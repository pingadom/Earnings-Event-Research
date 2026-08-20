"""Look-ahead regression tests.

Each of these plants a specific leak and asserts that the pipeline refuses it.
If any of these tests is ever deleted to make a build pass, the results this
repository produces stop meaning anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_engine.events.pit import (
    PointInTimeError,
    assert_point_in_time,
    restrict_to_known,
)
from earnings_engine.features import assemble_features


def test_clean_join_passes(events, fundamentals):
    joined = restrict_to_known(events, fundamentals, keep="all")
    assert not joined.empty
    assert_point_in_time(joined, label="test join")


def test_planted_future_fact_is_rejected(events, fundamentals):
    joined = restrict_to_known(events, fundamentals, keep="all")
    poisoned = joined.copy()
    idx = poisoned.index[:5]
    poisoned.loc[idx, "available_from_utc"] = poisoned.loc[idx, "trade_open_ts"] + pd.Timedelta(
        minutes=1
    )
    with pytest.raises(PointInTimeError, match="published after the trade timestamp"):
        assert_point_in_time(poisoned)


def test_missing_publication_time_is_rejected(events, fundamentals):
    joined = restrict_to_known(events, fundamentals, keep="all")
    poisoned = joined.copy()
    poisoned.loc[poisoned.index[:3], "available_from_utc"] = pd.NaT
    with pytest.raises(PointInTimeError, match="no available_from_utc"):
        assert_point_in_time(poisoned)


def test_restrict_to_known_never_returns_future_facts(events, fundamentals):
    """The as-of join is the only sanctioned way to attach facts to events."""
    joined = restrict_to_known(events, fundamentals, keep="all")
    known = pd.to_datetime(joined["available_from_utc"], utc=True)
    trade = pd.to_datetime(joined["trade_open_ts"], utc=True)
    assert (known <= trade).all()


def test_assembler_blanks_a_block_published_after_the_open(events, fundamentals):
    """A feature block dated after the trade open must be blanked, not kept."""
    from earnings_engine.features.fundamentals import build_fundamental_features

    feats = build_fundamental_features(fundamentals)
    late = feats.copy()
    late["available_from_utc"] = late["available_from_utc"] + pd.Timedelta(days=400)

    panel = assemble_features(events, {"fund": late})
    value_cols = [c for c in panel.columns if c.startswith(("revenue_", "gross_margin"))]
    assert value_cols, "expected fundamental feature columns to be present"
    assert panel[value_cols].notna().sum().sum() == 0


def test_assembler_rejects_a_non_unique_join(events, fundamentals):
    from earnings_engine.features.fundamentals import build_fundamental_features

    feats = build_fundamental_features(fundamentals)
    duplicated = pd.concat([feats, feats], ignore_index=True)
    with pytest.raises(ValueError, match="changed the row count"):
        assemble_features(events, {"fund": duplicated})


def test_surprise_expectation_uses_only_past_quarters(fundamentals):
    """SUE at quarter t must not move when quarter t+1 is revealed."""
    from earnings_engine.features.fundamentals import _pivot
    from earnings_engine.features.surprise import build_surprise_features

    wide = _pivot(fundamentals).sort_values(["ticker", "period_end"])
    full = build_surprise_features(wide).set_index(["ticker", "period_end"])["sue_timeseries"]

    cutoff = wide["period_end"].quantile(0.8)
    truncated = build_surprise_features(wide[wide["period_end"] <= cutoff]).set_index(
        ["ticker", "period_end"]
    )["sue_timeseries"]

    common = truncated.index.intersection(full.index)
    a, b = full.loc[common], truncated.loc[common]
    both = a.notna() & b.notna()
    assert both.sum() > 50, "not enough overlapping observations to make the test meaningful"
    np.testing.assert_allclose(a[both].to_numpy(), b[both].to_numpy(), rtol=1e-9)


def test_text_similarity_cannot_see_later_filings():
    """A future document must not change a similarity computed before it.

    Fitting one TF-IDF space over a firm's whole history lets the
    inverse-document-frequency weights of an early filing be shaped by
    vocabulary that only appears years later. The feature's timestamp stays
    honest, so this leak is invisible to the point-in-time validator and has to
    be pinned by a test instead.
    """
    from earnings_engine.features.text import TextFeatureExtractor

    extractor = TextFeatureExtractor()
    history = [
        "revenue grew strongly and margins expanded across every segment",
        "revenue grew again and margins expanded across most segments",
        "revenue declined and margins compressed across several segments",
    ]
    # A later filing full of vocabulary the earlier ones never used.
    future = history + ["restructuring impairment goodwill writedown litigation reserve" * 3]

    before, _ = extractor.similarity(history)
    after, _ = extractor.similarity(future)
    assert after[:3] == pytest.approx(before, nan_ok=True), (
        "adding a later filing changed an earlier similarity: the TF-IDF space "
        "is being fitted on the future"
    )


def test_text_similarity_still_measures_editorial_change():
    """The guard above must not have been bought by making the feature inert."""
    from earnings_engine.features.text import TextFeatureExtractor

    extractor = TextFeatureExtractor()
    prev, _ = extractor.similarity(
        [
            "revenue grew strongly and margins expanded across every segment",
            "revenue grew strongly and margins expanded across every segment",
            "restructuring impairment goodwill writedown litigation reserve entirely different",
        ]
    )
    assert prev[1] > prev[2], "a rewritten filing must score less similar than a repeated one"


def _refit_similarity(texts):
    """The obvious implementation: refit a TF-IDF space per document.

    Kept in the tests as the reference the fast path must reproduce.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(texts)
    prev = np.full(n, np.nan)
    for i in range(1, n):
        history = [t or "" for t in texts[: i + 1]]
        matrix = TfidfVectorizer(sublinear_tf=True, min_df=1).fit_transform(history)
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
        denominator = norms[i] * norms[i - 1]
        if denominator:
            prev[i] = float(matrix[i].multiply(matrix[i - 1]).sum() / denominator)
    return prev


def test_the_fast_similarity_reproduces_refitting_exactly():
    """Speed must not have been bought with a different answer.

    Refitting a vectoriser per document took twenty-three minutes of a
    thirty-minute study. Counting terms once and accumulating the document
    frequencies is the same arithmetic: a term absent from both documents being
    compared contributes nothing to their dot product, so restricting the
    vocabulary to the prefix cannot change the cosine.
    """
    import numpy as np

    from earnings_engine.features.text import TextFeatureExtractor

    rng = np.random.default_rng(0)
    vocabulary = [f"word{i}" for i in range(300)]
    corpus = [
        " ".join(rng.choice(vocabulary, size=rng.integers(120, 400)))
        for _ in range(12)
    ]
    fast, _ = TextFeatureExtractor().similarity(corpus)
    slow = _refit_similarity(corpus)
    assert fast[1:] == pytest.approx(slow[1:], abs=1e-9)


def test_the_fast_similarity_is_actually_fast():
    import time

    import numpy as np

    from earnings_engine.features.text import TextFeatureExtractor

    rng = np.random.default_rng(1)
    vocabulary = [f"word{i}" for i in range(4000)]
    corpus = [" ".join(rng.choice(vocabulary, size=2000)) for _ in range(60)]
    started = time.perf_counter()
    TextFeatureExtractor().similarity(corpus)
    assert time.perf_counter() - started < 5.0
