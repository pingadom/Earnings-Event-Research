"""Feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_engine.features import (
    assemble_features,
    build_fundamental_features,
    build_text_features,
    cross_sectional_normalise,
)
from earnings_engine.features.assemble import feature_columns
from earnings_engine.features.text import TextFeatureExtractor


def test_growth_handles_a_negative_base():
    from earnings_engine.features.fundamentals import _safe_growth

    current = pd.Series([10.0, 10.0, -5.0])
    prior = pd.Series([-5.0, 5.0, 10.0])
    got = _safe_growth(current, prior)
    # A swing from -5 to +10 is an improvement, so the sign must be positive.
    assert got.iloc[0] > 0
    assert got.iloc[1] == pytest.approx(1.0)
    assert got.iloc[2] < 0


def test_yoy_differencing_removes_seasonality():
    """A perfectly seasonal series must show zero year-on-year growth."""
    periods = pd.date_range("2015-03-31", periods=16, freq="QE")
    seasonal = np.tile([100.0, 120.0, 90.0, 200.0], 4)
    rows = []
    for p, v in zip(periods, seasonal, strict=False):
        rows.append(
            {
                "ticker": "AAA",
                "period_end": p,
                "available_from_utc": pd.Timestamp(p, tz="UTC") + pd.Timedelta(days=40),
                "item": "revenue",
                "value": v,
            }
        )
    feats = build_fundamental_features(pd.DataFrame(rows), winsorize=None)
    growth = feats["revenue_growth_yoy"].dropna()
    assert len(growth) == 12
    np.testing.assert_allclose(growth.to_numpy(), 0.0, atol=1e-12)


def test_fundamental_features_carry_a_publication_stamp(fundamentals):
    feats = build_fundamental_features(fundamentals)
    assert "available_from_utc" in feats.columns
    assert feats["available_from_utc"].notna().all()


def test_tone_responds_to_wording():
    ex = TextFeatureExtractor()
    good = ex.tone("strong growth improved margins record momentum confident outlook")
    bad = ex.tone("weak decline impairment loss deteriorating disappointing shortfall")
    assert good["tone_net"] > 0.5
    assert bad["tone_net"] < -0.5
    assert bad["pct_negative"] > good["pct_negative"]


def test_tone_is_empty_safe():
    ex = TextFeatureExtractor()
    assert np.isnan(ex.tone("")["tone_net"])


def test_similarity_detects_a_rewritten_filing():
    ex = TextFeatureExtractor()
    base = "the company operates three segments revenue increased customers grew steadily"
    texts = [base, base, "entirely different wording about litigation and restructuring costs"]
    prev, _ = ex.similarity(texts)
    assert prev[1] > 0.95  # unchanged filing
    assert prev[2] < 0.3  # heavily rewritten


def test_text_features_track_the_latent_surprise(provider, filings):
    feats = build_text_features(filings.head(400), provider.get_text)
    truth = provider.ground_truth().set_index("event_id")["z"]
    feats["z"] = feats["accession"].map(truth)
    assert feats[["tone_net", "z"]].corr().iloc[0, 1] > 0.5
    assert feats[["pct_negative", "z"]].corr().iloc[0, 1] < -0.5


def test_cross_sectional_normalisation_is_rank_preserving(events, fundamentals):
    feats = build_fundamental_features(fundamentals)
    panel = assemble_features(events, {"fund": feats})
    cols = feature_columns(panel)
    normed = cross_sectional_normalise(panel, cols, method="rank_gauss", min_obs=5)
    col = "revenue_growth_yoy"
    cohort = normed.groupby(normed["t0"].dt.to_period("M"))
    for _, grp in cohort:
        sub = pd.DataFrame({"raw": panel.loc[grp.index, col], "norm": grp[col]}).dropna()
        if len(sub) > 5:
            assert sub["raw"].corr(sub["norm"], method="spearman") == pytest.approx(1.0)
            break


def test_normalisation_leaves_no_cross_sectional_mean(events, fundamentals):
    feats = build_fundamental_features(fundamentals)
    panel = assemble_features(events, {"fund": feats})
    cols = feature_columns(panel)
    normed = cross_sectional_normalise(panel, cols, method="zscore", min_obs=15)
    means = normed.groupby(normed["t0"].dt.to_period("M"))["revenue_growth_yoy"].mean().dropna()
    assert means.abs().max() < 1e-9


def test_a_period_with_no_flow_item_is_not_a_reporting_quarter():
    """Balance-sheet instants dated outside a fiscal period must not add rows.

    The cover-page share count is stamped "as of" the filing date, weeks after
    the quarter it accompanies. Left in, it inserts an extra row between real
    quarters and every ``t-4`` comparison lands on the wrong one.
    """
    from earnings_engine.features.fundamentals import _pivot

    stamp = pd.Timestamp("2021-01-01", tz="UTC")
    rows = []
    for index, period in enumerate(pd.date_range("2020-03-31", periods=5, freq="QE")):
        rows.append(("AAA", period, "revenue", 100.0 + index, stamp))
        rows.append(("AAA", period, "total_assets", 900.0, stamp))
    # A phantom: an instant three weeks after the first quarter ended.
    rows.append(("AAA", pd.Timestamp("2020-04-21"), "total_assets", 950.0, stamp))
    frame = pd.DataFrame(
        rows, columns=["ticker", "period_end", "item", "value", "available_from_utc"]
    )
    out = _pivot(frame)
    assert len(out) == 5
    assert pd.Timestamp("2020-04-21") not in set(out["period_end"])


def test_year_ago_comparisons_land_on_the_same_quarter():
    """With phantoms removed, ``t-4`` is genuinely one year back."""
    stamp = pd.Timestamp("2021-01-01", tz="UTC")
    periods = pd.date_range("2019-03-31", periods=8, freq="QE")
    rows = []
    for index, period in enumerate(periods):
        rows.append(("AAA", period, "revenue", 100.0 * (1.1**index), stamp))
        rows.append(("AAA", pd.Timestamp(period) + pd.Timedelta(days=20), "shares_outstanding",
                     5.0, stamp))
    frame = pd.DataFrame(
        rows, columns=["ticker", "period_end", "item", "value", "available_from_utc"]
    )
    out = build_fundamental_features(frame, winsorize=None)
    growth = out.loc[out["period_end"] == periods[4], "revenue_growth_yoy"].iloc[0]
    assert growth == pytest.approx(1.1**4 - 1, rel=1e-9)
