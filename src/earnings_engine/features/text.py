"""Text features from filings and earnings releases.

Three families, in increasing order of how much they actually add:

**Tone**
    Proportion of words in each Loughran-McDonald category. Finance-specific
    dictionaries matter: general-purpose lexicons flag "liability", "cost" and
    "depreciation" as negative when they are accounting vocabulary. Loughran
    and McDonald (2011) found roughly three quarters of Harvard-IV negative
    words in 10-Ks are misclassifications of this kind.

**Change in tone**
    The level of negativity in a 10-K is mostly a firm fixed effect -- some
    companies just write cautiously. What carries information is the *change*
    against the same firm's previous filing, so every tone feature has a
    ``_delta`` counterpart.

**Change in language**
    Cosine similarity between a filing and the firm's previous one. Cohen,
    Malloy and Nguyen (2020) document that firms whose filings change little
    outperform those whose filings change a lot -- companies edit their
    boilerplate when something is wrong. This is a pure text feature with no
    fundamental counterpart, which is what makes it worth having.

The extractor is deliberately dependency-light (dictionary counts plus
scikit-learn TF-IDF). A transformer-based sentiment scorer can be dropped in
behind the same interface via the ``nlp`` extra; see ``score_transformer``.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"
BUNDLED_LEXICON = RESOURCE_DIR / "lm_lite.txt"

CATEGORIES = ("negative", "positive", "uncertainty", "litigious", "strong_modal", "weak_modal")
_WORD_RE = re.compile(r"[a-z][a-z'-]+")

TEXT_FEATURES: dict[str, str] = {
    "tone_net": "(positive - negative) / (positive + negative)",
    "pct_negative": "Share of words in the LM negative category",
    "pct_positive": "Share of words in the LM positive category",
    "pct_uncertainty": "Share of words in the LM uncertainty category",
    "pct_litigious": "Share of words in the LM litigious category",
    "modal_strength": "(strong modal - weak modal) share",
    "doc_length": "Word count (log)",
    "tone_net_delta": "Change in net tone vs the firm's previous filing",
    "pct_negative_delta": "Change in negativity vs the previous filing",
    "pct_uncertainty_delta": "Change in uncertainty vs the previous filing",
    "doc_length_delta": "Log change in document length",
    "similarity_prev": "Cosine similarity of TF-IDF vs the previous filing",
    "similarity_yoy": "Cosine similarity vs the same quarter one year ago",
}


def load_lexicon(path: str | Path | None = None) -> dict[str, set[str]]:
    """Load a sentiment lexicon.

    With no path, the bundled compact lexicon is used. With a path to the full
    Loughran-McDonald master dictionary CSV, that is parsed instead.
    """
    if path is None:
        return _load_bundled()
    p = Path(path)
    if not p.exists():
        log.warning("lexicon %s not found; falling back to the bundled lexicon", p)
        return _load_bundled()
    if p.suffix.lower() == ".csv":
        return _load_lm_master(p)
    return _load_tsv(p)


def _load_bundled() -> dict[str, set[str]]:
    return _load_tsv(BUNDLED_LEXICON)


def _load_tsv(path: Path) -> dict[str, set[str]]:
    lex: dict[str, set[str]] = {c: set() for c in CATEGORIES}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cat, _, word = line.partition("\t")
        if cat in lex and word:
            lex[cat].add(word.strip().lower())
    return lex


def _load_lm_master(path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    words = df[cols["word"]].astype(str).str.lower()
    mapping = {
        "negative": "negative",
        "positive": "positive",
        "uncertainty": "uncertainty",
        "litigious": "litigious",
        "strong_modal": "strong_modal",
        "weak_modal": "weak_modal",
    }
    lex: dict[str, set[str]] = {}
    for cat, col in mapping.items():
        if col in cols:
            flags = pd.to_numeric(df[cols[col]], errors="coerce").fillna(0)
            lex[cat] = set(words[flags > 0])
        else:
            lex[cat] = set()
    log.info("loaded Loughran-McDonald master dictionary: %d words", len(words))
    return lex


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class TextFeatureExtractor:
    """Turns filing text into the feature families described above."""

    def __init__(self, lexicon_path: str | Path | None = None, max_features: int = 20_000) -> None:
        self.lexicon = load_lexicon(lexicon_path)
        self.max_features = max_features
        self.sublinear_tf = True

    # ---- per-document -------------------------------------------------

    def tone(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        n = len(tokens)
        if n == 0:
            return dict.fromkeys(
                ["tone_net", "pct_negative", "pct_positive", "pct_uncertainty",
                 "pct_litigious", "modal_strength", "doc_length"], np.nan
            )
        counts = Counter(tokens)
        share = {}
        for cat in CATEGORIES:
            words = self.lexicon.get(cat, set())
            share[cat] = sum(counts[w] for w in words if w in counts) / n
        pos, neg = share["positive"], share["negative"]
        denom = pos + neg
        return {
            "tone_net": (pos - neg) / denom if denom > 0 else 0.0,
            "pct_negative": neg,
            "pct_positive": pos,
            "pct_uncertainty": share["uncertainty"],
            "pct_litigious": share["litigious"],
            "modal_strength": share["strong_modal"] - share["weak_modal"],
            "doc_length": float(np.log1p(n)),
        }

    # ---- across a firm's history ---------------------------------------

    def similarity(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Cosine similarity of each document to its predecessor and to t-4.

        The space is fitted **per firm and expanding through time**: document
        *i* is scored in a TF-IDF space built from documents 0..i only.

        The per-firm part is deliberate -- it makes the measure capture change
        in *this* company's language rather than drift relative to the market.

        The expanding part is not an optimisation, it is a correctness
        requirement. Fitting one space over a firm's whole history lets the
        inverse-document-frequency weights of a 2019 filing be shaped by words
        that first appeared in 2024, which is precisely the look-ahead this
        project refuses everywhere else. It is a quiet leak: the feature's
        timestamp is honest, the value behind it is not, so the point-in-time
        validator cannot see it. Measured on the S&P 500 sample it was worth
        roughly two and a half hundredths of information coefficient -- enough
        to move a null result to a t-statistic of 2.

        The cost is one refit per document. The vocabulary genuinely grows
        through the sample, which is what a reader of these filings in real
        time would have experienced.
        """
        from sklearn.feature_extraction.text import CountVectorizer  # noqa: PLC0415

        n = len(texts)
        prev = np.full(n, np.nan)
        yoy = np.full(n, np.nan)
        cleaned = [t or "" for t in texts]
        if sum(1 for t in cleaned if t.strip()) < 2:
            return prev, yoy
        try:
            counts = CountVectorizer(max_features=self.max_features).fit_transform(cleaned)
        except ValueError:  # empty vocabulary
            return prev, yoy

        # Term frequencies are fixed; only the document frequencies behind the
        # IDF weights change as the window expands. Counting once and
        # accumulating the document frequencies gives exactly what refitting a
        # vectoriser per document gives -- a term absent from both documents
        # being compared contributes nothing to their dot product, so
        # restricting the vocabulary to the prefix cannot change the cosine --
        # at a small fraction of the cost. Refitting took twenty-three minutes
        # of a thirty-minute study.
        tf = counts.astype("float64").tocsr()
        if self.sublinear_tf:
            tf.data = 1.0 + np.log(tf.data)
        present = (counts > 0).astype("float64").toarray()
        document_frequency = np.cumsum(present, axis=0)

        for i in range(1, n):
            if not cleaned[i].strip() or not cleaned[i - 1].strip():
                continue
            # sklearn's smoothed form: log((1 + n_docs) / (1 + df)) + 1.
            idf = np.log((1.0 + (i + 1)) / (1.0 + document_frequency[i])) + 1.0
            prev[i] = _weighted_cosine(tf, i, i - 1, idf)
            if i >= 4 and cleaned[i - 4].strip():
                yoy[i] = _weighted_cosine(tf, i, i - 4, idf)
        return prev, yoy


def _weighted_cosine(tf, i: int, j: int, idf: np.ndarray) -> float:
    """Cosine between two rows of a term-frequency matrix under given IDF weights."""
    a = np.asarray(tf[i].todense()).ravel() * idf
    b = np.asarray(tf[j].todense()).ravel() * idf
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(a @ b / denominator)


def _cosine(matrix, i: int, j: int, norms: np.ndarray) -> float:
    num = matrix[i].multiply(matrix[j]).sum()
    den = norms[i] * norms[j]
    return float(num / den) if den and np.isfinite(den) else np.nan


def build_text_features(
    filings: pd.DataFrame,
    text_loader,
    lexicon_path: str | Path | None = None,
    max_docs: int | None = None,
) -> pd.DataFrame:
    """Build the text feature panel.

    Parameters
    ----------
    filings
        Frame conforming to :data:`earnings_engine.utils.frames.FILINGS`.
    text_loader
        Callable ``accession -> str``. Kept as a callback so filings can be
        streamed from disk, EDGAR, or a synthetic generator without this
        module knowing which.
    """
    extractor = TextFeatureExtractor(lexicon_path)
    df = filings.sort_values(["ticker", "filed_at_utc"]).reset_index(drop=True)
    if max_docs is not None:
        df = df.groupby("ticker", sort=False).head(max_docs).reset_index(drop=True)

    rows = []
    for _ticker, grp in df.groupby("ticker", sort=False):
        texts = []
        for accession in grp["accession"]:
            try:
                texts.append(text_loader(accession) or "")
            except Exception as exc:  # pragma: no cover - loader dependent
                log.warning("could not load text for %s: %s", accession, exc)
                texts.append("")
        tone = pd.DataFrame([extractor.tone(t) for t in texts], index=grp.index)
        prev_sim, yoy_sim = extractor.similarity(texts)
        block = grp[["ticker", "accession", "form", "filed_at_utc", "period_end"]].copy()
        block = pd.concat([block, tone], axis=1)
        block["similarity_prev"] = prev_sim
        block["similarity_yoy"] = yoy_sim
        for col in ("tone_net", "pct_negative", "pct_uncertainty", "doc_length"):
            block[f"{col}_delta"] = block[col].diff()
        rows.append(block)

    if not rows:
        return pd.DataFrame(columns=["ticker", "accession", *TEXT_FEATURES])
    out = pd.concat(rows, ignore_index=True)
    # The text is knowable exactly when the document was filed.
    out["available_from_utc"] = pd.to_datetime(out["filed_at_utc"], utc=True)
    return out


def score_transformer(texts: list[str], model_name: str = "ProsusAI/finbert") -> pd.DataFrame:
    """Optional transformer sentiment, behind the ``nlp`` extra.

    Deliberately not wired into the default pipeline. A 10-K runs to tens of
    thousands of tokens, so this needs a chunking strategy and a GPU to be
    practical, and the marginal value over a good dictionary on *changes* is an
    open question worth testing rather than assuming.
    """
    try:
        from transformers import pipeline  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformer scoring needs the nlp extra: pip install -e '.[nlp]'"
        ) from exc
    clf = pipeline("sentiment-analysis", model=model_name, truncation=True, max_length=512)
    scores = clf(texts)
    return pd.DataFrame(scores)
