"""Earnings surprise.

The canonical PEAD sorting variable is standardised unexpected earnings (SUE).
There are two families and the choice matters more than it looks:

**Analyst-based**  ``SUE = (actual - consensus) / sigma(analyst estimates)``.
    Closest to "what the market did not expect", but requires a consensus
    history. Capital IQ and LSEG both carry one; free sources do not.

**Time-series**    ``SUE = (x_t - x_{t-4} - drift) / sigma(seasonal differences)``
    A seasonal random walk with drift, as in Foster, Olsen and Shevlin (1984).
    No vendor needed, works back as far as your fundamentals do, and is the
    default here precisely so the pipeline never silently degrades to nothing
    when consensus data is unavailable.

Both are computed when the inputs exist, and both are kept as separate
features: they disagree, and the disagreement is informative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

MIN_HISTORY = 8


def build_surprise_features(
    fundamentals_wide: pd.DataFrame,
    consensus: pd.DataFrame | None = None,
    eps_col: str = "eps_diluted",
    min_history: int = MIN_HISTORY,
) -> pd.DataFrame:
    """Compute time-series SUE and, if consensus is supplied, analyst SUE.

    Parameters
    ----------
    fundamentals_wide
        Frame with ``ticker``, ``period_end``, ``available_from_utc`` and the
        EPS column. (The output of :func:`._pivot` or any equivalent.)
    consensus
        Optional frame with ``ticker``, ``period_end``, ``consensus_eps``,
        ``consensus_std``, ``n_estimates`` and its own ``available_from_utc``.

        That last column is load-bearing. A consensus is a *forecast*, and an
        estimates screen queried without an as-of date hands back today's
        consensus for every historical period -- formed long after the actual
        was known. The output row therefore carries the **later** of the two
        stamps, so a row that mixes a reported figure with a snapshot of the
        forecast only becomes visible once both were. Taking the fundamentals
        stamp alone would let a forecast dated after the print be joined as
        though it had been available before it.

        Where the snapshot post-dates the figure it claims to forecast, the
        arithmetic is meaningless whatever the stamp says, and this function
        says so loudly rather than quietly. If that warning fires across a
        material share of the sample, the pull was run without an as-of date
        and needs redoing -- the warning is the finding, not noise.
    """
    df = fundamentals_wide.sort_values(["ticker", "period_end"]).copy()
    if eps_col not in df.columns:
        raise KeyError(f"surprise features need an {eps_col!r} column")

    g = df.groupby("ticker", sort=False)[eps_col]
    seasonal_diff = df[eps_col] - g.shift(4)
    by_ticker = seasonal_diff.groupby(df["ticker"], sort=False)

    # Expanding, strictly-past moments: .shift(1) after the expanding call is
    # what stops the current quarter contributing to its own expectation.
    drift = by_ticker.transform(lambda s: s.expanding(min_periods=min_history).mean().shift(1))
    sigma = by_ticker.transform(
        lambda s: s.expanding(min_periods=min_history).std(ddof=1).shift(1)
    )
    count = by_ticker.transform(lambda s: s.expanding(min_periods=1).count().shift(1))

    out = df[["ticker", "period_end", "available_from_utc"]].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["sue_timeseries"] = (seasonal_diff - drift) / sigma.where(sigma > 0)
    out["sue_timeseries"] = out["sue_timeseries"].replace([np.inf, -np.inf], np.nan)
    out.loc[count < min_history, "sue_timeseries"] = np.nan
    out["sue_ts_history"] = count

    if consensus is not None and not consensus.empty:
        c = consensus.copy()
        cols = ["ticker", "period_end", "consensus_eps", "consensus_std", "n_estimates"]
        if "available_from_utc" in c.columns:
            c = c.rename(columns={"available_from_utc": "_consensus_available_from_utc"})
            cols.append("_consensus_available_from_utc")
        else:
            log.warning(
                "consensus frame carries no available_from_utc; analyst SUE will inherit "
                "the reported figure's stamp, which ASSUMES the estimates pre-dated the "
                "print rather than establishing it. Supply the as-of date."
            )
        merged = out.merge(c[cols], on=["ticker", "period_end"], how="left")
        actual = df[eps_col].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = merged["consensus_std"].where(merged["consensus_std"] > 0)
            merged["sue_analyst"] = (actual - merged["consensus_eps"]) / denom
            # Percentage surprise: scale-free, robust when dispersion is tiny.
            base = merged["consensus_eps"].abs()
            merged["surprise_pct"] = (actual - merged["consensus_eps"]) / base.where(base > 0)
        merged["sue_analyst"] = merged["sue_analyst"].replace([np.inf, -np.inf], np.nan)
        merged["surprise_pct"] = merged["surprise_pct"].replace([np.inf, -np.inf], np.nan)
        if "_consensus_available_from_utc" in merged.columns:
            merged = _stamp_with_consensus(merged)
        out = merged

    return out.reset_index(drop=True)


def _stamp_with_consensus(merged: pd.DataFrame) -> pd.DataFrame:
    """Push each row's availability out to the later of its two stamps.

    Rows with no matched consensus keep the reported figure's stamp untouched;
    ``max`` against a missing snapshot must not turn a usable row into NaT.
    """
    figure = pd.to_datetime(merged["available_from_utc"], errors="coerce", utc=True)
    snapshot = pd.to_datetime(merged["_consensus_available_from_utc"], errors="coerce", utc=True)

    late = snapshot.notna() & figure.notna() & (snapshot > figure)
    n_late = int(late.sum())
    n_matched = int(snapshot.notna().sum())
    if n_late:
        share = n_late / max(n_matched, 1)
        log.warning(
            "consensus: %d of %d matched snapshot(s) (%.1f%%) are dated AFTER the figure "
            "they forecast. A forecast formed once the answer was known is not a forecast. "
            "The likely cause is a pull run without asOfDate, which returns today's "
            "consensus for every historical period -- see docs/capiq-pull-specification.xlsx. "
            "The affected rows are stamped at the snapshot date so the point-in-time gate "
            "still holds, but do not read sue_analyst on them as a surprise.",
            n_late,
            n_matched,
            100 * share,
        )
    merged["_consensus_stale"] = late
    merged["available_from_utc"] = figure.where(snapshot.isna(), snapshot.combine(figure, max))
    return merged
