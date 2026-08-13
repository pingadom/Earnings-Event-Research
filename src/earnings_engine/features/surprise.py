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
        The consensus stamp must pre-date the announcement; that is checked by
        the point-in-time guard downstream, not assumed here.
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
        merged = out.merge(
            c[["ticker", "period_end", "consensus_eps", "consensus_std", "n_estimates"]],
            on=["ticker", "period_end"],
            how="left",
        )
        actual = df[eps_col].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = merged["consensus_std"].where(merged["consensus_std"] > 0)
            merged["sue_analyst"] = (actual - merged["consensus_eps"]) / denom
            # Percentage surprise: scale-free, robust when dispersion is tiny.
            base = merged["consensus_eps"].abs()
            merged["surprise_pct"] = (actual - merged["consensus_eps"]) / base.where(base > 0)
        merged["sue_analyst"] = merged["sue_analyst"].replace([np.inf, -np.inf], np.nan)
        merged["surprise_pct"] = merged["surprise_pct"].replace([np.inf, -np.inf], np.nan)
        out = merged

    return out.reset_index(drop=True)
