"""Evaluation of a cross-sectional signal.

R-squared is close to useless here. A signal that explains 0.5% of the variance
of 20-day abnormal returns can be extremely profitable, and one with a flattering
in-sample R-squared can be worthless. The metrics that matter are about
*ordering*: does the signal rank next quarter's abnormal returns correctly, and
does it do so consistently through time?

* **Information coefficient (IC)** -- Spearman rank correlation between the
  signal and the realised abnormal return, computed *per cohort* and then
  averaged. Computing it pooled across all events at once conflates
  cross-sectional skill with market timing.
* **IC t-statistic** -- the mean IC over its standard error across cohorts.
  This is the number that tells you whether the signal is real; a high average
  IC driven by three good quarters is not a strategy.
* **Quantile spread** -- realised abnormal return of the top group minus the
  bottom group, which is what the backtest will actually trade.
* **Monotonicity** -- whether the quantile means increase in order. A signal
  that is strong only in the tails behaves very differently under transaction
  costs from one that is monotone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..returns.stats import newey_west_tstat


def information_coefficient(
    predictions: pd.DataFrame,
    pred_col: str = "prediction",
    target_col: str = "target",
    group_col: str = "cohort",
    min_obs: int = 15,
) -> pd.DataFrame:
    """Per-cohort rank IC."""
    rows = []
    for key, grp in predictions.groupby(group_col, sort=True):
        sub = grp[[pred_col, target_col]].dropna()
        if len(sub) < min_obs:
            continue
        rho, p = sps.spearmanr(sub[pred_col], sub[target_col])
        rows.append({group_col: key, "n": len(sub), "ic": rho, "p_value": p})
    return pd.DataFrame(rows)


def decile_spread(
    predictions: pd.DataFrame,
    pred_col: str = "prediction",
    target_col: str = "target",
    group_col: str = "cohort",
    quantiles: int = 5,
    min_obs: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean realised outcome per signal quantile, per cohort and pooled."""
    frames = []
    for key, grp in predictions.groupby(group_col, sort=True):
        sub = grp[[pred_col, target_col]].dropna()
        if len(sub) < max(min_obs, quantiles * 2):
            continue
        try:
            bucket = pd.qcut(sub[pred_col].rank(method="first"), quantiles, labels=False)
        except ValueError:
            continue
        means = sub.groupby(bucket)[target_col].mean()
        frames.append(means.rename(key))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    by_cohort = pd.concat(frames, axis=1).T
    by_cohort.index.name = group_col
    by_cohort.columns = [f"q{int(c) + 1}" for c in by_cohort.columns]
    by_cohort["spread"] = by_cohort.iloc[:, -1] - by_cohort.iloc[:, 0]

    pooled = pd.DataFrame(
        {
            "quantile": by_cohort.columns[:-1],
            "mean_target": by_cohort.iloc[:, :-1].mean().to_numpy(),
            "n_cohorts": by_cohort.iloc[:, :-1].notna().sum().to_numpy(),
        }
    )
    return by_cohort, pooled


def evaluate_predictions(
    predictions: pd.DataFrame,
    target_col: str,
    pred_col: str = "prediction",
    date_col: str = "t0",
    quantiles: int = 5,
    freq: str = "M",
) -> dict:
    """Headline evaluation of an out-of-sample prediction set."""
    df = predictions.copy()
    dates = pd.to_datetime(df[date_col])
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)
    df["cohort"] = dates.dt.to_period(freq).dt.to_timestamp()
    df = df.rename(columns={target_col: "target"})

    ic = information_coefficient(df, pred_col=pred_col, target_col="target")
    by_cohort, pooled = decile_spread(
        df, pred_col=pred_col, target_col="target", quantiles=quantiles
    )

    ic_mean = float(ic["ic"].mean()) if not ic.empty else np.nan
    ic_std = float(ic["ic"].std(ddof=1)) if len(ic) > 1 else np.nan
    ic_t, _ = newey_west_tstat(ic["ic"]) if not ic.empty else (np.nan, np.nan)
    hit = float((ic["ic"] > 0).mean()) if not ic.empty else np.nan

    spread_series = by_cohort["spread"] if "spread" in by_cohort else pd.Series(dtype=float)
    spread_t, spread_se = (
        newey_west_tstat(spread_series) if not spread_series.empty else (np.nan, np.nan)
    )

    monotone = np.nan
    if not pooled.empty:
        vals = pooled["mean_target"].to_numpy()
        monotone = float(sps.spearmanr(np.arange(len(vals)), vals).statistic)

    pooled_rho = np.nan
    sub = df[[pred_col, "target"]].dropna()
    if len(sub) > 10:
        pooled_rho = float(sps.spearmanr(sub[pred_col], sub["target"]).statistic)

    return {
        "n_predictions": int(len(df)),
        "n_cohorts": int(len(ic)),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_tstat_nw": float(ic_t) if ic_t == ic_t else np.nan,
        "ic_hit_rate": hit,
        "ic_pooled": pooled_rho,
        "quantile_spread_mean": float(spread_series.mean()) if not spread_series.empty else np.nan,
        "quantile_spread_tstat_nw": float(spread_t) if spread_t == spread_t else np.nan,
        "quantile_spread_se": float(spread_se) if spread_se == spread_se else np.nan,
        "monotonicity": monotone,
        "ic_by_cohort": ic,
        "quantiles_by_cohort": by_cohort,
        "quantiles_pooled": pooled,
    }
