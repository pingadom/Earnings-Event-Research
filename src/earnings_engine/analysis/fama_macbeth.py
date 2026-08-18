"""Fama-MacBeth: a second, independent read on the same question.

The portfolio sort in :mod:`earnings_engine.backtest` and this regression ask
the same question with different machinery, and that is the point. A sort is
non-parametric and robust but throws away most of the information in the
cross-section; a regression uses all of it but imposes linearity. When they
agree the result is much harder to dismiss than either alone. When they
disagree, that is a finding too -- typically it means the effect lives in the
tails rather than across the whole distribution.

The procedure (Fama & MacBeth 1973)
-----------------------------------
1. For each period *t*, run one cross-sectional regression of realised abnormal
   returns on the features known before *t*::

       CAR_{i,t} = gamma_{0,t} + sum_k gamma_{k,t} · x_{k,i,t} + e_{i,t}

2. That gives a *time series* of coefficients for each feature.
3. Test each series against zero, with Newey-West standard errors, because
   overlapping 20-day horizons make consecutive periods correlated.

The elegance is that cross-sectional correlation within a period -- the thing
that makes pooled OLS wildly overconfident here, since hundreds of firms report
in the same fortnight and share the same macro shocks -- is absorbed entirely
into each period's coefficient. The inference happens across periods, which are
far closer to independent.

Caveats worth stating rather than burying
-----------------------------------------
* **Errors-in-variables.** Where regressors are themselves estimated (betas, for
  instance), the standard errors are too small; Shanken (1992) gives the
  correction. Features here are observed accounting and text quantities rather
  than estimates, so the issue is mild, but it is not zero.
* **Monthly is a choice.** Grouping into months rather than days trades a
  shorter, less noisy coefficient series against fewer observations to test. The
  frequency is a parameter, and results should be reported for more than one.
* **Linearity is imposed.** A feature that only matters in its tails will look
  weak here and strong in a quantile sort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..returns.stats import newey_west_tstat
from ..utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class FamaMacBethResult:
    """Coefficient time series plus the second-stage tests."""

    coefficients: pd.DataFrame
    summary: pd.DataFrame
    n_periods: int
    mean_cross_section: float
    frequency: str

    def significant(self, level: float = 0.05) -> pd.DataFrame:
        return self.summary[self.summary["p_value"] < level]

    def to_markdown(self, decimals: int = 4) -> str:
        out = self.summary.copy()
        for col in ("coefficient", "std_error"):
            out[col] = (out[col] * 1e4).round(2)
        out = out.rename(columns={"coefficient": "coef (bp)", "std_error": "se (bp)"})
        return out.round(decimals).to_markdown(index=False)

    def verdict(self, level: float = 0.05) -> str:
        sig = self.significant(level)
        sig = sig[sig["term"] != "intercept"]
        if sig.empty:
            return (
                f"No feature is significant at the {level:.0%} level across "
                f"{self.n_periods} periods."
            )
        names = ", ".join(
            f"{r.term} ({r.coefficient * 1e4:+.1f} bp, t = {r.t_stat:.2f})"
            for r in sig.itertuples()
        )
        return f"Significant across {self.n_periods} periods: {names}."


def _ols(y: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """OLS with an intercept, returning None when the period is degenerate."""
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if ok.sum() < x.shape[1] + 5:
        return None
    xs = np.column_stack([np.ones(ok.sum()), x[ok]])
    if np.linalg.matrix_rank(xs) < xs.shape[1]:
        return None
    beta, *_ = np.linalg.lstsq(xs, y[ok], rcond=None)
    return beta


def fama_macbeth(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    date_col: str = "t0",
    frequency: str = "M",
    min_cross_section: int = 15,
    hac_lags: int | None = None,
    standardise: bool = True,
) -> FamaMacBethResult:
    """Run the two-stage Fama-MacBeth procedure.

    Parameters
    ----------
    standardise
        Cross-sectionally z-score the features inside each period, so
        coefficients are comparable across features and read as "basis points of
        abnormal return per one standard deviation of the feature". Strongly
        recommended; without it the units are whatever the feature happens to be.
    """
    missing = [c for c in [*features, target, date_col] if c not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing column(s) {missing}")

    # Drop degenerate regressors up front. A feature that is entirely missing,
    # or constant, makes the design matrix rank-deficient in every period, and
    # the whole regression silently returns nothing.
    usable = []
    for f in features:
        col = pd.to_numeric(panel[f], errors="coerce")
        if col.notna().sum() < 50:
            log.info("Fama-MacBeth: dropping %r (only %d non-null)", f, int(col.notna().sum()))
        elif not (col.std(skipna=True) > 0):
            log.info("Fama-MacBeth: dropping %r (constant)", f)
        else:
            usable.append(f)
    if not usable:
        raise ValueError("no usable features: all are constant or almost entirely missing")
    features = usable

    # A cross-sectional regression needs meaningfully more names than
    # regressors. With 30 features and 40 names per month the coefficients are
    # noise. Fail loudly rather than reporting them.
    required = len(features) + 10
    if min_cross_section < required:
        log.info(
            "Fama-MacBeth: raising min_cross_section from %d to %d for %d regressors",
            min_cross_section, required, len(features),
        )
        min_cross_section = required

    df = panel[[date_col, target, *features]].copy()
    dates = pd.to_datetime(df[date_col])
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)
    df["_period"] = (
        dates.dt.normalize()
        if frequency.upper() == "D"
        else dates.dt.to_period(frequency).dt.to_timestamp()
    )

    rows, sizes = [], []
    for period, grp in df.groupby("_period", sort=True):
        sub = grp.dropna(subset=[target])
        if len(sub) < min_cross_section:
            continue
        raw = sub[features].apply(pd.to_numeric, errors="coerce")
        # A feature can be fine overall and degenerate in one month. Regress on
        # the columns that carry variation *in this period* and leave the rest
        # NaN for this period, rather than discarding the period entirely.
        live = [f for f in features if raw[f].notna().sum() >= 5 and raw[f].std(skipna=True) > 0]
        if len(live) < 2 or len(sub) < len(live) + 10:
            continue
        x = raw[live].to_numpy(dtype="float64")
        if standardise:
            mu = np.nanmean(x, axis=0)
            sd = np.nanstd(x, axis=0, ddof=1)
            x = (x - mu) / np.where(sd > 0, sd, np.nan)
        med = np.nanmedian(x, axis=0)
        x = np.where(np.isfinite(x), x, np.where(np.isfinite(med), med, 0.0))

        beta = _ols(sub[target].to_numpy(dtype="float64"), x)
        if beta is None:
            continue
        row = pd.Series(np.nan, index=["intercept", *features], name=period)
        row.loc[["intercept", *live]] = beta
        rows.append(row)
        sizes.append(len(sub))

    if len(rows) < 6:
        raise ValueError(
            f"only {len(rows)} usable periods at frequency {frequency!r} with {len(features)} "
            f"regressors and a minimum cross-section of {min_cross_section}. Use a coarser "
            f"frequency, or -- better -- pass a smaller feature set: a cross-sectional "
            f"regression needs many more names than regressors to mean anything."
        )

    coefs = pd.DataFrame(rows)
    coefs.index.name = "period"

    summary = []
    for term in coefs.columns:
        series = coefs[term].dropna()
        t_stat, se = newey_west_tstat(series, lags=hac_lags)
        n = len(series)
        summary.append(
            {
                "term": term,
                "coefficient": float(series.mean()),
                "std_error": float(se) if np.isfinite(se) else np.nan,
                "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "p_value": float(2 * sps.norm.sf(abs(t_stat))) if np.isfinite(t_stat) else np.nan,
                "n_periods": n,
                "share_positive": float((series > 0).mean()),
            }
        )

    log.info(
        "Fama-MacBeth: %d periods at %s, mean cross-section %.0f names",
        len(coefs), frequency, float(np.mean(sizes)),
    )
    return FamaMacBethResult(
        coefficients=coefs,
        summary=pd.DataFrame(summary),
        n_periods=len(coefs),
        mean_cross_section=float(np.mean(sizes)),
        frequency=frequency,
    )
