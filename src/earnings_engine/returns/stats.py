"""Significance testing for event studies.

The naive test -- take the cross-section of CARs, divide the mean by
``std/sqrt(n)`` -- is wrong in three specific ways for this data, and each has
a fix implemented here.

1. **Event-induced variance.** Return variance *jumps* on announcement days,
   so the estimation-window sigma understates event-window sigma and the
   Patell-style test over-rejects. Boehmer, Musumeci and Poulsen (1991) fix
   this by standardising each CAR by its own predicted standard error and then
   taking a cross-sectional t-test of the standardised values --
   :func:`bmp_test`.
2. **Cross-sectional correlation.** Earnings cluster in calendar time: hundreds
   of firms report in the same fortnight, and their residuals share common
   shocks. Treating those as independent observations inflates ``n``. The fix
   is to resample *event dates*, not events --
   :func:`cluster_bootstrap_ci`.
3. **Overlapping horizons.** A 20-day CAR measured daily produces
   autocorrelated portfolio returns, so time-series t-stats need a HAC
   correction -- :func:`newey_west_tstat`.

None of this rescues a result that only exists at one horizon under one
estimator. It just stops you believing a result that is not there.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass(frozen=True)
class TestResult:
    n: int
    mean: float
    std_error: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    method: str

    def as_dict(self) -> dict:
        return asdict(self)


def cross_sectional_ttest(values, alpha: float = 0.05) -> TestResult:
    """Plain cross-sectional t-test. Reported as the baseline, not the answer."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return TestResult(n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "cross_sectional_t")
    mean = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n))
    t = mean / se if se > 0 else np.nan
    p = float(2 * sps.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan
    crit = sps.t.ppf(1 - alpha / 2, df=n - 1)
    return TestResult(n, mean, se, t, p, mean - crit * se, mean + crit * se, "cross_sectional_t")


def bmp_test(
    summary: pd.DataFrame,
    estimator: str = "market_model",
    window: tuple[int, int] = (0, 4),
    alpha: float = 0.05,
) -> TestResult:
    """Boehmer-Musumeci-Poulsen standardised cross-sectional test.

    Each CAR is divided by its predicted standard error

    .. math::
        s_i = \\sigma_i \\sqrt{L + \\frac{L^2}{T}
              + \\frac{\\left(\\sum_{t\\in W}(r_{mt}-\\bar r_m)\\right)^2}{S_{xx}}}

    (the Patell prediction-error correction: the last two terms account for the
    fact that ``alpha`` and ``beta`` were themselves estimated), and the test
    is then an ordinary t-test on the standardised values -- which is what
    makes it robust to the variance jump.
    """
    lo, hi = window
    car_col = f"car_{estimator}_{lo}_{hi}"
    dev_col = f"mktdev_{lo}_{hi}"
    needed = [car_col, "mm_sigma", "mm_n_obs", "est_market_sxx", dev_col]
    missing = [c for c in needed if c not in summary.columns]
    if missing:
        raise KeyError(f"bmp_test needs column(s) {missing}; recompute abnormal returns")

    df = summary[needed].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        return TestResult(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "bmp")

    L = float(hi - lo + 1)
    T = df["mm_n_obs"].to_numpy(dtype="float64")
    sigma = df["mm_sigma"].to_numpy(dtype="float64")
    sxx = df["est_market_sxx"].to_numpy(dtype="float64")
    dev = df[dev_col].to_numpy(dtype="float64")
    car = df[car_col].to_numpy(dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        var = sigma**2 * (L + (L**2) / T + (dev**2) / sxx)
        scar = car / np.sqrt(var)
    scar = scar[np.isfinite(scar)]
    n = scar.size
    if n < 2:
        return TestResult(n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "bmp")

    mean_scar = float(scar.mean())
    se_scar = float(scar.std(ddof=1) / np.sqrt(n))
    t = mean_scar / se_scar if se_scar > 0 else np.nan
    p = float(2 * sps.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan
    # Report the CI on the *raw* CAR scale, which is what a reader wants.
    raw_mean = float(car[np.isfinite(car)].mean())
    scale = raw_mean / mean_scar if mean_scar != 0 else np.nan
    crit = sps.t.ppf(1 - alpha / 2, df=n - 1)
    return TestResult(
        n=n,
        mean=raw_mean,
        std_error=abs(se_scar * scale) if np.isfinite(scale) else np.nan,
        t_stat=t,
        p_value=p,
        ci_low=raw_mean - crit * abs(se_scar * scale) if np.isfinite(scale) else np.nan,
        ci_high=raw_mean + crit * abs(se_scar * scale) if np.isfinite(scale) else np.nan,
        method="bmp",
    )


def cluster_bootstrap_ci(
    values,
    clusters,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260818,
) -> TestResult:
    """Bootstrap the mean by resampling *clusters* (event dates) with replacement.

    Earnings announcements cluster in calendar time, so events are not
    independent draws. Resampling whole dates preserves the within-date
    correlation instead of assuming it away.
    """
    v = pd.Series(np.asarray(values, dtype="float64"))
    c = pd.Series(np.asarray(clusters))
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if v.empty:
        return TestResult(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "cluster_bootstrap")

    groups = [g.to_numpy() for _, g in v.groupby(c.to_numpy(), sort=False)]
    k = len(groups)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        sample = np.concatenate([groups[i] for i in pick])
        means[b] = sample.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    mean = float(v.mean())
    se = float(means.std(ddof=1))
    t = mean / se if se > 0 else np.nan
    # Two-sided bootstrap p-value: how often does the resampled mean cross zero.
    p = float(2 * min((means <= 0).mean(), (means >= 0).mean()))
    return TestResult(
        n=int(v.size),
        mean=mean,
        std_error=se,
        t_stat=t,
        p_value=min(p, 1.0),
        ci_low=float(lo),
        ci_high=float(hi),
        method=f"cluster_bootstrap[{k} clusters]",
    )


def newey_west_tstat(series, lags: int | None = None) -> tuple[float, float]:
    """HAC t-statistic for the mean of an autocorrelated series.

    Returns ``(t_stat, standard_error)``. With ``lags=None`` the standard
    ``floor(4*(n/100)^(2/9))`` rule of thumb is used.
    """
    x = np.asarray(series, dtype="float64")
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.nan, np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 2))
    e = x - x.mean()
    gamma0 = float(e @ e) / n
    var = gamma0
    for lag in range(1, lags + 1):
        gamma = float(e[lag:] @ e[:-lag]) / n
        var += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    se = float(np.sqrt(max(var, 0.0) / n))
    return (float(x.mean() / se) if se > 0 else np.nan), se


def summarise_windows(
    summary: pd.DataFrame,
    estimators=("market_adjusted", "market_model", "sector_neutral"),
    windows=((0, 0), (0, 4), (0, 19)),
    cluster_col: str = "t0",
    n_boot: int = 1000,
    seed: int = 20260818,
) -> pd.DataFrame:
    """Full significance table: every estimator x window x test.

    Presenting the grid rather than the best cell is the point. If an effect
    only survives under one estimator at one horizon, the table shows that.
    """
    rows = []
    for est in estimators:
        for lo, hi in windows:
            col = f"car_{est}_{lo}_{hi}"
            if col not in summary.columns:
                continue
            values = summary[col]
            base = cross_sectional_ttest(values)
            rows.append({"estimator": est, "window": f"[{lo},{hi}]", **base.as_dict()})
            with contextlib.suppress(KeyError):
                rows.append(
                    {
                        "estimator": est,
                        "window": f"[{lo},{hi}]",
                        **bmp_test(summary, est, (lo, hi)).as_dict(),
                    }
                )
            if cluster_col in summary.columns:
                boot = cluster_bootstrap_ci(
                    values, summary[cluster_col], n_boot=n_boot, seed=seed
                )
                rows.append({"estimator": est, "window": f"[{lo},{hi}]", **boot.as_dict()})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out[
        [
            "estimator",
            "window",
            "method",
            "n",
            "mean",
            "std_error",
            "t_stat",
            "p_value",
            "ci_low",
            "ci_high",
        ]
    ]
