"""Factor attribution: is it alpha, or a known factor wearing a hat?

This is the first question anyone who does this professionally will ask, and a
backtest without an answer is unfinished. A long/short book sorted on earnings
information can easily end up long momentum (firms that beat tend to have been
rising), short value, or tilted to small caps -- all compensated risk premia
you can buy for a few basis points, not skill.

The regression
--------------
Daily strategy returns on the Fama-French five factors plus momentum::

    r_t = alpha + b_MKT·MKT_t + b_SMB·SMB_t + b_HML·HML_t
                + b_RMW·RMW_t + b_CMA·CMA_t + b_MOM·MOM_t + e_t

What matters in the output, in order:

1. **alpha and its t-statistic**, annualised, with Newey-West standard errors --
   overlapping 20-day holdings make daily residuals autocorrelated, and the OLS
   standard error would be too small.
2. **The loadings.** A strategy that is 40% momentum by construction should say
   so. Large loadings with a shrunken alpha is the common, honest outcome.
3. **R-squared.** High R-squared with near-zero alpha means the factors explain
   the strategy. *Low* R-squared with positive alpha is the good case -- and
   also the case where you should check hardest for a bug.
4. **Appraisal ratio** -- alpha over residual volatility, annualised: the Sharpe
   ratio of the part the factors cannot explain, and the number worth quoting.

Factor data comes from :mod:`earnings_engine.data.fama_french`, which downloads
Ken French's daily library. This module only consumes it, so there is one
downloader in the codebase rather than two.

Implemented on numpy for the same reason as the rest of the statistical core
(see ADR 0002): one fewer dependency, and the estimator stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

TRADING_DAYS = 252

#: Canonical factor names used throughout this module.
FACTORS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM")

#: Ken French publishes momentum as "Mom"; everything else already matches.
_ALIASES = {"Mom": "MOM", "MOM": "MOM", "UMD": "MOM", "Mkt_RF": "Mkt-RF"}


def normalise_factors(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Turn a Ken French download into a clean date-indexed factor frame.

    Accepts the output of
    :func:`earnings_engine.data.fama_french.download_fama_french_daily`
    (a ``date`` column, percent already converted to decimals, plus ``RF`` and a
    ``source`` label) and returns only the factor columns, indexed by date.
    """
    df = frame.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df.rename(columns={c: _ALIASES.get(str(c).strip(), str(c).strip()) for c in df.columns})
    keep = [c for c in FACTORS if c in df.columns]
    if not keep:
        raise ValueError(
            f"no recognised factor columns; got {sorted(map(str, df.columns))}. "
            f"Expected some of {FACTORS}."
        )
    missing = [c for c in FACTORS if c not in keep]
    if missing:
        log.info("factor frame is missing %s; regressing on %s", missing, keep)
    out = df[keep].apply(pd.to_numeric, errors="coerce")
    if out.abs().max().max() > 1.0:
        raise ValueError(
            "factor values look like percent, not decimals -- divide by 100 before use"
        )
    return out.sort_index()


@dataclass
class FactorModel:
    """Output of a factor regression, annualised where that is meaningful."""

    alpha_daily: float
    alpha_annual: float
    alpha_tstat: float
    alpha_pvalue: float
    loadings: pd.Series
    loading_tstats: pd.Series
    r_squared: float
    adj_r_squared: float
    residual_vol_annual: float
    appraisal_ratio: float
    n_obs: int
    hac_lags: int
    factors_used: tuple[str, ...]

    def summary_frame(self) -> pd.DataFrame:
        rows = [
            {
                "term": "alpha (annualised)",
                "estimate": self.alpha_annual,
                "t_stat": self.alpha_tstat,
                "p_value": self.alpha_pvalue,
            }
        ]
        for name in self.loadings.index:
            t = float(self.loading_tstats[name])
            rows.append(
                {
                    "term": name,
                    "estimate": float(self.loadings[name]),
                    "t_stat": t,
                    "p_value": float(2 * sps.norm.sf(abs(t))) if np.isfinite(t) else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def verdict(self) -> str:
        """A one-line reading, so the table is not left to speak for itself."""
        if not np.isfinite(self.alpha_tstat):
            return "Inconclusive: the regression did not estimate."
        biggest = self.loadings.abs().idxmax()
        tilt = f"largest exposure is {biggest} at {self.loadings[biggest]:+.2f}"
        if abs(self.alpha_tstat) < 2:
            return (
                f"No significant alpha (t = {self.alpha_tstat:.2f}). The known factors account "
                f"for the strategy's returns; {tilt}."
            )
        if self.r_squared > 0.5:
            return (
                f"Alpha survives at t = {self.alpha_tstat:.2f}, but {self.r_squared:.0%} of the "
                f"variance is factor exposure -- {tilt}. Treat the residual as the result."
            )
        return (
            f"Alpha of {self.alpha_annual:.2%} a year at t = {self.alpha_tstat:.2f}, with only "
            f"{self.r_squared:.0%} of variance explained by the factors ({tilt}). "
            f"Appraisal ratio {self.appraisal_ratio:.2f}."
        )


def _hac_cov(x: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance of the OLS coefficient vector.

    Overlapping holding periods make residuals autocorrelated; the OLS
    covariance would understate every standard error and overstate every t.
    """
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    u = x * resid[:, None]
    s = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = u[lag:].T @ u[:-lag]
        s += w * (gamma + gamma.T)
    return xtx_inv @ s @ xtx_inv * (n / max(n - k, 1))


def attribute_returns(
    returns: pd.Series,
    factors: pd.DataFrame,
    risk_free: pd.Series | None = None,
    hac_lags: int | None = None,
    already_excess: bool = True,
) -> FactorModel:
    """Regress a daily strategy return series on daily factor returns.

    Parameters
    ----------
    already_excess
        A market- and sector-neutral long/short book is self-financing, so its
        return is already an excess return and the risk-free rate must not be
        subtracted again. Pass ``False`` (with ``risk_free``) for a long-only
        series.
    """
    factors = factors if isinstance(factors.index, pd.DatetimeIndex) else normalise_factors(factors)
    y_series = returns.copy()
    y_series.index = pd.DatetimeIndex(y_series.index).tz_localize(None).normalize()

    df = pd.concat([y_series.rename("y"), factors], axis=1, join="inner").dropna()
    if len(df) < 60:
        raise ValueError(
            f"only {len(df)} overlapping observations between the strategy and the factors; "
            "need at least 60. Check the date ranges line up."
        )
    if not already_excess:
        if risk_free is None:
            raise ValueError("risk_free is required when already_excess is False")
        rf = risk_free.copy()
        rf.index = pd.DatetimeIndex(rf.index).tz_localize(None).normalize()
        df["y"] = df["y"] - rf.reindex(df.index).fillna(0.0)

    names = [c for c in factors.columns if c in df.columns]
    y = df["y"].to_numpy(dtype="float64")
    x = np.column_stack([np.ones(len(df)), df[names].to_numpy(dtype="float64")])

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    n, k = x.shape
    if hac_lags is None:
        hac_lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    se = np.sqrt(np.clip(np.diag(_hac_cov(x, resid, hac_lags)), 0, None))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1) if np.isfinite(r2) else np.nan

    alpha_d = float(beta[0])
    t_alpha = alpha_d / se[0] if se[0] > 0 else np.nan
    resid_vol = float(resid.std(ddof=k)) * np.sqrt(TRADING_DAYS)

    return FactorModel(
        alpha_daily=alpha_d,
        alpha_annual=alpha_d * TRADING_DAYS,
        alpha_tstat=float(t_alpha),
        alpha_pvalue=float(2 * sps.norm.sf(abs(t_alpha))) if np.isfinite(t_alpha) else np.nan,
        loadings=pd.Series(beta[1:], index=names),
        loading_tstats=pd.Series(
            np.divide(beta[1:], se[1:], out=np.full(k - 1, np.nan), where=se[1:] > 0), index=names
        ),
        r_squared=r2,
        adj_r_squared=adj_r2,
        residual_vol_annual=resid_vol,
        appraisal_ratio=(alpha_d * TRADING_DAYS) / resid_vol if resid_vol > 0 else np.nan,
        n_obs=n,
        hac_lags=hac_lags,
        factors_used=tuple(names),
    )


def synthetic_factors(panel, seed: int = 20260818) -> pd.DataFrame:
    """Factor proxies built from a :class:`ReturnPanel`, for offline use.

    Not a substitute for Ken French's data: it exists so the attribution code
    can be tested end to end without network access, and so the synthetic demo
    answers "is it just the market?" with an actual regression rather than an
    assertion. ``SMB`` and ``MOM`` are constructed from the panel's own
    cross-section; ``HML``, ``RMW`` and ``CMA`` have no analogue in the
    generating process and are drawn as noise, which is the honest thing to do
    -- they should come out insignificant, and if they do not, something is
    wrong with the regression rather than with the strategy.
    """
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(panel.sessions)
    rets = pd.DataFrame(panel.returns, index=idx, columns=panel.tickers)

    half = max(1, rets.shape[1] // 2)
    smb = rets.iloc[:, half:].mean(axis=1) - rets.iloc[:, :half].mean(axis=1)

    # 12-1 momentum: rank on the trailing 126 sessions, skipping the most
    # recent 21, which is the standard construction.
    trailing = rets.rolling(126, min_periods=60).mean().shift(21)
    rank = trailing.rank(axis=1, pct=True)
    mom = (rets.where(rank > 0.7).mean(axis=1) - rets.where(rank < 0.3).mean(axis=1)).fillna(0.0)

    return pd.DataFrame(
        {
            "Mkt-RF": pd.Series(panel.market, index=idx),
            "SMB": smb,
            "HML": pd.Series(rng.normal(0, 0.004, len(idx)), index=idx),
            "RMW": pd.Series(rng.normal(0, 0.003, len(idx)), index=idx),
            "CMA": pd.Series(rng.normal(0, 0.003, len(idx)), index=idx),
            "MOM": mom,
        }
    ).dropna(how="all")
