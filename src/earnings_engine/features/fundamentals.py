"""Fundamental features: levels are noise, *changes* are the signal.

The hypothesis under test is about information released at the announcement,
so the features are constructed as changes relative to the firm's own history
rather than as cross-sectional levels. A 32% gross margin means nothing on its
own; a gross margin 300bp above the same quarter last year, in a firm whose
revenue also accelerated, is a statement about the quarter.

Seasonality is handled by differencing against the *same quarter one year ago*
(``t-4``) rather than the previous quarter. Retailers do not have a bad Q1
because Q4 was good.

Every feature inherits ``available_from_utc`` from the *latest* input line item
it depends on -- if a feature needs four quarters of history, it is only
knowable once the most recent of those four was published.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

#: Feature name -> short description, used for reporting and documentation.
FUNDAMENTAL_FEATURES: dict[str, str] = {
    "revenue_growth_yoy": "Revenue vs the same quarter last year",
    "revenue_growth_accel": "Change in YoY revenue growth vs the prior quarter",
    "gross_margin": "Gross profit / revenue",
    "gross_margin_delta_yoy": "Gross margin change vs the same quarter last year",
    "operating_margin": "Operating income / revenue",
    "operating_margin_delta_yoy": "Operating margin change vs the same quarter last year",
    "net_margin": "Net income / revenue",
    "eps_growth_yoy": "Diluted EPS vs the same quarter last year (sign-safe)",
    "fcf": "Cash from operations less capital expenditure",
    "fcf_margin": "Free cash flow / revenue",
    "fcf_margin_delta_yoy": "FCF margin change vs the same quarter last year",
    "accruals": "(Net income - CFO) / total assets: earnings quality",
    "net_debt_to_assets": "(Total debt - cash) / total assets",
    "net_debt_delta_yoy": "Change in net debt / assets vs the same quarter last year",
    "asset_growth_yoy": "Total assets vs the same quarter last year",
    "share_count_change_yoy": "Diluted share count vs the same quarter last year",
}


def _pivot(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide per (ticker, period_end), keeping the publication stamp."""
    wide = fundamentals.pivot_table(
        index=["ticker", "period_end"], columns="item", values="value", aggfunc="last"
    )
    stamp = fundamentals.groupby(["ticker", "period_end"])["available_from_utc"].max()
    out = wide.join(stamp).reset_index()
    return out.sort_values(["ticker", "period_end"])


def _safe_growth(current: pd.Series, prior: pd.Series) -> pd.Series:
    """Growth rate that behaves when the base is negative or near zero.

    ``(x_t - x_{t-4}) / |x_{t-4}|`` -- using the absolute value of the base
    keeps the sign meaningful when a company swings from a loss to a profit,
    which the naive ratio gets backwards.
    """
    denom = prior.abs()
    out = (current - prior) / denom.where(denom > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def build_fundamental_features(
    fundamentals: pd.DataFrame, winsorize: tuple[float, float] | None = (0.01, 0.99)
) -> pd.DataFrame:
    """Build the fundamental feature panel from long-form line items.

    Returns one row per ``(ticker, period_end)`` with ``available_from_utc``
    preserved, ready for an as-of join onto events.
    """
    df = _pivot(fundamentals)
    for col in (
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps_diluted",
        "cfo",
        "capex",
        "total_debt",
        "cash",
        "total_assets",
        "shares_diluted",
    ):
        if col not in df.columns:
            df[col] = np.nan
            log.debug("fundamentals: %s missing, features depending on it will be NaN", col)

    g = df.groupby("ticker", sort=False)
    meta = {"ticker", "period_end", "available_from_utc"}
    lag4 = {c: g[c].shift(4) for c in df.columns if c not in meta}

    out = df[["ticker", "period_end", "available_from_utc"]].copy()
    rev, rev4 = df["revenue"], lag4["revenue"]

    out["revenue_growth_yoy"] = _safe_growth(rev, rev4)
    out["revenue_growth_accel"] = out.groupby(df["ticker"])["revenue_growth_yoy"].diff()

    with np.errstate(invalid="ignore", divide="ignore"):
        out["gross_margin"] = df["gross_profit"] / rev.where(rev != 0)
        out["operating_margin"] = df["operating_income"] / rev.where(rev != 0)
        out["net_margin"] = df["net_income"] / rev.where(rev != 0)
        gm4 = lag4["gross_profit"] / rev4.where(rev4 != 0)
        om4 = lag4["operating_income"] / rev4.where(rev4 != 0)
        out["gross_margin_delta_yoy"] = out["gross_margin"] - gm4
        out["operating_margin_delta_yoy"] = out["operating_margin"] - om4

        out["eps_growth_yoy"] = _safe_growth(df["eps_diluted"], lag4["eps_diluted"])

        fcf = df["cfo"] - df["capex"]
        fcf4 = lag4["cfo"] - lag4["capex"]
        out["fcf"] = fcf
        out["fcf_margin"] = fcf / rev.where(rev != 0)
        out["fcf_margin_delta_yoy"] = out["fcf_margin"] - fcf4 / rev4.where(rev4 != 0)

        assets = df["total_assets"]
        out["accruals"] = (df["net_income"] - df["cfo"]) / assets.where(assets > 0)
        net_debt = (df["total_debt"] - df["cash"]) / assets.where(assets > 0)
        net_debt4 = (lag4["total_debt"] - lag4["cash"]) / lag4["total_assets"].where(
            lag4["total_assets"] > 0
        )
        out["net_debt_to_assets"] = net_debt
        out["net_debt_delta_yoy"] = net_debt - net_debt4
        out["asset_growth_yoy"] = _safe_growth(assets, lag4["total_assets"])
        out["share_count_change_yoy"] = _safe_growth(df["shares_diluted"], lag4["shares_diluted"])

    feature_cols = [c for c in out.columns if c in FUNDAMENTAL_FEATURES]
    if winsorize is not None:
        out[feature_cols] = _winsorize(out[feature_cols], *winsorize)
    return out.reset_index(drop=True)


def _winsorize(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    lo = df.quantile(lower)
    hi = df.quantile(upper)
    return df.clip(lower=lo, upper=hi, axis=1)
