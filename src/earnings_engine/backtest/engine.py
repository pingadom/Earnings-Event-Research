"""The backtest loop and its performance statistics.

Construction
------------
Every traded event contributes a position that is live from ``entry_offset``
sessions after ``t0`` for ``holding_days`` sessions. Holding periods overlap, so
on any given day the book is a mixture of up to ``holding_days`` vintages -- the
Jegadeesh-Titman overlapping-portfolio construction.

The book is rebuilt from scratch each day rather than accumulated:

1. collect every live position and its raw side (+1/-1);
2. if sector-neutral, demean the raw weights *within each sector* so no sector
   carries net exposure -- imposed on the actual book, because overlapping
   vintages give no guarantee that entries balance;
3. scale so gross exposure equals the target, then cap per-name weight and
   renormalise once.

Traded notional is then the day-on-day change in the weight vector, summed in
absolute value. That captures entries, exits *and* the rebalancing implied by
step 2 -- which a turnover assumption based on entries alone would miss.

P&L uses *abnormal* returns, not raw ones: the strategy is market- and
sector-neutral by construction, so charging it with the market's return would
be measuring something else.

Statistics carry a Newey-West correction, because overlapping holding periods
make daily P&L autocorrelated and the naive Sharpe standard error too small.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..returns.stats import newey_west_tstat
from ..utils.logging_utils import get_logger
from .costs import CostModel, apply_costs, cost_sensitivity

log = get_logger(__name__)

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """Daily P&L plus the summary statistics you would put in a write-up."""

    daily: pd.DataFrame
    stats: dict
    positions: pd.DataFrame
    book: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def equity_curve(self) -> pd.Series:
        return (1.0 + self.daily["net"]).cumprod()

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.stats])


def build_daily_book(
    positions: pd.DataFrame,
    daily_ar: pd.DataFrame,
    *,
    ar_column: str,
    entry_offset: int,
    holding_days: int,
    sector_neutral: bool,
    gross_exposure: float,
    max_weight: float,
    sessions: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Expand positions into a (date, event) weight panel with abnormal returns."""
    first, last = entry_offset, entry_offset + holding_days - 1
    ar = daily_ar[(daily_ar["rel_day"] >= first) & (daily_ar["rel_day"] <= last)]
    cols = [c for c in ("event_id", "side", "sector") if c in positions.columns]
    merged = ar.merge(positions[cols], on="event_id", how="inner")
    if merged.empty:
        raise ValueError("positions and abnormal returns share no events")
    merged = _attach_calendar_dates(merged, sessions)
    merged = merged.dropna(subset=[ar_column])

    if sector_neutral and "sector" in merged.columns and merged["sector"].notna().any():
        # Demean within (date, sector): each sector's net exposure becomes zero.
        sector_mean = merged.groupby(["date", "sector"], observed=True)["side"].transform("mean")
        merged["raw_weight"] = merged["side"] - sector_mean
    else:
        if sector_neutral:
            log.warning("no sector labels available; running a non-neutral book")
        merged["raw_weight"] = merged["side"] - merged.groupby("date")["side"].transform("mean")

    gross = merged.groupby("date")["raw_weight"].transform(lambda s: s.abs().sum())
    merged["weight"] = merged["raw_weight"] / gross.where(gross > 0) * gross_exposure
    capped = merged["weight"].clip(-max_weight, max_weight)
    n_capped = int((capped != merged["weight"]).sum())
    if n_capped:
        merged["weight"] = capped
        gross2 = merged.groupby("date")["weight"].transform(lambda s: s.abs().sum())
        merged["weight"] = merged["weight"] / gross2.where(gross2 > 0) * gross_exposure
        log.info("weight cap at +/-%.3f bound %d position-days", max_weight, n_capped)

    merged["contrib"] = merged["weight"] * merged[ar_column]
    return merged.dropna(subset=["weight"])


def run_backtest(
    positions: pd.DataFrame,
    daily_ar: pd.DataFrame,
    *,
    ar_column: str = "ar_sector_neutral",
    entry_offset: int = 1,
    holding_days: int = 20,
    sector_neutral: bool = True,
    gross_exposure: float = 1.0,
    max_weight: float = 0.02,
    cost_model: CostModel | None = None,
    sessions: pd.DatetimeIndex | None = None,
) -> BacktestResult:
    """Run the overlapping-portfolio backtest and net it against costs."""
    cost_model = cost_model or CostModel()
    if positions.empty:
        raise ValueError("no positions to backtest")
    if ar_column not in daily_ar.columns:
        raise KeyError(f"daily abnormal returns have no column {ar_column!r}")

    book = build_daily_book(
        positions,
        daily_ar,
        ar_column=ar_column,
        entry_offset=entry_offset,
        holding_days=holding_days,
        sector_neutral=sector_neutral,
        gross_exposure=gross_exposure,
        max_weight=max_weight,
        sessions=sessions,
    )

    gross_ret = book.groupby("date")["contrib"].sum().sort_index()
    traded = _traded_notional(book).reindex(gross_ret.index).fillna(0.0)

    pnl = apply_costs(gross_ret, traded, cost_model)
    agg = book.groupby("date").agg(
        gross_exposure=("weight", lambda s: s.abs().sum()),
        net_exposure=("weight", "sum"),
        n_positions=("event_id", "nunique"),
    )
    pnl = pnl.join(agg)

    stats = performance_stats(pnl["net"], pnl["gross"], pnl["traded"])
    stats.update(
        {
            "ar_column": ar_column,
            "entry_offset": entry_offset,
            "holding_days": holding_days,
            "sector_neutral": sector_neutral,
            "one_way_cost_bps": float(cost_model.one_way_bps()),
            "avg_positions": float(pnl["n_positions"].mean()),
        }
    )
    sens = cost_sensitivity(pnl["gross"], pnl["traded"], cost_model)
    return BacktestResult(
        daily=pnl, stats=stats, positions=positions, book=book, cost_sensitivity=sens
    )


def _traded_notional(book: pd.DataFrame) -> pd.Series:
    """Sum of |w_t - w_{t-1}| per date, treating absent positions as zero weight."""
    wide = book.pivot_table(index="date", columns="event_id", values="weight", aggfunc="sum")
    wide = wide.sort_index().fillna(0.0)
    delta = wide.diff()
    delta.iloc[0] = wide.iloc[0]
    return delta.abs().sum(axis=1)


def _attach_calendar_dates(merged: pd.DataFrame, sessions: pd.DatetimeIndex | None) -> pd.DataFrame:
    """Turn (t0, rel_day) into an actual session date."""
    if sessions is None:
        from ..utils.calendar import default_calendar  # noqa: PLC0415

        sessions = default_calendar().sessions
    pos = pd.Series(np.arange(len(sessions)), index=pd.DatetimeIndex(sessions).normalize())
    t0 = pd.to_datetime(merged["t0"]).dt.normalize()
    base = pos.reindex(t0).to_numpy(dtype="float64")
    idx = base + merged["rel_day"].to_numpy()
    valid = np.isfinite(base) & (idx >= 0) & (idx < len(sessions))
    out = merged.loc[valid].copy()
    out["date"] = pd.DatetimeIndex(sessions)[idx[valid].astype(int)]
    return out


def performance_stats(
    net: pd.Series, gross: pd.Series | None = None, traded: pd.Series | None = None
) -> dict:
    """Annualised performance with HAC-corrected significance."""
    net = net.dropna()
    n = len(net)
    if n < 2:
        return {"n_days": n}
    ann_ret = float(net.mean() * TRADING_DAYS)
    ann_vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    t_nw, se_nw = newey_west_tstat(net)
    # Sharpe standard error implied by the HAC standard error of the mean.
    sharpe_se = (se_nw * TRADING_DAYS / ann_vol) if ann_vol > 0 else np.nan

    curve = (1.0 + net).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    downside = net[net < 0]
    sortino = (
        ann_ret / (float(downside.std(ddof=1)) * np.sqrt(TRADING_DAYS))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else np.nan
    )

    stats = {
        "n_days": n,
        "start": str(pd.Timestamp(net.index[0]).date()),
        "end": str(pd.Timestamp(net.index[-1]).date()),
        "ann_return_net": ann_ret,
        "ann_vol": ann_vol,
        "sharpe_net": sharpe,
        "sharpe_se_nw": float(sharpe_se) if sharpe_se == sharpe_se else np.nan,
        "tstat_nw": float(t_nw) if t_nw == t_nw else np.nan,
        "sortino": sortino,
        "max_drawdown": float(drawdown.min()),
        "hit_rate_daily": float((net > 0).mean()),
        "skew": float(net.skew()),
        "kurtosis": float(net.kurtosis()),
    }
    if gross is not None:
        g = gross.dropna()
        gvol = float(g.std(ddof=1) * np.sqrt(TRADING_DAYS))
        stats["ann_return_gross"] = float(g.mean() * TRADING_DAYS)
        stats["sharpe_gross"] = stats["ann_return_gross"] / gvol if gvol > 0 else np.nan
    if traded is not None:
        stats["ann_turnover"] = float(traded.mean() * TRADING_DAYS)
        stats["ann_cost_drag"] = stats.get("ann_return_gross", np.nan) - ann_ret
    return stats
