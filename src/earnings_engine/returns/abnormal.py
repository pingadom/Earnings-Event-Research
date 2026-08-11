"""Abnormal returns: CAR and BHAR under several definitions of "normal".

Why more than one estimator
---------------------------
"Abnormal return" is not a single quantity; it is whatever is left after you
subtract a benchmark, and the benchmark is a modelling choice. Reporting one
number hides how much of the result is the choice rather than the data. Three
are computed here, in increasing order of what they control for:

``market_adjusted``   ``AR = r_i - r_m``. No estimation, no parameters, nothing
                      to overfit. Fine at short horizons where beta dispersion
                      contributes little.
``market_model``      ``AR = r_i - (alpha_i + beta_i * r_m)`` with parameters
                      from a pre-event estimation window. The textbook
                      Brown-Warner specification.
``sector_neutral``    ``AR = r_i - mean(r_j : j in same sector, j != i)``. The
                      one that matters for the actual hypothesis: if you want
                      to claim earnings information predicts returns, you need
                      to show it is not just "energy did well this month".
                      Excluding the stock itself from its own benchmark is
                      essential; including it mechanically shrinks the measured
                      abnormal return toward zero.

CAR vs BHAR
-----------
``CAR`` sums daily abnormal returns; ``BHAR`` compounds the stock and the
benchmark separately and differences the result. They answer different
questions -- CAR is the right object for statistical testing, BHAR is the right
object for "what would I have made" -- and they diverge as the horizon grows.
Both are reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ReturnsConfig
from ..utils.logging_utils import get_logger
from .estimation import estimation_market_moments, fit_market_model

log = get_logger(__name__)

ESTIMATORS = ("market_adjusted", "market_model", "sector_neutral")


@dataclass
class ReturnPanel:
    """Wide, calendar-aligned daily returns plus benchmarks.

    Holding this as dense numpy rather than a long dataframe is what lets the
    event-window slicing be a gather instead of 50,000 groupby operations.
    """

    sessions: pd.DatetimeIndex
    tickers: list[str]
    returns: np.ndarray  # (T, N)
    market: np.ndarray  # (T,)
    sector_benchmark: np.ndarray | None  # (T, N) leave-one-out sector mean
    ticker_pos: dict[str, int]

    @classmethod
    def from_prices(
        cls,
        prices: pd.DataFrame,
        market_symbol: str = "SPY",
        sector_map: dict[str, str] | None = None,
        winsorize: tuple[float, float] | None = (0.005, 0.995),
    ) -> ReturnPanel:
        wide = (
            prices.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
            .sort_index()
        )
        rets = wide.pct_change(fill_method=None)

        if market_symbol not in rets.columns:
            raise KeyError(
                f"market symbol {market_symbol!r} missing from the price panel; "
                "add it to the ticker list before computing abnormal returns"
            )
        market = rets[market_symbol].to_numpy(dtype="float64")
        asset_rets = rets.drop(columns=[market_symbol])

        values = asset_rets.to_numpy(dtype="float64")
        if winsorize is not None:
            lo, hi = np.nanquantile(values, winsorize[0]), np.nanquantile(values, winsorize[1])
            n_clipped = int(np.nansum((values < lo) | (values > hi)))
            values = np.clip(values, lo, hi)
            log.debug("winsorised %d daily observations to [%.4f, %.4f]", n_clipped, lo, hi)

        tickers = list(asset_rets.columns)
        sector_bench = (
            cls._leave_one_out_sector(values, tickers, sector_map) if sector_map else None
        )
        return cls(
            sessions=pd.DatetimeIndex(wide.index),
            tickers=tickers,
            returns=values,
            market=market,
            sector_benchmark=sector_bench,
            ticker_pos={t: i for i, t in enumerate(tickers)},
        )

    @staticmethod
    def _leave_one_out_sector(values, tickers, sector_map) -> np.ndarray:
        """Equal-weighted sector return excluding the stock being benchmarked."""
        out = np.full_like(values, np.nan)
        sectors: dict[str, list[int]] = {}
        for i, t in enumerate(tickers):
            s = sector_map.get(t)
            if s is not None and s == s:  # not NaN
                sectors.setdefault(s, []).append(i)
        for _sector, idx in sectors.items():
            if len(idx) < 2:
                continue
            block = values[:, idx]
            present = np.isfinite(block)
            total = np.where(present, block, 0.0).sum(axis=1, keepdims=True)
            count = present.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                loo = (total - np.where(present, block, 0.0)) / np.maximum(count - present, 1)
            loo = np.where(count > 1, loo, np.nan)
            out[:, idx] = np.where(present, loo, np.nan)
        return out

    def positions(self, tickers: pd.Series) -> np.ndarray:
        return np.array([self.ticker_pos.get(t, -1) for t in tickers], dtype="int64")

    def session_positions(self, dates) -> np.ndarray:
        idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
        pos = pd.Series(np.arange(len(self.sessions)), index=self.sessions.normalize())
        out = pos.reindex(idx).to_numpy(dtype="float64")
        return np.where(np.isnan(out), -1, out).astype("int64")


@dataclass
class AbnormalReturnResult:
    """Everything an event study produces, in one place."""

    #: One row per (event, relative day) with the AR under each estimator.
    daily: pd.DataFrame
    #: One row per event with CAR/BHAR per estimator per window.
    summary: pd.DataFrame
    #: Market-model diagnostics per event.
    fit: pd.DataFrame
    windows: tuple[tuple[int, int], ...]
    estimators: tuple[str, ...]

    def car_column(self, estimator: str, window: tuple[int, int]) -> str:
        return f"car_{estimator}_{window[0]}_{window[1]}"

    def bhar_column(self, estimator: str, window: tuple[int, int]) -> str:
        return f"bhar_{estimator}_{window[0]}_{window[1]}"


def compute_abnormal_returns(
    events: pd.DataFrame,
    panel: ReturnPanel,
    config: ReturnsConfig | None = None,
) -> AbnormalReturnResult:
    """Run the event study.

    ``events`` must already carry ``t0`` (see
    :func:`earnings_engine.events.align_events`).
    """
    config = config or ReturnsConfig()
    unknown = set(config.estimators) - set(ESTIMATORS)
    if unknown:
        raise ValueError(f"unknown estimator(s) {sorted(unknown)}; valid: {ESTIMATORS}")
    if "t0" not in events.columns:
        raise KeyError("events frame needs a 't0' column; run align_events first")

    ev = events.copy().reset_index(drop=True)
    asset_idx = panel.positions(ev["ticker"])
    t0_pos = panel.session_positions(ev["t0"])

    keep = (asset_idx >= 0) & (t0_pos >= 0)
    if not keep.all():
        log.info("dropping %d event(s) with no matching price series/session", int((~keep).sum()))
    ev = ev.loc[keep].reset_index(drop=True)
    asset_idx = asset_idx[keep]
    t0_pos = t0_pos[keep]
    if ev.empty:
        raise ValueError("no events survived alignment against the price panel")

    max_offset = max(hi for _, hi in config.windows)
    min_offset = min(lo for lo, _ in config.windows)
    offsets = np.arange(min_offset, max_offset + 1)
    rows = t0_pos[:, None] + offsets[None, :]
    n_t = panel.returns.shape[0]
    in_range = (rows >= 0) & (rows < n_t)
    safe = np.clip(rows, 0, n_t - 1)

    r_i = np.where(in_range, panel.returns[safe, asset_idx[:, None]], np.nan)
    r_m = np.where(in_range, panel.market[safe], np.nan)

    fit = fit_market_model(
        panel.returns,
        panel.market,
        t0_pos,
        asset_idx,
        start_offset=config.estimation_start,
        end_offset=config.estimation_end,
        min_obs=config.min_estimation_obs,
    )
    mean_x, sxx = estimation_market_moments(
        panel.market, t0_pos, config.estimation_start, config.estimation_end
    )

    benchmarks: dict[str, np.ndarray] = {}
    if "market_adjusted" in config.estimators:
        benchmarks["market_adjusted"] = r_m
    if "market_model" in config.estimators:
        benchmarks["market_model"] = fit.alpha[:, None] + fit.beta[:, None] * r_m
    if "sector_neutral" in config.estimators:
        if panel.sector_benchmark is None:
            raise ValueError(
                "sector_neutral requested but the panel has no sector map; pass "
                "sector_map to ReturnPanel.from_prices"
            )
        benchmarks["sector_neutral"] = np.where(
            in_range, panel.sector_benchmark[safe, asset_idx[:, None]], np.nan
        )

    daily = _daily_frame(ev, offsets, r_i, benchmarks)
    summary = _summarise(ev, offsets, r_i, benchmarks, config.windows)
    summary = summary.merge(
        pd.DataFrame(
            {
                "event_id": ev["event_id"],
                "mm_alpha": fit.alpha,
                "mm_beta": fit.beta,
                "mm_sigma": fit.sigma,
                "mm_n_obs": fit.n_obs,
                "mm_r2": fit.r_squared,
                "est_market_mean": mean_x,
                "est_market_sxx": sxx,
            }
        ),
        on="event_id",
        how="left",
    )
    # Window sum of market deviations from its estimation-window mean. Needed
    # for the Patell prediction-error variance inflation used by the BMP test.
    off_pos = {int(o): i for i, o in enumerate(offsets)}
    for lo, hi in config.windows:
        sl = slice(off_pos[int(lo)], off_pos[int(hi)] + 1)
        summary[f"mktdev_{lo}_{hi}"] = np.nansum(r_m[:, sl] - mean_x[:, None], axis=1)

    fit_df = summary[
        ["event_id", "ticker", "mm_alpha", "mm_beta", "mm_sigma", "mm_n_obs", "mm_r2"]
    ].copy()
    return AbnormalReturnResult(
        daily=daily,
        summary=summary,
        fit=fit_df,
        windows=tuple(tuple(w) for w in config.windows),
        estimators=tuple(config.estimators),
    )


def _daily_frame(ev, offsets, r_i, benchmarks) -> pd.DataFrame:
    n_events, n_off = r_i.shape
    base = pd.DataFrame(
        {
            "event_id": np.repeat(ev["event_id"].to_numpy(), n_off),
            "ticker": np.repeat(ev["ticker"].to_numpy(), n_off),
            "t0": np.repeat(ev["t0"].to_numpy(), n_off),
            "rel_day": np.tile(offsets, n_events),
            "ret": r_i.reshape(-1),
        }
    )
    for name, bench in benchmarks.items():
        base[f"ar_{name}"] = (r_i - bench).reshape(-1)
        base[f"bench_{name}"] = np.broadcast_to(bench, r_i.shape).reshape(-1)
    return base


def _summarise(ev, offsets, r_i, benchmarks, windows) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "event_id": ev["event_id"].to_numpy(),
            "ticker": ev["ticker"].to_numpy(),
            "t0": ev["t0"].to_numpy(),
        }
    )
    for col in ("period_end", "fiscal_quarter", "timing", "timing_imputed", "trade_open_ts"):
        if col in ev.columns:
            out[col] = ev[col].to_numpy()

    off_pos = {int(o): i for i, o in enumerate(offsets)}
    for name, bench in benchmarks.items():
        ar = r_i - bench
        for lo, hi in windows:
            sl = slice(off_pos[int(lo)], off_pos[int(hi)] + 1)
            seg_ar = ar[:, sl]
            seg_r = r_i[:, sl]
            seg_b = bench[:, sl]
            complete = np.isfinite(seg_ar).all(axis=1)
            car = np.where(complete, np.nansum(seg_ar, axis=1), np.nan)
            bh = np.where(
                complete,
                np.nanprod(1.0 + seg_r, axis=1) - np.nanprod(1.0 + seg_b, axis=1),
                np.nan,
            )
            out[f"car_{name}_{lo}_{hi}"] = car
            out[f"bhar_{name}_{lo}_{hi}"] = bh
            out[f"nobs_{name}_{lo}_{hi}"] = np.isfinite(seg_ar).sum(axis=1)
    return out
