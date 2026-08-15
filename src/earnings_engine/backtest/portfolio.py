"""Position construction: turning a signal into a tradable book.

Three decisions carry most of the weight, and each of them is a place where a
backtest can quietly stop being honest.

**Ranking against the past, not the present.**
The obvious way to pick the top quintile is to rank every event in a month
against the others in that month. It is also a look-ahead: a firm reporting on
the 3rd would have to know what firms reporting on the 27th were going to
print. So quantile breakpoints here come from a *trailing* window of events
strictly before ``t0``. Slightly noisier, entirely implementable.

**Entering after the gap.**
Positions open ``entry_offset`` sessions after ``t0`` (one by default). The
announcement move happens on the opening gap, before anyone can transact on it.
Counting it as strategy profit is the largest single source of overstated PEAD
results.

**Sector neutrality at the book level.**
Because holding periods overlap, the set of live positions on any given day is
a mixture of vintages, and there is no guarantee the longs and shorts balance
within each sector. So neutrality is imposed daily on the *whole book* (see
:func:`earnings_engine.backtest.engine.build_daily_book`) rather than assumed
at entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger(__name__)


def build_positions(
    predictions: pd.DataFrame,
    *,
    pred_col: str = "prediction",
    date_col: str = "t0",
    sector_col: str = "sector",
    quantiles: int = 5,
    holding_days: int = 20,
    entry_offset: int = 1,
    lookback_days: int = 252,
    min_lookback_events: int = 100,
    sector_breakpoints: bool = False,
) -> pd.DataFrame:
    """Assign a side to each event using breakpoints estimated on past events only.

    Returns one row per traded event with ``side`` in ``{-1, +1}``.
    """
    df = predictions.dropna(subset=[pred_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=["event_id", "ticker", "t0", "side"])

    dates = pd.to_datetime(df[date_col])
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)
    df[date_col] = dates.dt.normalize()
    df = df.sort_values(date_col).reset_index(drop=True)

    if sector_breakpoints and sector_col in df.columns:
        pieces = [
            _assign_sides(g, pred_col, date_col, quantiles, lookback_days, min_lookback_events)
            for _, g in df.groupby(sector_col, sort=False)
        ]
        book = pd.concat([p for p in pieces if not p.empty], ignore_index=True)
    else:
        book = _assign_sides(df, pred_col, date_col, quantiles, lookback_days, min_lookback_events)

    if book.empty:
        log.warning(
            "no positions: the first %d events are used to seed the trailing breakpoints",
            min_lookback_events,
        )
        return book
    book["holding_days"] = holding_days
    book["entry_offset"] = entry_offset
    keep = [
        c
        for c in ("event_id", "ticker", "sector", date_col, "side", "pct_rank",
                  "holding_days", "entry_offset", pred_col)
        if c in book.columns
    ]
    log.info(
        "positions: %d long, %d short from %d scored events",
        int((book["side"] > 0).sum()),
        int((book["side"] < 0).sum()),
        len(df),
    )
    return book[keep].sort_values([date_col, "ticker"]).reset_index(drop=True)


def _assign_sides(
    df: pd.DataFrame,
    pred_col: str,
    date_col: str,
    quantiles: int,
    lookback_days: int,
    min_lookback_events: int,
) -> pd.DataFrame:
    """Percentile of each prediction within a strictly-past trailing window."""
    df = df.sort_values(date_col).reset_index(drop=True)
    values = df[pred_col].to_numpy(dtype="float64")
    days = df[date_col].to_numpy(dtype="datetime64[D]").astype("int64")

    lo_cut, hi_cut = 1.0 / quantiles, 1.0 - 1.0 / quantiles
    pct = np.full(len(df), np.nan)

    start = 0
    for i in range(len(df)):
        # Strictly earlier events only, within the trailing window.
        while days[start] < days[i] - lookback_days:
            start += 1
        end = np.searchsorted(days, days[i], side="left")
        if end - start < min_lookback_events:
            continue
        past = values[start:end]
        pct[i] = float((past < values[i]).mean())

    out = df.copy()
    out["pct_rank"] = pct
    out["side"] = np.where(pct >= hi_cut, 1.0, np.where(pct <= lo_cut, -1.0, 0.0))
    return out[np.isfinite(pct) & (out["side"] != 0)].reset_index(drop=True)
