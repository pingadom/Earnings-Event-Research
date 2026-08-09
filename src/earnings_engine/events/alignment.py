"""Map an announcement instant to the first session you could have traded on.

This is the highest-leverage 100 lines in the repository. Every abnormal return
is measured relative to ``t0``, so an off-by-one here does not add noise -- it
systematically moves the announcement-day jump into or out of your window and
can turn a null result into a "discovery".

The rules
---------
Let the announcement land at local exchange time ``T`` on calendar day ``D``.

* ``T`` before the open on a session ``D``  -> ``t0 = D``. The market opens
  after the news; the whole reaction is capturable.
* ``T`` after the close on session ``D``     -> ``t0 = next session``.
* ``T`` during the session                   -> configurable. The default,
  ``next_open``, sets ``t0 = next session``. It is the conservative choice: a
  daily-close-to-close return on day ``D`` contains a pre-announcement stub you
  could not have traded. Choosing ``same_day`` here is a *decision to accept
  partial look-ahead* and is flagged as such on the output.
* ``T`` unknown                              -> treated per
  ``events.unknown_time_policy`` (default ``amc``, the modal US convention),
  and flagged, because these events are where mis-timing concentrates.
* ``D`` is a holiday or weekend              -> ``t0 = next session``.

``trade_open_ts`` is the UTC instant of ``t0``'s opening bell. Nothing that
became known after that instant may enter a feature for this event; see
:mod:`earnings_engine.events.pit`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EventsConfig
from ..utils.calendar import REGULAR_OPEN, TradingCalendar, default_calendar
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

VALID_TIMINGS = ("bmo", "amc", "during", "unknown")


class EventAlignmentError(ValueError):
    """Raised when events cannot be aligned to the trading calendar."""


def classify_timing(
    announced_at_utc: pd.Series,
    calendar: TradingCalendar | None = None,
    exchange_tz: str = "America/New_York",
) -> pd.Series:
    """Derive bmo/amc/during from timestamps, honouring early-close days."""
    calendar = calendar or default_calendar()
    local = pd.to_datetime(announced_at_utc, utc=True).dt.tz_convert(exchange_tz)
    day = local.dt.normalize().dt.tz_localize(None)
    minutes = local.dt.hour * 60 + local.dt.minute
    open_min = REGULAR_OPEN.hour * 60 + REGULAR_OPEN.minute
    close_min = np.array(
        [
            (calendar.close_time(d).hour * 60 + calendar.close_time(d).minute)
            if calendar.is_session(d)
            else open_min
            for d in day
        ]
    )
    out = np.where(
        minutes < open_min, "bmo", np.where(minutes.to_numpy() >= close_min, "amc", "during")
    )
    return pd.Series(out, index=announced_at_utc.index, dtype="string")


def align_events(
    events: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    config: EventsConfig | None = None,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Add ``t0``, ``t0_pos`` and ``trade_open_ts`` to an events frame.

    Parameters
    ----------
    events
        Frame conforming to :data:`earnings_engine.utils.frames.EVENTS`.
    prices
        Optional price panel. When supplied, events without enough prior
        history for a market-model estimation window are dropped, and the
        reason is logged rather than silently swallowed.
    """
    config = config or EventsConfig()
    calendar = calendar or default_calendar()
    if events.empty:
        raise EventAlignmentError("no events to align")
    if config.intraday_policy not in {"next_open", "same_day"}:
        raise EventAlignmentError(f"unknown intraday_policy {config.intraday_policy!r}")
    if config.unknown_time_policy not in {"amc", "bmo", "drop"}:
        raise EventAlignmentError(f"unknown unknown_time_policy {config.unknown_time_policy!r}")

    df = events.copy()
    ts = pd.to_datetime(df["announced_at_utc"], utc=True)
    local = ts.dt.tz_convert(config.exchange_tz)
    day = local.dt.normalize().dt.tz_localize(None)

    declared = df.get("timing")
    if declared is None:
        declared = pd.Series("unknown", index=df.index, dtype="string")
    declared = (
        declared.astype("string").str.lower().where(lambda s: s.isin(VALID_TIMINGS), "unknown")
    )

    # Where the flag is missing but the clock time is informative, recover it.
    derived = classify_timing(ts, calendar, config.exchange_tz)
    has_clock = (local.dt.hour != 0) | (local.dt.minute != 0)
    recovered = derived.where(has_clock, pd.NA)
    timing = declared.where(declared != "unknown", recovered).fillna("unknown")

    # Provenance of the timing flag, because mis-timed events are where the
    # measurement error in this whole exercise concentrates.
    source = pd.Series("declared", index=df.index, dtype="string")
    source = source.where(declared != "unknown", "derived")
    source = source.where(~(declared.eq("unknown") & recovered.isna()), "assumed")

    df["timing"] = timing
    df["timing_source"] = source
    df["timing_imputed"] = source != "declared"

    effective = timing.copy()
    if config.unknown_time_policy == "drop":
        drop_mask = effective.eq("unknown")
        n_drop = int(drop_mask.sum())
        if n_drop:
            log.info("dropping %d event(s) with unknown announcement time", n_drop)
        if n_drop == len(df):
            raise EventAlignmentError(
                "unknown_time_policy='drop' removed every event: none of them carry a "
                "usable announcement time"
            )
        df = df.loc[~drop_mask].copy()
        day = day.loc[df.index]
        effective = effective.loc[df.index]
        timing = timing.loc[df.index]
    else:
        effective = effective.where(effective != "unknown", config.unknown_time_policy)

    if config.intraday_policy == "next_open":
        effective = effective.where(effective != "during", "amc")
        df["accepts_intraday_lookahead"] = False
    else:
        # Opting into same-day treatment means part of the day's close-to-close
        # return happened before the news. Recorded, not hidden.
        df["accepts_intraday_lookahead"] = timing.eq("during")

    # bmo on a session -> that session; everything else -> the next session.
    # ('during' only reaches here under intraday_policy='same_day'.)
    is_session = np.array([calendar.is_session(d) for d in day])
    same_day = effective.isin(["bmo", "during"]).to_numpy() & is_session

    t0 = np.empty(len(df), dtype="datetime64[ns]")
    ok = np.ones(len(df), dtype=bool)
    for i, (d, same) in enumerate(zip(day.to_numpy(), same_day, strict=False)):
        d_ts = pd.Timestamp(d)
        try:
            t0[i] = d_ts if same else calendar.next_session(d_ts)
        except IndexError:
            ok[i] = False
            t0[i] = np.datetime64("NaT")
    if (~ok).any():
        log.warning(
            "%d event(s) fall outside the calendar range and were dropped", int((~ok).sum())
        )
    df = df.loc[ok].copy()
    df["t0"] = pd.to_datetime(t0[ok])

    pos = calendar.positions_of(df["t0"])
    df["t0_pos"] = pos
    df = df.loc[df["t0_pos"] >= 0].copy()

    # The instant you could first transact: t0's opening bell.
    open_local = (
        df["t0"] + pd.Timedelta(hours=REGULAR_OPEN.hour, minutes=REGULAR_OPEN.minute)
    ).dt.tz_localize(config.exchange_tz, ambiguous=True, nonexistent="shift_forward")
    df["trade_open_ts"] = open_local.dt.tz_convert("UTC")

    # Under same-day intraday treatment the tradable instant is the
    # announcement itself, not the (already-past) opening bell.
    intraday = df["accepts_intraday_lookahead"].astype(bool)
    if intraday.any():
        df.loc[intraday, "trade_open_ts"] = pd.to_datetime(
            df.loc[intraday, "announced_at_utc"], utc=True
        )

    # Sanity: the announcement must not post-date the open we claim to trade at.
    bad = (pd.to_datetime(df["announced_at_utc"], utc=True) > df["trade_open_ts"]) & ~intraday
    if bad.any():
        n = int(bad.sum())
        log.warning("%d event(s) announced after their own t0 open; re-aligning to next session", n)
        fixed = [calendar.next_session(t) for t in df.loc[bad, "t0"]]
        df.loc[bad, "t0"] = pd.to_datetime(fixed)
        df.loc[bad, "t0_pos"] = calendar.positions_of(df.loc[bad, "t0"])
        open_local = (
            df.loc[bad, "t0"] + pd.Timedelta(hours=REGULAR_OPEN.hour, minutes=REGULAR_OPEN.minute)
        ).dt.tz_localize(config.exchange_tz, ambiguous=True, nonexistent="shift_forward")
        df.loc[bad, "trade_open_ts"] = open_local.dt.tz_convert("UTC")

    df["days_since_period_end"] = (
        df["t0"] - pd.to_datetime(df["period_end"], errors="coerce")
    ).dt.days

    if prices is not None and config.min_history_days > 0:
        df = _require_history(df, prices, calendar, config.min_history_days)

    return df.sort_values(["t0", "ticker"]).reset_index(drop=True)


def _require_history(df, prices, calendar, min_days: int) -> pd.DataFrame:
    """Drop events without ``min_days`` of prior price history."""
    first = (
        prices.groupby("ticker", observed=True)["date"].min().rename("first_price_date")
    )
    merged = df.merge(first, left_on="ticker", right_index=True, how="left")
    first_pos = calendar.positions_of(merged["first_price_date"].fillna(pd.Timestamp("2100-01-01")))
    have = merged["t0_pos"].to_numpy() - first_pos
    keep = have >= min_days
    dropped = int((~keep).sum())
    if dropped:
        log.info(
            "dropped %d/%d event(s) with fewer than %d prior sessions of price history",
            dropped,
            len(merged),
            min_days,
        )
    return merged.loc[keep].drop(columns=["first_price_date"]).copy()
