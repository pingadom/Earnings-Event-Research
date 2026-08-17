"""Announcement-to-session alignment: the highest-leverage logic in the repo."""

from __future__ import annotations

import pandas as pd
import pytest

from earnings_engine.config import EventsConfig
from earnings_engine.events import EventAlignmentError, align_events
from earnings_engine.utils.frames import EVENTS


def _event(ts_local: str, timing: str, ticker: str = "AAA") -> pd.DataFrame:
    ts = pd.Timestamp(ts_local).tz_localize("America/New_York").tz_convert("UTC")
    return EVENTS.validate(
        pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "event_id": f"{ticker}-{ts_local}",
                    "announced_at_utc": ts,
                    "timing": timing,
                    "period_end": pd.Timestamp("2024-03-31"),
                    "fiscal_quarter": "2024Q1",
                }
            ]
        )
    )


def test_bmo_trades_same_session():
    out = align_events(_event("2024-05-02 07:00", "bmo"), config=EventsConfig())
    assert out["t0"].iloc[0] == pd.Timestamp("2024-05-02")


def test_amc_trades_next_session():
    out = align_events(_event("2024-05-02 16:15", "amc"), config=EventsConfig())
    assert out["t0"].iloc[0] == pd.Timestamp("2024-05-03")


def test_amc_on_friday_rolls_to_monday():
    out = align_events(_event("2024-05-03 16:15", "amc"), config=EventsConfig())
    assert out["t0"].iloc[0] == pd.Timestamp("2024-05-06")


def test_bmo_on_a_holiday_rolls_forward():
    # Good Friday 2024. A release that morning is first tradable the next Monday.
    out = align_events(_event("2024-03-29 07:00", "bmo"), config=EventsConfig())
    assert out["t0"].iloc[0] == pd.Timestamp("2024-04-01")


def test_intraday_default_is_conservative():
    conservative = align_events(_event("2024-05-02 11:00", "during"), config=EventsConfig())
    assert conservative["t0"].iloc[0] == pd.Timestamp("2024-05-03")
    assert not conservative["accepts_intraday_lookahead"].iloc[0]


def test_intraday_same_day_is_flagged_as_lookahead():
    cfg = EventsConfig(intraday_policy="same_day")
    out = align_events(_event("2024-05-02 11:00", "during"), config=cfg)
    assert out["t0"].iloc[0] == pd.Timestamp("2024-05-02")
    assert out["accepts_intraday_lookahead"].iloc[0]


def test_clock_time_recovers_a_missing_timing_flag():
    """A source that gives a timestamp but no BMO/AMC flag is still usable --
    but the recovery is recorded, because it is a source of measurement error."""
    out = align_events(_event("2024-05-02 20:00", "unknown"), config=EventsConfig())
    assert out["timing"].iloc[0] == "amc"
    assert out["timing_source"].iloc[0] == "derived"
    assert out["timing_imputed"].iloc[0]


def test_timeless_announcement_is_assumed_and_flagged():
    out = align_events(_event("2024-05-02 00:00", "unknown"), config=EventsConfig())
    assert out["timing_source"].iloc[0] == "assumed"
    assert out["t0"].iloc[0] == pd.Timestamp("2024-05-03")


def test_timeless_announcements_can_be_dropped():
    cfg = EventsConfig(unknown_time_policy="drop")
    with pytest.raises(EventAlignmentError, match="removed every event"):
        align_events(_event("2024-05-02 00:00", "unknown"), config=cfg)


def test_trade_open_never_precedes_the_announcement(events):
    announced = pd.to_datetime(events["announced_at_utc"], utc=True)
    assert (announced <= events["trade_open_ts"]).all()


def test_t0_is_always_a_session(events):
    from earnings_engine.utils.calendar import default_calendar

    cal = default_calendar()
    assert all(cal.is_session(t) for t in events["t0"].unique())
