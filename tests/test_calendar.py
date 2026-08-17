"""NYSE calendar rules. These are the assertions that stop an off-by-one
propagating into every abnormal return in the study."""

from __future__ import annotations

import pandas as pd
import pytest

from earnings_engine.utils.calendar import EARLY_CLOSE, REGULAR_CLOSE, TradingCalendar


@pytest.fixture(scope="module")
def cal() -> TradingCalendar:
    return TradingCalendar("1995-01-01", "2030-12-31")


@pytest.mark.parametrize(
    "day",
    [
        "2001-09-11",  # attacks
        "2001-09-14",
        "2012-10-29",  # Sandy
        "2012-10-30",
        "2018-12-05",  # Bush funeral
        "2025-01-09",  # Carter funeral
        "2024-03-29",  # Good Friday
        "2022-06-20",  # Juneteenth observed
        "2024-06-19",  # Juneteenth
        "2024-01-01",
        "2024-12-25",
        "2024-11-28",  # Thanksgiving
    ],
)
def test_market_closed(cal, day):
    assert not cal.is_session(day)


@pytest.mark.parametrize(
    "day",
    [
        "2010-12-31",  # NY Day 2011 fell on a Saturday: NOT observed on the Friday
        "2021-06-18",  # Juneteenth only became a holiday in 2022
        "2024-07-05",
        "2024-11-29",  # half day, but a session
        "1997-01-20",  # MLK Day only became an NYSE holiday in 1998
    ],
)
def test_market_open(cal, day):
    assert cal.is_session(day)


def test_early_closes(cal):
    assert cal.close_time("2024-11-29") == EARLY_CLOSE  # day after Thanksgiving
    assert cal.close_time("2024-12-24") == EARLY_CLOSE
    assert cal.close_time("2024-11-27") == REGULAR_CLOSE


def test_shift_is_in_trading_days(cal):
    # Thanksgiving week: five calendar days forward is only three sessions.
    start = pd.Timestamp("2024-11-26")
    assert cal.shift(start, 3) == pd.Timestamp("2024-12-02")


def test_shift_round_trip(cal):
    day = pd.Timestamp("2019-05-15")
    assert cal.shift(cal.shift(day, 37), -37) == day


def test_window_length(cal):
    win = cal.window("2020-06-15", 0, 19)
    assert len(win) == 20
    assert win[0] == pd.Timestamp("2020-06-15")


def test_next_previous_session_skip_weekends(cal):
    assert cal.next_session("2024-07-06") == pd.Timestamp("2024-07-08")
    assert cal.previous_session("2024-07-06") == pd.Timestamp("2024-07-05")
    assert cal.next_session("2024-07-05", inclusive=True) == pd.Timestamp("2024-07-05")


def test_position_rejects_non_session(cal):
    with pytest.raises(KeyError):
        cal.position("2024-12-25")
