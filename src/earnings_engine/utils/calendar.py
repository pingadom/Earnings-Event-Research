"""NYSE trading calendar.

Why this exists rather than a dependency: event studies are indexed in
*trading days*, not calendar days, and the difference is not cosmetic. A
20-trading-day window spans a different amount of wall-clock time depending on
where Thanksgiving and Christmas fall, and getting it wrong shifts every
abnormal return by a day or two. `pandas_market_calendars` does this well but
pulls in a dependency chain we do not otherwise need, so the (well-specified,
rarely-changing) NYSE rules are encoded here and covered by tests.

Sources for the rule set: NYSE published holiday schedules, plus the list of
unscheduled closures since 1990.
"""

from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay

EXCHANGE_TZ = "America/New_York"
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Scheduled NYSE market holidays.

    Note the New Year's Day rule: when 1 January falls on a Saturday the NYSE
    does *not* close the preceding Friday, which is why it uses
    ``sunday_to_monday`` rather than ``nearest_workday``.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        Holiday(
            "Martin Luther King Jr. Day",
            month=1,
            day=1,
            offset=pd.DateOffset(weekday=0, weeks=2),  # 3rd Monday
            start_date=pd.Timestamp("1998-01-01"),
        ),
        Holiday(
            "Washington's Birthday",
            month=2,
            day=1,
            offset=pd.DateOffset(weekday=0, weeks=2),  # 3rd Monday
        ),
        GoodFriday,
        Holiday("Memorial Day", month=5, day=31, offset=pd.DateOffset(weekday=0, weeks=-1)),
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            observance=nearest_workday,
            start_date=pd.Timestamp("2022-06-19"),
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        Holiday("Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=0)),
        Holiday("Thanksgiving Day", month=11, day=1, offset=pd.DateOffset(weekday=3, weeks=3)),
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


#: Unscheduled full-day closures since 1990.
SPECIAL_CLOSURES: tuple[str, ...] = (
    "1994-04-27",  # Nixon funeral
    "2001-09-11",  # September 11 attacks
    "2001-09-12",
    "2001-09-13",
    "2001-09-14",
    "2004-06-11",  # Reagan funeral
    "2007-01-02",  # Ford funeral
    "2012-10-29",  # Hurricane Sandy
    "2012-10-30",
    "2018-12-05",  # G.H.W. Bush funeral
    "2025-01-09",  # Carter funeral
)

#: Sessions with a 13:00 ET close. Recurring rules cover the common cases;
#: this list holds the dates those rules would miss or get wrong.
SPECIAL_EARLY_CLOSES: tuple[str, ...] = (
    "2001-09-17",
    "2001-09-18",
    "2001-09-19",
    "2001-09-20",
    "2001-09-21",
)


class TradingCalendar:
    """Trading sessions and session-relative arithmetic for a single exchange.

    The calendar is materialised once over ``[start, end]`` and then answers
    everything by integer position, which keeps the hot paths (shifting an
    event by N sessions, slicing an estimation window) O(1).
    """

    def __init__(self, start: str | date = "1990-01-01", end: str | date = "2035-12-31") -> None:
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self._sessions = self._build_sessions()
        self._pos = pd.Series(np.arange(len(self._sessions)), index=self._sessions)
        self._early = self._build_early_closes()

    # ---- construction -------------------------------------------------

    def _build_sessions(self) -> pd.DatetimeIndex:
        holidays = NYSEHolidayCalendar().holidays(self.start, self.end)
        specials = pd.DatetimeIndex([pd.Timestamp(d) for d in SPECIAL_CLOSURES])
        specials = specials[(specials >= self.start) & (specials <= self.end)]
        all_holidays = holidays.union(specials)
        bday = CustomBusinessDay(holidays=all_holidays)
        sessions = pd.date_range(self.start, self.end, freq=bday)
        return pd.DatetimeIndex(sessions).normalize()

    def _build_early_closes(self) -> set[pd.Timestamp]:
        early: set[pd.Timestamp] = set()
        years = range(self.start.year, self.end.year + 1)
        for year in years:
            # Day after Thanksgiving.
            thanksgiving = pd.Timestamp(year=year, month=11, day=1) + pd.DateOffset(
                weekday=3, weeks=3
            )
            early.add(pd.Timestamp(thanksgiving) + pd.Timedelta(days=1))
            # 3 July when Independence Day is observed on the 4th and the 3rd
            # is itself a session.
            early.add(pd.Timestamp(year=year, month=7, day=3))
            # Christmas Eve when it is a session.
            early.add(pd.Timestamp(year=year, month=12, day=24))
        early.update(pd.Timestamp(d) for d in SPECIAL_EARLY_CLOSES)
        valid = set(self._sessions)
        return {d.normalize() for d in early if d.normalize() in valid}

    # ---- queries ------------------------------------------------------

    @property
    def sessions(self) -> pd.DatetimeIndex:
        """All trading sessions in the calendar range, normalised to midnight."""
        return self._sessions

    def is_session(self, day: pd.Timestamp | str | date) -> bool:
        return pd.Timestamp(day).normalize() in self._pos.index

    def close_time(self, day: pd.Timestamp | str | date) -> time:
        return EARLY_CLOSE if pd.Timestamp(day).normalize() in self._early else REGULAR_CLOSE

    def sessions_between(self, start, end) -> pd.DatetimeIndex:
        s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        return self._sessions[(self._sessions >= s) & (self._sessions <= e)]

    def position(self, day) -> int:
        """Integer index of ``day`` among sessions. Raises if not a session."""
        d = pd.Timestamp(day).normalize()
        try:
            return int(self._pos.loc[d])
        except KeyError as exc:
            raise KeyError(f"{d.date()} is not an NYSE trading session") from exc

    def next_session(self, day, inclusive: bool = False) -> pd.Timestamp:
        d = pd.Timestamp(day).normalize()
        idx = self._sessions.searchsorted(d, side="left" if inclusive else "right")
        if idx >= len(self._sessions):
            raise IndexError(f"no session on or after {d.date()} within calendar range")
        return self._sessions[idx]

    def previous_session(self, day, inclusive: bool = False) -> pd.Timestamp:
        d = pd.Timestamp(day).normalize()
        idx = self._sessions.searchsorted(d, side="right" if inclusive else "left") - 1
        if idx < 0:
            raise IndexError(f"no session on or before {d.date()} within calendar range")
        return self._sessions[idx]

    def shift(self, day, n: int) -> pd.Timestamp:
        """Return the session ``n`` trading days after ``day`` (negative = before)."""
        idx = self.position(day) + n
        if not 0 <= idx < len(self._sessions):
            raise IndexError(f"shifting {pd.Timestamp(day).date()} by {n} leaves the calendar")
        return self._sessions[idx]

    def window(self, day, start_offset: int, end_offset: int) -> pd.DatetimeIndex:
        """Sessions in ``[day + start_offset, day + end_offset]`` inclusive."""
        if end_offset < start_offset:
            raise ValueError("end_offset must be >= start_offset")
        base = self.position(day)
        lo, hi = base + start_offset, base + end_offset
        lo_c, hi_c = max(lo, 0), min(hi, len(self._sessions) - 1)
        if hi_c < lo_c:
            return pd.DatetimeIndex([])
        return self._sessions[lo_c : hi_c + 1]

    def positions_of(self, days) -> np.ndarray:
        """Vectorised :meth:`position` returning ``-1`` for non-sessions."""
        idx = pd.DatetimeIndex(pd.to_datetime(days)).normalize()
        out = self._pos.reindex(idx).to_numpy(dtype="float64")
        return np.where(np.isnan(out), -1, out).astype("int64")


_DEFAULT: TradingCalendar | None = None


def default_calendar() -> TradingCalendar:
    """Process-wide singleton; building the session index is not free."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TradingCalendar()
    return _DEFAULT
