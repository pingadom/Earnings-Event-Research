"""Point-in-time and legal-trading tests for real-data adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_engine.config import EventsConfig
from earnings_engine.data.providers.local import conservative_unknown_timestamp
from earnings_engine.data.universe import Universe
from earnings_engine.events.alignment import align_events
from earnings_engine.events.pit import PointInTimeError, assert_point_in_time
from earnings_engine.returns.abnormal import ReturnPanel


def test_unknown_earnings_time_cannot_trade_until_next_session():
    stamp = conservative_unknown_timestamp("2024-01-05")  # Friday
    events = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "event_id": ["AAPL-2024-01-05"],
            "announced_at_utc": [stamp],
            "timing": ["unknown"],
            "period_end": [pd.Timestamp("2023-12-31")],
            "fiscal_quarter": ["2023Q4"],
        }
    )
    aligned = align_events(events, config=EventsConfig(min_history_days=0))
    assert aligned.loc[0, "t0"] == pd.Timestamp("2024-01-08")
    assert aligned.loc[0, "trade_open_ts"] > stamp
    assert bool(aligned.loc[0, "timing_imputed"])


def test_sec_fact_after_trade_is_blocked():
    panel = pd.DataFrame(
        {
            "event_id": ["AAPL-event"],
            "ticker": ["AAPL"],
            "available_from_utc": [pd.Timestamp("2024-05-02 20:00", tz="UTC")],
            "trade_open_ts": [pd.Timestamp("2024-05-02 13:30", tz="UTC")],
        }
    )
    with pytest.raises(PointInTimeError, match="published after"):
        assert_point_in_time(panel)


def test_returns_do_not_bridge_missing_price_dates():
    prices = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "SPY", "SPY", "SPY"],
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-04", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "adj_close": [100.0, 110.0, 400.0, 404.0, 408.0],
        }
    )
    panel = ReturnPanel.from_prices(prices, "SPY", winsorize=None)
    position = panel.ticker_pos["AAPL"]
    # 2024-01-04 has no immediately preceding AAPL close. pct_change with
    # fill_method=None must leave it missing rather than manufacture a 10% return.
    assert np.isnan(panel.returns[2, position])


def test_historical_membership_filter_uses_event_date():
    universe = Universe.from_frame(
        pd.DataFrame(
            {
                "ticker": ["OLD", "NEW"],
                "start_date": ["2010-01-01", "2020-01-01"],
                "end_date": ["2019-12-31", "2100-01-01"],
                "sector": ["Tech", "Tech"],
            }
        )
    )
    events = pd.DataFrame(
        {
            "event_id": ["old-valid", "old-invalid", "new-valid"],
            "ticker": ["OLD", "OLD", "NEW"],
            "t0": pd.to_datetime(["2018-01-01", "2021-01-01", "2021-01-01"]),
        }
    )
    kept = universe.filter_events(events)
    assert set(kept["event_id"]) == {"old-valid", "new-valid"}
