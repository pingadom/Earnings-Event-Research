"""Focused validation regressions for durable acquisition outputs."""

from __future__ import annotations

import pandas as pd

from earnings_engine.data.validation import validate_frames


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "adjusted_close": [101.0, 102.0],
            "adj_close": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
            "source": "yahoo_finance",
        }
    )


def test_duplicate_ticker_date_prices_are_rejected():
    prices = pd.concat([_prices(), _prices().iloc[[0]]], ignore_index=True)
    report = validate_frames({"prices": prices})
    assert any(issue.code == "duplicate_ticker_date" for issue in report.errors)


def test_impossible_ohlc_is_rejected():
    prices = _prices()
    prices.loc[0, "high"] = 98.0
    report = validate_frames({"prices": prices})
    assert any(issue.code == "impossible_ohlc" for issue in report.errors)


def test_filing_cannot_be_available_before_its_filing_date():
    fundamentals = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "accession": ["0001"],
            "filing_date": [pd.Timestamp("2024-05-02")],
            "available_from_utc": [pd.Timestamp("2024-05-01 20:00", tz="UTC")],
            "period_end": [pd.Timestamp("2024-03-31")],
            "fiscal_period": ["Q2"],
        }
    )
    report = validate_frames({"fundamentals": fundamentals})
    assert any(issue.code == "availability_before_filing" for issue in report.errors)


def test_earnings_outside_price_history_is_reported_not_discarded():
    earnings = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "earnings_date": [pd.Timestamp("2023-01-01")],
            "announcement_time": ["unknown"],
            "source": ["yahoo_finance"],
        }
    )
    report = validate_frames({"prices": _prices(), "earnings": earnings})
    assert report.summary["earnings"]["rows"] == 1
    assert any(issue.code == "outside_price_history" for issue in report.warnings)
