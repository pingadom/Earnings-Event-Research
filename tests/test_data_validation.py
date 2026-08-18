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


def test_after_hours_acceptance_is_not_an_error():
    """EDGAR dates a filing accepted after 17:30 ET to the next business day.

    The acceptance timestamp then legitimately precedes the filing date, and
    treating that as look-ahead flags roughly a sixth of all real SEC rows.
    Apple's FY2019 10-K is the concrete case: accepted 18:12 ET on 30 October
    2019, filingDate 31 October.
    """
    fundamentals = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "accession": ["0000320193-19-000119"],
            "filing_date": [pd.Timestamp("2019-10-31")],
            "available_from_utc": [pd.Timestamp("2019-10-30 22:12:36", tz="UTC")],
            "period_end": [pd.Timestamp("2019-09-28")],
            "fiscal_period": ["FY"],
        }
    )
    report = validate_frames({"fundamentals": fundamentals})
    assert not any(i.code == "availability_before_filing" for i in report.errors)
    assert any(i.code == "after_hours_acceptance" for i in report.issues)


def test_weekend_filing_shift_is_not_an_error():
    """A Friday evening acceptance carries the following Monday as its filing
    date, so the gap is three days rather than one."""
    fundamentals = pd.DataFrame(
        {
            "ticker": ["XYZ"],
            "accession": ["0002"],
            "filing_date": [pd.Timestamp("2024-05-06")],  # Monday
            "available_from_utc": [pd.Timestamp("2024-05-03 22:30", tz="UTC")],  # Fri 18:30 ET
            "period_end": [pd.Timestamp("2024-03-31")],
            "fiscal_period": ["Q1"],
        }
    )
    report = validate_frames({"fundamentals": fundamentals})
    assert not any(i.code == "availability_before_filing" for i in report.errors)


def test_daytime_acceptance_before_the_filing_date_is_still_an_error():
    """No convention explains a 16:00 ET acceptance dated to the next day."""
    fundamentals = pd.DataFrame(
        {
            "ticker": ["XYZ"],
            "accession": ["0003"],
            "filing_date": [pd.Timestamp("2024-05-02")],
            "available_from_utc": [pd.Timestamp("2024-05-01 20:00", tz="UTC")],  # 16:00 EDT
            "period_end": [pd.Timestamp("2024-03-31")],
            "fiscal_period": ["Q1"],
        }
    )
    report = validate_frames({"fundamentals": fundamentals})
    assert any(i.code == "availability_before_filing" for i in report.errors)
