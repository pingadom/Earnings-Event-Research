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


def test_acceptance_before_filing_date_is_reported_but_not_an_error():
    """Raw data stays faithful to EDGAR; the adapter takes the conservative side.

    EDGAR itself returns acceptance timestamps earlier than the filing date for
    a meaningful minority of filings, and which of the two governs public
    availability is not documented unambiguously. Rewriting the source here
    would hide that, so this layer reports it and
    LocalProvider._conservative_availability resolves it.
    """
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
    assert not report.errors
    assert any(i.code == "acceptance_precedes_filing_date" for i in report.issues)


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


def test_daytime_acceptance_before_the_filing_date_is_flagged():
    """A 16:00 ET acceptance dated to the next day has no filing-time
    explanation, so it is called out separately from the after-hours case."""
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
    assert any(i.code == "acceptance_precedes_filing_date" for i in report.issues)


def test_csv_fallback_preserves_datetime_dtypes(tmp_path):
    """The gzipped-CSV fallback must be dtype-equivalent to parquet.

    Regression: without this, dates came back as strings and a downstream
    concat of "2015-01-02" with "2015-01-02 00:00:00" aborted the whole
    acquisition run with an unparseable-format error.
    """
    from earnings_engine.data.storage import read_table, write_parquet_atomic

    frame = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "date": [pd.Timestamp("2015-01-02"), pd.Timestamp("2016-03-04")],
            "period_end": [pd.Timestamp("2014-12-31"), pd.Timestamp("2016-03-31")],
            "filed_at_utc": [
                pd.Timestamp("2019-10-30 22:12", tz="UTC"),
                pd.Timestamp("2020-01-28 23:02", tz="UTC"),
            ],
            "close": [1.0, 2.0],
        }
    )
    back = read_table(write_parquet_atomic(frame, tmp_path / "x.parquet"))
    assert str(back["date"].dtype).startswith("datetime64")
    assert str(back["period_end"].dtype).startswith("datetime64")
    assert back["filed_at_utc"].dt.tz is not None
    pd.testing.assert_series_equal(back["close"], frame["close"])


def test_csv_fallback_handles_mixed_date_precision(tmp_path):
    """A CSV can hold a bare date and a full timestamp in one column."""
    from earnings_engine.data.storage import read_table

    path = tmp_path / "mixed.csv.gz"
    pd.DataFrame({"date": ["2015-01-02", "2015-01-05 00:00:00"]}).to_csv(
        path, index=False, compression="gzip"
    )
    back = read_table(path)
    assert str(back["date"].dtype).startswith("datetime64")
    assert back["date"].notna().all()


def test_adapter_never_claims_availability_before_the_filing_date():
    """The invariant that actually protects the study.

    Whatever EDGAR reports, the research adapter must never treat a document as
    public earlier than its stated filing date. Being a few hours conservative
    costs nothing -- an after-hours filing resolves to the following midnight
    Eastern, and the event aligner already pushes such announcements to the next
    session's 09:30 open.
    """
    from earnings_engine.data.providers.local import LocalProvider

    accepted = pd.Series(
        [
            pd.Timestamp("2018-04-25 19:38", tz="UTC"),  # 15:38 ET, dated next day
            pd.Timestamp("2019-10-30 22:12", tz="UTC"),  # 18:12 ET, after hours
            pd.Timestamp("2024-05-02 14:00", tz="UTC"),  # same day, normal
        ]
    )
    filed = pd.Series(
        [pd.Timestamp("2018-04-26"), pd.Timestamp("2019-10-31"), pd.Timestamp("2024-05-02")]
    )
    out = LocalProvider._conservative_availability(accepted, filed)
    floor = (
        filed.dt.normalize().dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    assert (out >= floor).all()
    # An unaffected same-day filing keeps its real acceptance time.
    assert out.iloc[2] == accepted.iloc[2]
    # And the floor lands before the opening bell, so nothing is lost.
    assert out.iloc[1].tz_convert("America/New_York").hour < 9
