"""Hermetic source-adapter tests; no test in this file uses the network."""

from __future__ import annotations

import sys
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from earnings_engine.data.constituents import download_sp500_membership, tickers_for_window
from earnings_engine.data.earnings import download_yahoo_earnings
from earnings_engine.data.fama_french import FF3_URL, FF5_URL, MOM_URL, download_fama_french_daily
from earnings_engine.data.providers.local import LocalProvider, _reject_annual_flows
from earnings_engine.data.schemas import FILING_COLUMNS
from earnings_engine.data.sec_download import SecConfigurationError, SecDownloader
from earnings_engine.data.storage import SymbolCache


def _zip_csv(text: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("factors.csv", text)
    return buffer.getvalue()


def test_fama_french_percent_values_become_decimal_returns():
    payloads = {
        FF3_URL: _zip_csv("notes\n,Mkt-RF,SMB,HML,RF\n20240102,1.00,0.20,-0.30,0.01\nfooter\n"),
        FF5_URL: _zip_csv("notes\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20240102,1,2,3,0.40,-0.50,0.01\n"),
        MOM_URL: _zip_csv("notes\n,Mom\n20240102,0.60\n"),
    }

    class Client:
        def get_bytes(self, url, **_kwargs):
            return payloads[url]

    frame = download_fama_french_daily(Client(), "2024-01-01", "2024-01-31")
    assert frame.loc[0, "Mkt-RF"] == pytest.approx(0.01)
    assert frame.loc[0, "RMW"] == pytest.approx(0.004)
    assert frame.loc[0, "Mom"] == pytest.approx(0.006)


def test_constituent_intervals_include_deleted_names():
    responses = {
        "ticker_start": "ticker,start_date,end_date\nOLD,2010-01-01,2018-01-01\nAAPL,2010-01-01,\n",
        "sp500.csv": ("Symbol,Security,GICS Sector\nAAPL,Apple,Information Technology\n"),
    }

    class Client:
        def get_text(self, url, **_kwargs):
            return responses["sp500.csv" if url.endswith("sp500.csv") else "ticker_start"]

    frame = download_sp500_membership(Client())
    assert set(tickers_for_window(frame, "2015-01-01", "2015-12-31")) == {"AAPL", "OLD"}
    assert pd.isna(frame.loc[frame["ticker"].eq("OLD"), "sector"]).all()


def test_empty_yahoo_response_is_cached_and_does_not_crash(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")

    class EmptyTicker:
        def get_earnings_dates(self, limit=100):
            assert limit == 100
            return pd.DataFrame()

    monkeypatch.setitem(
        sys.modules, "yfinance", SimpleNamespace(Ticker=lambda _ticker: EmptyTicker())
    )
    cache = SymbolCache(tmp_path, "earnings")
    frame = download_yahoo_earnings("AAPL", "2020-01-01", "2024-01-01", cache)
    assert frame.empty
    assert cache.covers("AAPL", "2020-01-01", "2024-01-01")


def test_symbol_cache_expands_coverage_instead_of_discarding_work(tmp_path):
    pytest.importorskip("pyarrow")
    cache = SymbolCache(tmp_path, "prices")
    frame = pd.DataFrame({"ticker": ["AAPL"], "date": [pd.Timestamp("2020-01-02")]})
    cache.store("AAPL", frame, start="2020-01-01", end="2021-01-01", source="yahoo_finance")
    start, end = cache.union_window("AAPL", "2019-01-01", "2020-06-01")
    assert (start, end) == ("2019-01-01", "2021-01-01")


def test_sec_concept_mapping_retains_accession_and_acceptance_time():
    filing = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "cik": 320193,
                "accession": "0000320193-24-000001",
                "form": "10-Q",
                "filing_date": pd.Timestamp("2024-05-02"),
                "accepted_at_utc": pd.Timestamp("2024-05-02 20:05", tz="UTC"),
                "period_end": pd.Timestamp("2024-03-30"),
                "items": "",
                "primary_document": "aapl.htm",
                "filing_url": "https://www.sec.gov/example",
                "timestamp_quality": "acceptance_timestamp",
                "source": "sec_submissions",
            }
        ],
        columns=FILING_COLUMNS,
    )
    fact = {
        "start": "2024-01-01",
        "end": "2024-03-30",
        "val": 100.0,
        "accn": "0000320193-24-000001",
        "fy": 2024,
        "fp": "Q2",
        "form": "10-Q",
        "filed": "2024-05-02",
    }
    payload = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [fact]}}
            }
        }
    }
    downloader = SecDownloader.__new__(SecDownloader)
    frame = downloader.standardise_facts(
        "AAPL", 320193, payload, filing, "2024-01-01", "2024-12-31"
    )
    assert frame.loc[0, "revenue"] == 100.0
    assert frame.loc[0, "available_from_utc"] == pd.Timestamp("2024-05-02 20:05", tz="UTC")
    assert "RevenueFromContract" in frame.loc[0, "provenance_json"]


def test_annual_flows_never_reach_the_quarterly_panel():
    """Quarters are derived once, upstream, where the filing timestamps live.

    This adapter used to derive Q4 itself by subtracting three standalone
    quarters from the annual figure. Once the acquisition layer began
    differencing year-to-date facts, that ran a second time on values it had
    already isolated -- and subtracting three quarters from one quarter still
    produces a plausible number, so nothing complained. The adapter's job is now
    only to refuse anything still annual.
    """
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 5 + ["MSFT"] * 3,
            "period_end": pd.to_datetime(
                [
                    "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2024-12-31",
                    "2024-03-31", "2024-06-30", "2024-12-31",
                ]
            ),
            # The Q4 rows carry the label the acquisition layer now assigns.
            "fiscal_period": ["Q1", "Q2", "Q3", "Q4", "FY", "Q1", "Q2", "FY"],
            "raw_item": ["revenue"] * 4 + ["eps"] + ["revenue"] * 3,
            "value": [10.0, 20.0, 30.0, 40.0, 5.0, 10.0, 20.0, 100.0],
        }
    )
    result = _reject_annual_flows(frame)

    kept = result.loc[result["ticker"].eq("AAPL") & result["fiscal_period"].eq("Q4"), "value"]
    assert kept.iloc[0] == 40.0, "an already-derived quarter must pass through untouched"
    assert not (result["fiscal_period"].astype(str).str.upper().eq("FY")).any()
    assert not result["raw_item"].eq("eps").any(), "annual EPS is not additive"
    # Every quarterly observation survives.
    assert len(result) == 6


def test_rejecting_annual_flows_is_not_quadratic():
    """The routine this replaced rebuilt two string columns per row.

    At real scale that came to two hundred million string operations and made
    the adapter the slowest step in the entire study.
    """
    import time

    n = 40_000
    frame = pd.DataFrame(
        {
            "ticker": [f"T{i % 400}" for i in range(n)],
            "period_end": pd.to_datetime("2024-03-31"),
            "fiscal_period": ["Q1", "Q2", "Q3", "FY"] * (n // 4),
            "raw_item": ["revenue"] * n,
            "value": 1.0,
        }
    )
    started = time.perf_counter()
    result = _reject_annual_flows(frame)
    assert time.perf_counter() - started < 2.0
    assert len(result) == n * 3 // 4


def test_sec_user_agent_is_required(tmp_path):
    with pytest.raises(SecConfigurationError, match="SEC_USER_AGENT"):
        SecDownloader("", tmp_path)


def test_local_cached_prices_load_without_network(tmp_path):
    pytest.importorskip("pyarrow")
    columns = {
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "adjusted_close": [101.0, 102.0],
        "adj_close": [101.0, 102.0],
        "volume": [1e6, 1.1e6],
        "source": ["cache", "cache"],
    }
    pd.DataFrame({"ticker": ["AAPL", "AAPL"], **columns}).to_parquet(
        tmp_path / "prices.parquet", index=False
    )
    pd.DataFrame({"ticker": ["SPY", "SPY"], **columns}).to_parquet(
        tmp_path / "benchmarks.parquet", index=False
    )
    provider = LocalProvider(tmp_path)
    frame = provider.get_prices(["AAPL", "SPY"], "2024-01-01", "2024-01-31")
    assert set(frame["ticker"]) == {"AAPL", "SPY"}
    assert len(frame) == 4
