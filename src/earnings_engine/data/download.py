"""Top-level, resumable acquisition orchestration."""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .constituents import download_sp500_membership, tickers_for_window
from .earnings import download_yahoo_earnings, reconcile_earnings, sec_earnings_from_filings
from .fama_french import download_fama_french_daily
from .http import HttpClient
from .providers.stooq import StooqProvider
from .providers.yahoo import YahooProvider
from .schemas import EARNINGS_COLUMNS, FILING_COLUMNS, FUNDAMENTAL_COLUMNS, PRICE_COLUMNS
from .sec_download import SecConfigurationError, SecDownloader
from .storage import SymbolCache, write_json_atomic, write_parquet_atomic
from .validation import ValidationReport, validate_directory

log = logging.getLogger(__name__)

BENCHMARKS = {"SPY": "S&P 500 ETF total-return proxy", "^GSPC": "S&P 500 price index"}


@dataclass(frozen=True)
class AcquisitionConfig:
    raw_dir: Path
    cache_dir: Path
    start: str
    end: str
    tickers: tuple[str, ...] = ()
    datasets: frozenset[str] = frozenset(
        {"membership", "prices", "benchmarks", "sec", "earnings", "factors"}
    )
    force: bool = False
    stooq_fallback: bool = True
    sec_user_agent: str = ""


@dataclass
class AcquisitionResult:
    report: ValidationReport
    failures: list[dict[str, str]] = field(default_factory=list)
    outputs: dict[str, Path] = field(default_factory=dict)


def _progress(values: Iterable[str], description: str) -> Iterable[str]:
    try:
        from tqdm import tqdm

        return tqdm(list(values), desc=description, unit="ticker")
    except ImportError:  # pragma: no cover - dependency is in requirements.txt
        return values


def _normalise_tickers(tickers: Iterable[str]) -> list[str]:
    return sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})


def _output_price_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy()
    out["ticker"] = out["ticker"].astype("string").str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["adjusted_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
    out["source"] = source
    return out.loc[:, PRICE_COLUMNS].dropna(subset=["ticker", "date", "adjusted_close"])


def _existing(path: Path, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            log.warning("could not reuse %s: %s", path, exc)
    return pd.DataFrame(columns=columns)


def _replace_window(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    successful_tickers: set[str],
    *,
    date_column: str,
    start: str,
    end: str,
    keys: list[str],
) -> pd.DataFrame:
    if existing.empty:
        combined = new.copy()
    elif not successful_tickers:
        combined = existing.copy()
    else:
        dates = pd.to_datetime(existing[date_column], errors="coerce")
        replace = existing["ticker"].astype(str).isin(successful_tickers) & dates.between(
            pd.Timestamp(start), pd.Timestamp(end)
        )
        combined = pd.concat([existing.loc[~replace], new], ignore_index=True)
    if combined.empty:
        return combined
    return combined.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def _download_price_symbols(
    symbols: list[str], config: AcquisitionConfig, namespace: str
) -> tuple[pd.DataFrame, set[str], list[dict[str, str]]]:
    cache = SymbolCache(config.cache_dir, namespace)
    yahoo = YahooProvider(batch_size=1, auto_adjust=False)
    http = HttpClient(
        user_agent="earnings-event-engine/0.1 local data acquisition",
        cache_dir=config.cache_dir / "stooq_http",
        timeout=30,
        retries=4,
        backoff=1.0,
    )
    stooq = StooqProvider(http)
    frames: list[pd.DataFrame] = []
    successes: set[str] = set()
    failures: list[dict[str, str]] = []
    for ticker in _progress(symbols, f"{namespace}: prices"):
        cached = cache.load(ticker)
        if (
            not config.force
            and cache.covers(ticker, config.start, config.end)
            and cached is not None
        ):
            frames.append(cached)
            successes.add(ticker)
            continue
        result = None
        error = None
        for attempt in range(3):
            try:
                canonical = yahoo.get_prices([ticker], config.start, config.end)
                result = _output_price_frame(canonical, "yahoo_finance")
                if result.empty:
                    raise RuntimeError("empty Yahoo response")
                break
            except Exception as exc:  # pragma: no cover - network dependent
                error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        if result is None and config.stooq_fallback:
            try:
                canonical = stooq.get_prices([ticker], config.start, config.end)
                result = _output_price_frame(canonical, "stooq")
                log.warning("%s: using Stooq because Yahoo failed (%s)", ticker, error)
            except Exception as exc:  # pragma: no cover - network dependent
                error = RuntimeError(f"Yahoo: {error}; Stooq: {exc}")
        if result is None or result.empty:
            if cached is not None and not cached.empty:
                frames.append(cached)
                successes.add(ticker)
                failures.append(
                    {
                        "dataset": namespace,
                        "ticker": ticker,
                        "error": f"refresh failed; retained stale cache: {error}",
                    }
                )
            else:
                failures.append({"dataset": namespace, "ticker": ticker, "error": str(error)})
            continue
        if cached is not None and not cached.empty:
            result = pd.concat([cached, result], ignore_index=True).drop_duplicates(
                ["ticker", "date"], keep="last"
            )
        source = "+".join(sorted(result["source"].dropna().astype(str).unique()))
        coverage_start, coverage_end = cache.union_window(ticker, config.start, config.end)
        cache.store(
            ticker,
            result,
            start=coverage_start,
            end=coverage_end,
            source=source,
        )
        frames.append(result)
        successes.add(ticker)
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS), successes, failures
    out = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(out["date"], errors="coerce")
    return (
        out.loc[dates.between(pd.Timestamp(config.start), pd.Timestamp(config.end))]
        .drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True),
        successes,
        failures,
    )


def _download_sec(
    tickers: list[str], config: AcquisitionConfig
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], list[dict[str, str]]]:
    downloader = SecDownloader(config.sec_user_agent, config.cache_dir / "sec", force=config.force)
    filings_cache = SymbolCache(config.cache_dir, "sec_filings")
    facts_cache = SymbolCache(config.cache_dir, "sec_fundamentals")
    filings_frames: list[pd.DataFrame] = []
    fact_frames: list[pd.DataFrame] = []
    successes: set[str] = set()
    failures: list[dict[str, str]] = []
    for ticker in _progress(tickers, "SEC"):
        cached_filings = filings_cache.load(ticker)
        cached_facts = facts_cache.load(ticker)
        covered = filings_cache.covers(ticker, config.start, config.end) and facts_cache.covers(
            ticker, config.start, config.end
        )
        if not config.force and covered and cached_filings is not None and cached_facts is not None:
            filings_frames.append(cached_filings)
            fact_frames.append(cached_facts)
            successes.add(ticker)
            continue
        try:
            filings, facts = downloader.download_ticker(ticker, config.start, config.end)
            if cached_filings is not None and not cached_filings.empty:
                filings = pd.concat([cached_filings, filings], ignore_index=True).drop_duplicates(
                    "accession", keep="last"
                )
            if cached_facts is not None and not cached_facts.empty:
                facts = pd.concat([cached_facts, facts], ignore_index=True).drop_duplicates(
                    ["ticker", "accession", "period_end", "fiscal_period"], keep="last"
                )
            coverage_start, coverage_end = filings_cache.union_window(
                ticker, config.start, config.end
            )
            filings_cache.store(
                ticker,
                filings,
                start=coverage_start,
                end=coverage_end,
                source="sec_submissions",
            )
            facts_cache.store(
                ticker,
                facts,
                start=coverage_start,
                end=coverage_end,
                source="sec_companyfacts",
            )
            filings_frames.append(filings)
            fact_frames.append(facts)
            successes.add(ticker)
        except Exception as exc:  # one issuer never aborts the whole job
            if cached_filings is not None and cached_facts is not None:
                filings_frames.append(cached_filings)
                fact_frames.append(cached_facts)
                successes.add(ticker)
                message = f"refresh failed; retained stale cache: {exc}"
            else:
                message = str(exc)
            failures.append({"dataset": "sec", "ticker": ticker, "error": message})
    filings = (
        pd.concat(filings_frames, ignore_index=True)
        if filings_frames
        else pd.DataFrame(columns=FILING_COLUMNS)
    )
    facts = (
        pd.concat(fact_frames, ignore_index=True)
        if fact_frames
        else pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
    )
    return filings, facts, successes, failures


def _download_earnings(
    tickers: list[str], config: AcquisitionConfig, sec_filings: pd.DataFrame
) -> tuple[pd.DataFrame, set[str], list[dict[str, str]]]:
    cache = SymbolCache(config.cache_dir, "earnings")
    frames: list[pd.DataFrame] = []
    successes: set[str] = set()
    failures: list[dict[str, str]] = []
    for ticker in _progress(tickers, "earnings"):
        try:
            frame = download_yahoo_earnings(
                ticker,
                config.start,
                config.end,
                cache,
                force=config.force,
            )
            frames.append(frame)
            successes.add(ticker)
        except Exception as exc:
            cached = cache.load(ticker)
            if cached is not None:
                frames.append(cached)
                successes.add(ticker)
                error = f"refresh failed; retained stale cache: {exc}"
            else:
                error = str(exc)
            failures.append({"dataset": "earnings", "ticker": ticker, "error": error})
    yahoo = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EARNINGS_COLUMNS)
    )
    sec = sec_earnings_from_filings(sec_filings)
    if not sec.empty:
        dates = pd.to_datetime(sec["earnings_date"], errors="coerce")
        sec = sec.loc[
            sec["ticker"].isin(tickers)
            & dates.between(pd.Timestamp(config.start), pd.Timestamp(config.end))
        ]
    return reconcile_earnings(yahoo, sec), successes, failures


def run_acquisition(config: AcquisitionConfig, *, validate_only: bool = False) -> AcquisitionResult:
    """Run selected acquisitions, persist after each stage, and validate all local outputs."""
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    outputs: dict[str, Path] = {}
    if validate_only:
        _frames, report = validate_directory(config.raw_dir)
        return AcquisitionResult(report=report)

    if "sec" in config.datasets and not config.sec_user_agent.strip():
        raise SecConfigurationError(
            "SEC_USER_AGENT is not set. Copy .env.example to .env and set your real name and "
            "contact email before running the full or --sec-only acquisition."
        )

    public_client = HttpClient(
        user_agent="earnings-event-engine/0.1 local data acquisition",
        cache_dir=config.cache_dir / "public_http",
        timeout=45.0,
        retries=4,
        backoff=1.0,
    )
    membership_path = config.raw_dir / "index_membership.parquet"
    if "membership" in config.datasets or not config.tickers:
        membership = download_sp500_membership(public_client, force=config.force)
        outputs["index_membership"] = write_parquet_atomic(membership, membership_path)
    else:
        membership = _existing(membership_path)

    if config.tickers:
        tickers = _normalise_tickers(config.tickers)
    else:
        if membership.empty:
            raise RuntimeError("no tickers supplied and historical membership is unavailable")
        tickers = tickers_for_window(membership, config.start, config.end)
    log.info("acquisition universe: %d symbols", len(tickers))

    if "prices" in config.datasets:
        new, successful, stage_failures = _download_price_symbols(tickers, config, "prices")
        failures.extend(stage_failures)
        path = config.raw_dir / "prices.parquet"
        merged = _replace_window(
            _existing(path, PRICE_COLUMNS),
            new,
            successful,
            date_column="date",
            start=config.start,
            end=config.end,
            keys=["ticker", "date"],
        )
        outputs["prices"] = write_parquet_atomic(merged, path)

    if "benchmarks" in config.datasets:
        symbols = list(BENCHMARKS)
        new, successful, stage_failures = _download_price_symbols(symbols, config, "benchmarks")
        failures.extend(stage_failures)
        if not new.empty:
            new["benchmark_name"] = new["ticker"].map(BENCHMARKS)
        path = config.raw_dir / "benchmarks.parquet"
        merged = _replace_window(
            _existing(path, PRICE_COLUMNS),
            new,
            successful,
            date_column="date",
            start=config.start,
            end=config.end,
            keys=["ticker", "date"],
        )
        outputs["benchmarks"] = write_parquet_atomic(merged, path)

    filings_path = config.raw_dir / "sec_filings.parquet"
    facts_path = config.raw_dir / "fundamentals.parquet"
    if "sec" in config.datasets:
        filings, facts, successful, stage_failures = _download_sec(tickers, config)
        failures.extend(stage_failures)
        old_filings = _existing(filings_path, FILING_COLUMNS)
        old_facts = _existing(facts_path, FUNDAMENTAL_COLUMNS)
        merged_filings = _replace_window(
            old_filings,
            filings,
            successful,
            date_column="filing_date",
            start=config.start,
            end=config.end,
            keys=["ticker", "accession"],
        )
        merged_facts = _replace_window(
            old_facts,
            facts,
            successful,
            date_column="period_end",
            start=config.start,
            end=config.end,
            keys=["ticker", "accession", "period_end", "fiscal_period"],
        )
        outputs["sec_filings"] = write_parquet_atomic(merged_filings, filings_path)
        outputs["fundamentals"] = write_parquet_atomic(merged_facts, facts_path)
        available_filings = merged_filings
    else:
        available_filings = _existing(filings_path, FILING_COLUMNS)

    if "earnings" in config.datasets:
        earnings, successful, stage_failures = _download_earnings(
            tickers, config, available_filings
        )
        failures.extend(stage_failures)
        path = config.raw_dir / "earnings.parquet"
        merged = _replace_window(
            _existing(path, EARNINGS_COLUMNS),
            earnings,
            successful,
            date_column="earnings_date",
            start=config.start,
            end=config.end,
            keys=["ticker", "earnings_date", "accession"],
        )
        outputs["earnings"] = write_parquet_atomic(merged, path)

    if "factors" in config.datasets:
        factors = download_fama_french_daily(
            public_client, config.start, config.end, force=config.force
        )
        path = config.raw_dir / "fama_french_daily.parquet"
        existing = _existing(path)
        if not existing.empty:
            dates = pd.to_datetime(existing["date"], errors="coerce")
            existing = existing.loc[
                ~dates.between(pd.Timestamp(config.start), pd.Timestamp(config.end))
            ]
            factors = pd.concat([existing, factors], ignore_index=True)
        factors = factors.drop_duplicates("date", keep="last").sort_values("date")
        outputs["fama_french_daily"] = write_parquet_atomic(factors, path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": config.start,
        "end": config.end,
        "tickers_requested": len(tickers),
        "datasets": sorted(config.datasets),
        "force_refresh": config.force,
        "stooq_fallback": config.stooq_fallback,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "failures": failures,
    }
    write_json_atomic(manifest, config.raw_dir / "acquisition_manifest.json")
    _frames, report = validate_directory(config.raw_dir)
    for dataset, count in sorted(Counter(item["dataset"] for item in failures).items()):
        report.add(
            "warning",
            dataset,
            "download_failures",
            f"{count} symbol download(s) failed or needed stale cache; "
            "see acquisition_manifest.json",
        )
    write_json_atomic(report.to_dict(), config.raw_dir / "validation_report.json")
    return AcquisitionResult(report=report, failures=failures, outputs=outputs)


def config_from_environment(
    *,
    raw_dir: str | Path,
    cache_dir: str | Path,
    start: str,
    end: str,
    tickers: Iterable[str],
    datasets: Iterable[str],
    force: bool,
    stooq_fallback: bool,
) -> AcquisitionConfig:
    return AcquisitionConfig(
        raw_dir=Path(raw_dir),
        cache_dir=Path(cache_dir),
        start=str(pd.Timestamp(start).date()),
        end=str(pd.Timestamp(end).date()),
        tickers=tuple(_normalise_tickers(tickers)),
        datasets=frozenset(datasets),
        force=force,
        stooq_fallback=stooq_fallback,
        sec_user_agent=os.environ.get("SEC_USER_AGENT", "").strip(),
    )
