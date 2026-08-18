"""Cross-dataset validation for durable acquisition outputs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    dataset: str
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, level: str, dataset: str, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(level, dataset, code, message))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "errors": [asdict(issue) for issue in self.errors],
            "warnings": [asdict(issue) for issue in self.warnings],
        }

    def render(self) -> str:
        lines = ["=== DATA ACQUISITION SUMMARY ===", ""]
        labels = {
            "prices": "Prices",
            "earnings": "Earnings",
            "fundamentals": "Fundamentals",
            "sec_filings": "SEC filings",
            "fama_french_daily": "Fama-French",
            "benchmarks": "Benchmarks",
            "index_membership": "Index membership",
        }
        for key, label in labels.items():
            if key not in self.summary:
                continue
            info = self.summary[key]
            detail = f"{info.get('rows', 0):,} rows"
            if "tickers" in info:
                detail = f"{info['tickers']:,} tickers, {detail}"
            if info.get("start") and info.get("end"):
                detail += f", {info['start']} -> {info['end']}"
            lines.append(f"{label}: {detail}")
        lines.extend(["", f"Errors: {len(self.errors)}", f"Warnings: {len(self.warnings)}"])
        for issue in self.issues:
            lines.append(f"  [{issue.level.upper()}] {issue.dataset}/{issue.code}: {issue.message}")
        return "\n".join(lines)


def _summary(frame: pd.DataFrame, date_col: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {"rows": int(len(frame))}
    if "ticker" in frame:
        info["tickers"] = int(frame["ticker"].nunique(dropna=True))
    if date_col and date_col in frame and not frame.empty:
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        if dates.notna().any():
            info["start"] = str(dates.min().date())
            info["end"] = str(dates.max().date())
    return info


def validate_frames(frames: dict[str, pd.DataFrame]) -> ValidationReport:
    """Validate every available output and the joins between them."""
    report = ValidationReport()
    for name, frame in frames.items():
        date_col = {
            "prices": "date",
            "earnings": "earnings_date",
            "fundamentals": "period_end",
            "sec_filings": "filing_date",
            "fama_french_daily": "date",
            "benchmarks": "date",
            "index_membership": "start_date",
        }.get(name)
        report.summary[name] = _summary(frame, date_col)
        if frame.empty:
            report.add("error", name, "empty", "dataset has no rows")

    prices = frames.get("prices")
    if prices is not None and not prices.empty:
        _validate_prices(prices, report, "prices")
    benchmarks = frames.get("benchmarks")
    if benchmarks is not None and not benchmarks.empty:
        _validate_prices(benchmarks, report, "benchmarks")
        _validate_benchmark_coverage(prices, benchmarks, report)
    earnings = frames.get("earnings")
    if earnings is not None and not earnings.empty:
        _validate_earnings(earnings, prices, report)
    fundamentals = frames.get("fundamentals")
    if fundamentals is not None and not fundamentals.empty:
        _validate_fundamentals(fundamentals, report)
    membership = frames.get("index_membership")
    if membership is not None and not membership.empty:
        _validate_membership(membership, report)
    factors = frames.get("fama_french_daily")
    if factors is not None and not factors.empty:
        _validate_factors(factors, report)
    return report


def validate_directory(raw_dir: str | Path) -> tuple[dict[str, pd.DataFrame], ValidationReport]:
    root = Path(raw_dir)
    names = (
        "prices",
        "earnings",
        "fundamentals",
        "sec_filings",
        "fama_french_daily",
        "benchmarks",
        "index_membership",
    )
    frames = {
        name: pd.read_parquet(root / f"{name}.parquet")
        for name in names
        if (root / f"{name}.parquet").exists()
    }
    report = validate_frames(frames)
    if not frames:
        report.add(
            "error",
            "acquisition",
            "no_datasets",
            f"no recognized Parquet datasets exist under {root}",
        )
    return frames, report


def _validate_prices(frame: pd.DataFrame, report: ValidationReport, dataset: str) -> None:
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    adjusted = "adjusted_close" if "adjusted_close" in frame else "adj_close"
    missing = required - set(frame.columns)
    if adjusted not in frame:
        missing.add("adjusted_close")
    if missing:
        report.add("error", dataset, "missing_columns", f"missing {sorted(missing)}")
        return
    dup = frame.duplicated(["ticker", "date"], keep=False)
    if dup.any():
        report.add(
            "error",
            dataset,
            "duplicate_ticker_date",
            f"{int(dup.sum())} rows share a ticker/date key",
        )
    dates = pd.to_datetime(frame["date"], errors="coerce")
    malformed = int(dates.isna().sum())
    if malformed:
        report.add("error", dataset, "malformed_dates", f"{malformed} dates are invalid")
    tickers = frame["ticker"].astype("string")
    bad_tickers = tickers.isna() | ~tickers.str.match(r"^[A-Z0-9^][A-Z0-9.\-^=]*$", na=False)
    if bad_tickers.any():
        report.add("error", dataset, "ticker_format", f"{int(bad_tickers.sum())} malformed tickers")
    numeric = ["open", "high", "low", "close", adjusted]
    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    negative = (values < 0).any(axis=1)
    if negative.any():
        report.add("error", dataset, "negative_prices", f"{int(negative.sum())} rows")
    impossible = (values["high"] < values[["open", "low", "close"]].max(axis=1)) | (
        values["low"] > values[["open", "high", "close"]].min(axis=1)
    )
    if impossible.any():
        report.add("error", dataset, "impossible_ohlc", f"{int(impossible.sum())} rows")
    if (pd.to_numeric(frame["volume"], errors="coerce") < 0).any():
        report.add("error", dataset, "negative_volume", "negative volume observations found")
    sorted_frame = frame.assign(_date=dates).sort_values(["ticker", "_date"])
    returns = sorted_frame.groupby("ticker", sort=False)[adjusted].pct_change(fill_method=None)
    extreme = returns.abs() > 5.0
    if extreme.any():
        report.add(
            "warning",
            dataset,
            "absurd_returns",
            f"{int(extreme.sum())} adjusted-close returns exceed 500%; inspect corporate actions",
        )
    rows_per_ticker = frame.groupby("ticker").size()
    low = int((rows_per_ticker < 100).sum())
    if low:
        report.add("warning", dataset, "low_history", f"{low} symbols have fewer than 100 rows")
    if "source" in frame:
        fallback = frame.loc[frame["source"].astype(str).str.contains("stooq"), "ticker"].nunique()
        if fallback:
            report.add(
                "warning",
                dataset,
                "fallback_source",
                f"{fallback} symbols use explicitly labelled Stooq fallback data",
            )


def _validate_earnings(
    earnings: pd.DataFrame, prices: pd.DataFrame | None, report: ValidationReport
) -> None:
    required = {"ticker", "earnings_date", "announcement_time", "source"}
    missing = required - set(earnings.columns)
    if missing:
        report.add("error", "earnings", "missing_columns", f"missing {sorted(missing)}")
        return
    dates = pd.to_datetime(earnings["earnings_date"], errors="coerce")
    if dates.isna().any():
        report.add(
            "error", "earnings", "malformed_dates", f"{int(dates.isna().sum())} invalid dates"
        )
    valid_times = {"before_market", "after_market", "during_market", "unknown"}
    invalid = ~earnings["announcement_time"].isin(valid_times)
    if invalid.any():
        report.add("error", "earnings", "announcement_time", f"{int(invalid.sum())} invalid labels")
    if prices is not None and not prices.empty:
        coverage = prices.groupby("ticker")["date"].agg(["min", "max"])
        merged = earnings.assign(_event_date=dates).merge(
            coverage, left_on="ticker", right_index=True, how="left"
        )
        outside = (
            merged["min"].isna()
            | (merged["_event_date"] < merged["min"])
            | (merged["_event_date"] > merged["max"])
        )
        if outside.any():
            report.add(
                "warning",
                "earnings",
                "outside_price_history",
                f"{int(outside.sum())} events lack price coverage on their event window",
            )


def _validate_fundamentals(frame: pd.DataFrame, report: ValidationReport) -> None:
    required = {"ticker", "accession", "filing_date", "period_end", "available_from_utc"}
    missing = required - set(frame.columns)
    if missing:
        report.add("error", "fundamentals", "missing_columns", f"missing {sorted(missing)}")
        return
    filed = pd.to_datetime(frame["filing_date"], errors="coerce")
    period = pd.to_datetime(frame["period_end"], errors="coerce")
    available = pd.to_datetime(frame["available_from_utc"], errors="coerce", utc=True)
    malformed = filed.isna() | period.isna() | available.isna()
    if malformed.any():
        report.add("error", "fundamentals", "malformed_dates", f"{int(malformed.sum())} rows")
    early = filed < period
    if early.any():
        report.add(
            "error",
            "fundamentals",
            "filing_before_period_end",
            f"{int(early.sum())} filings predate their financial period end",
        )
    dup = frame.duplicated(["ticker", "accession", "period_end", "fiscal_period"], keep=False)
    if dup.any():
        report.add(
            "error", "fundamentals", "duplicate_sec_facts", f"{int(dup.sum())} duplicate rows"
        )
    # UTC acceptance instants must never precede the reported period's end.
    before_period = available.dt.tz_localize(None) < period
    if before_period.any():
        report.add(
            "error",
            "fundamentals",
            "availability_before_period",
            f"{int(before_period.sum())} availability timestamps predate period end",
        )
    before_filing = available.dt.tz_localize(None).dt.normalize() < filed
    if before_filing.any():
        report.add(
            "error",
            "fundamentals",
            "availability_before_filing",
            f"{int(before_filing.sum())} observations become available before filing date",
        )


def _validate_membership(frame: pd.DataFrame, report: ValidationReport) -> None:
    required = {"ticker", "start_date", "end_date", "index"}
    missing = required - set(frame.columns)
    if missing:
        report.add("error", "index_membership", "missing_columns", f"missing {sorted(missing)}")
        return
    starts = pd.to_datetime(frame["start_date"], errors="coerce")
    ends = pd.to_datetime(frame["end_date"], errors="coerce")
    invalid = starts.isna() | (ends.notna() & (ends < starts))
    if invalid.any():
        report.add("error", "index_membership", "invalid_intervals", f"{int(invalid.sum())} rows")
    if frame["ticker"].nunique() < 400:
        report.add(
            "warning",
            "index_membership",
            "suspicious_row_count",
            "fewer than 400 unique symbols in the historical membership file",
        )


def _validate_factors(frame: pd.DataFrame, report: ValidationReport) -> None:
    required = {"date", "Mkt-RF", "SMB", "HML", "RF"}
    missing = required - set(frame.columns)
    if missing:
        report.add("error", "fama_french_daily", "missing_columns", f"missing {sorted(missing)}")
    if len(frame) < 100:
        report.add("warning", "fama_french_daily", "suspicious_row_count", "fewer than 100 days")
    numeric = frame[[column for column in required - {"date"} if column in frame]]
    if not numeric.empty and (numeric.abs() > 0.5).any().any():
        report.add(
            "warning",
            "fama_french_daily",
            "factor_units",
            "a daily factor exceeds 50%; verify percent-to-decimal conversion",
        )


def _validate_benchmark_coverage(
    prices: pd.DataFrame | None, benchmarks: pd.DataFrame, report: ValidationReport
) -> None:
    if "SPY" not in set(benchmarks["ticker"].astype(str)):
        report.add("error", "benchmarks", "missing_spy", "SPY market proxy is absent")
    if prices is None or prices.empty:
        return
    benchmark_dates = set(
        pd.to_datetime(benchmarks.loc[benchmarks["ticker"].eq("SPY"), "date"]).dt.normalize()
    )
    price_dates = set(pd.to_datetime(prices["date"]).dt.normalize())
    missing = price_dates - benchmark_dates
    if missing:
        ratio = len(missing) / max(len(price_dates), 1)
        level = "error" if ratio > 0.05 else "warning"
        report.add(
            level,
            "benchmarks",
            "coverage_gaps",
            f"SPY misses {len(missing)} of {len(price_dates)} equity trading dates ({ratio:.1%})",
        )


def ticker_format_is_consistent(tickers: pd.Series) -> bool:
    """Small public helper used by focused unit tests."""
    pattern = re.compile(r"^[A-Z0-9^][A-Z0-9.\-^=]*$")
    return bool(
        tickers.dropna().astype(str).map(lambda value: bool(pattern.fullmatch(value))).all()
    )
