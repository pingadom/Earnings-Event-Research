"""Historical earnings-date acquisition and SEC 8-K timestamp reconciliation."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from .schemas import EARNINGS_COLUMNS
from .storage import SymbolCache

log = logging.getLogger(__name__)


def _empty_earnings() -> pd.DataFrame:
    return pd.DataFrame(columns=EARNINGS_COLUMNS)


def _as_number(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else np.nan


def _classify_local_time(timestamp: pd.Timestamp) -> tuple[str, str]:
    minute = timestamp.hour * 60 + timestamp.minute
    if minute == 0:
        return "unknown", "unknown"
    if minute < 9 * 60 + 30:
        return "before_market", "bmo"
    if minute >= 16 * 60:
        return "after_market", "amc"
    return "during_market", "during"


def download_yahoo_earnings(
    ticker: str,
    start: str,
    end: str,
    cache: SymbolCache,
    *,
    force: bool = False,
    retries: int = 3,
) -> pd.DataFrame:
    """Download Yahoo's available earnings history for one symbol.

    Yahoo date-only rows retain a null ``announced_at_utc`` and an explicit
    ``timestamp_quality=date_only``. A conservative timestamp is introduced
    only by the offline analysis adapter, where it can be audited as a policy.
    """
    cached = cache.load(ticker)
    if not force and cache.covers(ticker, start, end):
        return cached if cached is not None else _empty_earnings()

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("yfinance is required; install requirements.txt") from exc

    dates = None
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            dates = yf.Ticker(ticker.replace(".", "-")).get_earnings_dates(limit=100)
            break
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    if dates is None or dates.empty:
        if last_error:
            raise RuntimeError(f"Yahoo earnings failed for {ticker}: {last_error}")
        frame = cached if cached is not None else _empty_earnings()
        coverage_start, coverage_end = cache.union_window(ticker, start, end)
        cache.store(
            ticker,
            frame,
            start=coverage_start,
            end=coverage_end,
            source="yahoo_finance",
        )
        return frame

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows: list[dict[str, Any]] = []
    for index_value, values in dates.iterrows():
        timestamp = pd.Timestamp(index_value)
        if timestamp.tzinfo is None:
            local = timestamp.tz_localize("America/New_York")
        else:
            local = timestamp.tz_convert("America/New_York")
        earnings_date = local.tz_localize(None).normalize()
        if not (start_ts <= earnings_date <= end_ts):
            continue
        announcement_time, timing = _classify_local_time(local)
        exact = announcement_time != "unknown"
        rows.append(
            {
                "ticker": ticker.upper(),
                "fiscal_period": pd.NaT,
                "earnings_date": earnings_date,
                "announced_at_utc": local.tz_convert("UTC") if exact else pd.NaT,
                "announcement_time": announcement_time,
                "timing": timing,
                "eps_actual": _as_number(values.get("Reported EPS")),
                "eps_estimate": _as_number(values.get("EPS Estimate")),
                "revenue_actual": np.nan,
                "revenue_estimate": np.nan,
                "source": "yahoo_finance",
                "timestamp_quality": "provider_timestamp" if exact else "date_only",
                "accession": pd.NA,
            }
        )
    frame = pd.DataFrame(rows, columns=EARNINGS_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values("earnings_date").drop_duplicates(
            ["ticker", "earnings_date"], keep="last"
        )
    if cached is not None and not cached.empty:
        frame = pd.concat([cached, frame], ignore_index=True).drop_duplicates(
            ["ticker", "earnings_date"], keep="last"
        )
    coverage_start, coverage_end = cache.union_window(ticker, start, end)
    cache.store(
        ticker,
        frame,
        start=coverage_start,
        end=coverage_end,
        source="yahoo_finance",
    )
    return frame.reset_index(drop=True)


def sec_earnings_from_filings(filings: pd.DataFrame) -> pd.DataFrame:
    """Build release events from accepted 8-K filings that declare Item 2.02."""
    if filings.empty:
        return _empty_earnings()
    items = filings.get("items", pd.Series("", index=filings.index)).fillna("").astype(str)
    subset = filings.loc[
        filings["form"].astype(str).str.startswith("8-K")
        & items.str.contains(r"(?:^|\D)2\.02(?:\D|$)", regex=True)
    ].copy()
    if subset.empty:
        return _empty_earnings()
    accepted = pd.to_datetime(subset["accepted_at_utc"], utc=True, errors="coerce")
    local = accepted.dt.tz_convert("America/New_York")
    announcement = []
    timing = []
    for timestamp, quality in zip(local, subset["timestamp_quality"], strict=False):
        if pd.isna(timestamp) or quality != "acceptance_timestamp":
            announcement.append("unknown")
            timing.append("unknown")
        else:
            label, short = _classify_local_time(timestamp)
            announcement.append(label)
            timing.append(short)
    return pd.DataFrame(
        {
            "ticker": subset["ticker"].astype("string").to_numpy(),
            "fiscal_period": pd.to_datetime(subset["period_end"], errors="coerce").to_numpy(),
            "earnings_date": local.dt.tz_localize(None).dt.normalize().to_numpy(),
            "announced_at_utc": accepted.to_numpy(),
            "announcement_time": announcement,
            "timing": timing,
            "eps_actual": np.nan,
            "eps_estimate": np.nan,
            "revenue_actual": np.nan,
            "revenue_estimate": np.nan,
            "source": "sec_8k_item_2.02",
            "timestamp_quality": subset["timestamp_quality"].to_numpy(),
            "accession": subset["accession"].to_numpy(),
        },
        columns=EARNINGS_COLUMNS,
    )


def reconcile_earnings(yahoo: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """Prefer SEC timestamps and enrich them with nearby Yahoo actual/estimate fields."""
    if sec.empty:
        return yahoo.copy()
    if yahoo.empty:
        return sec.copy()
    yahoo = yahoo.copy().reset_index(drop=True)
    sec = sec.copy().reset_index(drop=True)
    used: set[int] = set()
    output: list[dict[str, Any]] = []
    value_columns = ("eps_actual", "eps_estimate", "revenue_actual", "revenue_estimate")
    for sec_row in sec.itertuples(index=False):
        candidates = yahoo.index[yahoo["ticker"].eq(sec_row.ticker)]
        if len(candidates):
            distance = (
                pd.to_datetime(yahoo.loc[candidates, "earnings_date"])
                - pd.Timestamp(sec_row.earnings_date)
            ).abs()
            nearest = int(distance.idxmin())
            if distance.loc[nearest] <= pd.Timedelta(days=2):
                used.add(nearest)
                merged = sec_row._asdict()
                for column in value_columns:
                    merged[column] = yahoo.loc[nearest, column]
                merged["source"] = "sec_8k_item_2.02+yahoo_finance"
                if pd.isna(merged["fiscal_period"]):
                    merged["fiscal_period"] = yahoo.loc[nearest, "fiscal_period"]
                output.append(merged)
                continue
        output.append(sec_row._asdict())
    output.extend(yahoo.loc[~yahoo.index.isin(used)].to_dict("records"))
    frame = pd.DataFrame(output, columns=EARNINGS_COLUMNS)
    return frame.sort_values(["ticker", "earnings_date", "source"]).reset_index(drop=True)
