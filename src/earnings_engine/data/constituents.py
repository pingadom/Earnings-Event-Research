"""Historical S&P 500 interval acquisition."""

from __future__ import annotations

from io import StringIO

import pandas as pd

from .http import HttpClient
from .schemas import MEMBERSHIP_COLUMNS

INTERVAL_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
CURRENT_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv"
SOURCE_LABEL = "github:fja05680/sp500"


def download_sp500_membership(client: HttpClient, *, force: bool = False) -> pd.DataFrame:
    """Download interval membership and add current-name/sector metadata.

    The upstream interval file starts in 1996. Sector metadata is only joined
    for current members and must not be interpreted as a historical GICS
    classification. Deleted names remain in the interval table with a null
    sector rather than being discarded.
    """
    max_age = 7 * 24 * 60 * 60
    intervals = pd.read_csv(
        StringIO(client.get_text(INTERVAL_URL, force=force, max_age_seconds=max_age))
    )
    current = pd.read_csv(
        StringIO(client.get_text(CURRENT_URL, force=force, max_age_seconds=max_age))
    )
    intervals.columns = [str(c).strip().lower() for c in intervals.columns]
    current = current.rename(
        columns={
            "Symbol": "ticker",
            "Security": "security_name",
            "GICS Sector": "sector",
        }
    )
    required = {"ticker", "start_date", "end_date"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"historical constituent response missing columns: {sorted(missing)}")

    intervals["ticker"] = intervals["ticker"].astype("string").str.strip().str.upper()
    intervals["start_date"] = pd.to_datetime(intervals["start_date"], errors="coerce")
    intervals["end_date"] = pd.to_datetime(intervals["end_date"], errors="coerce")
    current["ticker"] = current["ticker"].astype("string").str.strip().str.upper()
    meta = current[["ticker", "security_name", "sector"]].drop_duplicates("ticker")
    out = intervals.merge(meta, on="ticker", how="left", validate="many_to_one")
    out["index"] = "S&P 500"
    out["source"] = SOURCE_LABEL
    out = out.dropna(subset=["ticker", "start_date"])
    out = out.drop_duplicates(subset=["ticker", "start_date", "end_date"])
    return (
        out.loc[:, MEMBERSHIP_COLUMNS].sort_values(["start_date", "ticker"]).reset_index(drop=True)
    )


def tickers_for_window(membership: pd.DataFrame, start: str, end: str) -> list[str]:
    """All tickers whose membership interval overlaps the requested window."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    ends = pd.to_datetime(membership["end_date"], errors="coerce").fillna(
        pd.Timestamp("2100-01-01")
    )
    starts = pd.to_datetime(membership["start_date"], errors="coerce")
    mask = (starts <= end_ts) & (ends >= start_ts)
    return sorted(membership.loc[mask, "ticker"].dropna().astype(str).unique().tolist())
