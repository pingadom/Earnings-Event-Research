"""Optional whole-symbol fallback for daily prices from Stooq."""

from __future__ import annotations

from io import StringIO

import pandas as pd

from ...utils.frames import PRICES
from ...utils.logging_utils import get_logger
from ..base import ProviderError
from ..http import HttpClient
from ..registry import register

log = get_logger(__name__)


def stooq_symbol(ticker: str) -> str:
    """Map a US equity/index ticker to Stooq's download convention."""
    symbol = ticker.strip().lower().replace(".", "-")
    if symbol == "^gspc":
        return "^spx"
    return symbol if symbol.startswith("^") else f"{symbol}.us"


@register("stooq")
class StooqProvider:
    """Daily Stooq data; used only when explicitly enabled as a fallback."""

    name = "stooq"

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient(user_agent="earnings-event-engine/0.1")

    def get_prices(self, tickers, start, end) -> pd.DataFrame:
        frames = []
        d1 = pd.Timestamp(start).strftime("%Y%m%d")
        d2 = pd.Timestamp(end).strftime("%Y%m%d")
        for ticker in tickers:
            symbol = stooq_symbol(ticker)
            url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
            try:
                raw = pd.read_csv(StringIO(self.client.get_text(url)))
            except Exception as exc:
                log.warning("stooq: %s failed (%s)", ticker, exc)
                continue
            if raw.empty or "Date" not in raw:
                continue
            raw = raw.rename(columns=str.lower)
            close = pd.to_numeric(raw.get("close"), errors="coerce")
            frame = pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": pd.to_datetime(raw["date"], errors="coerce"),
                    "open": pd.to_numeric(raw.get("open"), errors="coerce"),
                    "high": pd.to_numeric(raw.get("high"), errors="coerce"),
                    "low": pd.to_numeric(raw.get("low"), errors="coerce"),
                    "close": close,
                    # Stooq does not expose a separate total-return adjustment
                    # field in this endpoint. This equality is explicit in the
                    # output source/provenance and documented as a limitation.
                    "adj_close": close,
                    "volume": pd.to_numeric(raw.get("volume"), errors="coerce"),
                }
            ).dropna(subset=["date", "adj_close"])
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise ProviderError("Stooq returned no prices for the requested symbols")
        return PRICES.validate(pd.concat(frames, ignore_index=True))
