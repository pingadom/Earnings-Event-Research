"""Yahoo Finance provider (via ``yfinance``).

Free, no key, good enough to get a pipeline running end to end. Its limits are
real and are documented rather than hidden:

* it is an undocumented, unofficial endpoint that changes without notice;
* index membership is *current*, so building a universe from it is
  survivorship-biased (see ``docs/biases.md``);
* earnings *times of day* are frequently missing or wrong, and a mislabelled
  BMO/AMC flag shifts the whole event window by one session.

Use it for development. Use Capital IQ / LSEG for anything you would report.
"""

from __future__ import annotations

import pandas as pd

from ...utils.frames import EVENTS, PRICES
from ...utils.logging_utils import get_logger
from ..base import ProviderError
from ..registry import register

log = get_logger(__name__)


def _require_yfinance():
    try:
        import yfinance  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProviderError(
            "yfinance is not installed. Install the data extra: pip install -e '.[data]'"
        ) from exc
    return yfinance


@register("yahoo")
class YahooProvider:
    """Daily adjusted prices and earnings dates from Yahoo Finance."""

    name = "yahoo"

    def __init__(self, batch_size: int = 50, auto_adjust: bool = False) -> None:
        self.batch_size = batch_size
        self.auto_adjust = auto_adjust

    def get_prices(self, tickers, start, end) -> pd.DataFrame:
        yf = _require_yfinance()
        frames = []
        for i in range(0, len(tickers), self.batch_size):
            batch = list(tickers[i : i + self.batch_size])
            log.info("yahoo: prices for %d ticker(s) [%d/%d]", len(batch), i + 1, len(tickers))
            raw = yf.download(
                batch,
                start=str(pd.Timestamp(start).date()),
                end=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
                auto_adjust=self.auto_adjust,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if raw is None or len(raw) == 0:
                continue
            frames.append(self._tidy(raw, batch))
        if not frames:
            raise ProviderError("yahoo returned no price data for the requested tickers")
        return PRICES.validate(pd.concat(frames, ignore_index=True))

    def _tidy(self, raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        out = []
        multi = isinstance(raw.columns, pd.MultiIndex)
        for ticker in tickers:
            if multi:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                sub = raw[ticker].copy()
            else:
                sub = raw.copy()
            sub = sub.dropna(how="all")
            if sub.empty:
                continue
            adj = sub["Adj Close"] if "Adj Close" in sub.columns else sub["Close"]
            out.append(
                pd.DataFrame(
                    {
                        "ticker": ticker,
                        "date": pd.DatetimeIndex(sub.index).tz_localize(None).normalize(),
                        "open": sub.get("Open"),
                        "high": sub.get("High"),
                        "low": sub.get("Low"),
                        "close": sub.get("Close"),
                        "adj_close": adj,
                        "volume": sub.get("Volume"),
                    }
                ).reset_index(drop=True)
            )
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def get_events(self, tickers, start, end) -> pd.DataFrame:
        """Earnings dates. Yahoo's time-of-day is unreliable -- see module docstring."""
        yf = _require_yfinance()
        rows = []
        for ticker in tickers:
            try:
                dates = yf.Ticker(ticker).get_earnings_dates(limit=80)
            except Exception as exc:  # pragma: no cover - network dependent
                log.warning("yahoo: no earnings dates for %s (%s)", ticker, exc)
                continue
            if dates is None or dates.empty:
                continue
            for ts in pd.DatetimeIndex(dates.index):
                ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
                if not (pd.Timestamp(start, tz="UTC") <= ts_utc <= pd.Timestamp(end, tz="UTC")):
                    continue
                local = ts_utc.tz_convert("America/New_York")
                if local.hour == 0 and local.minute == 0:
                    timing = "unknown"
                elif local.hour < 9 or (local.hour == 9 and local.minute < 30):
                    timing = "bmo"
                elif local.hour >= 16:
                    timing = "amc"
                else:
                    timing = "during"
                approx_period = local.normalize().tz_localize(None) - pd.Timedelta(days=30)
                period_end = approx_period.to_period("Q").end_time.normalize()
                rows.append(
                    {
                        "ticker": ticker,
                        "event_id": f"{ticker}-{local.date()}",
                        "announced_at_utc": ts_utc,
                        "timing": timing,
                        "period_end": period_end,
                        "fiscal_quarter": f"{period_end.year}Q{period_end.quarter}",
                    }
                )
        if not rows:
            raise ProviderError("yahoo returned no earnings dates for the requested tickers")
        df = pd.DataFrame(rows).drop_duplicates(subset=["event_id"])
        return EVENTS.validate(df.reset_index(drop=True))
