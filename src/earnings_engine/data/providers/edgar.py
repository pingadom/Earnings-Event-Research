"""SEC EDGAR provider: filing metadata, XBRL fundamentals, filing text.

EDGAR is the reason the US universe is the right place to do this research on a
student budget. Two properties matter:

1. ``acceptanceDateTime`` in the submissions feed is the instant the document
   became public, to the minute. That is a genuine point-in-time stamp, so
   look-ahead prevention becomes verifiable rather than assumed.
2. ``companyfacts`` returns every XBRL fact ever reported *with the accession
   that first reported it*, so restatements do not silently overwrite what the
   market actually saw at the time. We keep the first-reported value.

Conditions of use: EDGAR requires a descriptive ``User-Agent`` containing a
contact address and rate-limits to 10 requests/second. Both are honoured here.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd

from ...utils.frames import FILINGS, FUNDAMENTALS
from ...utils.logging_utils import get_logger
from ..base import ProviderError
from ..registry import register

log = get_logger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

#: Map us-gaap XBRL tags onto our canonical line items. Order matters: the
#: first tag present for a period wins, because issuers use different tags for
#: the same economic quantity (and change which one they use over time).
TAG_MAP: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps_diluted": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "cfo": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    "total_debt": (
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "DebtLongtermAndShorttermCombinedAmount",
    ),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "total_assets": ("Assets",),
    "shares_diluted": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
}


class _Throttle:
    """Simple token-free rate limiter: at most ``rate`` calls per second."""

    def __init__(self, rate: float = 8.0) -> None:
        self.min_interval = 1.0 / rate
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


@register("edgar")
class EdgarProvider:
    """Filings and XBRL fundamentals from SEC EDGAR."""

    name = "edgar"

    def __init__(
        self,
        user_agent: str | None = None,
        cache_dir: str | Path | None = None,
        rate: float = 8.0,
    ) -> None:
        self.user_agent = user_agent or os.environ.get("SEC_USER_AGENT", "").strip()
        if not self.user_agent:
            raise ProviderError(
                "SEC EDGAR requires a descriptive User-Agent with a contact address. "
                "Set SEC_USER_AGENT in your .env, e.g. "
                '"earnings-event-engine research (you@example.com)"'
            )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._throttle = _Throttle(rate)
        self._session = None
        self._cik_map: dict[str, int] | None = None

    # ---- plumbing ------------------------------------------------------

    def _get(self, url: str, *, as_json: bool = True):
        try:
            import requests  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("requests is required: pip install -e '.[data]'") from exc
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
            )
        cache_key = None
        if self.cache_dir is not None:
            cache_key = self.cache_dir / (re.sub(r"[^A-Za-z0-9]+", "_", url)[-150:] + ".cache")
            if cache_key.exists():
                text = cache_key.read_text(encoding="utf-8")
                return json.loads(text) if as_json else text
        self._throttle.wait()
        resp = self._session.get(url, timeout=30)
        if resp.status_code == 404:
            raise ProviderError(f"EDGAR 404 for {url}")
        resp.raise_for_status()
        text = resp.text
        if cache_key is not None:
            cache_key.write_text(text, encoding="utf-8")
        return resp.json() if as_json else text

    def cik_for(self, ticker: str) -> int:
        if self._cik_map is None:
            data = self._get(TICKER_MAP_URL)
            self._cik_map = {
                str(v["ticker"]).upper(): int(v["cik_str"]) for v in data.values()
            }
        key = ticker.upper().replace(".", "-")
        if key not in self._cik_map:
            raise ProviderError(f"no CIK found for ticker {ticker!r}")
        return self._cik_map[key]

    # ---- provider interface --------------------------------------------

    def get_filings(self, tickers, start, end) -> pd.DataFrame:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        rows = []
        for ticker in tickers:
            try:
                cik = self.cik_for(ticker)
                data = self._get(SUBMISSIONS_URL.format(cik=cik))
            except ProviderError as exc:
                log.warning("edgar: skipping %s (%s)", ticker, exc)
                continue
            recent = data.get("filings", {}).get("recent", {})
            if not recent:
                continue
            n = len(recent.get("accessionNumber", []))
            for i in range(n):
                form = recent["form"][i]
                if form not in {"10-K", "10-Q", "20-F", "40-F"}:
                    continue
                # acceptanceDateTime is the public-availability instant and is
                # stated in Eastern time.
                accepted = recent.get("acceptanceDateTime", [None] * n)[i]
                if accepted:
                    ts = pd.Timestamp(accepted)
                    ts = (
                        ts.tz_localize("America/New_York").tz_convert("UTC")
                        if ts.tzinfo is None
                        else ts.tz_convert("UTC")
                    )
                else:  # fall back to the filing date at the close
                    ts = pd.Timestamp(
                        f"{recent['filingDate'][i]} 17:30", tz="America/New_York"
                    ).tz_convert("UTC")
                if not (start_ts <= ts <= end_ts):
                    continue
                accession = recent["accessionNumber"][i]
                rows.append(
                    {
                        "ticker": ticker,
                        "accession": accession,
                        "form": form,
                        "filed_at_utc": ts,
                        "period_end": pd.Timestamp(recent["reportDate"][i] or pd.NaT),
                        "path": ARCHIVE_URL.format(
                            cik=cik,
                            accession_nodash=accession.replace("-", ""),
                            document=recent["primaryDocument"][i],
                        ),
                    }
                )
        if not rows:
            raise ProviderError("edgar returned no filings for the requested tickers/window")
        return FILINGS.validate(pd.DataFrame(rows))

    def get_fundamentals(self, tickers, start, end) -> pd.DataFrame:
        """First-reported XBRL facts, stamped with when they became public."""
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        rows = []
        for ticker in tickers:
            try:
                cik = self.cik_for(ticker)
                facts = self._get(COMPANYFACTS_URL.format(cik=cik))
            except ProviderError as exc:
                log.warning("edgar: skipping %s (%s)", ticker, exc)
                continue
            gaap = facts.get("facts", {}).get("us-gaap", {})
            for item, tags in TAG_MAP.items():
                collected: dict[pd.Timestamp, dict] = {}
                for tag in tags:
                    node = gaap.get(tag)
                    if not node:
                        continue
                    for unit_facts in node.get("units", {}).values():
                        for f in unit_facts:
                            if f.get("form") not in {"10-K", "10-Q"}:
                                continue
                            end_date = f.get("end")
                            filed = f.get("filed")
                            if not end_date or not filed:
                                continue
                            # Quarterly flows only (fp/frame heuristics vary), plus
                            # instants for balance-sheet items.
                            period_end = pd.Timestamp(end_date)
                            if not (start_ts <= period_end <= end_ts):
                                continue
                            filed_ts = pd.Timestamp(f"{filed} 17:30", tz="America/New_York")
                            prev = collected.get(period_end)
                            # Keep the *earliest* disclosure: that is what the
                            # market saw. Later restatements are ignored.
                            if prev is None or filed_ts < prev["available_from_utc"]:
                                collected[period_end] = {
                                    "value": float(f["val"]),
                                    "available_from_utc": filed_ts,
                                }
                    if collected:
                        break  # first tag that yields data wins
                for period_end, rec in collected.items():
                    rows.append(
                        {
                            "ticker": ticker,
                            "period_end": period_end,
                            "available_from_utc": rec["available_from_utc"].tz_convert("UTC"),
                            "item": item,
                            "value": rec["value"],
                        }
                    )
        if not rows:
            raise ProviderError("edgar returned no XBRL facts for the requested tickers/window")
        return FUNDAMENTALS.validate(pd.DataFrame(rows))

    def get_text(self, accession: str, url: str | None = None) -> str:
        """Fetch a filing document and strip it to plain text."""
        if url is None:
            raise ProviderError(
                "pass the document url (the 'path' column of get_filings) to get_text"
            )
        html = self._get(url, as_json=False)
        return html_to_text(html)


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
_ENTITIES = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&#39;": "'", "&quot;": '"'}


def html_to_text(html: str) -> str:
    """Cheap HTML-to-text good enough for bag-of-words features.

    Deliberately dependency-free. If you need structure (Item 7 extraction,
    tables), install the ``data`` extra and use BeautifulSoup + lxml instead.
    """
    text = _SCRIPT_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    for entity, repl in _ENTITIES.items():
        text = text.replace(entity, repl)
    return _WS_RE.sub(" ", text).strip()
