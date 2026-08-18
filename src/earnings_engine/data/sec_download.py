"""Production SEC submissions and Company Facts acquisition.

The durable fundamental table preserves both the fiscal period described by a
fact and the filing/acceptance time when that fact became public. Later
restatements remain separate rows identified by accession number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .http import HttpClient
from .schemas import FILING_COLUMNS, FUNDAMENTAL_COLUMNS, FUNDAMENTAL_VALUE_COLUMNS

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
OLD_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


class SecConfigurationError(ValueError):
    """SEC identification is absent or clearly still a placeholder."""


@dataclass(frozen=True)
class ConceptSpec:
    kind: str
    units: tuple[str, ...]
    concepts: tuple[tuple[str, str], ...]


CONCEPTS: dict[str, ConceptSpec] = {
    "revenue": ConceptSpec(
        "flow",
        ("USD",),
        (
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "Revenues"),
            ("us-gaap", "SalesRevenueNet"),
        ),
    ),
    "operating_income": ConceptSpec("flow", ("USD",), (("us-gaap", "OperatingIncomeLoss"),)),
    "net_income": ConceptSpec(
        "flow",
        ("USD",),
        (("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")),
    ),
    "eps": ConceptSpec(
        "flow",
        ("USD/shares",),
        (
            ("us-gaap", "EarningsPerShareDiluted"),
            ("us-gaap", "EarningsPerShareBasicAndDiluted"),
            ("us-gaap", "EarningsPerShareBasic"),
        ),
    ),
    "total_assets": ConceptSpec("instant", ("USD",), (("us-gaap", "Assets"),)),
    "total_liabilities": ConceptSpec("instant", ("USD",), (("us-gaap", "Liabilities"),)),
    "cash": ConceptSpec(
        "instant",
        ("USD",),
        (
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
            ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        ),
    ),
    "debt": ConceptSpec(
        "instant",
        ("USD",),
        (
            ("us-gaap", "LongTermDebtAndFinanceLeaseObligations"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
            ("us-gaap", "LongTermDebt"),
            ("us-gaap", "LongTermDebtNoncurrent"),
        ),
    ),
    "shareholders_equity": ConceptSpec(
        "instant",
        ("USD",),
        (
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ),
    ),
    "operating_cash_flow": ConceptSpec(
        "flow",
        ("USD",),
        (("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),),
    ),
    "capital_expenditure": ConceptSpec(
        "flow",
        ("USD",),
        (
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
            ("us-gaap", "PaymentsToAcquireProductiveAssets"),
        ),
    ),
    "shares_outstanding": ConceptSpec(
        "instant",
        ("shares",),
        (
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
        ),
    ),
}

RELEVANT_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A", "20-F", "40-F"}
FACT_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "40-F"}


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _normalise_ticker(ticker: str) -> str:
    return ticker.upper().replace(".", "-")


def _accepted_timestamp(value: Any, filing_date: Any) -> tuple[pd.Timestamp, str]:
    if value is not None and str(value).strip() not in {"", "nan", "None"}:
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("America/New_York")
        return parsed.tz_convert("UTC"), "acceptance_timestamp"
    filed = pd.Timestamp(filing_date)
    fallback = (filed + pd.Timedelta(hours=17, minutes=30)).tz_localize("America/New_York")
    return fallback.tz_convert("UTC"), "filing_date_conservative"


class SecDownloader:
    """Sequential SEC downloader that respects a maximum of eight calls/second."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: str | Path,
        *,
        force: bool = False,
    ) -> None:
        value = (user_agent or "").strip()
        if not value or "your.email@example.com" in value.lower() or "your name" in value.lower():
            raise SecConfigurationError(
                "SEC_USER_AGENT is required and must identify you with a real contact address. "
                "Copy .env.example to .env and set, for example, "
                "SEC_USER_AGENT=Your Name your.email@domain.com"
            )
        self.force = force
        self.client = HttpClient(
            user_agent=value,
            cache_dir=Path(cache_dir) / "http",
            min_interval=0.125,
            timeout=30.0,
            retries=5,
            backoff=1.0,
        )
        self._ticker_map: dict[str, int] | None = None

    def cik_for(self, ticker: str) -> int:
        if self._ticker_map is None:
            payload = self.client.get_json(
                TICKERS_URL, force=self.force, max_age_seconds=24 * 60 * 60
            )
            self._ticker_map = {
                _normalise_ticker(str(record["ticker"])): int(record["cik_str"])
                for record in payload.values()
            }
        key = _normalise_ticker(ticker)
        if key not in self._ticker_map:
            raise KeyError(f"SEC ticker map has no current CIK for {ticker}")
        return self._ticker_map[key]

    def download_ticker(
        self, ticker: str, start: str, end: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cik = self.cik_for(ticker)
        filings = self.download_filings(ticker, cik, start, end)
        facts = self.client.get_json(
            COMPANY_FACTS_URL.format(cik=cik),
            force=self.force,
            max_age_seconds=24 * 60 * 60,
        )
        fundamentals = self.standardise_facts(ticker, cik, facts, filings, start, end)
        return filings, fundamentals

    def download_filings(self, ticker: str, cik: int, start: str, end: str) -> pd.DataFrame:
        payload = self.client.get_json(
            SUBMISSIONS_URL.format(cik=cik),
            force=self.force,
            max_age_seconds=24 * 60 * 60,
        )
        blocks: list[dict[str, Any]] = []
        recent = payload.get("filings", {}).get("recent", {})
        if recent:
            blocks.append(recent)
        for old_file in payload.get("filings", {}).get("files", []):
            name = old_file.get("name")
            if not name:
                continue
            old = self.client.get_json(
                OLD_SUBMISSIONS_URL.format(name=name),
                force=self.force,
                max_age_seconds=30 * 24 * 60 * 60,
            )
            blocks.append(old.get("filings", {}).get("recent", old))

        rows: list[dict[str, Any]] = []
        end_ts = pd.Timestamp(end)
        # Include earlier filings because Company Facts may refer to an older
        # accession needed as provenance for a fact inside the requested period.
        provenance_floor = pd.Timestamp(start) - pd.DateOffset(years=2)
        for block in blocks:
            n = len(block.get("accessionNumber", []))
            for index in range(n):
                form = str(_value(block, "form", index, ""))
                if form not in RELEVANT_FORMS:
                    continue
                filing_date = pd.to_datetime(
                    _value(block, "filingDate", index, None), errors="coerce"
                )
                if pd.isna(filing_date) or not (provenance_floor <= filing_date <= end_ts):
                    continue
                accession = str(_value(block, "accessionNumber", index, ""))
                document = str(_value(block, "primaryDocument", index, ""))
                accepted, quality = _accepted_timestamp(
                    _value(block, "acceptanceDateTime", index, None), filing_date
                )
                rows.append(
                    {
                        "ticker": ticker.upper(),
                        "cik": int(cik),
                        "accession": accession,
                        "form": form,
                        "filing_date": filing_date.normalize(),
                        "accepted_at_utc": accepted,
                        "period_end": pd.to_datetime(
                            _value(block, "reportDate", index, None), errors="coerce"
                        ),
                        "items": str(_value(block, "items", index, "") or ""),
                        "primary_document": document,
                        "filing_url": ARCHIVE_URL.format(
                            cik=cik,
                            accession=accession.replace("-", ""),
                            document=document,
                        ),
                        "timestamp_quality": quality,
                        "source": "sec_submissions",
                    }
                )
        if not rows:
            return _empty(FILING_COLUMNS)
        return (
            pd.DataFrame(rows, columns=FILING_COLUMNS)
            .drop_duplicates("accession", keep="first")
            .sort_values(["filing_date", "accession"])
            .reset_index(drop=True)
        )

    def standardise_facts(
        self,
        ticker: str,
        cik: int,
        payload: dict[str, Any],
        filings: pd.DataFrame,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Map issuer concepts into one row per original filing and fiscal period."""
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        metadata = filings.set_index("accession").to_dict("index") if not filings.empty else {}
        selected: dict[tuple[str, pd.Timestamp, str, Any, Any], dict[str, Any]] = {}

        for standard_name, spec in CONCEPTS.items():
            candidates: dict[tuple[str, pd.Timestamp, str, Any, Any], tuple[tuple, dict]] = {}
            for concept_rank, (namespace, concept) in enumerate(spec.concepts):
                node = payload.get("facts", {}).get(namespace, {}).get(concept)
                if not node:
                    continue
                for unit_rank, unit in enumerate(spec.units):
                    for fact in node.get("units", {}).get(unit, []):
                        record = _candidate_fact(fact, spec, start_ts, end_ts)
                        if record is None:
                            continue
                        key = (
                            record["accession"],
                            record["period_end"],
                            record["form"],
                            record["fiscal_year"],
                            record["fiscal_period"],
                        )
                        score = (concept_rank, unit_rank, record.pop("duration_score"))
                        provenance = {
                            "namespace": namespace,
                            "concept": concept,
                            "unit": unit,
                            "start": record.pop("fact_start"),
                            "end": str(record["period_end"].date()),
                            "accession": record["accession"],
                        }
                        record["provenance"] = provenance
                        if key not in candidates or score < candidates[key][0]:
                            candidates[key] = (score, record)
            for key, (_score, record) in candidates.items():
                row = selected.setdefault(
                    key,
                    {
                        "ticker": ticker.upper(),
                        "cik": int(cik),
                        "accession": record["accession"],
                        "period_end": record["period_end"],
                        "fiscal_year": record["fiscal_year"],
                        "fiscal_period": record["fiscal_period"],
                        "form": record["form"],
                        "_filed": record["filed"],
                        "_provenance": {},
                    },
                )
                row[standard_name] = record["value"]
                row["_provenance"][standard_name] = record["provenance"]

        rows = []
        for row in selected.values():
            accession = row["accession"]
            filing_meta = metadata.get(accession, {})
            filed = pd.Timestamp(filing_meta.get("filing_date", row.pop("_filed"))).normalize()
            if filed > end_ts:
                continue
            if filing_meta:
                available = pd.Timestamp(filing_meta["accepted_at_utc"])
            else:
                available, _quality = _accepted_timestamp(None, filed)
            for column in FUNDAMENTAL_VALUE_COLUMNS:
                row.setdefault(column, np.nan)
            if pd.notna(row["operating_cash_flow"]) and pd.notna(row["capital_expenditure"]):
                row["free_cash_flow"] = row["operating_cash_flow"] - row["capital_expenditure"]
                row["_provenance"]["free_cash_flow"] = {
                    "derived": "operating_cash_flow - capital_expenditure"
                }
            row["filing_date"] = filed
            row["available_from_utc"] = available
            row["provenance_json"] = json.dumps(row.pop("_provenance"), sort_keys=True)
            row["source"] = "sec_companyfacts"
            rows.append(row)
        if not rows:
            return _empty(FUNDAMENTAL_COLUMNS)
        frame = pd.DataFrame(rows)
        frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce").astype("Int64")
        return (
            frame.loc[:, FUNDAMENTAL_COLUMNS]
            .drop_duplicates(["ticker", "accession", "period_end", "fiscal_period"])
            .sort_values(["filing_date", "ticker", "period_end"])
            .reset_index(drop=True)
        )


def _value(block: dict[str, Any], key: str, index: int, default: Any) -> Any:
    values = block.get(key, [])
    return values[index] if index < len(values) else default


def _candidate_fact(
    fact: dict[str, Any], spec: ConceptSpec, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any] | None:
    form = str(fact.get("form", ""))
    if form not in FACT_FORMS or not fact.get("accn") or fact.get("val") is None:
        return None
    period_end = pd.to_datetime(fact.get("end"), errors="coerce")
    filed = pd.to_datetime(fact.get("filed"), errors="coerce")
    if pd.isna(period_end) or pd.isna(filed) or not (start <= period_end <= end) or filed > end:
        return None
    fact_start = pd.to_datetime(fact.get("start"), errors="coerce")
    duration_score = 0
    if spec.kind == "flow":
        if pd.isna(fact_start):
            return None
        days = int((period_end - fact_start).days) + 1
        is_annual = form.startswith(("10-K", "20-F", "40-F")) or fact.get("fp") == "FY"
        target = 365 if is_annual else 91
        if is_annual and not 250 <= days <= 430:
            return None
        if not is_annual and not 60 <= days <= 120:
            return None
        duration_score = abs(days - target)
    try:
        value = float(fact["val"])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    fiscal_year = fact.get("fy")
    try:
        fiscal_year = int(fiscal_year) if fiscal_year is not None else pd.NA
    except (TypeError, ValueError):
        fiscal_year = pd.NA
    return {
        "accession": str(fact["accn"]),
        "period_end": period_end.normalize(),
        "form": form,
        "fiscal_year": fiscal_year,
        "fiscal_period": str(fact.get("fp") or "unknown"),
        "filed": filed.normalize(),
        "value": value,
        "fact_start": str(fact_start.date()) if pd.notna(fact_start) else None,
        "duration_score": duration_score,
    }
