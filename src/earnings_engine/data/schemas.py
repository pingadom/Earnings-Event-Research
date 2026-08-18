"""Column contracts for the durable files written under ``data/raw``."""

from __future__ import annotations

PRICE_COLUMNS = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "adj_close",
    "volume",
    "source",
)

EARNINGS_COLUMNS = (
    "ticker",
    "fiscal_period",
    "earnings_date",
    "announced_at_utc",
    "announcement_time",
    "timing",
    "eps_actual",
    "eps_estimate",
    "revenue_actual",
    "revenue_estimate",
    "source",
    "timestamp_quality",
    "accession",
)

FUNDAMENTAL_VALUE_COLUMNS = (
    "revenue",
    "operating_income",
    "net_income",
    "eps",
    "total_assets",
    "total_liabilities",
    "cash",
    "debt",
    "shareholders_equity",
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
    "shares_outstanding",
)

FUNDAMENTAL_COLUMNS = (
    "ticker",
    "cik",
    "accession",
    "filing_date",
    "available_from_utc",
    "period_end",
    "fiscal_year",
    "fiscal_period",
    "form",
    *FUNDAMENTAL_VALUE_COLUMNS,
    "provenance_json",
    "source",
)

FILING_COLUMNS = (
    "ticker",
    "cik",
    "accession",
    "form",
    "filing_date",
    "accepted_at_utc",
    "period_end",
    "items",
    "primary_document",
    "filing_url",
    "timestamp_quality",
    "source",
)

MEMBERSHIP_COLUMNS = (
    "ticker",
    "start_date",
    "end_date",
    "index",
    "sector",
    "security_name",
    "source",
)
