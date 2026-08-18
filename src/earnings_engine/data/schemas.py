"""Column contracts for the durable files written under ``data/raw``.

This is the **storage** layer: what a downloaded Parquet file on disk must
contain, including provenance columns (``source``, ``accession``,
``timestamp_quality``) that exist so a row can be traced back to the request
that produced it.

Not to be confused with :mod:`earnings_engine.utils.frames`, which is the
**pipeline** layer: what a frame must look like as it passes between research
stages. The two differ deliberately -- storage keeps everything a vendor gave
us, the pipeline keeps only what the research needs, in canonical units. The
adapters in :mod:`earnings_engine.data.providers.local` map one onto the other.
"""

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
    "gross_profit",
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
    "shares_diluted",
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
