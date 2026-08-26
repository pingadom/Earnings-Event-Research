"""Schema contracts for the tidy frames that move between pipeline stages.

This is the **pipeline** layer. For the *storage* contracts that govern
downloaded files under ``data/raw`` -- which carry extra provenance columns and
raw vendor units -- see :mod:`earnings_engine.data.schemas`. Providers map from
storage into these frames; nothing downstream sees the storage shape.

Every stage declares the columns and dtypes it emits, and validation runs at
the boundary. This is cheap insurance: the failure mode it prevents is a
silently-misaligned join that produces a plausible-looking Sharpe ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SchemaError(ValueError):
    """Raised when a dataframe does not satisfy its declared schema."""


@dataclass(frozen=True)
class Schema:
    name: str
    columns: dict[str, str]
    required: tuple[str, ...] = ()
    unique_on: tuple[str, ...] = ()

    def validate(self, df: pd.DataFrame, *, coerce: bool = True) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise SchemaError(
                f"{self.name}: missing column(s) {missing}; got {sorted(df.columns)}"
            )
        out = df.copy()
        if coerce:
            for col, dtype in self.columns.items():
                out[col] = _coerce_column(out[col], dtype, f"{self.name}.{col}")
        for col in self.required or tuple(self.columns):
            if col in self.required and out[col].isna().any():
                n = int(out[col].isna().sum())
                raise SchemaError(f"{self.name}: column {col!r} has {n} null value(s)")
        if self.unique_on:
            dup = out.duplicated(subset=list(self.unique_on))
            if dup.any():
                sample = out.loc[dup, list(self.unique_on)].head(3).to_dict("records")
                raise SchemaError(
                    f"{self.name}: {int(dup.sum())} duplicate row(s) on {self.unique_on}; "
                    f"e.g. {sample}"
                )
        ordered = list(self.columns) + [c for c in out.columns if c not in self.columns]
        return out[ordered]


def _coerce_column(s: pd.Series, dtype: str, label: str) -> pd.Series:
    try:
        if dtype == "datetime64[ns]":
            out = pd.to_datetime(s, errors="coerce")
            if isinstance(out.dtype, pd.DatetimeTZDtype):
                out = out.dt.tz_convert("UTC").dt.tz_localize(None)
            return out
        if dtype == "datetime64[ns, UTC]":
            out = pd.to_datetime(s, errors="coerce", utc=True)
            return out
        if dtype.startswith("float"):
            return pd.to_numeric(s, errors="coerce").astype("float64")
        if dtype.startswith("int"):
            return pd.to_numeric(s, errors="coerce").astype("Int64")
        if dtype == "string":
            return s.astype("string")
        if dtype == "bool":
            return s.astype("boolean")
        return s.astype(dtype)
    except Exception as exc:  # pragma: no cover - defensive
        raise SchemaError(f"{label}: cannot coerce to {dtype}: {exc}") from exc


# --- the contracts ----------------------------------------------------------

PRICES = Schema(
    name="prices",
    columns={
        "ticker": "string",
        "date": "datetime64[ns]",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "adj_close": "float64",
        "volume": "float64",
    },
    required=("ticker", "date", "adj_close"),
    unique_on=("ticker", "date"),
)

EVENTS = Schema(
    name="events",
    columns={
        "ticker": "string",
        "event_id": "string",
        # Wall-clock instant the results hit the tape, in UTC.
        "announced_at_utc": "datetime64[ns, UTC]",
        # 'bmo' | 'amc' | 'during' | 'unknown'
        "timing": "string",
        "period_end": "datetime64[ns]",
        "fiscal_quarter": "string",
    },
    required=("ticker", "event_id", "announced_at_utc"),
    unique_on=("event_id",),
)

FUNDAMENTALS = Schema(
    name="fundamentals",
    columns={
        "ticker": "string",
        "period_end": "datetime64[ns]",
        # The instant this line item became public. This column is the whole
        # point of the schema: without it, look-ahead bias is unprovable.
        "available_from_utc": "datetime64[ns, UTC]",
        "item": "string",
        "value": "float64",
    },
    required=("ticker", "period_end", "available_from_utc", "item"),
    unique_on=("ticker", "period_end", "item"),
)

CONSENSUS = Schema(
    name="consensus",
    columns={
        "ticker": "string",
        "period_end": "datetime64[ns]",
        # When this *snapshot of the forecast* was observable -- which is not
        # period_end and not the announcement. A consensus is a forecast, and
        # the entire question it answers is what analysts expected *before* the
        # print. Capital IQ's IQ_EPS_EST returns today's consensus unless an
        # asOfDate is supplied, so a careless pull stamps a 2019 period with a
        # number formed years after the fact. This column is what makes that
        # mistake detectable instead of invisible.
        "available_from_utc": "datetime64[ns, UTC]",
        "consensus_eps": "float64",
        # Cross-sectional dispersion of the individual estimates. It is the
        # denominator of analyst SUE, so a zero or missing value must produce
        # NaN rather than an infinite surprise.
        "consensus_std": "float64",
        "n_estimates": "Int64",
    },
    required=("ticker", "period_end", "available_from_utc", "consensus_eps"),
    unique_on=("ticker", "period_end"),
)

FILINGS = Schema(
    name="filings",
    columns={
        "ticker": "string",
        "accession": "string",
        "form": "string",
        "filed_at_utc": "datetime64[ns, UTC]",
        "period_end": "datetime64[ns]",
        "path": "string",
    },
    required=("ticker", "accession", "form", "filed_at_utc"),
    # Not unique on accession alone. A filer with more than one listed share
    # class -- GOOGL and GOOG, FOXA and FOX -- files a single document under a
    # single CIK, and it is a separate observation for each ticker because each
    # class has its own price series. Keying on the accession alone rejected the
    # whole filings frame, which silently disabled every text feature.
    unique_on=("ticker", "accession"),
)

UNIVERSE = Schema(
    name="universe",
    columns={
        "ticker": "string",
        "start_date": "datetime64[ns]",
        "end_date": "datetime64[ns]",
        "sector": "string",
    },
    required=("ticker", "start_date"),
)
