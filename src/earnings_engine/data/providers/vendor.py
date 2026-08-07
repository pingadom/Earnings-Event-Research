"""Vendor export providers: Capital IQ, LSEG Workspace, Finaeon.

None of these are free programmatic APIs. In practice you drive them from the
Excel plug-in or Workspace UI, export to CSV/XLSX, and drop the files into
``data/vendor/<dataset>/``. This module turns that drop folder into a first-
class ingestion path: declared column mappings, schema validation, and an
explicit point-in-time policy.

Layout expected::

    data/vendor/
      prices/          any number of .csv/.xlsx exports
      events/
      fundamentals/
      filings/
      universe/

A ``_manifest.yaml`` may sit beside the files to record what was exported,
when, and with which template -- provenance you will want when a result needs
defending.

Point-in-time warning
---------------------
A standard Capital IQ or LSEG fundamentals export gives you the *latest
restated* figures, not what was on the tape at the time. Restatements are not
rare and they bias any study that conditions on reported fundamentals. Two
ways out, in order of preference:

1. export the point-in-time / "as-first-reported" variant (Capital IQ's
   point-in-time financials; LSEG's ``.PIT`` fields);
2. failing that, set ``available_from_policy="filing_lag"`` below, which stamps
   each period with a conservative assumed disclosure lag and records that the
   data is restated so the caveat survives into the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ...utils.frames import EVENTS, FILINGS, FUNDAMENTALS, PRICES, UNIVERSE, Schema
from ...utils.logging_utils import get_logger
from ..base import ProviderError
from ..registry import register

log = get_logger(__name__)

DATASETS = ("prices", "events", "fundamentals", "filings", "universe")

_SCHEMAS: dict[str, Schema] = {
    "prices": PRICES,
    "events": EVENTS,
    "fundamentals": FUNDAMENTALS,
    "filings": FILINGS,
    "universe": UNIVERSE,
}


@dataclass(frozen=True)
class ColumnMap:
    """Vendor column name -> canonical column name, per dataset."""

    prices: dict[str, str] = field(default_factory=dict)
    events: dict[str, str] = field(default_factory=dict)
    fundamentals: dict[str, str] = field(default_factory=dict)
    filings: dict[str, str] = field(default_factory=dict)
    universe: dict[str, str] = field(default_factory=dict)

    def for_dataset(self, dataset: str) -> dict[str, str]:
        return getattr(self, dataset)


#: Capital IQ Excel plug-in default headers (CIQ template names vary by
#: template; adjust here rather than renaming columns by hand in Excel).
CAPITALIQ_MAP = ColumnMap(
    prices={
        "Ticker": "ticker",
        "Date": "date",
        "IQ_OPENPRICE": "open",
        "IQ_HIGHPRICE": "high",
        "IQ_LOWPRICE": "low",
        "IQ_CLOSEPRICE": "close",
        "IQ_CLOSEPRICE_ADJ": "adj_close",
        "IQ_VOLUME": "volume",
    },
    events={
        "Ticker": "ticker",
        "IQ_EARNINGS_ANNOUNCE_DATE": "announced_at_utc",
        "IQ_EARNINGS_ANNOUNCE_TIME": "timing",
        "IQ_PERIODDATE": "period_end",
    },
    fundamentals={
        "Ticker": "ticker",
        "IQ_PERIODDATE": "period_end",
        "IQ_FILINGDATE": "available_from_utc",
        "Item": "item",
        "Value": "value",
    },
    universe={
        "Ticker": "ticker",
        "IQ_INDEX_START": "start_date",
        "IQ_INDEX_END": "end_date",
        "IQ_PRIMARY_INDUSTRY": "sector",
    },
)

#: LSEG Workspace / Datastream export headers.
LSEG_MAP = ColumnMap(
    prices={
        "Instrument": "ticker",
        "Date": "date",
        "Price Open": "open",
        "Price High": "high",
        "Price Low": "low",
        "Price Close": "close",
        "Adjusted Close Price": "adj_close",
        "Volume": "volume",
    },
    events={
        "Instrument": "ticker",
        "Earnings Announcement Date": "announced_at_utc",
        "Announcement Time": "timing",
        "Period End Date": "period_end",
    },
    fundamentals={
        "Instrument": "ticker",
        "Period End Date": "period_end",
        "Original Announcement Date": "available_from_utc",
        "Field": "item",
        "Value": "value",
    },
    universe={
        "Instrument": "ticker",
        "Index Join Date": "start_date",
        "Index Leave Date": "end_date",
        "TRBC Economic Sector Name": "sector",
    },
)

#: Finaeon / Global Financial Data long-history exports.
FINAEON_MAP = ColumnMap(
    prices={
        "Symbol": "ticker",
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adjusted Close": "adj_close",
        "Volume": "volume",
    },
    universe={
        "Symbol": "ticker",
        "Start Date": "start_date",
        "End Date": "end_date",
        "Sector": "sector",
    },
)

_TIMING_ALIASES = {
    "before market open": "bmo",
    "before open": "bmo",
    "bmo": "bmo",
    "pre-market": "bmo",
    "premarket": "bmo",
    "after market close": "amc",
    "after close": "amc",
    "amc": "amc",
    "post-market": "amc",
    "postmarket": "amc",
    "during market hours": "during",
    "intraday": "during",
    "during": "during",
    "unspecified": "unknown",
    "n/a": "unknown",
    "": "unknown",
}


class VendorExportProvider:
    """Reads a vendor drop folder and normalises it to the canonical schemas."""

    name = "vendor"
    column_map: ColumnMap = ColumnMap()
    #: "as_reported" if the export carries a true first-disclosure date,
    #: "filing_lag" to assume a conservative lag from period end.
    available_from_policy: str = "as_reported"
    assumed_filing_lag_days: int = 45

    def __init__(
        self,
        vendor_dir: str | Path = "data/vendor",
        exchange_tz: str = "America/New_York",
        strict: bool = True,
    ) -> None:
        self.vendor_dir = Path(vendor_dir)
        self.exchange_tz = exchange_tz
        self.strict = strict

    # ---- file discovery -------------------------------------------------

    def _files(self, dataset: str) -> list[Path]:
        folder = self.vendor_dir / dataset
        if not folder.exists():
            return []
        return sorted(
            p
            for p in folder.iterdir()
            if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".txt"} and not p.name.startswith("~$")
        )

    def _read_one(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                return pd.read_excel(path)
            except ImportError as exc:  # pragma: no cover
                raise ProviderError(
                    f"reading {path.name} needs openpyxl: pip install -e '.[data]'"
                ) from exc
        sep = "\t" if path.suffix.lower() == ".txt" else ","
        return pd.read_csv(path, sep=sep)

    def load(self, dataset: str) -> pd.DataFrame:
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
        files = self._files(dataset)
        if not files:
            raise ProviderError(
                f"no export files found in {self.vendor_dir / dataset}. "
                f"See data/vendor/README.md for the export recipes."
            )
        frames = []
        for path in files:
            df = self._read_one(path)
            df = self._rename(df, dataset, path)
            df["_source_file"] = path.name
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        df = self._post_process(df, dataset)
        schema = _SCHEMAS[dataset]
        try:
            return schema.validate(df)
        except Exception as exc:
            if self.strict:
                raise ProviderError(f"{self.name}/{dataset}: {exc}") from exc
            log.warning("%s/%s failed validation: %s", self.name, dataset, exc)
            return df

    def _rename(self, df: pd.DataFrame, dataset: str, path: Path) -> pd.DataFrame:
        mapping = self.column_map.for_dataset(dataset)
        lower = {str(c).strip().lower(): c for c in df.columns}
        rename: dict[str, str] = {}
        for vendor_col, canon in mapping.items():
            key = vendor_col.strip().lower()
            if key in lower:
                rename[lower[key]] = canon
        # Allow already-canonical headers to pass through untouched.
        out = df.rename(columns=rename)
        expected = set(_SCHEMAS[dataset].columns)
        got = set(out.columns) & expected
        if not got:
            raise ProviderError(
                f"{path.name}: none of the expected columns were found after mapping. "
                f"Columns present: {sorted(map(str, df.columns))[:12]}"
            )
        return out

    def _post_process(self, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        df = df.copy()
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        if dataset == "events":
            df = self._process_events(df)
        if dataset == "fundamentals":
            df = self._process_fundamentals(df)
        if dataset == "universe" and "end_date" in df.columns:
            df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").fillna(
                pd.Timestamp("2100-01-01")
            )
        for key in ("event_id", "accession"):
            if dataset in {"events", "filings"} and key not in df.columns:
                df[key] = self._synthesise_id(df, dataset)
        return df

    def _process_events(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timing" in df.columns:
            df["timing"] = (
                df["timing"].astype(str).str.strip().str.lower().map(_TIMING_ALIASES).fillna("unknown")
            )
        else:
            df["timing"] = "unknown"
        df["announced_at_utc"] = self._localise(df["announced_at_utc"], df["timing"])
        if "period_end" not in df.columns:
            df["period_end"] = pd.NaT
        if "fiscal_quarter" not in df.columns:
            pe = pd.to_datetime(df["period_end"], errors="coerce")
            year = pe.dt.year.astype("Int64").astype(str)
            quarter = pe.dt.quarter.astype("Int64").astype(str)
            df["fiscal_quarter"] = year + "Q" + quarter
        return df

    def _localise(self, series: pd.Series, timing: pd.Series) -> pd.Series:
        ts = pd.to_datetime(series, errors="coerce")
        if isinstance(ts.dtype, pd.DatetimeTZDtype):
            return ts.dt.tz_convert("UTC")
        # Date-only exports carry no clock time; assign one from the timing flag
        # so downstream alignment is explicit rather than accidental.
        has_time = (ts.dt.hour != 0) | (ts.dt.minute != 0)
        default_hour = timing.map({"bmo": 7, "amc": 16, "during": 12}).fillna(20)
        ts = ts.where(has_time, ts.dt.normalize() + pd.to_timedelta(default_hour, unit="h"))
        localised = ts.dt.tz_localize(
            self.exchange_tz, ambiguous=True, nonexistent="shift_forward"
        )
        return localised.dt.tz_convert("UTC")

    def _process_fundamentals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        if self.available_from_policy == "filing_lag" or "available_from_utc" not in df.columns:
            log.warning(
                "%s: no first-disclosure date in export; assuming a %d-day filing lag. "
                "These fundamentals are RESTATED, not point-in-time - see docs/biases.md.",
                self.name,
                self.assumed_filing_lag_days,
            )
            stamp = df["period_end"] + pd.Timedelta(days=self.assumed_filing_lag_days)
            df["available_from_utc"] = stamp.dt.tz_localize(self.exchange_tz).dt.tz_convert("UTC")
            df["_restated"] = True
        else:
            stamp = pd.to_datetime(df["available_from_utc"], errors="coerce")
            if isinstance(stamp.dtype, pd.DatetimeTZDtype):
                df["available_from_utc"] = stamp.dt.tz_convert("UTC")
            else:
                df["available_from_utc"] = (
                    (stamp.dt.normalize() + pd.Timedelta(hours=17, minutes=30))
                    .dt.tz_localize(self.exchange_tz, ambiguous=True, nonexistent="shift_forward")
                    .dt.tz_convert("UTC")
                )
            df["_restated"] = False
        if "item" in df.columns:
            df["item"] = df["item"].astype(str).str.strip().str.lower().str.replace(" ", "_")
        return df

    @staticmethod
    def _synthesise_id(df: pd.DataFrame, dataset: str) -> pd.Series:
        if dataset == "events":
            stamp = pd.to_datetime(df["announced_at_utc"], errors="coerce", utc=True).dt.date
        else:
            stamp = pd.to_datetime(df["filed_at_utc"], errors="coerce", utc=True).dt.date
        return df["ticker"].astype(str) + "-" + stamp.astype(str)

    # ---- provider interface ---------------------------------------------

    def get_prices(self, tickers, start, end) -> pd.DataFrame:
        return _slice(self.load("prices"), tickers, start, end, "date")

    def get_events(self, tickers, start, end) -> pd.DataFrame:
        return _slice(self.load("events"), tickers, start, end, "announced_at_utc", tz=True)

    def get_fundamentals(self, tickers, start, end) -> pd.DataFrame:
        return _slice(self.load("fundamentals"), tickers, start, end, "period_end")

    def get_filings(self, tickers, start, end) -> pd.DataFrame:
        return _slice(self.load("filings"), tickers, start, end, "filed_at_utc", tz=True)

    def get_universe(self) -> pd.DataFrame:
        return self.load("universe")


def _slice(df, tickers, start, end, col, tz: bool = False) -> pd.DataFrame:
    out = df[df["ticker"].isin(set(tickers))] if tickers else df
    lo = pd.Timestamp(start, tz="UTC") if tz else pd.Timestamp(start)
    hi = pd.Timestamp(end, tz="UTC") if tz else pd.Timestamp(end)
    out = out[(out[col] >= lo) & (out[col] <= hi)]
    return out.reset_index(drop=True)


@register("capitaliq")
class CapitalIQProvider(VendorExportProvider):
    """S&P Capital IQ Excel plug-in exports."""

    name = "capitaliq"
    column_map = CAPITALIQ_MAP


@register("lseg")
class LSEGProvider(VendorExportProvider):
    """LSEG Workspace / Datastream exports."""

    name = "lseg"
    column_map = LSEG_MAP


@register("finaeon")
class FinaeonProvider(VendorExportProvider):
    """Finaeon (Global Financial Data) long-history exports."""

    name = "finaeon"
    column_map = FINAEON_MAP
