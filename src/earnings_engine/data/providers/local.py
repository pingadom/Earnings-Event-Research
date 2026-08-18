"""Offline-only provider backed by the consolidated ``data/raw`` Parquets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...utils.frames import EVENTS, FILINGS, FUNDAMENTALS, PRICES, UNIVERSE
from ...utils.logging_utils import get_logger
from ..base import ProviderError
from ..filing_text import is_earnings_release, read_text
from ..registry import register
from ..storage import read_table, table_path

log = get_logger(__name__)


def conservative_unknown_timestamp(date_value) -> pd.Timestamp:
    """End-of-day Eastern timestamp used only as a conservative trade policy.

    This is deliberately not presented as an observed release time. The raw
    event retains ``announcement_time=unknown`` and ``timestamp_quality`` so
    the downstream alignment records that timing was assumed.
    """
    date = pd.Timestamp(date_value).normalize()
    return (
        (date + pd.Timedelta(hours=23, minutes=59, seconds=59))
        .tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward")
        .tz_convert("UTC")
    )


_ADDITIVE_FLOWS = {
    "revenue",
    "operating_income",
    "net_income",
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
}


def _quarterise_annual_flows(long: pd.DataFrame) -> pd.DataFrame:
    """Derive Q4 from an annual additive flow only when Q1-Q3 are observed.

    This is an exact residual, not interpolation. If any standalone quarterly
    component is absent, the annual flow is removed from the quarterly research
    adapter and remains available only in the raw wide fundamentals file.
    """
    out = long.copy()
    fiscal = out["fiscal_period"].astype("string").str.upper()
    annual = fiscal.eq("FY")
    for index in out.index[annual & out["raw_item"].isin(_ADDITIVE_FLOWS)]:
        row = out.loc[index]
        period_end = pd.Timestamp(row["period_end"])
        candidate_fiscal = out["fiscal_period"].astype("string").str.upper()
        candidate_dates = pd.to_datetime(out["period_end"])
        candidates = out.loc[
            out["ticker"].eq(row["ticker"])
            & out["raw_item"].eq(row["raw_item"])
            & candidate_fiscal.isin({"Q1", "Q2", "Q3"})
            & candidate_dates.between(
                period_end - pd.Timedelta(days=370), period_end - pd.Timedelta(days=1)
            )
        ].sort_values("period_end")
        candidates = candidates.drop_duplicates("fiscal_period", keep="last")
        labels = set(candidates["fiscal_period"].astype(str).str.upper())
        if labels == {"Q1", "Q2", "Q3"}:
            out.loc[index, "value"] = float(row["value"]) - float(candidates["value"].sum())
            out.loc[index, "fiscal_period"] = "Q4_DERIVED"
        else:
            out.loc[index, "value"] = pd.NA
    # Annual EPS is not additive because the denominator is weighted shares.
    out.loc[annual & out["raw_item"].eq("eps"), "value"] = pd.NA
    return out.dropna(subset=["value"])


@register("local")
class LocalProvider:
    """Load real data without performing any network access or synthetic fallback."""

    name = "local"

    def __init__(self, raw_dir: str | Path = "data/raw") -> None:
        self.raw_dir = Path(raw_dir)
        self.text_dir = self.raw_dir / "sec_filing_text"

    def _load(self, name: str) -> pd.DataFrame:
        path = table_path(self.raw_dir / f"{name}.parquet")
        if not path.exists():
            raise ProviderError(
                f"offline dataset missing: {path}. Run python scripts/download_data.py first; "
                "the local provider never falls back to synthetic data."
            )
        return read_table(path)

    def get_universe(self) -> pd.DataFrame:
        frame = self._load("index_membership").copy()
        if "sector" not in frame:
            frame["sector"] = pd.NA
        frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce").fillna(
            pd.Timestamp("2100-01-01")
        )
        return UNIVERSE.validate(frame[["ticker", "start_date", "end_date", "sector"]])

    def get_prices(self, tickers, start, end) -> pd.DataFrame:
        securities = self._load("prices")
        benchmarks = self._load("benchmarks")
        frame = pd.concat([securities, benchmarks], ignore_index=True)
        if "adj_close" not in frame and "adjusted_close" in frame:
            frame["adj_close"] = frame["adjusted_close"]
        wanted = {str(ticker).upper() for ticker in tickers}
        frame["ticker"] = frame["ticker"].astype("string").str.upper()
        dates = pd.to_datetime(frame["date"], errors="coerce")
        mask = frame["ticker"].isin(wanted) & dates.between(pd.Timestamp(start), pd.Timestamp(end))
        frame = frame.loc[mask].drop_duplicates(["ticker", "date"], keep="last")
        if frame.empty:
            raise ProviderError("local price files contain none of the requested symbols/window")
        return PRICES.validate(frame.reset_index(drop=True))

    def _events_from_8k(self, tickers, start, end) -> pd.DataFrame:
        """Earnings events from SEC 8-K Item 2.02 filings.

        Item 2.02 is "Results of Operations and Financial Condition" -- the
        filing a US issuer makes when it releases earnings. Two reasons this is
        the primary source rather than a fallback:

        * ``acceptanceDateTime`` is a real, minute-level, point-in-time stamp.
          Vendor earnings calendars carry a date and frequently no time, and a
          wrong BMO/AMC flag shifts the whole event window by a session.
        * It needs no third-party package and no scraping, so the event table
          is reproducible from a public API a decade from now.

        The fiscal ``period_end`` is *inferred* as the calendar quarter end
        preceding the announcement. Nothing downstream keys on it -- fundamental
        features attach by publication time -- but it is an approximation and is
        flagged as such rather than presented as reported.
        """
        filings = self._load("sec_filings")
        wanted = {str(t).upper() for t in tickers}
        form = filings["form"].astype("string").str.upper()
        items = filings["items"].astype("string").fillna("")
        mask = (
            filings["ticker"].astype("string").str.upper().isin(wanted)
            & form.str.startswith("8-K")
            & items.str.contains("2.02", regex=False)
        )
        eight_k = filings.loc[mask].copy()
        if eight_k.empty:
            raise ProviderError("no 8-K Item 2.02 filings for the requested tickers")

        stamp = pd.to_datetime(eight_k["accepted_at_utc"], utc=True, errors="coerce")
        eight_k = eight_k.loc[stamp.notna()].copy()
        stamp = stamp.loc[eight_k.index]
        window = (stamp >= pd.Timestamp(start, tz="UTC")) & (stamp <= pd.Timestamp(end, tz="UTC"))
        eight_k, stamp = eight_k.loc[window], stamp.loc[window]
        if eight_k.empty:
            raise ProviderError("no 8-K Item 2.02 filings inside the requested window")

        local = stamp.dt.tz_convert("America/New_York")
        eight_k["announced_at_utc"] = stamp
        eight_k["ticker"] = eight_k["ticker"].astype("string").str.upper()
        # The clock time is real, so the BMO/AMC flag is derived rather than
        # assumed -- which is exactly what a vendor calendar cannot give us.
        minutes = local.dt.hour * 60 + local.dt.minute
        eight_k["timing"] = np.where(
            minutes < 9 * 60 + 30, "bmo", np.where(minutes >= 16 * 60, "amc", "during")
        )
        quarter = local.dt.tz_localize(None).dt.to_period("Q")
        eight_k["period_end"] = (quarter - 1).dt.end_time.dt.normalize()
        eight_k["fiscal_quarter"] = (quarter - 1).astype(str)
        # The accession alone is not unique. A filer with more than one listed
        # share class -- GOOGL and GOOG, FOXA and FOX, NWSA and NWS -- files a
        # single 8-K under a single CIK, and it arrives here once per ticker.
        # Each class is separately tradable with its own price series, so both
        # are real events; they just are not the same event.
        eight_k["event_id"] = (
            eight_k["ticker"].astype("string") + "-" + eight_k["accession"].astype("string")
        )

        # A company can file more than one Item 2.02 in a quarter (a
        # pre-announcement, then the full release). Keep the earliest: that is
        # when the market first learned something.
        eight_k = eight_k.sort_values("announced_at_utc").drop_duplicates(
            ["ticker", "fiscal_quarter"], keep="first"
        )
        columns = [
            "ticker", "event_id", "announced_at_utc", "timing", "period_end", "fiscal_quarter",
        ]
        return EVENTS.validate(eight_k[columns].reset_index(drop=True))

    def get_events(self, tickers, start, end) -> pd.DataFrame:
        """Earnings events, preferring SEC 8-K Item 2.02 over the vendor file."""
        try:
            return self._events_from_8k(tickers, start, end)
        except ProviderError as exc:
            log.info("8-K Item 2.02 events unavailable (%s); using the earnings file", exc)
        raw = self._load("earnings").copy()
        wanted = {str(ticker).upper() for ticker in tickers}
        raw["ticker"] = raw["ticker"].astype("string").str.upper()
        dates = pd.to_datetime(raw["earnings_date"], errors="coerce")
        raw = raw.loc[
            raw["ticker"].isin(wanted) & dates.between(pd.Timestamp(start), pd.Timestamp(end))
        ].copy()
        if raw.empty:
            raise ProviderError("local earnings file contains no requested events")
        announced = pd.to_datetime(raw["announced_at_utc"], errors="coerce", utc=True)
        unknown = announced.isna()
        if unknown.any():
            announced.loc[unknown] = [
                conservative_unknown_timestamp(value) for value in raw.loc[unknown, "earnings_date"]
            ]
        period_end = pd.to_datetime(raw.get("fiscal_period"), errors="coerce")
        fiscal_quarter = period_end.map(
            lambda value: f"{value.year}Q{value.quarter}" if pd.notna(value) else pd.NA
        )
        base_id = raw.get("accession", pd.Series(pd.NA, index=raw.index)).astype("string")
        fallback_id = (
            raw["ticker"].astype(str)
            + "-"
            + pd.to_datetime(raw["earnings_date"]).dt.strftime("%Y-%m-%d")
            + "-"
            + raw.groupby(["ticker", "earnings_date"]).cumcount().astype(str)
        )
        event_id = base_id.where(base_id.notna() & base_id.ne(""), fallback_id)
        frame = pd.DataFrame(
            {
                "ticker": raw["ticker"].to_numpy(),
                "event_id": event_id.to_numpy(),
                "announced_at_utc": announced.to_numpy(),
                "timing": raw["timing"].fillna("unknown").to_numpy(),
                "period_end": period_end.to_numpy(),
                "fiscal_quarter": fiscal_quarter.to_numpy(),
                "timestamp_quality": raw.get(
                    "timestamp_quality", pd.Series("unknown", index=raw.index)
                ).to_numpy(),
                "source": raw.get("source", pd.Series("unknown", index=raw.index)).to_numpy(),
            }
        ).drop_duplicates("event_id", keep="first")
        return EVENTS.validate(frame.reset_index(drop=True))

    @staticmethod
    def _conservative_availability(
        accepted: pd.Series, filing_date: pd.Series
    ) -> pd.Series:
        """The later of EDGAR's acceptance timestamp and its filing date.

        The two disagree on a meaningful minority of filings, in both
        directions, and SEC does not document unambiguously which governs
        public availability. Taking the later of the two can only ever make the
        study more conservative: a feature is treated as knowable slightly
        after it may really have been, never before.

        In practice this costs nothing. An after-hours acceptance resolves to
        the following midnight Eastern, and the event aligner already pushes
        such announcements to the next session's 09:30 open -- so the feature is
        still available when it is needed.
        """
        accepted = pd.to_datetime(accepted, utc=True, errors="coerce")
        floor = (
            pd.to_datetime(filing_date, errors="coerce")
            .dt.normalize()
            .dt.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward")
            .dt.tz_convert("UTC")
        )
        return accepted.where(accepted >= floor, floor)

    def get_fundamentals(self, tickers, start, end) -> pd.DataFrame:
        raw = self._load("fundamentals").copy()
        wanted = {str(ticker).upper() for ticker in tickers}
        periods = pd.to_datetime(raw["period_end"], errors="coerce")
        raw = raw.loc[
            raw["ticker"].astype("string").str.upper().isin(wanted)
            & periods.between(pd.Timestamp(start), pd.Timestamp(end))
        ].copy()
        mapping = {
            "revenue": "revenue",
            "gross_profit": "gross_profit",
            "operating_income": "operating_income",
            "net_income": "net_income",
            "eps": "eps_diluted",
            "total_assets": "total_assets",
            "total_liabilities": "total_liabilities",
            "cash": "cash",
            "debt": "total_debt",
            "shareholders_equity": "shareholders_equity",
            "operating_cash_flow": "cfo",
            "capital_expenditure": "capex",
            "free_cash_flow": "free_cash_flow",
            # Weighted-average diluted shares is the denominator earnings per
            # share is actually computed on, and it is dated to the fiscal
            # period like every other line item.
            #
            # ``shares_outstanding`` is deliberately absent. It comes from the
            # 10-Q cover page and is stamped "as of" the *filing* date, which
            # is two to three weeks after the period it accompanies. Emitting
            # it created a (ticker, period_end) row on a date no fiscal quarter
            # ever ended -- 68 of Apple's 127 periods were phantoms of exactly
            # this kind -- and every feature that looks back four rows for "the
            # same quarter last year" was counting those phantoms. It stays in
            # ``fundamentals.parquet`` for provenance; it does not enter the
            # research frame.
            "shares_diluted": "shares_diluted",
        }
        value_columns = [column for column in mapping if column in raw]
        if raw.empty or not value_columns:
            raise ProviderError("local fundamentals file has no requested SEC facts")
        long = raw.melt(
            id_vars=[
                "ticker",
                "period_end",
                "available_from_utc",
                "filing_date",
                "accession",
                "fiscal_year",
                "fiscal_period",
                "form",
            ],
            value_vars=value_columns,
            var_name="raw_item",
            value_name="value",
        ).dropna(subset=["value", "available_from_utc"])
        long["item"] = long["raw_item"].map(mapping)
        # Conservative availability: never earlier than EDGAR's own filing date.
        long["available_from_utc"] = self._conservative_availability(
            long["available_from_utc"], long["filing_date"]
        )
        # Canonical research frames are unique per period/item. Select the
        # earliest public observation; later amendment rows remain preserved
        # in fundamentals.parquet for provenance/audit.
        long = long.sort_values("available_from_utc").drop_duplicates(
            ["ticker", "period_end", "item"], keep="first"
        )
        long = _quarterise_annual_flows(long)
        frame = long[["ticker", "period_end", "available_from_utc", "item", "value"]]
        return FUNDAMENTALS.validate(frame.reset_index(drop=True))

    def get_filings(self, tickers, start, end) -> pd.DataFrame:
        """Earnings releases, in filing order, for the text feature block.

        The corpus is deliberately one document type: the Item 2.02 press
        release. A similarity measured across a mixture of releases and 10-Qs
        would be measuring genre, not editorial change -- see
        :mod:`earnings_engine.data.filing_text`.
        """
        if not self.text_dir.exists() or not any(self.text_dir.glob("*.txt*")):
            raise ProviderError(
                "SEC filing metadata is local, but filing bodies were not bulk-downloaded. "
                "Run `eee download --text-only`, or leave features.text off: offline research "
                "will skip text features explicitly; it will not contact SEC."
            )
        raw = self._load("sec_filings").copy()
        wanted = {str(ticker).upper() for ticker in tickers}
        filed = pd.to_datetime(raw["filing_date"], errors="coerce")
        is_release = raw["form"].isin(["8-K", "8-K/A"]) & raw["items"].map(is_earnings_release)
        raw = raw.loc[
            raw["ticker"].astype("string").str.upper().isin(wanted)
            & filed.between(pd.Timestamp(start), pd.Timestamp(end))
            & is_release
        ].copy()
        if raw.empty:
            raise ProviderError("local SEC filings file has no earnings releases in this window")
        paths = [
            str(self.text_dir / f"{accession}.txt.gz")
            if (self.text_dir / f"{accession}.txt.gz").exists()
            else ""
            for accession in raw["accession"]
        ]
        # A release whose body was never acquired would enter the corpus as an
        # empty string and drag every similarity around it towards zero.
        raw = raw.loc[[bool(path) for path in paths]]
        paths = [path for path in paths if path]
        if raw.empty:
            raise ProviderError("no earnings-release bodies were acquired for these tickers")
        frame = pd.DataFrame(
            {
                "ticker": raw["ticker"],
                "accession": raw["accession"],
                "form": raw["form"],
                "filed_at_utc": pd.to_datetime(raw["accepted_at_utc"], utc=True),
                "period_end": pd.to_datetime(raw["period_end"], errors="coerce"),
                "path": paths,
            }
        )
        return FILINGS.validate(frame.reset_index(drop=True))

    def get_text(self, accession: str) -> str:
        text = read_text(self.text_dir, accession)
        if not text:
            raise ProviderError(
                f"filing text was not acquired for {accession}; offline mode will leave its "
                "text features missing rather than contact SEC or fabricate content"
            )
        return text
