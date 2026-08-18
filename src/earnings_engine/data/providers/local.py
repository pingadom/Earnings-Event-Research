"""Offline-only provider backed by the consolidated ``data/raw`` Parquets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...utils.frames import EVENTS, FILINGS, FUNDAMENTALS, PRICES, UNIVERSE
from ..base import ProviderError
from ..registry import register


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
        path = self.raw_dir / f"{name}.parquet"
        if not path.exists():
            raise ProviderError(
                f"offline dataset missing: {path}. Run python scripts/download_data.py first; "
                "the local provider never falls back to synthetic data."
            )
        return pd.read_parquet(path)

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

    def get_events(self, tickers, start, end) -> pd.DataFrame:
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
            "shares_outstanding": "shares_diluted",
        }
        value_columns = [column for column in mapping if column in raw]
        if raw.empty or not value_columns:
            raise ProviderError("local fundamentals file has no requested SEC facts")
        long = raw.melt(
            id_vars=[
                "ticker",
                "period_end",
                "available_from_utc",
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
        long["available_from_utc"] = pd.to_datetime(long["available_from_utc"], utc=True)
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
        if not self.text_dir.exists() or not any(self.text_dir.glob("*.txt")):
            raise ProviderError(
                "SEC filing metadata is local, but filing bodies were not bulk-downloaded. "
                "Offline research will skip text features explicitly; it will not contact SEC."
            )
        raw = self._load("sec_filings").copy()
        wanted = {str(ticker).upper() for ticker in tickers}
        filed = pd.to_datetime(raw["filing_date"], errors="coerce")
        raw = raw.loc[
            raw["ticker"].astype("string").str.upper().isin(wanted)
            & filed.between(pd.Timestamp(start), pd.Timestamp(end))
            & raw["form"].isin(["10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "40-F"])
        ].copy()
        if raw.empty:
            raise ProviderError("local SEC filings file has no requested filings")
        paths = [
            str(self.text_dir / f"{accession}.txt")
            if (self.text_dir / f"{accession}.txt").exists()
            else ""
            for accession in raw["accession"]
        ]
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
        path = self.text_dir / f"{accession}.txt"
        if not path.exists():
            raise ProviderError(
                f"filing text was not acquired for {accession}; offline mode will leave its "
                "text features missing rather than contact SEC or fabricate content"
            )
        return path.read_text(encoding="utf-8", errors="replace")
