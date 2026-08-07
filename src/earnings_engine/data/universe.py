"""Point-in-time index membership.

Survivorship bias is the single easiest way to manufacture a fake alpha in this
kind of study, and it enters through the universe definition rather than
through the model. If you take today's S&P 500 constituents and run the
backtest from 2014, every company that blew up and was deleted is missing, and
every company that was added had already had the run-up that got it added.

So the universe here is an interval table -- ``(ticker, start_date, end_date)``
-- and membership is asked *as of a date*. A static list is still allowed,
because sometimes it is all you have, but it must be opted into and it stamps
``static_membership=True`` on the result so the caveat survives into the
write-up rather than being forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..utils.frames import UNIVERSE
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

FAR_FUTURE = pd.Timestamp("2100-01-01")


class SurvivorshipBiasError(RuntimeError):
    """Raised when a static constituent list is used without acknowledgement."""


@dataclass
class Universe:
    """Interval-based index membership."""

    members: pd.DataFrame
    name: str = "universe"
    static_membership: bool = False

    @classmethod
    def from_frame(cls, df: pd.DataFrame, name: str = "universe", static: bool = False):
        out = df.copy()
        if "end_date" not in out.columns:
            out["end_date"] = FAR_FUTURE
        out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce").fillna(FAR_FUTURE)
        if "sector" not in out.columns:
            out["sector"] = pd.NA
        return cls(members=UNIVERSE.validate(out), name=name, static_membership=static)

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None):
        path = Path(path)
        df = pd.read_csv(path)
        return cls.from_frame(df, name=name or path.stem, static=False)

    @classmethod
    def static(cls, tickers, sectors=None, *, acknowledge_bias: bool = False, name="static"):
        """Build a universe from a current constituent list.

        ``acknowledge_bias=True`` is required, and is not a formality: results
        from a static universe are not comparable to published PEAD numbers.
        """
        if not acknowledge_bias:
            raise SurvivorshipBiasError(
                "Building a universe from a current constituent list is survivorship-"
                "biased: firms deleted from the index over the sample are missing "
                "entirely. Supply a point-in-time membership file, or pass "
                "acknowledge_bias=True to proceed and have results tagged as biased. "
                "See docs/biases.md."
            )
        log.warning(
            "using a STATIC universe of %d tickers - results are survivorship-biased",
            len(tickers),
        )
        df = pd.DataFrame(
            {
                "ticker": list(tickers),
                "start_date": pd.Timestamp("1900-01-01"),
                "end_date": FAR_FUTURE,
                "sector": list(sectors) if sectors is not None else pd.NA,
            }
        )
        return cls.from_frame(df, name=name, static=True)

    # ---- queries ---------------------------------------------------------

    def as_of(self, date) -> list[str]:
        """Tickers that were members on ``date``."""
        d = pd.Timestamp(date)
        m = self.members
        mask = (m["start_date"] <= d) & (m["end_date"] >= d)
        return sorted(m.loc[mask, "ticker"].unique().tolist())

    def all_tickers(self) -> list[str]:
        """Every ticker that was *ever* a member over the sample."""
        return sorted(self.members["ticker"].unique().tolist())

    def sector_map(self) -> dict[str, str]:
        m = self.members.dropna(subset=["sector"])
        return dict(zip(m["ticker"], m["sector"], strict=False))

    def filter_events(self, events: pd.DataFrame, date_col: str = "t0") -> pd.DataFrame:
        """Keep only events whose issuer was in the index on the event date.

        This is the step people skip. Without it you pick up announcements from
        firms that were not in your investable universe at the time, which
        quietly changes what the backtest is claiming.
        """
        if date_col not in events.columns:
            raise KeyError(f"events frame has no column {date_col!r}")
        m = self.members[["ticker", "start_date", "end_date"]]
        merged = events.merge(m, on="ticker", how="left")
        dates = pd.to_datetime(merged[date_col])
        if isinstance(dates.dtype, pd.DatetimeTZDtype):
            dates = dates.dt.tz_localize(None)
        keep = (merged["start_date"] <= dates) & (merged["end_date"] >= dates)
        kept = merged.loc[keep.fillna(False)].drop(columns=["start_date", "end_date"])
        dropped = len(events) - len(kept)
        if dropped:
            log.info("universe filter dropped %d/%d events", dropped, len(events))
        subset = [c for c in ("event_id",) if c in kept.columns]
        return kept.drop_duplicates(subset=subset).reset_index(drop=True)

    def liquidity_filter(
        self, prices: pd.DataFrame, min_price: float, min_dollar_volume: float
    ) -> pd.DataFrame:
        """Per (ticker, date) tradability flags from a 20-day median.

        Applied *as of* each date with a trailing window, so it never uses
        information from after the date it labels.
        """
        px = prices.sort_values(["ticker", "date"]).copy()
        px["dollar_volume"] = px["adj_close"] * px["volume"]
        grp = px.groupby("ticker", sort=False)
        px["adv20"] = grp["dollar_volume"].transform(
            lambda s: s.rolling(20, min_periods=10).median().shift(1)
        )
        px["prev_close"] = grp["adj_close"].shift(1)
        px["tradable"] = (px["prev_close"] >= min_price) & (px["adv20"] >= min_dollar_volume)
        return px[["ticker", "date", "adv20", "tradable"]]

    def __len__(self) -> int:
        return int(self.members["ticker"].nunique())

    def __repr__(self) -> str:
        tag = " STATIC/BIASED" if self.static_membership else ""
        return f"<Universe {self.name!r} n={len(self)}{tag}>"
