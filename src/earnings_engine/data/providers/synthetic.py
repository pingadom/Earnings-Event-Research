"""A deterministic synthetic market.

This is not a toy convenience: it is the backbone of the test suite and of the
offline demo. Because the data-generating process is known, we can assert that
the event-study machinery recovers an effect we planted, and that it finds
*nothing* when we plant nothing. Those two tests catch most sign errors,
off-by-one window bugs and accidental look-ahead.

The generating process
----------------------
Daily excess returns follow a two-factor structure::

    r_it = beta_i * r_mt + gamma_i * f_{s(i),t} + eps_it + announcement_effect

Each ticker announces quarterly. Every announcement draws a latent standardised
surprise ``z ~ N(0, 1)``. That single latent variable drives, jointly:

* an immediate jump on the first tradable session, ``jump_coef * z``;
* a slow post-announcement drift spread over the next ``drift_days`` sessions,
  ``drift_coef * z`` in total -- i.e. a planted PEAD effect;
* the reported fundamentals (revenue growth, margin change, EPS);
* the tone of the accompanying filing text.

so the fundamental, textual and return signatures of a "good quarter" all line
up the way they do in real data, with a signal-to-noise ratio you control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...utils.calendar import default_calendar
from ...utils.frames import EVENTS, FILINGS, FUNDAMENTALS, PRICES, UNIVERSE
from ..registry import register

SECTORS = (
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Industrials",
    "Energy",
    "Consumer Staples",
    "Utilities",
)

_POSITIVE_WORDS = (
    "strong", "growth", "outperformed", "record", "momentum", "improved", "expanded",
    "confident", "robust", "favourable", "accelerating", "gains", "efficient"
)
_NEGATIVE_WORDS = (
    "weak", "decline", "shortfall", "pressure", "impairment", "restructuring", "headwinds",
    "disappointing", "loss", "deteriorating", "adverse", "writedown", "litigation",
    "lawsuit", "regulatory"
)
_UNCERTAIN_WORDS = (
    "may", "could", "approximately", "uncertain", "possibly", "risk", "exposure", "depends",
    "volatility", "contingent", "anticipate", "believe", "estimate"
)
_NEUTRAL_WORDS = (
    "quarter", "revenue", "segment", "operations", "customers", "products", "market",
    "results", "period", "business", "fiscal", "reported", "compared", "management"
)


@dataclass(frozen=True)
class SyntheticSpec:
    """Knobs on the data-generating process."""

    n_tickers: int = 60
    start: str = "2014-01-02"
    end: str = "2024-12-31"
    seed: int = 20260818
    market_vol: float = 0.010
    sector_vol: float = 0.007
    idio_vol: float = 0.016
    market_drift: float = 0.0003
    #: Same-session price reaction per unit of standardised surprise.
    jump_coef: float = 0.020
    #: Total post-announcement drift per unit of surprise, spread over
    #: ``drift_days`` sessions. Set to 0.0 for the null-hypothesis fixture.
    drift_coef: float = 0.008
    drift_days: int = 20
    #: Probability an announcement is released before the open.
    p_bmo: float = 0.45
    #: Probability the release time is unrecorded (mirrors free data sources).
    p_unknown_time: float = 0.10


@register("synthetic")
class SyntheticProvider:
    """Implements every provider protocol at once, from one seeded process."""

    name = "synthetic"

    def __init__(self, spec: SyntheticSpec | None = None, **kwargs) -> None:
        if spec is None:
            spec = SyntheticSpec(**kwargs)
        elif kwargs:
            raise TypeError("pass either a SyntheticSpec or keyword arguments, not both")
        self.spec = spec
        self.calendar = default_calendar()
        self._rng = np.random.default_rng(spec.seed)
        self._built = False

    # ---- generation ---------------------------------------------------

    def _build(self) -> None:
        if self._built:
            return
        spec = self.spec
        rng = self._rng
        sessions = self.calendar.sessions_between(spec.start, spec.end)
        n_days = len(sessions)
        n = spec.n_tickers

        tickers = [f"SYN{i:03d}" for i in range(n)]
        sectors = [SECTORS[i % len(SECTORS)] for i in range(n)]
        betas = rng.normal(1.0, 0.25, size=n).clip(0.3, 2.0)
        gammas = rng.normal(0.8, 0.20, size=n).clip(0.1, 1.6)

        market = rng.normal(spec.market_drift, spec.market_vol, size=n_days)
        sector_f = rng.normal(0.0, spec.sector_vol, size=(n_days, len(SECTORS)))
        idio = rng.normal(0.0, spec.idio_vol, size=(n_days, n))

        sector_idx = np.array([SECTORS.index(s) for s in sectors])
        returns = betas * market[:, None] + gammas * sector_f[:, sector_idx] + idio

        events = self._make_events(tickers, sessions, rng)
        returns = self._inject_effects(returns, events, sessions)

        self._sessions = sessions
        self._tickers = tickers
        self._sectors = dict(zip(tickers, sectors, strict=False))
        self._market = market
        self._returns = returns
        self._events = events
        self._prices = self._make_prices(tickers, sessions, returns, market, rng)
        self._built = True

    def _make_events(self, tickers, sessions, rng) -> pd.DataFrame:
        spec = self.spec
        rows = []
        first, last = sessions[0], sessions[-1]
        for i, ticker in enumerate(tickers):
            # Stagger fiscal calendars so announcements are not all on one day.
            offset_months = i % 3
            period = pd.Timestamp(spec.start) + pd.offsets.QuarterEnd(0)
            period = period + pd.DateOffset(months=offset_months)
            q = 0
            while period < last:
                lag = int(rng.integers(21, 46))  # reporting lag in calendar days
                announce_day = period + pd.Timedelta(days=lag)
                if announce_day <= first or announce_day >= last:
                    period = period + pd.offsets.QuarterEnd(1)
                    continue
                session = self.calendar.next_session(announce_day, inclusive=True)
                u = rng.random()
                if u < spec.p_unknown_time:
                    timing = "unknown"
                    local_time = "20:00"
                elif u < spec.p_unknown_time + spec.p_bmo:
                    timing = "bmo"
                    local_time = "07:00"
                else:
                    timing = "amc"
                    local_time = "16:15"
                # A 'bmo' release is stamped on the session it precedes; an
                # 'amc' release is stamped on the previous session's evening.
                stamp_day = session if timing == "bmo" else self.calendar.previous_session(session)
                ts_local = pd.Timestamp(f"{stamp_day.date()} {local_time}").tz_localize(
                    "America/New_York"
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "event_id": f"{ticker}-{period.date()}",
                        "announced_at_utc": ts_local.tz_convert("UTC"),
                        "timing": timing,
                        "period_end": period,
                        "fiscal_quarter": f"{period.year}Q{period.quarter}",
                        "z": float(rng.normal()),
                    }
                )
                period = period + pd.offsets.QuarterEnd(1)
                q += 1
        return pd.DataFrame(rows)

    def _inject_effects(self, returns, events, sessions) -> np.ndarray:
        spec = self.spec
        pos = pd.Series(np.arange(len(sessions)), index=sessions)
        tick_idx = {t: i for i, t in enumerate(self._ticker_order(events))}
        per_day = spec.drift_coef / max(spec.drift_days, 1)
        for row in events.itertuples(index=False):
            t0 = self._t0_for(row.announced_at_utc, row.timing)
            if t0 not in pos.index:
                continue
            j = int(pos.loc[t0])
            i = tick_idx[row.ticker]
            returns[j, i] += spec.jump_coef * row.z
            hi = min(j + spec.drift_days, len(sessions) - 1)
            if hi > j:
                returns[j + 1 : hi + 1, i] += per_day * row.z
        return returns

    def _ticker_order(self, events) -> list[str]:
        return [f"SYN{i:03d}" for i in range(self.spec.n_tickers)]

    def _t0_for(self, announced_at_utc, timing) -> pd.Timestamp:
        local = pd.Timestamp(announced_at_utc).tz_convert("America/New_York")
        day = local.normalize().tz_localize(None)
        if timing == "bmo" and self.calendar.is_session(day):
            return day
        return self.calendar.next_session(day)

    def _make_prices(self, tickers, sessions, returns, market, rng) -> pd.DataFrame:
        n_days, n = returns.shape
        start_px = rng.uniform(15, 400, size=n)
        levels = start_px * np.cumprod(1.0 + returns, axis=0)
        base_vol = rng.uniform(3e5, 8e6, size=n)
        vol_noise = np.exp(rng.normal(0, 0.35, size=(n_days, n)))
        volume = base_vol * vol_noise

        frames = []
        intraday = np.abs(rng.normal(0, 0.006, size=(n_days, n)))
        for i, ticker in enumerate(tickers):
            close = levels[:, i]
            prev = np.concatenate([[start_px[i]], close[:-1]])
            frames.append(
                pd.DataFrame(
                    {
                        "ticker": ticker,
                        "date": sessions,
                        "open": prev * (1 + rng.normal(0, 0.002, size=n_days)),
                        "high": np.maximum(close, prev) * (1 + intraday[:, i]),
                        "low": np.minimum(close, prev) * (1 - intraday[:, i]),
                        "close": close,
                        "adj_close": close,
                        "volume": volume[:, i],
                    }
                )
            )
        # The market proxy, so `market_symbol` resolves like any other ticker.
        spy = 400.0 * np.cumprod(1.0 + market)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": "SPY",
                    "date": sessions,
                    "open": spy,
                    "high": spy,
                    "low": spy,
                    "close": spy,
                    "adj_close": spy,
                    "volume": 8e7,
                }
            )
        )
        return pd.concat(frames, ignore_index=True)

    # ---- provider interface -------------------------------------------

    def get_universe(self) -> pd.DataFrame:
        self._build()
        df = pd.DataFrame(
            {
                "ticker": self._tickers,
                "start_date": self._sessions[0],
                "end_date": self._sessions[-1],
                "sector": [self._sectors[t] for t in self._tickers],
            }
        )
        return UNIVERSE.validate(df)

    def get_prices(self, tickers, start, end) -> pd.DataFrame:
        self._build()
        wanted = set(tickers)
        df = self._prices[self._prices["ticker"].isin(wanted)]
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        return PRICES.validate(df.reset_index(drop=True))

    def get_events(self, tickers, start, end) -> pd.DataFrame:
        self._build()
        wanted = set(tickers)
        df = self._events[self._events["ticker"].isin(wanted)].copy()
        ts = df["announced_at_utc"]
        df = df[(ts >= pd.Timestamp(start, tz="UTC")) & (ts <= pd.Timestamp(end, tz="UTC"))]
        return EVENTS.validate(df.drop(columns=["z"]).reset_index(drop=True))

    def get_fundamentals(self, tickers, start, end) -> pd.DataFrame:
        """Line items whose growth/margins are driven by the same latent ``z``."""
        self._build()
        rng = np.random.default_rng(self.spec.seed + 1)
        wanted = set(tickers)
        ev = self._events[self._events["ticker"].isin(wanted)].sort_values(
            ["ticker", "period_end"]
        )
        rows = []
        for ticker, grp in ev.groupby("ticker", sort=False):
            revenue = float(rng.uniform(500e6, 20e9))
            margin = float(rng.uniform(0.06, 0.28))
            shares = float(rng.uniform(50e6, 3e9))
            assets = revenue * float(rng.uniform(1.2, 3.0))
            debt = assets * float(rng.uniform(0.05, 0.45))
            for row in grp.itertuples(index=False):
                growth = 0.01 + 0.02 * row.z + rng.normal(0, 0.02)
                revenue *= 1.0 + growth
                margin = float(np.clip(margin + 0.004 * row.z + rng.normal(0, 0.004), 0.01, 0.6))
                op_income = revenue * margin
                net_income = op_income * float(rng.uniform(0.6, 0.85))
                capex = revenue * float(rng.uniform(0.02, 0.09))
                cfo = net_income * float(rng.uniform(1.0, 1.6))
                debt *= 1.0 + rng.normal(0.0, 0.03)
                shares *= 1.0 + rng.normal(-0.002, 0.004)
                assets *= 1.0 + growth * 0.6
                items = {
                    "revenue": revenue,
                    "gross_profit": revenue * min(margin * 2.2, 0.85),
                    "operating_income": op_income,
                    "net_income": net_income,
                    "eps_diluted": net_income / shares,
                    "cfo": cfo,
                    "capex": capex,
                    "total_debt": debt,
                    "cash": assets * 0.08,
                    "total_assets": assets,
                    "shares_diluted": shares,
                }
                for item, value in items.items():
                    rows.append(
                        {
                            "ticker": ticker,
                            "period_end": row.period_end,
                            # Fundamentals become public with the release.
                            "available_from_utc": row.announced_at_utc,
                            "item": item,
                            "value": float(value),
                        }
                    )
        df = pd.DataFrame(rows)
        mask = (df["period_end"] >= pd.Timestamp(start)) & (df["period_end"] <= pd.Timestamp(end))
        return FUNDAMENTALS.validate(df[mask].reset_index(drop=True))

    def get_filings(self, tickers, start, end) -> pd.DataFrame:
        self._build()
        wanted = set(tickers)
        ev = self._events[self._events["ticker"].isin(wanted)].copy()
        ts = ev["announced_at_utc"]
        ev = ev[(ts >= pd.Timestamp(start, tz="UTC")) & (ts <= pd.Timestamp(end, tz="UTC"))]
        df = pd.DataFrame(
            {
                "ticker": ev["ticker"].to_numpy(),
                "accession": ev["event_id"].to_numpy(),
                "form": np.where(ev["period_end"].dt.quarter.to_numpy() == 4, "10-K", "10-Q"),
                "filed_at_utc": ev["announced_at_utc"].to_numpy(),
                "period_end": ev["period_end"].to_numpy(),
                "path": "",
            }
        )
        self._filing_z = dict(zip(ev["event_id"], ev["z"], strict=False))
        return FILINGS.validate(df.reset_index(drop=True))

    def get_text(self, accession: str) -> str:
        """Synthesise MD&A-like prose whose tone tracks the latent surprise."""
        self._build()
        if not hasattr(self, "_filing_z"):
            self._filing_z = dict(zip(self._events["event_id"], self._events["z"], strict=False))
        if accession not in self._filing_z:
            raise KeyError(f"unknown accession {accession!r}")
        z = self._filing_z[accession]
        rng = np.random.default_rng(abs(hash(accession)) % (2**32))
        # Length varies with the quarter: firms write more when there is more
        # to explain, which is itself part of the "Lazy Prices" effect.
        n_words = int(rng.integers(420, 700) - 60 * z)
        p_pos = float(np.clip(0.04 + 0.020 * z, 0.005, 0.20))
        p_neg = float(np.clip(0.04 - 0.020 * z, 0.005, 0.20))
        p_unc = float(np.clip(0.05 - 0.010 * z, 0.005, 0.20))
        pools = [_POSITIVE_WORDS, _NEGATIVE_WORDS, _UNCERTAIN_WORDS, _NEUTRAL_WORDS]
        probs = np.array([p_pos, p_neg, p_unc, max(1e-6, 1 - p_pos - p_neg - p_unc)])
        probs = probs / probs.sum()
        choice = rng.choice(4, size=n_words, p=probs)
        words = [pools[c][rng.integers(len(pools[c]))] for c in choice]
        return " ".join(words)

    # ---- introspection for tests ---------------------------------------

    def ground_truth(self) -> pd.DataFrame:
        """The latent surprise per event. Tests only -- never a model input."""
        self._build()
        return self._events[["event_id", "ticker", "period_end", "z"]].copy()
