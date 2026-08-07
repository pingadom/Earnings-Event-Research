"""Provider protocols.

Everything the pipeline needs from the outside world is expressed as one of
four narrow interfaces. Swapping yfinance for Capital IQ, or a live pull for a
deterministic synthetic generator, is then a config change rather than a
rewrite -- which is also what makes the test suite hermetic.

Each method returns a tidy dataframe conforming to the matching schema in
:mod:`earnings_engine.utils.frames`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


class ProviderError(RuntimeError):
    """A provider could not satisfy a request (network, auth, bad export...)."""


@runtime_checkable
class PriceProvider(Protocol):
    """Daily OHLCV, split- and dividend-adjusted."""

    name: str

    def get_prices(
        self, tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DataFrame:
        """Return a frame matching :data:`earnings_engine.utils.frames.PRICES`."""
        ...


@runtime_checkable
class EventProvider(Protocol):
    """Earnings announcement dates and, critically, times of day."""

    name: str

    def get_events(
        self, tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DataFrame:
        """Return a frame matching :data:`earnings_engine.utils.frames.EVENTS`."""
        ...


@runtime_checkable
class FundamentalProvider(Protocol):
    """Line items in long form, each stamped with when it became public."""

    name: str

    def get_fundamentals(
        self, tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DataFrame:
        """Return a frame matching :data:`earnings_engine.utils.frames.FUNDAMENTALS`."""
        ...


@runtime_checkable
class FilingProvider(Protocol):
    """Filing metadata plus a handle to the document text."""

    name: str

    def get_filings(
        self, tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DataFrame:
        """Return a frame matching :data:`earnings_engine.utils.frames.FILINGS`."""
        ...

    def get_text(self, accession: str) -> str:
        """Return the plain text of a filing."""
        ...
