"""Shared fixtures.

The whole suite runs against the synthetic provider: deterministic, offline,
and with a known data-generating process, so tests can assert that the
machinery recovers an effect that was planted and finds nothing when it was
not. That is a much stronger guarantee than asserting shapes.
"""

from __future__ import annotations

import pytest

from earnings_engine.config import Config, EventsConfig, ReturnsConfig
from earnings_engine.data.providers.synthetic import SyntheticProvider, SyntheticSpec
from earnings_engine.events import align_events
from earnings_engine.returns import ReturnPanel, compute_abnormal_returns

START = "2016-01-04"
END = "2022-12-30"


def _provider(**kwargs) -> SyntheticProvider:
    spec = SyntheticSpec(n_tickers=40, start=START, end=END, seed=7, **kwargs)
    return SyntheticProvider(spec)


@pytest.fixture(scope="session")
def provider() -> SyntheticProvider:
    return _provider()


@pytest.fixture(scope="session")
def null_provider() -> SyntheticProvider:
    """Same process with no planted drift and no announcement jump."""
    return _provider(drift_coef=0.0, jump_coef=0.0)


@pytest.fixture(scope="session")
def universe(provider):
    return provider.get_universe()


@pytest.fixture(scope="session")
def tickers(universe):
    return list(universe["ticker"])


@pytest.fixture(scope="session")
def prices(provider, tickers):
    return provider.get_prices([*tickers, "SPY"], START, END)


@pytest.fixture(scope="session")
def raw_events(provider, tickers):
    return provider.get_events(tickers, START, END)


@pytest.fixture(scope="session")
def events(raw_events, prices):
    return align_events(raw_events, prices, EventsConfig())


@pytest.fixture(scope="session")
def sector_map(universe):
    return dict(zip(universe["ticker"], universe["sector"], strict=False))


@pytest.fixture(scope="session")
def panel(prices, sector_map):
    return ReturnPanel.from_prices(prices, "SPY", sector_map)


@pytest.fixture(scope="session")
def study(events, panel):
    return compute_abnormal_returns(events, panel, ReturnsConfig())


@pytest.fixture(scope="session")
def fundamentals(provider, tickers):
    return provider.get_fundamentals(tickers, START, END)


@pytest.fixture(scope="session")
def filings(provider, tickers):
    return provider.get_filings(tickers, START, END)


@pytest.fixture()
def config() -> Config:
    return Config()
