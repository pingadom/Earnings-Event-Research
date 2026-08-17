"""Schemas, providers, universe handling and configuration."""

from __future__ import annotations

import pandas as pd
import pytest

from earnings_engine.config import ConfigError, load_config
from earnings_engine.data.universe import SurvivorshipBiasError, Universe
from earnings_engine.utils.frames import EVENTS, PRICES, SchemaError


def test_schema_rejects_missing_columns():
    with pytest.raises(SchemaError, match="missing column"):
        PRICES.validate(pd.DataFrame({"ticker": ["A"]}))


def test_schema_rejects_duplicate_keys(prices):
    with pytest.raises(SchemaError, match="duplicate row"):
        PRICES.validate(pd.concat([prices.head(5), prices.head(5)]))


def test_schema_rejects_nulls_in_required_columns(prices):
    broken = prices.head(20).copy()
    broken.loc[broken.index[0], "adj_close"] = None
    with pytest.raises(SchemaError, match="null value"):
        PRICES.validate(broken)


def test_events_are_timezone_aware(raw_events):
    validated = EVENTS.validate(raw_events)
    assert str(validated["announced_at_utc"].dt.tz) == "UTC"


def test_static_universe_requires_acknowledgement():
    with pytest.raises(SurvivorshipBiasError, match="survivorship"):
        Universe.static(["AAPL", "MSFT"])


def test_static_universe_is_flagged():
    u = Universe.static(["AAPL"], ["Tech"], acknowledge_bias=True)
    assert u.static_membership
    assert "STATIC" in repr(u)


def test_membership_is_asked_as_of_a_date():
    u = Universe.from_frame(
        pd.DataFrame(
            {
                "ticker": ["SURVIVOR", "DELETED"],
                "start_date": ["2010-01-01", "2010-01-01"],
                "end_date": ["2100-01-01", "2018-06-30"],
                "sector": ["Tech", "Financials"],
            }
        )
    )
    assert u.as_of("2015-01-01") == ["DELETED", "SURVIVOR"]
    assert u.as_of("2020-01-01") == ["SURVIVOR"]
    # The deleted name is still reachable for the period it was a member --
    # that is what makes the study survivorship-bias free.
    assert "DELETED" in u.all_tickers()


def test_universe_filter_drops_events_outside_membership(events):
    u = Universe.from_frame(
        pd.DataFrame(
            {
                "ticker": sorted(events["ticker"].unique()),
                "start_date": pd.Timestamp("2019-01-01"),
                "end_date": pd.Timestamp("2021-12-31"),
                "sector": "Tech",
            }
        )
    )
    kept = u.filter_events(events, "t0")
    assert len(kept) < len(events)
    assert kept["t0"].min() >= pd.Timestamp("2019-01-01")
    assert kept["t0"].max() <= pd.Timestamp("2021-12-31")


def test_liquidity_filter_uses_only_trailing_data(prices):
    u = Universe.static(["X"], acknowledge_bias=True)
    flags = u.liquidity_filter(prices, min_price=5.0, min_dollar_volume=1e6)
    first = flags.groupby("ticker").head(1)
    # No trailing window on day one, so nothing can be marked tradable.
    assert not first["tradable"].any()


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("retruns:\n  market_symbol: SPY\n")
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_config(path)


def test_config_rejects_unknown_nested_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("returns:\n  market_symbl: SPY\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_config_roundtrips_the_shipped_file():
    cfg = load_config()
    assert cfg.returns.market_symbol
    assert cfg.model.target.startswith("car_")
    # The default target must be a drift window, not one containing the gap.
    horizon_start = int(cfg.model.target.rsplit("_", 2)[1])
    assert horizon_start >= 1, "the default model target must exclude the announcement day"


def test_synthetic_provider_is_deterministic():
    from earnings_engine.data.providers.synthetic import SyntheticProvider, SyntheticSpec

    spec = SyntheticSpec(n_tickers=5, start="2018-01-02", end="2019-12-31", seed=99)
    a = SyntheticProvider(spec).get_prices(["SYN000"], "2018-01-02", "2019-12-31")
    b = SyntheticProvider(spec).get_prices(["SYN000"], "2018-01-02", "2019-12-31")
    pd.testing.assert_frame_equal(a, b)


def test_vendor_provider_reads_a_drop_folder(tmp_path):
    from earnings_engine.data.providers.vendor import CapitalIQProvider

    folder = tmp_path / "prices"
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ticker": ["AAPL", "AAPL"],
            "Date": ["2024-01-02", "2024-01-03"],
            "IQ_OPENPRICE": [100.0, 101.0],
            "IQ_HIGHPRICE": [102.0, 103.0],
            "IQ_LOWPRICE": [99.0, 100.0],
            "IQ_CLOSEPRICE": [101.0, 102.0],
            "IQ_CLOSEPRICE_ADJ": [101.0, 102.0],
            "IQ_VOLUME": [1e6, 1.1e6],
        }
    ).to_csv(folder / "px.csv", index=False)

    df = CapitalIQProvider(vendor_dir=tmp_path).load("prices")
    assert list(df["ticker"]) == ["AAPL", "AAPL"]
    assert df["adj_close"].iloc[1] == 102.0


def test_vendor_provider_flags_restated_fundamentals(tmp_path, caplog):
    from earnings_engine.data.providers.vendor import CapitalIQProvider

    folder = tmp_path / "fundamentals"
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "Ticker": ["AAPL"],
            "IQ_PERIODDATE": ["2024-03-31"],
            "Item": ["Revenue"],
            "Value": [90e9],
        }
    ).to_csv(folder / "f.csv", index=False)

    provider = CapitalIQProvider(vendor_dir=tmp_path)
    provider.available_from_policy = "filing_lag"
    with caplog.at_level("WARNING"):
        df = provider.load("fundamentals")
    assert "RESTATED" in caplog.text
    assert df["available_from_utc"].notna().all()
