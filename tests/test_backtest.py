"""Portfolio construction, costs and the backtest loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_engine.backtest.costs import CostModel, apply_costs, cost_sensitivity
from earnings_engine.backtest.engine import build_daily_book, performance_stats, run_backtest
from earnings_engine.backtest.portfolio import build_positions


@pytest.fixture()
def scored(study):
    rng = np.random.default_rng(21)
    df = study.summary[["event_id", "ticker", "t0"]].copy()
    df["sector"] = np.resize(["Tech", "Health", "Energy", "Financials"], len(df))
    df["prediction"] = rng.normal(size=len(df))
    return df.sort_values("t0").reset_index(drop=True)


def test_breakpoints_use_only_past_events(scored):
    book = build_positions(scored, min_lookback_events=50, lookback_days=252)
    assert not book.empty
    # The earliest events cannot be scored: there is no history to rank against.
    assert book["t0"].min() > scored["t0"].min()


def test_long_and_short_legs_are_both_populated(scored):
    book = build_positions(scored, quantiles=5, min_lookback_events=50)
    assert (book["side"] > 0).sum() > 0
    assert (book["side"] < 0).sum() > 0


def test_entry_offset_excludes_the_announcement_day(scored, study):
    book = build_positions(scored, min_lookback_events=50)
    daily = build_daily_book(
        book,
        study.daily,
        ar_column="ar_market_model",
        entry_offset=1,
        holding_days=20,
        sector_neutral=True,
        gross_exposure=1.0,
        max_weight=0.05,
    )
    assert daily["rel_day"].min() == 1, "the untradable announcement gap leaked into the book"
    assert daily["rel_day"].max() == 20


def test_book_is_sector_neutral_each_day(scored, study):
    book = build_positions(scored, min_lookback_events=50)
    daily = build_daily_book(
        book,
        study.daily,
        ar_column="ar_market_model",
        entry_offset=1,
        holding_days=20,
        sector_neutral=True,
        gross_exposure=1.0,
        max_weight=1.0,
    )
    net_by_sector = daily.groupby(["date", "sector"], observed=True)["weight"].sum().abs()
    assert net_by_sector.max() < 1e-9


def test_gross_exposure_is_respected(scored, study):
    book = build_positions(scored, min_lookback_events=50)
    daily = build_daily_book(
        book,
        study.daily,
        ar_column="ar_market_model",
        entry_offset=1,
        holding_days=20,
        sector_neutral=True,
        gross_exposure=1.0,
        max_weight=1.0,
    )
    gross = daily.groupby("date")["weight"].apply(lambda s: s.abs().sum())
    assert gross.max() == pytest.approx(1.0, abs=1e-9)


def test_costs_reduce_returns_monotonically():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    gross = pd.Series(0.001, index=idx)
    traded = pd.Series(0.1, index=idx)
    cheap = apply_costs(gross, traded, CostModel(half_spread_bps=1))["net"].sum()
    dear = apply_costs(gross, traded, CostModel(half_spread_bps=20))["net"].sum()
    assert cheap > dear


def test_cost_sensitivity_is_decreasing():
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(31)
    gross = pd.Series(rng.normal(0.0004, 0.004, len(idx)), index=idx)
    traded = pd.Series(0.08, index=idx)
    sens = cost_sensitivity(gross, traded, CostModel())
    assert sens["sharpe"].is_monotonic_decreasing


def test_impact_is_concave_in_participation():
    m = CostModel(impact_coef_bps=10.0, half_spread_bps=0.0, commission_bps=0.0)
    assert m.one_way_bps(0.04) == pytest.approx(2.0)
    assert m.one_way_bps(0.16) == pytest.approx(4.0)  # 4x size -> 2x cost


def test_performance_stats_on_a_known_series():
    idx = pd.date_range("2020-01-01", periods=252 * 4, freq="B")
    net = pd.Series(np.full(len(idx), 0.0004), index=idx)
    stats = performance_stats(net)
    assert stats["ann_return_net"] == pytest.approx(0.1008)
    assert stats["max_drawdown"] == pytest.approx(0.0)
    assert stats["hit_rate_daily"] == 1.0


def test_backtest_runs_end_to_end(scored, study):
    book = build_positions(scored, min_lookback_events=50)
    result = run_backtest(book, study.daily, ar_column="ar_market_model", max_weight=0.05)
    assert len(result.daily) > 100
    assert {"gross", "net", "cost", "traded"} <= set(result.daily.columns)
    assert (result.daily["net"] <= result.daily["gross"] + 1e-12).all()
    assert result.stats["ann_turnover"] > 0


def test_random_signal_earns_nothing(scored, study):
    """A signal drawn from noise must not produce a significant net return."""
    book = build_positions(scored, min_lookback_events=50)
    result = run_backtest(book, study.daily, ar_column="ar_market_model", max_weight=0.05)
    assert abs(result.stats["tstat_nw"]) < 3.0
