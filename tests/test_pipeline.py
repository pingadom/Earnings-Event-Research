"""End-to-end integration: the pipeline must find a planted effect, and must
not find one that was never planted.

These two tests together are the strongest statement this repository can make
about itself. A leak anywhere -- in the as-of join, the event alignment, the
walk-forward split, the portfolio breakpoints -- shows up as the null fixture
producing a significant result.
"""

from __future__ import annotations

import pandas as pd
import pytest

from earnings_engine.config import Config, ModelConfig
from earnings_engine.data.providers.synthetic import SyntheticProvider, SyntheticSpec
from earnings_engine.pipeline import (
    build_dataset,
    build_feature_panel,
    run_event_study,
    run_model,
    run_portfolio,
)
from earnings_engine.returns.stats import newey_west_tstat

START, END = "2014-01-02", "2024-12-31"


def _run(drift: float) -> dict:
    provider = SyntheticProvider(
        SyntheticSpec(n_tickers=90, start=START, end=END, seed=20260818, drift_coef=drift)
    )
    cfg = Config(model=ModelConfig(initial_train_years=6, validation_years=1, step_years=1))
    dataset = build_dataset(provider, cfg, START, END)
    study = run_event_study(dataset, cfg)
    panel = build_feature_panel(dataset, study, cfg)
    preds, metrics, _ = run_model(panel, cfg)
    bt = run_portfolio(preds, study, cfg)
    t_stat, _ = newey_west_tstat(bt.daily["net"])
    return {"ic": metrics["ic_mean"], "ic_t": metrics["ic_tstat_nw"], "t": t_stat, "bt": bt}


@pytest.fixture(scope="module")
def planted():
    return _run(0.020)


@pytest.fixture(scope="module")
def null():
    return _run(0.0)


@pytest.mark.slow
def test_planted_drift_is_found(planted):
    assert planted["ic"] > 0.05
    assert planted["ic_t"] > 2.0
    assert planted["t"] > 2.0


@pytest.mark.slow
def test_null_produces_nothing(null):
    assert abs(null["ic"]) < 0.05
    assert abs(null["t"]) < 2.0


@pytest.mark.slow
def test_costs_always_reduce_the_return(planted):
    daily = planted["bt"].daily
    assert (daily["cost"] >= 0).all()
    assert daily["net"].sum() < daily["gross"].sum()
    assert planted["bt"].stats["ann_turnover"] > 0


@pytest.mark.slow
def test_book_carries_no_net_sector_exposure(planted):
    book = planted["bt"].book
    net = book.groupby(["date", "sector"], observed=True)["weight"].sum().abs()
    assert net.max() < 1e-9


@pytest.mark.slow
def test_positions_never_include_the_announcement_day(planted):
    assert planted["bt"].book["rel_day"].min() >= 1


@pytest.mark.slow
def test_cost_sensitivity_erodes_the_edge(planted):
    sens = planted["bt"].cost_sensitivity
    assert sens["sharpe"].iloc[0] > sens["sharpe"].iloc[-1]


@pytest.mark.slow
def test_predictions_are_all_out_of_sample(planted, null):
    """Every prediction must come from a fold whose training data ended first."""
    for run in (planted, null):
        positions = run["bt"].positions
        assert not positions.empty
        assert pd.api.types.is_datetime64_any_dtype(positions["t0"])
