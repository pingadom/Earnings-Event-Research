"""Abnormal return estimation, checked against a known data-generating process."""

from __future__ import annotations

import numpy as np
import pytest

from earnings_engine.config import EventsConfig, ReturnsConfig
from earnings_engine.events import align_events
from earnings_engine.returns import ReturnPanel, compute_abnormal_returns
from earnings_engine.returns.estimation import fit_market_model


def test_market_model_recovers_known_beta():
    rng = np.random.default_rng(0)
    n_t = 400
    market = rng.normal(0, 0.01, n_t)
    true_alpha, true_beta = 0.0003, 1.4
    asset = true_alpha + true_beta * market + rng.normal(0, 0.004, n_t)
    returns = asset.reshape(-1, 1)

    fit = fit_market_model(
        returns, market, np.array([350]), np.array([0]), start_offset=-250, end_offset=-31
    )
    assert fit.beta[0] == pytest.approx(true_beta, abs=0.05)
    assert fit.alpha[0] == pytest.approx(true_alpha, abs=0.0005)
    assert fit.sigma[0] == pytest.approx(0.004, rel=0.15)
    assert fit.n_obs[0] == 220


def test_market_model_marks_short_windows_invalid():
    rng = np.random.default_rng(1)
    market = rng.normal(0, 0.01, 100)
    returns = rng.normal(0, 0.01, (100, 1))
    fit = fit_market_model(returns, market, np.array([60]), np.array([0]), min_obs=100)
    assert np.isnan(fit.beta[0])


def test_estimation_window_excludes_the_event(study):
    """The default window ends 31 sessions before t0, so no event-window return
    can influence the parameters that define 'normal'."""
    cfg = ReturnsConfig()
    assert cfg.estimation_end <= -1
    assert cfg.estimation_end < min(lo for lo, _ in cfg.windows)


def test_sector_benchmark_excludes_the_stock_itself(prices, sector_map):
    panel = ReturnPanel.from_prices(prices, "SPY", sector_map)
    assert panel.sector_benchmark is not None
    i = panel.ticker_pos[panel.tickers[0]]
    own = panel.returns[:, i]
    bench = panel.sector_benchmark[:, i]
    ok = np.isfinite(own) & np.isfinite(bench)
    # If the stock were included in its own benchmark the correlation would be
    # mechanically inflated toward 1 for a small sector.
    assert np.corrcoef(own[ok], bench[ok])[0, 1] < 0.9


def test_car_matches_a_hand_computed_sum(study, panel):
    row = study.summary.dropna(subset=["car_market_adjusted_0_4"]).iloc[0]
    daily = study.daily[study.daily["event_id"] == row["event_id"]]
    manual = daily[(daily["rel_day"] >= 0) & (daily["rel_day"] <= 4)]["ar_market_adjusted"].sum()
    assert row["car_market_adjusted_0_4"] == pytest.approx(manual, abs=1e-12)


def test_bhar_compounds_rather_than_sums(study):
    row = study.summary.dropna(subset=["car_market_model_0_19", "bhar_market_model_0_19"]).iloc[0]
    assert row["bhar_market_model_0_19"] != pytest.approx(row["car_market_model_0_19"], abs=1e-9)


def test_planted_effect_is_recovered(provider, study):
    """CAR should load on the latent surprise with roughly the planted slope."""
    truth = provider.ground_truth().set_index("event_id")["z"]
    df = study.summary.assign(z=study.summary["event_id"].map(truth)).dropna(
        subset=["car_market_model_0_0", "z"]
    )
    slope = np.polyfit(df["z"], df["car_market_model_0_0"], 1)[0]
    assert slope == pytest.approx(provider.spec.jump_coef, rel=0.25)
    assert df[["car_market_model_0_0", "z"]].corr().iloc[0, 1] > 0.6


def test_no_effect_when_none_was_planted(null_provider):
    """The null fixture must produce a mean CAR indistinguishable from zero."""
    from earnings_engine.returns.stats import cross_sectional_ttest

    tickers = list(null_provider.get_universe()["ticker"])
    px = null_provider.get_prices([*tickers, "SPY"], "2016-01-04", "2022-12-30")
    ev = null_provider.get_events(tickers, "2016-01-04", "2022-12-30")
    aligned = align_events(ev, px, EventsConfig())
    sectors = dict(
        zip(null_provider.get_universe()["ticker"], null_provider.get_universe()["sector"], strict=False)
    )
    res = compute_abnormal_returns(aligned, ReturnPanel.from_prices(px, "SPY", sectors))
    test = cross_sectional_ttest(res.summary["car_market_model_0_19"])
    assert abs(test.t_stat) < 3.0, f"spurious effect under the null: t={test.t_stat:.2f}"


def test_windows_are_complete_or_missing(study):
    """A CAR is either computed over the full window or is NaN -- never partial."""
    col, nobs = "car_market_model_0_19", "nobs_market_model_0_19"
    have = study.summary[col].notna()
    assert (study.summary.loc[have, nobs] == 20).all()
