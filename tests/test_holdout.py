"""Rolling annual holdouts.

Two layers. The fast tests fabricate a panel with known dates so the leakage
guarantees can be checked exactly. The slow tests run the whole pipeline on the
planted-effect and null fixtures and assert the pair separates -- which is the
claim `docs/results.md` rests on.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from earnings_engine.config import Config, ModelConfig
from earnings_engine.data.providers.synthetic import SyntheticProvider, SyntheticSpec
from earnings_engine.holdout import _calibration, run_annual_holdouts
from earnings_engine.pipeline import build_dataset, build_feature_panel, run_event_study
from earnings_engine.reporting.dashboard import build_payload, write_dashboard
from earnings_engine.returns.abnormal import AbnormalReturnResult

TARGET = "car_market_model_1_20"


class _EmptyStudy(AbnormalReturnResult):
    """A study with no daily returns, so the per-year backtests bow out cleanly."""

    def __init__(self):
        empty = pd.DataFrame(columns=["event_id", "rel_day", "t0", "ar_market_model"])
        super().__init__(daily=empty, summary=empty, fit=empty, windows=((1, 20),),
                         estimators=("market_model",))


@pytest.fixture()
def fake_panel() -> pd.DataFrame:
    """A panel with a known linear relationship and one event per business day."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2012-01-02", "2024-12-31", freq="B")
    n = len(dates)
    x = rng.normal(size=n)
    noise = rng.normal(scale=0.02, size=n)
    return pd.DataFrame(
        {
            "event_id": [f"E{i:05d}" for i in range(n)],
            "ticker": np.resize([f"T{i:02d}" for i in range(40)], n),
            "sector": np.resize(["Tech", "Health", "Energy", "Financials"], n),
            "t0": dates,
            "signal": x,
            "sue_timeseries": x * 0.4 + rng.normal(scale=0.9, size=n),
            "junk": rng.normal(size=n),
            TARGET: 0.01 * x + noise,
        }
    )


@pytest.fixture()
def cfg() -> Config:
    return Config(model=ModelConfig(target=TARGET, embargo_days=25))


# --- leakage guarantees ----------------------------------------------------


def test_training_data_always_predates_the_holdout_year(fake_panel, cfg):
    res = run_annual_holdouts(fake_panel, _EmptyStudy(), cfg, range(2019, 2025), min_train=100)
    for row in res.by_year.itertuples():
        assert pd.Timestamp(row.train_end) < pd.Timestamp(f"{row.year}-01-01")


def test_embargo_gap_is_respected(fake_panel, cfg):
    res = run_annual_holdouts(fake_panel, _EmptyStudy(), cfg, range(2019, 2025), min_train=100)
    for row in res.by_year.itertuples():
        gap = (pd.Timestamp(f"{row.year}-01-01") - pd.Timestamp(row.train_end)).days
        assert gap >= cfg.model.embargo_days


def test_overlapping_labels_are_purged(fake_panel, cfg):
    res = run_annual_holdouts(fake_panel, _EmptyStudy(), cfg, range(2019, 2025), min_train=100)
    assert (res.by_year["n_purged"] > 0).all()


def test_predictions_cover_each_holdout_year_exactly_once(fake_panel, cfg):
    res = run_annual_holdouts(fake_panel, _EmptyStudy(), cfg, range(2019, 2025), min_train=100)
    assert res.predictions["event_id"].is_unique
    for year, grp in res.predictions.groupby("holdout_year"):
        assert grp["t0"].dt.year.eq(year).all()


def test_training_set_grows_each_year(fake_panel, cfg):
    res = run_annual_holdouts(fake_panel, _EmptyStudy(), cfg, range(2019, 2025), min_train=100)
    assert res.by_year["n_train"].is_monotonic_increasing


def test_a_year_without_enough_history_is_skipped(fake_panel, cfg):
    res = run_annual_holdouts(fake_panel, _EmptyStudy(), cfg, [2012, 2013, 2024], min_train=1000)
    assert list(res.by_year["year"]) == [2024]


# --- the metrics themselves ------------------------------------------------


def test_calibration_recovers_a_known_slope():
    rng = np.random.default_rng(5)
    pred = rng.normal(size=4000)
    real = 0.5 * pred + rng.normal(scale=0.4, size=4000)
    slope, _, r2 = _calibration(pred, real)
    assert slope == pytest.approx(0.5, abs=0.03)
    assert 0 < r2 < 1


def test_calibration_is_flat_when_predictions_are_noise():
    rng = np.random.default_rng(6)
    slope, _, _ = _calibration(rng.normal(size=3000), rng.normal(size=3000))
    assert abs(slope) < 0.1


def test_a_real_relationship_is_recovered(fake_panel, cfg):
    res = run_annual_holdouts(fake_panel, _EmptyStudy(), cfg, range(2019, 2025), min_train=100)
    assert res.aggregate["mean_ic"] > 0.2
    assert res.aggregate["positive_ic_years"] == res.aggregate["n_years"]
    assert res.by_year["calib_slope"].mean() > 0.5


def test_pure_noise_features_find_nothing(fake_panel, cfg):
    """Same machinery, same dates, target replaced by noise."""
    rng = np.random.default_rng(9)
    panel = fake_panel.copy()
    panel[TARGET] = rng.normal(scale=0.02, size=len(panel))
    res = run_annual_holdouts(panel, _EmptyStudy(), cfg, range(2019, 2025), min_train=100)
    assert abs(res.aggregate["mean_ic"]) < 0.06
    assert abs(res.aggregate["ic_tstat_across_years"]) < 3.0


def test_baseline_is_scored_alongside_the_model(fake_panel, cfg):
    res = run_annual_holdouts(fake_panel, _EmptyStudy(), cfg, range(2019, 2025), min_train=100)
    assert res.baseline_feature == "sue_timeseries"
    assert res.by_year["baseline_ic"].notna().all()
    # The model sees the clean feature; the baseline only sees a noisy proxy.
    assert res.aggregate["mean_ic"] > res.aggregate["mean_baseline_ic"]


# --- dashboard -------------------------------------------------------------


def test_dashboard_is_self_contained_and_parseable(fake_panel, cfg, tmp_path):
    res = run_annual_holdouts(fake_panel, _EmptyStudy(), cfg, range(2019, 2025), min_train=100)
    path = write_dashboard(tmp_path / "d.html", res, {"label": "test", "model": "ridge"})
    html = path.read_text()

    assert "__PAYLOAD__" not in html
    # No network dependencies: the file must still work from disk in ten years.
    # The only permitted absolute URL is the SVG namespace, which is an
    # identifier rather than something the browser fetches.
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", html))
    assert urls <= {"http://www.w3.org/2000/svg"}, f"external references: {urls}"
    for tag in ("<script src=", "<link ", "@import", "url(http"):
        assert tag not in html

    payload = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    assert len(payload["byYear"]) == len(res.by_year)
    assert payload["calibration"]
    # NaN is not valid JSON; the payload must already be clean.
    assert "NaN" not in html.split("const DATA = ")[1].split(";\n")[0]


def test_payload_has_no_nan(fake_panel, cfg):
    res = run_annual_holdouts(fake_panel, _EmptyStudy(), cfg, range(2019, 2025), min_train=100)
    blob = json.dumps(build_payload(res, {"label": "t"}))
    assert "NaN" not in blob and "Infinity" not in blob


# --- end to end ------------------------------------------------------------


def _pipeline_holdout(drift: float):
    start, end = "2014-01-02", "2023-12-31"
    provider = SyntheticProvider(
        SyntheticSpec(n_tickers=80, start=start, end=end, seed=20260818, drift_coef=drift)
    )
    config = Config()
    dataset = build_dataset(provider, config, start, end)
    study = run_event_study(dataset, config)
    panel = build_feature_panel(dataset, study, config)
    return run_annual_holdouts(panel, study, config, range(2021, 2024))


@pytest.fixture(scope="module")
def planted_holdout():
    return _pipeline_holdout(0.020)


@pytest.fixture(scope="module")
def null_holdout():
    return _pipeline_holdout(0.0)


@pytest.mark.slow
def test_planted_effect_is_found_in_every_year(planted_holdout):
    agg = planted_holdout.aggregate
    assert agg["mean_ic"] > 0.08
    assert agg["positive_ic_years"] == agg["n_years"]
    assert agg["stitched_tstat_nw"] > 2.0


@pytest.mark.slow
def test_null_control_finds_no_ranking_skill(null_holdout):
    """The scientific claim: with nothing planted, nothing is ranked.

    Skill is the thing the null control exists to test, so skill is what is
    asserted. The realised profit and loss is a noisier quantity and is checked
    separately below, for reasons that are worth writing down.
    """
    agg = null_holdout.aggregate
    assert abs(agg["mean_ic"]) < 0.08
    assert abs(agg["ic_tstat_across_years"]) < 2.0


@pytest.mark.slow
def test_null_control_book_carries_a_known_negative_drift(null_holdout):
    """A documented artefact, pinned so it cannot quietly grow.

    The null book has no ranking skill, is exactly dollar-neutral, and still
    loses roughly three percent a year *before* costs. That is a property of
    how the book is constructed -- overlapping twenty-day holdings rebalanced
    daily into a per-name cap that binds on most days -- not of the signal.

    It matters for how the real study is read: some part of any negative Sharpe
    ratio reported there is mechanical rather than evidence against the
    hypothesis, which is why the information coefficient, not the Sharpe ratio,
    is treated as the primary evidence in docs/results.md.

    The bound is wide and one-sided on purpose. It is here to catch the drift
    getting worse, and to stop anyone reading a negative null Sharpe as a
    finding.
    """
    agg = null_holdout.aggregate
    assert -4.0 < agg["stitched_tstat_nw"] < 2.0, (
        "a null control that makes money is look-ahead; one that loses much more "
        "than it used to is a regression in the book construction"
    )


@pytest.mark.slow
def test_the_two_runs_separate(planted_holdout, null_holdout):
    """The claim docs/results.md makes, asserted."""
    assert planted_holdout.aggregate["mean_ic"] > 2 * abs(null_holdout.aggregate["mean_ic"])
    assert (
        planted_holdout.aggregate["stitched_sharpe_net"]
        > null_holdout.aggregate["stitched_sharpe_net"] + 1.0
    )


@pytest.mark.slow
def test_predictions_are_never_from_a_model_that_saw_them(planted_holdout):
    preds = planted_holdout.predictions
    for row in planted_holdout.by_year.itertuples():
        in_year = preds[preds["holdout_year"] == row.year]
        assert in_year["t0"].min() >= pd.Timestamp(f"{row.year}-01-01")
        assert pd.Timestamp(row.train_end) < in_year["t0"].min()
