"""Does the drift exist at all: sorted spreads with date-clustered inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_engine.analysis.pead import (
    drift_by_horizon,
    drift_by_subsample,
    sorted_drift,
)


def a_panel(effect: float, n_dates: int = 120, per_date: int = 12, seed: int = 0,
            noise: float = 0.05) -> pd.DataFrame:
    """Events clustered on dates, with a planted drift proportional to surprise."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2019-01-02", periods=n_dates, freq="B"), per_date)
    surprise = rng.normal(size=len(dates))
    car = effect * surprise + rng.normal(scale=noise, size=len(dates))
    return pd.DataFrame(
        {
            "t0": dates,
            "sue": surprise,
            "car_1_20": car,
            "car_1_5": 0.4 * car,
            "car_0_0": 0.05 * car,
            "adv20": rng.lognormal(mean=16, sigma=1.5, size=len(dates)),
        }
    )


def test_a_planted_drift_is_recovered_by_the_sort():
    table = sorted_drift(a_panel(0.02), "sue", "car_1_20", n_boot=400)
    spread = table.iloc[-1]
    assert spread["group"] == "Q5-Q1"
    # Q5 and Q1 mean surprises are about +/-1.4 sigma, so the spread is ~2.8 * effect.
    assert spread["mean_car"] == pytest.approx(0.02 * 2.8, rel=0.25)
    assert spread["t_stat"] > 3
    assert spread["p_value"] < 0.01


def test_no_drift_is_reported_as_no_drift():
    table = sorted_drift(a_panel(0.0), "sue", "car_1_20", n_boot=400)
    spread = table.iloc[-1]
    assert abs(spread["t_stat"]) < 2.5
    assert spread["p_value"] > 0.05
    assert spread["ci_low"] < 0 < spread["ci_high"]


def test_the_quantile_rows_show_where_the_effect_sits():
    table = sorted_drift(a_panel(0.02), "sue", "car_1_20", n_boot=200)
    quantiles = table.iloc[:-1]
    assert list(quantiles["group"]) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert quantiles["mean_sort_value"].is_monotonic_increasing
    assert quantiles["mean_car"].is_monotonic_increasing


def test_clustering_by_date_widens_the_interval():
    """The correction has to bite, or it is decoration.

    Every event on a date shares a common shock here, so treating them as
    independent would overstate precision several-fold.
    """
    rng = np.random.default_rng(7)
    dates = np.repeat(pd.date_range("2019-01-02", periods=100, freq="B"), 20)
    shock = np.repeat(rng.normal(scale=0.03, size=100), 20)
    panel = pd.DataFrame(
        {
            "t0": dates,
            "sue": rng.normal(size=len(dates)),
            "car_1_20": shock + rng.normal(scale=0.005, size=len(dates)),
        }
    )
    spread = sorted_drift(panel, "sue", "car_1_20", n_boot=400).iloc[-1]
    naive = panel["car_1_20"].std() / np.sqrt(len(panel))
    clustered = (spread["ci_high"] - spread["ci_low"]) / (2 * 1.96)
    assert clustered > naive


def test_the_term_structure_of_the_spread_is_reported():
    """Drift accumulates; a spread present on day one is an announcement reaction."""
    table = drift_by_horizon(
        a_panel(0.02),
        "sue",
        {"[0,0]": "car_0_0", "[1,5]": "car_1_5", "[1,20]": "car_1_20"},
        n_boot=200,
    )
    assert list(table["horizon"]) == ["[0,0]", "[1,5]", "[1,20]"]
    assert table["spread"].is_monotonic_increasing


def test_a_missing_horizon_is_skipped_not_invented():
    table = drift_by_horizon(
        a_panel(0.02), "sue", {"[1,20]": "car_1_20", "[1,60]": "car_1_60"}, n_boot=100
    )
    assert list(table["horizon"]) == ["[1,20]"]


def test_subsamples_are_tested_separately():
    table = drift_by_subsample(
        a_panel(0.02), "sue", "car_1_20", "adv20", n_groups=3,
        group_names=("least liquid", "middle", "most liquid"), n_boot=200,
    )
    assert list(table["subsample"]) == ["least liquid", "middle", "most liquid"]
    assert table["median_group_value"].is_monotonic_increasing
    assert (table["t_stat"] > 1.5).all()


def test_an_effect_confined_to_one_subsample_is_localised():
    """The whole point of splitting: find where the effect lives, if anywhere."""
    illiquid = a_panel(0.04, seed=1)
    illiquid["adv20"] = 1e5
    liquid = a_panel(0.0, seed=2)
    liquid["adv20"] = 1e10
    table = drift_by_subsample(
        pd.concat([illiquid, liquid], ignore_index=True),
        "sue", "car_1_20", "adv20", n_groups=2,
        group_names=("illiquid", "liquid"), n_boot=300,
    )
    by_name = table.set_index("subsample")
    assert by_name.loc["illiquid", "t_stat"] > 3
    assert abs(by_name.loc["liquid", "t_stat"]) < 2.5


def test_too_few_events_is_refused_rather_than_reported():
    with pytest.raises(ValueError, match="too few"):
        sorted_drift(a_panel(0.02, n_dates=5, per_date=2), "sue", "car_1_20", n_boot=50)


def test_a_missing_column_is_named():
    with pytest.raises(KeyError, match="sue"):
        sorted_drift(a_panel(0.0).drop(columns=["sue"]), "sue", "car_1_20", n_boot=50)


def test_a_bootstrap_p_value_is_never_exactly_zero():
    """No test with finitely many draws is entitled to claim zero."""
    spread = sorted_drift(a_panel(0.5, noise=0.001), "sue", "car_1_20", n_boot=200).iloc[-1]
    assert spread["t_stat"] > 10
    assert spread["p_value"] >= 2 / 201
