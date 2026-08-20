"""Significance machinery, cross-checked against closed-form results."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from earnings_engine.returns.stats import (
    bmp_test,
    cluster_bootstrap_ci,
    cross_sectional_ttest,
    newey_west_tstat,
    summarise_windows,
)


def test_cross_sectional_ttest_matches_scipy():
    rng = np.random.default_rng(3)
    x = rng.normal(0.002, 0.05, 500)
    got = cross_sectional_ttest(x)
    want = sps.ttest_1samp(x, 0.0)
    assert got.t_stat == pytest.approx(want.statistic)
    assert got.p_value == pytest.approx(want.pvalue)


def test_newey_west_equals_ols_se_at_zero_lags():
    rng = np.random.default_rng(4)
    x = rng.normal(0.01, 0.2, 400)
    t_nw, se = newey_west_tstat(x, lags=0)
    assert se == pytest.approx(x.std(ddof=0) / np.sqrt(len(x)), rel=1e-9)
    assert t_nw == pytest.approx(x.mean() / se)


def test_newey_west_widens_se_under_positive_autocorrelation():
    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, 3000)
    ar1 = np.zeros(3000)
    for i in range(1, 3000):
        ar1[i] = 0.7 * ar1[i - 1] + e[i]
    _, se_iid = newey_west_tstat(ar1, lags=0)
    _, se_hac = newey_west_tstat(ar1, lags=20)
    assert se_hac > 1.5 * se_iid


def test_cluster_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(6)
    clusters = np.repeat(np.arange(60), 20)
    shock = rng.normal(0, 0.05, 60)  # a common shock inside each cluster
    values = shock[clusters] + rng.normal(0.01, 0.02, clusters.size)
    res = cluster_bootstrap_ci(values, clusters, n_boot=800, seed=1)
    assert res.ci_low < values.mean() < res.ci_high


def test_cluster_bootstrap_se_exceeds_the_naive_se():
    """With correlation inside clusters, treating events as independent
    understates the standard error. That is the whole point of clustering."""
    rng = np.random.default_rng(7)
    clusters = np.repeat(np.arange(40), 25)
    shock = rng.normal(0, 0.08, 40)
    values = shock[clusters] + rng.normal(0.0, 0.01, clusters.size)
    naive = cross_sectional_ttest(values)
    clustered = cluster_bootstrap_ci(values, clusters, n_boot=800, seed=2)
    assert clustered.std_error > naive.std_error * 1.5


def test_bmp_runs_and_agrees_in_sign(study):
    naive = cross_sectional_ttest(study.summary["car_market_model_0_4"])
    bmp = bmp_test(study.summary, "market_model", (0, 4))
    assert bmp.n > 100
    assert np.sign(bmp.t_stat) == np.sign(naive.t_stat)


def test_summary_table_covers_every_estimator_and_window(study):
    table = summarise_windows(
        study.summary, windows=((0, 0), (0, 4), (1, 20)), n_boot=200
    )
    assert set(table["estimator"]) == {"market_adjusted", "market_model", "sector_neutral"}
    assert set(table["method"].str.split("[").str[0]) == {
        "cross_sectional_t",
        "bmp",
        "cluster_bootstrap",
    }
    assert table["n"].min() > 0


def test_cluster_bootstrap_is_fast_enough_to_use_at_scale():
    """The obvious implementation rebuilt an array per draw and was unusable.

    Not a micro-benchmark for its own sake: at 13,000 events and 2,000 draws the
    naive version took longer than the rest of the study combined, which is the
    difference between a test that gets run and one that gets skipped.
    """
    import time

    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(2700), 5)
    values = rng.normal(size=clusters.size)
    started = time.perf_counter()
    result = cluster_bootstrap_ci(values, clusters, n_boot=2000, seed=1)
    assert time.perf_counter() - started < 10.0
    assert result.n == clusters.size
    assert result.ci_low < result.mean < result.ci_high
