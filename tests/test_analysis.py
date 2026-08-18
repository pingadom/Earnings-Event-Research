"""Post-hoc analysis: attribution, multiple testing, Fama-MacBeth.

Each estimator is checked against a case with a known answer -- a series built
to have a specific alpha, a Sharpe distribution with a known maximum, a panel
with a planted cross-sectional relationship -- rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from earnings_engine.analysis import (
    TrialsLog,
    attribute_returns,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    fama_macbeth,
    min_track_record_length,
    normalise_factors,
    probabilistic_sharpe_ratio,
)

DAYS = 252


@pytest.fixture()
def factors() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2015-01-01", periods=DAYS * 6)
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.010, len(idx)),
            "SMB": rng.normal(0.0000, 0.004, len(idx)),
            "HML": rng.normal(0.0000, 0.004, len(idx)),
            "RMW": rng.normal(0.0000, 0.003, len(idx)),
            "CMA": rng.normal(0.0000, 0.003, len(idx)),
            "MOM": rng.normal(0.0001, 0.006, len(idx)),
        },
        index=idx,
    )


# --- attribution ------------------------------------------------------------


def test_alpha_and_loadings_are_recovered(factors):
    """Build a series with a known alpha and known betas; get them back."""
    rng = np.random.default_rng(12)
    true_alpha_daily = 0.05 / DAYS
    strategy = (
        true_alpha_daily
        + 0.30 * factors["Mkt-RF"]
        + 0.50 * factors["MOM"]
        + rng.normal(0, 0.003, len(factors))
    )
    res = attribute_returns(strategy, factors)
    assert res.alpha_annual == pytest.approx(0.05, abs=0.012)
    assert res.loadings["Mkt-RF"] == pytest.approx(0.30, abs=0.05)
    assert res.loadings["MOM"] == pytest.approx(0.50, abs=0.08)
    assert res.loadings["HML"] == pytest.approx(0.0, abs=0.12)
    assert res.alpha_tstat > 2


def test_pure_factor_exposure_shows_no_alpha(factors):
    """A strategy that is 100% momentum must not be credited with alpha."""
    rng = np.random.default_rng(13)
    strategy = 1.0 * factors["MOM"] + rng.normal(0, 0.001, len(factors))
    res = attribute_returns(strategy, factors)
    assert abs(res.alpha_tstat) < 2.5
    assert res.loadings["MOM"] == pytest.approx(1.0, abs=0.05)
    assert res.r_squared > 0.9
    assert "No significant alpha" in res.verdict() or "factor exposure" in res.verdict()


def test_hac_standard_errors_exceed_ols_under_autocorrelation(factors):
    """Overlapping holdings inflate the OLS t-stat; HAC must pull it back."""
    rng = np.random.default_rng(14)
    shock = rng.normal(0, 0.003, len(factors))
    smoothed = pd.Series(shock, index=factors.index).rolling(20, min_periods=1).mean()
    strategy = 0.0002 + smoothed
    hac = attribute_returns(strategy, factors, hac_lags=25)
    naive = attribute_returns(strategy, factors, hac_lags=0)
    assert abs(hac.alpha_tstat) < abs(naive.alpha_tstat)


def test_appraisal_ratio_is_alpha_over_residual_vol(factors):
    rng = np.random.default_rng(15)
    strategy = 0.0003 + 0.2 * factors["Mkt-RF"] + rng.normal(0, 0.004, len(factors))
    res = attribute_returns(strategy, factors)
    assert res.appraisal_ratio == pytest.approx(
        res.alpha_annual / res.residual_vol_annual, rel=1e-9
    )


def test_percent_scaled_factors_are_rejected(factors):
    with pytest.raises(ValueError, match="percent"):
        normalise_factors(factors * 100.0)


def test_momentum_alias_is_normalised(factors):
    renamed = factors.rename(columns={"MOM": "Mom"}).reset_index(names="date")
    assert "MOM" in normalise_factors(renamed).columns


def test_too_little_overlap_is_an_error(factors):
    short = pd.Series(0.001, index=factors.index[:20])
    with pytest.raises(ValueError, match="overlapping"):
        attribute_returns(short, factors)


# --- multiple testing -------------------------------------------------------


def test_expected_max_sharpe_grows_with_trials():
    v = 0.5**2
    assert expected_max_sharpe(1, v) == 0.0
    seq = [expected_max_sharpe(n, v) for n in (2, 5, 20, 100, 1000)]
    assert seq == sorted(seq)


def test_expected_max_sharpe_scales_with_dispersion():
    assert expected_max_sharpe(50, 4.0) == pytest.approx(2 * expected_max_sharpe(50, 1.0))


def _series_with_exact_sharpe(n: int, sharpe: float, seed: int) -> np.ndarray:
    """A sample whose *realised* per-period Sharpe is exactly ``sharpe``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x = (x - x.mean()) / x.std(ddof=1)
    return x * 0.01 + sharpe * 0.01


def test_psr_rises_with_track_record_length():
    """Same realised Sharpe, more observations -> more confidence.

    The realised Sharpe has to be held fixed for this to test anything: two
    independent draws differ in Sharpe as well as in length, and the shorter one
    can easily score higher.
    """
    short = _series_with_exact_sharpe(120, 0.05, 16)
    long_run = _series_with_exact_sharpe(3000, 0.05, 26)
    assert probabilistic_sharpe_ratio(short)["sharpe"] == pytest.approx(
        probabilistic_sharpe_ratio(long_run)["sharpe"], abs=1e-12
    )
    assert probabilistic_sharpe_ratio(long_run)["psr"] > probabilistic_sharpe_ratio(short)["psr"]


def test_psr_penalises_negative_skew():
    """Two series, same Sharpe, different skew: the skewed one must score worse.

    This is the whole reason PSR exists -- a strategy that sells optionality
    looks fine until it does not, and a short record is exactly where that hides.
    """
    rng = np.random.default_rng(17)
    n = 2000
    symmetric = rng.normal(0, 1, n)
    skewed = -sps.skewnorm.rvs(a=8, size=n, random_state=18)
    skewed = (skewed - skewed.mean()) / skewed.std(ddof=1)
    symmetric = (symmetric - symmetric.mean()) / symmetric.std(ddof=1)
    target_sr = 0.05
    a = pd.Series(symmetric * 0.01 + target_sr * 0.01)
    b = pd.Series(skewed * 0.01 + target_sr * 0.01)
    assert probabilistic_sharpe_ratio(b)["psr"] < probabilistic_sharpe_ratio(a)["psr"]


def test_deflation_reduces_confidence():
    rng = np.random.default_rng(19)
    r = rng.normal(0.0006, 0.008, DAYS * 5)
    undeflated = probabilistic_sharpe_ratio(r)["psr"]
    deflated = deflated_sharpe_ratio(r, n_trials=100, variance_of_trials=(0.8 / np.sqrt(DAYS)) ** 2)
    assert deflated["dsr"] < undeflated
    assert deflated["expected_max_sharpe"] > 0


def test_deflation_requires_a_stated_dispersion():
    rng = np.random.default_rng(20)
    with pytest.raises(ValueError, match="trial_sharpes or an explicit"):
        deflated_sharpe_ratio(rng.normal(0, 0.01, 500), n_trials=10)


def test_min_track_record_length_is_infinite_below_the_benchmark():
    rng = np.random.default_rng(21)
    losing = rng.normal(-0.001, 0.01, 500)
    assert min_track_record_length(losing) == np.inf


def test_trials_log_round_trips(tmp_path):
    log = TrialsLog(path=tmp_path / "t.json")
    log.record("spec A", sharpe_annual=0.8, notes="baseline")
    log.record("spec B", sharpe_annual=1.4, params={"model": "ridge"})
    path = log.save()
    again = TrialsLog.load(path)
    assert again.n == 2
    assert list(again.sharpes()) == [0.8, 1.4]


def test_trials_log_refuses_to_deflate_on_thin_evidence(tmp_path):
    """Two recorded Sharpes cannot estimate a dispersion; say so, don't guess."""
    rng = np.random.default_rng(22)
    r = pd.Series(rng.normal(0.0005, 0.008, 1000))
    log = TrialsLog(path=tmp_path / "t.json")
    log.record("a", sharpe_annual=1.0)
    log.record("b", sharpe_annual=1.1)
    for i in range(6):
        log.record(f"abandoned {i}")
    out = log.deflate(r)
    assert out["not_deflated"] is True
    assert "NOT DEFLATED" in out["verdict"]
    assert out["n_trials"] == 8


def test_dispersion_sensitivity_is_monotone(tmp_path):
    rng = np.random.default_rng(23)
    r = pd.Series(rng.normal(0.0004, 0.008, 1500))
    log = TrialsLog(path=tmp_path / "t.json")
    for i in range(12):
        log.record(f"spec {i}")
    sens = log.deflate_sensitivity(r)
    assert sens["hurdle_sharpe_annual"].is_monotonic_increasing
    assert sens["deflated_sharpe"].is_monotonic_decreasing


# --- Fama-MacBeth -----------------------------------------------------------


@pytest.fixture()
def cross_section() -> pd.DataFrame:
    """A panel where `signal` earns 40bp per sd and `junk` earns nothing."""
    rng = np.random.default_rng(24)
    rows = []
    for period in pd.date_range("2015-01-31", periods=60, freq="ME"):
        n = 90
        signal = rng.normal(size=n)
        junk = rng.normal(size=n)
        rows.append(
            pd.DataFrame(
                {
                    "t0": period,
                    "signal": signal,
                    "junk": junk,
                    "y": 0.0040 * signal + rng.normal(0, 0.05, n),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_fama_macbeth_recovers_a_planted_coefficient(cross_section):
    res = fama_macbeth(cross_section, ["signal", "junk"], "y", min_cross_section=30)
    row = res.summary.set_index("term")
    assert res.n_periods == 60
    assert row.loc["signal", "coefficient"] == pytest.approx(0.0040, abs=0.001)
    assert row.loc["signal", "t_stat"] > 3
    assert abs(row.loc["junk", "t_stat"]) < 2.5
    assert "signal" in res.verdict()


def test_fama_macbeth_drops_constant_features(cross_section):
    """A constant column makes every period rank-deficient; it must be dropped,
    not allowed to silently kill the whole regression."""
    df = cross_section.assign(dead=1.0, empty=np.nan)
    res = fama_macbeth(df, ["signal", "junk", "dead", "empty"], "y", min_cross_section=30)
    assert "dead" not in res.summary["term"].tolist()
    assert "empty" not in res.summary["term"].tolist()
    assert res.n_periods == 60


def test_fama_macbeth_refuses_too_many_regressors(cross_section):
    rng = np.random.default_rng(25)
    names = [f"f{i}" for i in range(120)]
    extra = pd.DataFrame(
        rng.normal(size=(len(cross_section), len(names))), columns=names,
        index=cross_section.index,
    )
    df = pd.concat([cross_section, extra], axis=1)
    with pytest.raises(ValueError, match="more names than regressors"):
        fama_macbeth(df, names, "y")


def test_fama_macbeth_coefficients_are_a_time_series(cross_section):
    res = fama_macbeth(cross_section, ["signal", "junk"], "y", min_cross_section=30)
    assert list(res.coefficients.columns) == ["intercept", "signal", "junk"]
    assert len(res.coefficients) == res.n_periods
    assert res.coefficients.index.is_monotonic_increasing
