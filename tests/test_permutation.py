"""The shuffled-prediction null: does the book produce this from noise?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_engine.analysis.permutation import permutation_null


class FakeBacktest:
    """A book whose Sharpe is the correlation between prediction and outcome.

    Plus a constant negative drift, which is the property the real book has and
    the reason this module exists: a worthless signal does not score zero here.
    """

    DRIFT = -0.4

    def __init__(self, frame):
        ok = frame[["prediction", "outcome"]].dropna()
        corr = float(np.corrcoef(ok["prediction"], ok["outcome"])[0, 1]) if len(ok) > 2 else 0.0
        self.stats = {"sharpe_net": 3.0 * corr + self.DRIFT}


def a_panel(signal_strength: float, n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    outcome = rng.normal(size=n)
    prediction = signal_strength * outcome + rng.normal(size=n)
    return pd.DataFrame(
        {
            "prediction": prediction,
            "outcome": outcome,
            "holdout_year": np.repeat([2021, 2022, 2023, 2024], n // 4),
        }
    )


def test_a_worthless_signal_lands_in_the_middle_of_the_null():
    """Across independent panels, not one lucky draw.

    A single worthless panel can still land in the tail -- that is what a tail
    is. The claim being tested is distributional, so it is tested that way:
    most worthless panels must be unremarkable against their own null.
    """
    inside = [
        5 < permutation_null(a_panel(0.0, seed=s), FakeBacktest, n_permutations=120, seed=s).percentile < 95
        for s in range(7)
    ]
    assert sum(inside) >= 5, f"only {sum(inside)}/7 worthless panels were unremarkable"


def test_a_real_signal_beats_the_null():
    result = permutation_null(a_panel(1.2), FakeBacktest, n_permutations=120, seed=1)
    assert result.percentile > 95
    assert result.p_value_one_sided < 0.05


def test_the_null_recovers_the_book_s_own_drift_not_zero():
    """The whole point: 'worthless' is -0.4 here, not 0."""
    result = permutation_null(a_panel(0.0), FakeBacktest, n_permutations=200, seed=2)
    assert result.mean == pytest.approx(FakeBacktest.DRIFT, abs=0.1)
    assert abs(result.excess) < 0.35


def test_shuffling_happens_within_a_year_not_across_it():
    """Each year's set of predicted values must be preserved exactly."""
    panel = a_panel(0.0, n=200, seed=3)
    seen = []

    class Recorder:
        def __init__(self, frame):
            seen.append(frame.groupby("holdout_year")["prediction"].apply(
                lambda s: tuple(sorted(np.round(s, 9)))
            ))
            self.stats = {"sharpe_net": 0.0}

    permutation_null(panel, Recorder, n_permutations=4, seed=4)
    assert all(draw.equals(seen[0]) for draw in seen[1:])


def test_the_p_value_can_never_be_exactly_zero():
    """No permutation test with finitely many draws is entitled to claim zero."""
    result = permutation_null(a_panel(4.0), FakeBacktest, n_permutations=50, seed=5)
    assert result.p_value_one_sided >= 1 / 51


def test_a_missing_column_is_refused_rather_than_guessed():
    with pytest.raises(KeyError, match="prediction"):
        permutation_null(pd.DataFrame({"holdout_year": [2021]}), FakeBacktest)
    with pytest.raises(KeyError, match="holdout_year"):
        permutation_null(pd.DataFrame({"prediction": [1.0]}), FakeBacktest)


def test_one_degenerate_draw_does_not_lose_the_rest():
    calls = {"n": 0}

    class Flaky(FakeBacktest):
        def __init__(self, frame):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("degenerate cross-section")
            super().__init__(frame)

    result = permutation_null(a_panel(0.0), Flaky, n_permutations=20, seed=6)
    assert np.isnan(result.draws).sum() == 1
    assert np.isfinite(result.mean)
