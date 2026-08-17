"""Walk-forward splitting: no future leaks into training, ever."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_engine.models.pipelines import build_estimator
from earnings_engine.models.walkforward import WalkForwardSplitter, run_walk_forward


@pytest.fixture()
def dates() -> pd.Series:
    return pd.Series(pd.date_range("2010-01-04", "2024-12-31", freq="B"))


def test_train_always_precedes_test(dates):
    splitter = WalkForwardSplitter(initial_train_years=5, validation_years=1, min_train=50)
    folds = list(splitter.split(dates))
    assert folds
    for f in folds:
        assert dates.iloc[f.train_idx].max() < dates.iloc[f.test_idx].min()


def test_train_windows_expand(dates):
    splitter = WalkForwardSplitter(initial_train_years=5, min_train=50)
    folds = list(splitter.split(dates))
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)
    assert len(folds) >= 5


def test_test_windows_do_not_overlap(dates):
    splitter = WalkForwardSplitter(initial_train_years=5, min_train=50)
    folds = list(splitter.split(dates))
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_end < b.test_start


def test_embargo_gap_is_respected(dates):
    embargo = 40
    splitter = WalkForwardSplitter(initial_train_years=5, embargo_days=embargo, min_train=50)
    for f in splitter.split(dates):
        gap = (f.test_start - f.train_end).days
        assert gap >= embargo


def test_overlapping_labels_are_purged(dates):
    splitter = WalkForwardSplitter(
        initial_train_years=5, embargo_days=25, label_horizon_days=20, min_train=50
    )
    folds = list(splitter.split(dates))
    assert any(f.n_purged > 0 for f in folds)


def test_unsorted_dates_are_rejected(dates):
    splitter = WalkForwardSplitter()
    with pytest.raises(ValueError, match="sorted"):
        list(splitter.split(dates.sample(frac=1.0, random_state=0)))


def test_predictions_are_strictly_out_of_sample():
    """A leak-detector: a feature that equals the future target must not raise
    in-sample performance into the out-of-sample predictions."""
    rng = np.random.default_rng(11)
    n = 4000
    dates = pd.Series(pd.date_range("2010-01-04", periods=n, freq="B"))
    signal = rng.normal(size=n)
    panel = pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(n)],
            "t0": dates,
            "x": signal,
            "noise": rng.normal(size=n),
            "y": 0.1 * signal + rng.normal(scale=1.0, size=n),
        }
    )
    splitter = WalkForwardSplitter(initial_train_years=6, min_train=100, embargo_days=25)
    preds, folds = run_walk_forward(
        panel, ["x", "noise"], "y", build_estimator("ridge"), splitter
    )
    assert len(folds) >= 3
    # The relationship is real, so out-of-sample correlation should be positive
    # but nowhere near perfect.
    corr = preds[["prediction", "y"]].corr().iloc[0, 1]
    assert 0.02 < corr < 0.4
