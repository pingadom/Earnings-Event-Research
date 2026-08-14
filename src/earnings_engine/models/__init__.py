"""Modelling: leakage-free walk-forward evaluation and signal construction."""

from .evaluate import decile_spread, evaluate_predictions, information_coefficient
from .pipelines import SignalModel, build_estimator
from .walkforward import WalkForwardSplit, WalkForwardSplitter, run_walk_forward

__all__ = [
    "SignalModel",
    "WalkForwardSplit",
    "WalkForwardSplitter",
    "build_estimator",
    "decile_spread",
    "evaluate_predictions",
    "information_coefficient",
    "run_walk_forward",
]
