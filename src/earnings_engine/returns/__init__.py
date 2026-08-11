"""Abnormal return measurement and its statistical significance."""

from .abnormal import AbnormalReturnResult, ReturnPanel, compute_abnormal_returns
from .estimation import MarketModelFit, fit_market_model
from .stats import (
    bmp_test,
    cluster_bootstrap_ci,
    cross_sectional_ttest,
    newey_west_tstat,
    summarise_windows,
)

__all__ = [
    "AbnormalReturnResult",
    "MarketModelFit",
    "ReturnPanel",
    "bmp_test",
    "cluster_bootstrap_ci",
    "compute_abnormal_returns",
    "cross_sectional_ttest",
    "fit_market_model",
    "newey_west_tstat",
    "summarise_windows",
]
