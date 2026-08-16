"""Figures and tables for the write-up."""

from .plots import (
    plot_car_by_quantile,
    plot_cost_sensitivity,
    plot_equity_curve,
    plot_event_study,
    plot_ic_timeseries,
)
from .tables import format_significance_table, write_report

__all__ = [
    "format_significance_table",
    "plot_car_by_quantile",
    "plot_cost_sensitivity",
    "plot_equity_curve",
    "plot_event_study",
    "plot_ic_timeseries",
    "write_report",
]
