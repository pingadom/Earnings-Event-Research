"""Figures, tables and the interactive dashboard."""

from .dashboard import write_dashboard
from .plots import (
    plot_calibration,
    plot_car_by_quantile,
    plot_coefficient_stability,
    plot_cost_sensitivity,
    plot_equity_curve,
    plot_event_study,
    plot_holdout_ic,
    plot_ic_timeseries,
    plot_predicted_vs_realised,
)
from .tables import format_significance_table, format_stats, write_report

__all__ = [
    "format_significance_table",
    "format_stats",
    "plot_calibration",
    "plot_car_by_quantile",
    "plot_coefficient_stability",
    "plot_cost_sensitivity",
    "plot_equity_curve",
    "plot_event_study",
    "plot_holdout_ic",
    "plot_ic_timeseries",
    "plot_predicted_vs_realised",
    "write_dashboard",
    "write_report",
]
