"""Post-hoc analysis: is the result real, and is it new?

Three questions any quantitative reviewer will ask about a backtest, each with
a module here:

``attribution``      Is this alpha, or a known factor wearing a hat?
``multiple_testing`` How many specifications were tried before this one?
``fama_macbeth``     Does a second, independent methodology agree?

None of these can rescue a bad result. They exist to stop a *good-looking* one
from being believed for the wrong reason.
"""

from .attribution import FactorModel, attribute_returns, normalise_factors, synthetic_factors
from .diagnostics import Diagnostics, run_diagnostics
from .fama_macbeth import FamaMacBethResult, fama_macbeth
from .multiple_testing import (
    TrialsLog,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)

__all__ = [
    "Diagnostics",
    "FactorModel",
    "FamaMacBethResult",
    "TrialsLog",
    "attribute_returns",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "fama_macbeth",
    "min_track_record_length",
    "normalise_factors",
    "probabilistic_sharpe_ratio",
    "run_diagnostics",
    "synthetic_factors",
]
