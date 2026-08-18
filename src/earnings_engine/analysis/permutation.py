"""An empirical null for the backtest, obtained by shuffling the predictions.

Why this exists
---------------
A long-short book is not a neutral instrument. Overlapping twenty-day holdings,
rebalanced daily into a per-name cap that binds on most days, do not return zero
when the signal driving them is worthless -- they return something slightly
negative, reliably. The synthetic null control makes this visible: no ranking
skill, exactly dollar-neutral, and still roughly three percent a year of drift
before costs.

That matters for reading the real study. If a Sharpe ratio of -0.6 is reported,
the reader deserves to know how much of it is evidence against the hypothesis
and how much is what this book does to any signal at all.

The clean way to answer that is not to reason about it. It is to run the same
book on the same events with the *predictions shuffled*, many times, and look at
the distribution that comes back. Shuffling within each holdout year preserves
everything except the thing under test: the same events, the same calendar, the
same cross-sectional spread of predicted values, the same sector composition --
only the correspondence between a prediction and the stock it belongs to is
destroyed.

What comes back is a null distribution for whatever statistic you care about,
and the realised value can be read against it as a percentile. If the observed
Sharpe sits in the middle of the shuffled distribution, it is not evidence of
anything; the book would have produced it from noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class PermutationNull:
    """The shuffled-prediction distribution of a backtest statistic."""

    statistic: str
    observed: float
    draws: np.ndarray = field(repr=False)
    n_permutations: int

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.draws))

    @property
    def std(self) -> float:
        return float(np.nanstd(self.draws, ddof=1))

    @property
    def percentile(self) -> float:
        """Where the observed value falls in the shuffled distribution, 0-100."""
        finite = self.draws[np.isfinite(self.draws)]
        if not len(finite):
            return float("nan")
        return float((finite < self.observed).mean() * 100)

    @property
    def p_value_one_sided(self) -> float:
        """P(shuffled >= observed): small means the result beats noise.

        The ``+1`` in both terms is the standard finite-sample correction. It
        keeps the p-value from ever being exactly zero, which no permutation
        test with finitely many draws is entitled to claim.
        """
        finite = self.draws[np.isfinite(self.draws)]
        if not len(finite):
            return float("nan")
        return float((np.sum(finite >= self.observed) + 1) / (len(finite) + 1))

    @property
    def excess(self) -> float:
        """Observed minus what the book produces from noise alone."""
        return self.observed - self.mean

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "statistic": self.statistic,
            "observed": self.observed,
            "null_mean": self.mean,
            "null_std": self.std,
            "percentile": self.percentile,
            "p_value_one_sided": self.p_value_one_sided,
            "excess_over_null": self.excess,
            "n_permutations": self.n_permutations,
        }

    def render(self) -> str:
        d = self.as_dict()
        return (
            f"{self.statistic}: observed {d['observed']:.3f}, "
            f"shuffled null {d['null_mean']:.3f} +/- {d['null_std']:.3f} "
            f"({self.n_permutations} permutations), "
            f"percentile {d['percentile']:.0f}, p = {d['p_value_one_sided']:.3f}"
        )


def permutation_null(
    predictions: pd.DataFrame,
    backtest_fn,
    *,
    statistic: str = "sharpe_net",
    observed: float | None = None,
    n_permutations: int = 200,
    seed: int = 20260818,
    year_col: str = "holdout_year",
) -> PermutationNull:
    """Shuffle predictions within each holdout year and re-run the book.

    Parameters
    ----------
    predictions
        One row per scored event, carrying ``prediction`` and ``holdout_year``.
    backtest_fn
        ``frame -> object with .stats``. Passed in rather than imported so this
        module stays independent of how a book happens to be constructed.
    observed
        The realised statistic. Recomputed from ``predictions`` if omitted.
    """
    if "prediction" not in predictions.columns:
        raise KeyError("predictions frame has no 'prediction' column")
    if year_col not in predictions.columns:
        raise KeyError(f"predictions frame has no {year_col!r} column to shuffle within")

    if observed is None:
        result = backtest_fn(predictions)
        observed = float(result.stats.get(statistic, np.nan))

    rng = np.random.default_rng(seed)
    frame = predictions.copy()
    groups = [np.asarray(idx) for _year, idx in frame.groupby(year_col).groups.items()]
    values = frame["prediction"].to_numpy(copy=True)

    draws = np.full(n_permutations, np.nan)
    for i in range(n_permutations):
        shuffled = values.copy()
        for positions in groups:
            block = shuffled[positions]
            rng.shuffle(block)
            shuffled[positions] = block
        frame["prediction"] = shuffled
        try:
            draws[i] = float(backtest_fn(frame).stats.get(statistic, np.nan))
        except Exception as exc:  # one degenerate draw must not lose the rest
            log.warning("permutation %d failed (%s)", i, exc)
        if (i + 1) % 50 == 0:
            log.info("permutation null: %d/%d", i + 1, n_permutations)

    out = PermutationNull(
        statistic=statistic,
        observed=float(observed),
        draws=draws,
        n_permutations=n_permutations,
    )
    log.info("%s", out.render())
    return out
