"""Walk-forward evaluation with purging and embargo.

Why not k-fold
--------------
Random k-fold cross-validation on financial panel data is not a mild
approximation, it is wrong. It trains on 2022 to predict 2016, so the model can
exploit relationships that were only discoverable later, and it splits
overlapping observations across the fold boundary so information leaks
directly. Reported out-of-sample R-squared under k-fold routinely doubles what
an honest split produces.

The scheme here
---------------
Expanding-window walk-forward. Fold *k* trains on everything up to a cutoff,
validates on the following ``validation_years``, then the cutoff rolls forward
by ``step_years`` and the process repeats. That mirrors how the strategy would
actually have been run: at every point, only the past is available.

Purging and embargo (Lopez de Prado)
------------------------------------
A 20-day CAR label observed at ``t0`` is not resolved until ``t0 + 20``. If a
training event's label window overlaps the test period, its label contains
information from the test period. So training events whose label window reaches
into the test set are *purged*, and a further ``embargo_days`` gap is imposed
after the test window before training resumes, to break serial correlation
across the boundary.

Skipping this is the single most common way a promising backtest turns out to
be nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class WalkForwardSplit:
    """One fold: integer positions into the (date-sorted) panel."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_purged: int

    def describe(self) -> str:
        return (
            f"fold {self.fold}: train {self.train_start.date()}..{self.train_end.date()} "
            f"(n={len(self.train_idx)}, purged {self.n_purged}) -> "
            f"test {self.test_start.date()}..{self.test_end.date()} (n={len(self.test_idx)})"
        )


class WalkForwardSplitter:
    """Expanding-window splitter with label-overlap purging and an embargo."""

    def __init__(
        self,
        initial_train_years: int = 6,
        validation_years: int = 1,
        step_years: int = 1,
        embargo_days: int = 25,
        label_horizon_days: int = 20,
        min_train: int = 200,
    ) -> None:
        if embargo_days < label_horizon_days:
            log.warning(
                "embargo_days (%d) is shorter than the label horizon (%d); overlapping "
                "labels can leak across the fold boundary",
                embargo_days,
                label_horizon_days,
            )
        self.initial_train_years = initial_train_years
        self.validation_years = validation_years
        self.step_years = step_years
        self.embargo_days = embargo_days
        self.label_horizon_days = label_horizon_days
        self.min_train = min_train

    def split(self, dates: pd.Series) -> Iterator[WalkForwardSplit]:
        d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        if isinstance(d.dtype, pd.DatetimeTZDtype):
            d = d.dt.tz_localize(None)
        if not d.is_monotonic_increasing:
            raise ValueError("dates must be sorted ascending before splitting")

        start, end = d.iloc[0], d.iloc[-1]
        # Label resolution date: when the outcome for an event is fully known.
        resolved = d + pd.Timedelta(days=int(self.label_horizon_days * 1.45))

        fold = 0
        train_end = start + pd.DateOffset(years=self.initial_train_years)
        while True:
            test_start = train_end
            test_end = test_start + pd.DateOffset(years=self.validation_years)
            if test_start >= end:
                break

            # Purge: drop training events whose label window reaches the test set.
            train_mask = (d < train_end) & (resolved < test_start)
            naive_mask = d < train_end
            n_purged = int(naive_mask.sum() - train_mask.sum())

            # Embargo: also exclude a buffer immediately before the test start.
            embargo_cut = test_start - pd.Timedelta(days=self.embargo_days)
            train_mask &= d <= embargo_cut

            test_mask = (d >= test_start) & (d < test_end)
            if test_mask.sum() == 0:
                train_end = train_end + pd.DateOffset(years=self.step_years)
                continue
            if train_mask.sum() < self.min_train:
                log.info(
                    "fold %d skipped: only %d training rows (min %d)",
                    fold,
                    int(train_mask.sum()),
                    self.min_train,
                )
                train_end = train_end + pd.DateOffset(years=self.step_years)
                if train_end >= end:
                    break
                continue

            train_idx = np.flatnonzero(train_mask.to_numpy())
            test_idx = np.flatnonzero(test_mask.to_numpy())
            yield WalkForwardSplit(
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                train_start=d.iloc[train_idx[0]],
                train_end=d.iloc[train_idx[-1]],
                test_start=d.iloc[test_idx[0]],
                test_end=d.iloc[test_idx[-1]],
                n_purged=n_purged,
            )
            fold += 1
            train_end = train_end + pd.DateOffset(years=self.step_years)
            if train_end >= end:
                break


def run_walk_forward(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    model,
    splitter: WalkForwardSplitter,
    date_col: str = "t0",
) -> tuple[pd.DataFrame, list[WalkForwardSplit]]:
    """Fit ``model`` fold by fold and collect strictly out-of-sample predictions.

    A fresh clone of the estimator is fitted per fold; nothing is carried over,
    including scaler statistics, which would otherwise leak test-period moments
    into training.
    """
    from sklearn.base import clone  # noqa: PLC0415

    df = panel.sort_values(date_col).reset_index(drop=True)
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise KeyError(f"panel is missing column(s) {missing}")

    preds: list[pd.DataFrame] = []
    splits: list[WalkForwardSplit] = []
    for split in splitter.split(df[date_col]):
        train = df.iloc[split.train_idx]
        test = df.iloc[split.test_idx]

        x_tr = train[features]
        y_tr = train[target]
        ok = y_tr.notna() & x_tr.notna().any(axis=1)
        if ok.sum() < splitter.min_train:
            log.info("fold %d has too few labelled training rows; skipping", split.fold)
            continue

        est = clone(model)
        est.fit(x_tr[ok], y_tr[ok])
        yhat = est.predict(test[features])

        carry = ("event_id", "ticker", date_col, "sector", target)
        out = test[[c for c in carry if c in test.columns]].copy()
        out["prediction"] = yhat
        out["fold"] = split.fold
        preds.append(out)
        splits.append(split)
        log.info(split.describe())

    if not preds:
        raise RuntimeError(
            "walk-forward produced no folds; the sample is too short for "
            f"initial_train_years={splitter.initial_train_years}"
        )
    return pd.concat(preds, ignore_index=True), splits
