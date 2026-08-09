"""Point-in-time enforcement.

Look-ahead bias does not usually arrive as a dramatic mistake. It arrives as a
merge on ``period_end`` instead of ``available_from``, and it inflates results
by an amount that looks exactly like skill.

The invariant this module enforces is a single inequality:

    for every (event, feature):  feature.available_from_utc <= event.trade_open_ts

Nothing else in the codebase is allowed to build a feature panel except through
:func:`restrict_to_known`, and :func:`assert_point_in_time` is called again at
the model boundary as a belt-and-braces check. Both are covered by regression
tests that deliberately plant a leak and assert that it is caught.
"""

from __future__ import annotations

import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger(__name__)


class PointInTimeError(AssertionError):
    """Raised when a feature would have been unknown at the moment of trading."""


def assert_point_in_time(
    panel: pd.DataFrame,
    *,
    known_col: str = "available_from_utc",
    trade_col: str = "trade_open_ts",
    label: str = "feature panel",
) -> None:
    """Raise if any row carries information dated after its trade timestamp."""
    for col in (known_col, trade_col):
        if col not in panel.columns:
            raise KeyError(f"{label}: expected column {col!r} for the point-in-time check")
    known = pd.to_datetime(panel[known_col], utc=True)
    trade = pd.to_datetime(panel[trade_col], utc=True)
    violations = known > trade
    n = int(violations.fillna(False).sum())
    if n:
        sample = (
            panel.loc[violations, [c for c in ("event_id", "ticker", "item") if c in panel.columns]]
            .head(5)
            .to_dict("records")
        )
        worst = (known - trade)[violations].max()
        raise PointInTimeError(
            f"{label}: {n} row(s) use information published after the trade timestamp "
            f"(worst offender is {worst} late). Examples: {sample}"
        )
    missing = int(known.isna().sum())
    if missing:
        raise PointInTimeError(
            f"{label}: {missing} row(s) have no {known_col}; a feature with no known "
            "publication time cannot be proven free of look-ahead."
        )


def restrict_to_known(
    events: pd.DataFrame,
    facts: pd.DataFrame,
    *,
    on: str = "ticker",
    known_col: str = "available_from_utc",
    trade_col: str = "trade_open_ts",
    keep: str = "last",
) -> pd.DataFrame:
    """As-of join: attach to each event only facts already public at its open.

    This is the only sanctioned way to join a fact table onto events. It is an
    ``asof`` merge in the *publication-time* dimension, not the period
    dimension, which is precisely the distinction people get wrong.

    Parameters
    ----------
    keep
        ``"last"`` attaches the most recently published matching fact;
        ``"all"`` returns every fact known at the open (useful for building
        trailing-window features such as year-on-year growth).
    """
    if events.empty or facts.empty:
        return facts.head(0).copy()
    if trade_col not in events.columns:
        raise KeyError(f"events frame needs {trade_col!r}; run align_events first")
    if known_col not in facts.columns:
        raise KeyError(f"facts frame needs {known_col!r}")

    ev = events[[c for c in ("event_id", on, trade_col) if c in events.columns]].copy()
    ev[trade_col] = pd.to_datetime(ev[trade_col], utc=True)
    fc = facts.copy()
    fc[known_col] = pd.to_datetime(fc[known_col], utc=True)

    if keep == "all":
        merged = ev.merge(fc, on=on, how="inner")
        merged = merged.loc[merged[known_col] <= merged[trade_col]]
        return merged.reset_index(drop=True)

    if keep != "last":
        raise ValueError("keep must be 'last' or 'all'")

    ev = ev.sort_values(trade_col)
    fc = fc.sort_values(known_col)
    merged = pd.merge_asof(
        ev,
        fc,
        left_on=trade_col,
        right_on=known_col,
        by=on,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.dropna(subset=[known_col]).reset_index(drop=True)


def audit_panel(panel: pd.DataFrame, known_col: str = "available_from_utc") -> pd.DataFrame:
    """Summarise the information lag per feature. Useful in a write-up.

    A lag distribution with mass at or below zero days is a red flag: it means
    the "fact" was dated no later than the moment you traded on it, which
    usually indicates the publication timestamp is the period end in disguise.
    """
    if known_col not in panel.columns or "trade_open_ts" not in panel.columns:
        raise KeyError("panel needs both available_from_utc and trade_open_ts")
    lag = (
        pd.to_datetime(panel["trade_open_ts"], utc=True)
        - pd.to_datetime(panel[known_col], utc=True)
    ).dt.total_seconds() / 86400.0
    group = panel["item"] if "item" in panel.columns else pd.Series("all", index=panel.index)
    return (
        lag.groupby(group)
        .agg(n="size", min_lag_days="min", p05=lambda s: s.quantile(0.05), median="median")
        .reset_index()
        .rename(columns={group.name or "index": "item"})
    )
