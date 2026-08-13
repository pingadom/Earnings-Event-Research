"""Assemble the feature panel -- the one place features meet events.

Everything funnels through here so the point-in-time check happens exactly
once, in exactly one place, and cannot be bypassed by a convenient merge
somewhere else in the codebase.

The cross-sectional transform is not cosmetic either. Fitting a model on raw
feature levels pooled across a decade means the model spends its capacity
learning that 2020 was strange. Ranking within each event date and mapping to a
normal score means every observation is "how does this quarter compare to the
other quarters reported around the same time", which is the comparison the
hypothesis is actually about -- and it neutralises time-varying scale, fat
tails and level drift in one step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

from ..events.pit import assert_point_in_time
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

_META = {
    "event_id",
    "ticker",
    "t0",
    "t0_pos",
    "period_end",
    "fiscal_quarter",
    "trade_open_ts",
    "announced_at_utc",
    "available_from_utc",
    "timing",
    "timing_imputed",
    "accepts_intraday_lookahead",
    "sector",
    "accession",
    "form",
    "filed_at_utc",
    "days_since_period_end",
    "_source_file",
    "_restated",
}


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Columns that are features rather than metadata, labels or diagnostics."""
    return [
        c
        for c in panel.columns
        if c not in _META
        and not c.startswith(("car_", "bhar_", "nobs_", "mm_", "est_", "mktdev_", "_"))
        and pd.api.types.is_numeric_dtype(panel[c])
    ]


def assemble_features(
    events: pd.DataFrame,
    blocks: dict[str, pd.DataFrame],
    *,
    join_keys: dict[str, list[str]] | None = None,
    sector_map: dict[str, str] | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Join feature blocks onto aligned events under point-in-time discipline.

    Parameters
    ----------
    events
        Output of :func:`earnings_engine.events.align_events`; must carry
        ``trade_open_ts``.
    blocks
        ``{name: frame}``. Each frame needs ``available_from_utc`` and enough
        keys to join (``ticker`` + ``period_end`` by default).
    """
    if "trade_open_ts" not in events.columns:
        raise KeyError("events must be aligned first (missing 'trade_open_ts')")
    join_keys = join_keys or {}
    panel = events.copy()
    panel["trade_open_ts"] = pd.to_datetime(panel["trade_open_ts"], utc=True)

    stamps = []
    for name, block in blocks.items():
        if block is None or block.empty:
            log.warning("feature block %r is empty; skipping", name)
            continue
        if "available_from_utc" not in block.columns:
            raise KeyError(f"feature block {name!r} has no available_from_utc column")
        keys = join_keys.get(name, ["ticker", "period_end"])
        missing = [k for k in keys if k not in block.columns or k not in panel.columns]
        if missing:
            raise KeyError(f"feature block {name!r} cannot join on {missing}")

        b = block.copy()
        b["available_from_utc"] = pd.to_datetime(b["available_from_utc"], utc=True)
        stamp_col = f"available_from_{name}"
        b = b.rename(columns={"available_from_utc": stamp_col})
        drop = [c for c in b.columns if c in panel.columns and c not in keys]
        if drop:
            b = b.drop(columns=drop)
        before = len(panel)
        panel = panel.merge(b, on=keys, how="left", suffixes=("", f"_{name}"))
        if len(panel) != before:
            raise ValueError(
                f"joining block {name!r} changed the row count {before} -> {len(panel)}; "
                f"the join keys {keys} are not unique in that block"
            )
        stamps.append(stamp_col)

        # Anything published after the open is not knowable: blank the whole
        # block for that event rather than keeping a half-filled row.
        late = panel[stamp_col] > panel["trade_open_ts"]
        n_late = int(late.fillna(False).sum())
        if n_late:
            block_cols = [c for c in b.columns if c not in keys and c != stamp_col]
            panel.loc[late, block_cols] = np.nan
            panel.loc[late, stamp_col] = pd.NaT
            log.info(
                "block %r: blanked %d/%d event(s) whose data post-dated the trade open",
                name,
                n_late,
                len(panel),
            )

    if sector_map is not None and "sector" not in panel.columns:
        panel["sector"] = panel["ticker"].map(sector_map)

    if validate and stamps:
        # The binding constraint is the *latest* input a row depends on.
        panel["available_from_utc"] = panel[stamps].max(axis=1)
        check = panel.dropna(subset=["available_from_utc"])
        assert_point_in_time(check, label="assembled feature panel")

    return panel


def cohort_key(panel: pd.DataFrame, date_col: str = "t0", freq: str = "M") -> pd.Series:
    """Group events into comparison cohorts.

    Normalising against the handful of firms that happened to report on the
    exact same calendar day gives a tiny, noisy cross-section. Firms reporting
    within the same reporting *month* are the natural comparison set: they
    share the macro backdrop, and there are enough of them for a rank to mean
    something. ``freq="D"`` recovers same-day grouping if you want it.
    """
    dates = pd.to_datetime(panel[date_col])
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)
    if freq.upper() == "D":
        return dates.dt.normalize()
    return dates.dt.to_period(freq).dt.to_timestamp()


def cross_sectional_normalise(
    panel: pd.DataFrame,
    columns: list[str] | None = None,
    date_col: str = "t0",
    method: str = "rank_gauss",
    min_obs: int = 15,
    freq: str = "M",
) -> pd.DataFrame:
    """Standardise features within each reporting cohort.

    ``rank_gauss`` maps within-cohort ranks through the inverse normal CDF,
    which is robust to the fat tails fundamentals data is full of. ``zscore``
    and ``rank`` are also available; ``none`` passes through.

    Cohorts with fewer than ``min_obs`` non-null observations are set to NaN
    rather than normalised: a rank among four names is not a cross-section.
    """
    if method == "none":
        return panel.copy()
    columns = columns or feature_columns(panel)
    out = panel.copy()
    key = cohort_key(out, date_col, freq)
    grouped = out.groupby(key, sort=False)

    for col in columns:
        if col not in out.columns:
            continue
        s = out[col]
        count = grouped[col].transform("count")
        if method == "zscore":
            mu = grouped[col].transform("mean")
            sd = grouped[col].transform("std")
            out[col] = (s - mu) / sd.where(sd > 0)
        elif method in {"rank", "rank_gauss"}:
            rank = grouped[col].rank(method="average", na_option="keep")
            uniform = (rank - 0.5) / count
            if method == "rank":
                out[col] = uniform
            else:
                out[col] = pd.Series(
                    ndtri(uniform.clip(1e-6, 1 - 1e-6)), index=out.index
                ).where(uniform.notna())
        else:
            raise ValueError(f"unknown normalisation method {method!r}")
        out.loc[count < min_obs, col] = np.nan

    thin = int((grouped[columns[0]].transform("count") < min_obs).sum()) if columns else 0
    if thin:
        log.info(
            "cross-sectional normalisation: %d row(s) in cohorts smaller than %d were dropped",
            thin,
            min_obs,
        )
    return out
