"""Does post-earnings drift exist in this sample at all?

The rest of this repository asks a harder question -- can a model rank
companies by their subsequent abnormal return -- and answers it in the
negative. That answer is weak on its own, because a weak model explains it
just as well as an absent effect does. "My ridge regression did not predict
the drift" and "there is no drift to predict" look identical from the outside,
and only the second is a finding.

This module tests the second directly, with no model in the way.

**The sort.** Rank every announcement by how much it surprised the market --
standardised unexpected earnings, the same measure Bernard and Thomas (1989)
used -- and look at what the top and bottom groups did afterwards. If drift
exists, the top group out-performs. No fitting, no training window, no
hyperparameter: a sort and a mean.

**The subsample.** Where the effect is *supposed* to live matters as much as
whether it lives. Drift is documented to concentrate in small, illiquid,
lightly-covered firms, on the theory that those are where information
diffuses slowly. An S&P 500 universe is the most heavily arbitraged corner of
the market, so finding nothing there is unsurprising and proves little. Finding
nothing in the *least* liquid quintile of the S&P 500 -- the closest this
sample gets to where the effect should be -- says considerably more.

Liquidity is measured by the trailing twenty-day median dollar volume the
universe filter already computes, lagged one day, so the split uses nothing
that was not knowable before the announcement.

**The statistics.** Announcements cluster in calendar time: hundreds land in
the same week, and their abnormal returns share whatever the market did that
week. Treating them as independent draws overstates significance badly. Every
test here resamples whole event dates rather than individual events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..returns.stats import cluster_bootstrap_ci
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

#: Below this many events a quantile mean is too noisy to report.
MIN_GROUP = 30


@dataclass(frozen=True)
class SpreadResult:
    """A long-short spread with its date-clustered uncertainty."""

    n: int
    mean: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    n_clusters: int


def _quantile_labels(values: pd.Series, n: int) -> pd.Series:
    """Rank into ``n`` equal groups, tolerating ties and missing values."""
    ranked = values.rank(method="first", na_option="keep")
    try:
        return pd.qcut(ranked, n, labels=False, duplicates="drop")
    except ValueError:  # too few distinct values to cut
        return pd.Series(np.nan, index=values.index)


def sorted_drift(
    events: pd.DataFrame,
    sort_col: str,
    target_col: str,
    *,
    n_quantiles: int = 5,
    cluster_col: str = "t0",
    n_boot: int = 2000,
    seed: int = 20260818,
) -> pd.DataFrame:
    """Mean abnormal return by quantile of a sorting variable, plus the spread.

    The returned frame has one row per quantile and a final ``Q5-Q1`` row
    carrying the long-short spread with its clustered confidence interval. The
    spread row is the test; the per-quantile rows are there so a reader can see
    whether the relationship is monotone or driven by one tail.
    """
    needed = {sort_col, target_col, cluster_col}
    missing = needed - set(events.columns)
    if missing:
        raise KeyError(f"sorted_drift needs {sorted(missing)}")

    df = events[[sort_col, target_col, cluster_col]].dropna()
    if len(df) < MIN_GROUP * n_quantiles:
        raise ValueError(
            f"{len(df)} usable events is too few to cut into {n_quantiles} quantiles; "
            f"at least {MIN_GROUP} per group are needed for the means to mean anything"
        )
    df = df.assign(bucket=_quantile_labels(df[sort_col], n_quantiles))
    df = df.dropna(subset=["bucket"])

    rows = []
    for bucket, group in df.groupby("bucket", sort=True):
        if len(group) < MIN_GROUP:
            continue
        test = cluster_bootstrap_ci(
            group[target_col], group[cluster_col], n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "group": f"Q{int(bucket) + 1}",
                "n": test.n,
                "mean_sort_value": float(group[sort_col].mean()),
                "mean_car": test.mean,
                "t_stat": test.t_stat,
                "p_value": test.p_value,
                "ci_low": test.ci_low,
                "ci_high": test.ci_high,
            }
        )
    if len(rows) < 2:
        raise ValueError("fewer than two usable quantiles; nothing to compare")

    top, bottom = int(df["bucket"].max()), int(df["bucket"].min())
    spread = _spread_test(df, target_col, cluster_col, top, bottom, n_boot, seed)
    rows.append(
        {
            "group": f"Q{top + 1}-Q{bottom + 1}",
            "n": spread.n,
            "mean_sort_value": np.nan,
            "mean_car": spread.mean,
            "t_stat": spread.t_stat,
            "p_value": spread.p_value,
            "ci_low": spread.ci_low,
            "ci_high": spread.ci_high,
        }
    )
    return pd.DataFrame(rows)


def _spread_test(df, target_col, cluster_col, top, bottom, n_boot, seed):
    """Bootstrap the top-minus-bottom difference by resampling event dates.

    Not a rescaled pooled mean: each draw resamples whole dates and recomputes
    *both* group means inside that draw, so the statistic bootstrapped is the
    difference itself. Resampling dates rather than events is what keeps the
    hundreds of announcements that share a week from counting as hundreds of
    independent observations -- the correction that usually decides whether a
    long-short spread looks significant.

    Each date is reduced once to four numbers -- the sum and count of its top
    observations and of its bottom ones -- because a mean of a union of groups
    is recoverable from sums and counts. The resampling loop then never touches
    a DataFrame, which turns an operation that took minutes per test into one
    that takes a fraction of a second.
    """
    subset = df.loc[df["bucket"].isin([top, bottom]), [target_col, cluster_col, "bucket"]]
    subset = subset.dropna(subset=[target_col])
    if subset.empty:
        raise ValueError("no events in the top or bottom bucket")

    is_top = (subset["bucket"] == top).to_numpy()
    values = subset[target_col].to_numpy(dtype="float64")
    codes, _uniques = pd.factorize(subset[cluster_col], sort=False)
    n_clusters = int(codes.max()) + 1

    top_sum = np.bincount(codes, weights=np.where(is_top, values, 0.0), minlength=n_clusters)
    top_n = np.bincount(codes, weights=is_top.astype("float64"), minlength=n_clusters)
    bottom_sum = np.bincount(codes, weights=np.where(is_top, 0.0, values), minlength=n_clusters)
    bottom_n = np.bincount(codes, weights=(~is_top).astype("float64"), minlength=n_clusters)

    def difference(picks: np.ndarray) -> float:
        n_t, n_b = top_n[picks].sum(), bottom_n[picks].sum()
        if n_t == 0 or n_b == 0:
            return np.nan
        return float(top_sum[picks].sum() / n_t - bottom_sum[picks].sum() / n_b)

    everything = np.arange(n_clusters)
    observed = difference(everything)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    with np.errstate(invalid="ignore", divide="ignore"):
        n_t = top_n[picks].sum(axis=1)
        n_b = bottom_n[picks].sum(axis=1)
        draws = np.where(
            (n_t > 0) & (n_b > 0),
            top_sum[picks].sum(axis=1) / np.where(n_t > 0, n_t, np.nan)
            - bottom_sum[picks].sum(axis=1) / np.where(n_b > 0, n_b, np.nan),
            np.nan,
        )
    draws = draws[np.isfinite(draws)]
    if draws.size < n_boot // 2:
        raise ValueError("too many bootstrap draws lacked one of the two groups")

    se = float(draws.std(ddof=1))
    low, high = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
    # Centre the draws on zero for the p-value: how often would a distribution
    # with no true difference reach the observed one?
    # The +1 terms are the standard finite-sample correction: no test with
    # finitely many draws is entitled to report a p-value of exactly zero, and
    # printing one invites a reader to believe a precision that is not there.
    centred = draws - draws.mean()
    exceed = int(np.sum(np.abs(centred) >= abs(observed)))
    p_value = float(min(1.0, 2 * (exceed + 1) / (draws.size + 1)))
    return SpreadResult(
        n=int(len(subset)),
        mean=observed,
        t_stat=observed / se if se > 0 else np.nan,
        p_value=p_value,
        ci_low=low,
        ci_high=high,
        n_clusters=n_clusters,
    )


def drift_by_horizon(
    events: pd.DataFrame,
    sort_col: str,
    target_cols: dict[str, str],
    **kwargs,
) -> pd.DataFrame:
    """The sorted spread at each horizon, so the term structure is visible.

    Drift is a claim about *accumulation*: the gap should widen from one day to
    five to twenty. A spread that is already fully present on day one is an
    announcement-day reaction, not drift, and the two are routinely conflated.
    """
    rows = []
    for label, column in target_cols.items():
        if column not in events.columns:
            log.info("drift_by_horizon: %s not in the panel, skipping", column)
            continue
        table = sorted_drift(events, sort_col, column, **kwargs)
        spread = table.iloc[-1]
        rows.append(
            {
                "horizon": label,
                "n": int(spread["n"]),
                "spread": float(spread["mean_car"]),
                "t_stat": float(spread["t_stat"]),
                "p_value": float(spread["p_value"]),
                "ci_low": float(spread["ci_low"]),
                "ci_high": float(spread["ci_high"]),
            }
        )
    if not rows:
        raise ValueError("no requested horizon was present in the panel")
    return pd.DataFrame(rows)


def drift_by_subsample(
    events: pd.DataFrame,
    sort_col: str,
    target_col: str,
    group_col: str,
    *,
    n_groups: int = 3,
    group_names: tuple[str, ...] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """The sorted spread computed separately within each subsample.

    Splitting by liquidity is the sharp version of the test. If the effect has
    been arbitraged out of the largest names but survives in the smallest, the
    spread rises monotonically as liquidity falls. If it is absent everywhere,
    including where the literature says it should be strongest, that is a much
    harder result to explain away as a limitation of the universe.
    """
    if group_col not in events.columns:
        raise KeyError(f"drift_by_subsample needs {group_col!r} in the panel")
    df = events.dropna(subset=[group_col]).copy()
    df["_group"] = _quantile_labels(df[group_col], n_groups)
    df = df.dropna(subset=["_group"])

    rows = []
    for group, subset in df.groupby("_group", sort=True):
        index = int(group)
        name = (
            group_names[index]
            if group_names and index < len(group_names)
            else f"{group_col} group {index + 1}"
        )
        try:
            table = sorted_drift(subset, sort_col, target_col, **kwargs)
        except ValueError as exc:
            log.info("subsample %s skipped: %s", name, exc)
            continue
        spread = table.iloc[-1]
        rows.append(
            {
                "subsample": name,
                "n": int(spread["n"]),
                "median_group_value": float(subset[group_col].median()),
                "spread": float(spread["mean_car"]),
                "t_stat": float(spread["t_stat"]),
                "p_value": float(spread["p_value"]),
                "ci_low": float(spread["ci_low"]),
                "ci_high": float(spread["ci_high"]),
            }
        )
    if not rows:
        raise ValueError("no subsample had enough events to test")
    return pd.DataFrame(rows)
