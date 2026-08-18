"""Rolling annual holdouts: train through year Y-1, freeze, predict year Y.

Why this and not just the walk-forward loop
-------------------------------------------
The walk-forward splitter already prevents leakage, but its folds are anchored
to the first event in the sample, which makes the results awkward to talk
about. Anchoring folds to calendar years turns the evaluation into a claim a
reader can check: *"using only data available on 31 December 2018, the model
predicted 2019; here is what actually happened."* Repeated for every year in
the range, that is six independent out-of-sample tests rather than one.

One good year is luck. Six consistent years is evidence. Six years with a clear
downward trend is the decay you would expect of a real anomaly that got
arbitraged, and is itself worth reporting.

What gets measured
------------------
For each holdout year, three different questions:

**Does it rank?**       Rank IC between prediction and realised abnormal return,
                        computed per month and aggregated, plus the top-minus-
                        bottom quintile spread.
**Is it calibrated?**   Regress realised on predicted. A slope of 1 means the
                        predicted magnitudes are right; a slope well below 1
                        (the usual outcome) means the ranking has information
                        but the model is overconfident about size. Ranking and
                        calibration are different properties and a signal can
                        have the first without the second.
**Does it pay?**        Net-of-cost return of the sector-neutral book restricted
                        to that year.

Every year is also compared against a **single-feature baseline** (earnings
surprise alone). A model that cannot beat one column is not earning its
complexity, and reporting the baseline stops that going unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from .backtest.costs import CostModel
from .backtest.engine import run_backtest
from .backtest.portfolio import build_positions
from .config import Config
from .features.assemble import feature_columns
from .models.evaluate import evaluate_predictions
from .models.pipelines import build_estimator
from .returns.abnormal import AbnormalReturnResult
from .utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class YearResult:
    """Everything measured for one held-out year."""

    year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_train: int
    n_test: int
    n_purged: int
    # ranking
    ic_mean: float
    ic_tstat: float
    ic_hit_rate: float
    ic_pooled: float
    spread_tstat: float
    monotonicity: float
    # calibration
    calib_slope: float
    calib_intercept: float
    calib_r2: float
    predicted_spread: float
    realised_spread: float
    # economics
    ann_return_gross: float
    ann_return_net: float
    sharpe_net: float
    tstat_nw: float
    max_drawdown: float
    turnover: float
    avg_positions: float
    # baseline
    baseline_ic: float
    baseline_spread: float

    def as_row(self) -> dict:
        d = dict(self.__dict__)
        d["train_start"] = str(pd.Timestamp(self.train_start).date())
        d["train_end"] = str(pd.Timestamp(self.train_end).date())
        return d


@dataclass
class HoldoutResult:
    by_year: pd.DataFrame
    predictions: pd.DataFrame
    backtest: object | None
    coefficients: pd.DataFrame
    target: str
    baseline_feature: str
    aggregate: dict = field(default_factory=dict)

    def summary_markdown(self) -> str:
        cols = [
            "year", "n_train", "n_test", "ic_mean", "ic_tstat",
            "predicted_spread", "realised_spread", "spread_tstat",
            "calib_slope", "sharpe_net", "ann_return_net", "baseline_ic",
        ]
        df = self.by_year[cols].copy()
        for c in ("predicted_spread", "realised_spread", "ann_return_net"):
            df[c] = (df[c] * 10_000).round(0)
        df = df.round(3)
        df = df.rename(columns={
            "predicted_spread": "predicted (bp)",
            "realised_spread": "realised (bp)",
            "spread_tstat": "spread t",
            "ann_return_net": "net return (bp)",
            "calib_slope": "calib slope",
            "baseline_ic": "baseline IC",
        })
        return df.to_markdown(index=False)


def _year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def _calibration(pred: np.ndarray, real: np.ndarray) -> tuple[float, float, float]:
    """OLS of realised on predicted.

    Slope 1 = predicted magnitudes are right. Slope near 0 with a positive IC =
    the ordering is informative but the scale is not, which is the normal
    outcome and matters for position sizing.
    """
    ok = np.isfinite(pred) & np.isfinite(real)
    if ok.sum() < 30 or np.std(pred[ok]) == 0:
        return np.nan, np.nan, np.nan
    res = sps.linregress(pred[ok], real[ok])
    return float(res.slope), float(res.intercept), float(res.rvalue**2)


def _quantile_spread(pred: pd.Series, real: pd.Series, q: int) -> tuple[float, float, float]:
    """Realised top-minus-bottom spread, and the *predicted* spread alongside it."""
    df = pd.DataFrame({"p": pred, "r": real}).dropna()
    if len(df) < q * 5:
        return np.nan, np.nan, np.nan
    bucket = pd.qcut(df["p"].rank(method="first"), q, labels=False)
    realised = df.groupby(bucket)["r"].mean()
    predicted = df.groupby(bucket)["p"].mean()
    top, bottom = q - 1, 0
    realised_spread = float(realised.loc[top] - realised.loc[bottom])
    predicted_spread = float(predicted.loc[top] - predicted.loc[bottom])
    mono = float(sps.spearmanr(np.arange(q), realised.reindex(range(q))).statistic)
    return realised_spread, predicted_spread, mono


def run_annual_holdouts(
    panel: pd.DataFrame,
    study: AbnormalReturnResult,
    config: Config,
    years: range | list[int],
    *,
    baseline_feature: str = "sue_timeseries",
    min_train: int = 400,
    date_col: str = "t0",
) -> HoldoutResult:
    """Fit and evaluate one frozen model per holdout year."""
    target = config.model.target
    if target not in panel.columns:
        raise KeyError(f"target {target!r} missing from the panel")

    features = [c for c in feature_columns(panel) if not c.startswith(("car_", "bhar_"))]
    if baseline_feature not in features:
        log.warning("baseline feature %r not available; skipping baseline", baseline_feature)
        baseline_feature = ""

    df = panel.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(df[date_col])
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)
    df[date_col] = dates

    horizon = max(hi for _, hi in config.returns.windows) + 1
    # A label observed at t0 is not resolved until ~t0 + horizon sessions.
    resolved = df[date_col] + pd.Timedelta(days=int(horizon * 1.45))
    embargo = pd.Timedelta(days=config.model.embargo_days)

    rows: list[YearResult] = []
    all_preds: list[pd.DataFrame] = []
    coefs: list[pd.Series] = []

    for year in years:
        start, end = _year_bounds(int(year))
        # Purge: only training events whose outcome was already known, and
        # stop an embargo period before the holdout year opens.
        train_mask = (df[date_col] < start - embargo) & (resolved < start)
        test_mask = (df[date_col] >= start) & (df[date_col] <= end)
        n_purged = int(((df[date_col] < start) & ~train_mask).sum())

        train, test = df[train_mask], df[test_mask]
        labelled = train[target].notna()
        if labelled.sum() < min_train or len(test) < 50:
            log.info(
                "holdout %d skipped: %d labelled training events, %d test events",
                year, int(labelled.sum()), len(test),
            )
            continue

        model = build_estimator(config.model.kind, random_state=config.model.random_state)
        model.fit(train.loc[labelled, features], train.loc[labelled, target])
        pred = model.predict(test[features])

        carry = ("event_id", "ticker", date_col, "sector", target)
        out = test[[c for c in carry if c in test.columns]].copy()
        out["prediction"] = pred
        out["holdout_year"] = int(year)
        if baseline_feature:
            out["baseline"] = test[baseline_feature].to_numpy()
        all_preds.append(out)

        linear = getattr(model.named_steps.get("model"), "coef_", None)
        if linear is not None:
            flat = np.ravel(linear)
            if len(flat) == len(features):
                coefs.append(pd.Series(flat, index=features, name=int(year)))
            else:
                # Belt and braces: a preprocessing step dropped columns, so the
                # coefficients no longer line up with the feature list. Skip the
                # stability record rather than mislabel it.
                log.warning(
                    "holdout %d: %d coefficients for %d features; skipping the "
                    "coefficient record for this fold",
                    year, len(flat), len(features),
                )

        rows.append(
            _score_year(
                int(year), out, target, study, config, train, labelled, n_purged, baseline_feature
            )
        )

    if not rows:
        raise RuntimeError(
            "no holdout year had enough training data; widen the sample or lower min_train"
        )

    by_year = pd.DataFrame([r.as_row() for r in rows])
    predictions = pd.concat(all_preds, ignore_index=True)
    coefficients = pd.DataFrame(coefs).T if coefs else pd.DataFrame()

    backtest = _stitched_backtest(predictions, study, config)
    aggregate = _aggregate(by_year, predictions, target, backtest)
    return HoldoutResult(
        by_year=by_year,
        predictions=predictions,
        backtest=backtest,
        coefficients=coefficients,
        target=target,
        baseline_feature=baseline_feature,
        aggregate=aggregate,
    )


def _score_year(
    year, out, target, study, config, train, labelled, n_purged, baseline_feature
) -> YearResult:
    q = config.backtest.quantiles
    metrics = evaluate_predictions(out, target_col=target, quantiles=q)
    real = out[target]
    pred = out["prediction"]

    realised_spread, predicted_spread, mono = _quantile_spread(pred, real, q)
    slope, intercept, r2 = _calibration(pred.to_numpy(), real.to_numpy())

    base_ic, base_spread = np.nan, np.nan
    if baseline_feature and "baseline" in out.columns:
        sub = out[["baseline", target]].dropna()
        if len(sub) > 50:
            base_ic = float(sps.spearmanr(sub["baseline"], sub[target]).statistic)
            base_spread, _, _ = _quantile_spread(out["baseline"], real, q)

    bt = _year_backtest(out, study, config)
    stats = bt.stats if bt is not None else {}

    return YearResult(
        year=year,
        train_start=train["t0"].min(),
        train_end=train["t0"].max(),
        n_train=int(labelled.sum()),
        n_test=int(len(out)),
        n_purged=n_purged,
        ic_mean=float(metrics["ic_mean"]),
        ic_tstat=float(metrics["ic_tstat_nw"]),
        ic_hit_rate=float(metrics["ic_hit_rate"]),
        ic_pooled=float(metrics["ic_pooled"]),
        spread_tstat=float(metrics["quantile_spread_tstat_nw"]),
        monotonicity=mono,
        calib_slope=slope,
        calib_intercept=intercept,
        calib_r2=r2,
        predicted_spread=predicted_spread,
        realised_spread=realised_spread,
        ann_return_gross=float(stats.get("ann_return_gross", np.nan)),
        ann_return_net=float(stats.get("ann_return_net", np.nan)),
        sharpe_net=float(stats.get("sharpe_net", np.nan)),
        tstat_nw=float(stats.get("tstat_nw", np.nan)),
        max_drawdown=float(stats.get("max_drawdown", np.nan)),
        turnover=float(stats.get("ann_turnover", np.nan)),
        avg_positions=float(stats.get("avg_positions", np.nan)),
        baseline_ic=base_ic,
        baseline_spread=base_spread,
    )


def _cost_model(config: Config) -> CostModel:
    c = config.backtest.costs
    return CostModel(
        half_spread_bps=c.half_spread_bps,
        commission_bps=c.commission_bps,
        impact_coef_bps=c.impact_coef_bps,
        participation=c.participation,
    )


def _year_backtest(out, study, config):
    """Backtest a single holdout year in isolation.

    Breakpoints inside ``build_positions`` still look only at past events, so a
    single-year slice starts cold and trades fewer names early in the year --
    which is the honest picture of running the strategy from a standing start.
    """
    try:
        positions = build_positions(
            out,
            quantiles=config.backtest.quantiles,
            holding_days=config.backtest.holding_days,
            entry_offset=config.backtest.entry_offset,
            min_lookback_events=60,
        )
        if positions.empty:
            return None
        return run_backtest(
            positions,
            study.daily,
            ar_column=(
                "ar_sector_neutral"
                if "ar_sector_neutral" in study.daily.columns
                else "ar_market_model"
            ),
            entry_offset=config.backtest.entry_offset,
            holding_days=config.backtest.holding_days,
            sector_neutral=config.backtest.sector_neutral,
            gross_exposure=config.backtest.gross_exposure,
            max_weight=config.backtest.max_weight,
            cost_model=_cost_model(config),
        )
    except (ValueError, KeyError) as exc:
        log.warning("backtest failed for holdout slice: %s", exc)
        return None


def _stitched_backtest(predictions, study, config):
    """One continuous backtest over every holdout year, run end to end."""
    return _year_backtest(predictions.sort_values("t0"), study, config)


def _aggregate(by_year: pd.DataFrame, predictions: pd.DataFrame, target: str, backtest) -> dict:
    ic = by_year["ic_mean"].dropna()
    # A plain t-test, not Newey-West. Annual ICs are far enough apart to treat
    # as independent, and a HAC estimator on six observations is unstable
    # enough to produce confident-looking nonsense. Six points is six points:
    # this statistic has very little power and should be read as a sanity
    # check on consistency, not as the evidence.
    t_across = float(sps.ttest_1samp(ic, 0.0).statistic) if len(ic) > 2 else np.nan
    pooled = predictions[["prediction", target]].dropna()
    agg = {
        "n_years": int(len(by_year)),
        "years": f"{int(by_year['year'].min())}-{int(by_year['year'].max())}",
        "n_predictions": int(len(predictions)),
        "mean_ic": float(ic.mean()) if len(ic) else np.nan,
        "ic_tstat_across_years": float(t_across) if t_across == t_across else np.nan,
        "ic_std_across_years": float(ic.std(ddof=1)) if len(ic) > 1 else np.nan,
        "positive_ic_years": int((ic > 0).sum()),
        "mean_realised_spread": float(by_year["realised_spread"].mean()),
        "mean_predicted_spread": float(by_year["predicted_spread"].mean()),
        "mean_calibration_slope": float(by_year["calib_slope"].mean()),
        "mean_baseline_ic": float(by_year["baseline_ic"].mean()),
        "pooled_ic": float(sps.spearmanr(pooled["prediction"], pooled[target]).statistic)
        if len(pooled) > 100 else np.nan,
    }
    # Is the edge decaying? Regress annual IC on the year index.
    if len(ic) >= 4:
        trend = sps.linregress(by_year.loc[ic.index, "year"], ic)
        agg["ic_trend_per_year"] = float(trend.slope)
        agg["ic_trend_p"] = float(trend.pvalue)
    if backtest is not None:
        agg.update({
            "stitched_sharpe_net": float(backtest.stats.get("sharpe_net", np.nan)),
            "stitched_tstat_nw": float(backtest.stats.get("tstat_nw", np.nan)),
            "stitched_ann_return_net": float(backtest.stats.get("ann_return_net", np.nan)),
            "stitched_max_drawdown": float(backtest.stats.get("max_drawdown", np.nan)),
        })
    return agg
