"""End-to-end orchestration.

    universe -> prices/events/fundamentals/filings
             -> align -> abnormal returns
             -> features (point-in-time)
             -> walk-forward model -> signal
             -> sector-neutral portfolio -> costs -> evaluation

Each stage is a function that takes and returns tidy frames, so any of them can
be run, cached and inspected on its own. The whole thing is also runnable
against the synthetic provider with no network, which is what the ``demo``
command does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .backtest.costs import CostModel
from .backtest.engine import BacktestResult, run_backtest
from .backtest.portfolio import build_positions
from .config import Config
from .data.universe import Universe
from .events.alignment import align_events
from .features.assemble import assemble_features, cross_sectional_normalise, feature_columns
from .features.fundamentals import _pivot, build_fundamental_features
from .features.surprise import build_surprise_features
from .features.text import build_text_features
from .models.evaluate import evaluate_predictions
from .models.pipelines import build_estimator
from .models.walkforward import WalkForwardSplitter, run_walk_forward
from .returns.abnormal import AbnormalReturnResult, ReturnPanel, compute_abnormal_returns
from .returns.stats import summarise_windows
from .utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Dataset:
    """Everything the research stages need, already aligned and validated."""

    universe: Universe
    prices: pd.DataFrame
    events: pd.DataFrame
    fundamentals: pd.DataFrame | None = None
    filings: pd.DataFrame | None = None
    text_loader: object | None = None
    panel: ReturnPanel | None = None
    meta: dict = field(default_factory=dict)


def build_dataset(
    provider,
    config: Config,
    start: str,
    end: str,
    tickers: list[str] | None = None,
    with_fundamentals: bool = True,
    with_filings: bool = True,
) -> Dataset:
    """Pull and align everything from a provider (or bundle of providers)."""
    universe = _universe_from(provider, tickers, config)
    tickers = tickers or universe.all_tickers()
    market = config.returns.market_symbol

    prices = provider.get_prices([*tickers, market], start, end)
    events = provider.get_events(tickers, start, end)
    aligned = align_events(events, prices, config.events)
    aligned = universe.filter_events(aligned, "t0")

    fundamentals = None
    if with_fundamentals and hasattr(provider, "get_fundamentals"):
        try:
            fundamentals = provider.get_fundamentals(tickers, start, end)
        except Exception as exc:
            log.warning("fundamentals unavailable: %s", exc)

    filings = None
    text_loader = None
    if with_filings and hasattr(provider, "get_filings"):
        try:
            filings = provider.get_filings(tickers, start, end)
            text_loader = getattr(provider, "get_text", None)
        except Exception as exc:
            log.warning("filings unavailable: %s", exc)

    panel = ReturnPanel.from_prices(
        prices,
        market_symbol=market,
        sector_map=universe.sector_map(),
        winsorize=config.returns.winsorize,
    )
    return Dataset(
        universe=universe,
        prices=prices,
        events=aligned,
        fundamentals=fundamentals,
        filings=filings,
        text_loader=text_loader,
        panel=panel,
        meta={
            "start": start,
            "end": end,
            "n_tickers": len(tickers),
            "n_events": int(len(aligned)),
            "static_universe": universe.static_membership,
        },
    )


def _universe_from(provider, tickers, config: Config) -> Universe:
    if config.universe.membership_file:
        return Universe.from_csv(config.universe.membership_file, name=config.universe.name)
    if hasattr(provider, "get_universe"):
        return Universe.from_frame(provider.get_universe(), name=config.universe.name)
    if tickers is None:
        raise ValueError("no universe available: supply tickers or a membership file")
    return Universe.static(
        tickers, acknowledge_bias=config.universe.allow_static_membership, name=config.universe.name
    )


def run_event_study(dataset: Dataset, config: Config) -> AbnormalReturnResult:
    """Abnormal returns for every aligned event."""
    assert dataset.panel is not None
    return compute_abnormal_returns(dataset.events, dataset.panel, config.returns)


def build_feature_panel(
    dataset: Dataset, result: AbnormalReturnResult, config: Config
) -> pd.DataFrame:
    """Point-in-time features joined to labels."""
    blocks: dict[str, pd.DataFrame] = {}
    if dataset.fundamentals is not None and config.features.fundamentals:
        wide = _pivot(dataset.fundamentals)
        blocks["fund"] = build_fundamental_features(dataset.fundamentals)
        if config.features.surprise:
            blocks["sue"] = build_surprise_features(wide)
    if dataset.filings is not None and dataset.text_loader is not None and config.features.text:
        blocks["text"] = build_text_features(
            dataset.filings, dataset.text_loader, config.features.lm_dictionary_path
        )

    # Fundamentals and surprise attach by publication time, not by period_end:
    # against real SEC data the two calendars do not agree and a key-based join
    # silently matches nothing. Text stays keyed, since a filing is tied to the
    # event that produced it.
    join_keys = {name: "asof" for name in ("fund", "sue") if name in blocks}
    panel = assemble_features(
        dataset.events, blocks, join_keys=join_keys, sector_map=dataset.universe.sector_map()
    )
    labels = result.summary.drop(
        columns=[c for c in result.summary.columns if c in panel.columns and c != "event_id"]
    )
    panel = panel.merge(labels, on="event_id", how="inner")
    feats = feature_columns(panel)
    panel = cross_sectional_normalise(
        panel, feats, method=config.features.cross_sectional_transform
    )
    log.info("feature panel: %d events x %d features", len(panel), len(feats))
    return panel


def run_model(panel: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict, list]:
    """Walk-forward fit and out-of-sample evaluation."""
    target = config.model.target
    if target not in panel.columns:
        raise KeyError(
            f"target {target!r} not in the panel; available labels: "
            f"{[c for c in panel.columns if c.startswith('car_')][:6]}"
        )
    features = [c for c in feature_columns(panel) if not c.startswith(("car_", "bhar_"))]
    if not features:
        raise ValueError("no usable features in the panel")

    splitter = WalkForwardSplitter(
        initial_train_years=config.model.initial_train_years,
        validation_years=config.model.validation_years,
        step_years=config.model.step_years,
        embargo_days=config.model.embargo_days,
        label_horizon_days=max(hi for _, hi in config.returns.windows) + 1,
    )
    estimator = build_estimator(config.model.kind, random_state=config.model.random_state)
    preds, splits = run_walk_forward(panel, features, target, estimator, splitter)
    metrics = evaluate_predictions(preds, target_col=target, quantiles=config.backtest.quantiles)
    return preds, metrics, splits


def run_portfolio(
    preds: pd.DataFrame, result: AbnormalReturnResult, config: Config
) -> BacktestResult:
    """Sector-neutral long/short book, net of costs."""
    positions = build_positions(
        preds,
        quantiles=config.backtest.quantiles,
        holding_days=config.backtest.holding_days,
        entry_offset=config.backtest.entry_offset,
    )
    if positions.empty:
        raise RuntimeError(
            "no positions were opened; the out-of-sample sample is too short to "
            "seed the trailing quantile breakpoints"
        )
    cost_model = CostModel(
        half_spread_bps=config.backtest.costs.half_spread_bps,
        commission_bps=config.backtest.costs.commission_bps,
        impact_coef_bps=config.backtest.costs.impact_coef_bps,
        participation=config.backtest.costs.participation,
    )
    ar_col = (
        "ar_sector_neutral"
        if "ar_sector_neutral" in result.daily.columns
        else "ar_market_model"
    )
    return run_backtest(
        positions,
        result.daily,
        ar_column=ar_col,
        entry_offset=config.backtest.entry_offset,
        holding_days=config.backtest.holding_days,
        sector_neutral=config.backtest.sector_neutral,
        gross_exposure=config.backtest.gross_exposure,
        max_weight=config.backtest.max_weight,
        cost_model=cost_model,
    )


def save_stage(df: pd.DataFrame, out_dir: str | Path, name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def significance_table(result: AbnormalReturnResult, config: Config, n_boot: int = 1000):
    return summarise_windows(
        result.summary,
        estimators=config.returns.estimators,
        windows=config.returns.windows,
        n_boot=n_boot,
        seed=config.seed,
    )
