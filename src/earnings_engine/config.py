"""Configuration loading.

Configuration is a plain YAML file resolved into a frozen dataclass tree, so
that a typo in a config key fails at load time rather than silently changing
a backtest six modules later.

Precedence (highest first):
    1. explicit keyword overrides passed to :func:`load_config`
    2. environment variables (``EEE_DATA_DIR``, ``EEE_VENDOR_DIR``)
    3. the YAML file
    4. dataclass defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "config.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or contains unknown keys."""


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "data"
    vendor_dir: str = "data/vendor"
    reports_dir: str = "reports"

    def resolve(self, root: Path = REPO_ROOT) -> ResolvedPaths:
        return ResolvedPaths(
            data=_abs(self.data_dir, root),
            vendor=_abs(self.vendor_dir, root),
            reports=_abs(self.reports_dir, root),
        )


@dataclass(frozen=True)
class ResolvedPaths:
    data: Path
    vendor: Path
    reports: Path

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    def ensure(self) -> ResolvedPaths:
        for p in (self.raw, self.interim, self.processed, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class UniverseConfig:
    name: str = "sp500"
    membership_file: str | None = None
    #: If no point-in-time membership file is supplied we fall back to the
    #: current index constituents, which is survivorship-biased. That is only
    #: allowed when explicitly acknowledged -- see docs/biases.md.
    allow_static_membership: bool = False
    min_price: float = 5.0
    min_dollar_volume: float = 1_000_000.0


@dataclass(frozen=True)
class EventsConfig:
    #: Exchange timezone used to interpret announcement timestamps.
    exchange_tz: str = "America/New_York"
    #: How to treat an announcement that lands while the market is open.
    #: "next_open" is the conservative choice and the default.
    intraday_policy: str = "next_open"
    #: Announcements with no known time of day are assumed to be after the
    #: close, which is the modal convention for US issuers.
    unknown_time_policy: str = "amc"
    #: Drop events whose t0 has fewer than this many prior trading days of
    #: price history (needed for the market-model estimation window).
    min_history_days: int = 150


@dataclass(frozen=True)
class ReturnsConfig:
    #: Which abnormal-return estimators to compute.
    estimators: tuple[str, ...] = ("market_adjusted", "market_model", "sector_neutral")
    #: Estimation window for the market model, in trading days relative to t0.
    estimation_start: int = -250
    estimation_end: int = -31
    min_estimation_obs: int = 100
    #: Event windows as (start, end) inclusive offsets in trading days from t0.
    #: Windows starting at 0 include the announcement jump, which a strategy
    #: entering at the t0 open cannot capture. Windows starting at 1 are the
    #: tradable post-announcement drift. Both are reported; only the latter
    #: should ever be used as a model target.
    windows: tuple[tuple[int, int], ...] = ((0, 0), (0, 4), (0, 19), (1, 5), (1, 20))
    market_symbol: str = "SPY"
    #: Winsorise daily returns at these quantiles before estimation.
    winsorize: tuple[float, float] = (0.005, 0.995)


@dataclass(frozen=True)
class FeaturesConfig:
    fundamentals: bool = True
    surprise: bool = True
    text: bool = True
    #: Cross-sectional standardisation applied per event date.
    cross_sectional_transform: str = "rank_gauss"
    #: Path to the Loughran-McDonald master dictionary if you have it.
    lm_dictionary_path: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "ridge"
    #: Must be a drift window (starting at rel_day >= 1); see BacktestConfig.entry_offset.
    target: str = "car_market_model_1_20"
    #: Walk-forward: initial training span in calendar years, then step forward.
    initial_train_years: int = 6
    validation_years: int = 1
    step_years: int = 1
    #: Purge/embargo (trading days) to stop overlapping event windows leaking
    #: from the training set into the test set. Must be >= the longest window.
    embargo_days: int = 25
    random_state: int = 20260818


@dataclass(frozen=True)
class CostsConfig:
    #: One-way cost in basis points applied to traded notional.
    half_spread_bps: float = 3.0
    commission_bps: float = 0.5
    #: Square-root market-impact coefficient: impact_bps = coef * sqrt(participation)
    impact_coef_bps: float = 10.0
    #: Assumed participation rate of ADV per name per day.
    participation: float = 0.05


@dataclass(frozen=True)
class BacktestConfig:
    #: First relative day on which the position is live. 1, not 0: the
    #: announcement gap opens before you can trade, so a strategy that
    #: enters at the t0 open never earns it. Setting this to 0 books the
    #: jump as profit and is the most common way this study is faked.
    entry_offset: int = 1
    holding_days: int = 20
    quantiles: int = 5
    sector_neutral: bool = True
    gross_exposure: float = 1.0
    max_weight: float = 0.02
    costs: CostsConfig = field(default_factory=CostsConfig)


@dataclass(frozen=True)
class Config:
    seed: int = 20260818
    paths: PathsConfig = field(default_factory=PathsConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    returns: ReturnsConfig = field(default_factory=ReturnsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def resolved_paths(self) -> ResolvedPaths:
        return self.paths.resolve()


def _abs(value: str, root: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p)


_NESTED = {
    "paths": PathsConfig,
    "universe": UniverseConfig,
    "events": EventsConfig,
    "returns": ReturnsConfig,
    "features": FeaturesConfig,
    "model": ModelConfig,
    "backtest": BacktestConfig,
}


def _coerce(annotation: Any, value: Any) -> Any:
    """Turn YAML lists into tuples so the dataclasses stay hashable/frozen."""
    if isinstance(value, list):
        return tuple(tuple(v) if isinstance(v, list) else v for v in value)
    return value


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load configuration from YAML, applying env vars then keyword overrides."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigError(f"{path} must contain a top-level mapping")
            raw = loaded

    top_known = {f.name for f in fields(Config)}
    unknown = set(raw) - top_known
    if unknown:
        raise ConfigError(f"unknown top-level config key(s): {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _NESTED:
            kwargs[key] = _build_nested(_NESTED[key], value, key)
        else:
            kwargs[key] = _coerce(None, value)

    cfg = Config(**kwargs)
    cfg = _apply_env(cfg)

    for key, value in overrides.items():
        if key not in top_known:
            raise ConfigError(f"unknown override {key!r}")
        cfg = replace(cfg, **{key: value})
    return cfg


def _build_nested(cls: type, data: Any, name: str) -> Any:
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"config section {name!r} must be a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)} in section {name!r}")
    kwargs = {}
    for key, value in data.items():
        if cls is BacktestConfig and key == "costs":
            kwargs[key] = _build_nested(CostsConfig, value, "backtest.costs")
        else:
            kwargs[key] = _coerce(None, value)
    return cls(**kwargs)


def _apply_env(cfg: Config) -> Config:
    data_dir = os.environ.get("EEE_DATA_DIR")
    vendor_dir = os.environ.get("EEE_VENDOR_DIR")
    if not (data_dir or vendor_dir):
        return cfg
    paths = cfg.paths
    if data_dir:
        paths = replace(paths, data_dir=data_dir)
    if vendor_dir:
        paths = replace(paths, vendor_dir=vendor_dir)
    return replace(cfg, paths=paths)
