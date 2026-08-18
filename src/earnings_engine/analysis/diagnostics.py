"""Run the three post-hoc checks together and report them as one block.

Kept separate from the modules that implement them so the checks stay usable
individually, and so the orchestration -- which factor set, which trial count,
which frequency -- lives in one readable place rather than being scattered
through the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger
from .attribution import FactorModel, attribute_returns, normalise_factors, synthetic_factors
from .fama_macbeth import FamaMacBethResult, fama_macbeth
from .multiple_testing import TrialsLog, min_track_record_length

log = get_logger(__name__)


@dataclass
class Diagnostics:
    attribution: FactorModel | None = None
    deflated: dict = field(default_factory=dict)
    fama_macbeth: FamaMacBethResult | None = None
    min_track_record: float = float("nan")
    factor_source: str = ""
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> dict[str, str]:
        """One markdown block per check, ready to drop into a report."""
        out: dict[str, str] = {}
        if self.attribution is not None:
            table = self.attribution.summary_frame().copy()
            table["estimate"] = table["estimate"].round(4)
            out["Factor attribution"] = (
                f"_{self.attribution.verdict()}_\n\n"
                f"Factors: {self.factor_source}. "
                f"{self.attribution.n_obs} daily observations, Newey-West with "
                f"{self.attribution.hac_lags} lags.\n\n"
                + table.round(4).to_markdown(index=False)
                + f"\n\nR-squared {self.attribution.r_squared:.3f} · "
                f"residual volatility {self.attribution.residual_vol_annual:.2%} · "
                f"appraisal ratio {self.attribution.appraisal_ratio:.2f}"
            )
        if self.deflated:
            d = self.deflated
            out["Multiple testing"] = (
                f"_{d.get('verdict', '')}_\n\n"
                f"| quantity | value |\n|---|---:|\n"
                f"| Observed Sharpe (annualised) | {d.get('sharpe_annual', float('nan')):.3f} |\n"
                f"| Skew / excess kurtosis | {d.get('skew', float('nan')):.2f} / "
                f"{d.get('kurtosis', float('nan')) - 3:.2f} |\n"
                f"| Specifications tried | {d.get('n_trials', 0)} |\n"
                f"| Sharpe the best of those would reach by luck | "
                f"{d.get('expected_max_sharpe', float('nan')) * np.sqrt(252):.3f} |\n"
                f"| **Deflated Sharpe ratio** | **{d.get('dsr', float('nan')):.3f}** |\n"
                f"| Minimum track record length (days) | {self.min_track_record:,.0f} |\n"
            )
        if self.fama_macbeth is not None:
            out["Fama-MacBeth cross-sectional regressions"] = (
                f"_{self.fama_macbeth.verdict()}_\n\n"
                f"{self.fama_macbeth.n_periods} periods at frequency "
                f"'{self.fama_macbeth.frequency}', mean cross-section "
                f"{self.fama_macbeth.mean_cross_section:.0f} names. Coefficients are basis "
                f"points of abnormal return per one cross-sectional standard deviation of the "
                f"feature.\n\n" + self.fama_macbeth.to_markdown()
            )
        if self.notes:
            out["Diagnostic notes"] = "\n".join(f"- {n}" for n in self.notes)
        return out


def resolve_factors(
    panel=None, factor_file: str | Path | None = None, seed: int = 20260818
) -> tuple[pd.DataFrame, str]:
    """Get daily factor returns, preferring real data over the synthetic proxy."""
    if factor_file:
        path = Path(factor_file)
        raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        return normalise_factors(raw), f"Ken French daily library ({path.name})"
    if panel is None:
        raise ValueError("supply either a factor file or a ReturnPanel")
    return (
        synthetic_factors(panel, seed=seed),
        "synthetic proxies built from the generated panel (NOT Ken French data)",
    )


def _top_features(panel: pd.DataFrame, features: list[str], target: str, k: int) -> list[str]:
    """The k features most correlated with the target, by |Spearman rho|."""
    scores = {}
    for f in features:
        sub = panel[[f, target]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 100 or not (sub[f].std() > 0):
            continue
        rho = sub[f].corr(sub[target], method="spearman")
        if pd.notna(rho):
            scores[f] = abs(float(rho))
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:k] if ranked else features[:k]


def run_diagnostics(
    daily_returns: pd.Series,
    panel_frame: pd.DataFrame | None = None,
    features: list[str] | None = None,
    target: str | None = None,
    *,
    return_panel=None,
    factor_file: str | Path | None = None,
    trials_path: str | Path | None = None,
    fm_frequency: str = "M",
    fm_max_features: int = 8,
    seed: int = 20260818,
) -> Diagnostics:
    """Attribution, multiple-testing correction and Fama-MacBeth in one call."""
    diag = Diagnostics()

    try:
        factors, source = resolve_factors(return_panel, factor_file, seed)
        diag.factor_source = source
        diag.attribution = attribute_returns(daily_returns, factors, already_excess=True)
        if "synthetic" in source:
            diag.notes.append(
                "Factor loadings are against synthetic proxies, so they establish that the "
                "regression works, not that the strategy is factor-neutral in real markets. "
                "Supply --factor-file with Ken French data for a real attribution."
            )
    except Exception as exc:
        log.warning("factor attribution unavailable: %s", exc)
        diag.notes.append(f"Factor attribution skipped: {exc}")

    try:
        log_path = Path(trials_path) if trials_path else Path("conf/trials.json")
        trials = TrialsLog.load(log_path)
        if trials.n == 0:
            diag.notes.append(
                f"No trials logged at {log_path}, so the deflated Sharpe has nothing to deflate "
                "against. Record every specification you evaluate with TrialsLog."
            )
        else:
            diag.deflated = trials.deflate(daily_returns)
            diag.min_track_record = min_track_record_length(daily_returns)
    except Exception as exc:
        log.warning("deflated Sharpe unavailable: %s", exc)
        diag.notes.append(f"Multiple-testing correction skipped: {exc}")

    if panel_frame is not None and features and target:
        try:
            # Regress on the features the model actually leans on. Throwing
            # thirty regressors at a forty-name monthly cross-section produces
            # coefficients that are pure noise, however good they look.
            chosen = _top_features(panel_frame, features, target, fm_max_features)
            diag.fama_macbeth = fama_macbeth(
                panel_frame, chosen, target, frequency=fm_frequency
            )
            if len(chosen) < len(features):
                diag.notes.append(
                    f"Fama-MacBeth uses the {len(chosen)} features with the strongest "
                    f"univariate rank correlation to the target, out of {len(features)}. A "
                    f"monthly cross-section cannot support thirty regressors."
                )
        except Exception as exc:
            log.warning("Fama-MacBeth unavailable: %s", exc)
            diag.notes.append(f"Fama-MacBeth skipped: {exc}")

    return diag
