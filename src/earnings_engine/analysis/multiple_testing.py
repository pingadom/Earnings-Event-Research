"""Multiple testing: how many things did you try before this one?

The question that kills most backtests, and the one most write-ups skip. If you
test twenty specifications at the 5% level, one of them clears the bar by
construction. The reported Sharpe ratio of the winner is then a *maximum* of a
sample, not an estimate of anything, and it is biased upward by an amount that
can be computed rather than guessed at.

Three tools, in the order you would use them:

**Probabilistic Sharpe Ratio** (Bailey & Lopez de Prado 2012)
    The probability that the true Sharpe exceeds a benchmark, given the observed
    Sharpe, the track record length, and -- critically -- the *skew and kurtosis*
    of the returns. A strategy with negative skew and fat tails needs a longer
    record to establish the same Sharpe, because occasional large losses are
    exactly what a short sample is likely to miss. Strategies that sell
    optionality look wonderful right up until they do not, and this is the
    correction that says so.

**Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014)
    The PSR with the benchmark set to the Sharpe you would *expect* the best of
    N independent trials to produce by luck alone. Deflated Sharpe above 0.95
    means the result survives the number of attempts made. It requires an honest
    N, which is what :class:`TrialsLog` is for.

**Minimum Track Record Length**
    How many observations you would need before a Sharpe of this size, with
    these higher moments, could be called significant. Frequently longer than
    the sample you have, which is worth knowing before making claims.

The honest N
------------
``expected_max_sharpe`` is only meaningful with a real trial count. A
:class:`TrialsLog` records every specification actually evaluated -- target
window, model family, feature set, estimator -- so the count is a fact in the
repository rather than a number chosen after the fact. Under-reporting N inflates
the deflated Sharpe, which defeats the entire point.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


def _moments(returns: np.ndarray) -> tuple[float, float, float, int]:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return np.nan, np.nan, np.nan, n
    sharpe = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else np.nan
    return sharpe, float(sps.skew(r)), float(sps.kurtosis(r, fisher=False)), n


def probabilistic_sharpe_ratio(
    returns, benchmark_sharpe: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> dict:
    """Probability that the true (per-period) Sharpe exceeds ``benchmark_sharpe``.

    ``benchmark_sharpe`` is in the same per-period units as the returns; pass an
    annualised figure divided by ``sqrt(periods_per_year)`` if that is what you
    have.
    """
    sr, skew, kurt, n = _moments(returns)
    if not np.isfinite(sr) or n < 3:
        return {"psr": np.nan, "sharpe": sr, "n": n}
    denom = np.sqrt(max(1e-12, 1 - skew * sr + ((kurt - 1) / 4.0) * sr**2))
    z = (sr - benchmark_sharpe) * np.sqrt(n - 1) / denom
    return {
        "psr": float(sps.norm.cdf(z)),
        "sharpe": sr,
        "sharpe_annual": sr * np.sqrt(periods_per_year),
        "skew": skew,
        "kurtosis": kurt,
        "n": n,
        "benchmark_sharpe": benchmark_sharpe,
    }


def expected_max_sharpe(n_trials: int, variance_of_trials: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent, worthless strategies.

    ``variance_of_trials`` is the variance of the Sharpe ratios *across the
    specifications you tried*, in the **same per-period units** as the Sharpe
    being tested. Units matter here and getting them wrong is the easy mistake:
    a daily Sharpe of 0.06 is an annualised Sharpe of ~0.95, and comparing one
    against a threshold computed from the other is meaningless.

    The standard extreme-value approximation:

    .. math::
        E[\\max SR] \\approx \\sigma\\left[(1-\\gamma)\\,Z^{-1}\\!\\left(1-\\tfrac1N\\right)
        + \\gamma\\,Z^{-1}\\!\\left(1-\\tfrac1{Ne}\\right)\\right]

    Test 50 things that are all noise and the best will still show a Sharpe
    around 0.6 per unit of trial dispersion. That is the bar a real result has
    to clear, not zero.
    """
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.0
    sigma = float(np.sqrt(max(variance_of_trials, 0.0)))
    a = sps.norm.ppf(1 - 1.0 / n)
    b = sps.norm.ppf(1 - 1.0 / (n * np.e))
    return float(sigma * ((1 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe_ratio(
    returns,
    n_trials: int,
    variance_of_trials: float | None = None,
    trial_sharpes=None,
    trial_sharpes_are_annual: bool = True,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """PSR against the Sharpe the best of ``n_trials`` would reach by luck.

    Parameters
    ----------
    trial_sharpes
        Sharpe ratios of every specification tried. Annualised by default, since
        that is how people record them; converted internally to the per-period
        units the PSR works in.
    variance_of_trials
        Supply directly instead of ``trial_sharpes``, in **per-period** units.

    One of the two is required. There is deliberately no default: a made-up
    trial variance produces a made-up deflated Sharpe, and silently guessing
    would defeat the point of the correction.
    """
    if trial_sharpes is not None:
        s = np.asarray(trial_sharpes, dtype="float64")
        s = s[np.isfinite(s)]
        if trial_sharpes_are_annual:
            s = s / np.sqrt(periods_per_year)
        if s.size > 1:
            variance_of_trials = float(s.var(ddof=1))
        n_trials = max(n_trials, int(s.size))
    if variance_of_trials is None:
        if n_trials <= 1:
            variance_of_trials = 0.0
        else:
            raise ValueError(
                "deflated_sharpe_ratio needs either trial_sharpes or an explicit "
                "variance_of_trials (in per-period units). Guessing a value would "
                "produce a deflated Sharpe that means nothing -- record the trials "
                "you actually ran with TrialsLog instead."
            )

    threshold = expected_max_sharpe(n_trials, variance_of_trials)
    out = probabilistic_sharpe_ratio(returns, threshold, periods_per_year)
    out.update(
        {
            "dsr": out.pop("psr"),
            "n_trials": int(n_trials),
            "variance_of_trials": float(variance_of_trials),
            "expected_max_sharpe": threshold,
        }
    )
    out["verdict"] = _dsr_verdict(out)
    return out


def _dsr_verdict(res: dict) -> str:
    dsr, n, thresh = res.get("dsr"), res.get("n_trials"), res.get("expected_max_sharpe")
    if not np.isfinite(dsr):
        return "Inconclusive."
    got = res.get("sharpe", np.nan)
    if dsr >= 0.95:
        return (
            f"Survives {n} trials: deflated Sharpe {dsr:.3f}. The observed Sharpe of {got:.3f} "
            f"clears the {thresh:.3f} that the best of {n} noise strategies would reach."
        )
    if dsr >= 0.80:
        return (
            f"Marginal: deflated Sharpe {dsr:.3f} after {n} trials. Suggestive, not established."
        )
    return (
        f"Does not survive multiple testing: deflated Sharpe {dsr:.3f}. With {n} trials, luck "
        f"alone would be expected to produce a Sharpe of {thresh:.3f} against the {got:.3f} "
        f"observed."
    )


def min_track_record_length(
    returns, benchmark_sharpe: float = 0.0, confidence: float = 0.95
) -> float:
    """Observations needed before this Sharpe could be called significant.

    Routinely larger than the sample actually held, which is the useful part.
    """
    sr, skew, kurt, n = _moments(returns)
    if not np.isfinite(sr) or sr <= benchmark_sharpe:
        return np.inf
    z = sps.norm.ppf(confidence)
    variance_term = 1 - skew * sr + ((kurt - 1) / 4.0) * sr**2
    return float(1 + variance_term * (z / (sr - benchmark_sharpe)) ** 2)


@dataclass
class Trial:
    """One specification that was actually evaluated."""

    name: str
    #: Annualised, because that is how a Sharpe is normally quoted and recorded.
    sharpe_annual: float | None = None
    ic: float | None = None
    notes: str = ""
    params: dict = field(default_factory=dict)
    recorded_at: str = ""


@dataclass
class TrialsLog:
    """A machine-readable record of every specification tried.

    The deflated Sharpe ratio is only as honest as its trial count, and a count
    reconstructed from memory after the fact is not a count. Append to this as
    you work, commit it, and let the number be a fact about the repository.
    """

    trials: list[Trial] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> TrialsLog:
        p = Path(path)
        if not p.exists():
            return cls(path=p)
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls(trials=[Trial(**t) for t in raw.get("trials", [])], path=p)

    def record(self, name: str, **kwargs) -> TrialsLog:
        kwargs.setdefault("recorded_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.trials.append(Trial(name=name, **kwargs))
        return self

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path or self.path or "reports/trials.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"n_trials": len(self.trials), "trials": [asdict(t) for t in self.trials]},
                       indent=2),
            encoding="utf-8",
        )
        return p

    @property
    def n(self) -> int:
        return len(self.trials)

    def sharpes(self) -> np.ndarray:
        """Annualised Sharpe of every trial that recorded one."""
        return np.array(
            [t.sharpe_annual for t in self.trials if t.sharpe_annual is not None], dtype="float64"
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.trials])

    def deflate(
        self,
        returns,
        assumed_sharpe_dispersion: float | None = None,
        periods_per_year: int = TRADING_DAYS,
        min_recorded: int = 3,
    ) -> dict:
        """Deflated Sharpe for ``returns`` against this log's trial count.

        The trial *count* is usually easy to state honestly; the *dispersion* of
        Sharpe ratios across trials is not, because abandoned specifications
        rarely get a Sharpe recorded before being abandoned. Estimating a
        variance from two numbers would produce a spuriously tight threshold and
        a flattering deflated Sharpe -- exactly the failure this correction is
        meant to prevent.

        So: with at least ``min_recorded`` recorded Sharpes, dispersion is
        estimated from them. Otherwise no deflation is applied and the result
        says so, unless the caller supplies ``assumed_sharpe_dispersion``
        (annualised) as a stated assumption.
        """
        s = self.sharpes()
        if s.size >= min_recorded:
            return deflated_sharpe_ratio(
                returns, n_trials=max(self.n, 1), trial_sharpes=s,
                periods_per_year=periods_per_year,
            )
        if assumed_sharpe_dispersion is not None:
            var = (assumed_sharpe_dispersion / np.sqrt(periods_per_year)) ** 2
            out = deflated_sharpe_ratio(
                returns, n_trials=max(self.n, 1), variance_of_trials=var,
                periods_per_year=periods_per_year,
            )
            out["dispersion_assumed"] = float(assumed_sharpe_dispersion)
            out["verdict"] = (
                f"{out['verdict']} Trial dispersion was ASSUMED at "
                f"{assumed_sharpe_dispersion:.2f} annualised Sharpe, not measured -- only "
                f"{s.size} of {self.n} logged trials carry a Sharpe."
            )
            return out

        out = probabilistic_sharpe_ratio(returns, 0.0, periods_per_year)
        out["dsr"] = out.pop("psr")
        out.update(
            {
                "n_trials": self.n,
                "variance_of_trials": None,
                "expected_max_sharpe": 0.0,
                "not_deflated": True,
                "verdict": (
                    f"NOT DEFLATED. {self.n} specifications are logged but only {s.size} carry a "
                    f"Sharpe ratio, which is too few to estimate the dispersion across trials. "
                    f"The figure shown is the undeflated probabilistic Sharpe ratio and is "
                    f"therefore optimistic. Record a Sharpe for every specification you evaluate "
                    f"to make this meaningful."
                ),
            }
        )
        return out

    def deflate_sensitivity(
        self,
        returns,
        dispersions=(0.25, 0.5, 1.0, 1.5),
        periods_per_year: int = TRADING_DAYS,
    ) -> pd.DataFrame:
        """Deflated Sharpe across a range of assumed trial dispersions.

        More useful than a single deflated number when the dispersion cannot be
        measured: it converts an unverifiable assumption into a statement about
        how wrong you would have to be for the conclusion to flip.
        """
        rows = []
        for d in dispersions:
            var = (d / np.sqrt(periods_per_year)) ** 2
            res = deflated_sharpe_ratio(
                returns, n_trials=max(self.n, 1), variance_of_trials=var,
                periods_per_year=periods_per_year,
            )
            rows.append(
                {
                    "assumed_sharpe_dispersion": d,
                    "hurdle_sharpe_annual": res["expected_max_sharpe"] * np.sqrt(periods_per_year),
                    "deflated_sharpe": res["dsr"],
                }
            )
        return pd.DataFrame(rows)
