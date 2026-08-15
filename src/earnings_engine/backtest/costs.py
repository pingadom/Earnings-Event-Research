"""Transaction costs.

A PEAD-style signal trades on a schedule set by the earnings calendar, holds
for weeks, and turns the book over several times a year. At those horizons
costs are not a rounding error -- a gross Sharpe of 1.0 on a 20-day signal can
land under 0.4 net once you pay the spread twice and move the market on the
way in. Reporting gross-only performance is the most common way a research
backtest overstates itself.

The model
---------
One-way cost, in basis points of traded notional::

    cost = half_spread + commission + impact_coef * sqrt(participation)

The square-root term is the standard functional form for market impact (Almgren
et al.): cost grows with the square root of the fraction of daily volume you
consume, not linearly. ``participation`` defaults to 5% of ADV, which is
already assertive for a small book and conservative for a large one.

Defaults are for liquid US large caps. They are *assumptions*, not
measurements, and the right thing to do with them is a sensitivity analysis --
see :func:`cost_sensitivity`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    """One-way transaction cost in basis points of traded notional."""

    half_spread_bps: float = 3.0
    commission_bps: float = 0.5
    impact_coef_bps: float = 10.0
    participation: float = 0.05

    def one_way_bps(self, participation: float | np.ndarray | None = None) -> float | np.ndarray:
        part = self.participation if participation is None else participation
        return self.half_spread_bps + self.commission_bps + self.impact_coef_bps * np.sqrt(part)

    def cost_on(self, traded_notional, participation=None):
        """Cost as a fraction of capital, given traded notional as a fraction."""
        return np.abs(traded_notional) * self.one_way_bps(participation) * BPS

    def scaled(self, factor: float) -> CostModel:
        """A version with every component multiplied by ``factor``."""
        return replace(
            self,
            half_spread_bps=self.half_spread_bps * factor,
            commission_bps=self.commission_bps * factor,
            impact_coef_bps=self.impact_coef_bps * factor,
        )


def apply_costs(
    gross_returns: pd.Series,
    traded: pd.Series,
    model: CostModel,
    participation: pd.Series | None = None,
) -> pd.DataFrame:
    """Net a gross return series against per-period traded notional."""
    idx = gross_returns.index
    traded = traded.reindex(idx).fillna(0.0)
    part = participation.reindex(idx) if participation is not None else None
    part_arr = None if part is None else part.to_numpy()
    cost = pd.Series(model.cost_on(traded.to_numpy(), part_arr), index=idx)
    return pd.DataFrame(
        {"gross": gross_returns, "traded": traded, "cost": cost, "net": gross_returns - cost}
    )


def cost_sensitivity(
    gross_returns: pd.Series,
    traded: pd.Series,
    model: CostModel,
    factors=(0.5, 1.0, 1.5, 2.0, 3.0),
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """How much worse do the costs have to be before the edge disappears?

    This is a more useful number than a single net Sharpe, because it converts
    an unverifiable assumption into a statement about how much you would have
    to be wrong for the conclusion to flip.
    """
    rows = []
    for f in factors:
        m = model.scaled(f)
        net = apply_costs(gross_returns, traded, m)["net"]
        vol = net.std(ddof=1) * np.sqrt(periods_per_year)
        rows.append(
            {
                "cost_multiple": f,
                "one_way_bps": float(m.one_way_bps()),
                "ann_return": float(net.mean() * periods_per_year),
                "ann_vol": float(vol),
                "sharpe": float(net.mean() * periods_per_year / vol) if vol > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
