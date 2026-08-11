"""Estimation-window regressions, vectorised across events.

The market model
----------------
For event *i* we estimate, over an estimation window that ends well before the
announcement (default ``[t0-250, t0-31]``)::

    r_it = alpha_i + beta_i * r_mt + e_it

and then define the abnormal return in the event window as the realised return
minus the model's prediction. The 30-session gap between the end of the
estimation window and the event is not decoration: leakage and pre-announcement
run-up would otherwise contaminate the very parameters used to define "normal".

Two outputs matter downstream:

* ``alpha``/``beta`` -- used to build the counterfactual return;
* ``sigma`` -- the residual standard deviation, which is what makes the
  Boehmer-Musumeci-Poulsen standardised test possible. Event studies violate
  the constant-variance assumption badly (variance *jumps* on announcement
  days), and the naive cross-sectional t-test over-rejects as a result.

Everything is done with numpy gathers rather than a Python loop over events, so
a 50k-event study fits comfortably in memory and runs in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketModelFit:
    """Per-event OLS results. All arrays are aligned to the event ordering."""

    alpha: np.ndarray
    beta: np.ndarray
    sigma: np.ndarray
    n_obs: np.ndarray
    r_squared: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.alpha) & np.isfinite(self.beta) & (self.sigma > 0)


def _gather(matrix: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Gather ``matrix[rows, cols]`` with out-of-range positions as NaN.

    ``rows`` is (n_events, window) and ``cols`` is (n_events,).
    """
    n_t = matrix.shape[0]
    safe = np.clip(rows, 0, n_t - 1)
    out = matrix[safe, cols[:, None]]
    return np.where((rows >= 0) & (rows < n_t), out, np.nan)


def fit_market_model(
    returns: np.ndarray,
    market: np.ndarray,
    event_pos: np.ndarray,
    asset_idx: np.ndarray,
    start_offset: int = -250,
    end_offset: int = -31,
    min_obs: int = 100,
) -> MarketModelFit:
    """Estimate ``alpha``, ``beta`` and residual ``sigma`` for every event.

    Parameters
    ----------
    returns
        ``(n_sessions, n_assets)`` simple daily returns; NaN where missing.
    market
        ``(n_sessions,)`` market returns.
    event_pos
        ``(n_events,)`` session index of each event's ``t0``.
    asset_idx
        ``(n_events,)`` column index of the event's asset.
    """
    if end_offset < start_offset:
        raise ValueError("end_offset must be >= start_offset")
    offsets = np.arange(start_offset, end_offset + 1)
    rows = event_pos[:, None] + offsets[None, :]

    y = _gather(returns, rows, asset_idx)
    n_t = market.shape[0]
    safe_rows = np.clip(rows, 0, n_t - 1)
    x = np.where((rows >= 0) & (rows < n_t), market[safe_rows], np.nan)

    mask = np.isfinite(y) & np.isfinite(x)
    n = mask.sum(axis=1)

    y0 = np.where(mask, y, 0.0)
    x0 = np.where(mask, x, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_f = n.astype(float)
        mean_x = x0.sum(axis=1) / n_f
        mean_y = y0.sum(axis=1) / n_f
        dx = np.where(mask, x - mean_x[:, None], 0.0)
        dy = np.where(mask, y - mean_y[:, None], 0.0)
        sxx = (dx * dx).sum(axis=1)
        sxy = (dx * dy).sum(axis=1)
        syy = (dy * dy).sum(axis=1)
        beta = sxy / sxx
        alpha = mean_y - beta * mean_x
        ss_res = syy - beta * sxy
        dof = n_f - 2.0
        sigma = np.sqrt(np.clip(ss_res, 0.0, None) / dof)
        r2 = 1.0 - ss_res / syy

    bad = (n < min_obs) | ~np.isfinite(beta) | (sxx <= 0)
    alpha = np.where(bad, np.nan, alpha)
    beta = np.where(bad, np.nan, beta)
    sigma = np.where(bad, np.nan, sigma)
    r2 = np.where(bad, np.nan, r2)
    return MarketModelFit(alpha=alpha, beta=beta, sigma=sigma, n_obs=n, r_squared=r2)


def estimation_market_moments(
    market: np.ndarray,
    event_pos: np.ndarray,
    start_offset: int,
    end_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sum of squared deviations of the market over each estimation window.

    Needed for the exact BMP/Patell prediction-error variance inflation term.
    """
    offsets = np.arange(start_offset, end_offset + 1)
    rows = event_pos[:, None] + offsets[None, :]
    n_t = market.shape[0]
    safe = np.clip(rows, 0, n_t - 1)
    x = np.where((rows >= 0) & (rows < n_t), market[safe], np.nan)
    mask = np.isfinite(x)
    n = mask.sum(axis=1).astype(float)
    x0 = np.where(mask, x, 0.0)
    mean_x = x0.sum(axis=1) / n
    dx = np.where(mask, x - mean_x[:, None], 0.0)
    return mean_x, (dx * dx).sum(axis=1)
