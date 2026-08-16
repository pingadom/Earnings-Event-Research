"""Figures.

House style, applied consistently so a set of figures reads as one document:

* one accent colour for the thing being measured, grey for context;
* a diverging pair (blue / orange) for quantile fans -- safe for the common
  forms of colour blindness, unlike red/green, which is the usual default for
  exactly this kind of chart and the worst possible one;
* zero lines drawn explicitly, because in an abnormal-return chart zero is the
  null hypothesis and the reader needs to see where it is;
* no chart junk, no gradients, no 3-D.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#1c1c1e"
MUTED = "#8a8a8e"
GRID = "#e5e5ea"
ACCENT = "#0b6bcb"
COOL = "#0b6bcb"
WARM = "#d1600b"
POSITIVE = "#1a7f5a"
NEGATIVE = "#b3261e"


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12, fontweight="medium")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def _save(fig, path: str | Path | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return path


def plot_event_study(
    daily: pd.DataFrame,
    ar_column: str = "ar_market_model",
    path: str | Path | None = None,
    title: str = "Average cumulative abnormal return around the announcement",
):
    """Mean CAR by relative day, with a bootstrap confidence band."""
    grp = daily.groupby("rel_day")[ar_column]
    mean = grp.mean()
    se = grp.std(ddof=1) / np.sqrt(grp.count())
    car = mean.sort_index().cumsum()
    band = np.sqrt((se.sort_index() ** 2).cumsum()) * 1.96

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
    ax.fill_between(car.index, car - band, car + band, color=ACCENT, alpha=0.15, linewidth=0)
    ax.plot(car.index, car, color=ACCENT, linewidth=2)
    _style(
        ax,
        title,
        "Trading days from announcement (t0 = first tradable session)",
        "Cumulative abnormal return",
    )
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    return _save(fig, path) or (fig, ax)


def plot_car_by_quantile(
    daily: pd.DataFrame,
    signal: pd.Series,
    ar_column: str = "ar_market_model",
    quantiles: int = 5,
    path: str | Path | None = None,
    title: str = "Cumulative abnormal return by signal quantile",
):
    """The picture the whole project is trying to produce.

    ``signal`` must be indexed by ``event_id``. If the top and bottom fans do
    not separate here, no amount of modelling downstream will rescue it.
    """
    df = daily.copy()
    df["signal"] = df["event_id"].map(signal)
    df = df.dropna(subset=["signal", ar_column])
    if df.empty:
        raise ValueError("no overlap between the signal and the abnormal returns")

    ranks = df.drop_duplicates("event_id").set_index("event_id")["signal"].rank(method="first")
    bucket = pd.qcut(ranks, quantiles, labels=False)
    df["bucket"] = df["event_id"].map(bucket)

    cmap = _diverging(quantiles)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
    for q in range(quantiles):
        sub = df[df["bucket"] == q]
        car = sub.groupby("rel_day")[ar_column].mean().sort_index().cumsum()
        label = f"Q{q + 1}" + (" (low)" if q == 0 else " (high)" if q == quantiles - 1 else "")
        ax.plot(car.index, car, color=cmap[q], linewidth=2 if q in (0, quantiles - 1) else 1.2,
                label=label, alpha=1.0 if q in (0, quantiles - 1) else 0.75)
    _style(ax, title, "Trading days from announcement", "Cumulative abnormal return")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=2)
    for text in leg.get_texts():
        text.set_color(INK)
    return _save(fig, path) or (fig, ax)


def _diverging(n: int) -> list[str]:
    """Blue -> grey -> orange. Colour-blind safe, unlike the usual red/green."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex  # noqa: PLC0415

    cmap = LinearSegmentedColormap.from_list("eee", [COOL, "#b8b8bd", WARM])
    return [to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def plot_ic_timeseries(ic: pd.DataFrame, path: str | Path | None = None,
                       title: str = "Information coefficient through time"):
    """Per-cohort rank IC with a rolling mean. Consistency is the point."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = pd.to_datetime(ic.iloc[:, 0])
    y = ic["ic"]
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.bar(x, y, width=20, color=np.where(y >= 0, POSITIVE, NEGATIVE), alpha=0.55, linewidth=0)
    if len(y) >= 6:
        ax.plot(x, y.rolling(6, min_periods=3).mean(), color=INK, linewidth=1.8,
                label="6-cohort rolling mean")
        leg = ax.legend(frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(INK)
    _style(ax, title, "", "Spearman rank IC")
    return _save(fig, path) or (fig, ax)


def plot_equity_curve(daily: pd.DataFrame, path: str | Path | None = None,
                      title: str = "Cumulative strategy return"):
    """Gross vs net. Showing only the gross line is how backtests mislead."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    gross = (1.0 + daily["gross"]).cumprod() - 1.0
    net = (1.0 + daily["net"]).cumprod() - 1.0
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.plot(gross.index, gross, color=MUTED, linewidth=1.4, linestyle="--", label="Gross")
    ax.plot(net.index, net, color=ACCENT, linewidth=2, label="Net of costs")
    _style(ax, title, "", "Cumulative return")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    return _save(fig, path) or (fig, ax)


def plot_cost_sensitivity(sens: pd.DataFrame, path: str | Path | None = None,
                          title: str = "How wrong would the cost assumption have to be?"):
    """Net Sharpe as a function of the cost multiple."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.plot(sens["cost_multiple"], sens["sharpe"], color=ACCENT, linewidth=2, marker="o",
            markersize=5, markerfacecolor="white", markeredgewidth=1.6)
    for _, row in sens.iterrows():
        ax.annotate(f"{row['one_way_bps']:.0f}bp", (row["cost_multiple"], row["sharpe"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color=MUTED)
    _style(ax, title, "Cost multiple vs the baseline assumption", "Net Sharpe ratio")
    return _save(fig, path) or (fig, ax)
