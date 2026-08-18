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


# --- holdout figures --------------------------------------------------------


def plot_holdout_ic(by_year: pd.DataFrame, path: str | Path | None = None,
                    title: str = "Out-of-sample rank IC by held-out year"):
    """One bar per held-out year, with the single-feature baseline behind it."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    years = by_year["year"].to_numpy()
    ic = by_year["ic_mean"].to_numpy()
    ax.axhline(0, color=MUTED, linewidth=1)
    if "baseline_ic" in by_year.columns:
        ax.bar(years, by_year["baseline_ic"], width=0.62, color=GRID,
               edgecolor=MUTED, linewidth=0.6, label="Earnings surprise alone", zorder=1)
    ax.bar(years, ic, width=0.38, color=np.where(ic >= 0, ACCENT, NEGATIVE),
           label="Model", zorder=2)
    for x, y, t in zip(years, ic, by_year["ic_tstat"], strict=False):
        if np.isfinite(t):
            ax.annotate(f"t={t:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 5 if y >= 0 else -13), ha="center",
                        fontsize=8, color=MUTED)
    _style(ax, title, "", "Spearman rank IC")
    ax.set_xticks(years)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    return _save(fig, path) or (fig, ax)


def plot_predicted_vs_realised(by_year: pd.DataFrame, path: str | Path | None = None,
                               title: str = "Predicted vs realised quintile spread, by year"):
    """The comparison the whole exercise is for: what the model said, then what happened."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    years = by_year["year"].to_numpy()
    width = 0.36
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.bar(years - width / 2, by_year["predicted_spread"] * 1e4, width,
           color=COOL, label="Predicted", zorder=2)
    ax.bar(years + width / 2, by_year["realised_spread"] * 1e4, width,
           color=WARM, label="Realised", zorder=2)
    _style(ax, title, "", "Top-minus-bottom quintile (bp)")
    ax.set_xticks(years)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    return _save(fig, path) or (fig, ax)


def plot_calibration(predictions: pd.DataFrame, target: str,
                     path: str | Path | None = None, bins: int = 20,
                     title: str = "Calibration: predicted vs realised abnormal return"):
    """Binned predictions against realised outcomes, with the 45-degree line.

    A signal can rank correctly and still be badly calibrated. The gap between
    the fitted line and the diagonal is exactly the overconfidence you would
    otherwise size positions on.
    """
    df = predictions[["prediction", target]].dropna()
    if len(df) < bins * 5:
        raise ValueError("not enough predictions to calibrate")
    df = df.assign(bucket=pd.qcut(df["prediction"].rank(method="first"), bins, labels=False))
    grouped = df.groupby("bucket")
    x = grouped["prediction"].mean() * 1e4
    y = grouped[target].mean() * 1e4
    se = (grouped[target].std() / np.sqrt(grouped[target].count())) * 1e4

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    lim = float(max(abs(x).max(), abs(y).max()) * 1.15)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=1,
            linestyle="--", label="Perfect calibration")
    ax.errorbar(x, y, yerr=1.96 * se, fmt="o", color=ACCENT, markersize=5,
                markerfacecolor="white", markeredgewidth=1.6,
                ecolor=ACCENT, elinewidth=1, capsize=0, alpha=0.9,
                label=f"{bins} prediction bins")
    fit = np.polyfit(x, y, 1)
    ax.plot([-lim, lim], np.polyval(fit, [-lim, lim]), color=WARM, linewidth=1.8,
            label=f"Fitted (slope {fit[0]:.2f})")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _style(ax, title, "Predicted abnormal return (bp)", "Realised abnormal return (bp)")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    return _save(fig, path) or (fig, ax)


def plot_coefficient_stability(coefficients: pd.DataFrame, top_n: int = 12,
                               path: str | Path | None = None,
                               title: str = "Model coefficients across refits"):
    """Does the model keep saying the same thing as it is refitted each year?

    Coefficients that flip sign between refits mean the model is chasing noise,
    even when the headline out-of-sample numbers look acceptable.
    """
    if coefficients.empty:
        raise ValueError("no coefficients (the estimator is not linear)")
    order = coefficients.abs().mean(axis=1).sort_values(ascending=False).head(top_n).index
    sub = coefficients.loc[order]

    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(order) + 1.6))
    ax.axvline(0, color=MUTED, linewidth=1)
    ypos = np.arange(len(order))[::-1]
    for j, year in enumerate(sub.columns):
        shade = 0.35 + 0.65 * (j + 1) / len(sub.columns)
        ax.scatter(sub[year], ypos, s=26, color=ACCENT, alpha=shade,
                   label=str(year) if j in (0, len(sub.columns) - 1) else None,
                   zorder=3, edgecolors="none")
    ax.scatter(sub.mean(axis=1), ypos, s=90, marker="|", color=INK, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=9)
    _style(ax, title, "Standardised coefficient", "")
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right", title="refit year")
    leg.get_title().set_color(MUTED)
    leg.get_title().set_fontsize(8)
    for t in leg.get_texts():
        t.set_color(INK)
    return _save(fig, path) or (fig, ax)
