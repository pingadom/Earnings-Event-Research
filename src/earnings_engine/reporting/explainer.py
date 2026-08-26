"""A plain-language explainer page for readers who are not quants.

The dashboard answers "what were the numbers"; this page answers "what was the
question, and did it work". It is written for someone with no finance
background -- a recruiter, a course-mate, a family member -- and it is honest
about a null result rather than dressing one up.

Design constraints, all deliberate:

* **Self-contained.** One HTML file, no network fetch, no CDN. It opens from a
  USB stick, an email attachment, or GitHub Pages, and it will still open in
  ten years.
* **Static SVG, not JavaScript.** Charts are rendered server-side into the
  document, so the page prints correctly and works with scripting disabled.
* **A table for every chart.** Anyone who cannot read the chart -- screen
  reader, colour vision deficiency, printed in greyscale -- gets the same
  numbers in a table beside it.
* **Generated, never hand-written.** Every figure on the page is read out of
  the run artefacts, so the page cannot drift away from the study it describes.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

# Colour is carried by CSS custom properties rather than literal hex, so the
# same SVG renders correctly in light and dark mode. Two *different* jobs are at
# work and they get different palettes, which is the whole point of separating
# them: the bar charts encode polarity (better or worse than zero) and use a
# diverging pair anchored on a neutral baseline; the line chart encodes identity
# (predicted against realised) and uses a categorical pair. Every pair below was
# checked for colour-vision-deficient separation against both surfaces rather
# than chosen by eye.
POS = "var(--pos)"
NEG = "var(--neg)"
SERIES_1 = "var(--s1)"
SERIES_2 = "var(--s2)"
GRID = "var(--grid)"
AXIS = "var(--ink)"
MUTED = "var(--muted)"


_TEXT_STEP = """<li><strong>Read the language</strong>
<span>The earnings press release itself is downloaded and scored for tone, uncertainty and how
much it has been rewritten since last quarter. Companies edit their boilerplate when something
has changed.</span></li>"""

#: The same step, when the corpus was not available for the run being described.
#: Describing analysis that did not happen is the easiest way for a summary page
#: to become fiction, so the page says which half of the question was tested.
_TEXT_STEP_OFF = """<li><strong>Read the language &mdash; not in this run</strong>
<span>The press release behind every announcement is downloaded and can be scored for tone,
uncertainty and how much it has been rewritten since last quarter. The corpus was still being
assembled when these results were produced, so this half of the question is untested here. It is
switched off explicitly rather than left to quietly fill in blanks.</span></li>"""

_TEXT_CAVEAT = """<p><strong>Half the question is untested.</strong> The original hypothesis was
about the numbers <em>and</em> the language. These results cover the numbers only, because the
release corpus was incomplete when the study was run. Whether management's choice of words adds
anything the accounts do not is still open.</p>
"""


#: Headline figures from the real-data study, quoted by the synthetic-data
#: banner so a reader who lands on a demonstration run is told what the actual
#: answer was in the same breath. These are the only hand-copied numbers on the
#: page: everything else is read out of the run that produced it. They come
#: from docs/results.md and must be updated with it.
REAL_STUDY = {
    "ic": "0.067",
    "ic_t": "3.40",
    "net_sharpe": "-0.30",
    "perm_percentile": "85th",
    "source": "docs/results.md",
}


def _synthetic_banner(metadata: dict) -> str:
    """The warning that this page is a demonstration, not a finding.

    `eee holdout` defaults to the synthetic provider, and the Pages workflow
    calls it with no data of its own -- so the published landing page is built
    from invented markets with a drift coefficient deliberately planted in
    them. Every number on it is then a measurement of the machinery, and a
    reader has no way of knowing that from the prose, which is written in the
    voice of a study reporting what it found.

    It went out that way. The site spent a week telling anyone who read it that
    a correlation of 0.197 with a t-statistic of 18 was "a real result". For a
    project whose entire argument is that numbers like those mean you have a
    bug, that is the worst available thing to get wrong, and it is exactly the
    error already corrected once in the README -- which is why the banner is
    generated here, above the answer, rather than trusted to prose further
    down where it can be skimmed past.
    """
    if metadata.get("provider") != "synthetic":
        return ""
    planted = metadata.get("synthetic_drift")
    effect = (
        "with no effect planted in it at all -- the null control, which should find nothing"
        if planted == 0
        else "with a drift effect deliberately planted in it"
    )
    return (
        '<div class="demo-banner">'
        "<strong>This page is a demonstration, not a result.</strong> "
        f"It was generated from <em>invented</em> price and earnings data {effect}. "
        "The figures below measure whether the machinery recovers an effect that was "
        "put there on purpose. They are not findings about any real company, and a "
        "number like a t-statistic of 18 is the planted signal being found, not skill."
        "<br><br>"
        "The real study, on actual S&amp;P 500 announcements, found something much "
        f"smaller: a mean information coefficient of <strong>{REAL_STUDY['ic']}</strong> "
        f"(t = {REAL_STUDY['ic_t']}) — genuine ranking skill — and a net Sharpe ratio of "
        f"<strong>{REAL_STUDY['net_sharpe']}</strong>, which is the "
        f"{REAL_STUDY['perm_percentile']} percentile of a shuffled-prediction null and "
        "therefore better than noise but not significantly so, and still a loss. "
        f"Those numbers are in <code>{REAL_STUDY['source']}</code>."
        "</div>"
    )


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float | None, places: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.{places}f}%"


def _num(value: float | None, places: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{places}f}"


@dataclass
class Verdict:
    """One pre-registered falsification test and whether the study passed it."""

    criterion: str
    threshold: str
    observed: str
    passed: bool


def _verdicts(aggregate: dict, metadata: dict) -> list[Verdict]:
    """Score the study against the criteria fixed *before* it was run.

    These thresholds were written down in ``docs/methodology.md`` in advance.
    Reporting them afterwards, unchanged, is the only thing that stops a study
    from quietly becoming whatever its results happened to support.
    """
    mean_ic = aggregate.get("mean_ic")
    return [
        Verdict(
            "Ranking skill is real",
            "mean information coefficient above 0.02 with t > 2",
            f"IC {_num(mean_ic, 3)}, t = {_num(aggregate.get('ic_tstat_across_years'))}",
            bool(mean_ic and mean_ic > 0.02 and aggregate.get("ic_tstat_across_years", 0) > 2),
        ),
        Verdict(
            "Skill is consistent",
            "positive in at least five of six holdout years",
            f"{aggregate.get('positive_ic_years')} of {aggregate.get('n_years')} years",
            aggregate.get("positive_ic_years", 0) >= 5,
        ),
        Verdict(
            "It survives trading costs",
            "net Sharpe ratio above 0.5",
            f"net Sharpe {_num(aggregate.get('stitched_sharpe_net'))}",
            aggregate.get("stitched_sharpe_net", 0) > 0.5,
        ),
        Verdict(
            "It is not just market exposure",
            "positive alpha against the Fama-French factors",
            f"alpha {_pct(metadata.get('alpha_annual'))} a year, "
            f"t = {_num(metadata.get('alpha_tstat'))}",
            metadata.get("alpha_annual", 0) > 0,
        ),
        Verdict(
            "It survives the search for it",
            "deflated Sharpe ratio above 0.95 after 8 specifications tried",
            f"deflated Sharpe {_num(metadata.get('deflated_sharpe'))}",
            metadata.get("deflated_sharpe", 0) > 0.95,
        ),
    ]


def _significance_note(t_stat: float | None) -> str:
    """One line under the t-statistic tile, on the correct side of the threshold."""
    if t_stat is None or pd.isna(t_stat):
        return "A measure of how likely the result is to be luck."
    if abs(t_stat) >= 2:
        return "You need roughly 2 before a result is unlikely to be chance. This clears it."
    return "You need roughly 2 before a result is unlikely to be chance. This is below it."


def _trend_caption(trend: float | None, p_value: float | None) -> str:
    """Describe the shape of the year-by-year bars, whatever shape they are."""
    if trend is None or p_value is None or pd.isna(trend) or pd.isna(p_value):
        return "The bars below the line are the years it pointed the wrong way."
    if p_value >= 0.10:
        return (
            "The bars below the line are the years it pointed the wrong way. There is no "
            "pattern to which years those are: good and bad years are scattered, not trending."
        )
    direction = "downward" if trend < 0 else "upward"
    return (
        "The bars below the line are the years it pointed the wrong way, and they are not "
        f"scattered at random: the series has a {direction} slope."
    )


def _trend_paragraph(trend: float | None, p_value: float | None) -> str:
    """The decay question, answered by the data rather than by the literature.

    A published anomaly is supposed to decay once it is widely traded, so it is
    tempting to report a decline whether or not one is there. This states what
    was measured either way, including when the answer is "nothing".
    """
    if trend is None or p_value is None or pd.isna(trend) or pd.isna(p_value):
        return ""
    if p_value < 0.10:
        direction = "fell" if trend < 0 else "rose"
        return (
            f"<p>Skill {direction} by <strong>{abs(trend):.3f} per year</strong>, with a p-value "
            f"of <strong>{p_value:.3f}</strong> &mdash; roughly a {p_value:.0%} chance of a trend "
            "this steep if nothing were really changing. That is what the academic literature "
            "predicts should happen once a published anomaly becomes widely traded.</p>"
        )
    return (
        "<p>A tempting story to tell here is decay: the effect was documented in 1968, published "
        "repeatedly, and should have been traded away. This sample does not support that story. "
        f"The year-on-year trend in skill is {trend:+.3f} with a p-value of {p_value:.2f} &mdash; "
        "indistinguishable from no trend at all. The good and bad years are scattered, not "
        "ordered.</p>"
        "<p>That is worth stating plainly because an earlier version of this study, run on a "
        "quarter as many companies, found a decline that looked significant. It did not survive "
        "quadrupling the sample. A pattern across six points that disappears when you add more "
        "data was never a pattern.</p>"
    )


def _answer(aggregate: dict, metadata: dict, verdicts: list[Verdict]) -> str:
    """The verdict, assembled from what the run found rather than written once.

    This paragraph is the first thing anyone reads, so it is the one most likely
    to go stale. It was written when the study reported a null; a later run
    tripled the information coefficient and the sentence "well inside what luck
    produces" was still sitting above a t-statistic of 3.4. Every clause here is
    now conditional on a number.
    """
    passed = sum(1 for v in verdicts if v.passed)
    percentile = aggregate.get("perm_percentile")
    ic = aggregate.get("mean_ic") or 0.0
    t_stat = aggregate.get("ic_tstat_across_years") or 0.0
    baseline = aggregate.get("mean_baseline_ic")

    ranks = abs(t_stat) >= 2 and ic > 0
    if ranks and passed >= 4:
        headline = "The short answer: yes, and it survives the checks."
    elif ranks:
        headline = "The short answer: it ranks, but it does not pay."
    elif passed >= 2:
        headline = "The short answer: partly, and not enough to trade."
    else:
        headline = "The short answer: no."

    # On invented data the sentence above is a statement about the plumbing, and
    # it reads as a statement about markets. Say which it is, in the sentence
    # itself, because the banner above can be scrolled past.
    if metadata.get("provider") == "synthetic":
        headline = headline.replace(
            "The short answer:", "On invented data, the short answer is:"
        )

    synthetic = metadata.get("provider") == "synthetic"
    if ranks and synthetic:
        body = (
            f"The machinery recovers the effect that was planted for it to find: a "
            f"correlation of {_num(ic, 3)} between prediction and outcome, with a "
            f"t-statistic of {_num(t_stat)}. On real markets a t-statistic that size would "
            "mean a bug rather than a discovery; here it means the pipeline works end to "
            "end, which is all this run is testing. "
        )
    elif ranks:
        body = (
            f"The model really does sort companies better than chance: a correlation of "
            f"{_num(ic, 3)} between its prediction and what actually happened, with a "
            f"t-statistic of {_num(t_stat)}. Anything above about 2 is unlikely to be luck, "
            "so this part is a real result. "
        )
        if baseline is not None and not pd.isna(baseline) and baseline < 0:
            body += (
                "The obvious shortcut — just ranking companies by how much they beat "
                f"expectations — scores {_num(baseline, 3)} over the same announcements, so the "
                "work is being done by the detail in the accounts rather than by the headline "
                "number. "
            )
    else:
        body = (
            f"Ranking companies by the model's prediction beat chance by a whisker on average "
            f"— a correlation of {_num(ic, 3)} where zero is no skill — but with a t-statistic "
            f"of {_num(t_stat)} that is well inside what luck produces. "
        )

    if percentile is not None and not pd.isna(percentile):
        body += (
            "Turning that into money is a different matter, and here the answer is no. "
            f"Shuffling the predictions at random reproduces the trading results, which puts "
            f"the real ones at the {percentile:.0f}th percentile of pure noise — a measured "
            "claim rather than a hedge."
        )
    else:
        body += "The trading results are negative after costs."
    return f"<p><strong>{headline}</strong> {body}</p>"



def _bar_path(x: float, y: float, width: float, height: float, positive: bool) -> str:
    """A bar rounded only at the data end and square where it meets the baseline.

    Rounding all four corners detaches the bar from the axis it is measured
    against; rounding only the end keeps the reading unambiguous.
    """
    radius = min(4.0, width / 2, height)
    if positive:
        return (
            f"M{x:.1f},{y + height:.1f} V{y + radius:.1f} Q{x:.1f},{y:.1f} {x + radius:.1f},{y:.1f} "
            f"H{x + width - radius:.1f} Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + radius:.1f} "
            f"V{y + height:.1f} Z"
        )
    bottom = y + height
    return (
        f"M{x:.1f},{y:.1f} V{bottom - radius:.1f} "
        f"Q{x:.1f},{bottom:.1f} {x + radius:.1f},{bottom:.1f} "
        f"H{x + width - radius:.1f} Q{x + width:.1f},{bottom:.1f} {x + width:.1f},{bottom - radius:.1f} "
        f"V{y:.1f} Z"
    )


def _bar_chart(
    labels: list[str],
    values: list[float],
    *,
    width: int = 560,
    height: int = 260,
    fmt=lambda v: f"{v:.3f}",
    title: str = "",
) -> str:
    """A zero-anchored bar chart. One axis, one series, no decoration."""
    pad_l, pad_r, pad_t, pad_b = 56, 12, 16, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    finite = [v for v in values if v is not None and not pd.isna(v)]
    if not finite:
        return ""
    top = max(max(finite), 0.0)
    bottom = min(min(finite), 0.0)
    span = (top - bottom) or 1.0
    top += span * 0.12
    bottom -= span * 0.12
    span = top - bottom

    def y_of(value: float) -> float:
        return pad_t + plot_h * (top - value) / span

    zero_y = y_of(0.0)
    step = plot_w / max(len(values), 1)
    bar_w = step * 0.56
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{_e(title)}" style="max-width:{width}px">'
    ]
    for fraction in (0.0, 0.5, 1.0):
        y = pad_t + plot_h * fraction
        value = top - span * fraction
        parts.append(
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="{MUTED}">{fmt(value)}</text>'
        )
    parts.append(
        f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" '
        f'stroke="{AXIS}" stroke-width="1.2"/>'
    )
    for index, (label, value) in enumerate(zip(labels, values, strict=False)):
        if value is None or pd.isna(value):
            continue
        x = pad_l + step * index + (step - bar_w) / 2
        y = min(y_of(value), zero_y)
        bar_h = abs(y_of(value) - zero_y)
        colour = POS if value >= 0 else NEG
        parts.append(
            f'<path d="{_bar_path(x, y, bar_w, max(bar_h, 1), value >= 0)}" fill="{colour}">'
            f"<title>{_e(label)}: {_e(fmt(value))}</title></path>"
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 12}" text-anchor="middle" '
            f'font-size="11" fill="{MUTED}">{_e(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _line_chart(
    x_labels: list[str],
    series: list[tuple[str, list[float], str]],
    *,
    width: int = 560,
    height: int = 260,
    fmt=lambda v: f"{v:.2f}",
    title: str = "",
) -> str:
    """A zero-anchored line chart. Every series shares the one axis."""
    pad_l, pad_r, pad_t, pad_b = 60, 12, 26, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    finite = [v for _n, values, _c in series for v in values if v is not None and not pd.isna(v)]
    if not finite:
        return ""
    top, bottom = max(max(finite), 0.0), min(min(finite), 0.0)
    span = (top - bottom) or 1.0
    top += span * 0.12
    bottom -= span * 0.12
    span = top - bottom
    count = max(len(x_labels) - 1, 1)

    def x_of(index: int) -> float:
        return pad_l + plot_w * index / count

    def y_of(value: float) -> float:
        return pad_t + plot_h * (top - value) / span

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{_e(title)}" style="max-width:{width}px">'
    ]
    for fraction in (0.0, 0.5, 1.0):
        y = pad_t + plot_h * fraction
        parts.append(
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="{MUTED}">{fmt(top - span * fraction)}</text>'
        )
    parts.append(
        f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y_of(0):.1f}" y2="{y_of(0):.1f}" '
        f'stroke="{AXIS}" stroke-width="1.2"/>'
    )
    for offset, (name, values, colour) in enumerate(series):
        points = " ".join(
            f"{x_of(i):.1f},{y_of(v):.1f}"
            for i, v in enumerate(values)
            if v is not None and not pd.isna(v)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-linejoin="round"/>'
        )
        swatch_x = pad_l + offset * 160
        parts.append(
            f'<rect x="{swatch_x}" y="6" width="10" height="10" rx="2" fill="{colour}"/>'
            f'<text x="{swatch_x + 15}" y="15" font-size="11.5" fill="{MUTED}">{_e(name)}</text>'
        )
        for index, value in enumerate(values):
            if value is None or pd.isna(value):
                continue
            parts.append(
                f'<circle cx="{x_of(index):.1f}" cy="{y_of(value):.1f}" r="4" fill="{colour}" '
                f'stroke="var(--surface)" stroke-width="2">'
                f"<title>{_e(name)} {_e(x_labels[index])}: {_e(fmt(value))}</title></circle>"
            )
    for index, label in enumerate(x_labels):
        parts.append(
            f'<text x="{x_of(index):.1f}" y="{height - 12}" text-anchor="middle" '
            f'font-size="11" fill="{MUTED}">{_e(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_e(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        f'<table><caption>{_e(caption)}</caption><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def build_explainer(
    summary_path: str | Path,
    by_year_path: str | Path,
    out_path: str | Path,
    *,
    dashboard_href: str = "dashboard.html",
    text_enabled: bool = True,
    repo_url: str = "https://github.com/pingadom/Earnings-Event-Research",
) -> Path:
    """Render the explainer page from a completed holdout run."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    by_year = pd.read_csv(by_year_path)
    aggregate = summary.get("aggregate", {})
    metadata = summary.get("metadata", {})
    verdicts = _verdicts(aggregate, metadata)

    years = [str(int(y)) for y in by_year["year"]]
    ic_chart = _bar_chart(
        years,
        list(by_year["ic_mean"]),
        fmt=lambda v: f"{v:.02f}",
        title="Information coefficient by holdout year",
    )
    ic_table = _table(
        ["Year", "Correlation", "Events tested", "Was it positive?"],
        [
            [
                str(int(row.year)),
                _num(row.ic_mean, 3),
                str(int(row.n_test)),
                "yes" if row.ic_mean > 0 else "no",
            ]
            for row in by_year.itertuples()
        ],
        "Predictive correlation in each year the model had never seen.",
    )

    spread_chart = _line_chart(
        years,
        [
            ("Predicted", list(by_year["predicted_spread"]), SERIES_1),
            ("Actually realised", list(by_year["realised_spread"]), SERIES_2),
        ],
        fmt=lambda v: f"{v * 100:.1f}%",
        title="Predicted versus realised spread by year",
    )
    spread_table = _table(
        ["Year", "Predicted gap", "Realised gap", "Shortfall"],
        [
            [
                str(int(row.year)),
                _pct(row.predicted_spread),
                _pct(row.realised_spread),
                _pct(row.realised_spread - row.predicted_spread),
            ]
            for row in by_year.itertuples()
        ],
        "The model's forecast gap between its best and worst picks, against what happened.",
    )

    returns_chart = _bar_chart(
        years,
        list(by_year["ann_return_net"]),
        fmt=lambda v: f"{v * 100:.1f}%",
        title="Net return by year after costs",
    )
    returns_table = _table(
        ["Year", "Before costs", "After costs", "Cost drag"],
        [
            [
                str(int(row.year)),
                _pct(row.ann_return_gross),
                _pct(row.ann_return_net),
                _pct(row.ann_return_net - row.ann_return_gross),
            ]
            for row in by_year.itertuples()
        ],
        "Annualised return of the long-short portfolio in each holdout year.",
    )

    verdict_rows = "".join(
        f'<tr class="{"pass" if v.passed else "fail"}">'
        f"<td><strong>{_e(v.criterion)}</strong></td><td>{_e(v.threshold)}</td>"
        f"<td>{_e(v.observed)}</td>"
        f'<td class="mark">{"PASS" if v.passed else "FAIL"}</td></tr>'
        for v in verdicts
    )
    passed = sum(1 for v in verdicts if v.passed)

    trend = aggregate.get("ic_trend_per_year")
    trend_p = aggregate.get("ic_trend_p")
    over_promised = int((by_year["predicted_spread"] > by_year["realised_spread"]).sum())
    profitable = int((by_year["ann_return_net"] > 0).sum())
    cost_drag = float((by_year["ann_return_gross"] - by_year["ann_return_net"]).mean())

    context = {
        "over_promised": over_promised,
        "over_promised_words": "every single year" if over_promised == len(by_year) else
        f"{over_promised} of the {len(by_year)} years",
        "profitable_years": profitable,
        "losing_years": len(by_year) - profitable,
        "cost_drag": _pct(cost_drag, 2),
        "trend_caption": _trend_caption(trend, trend_p),
        "trend_paragraph": _trend_paragraph(trend, trend_p),
        "answer": _answer(aggregate, metadata, verdicts),
        "ic_t_note": _significance_note(aggregate.get("ic_tstat_across_years")),
        "perm_mean": _num(aggregate.get("perm_null_mean")),
        "perm_pct": _num(aggregate.get("perm_percentile"), 0),
        "perm_p": _num(aggregate.get("perm_p_value_one_sided"), 3),
        "perm_n": aggregate.get("perm_n_permutations", 0),
        "perm_excess": _num(aggregate.get("perm_excess_over_null")),
        "n_events": f"{int(metadata.get('n_events', 0)):,}",
        "synthetic_banner": _synthetic_banner(metadata),
        "universe_label": (
            f"{int(metadata.get('n_tickers', 0))} invented companies"
            if metadata.get("provider") == "synthetic"
            else "S&amp;P 500"
        ),
        "n_years": aggregate.get("n_years", 0),
        "years": aggregate.get("years", ""),
        "n_predictions": f"{int(aggregate.get('n_predictions', 0)):,}",
        "start": metadata.get("start", ""),
        "end": metadata.get("end", ""),
        "mean_ic": _num(aggregate.get("mean_ic"), 3),
        "ic_t": _num(aggregate.get("ic_tstat_across_years")),
        "positive_years": aggregate.get("positive_ic_years", 0),
        "sharpe": _num(aggregate.get("stitched_sharpe_net")),
        "sharpe_t": _num(aggregate.get("stitched_tstat_nw")),
        "alpha": _pct(metadata.get("alpha_annual")),
        "alpha_t": _num(metadata.get("alpha_tstat")),
        "dsr": _num(metadata.get("deflated_sharpe")),
        "n_trials": metadata.get("n_trials", 0),
        "trend": _num(aggregate.get("ic_trend_per_year"), 3),
        "trend_p": _num(aggregate.get("ic_trend_p"), 3),
        "calib": _num(aggregate.get("mean_calibration_slope")),
        "predicted": _pct(aggregate.get("mean_predicted_spread")),
        "realised": _pct(aggregate.get("mean_realised_spread")),
        "passed": passed,
        "total": len(verdicts),
        "ic_chart": ic_chart,
        "ic_table": ic_table,
        "spread_chart": spread_chart,
        "spread_table": spread_table,
        "returns_chart": returns_chart,
        "returns_table": returns_table,
        "verdict_rows": verdict_rows,
        "text_step": _TEXT_STEP if text_enabled else _TEXT_STEP_OFF,
        "text_caveat": "" if text_enabled else _TEXT_CAVEAT,
        "dashboard_href": _e(dashboard_href),
        "repo_url": _e(repo_url),
    }
    page = _TEMPLATE
    for key, value in context.items():
        page = page.replace("{{" + key + "}}", str(value))
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    log.info("explainer written to %s", destination)
    return destination


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Does an earnings report tell you where the share price goes next?</title>
<meta name="description" content="A walk-forward study of post-earnings drift in the S&amp;P 500: what was tested, what was found, and how well it predicts.">
<style>
  :root {
    color-scheme: light;
    --plane:#f9f9f7; --surface:#ffffff; --ink:#0b0b0b; --ink-2:#3f3e3b;
    --muted:#6b6a66; --border:rgba(11,11,11,.12); --wash:rgba(42,120,214,.07);
    --grid:#e4e3dc;
    /* Categorical identity: two series in one chart. */
    --s1:#2a78d6; --s2:#eb6834;
    /* Diverging polarity: above or below a neutral zero baseline. */
    --pos:#2a78d6; --neg:#c0392b;
    --good:#006300; --bad:#c0392b;
  }
  /* Dark mode is a separate set of steps validated against the dark surface,
     not an inversion of the light one: an inverted hue loses chroma against
     black and the two series stop separating. */
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --plane:#0d0d0d; --surface:#191918; --ink:#f4f4f2; --ink-2:#c9c8c2;
      --muted:#918f88; --border:rgba(255,255,255,.13); --wash:rgba(57,135,229,.13);
      --grid:#2e2e2b;
      --s1:#3987e5; --s2:#d95926;
      --pos:#3987e5; --neg:#e66767;
      --good:#3fbf3f; --bad:#e66767;
    }
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--plane); color:var(--ink);
    font:400 17px/1.65 Georgia,"Iowan Old Style",'Times New Roman',serif;
    -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:760px; margin:0 auto; padding:56px 22px 96px; }
  .wide { max-width:940px; }
  h1 { font-size:34px; line-height:1.2; font-weight:600; letter-spacing:-.02em; margin:0 0 14px; }
  h2 {
    font-size:14px; letter-spacing:.09em; text-transform:uppercase; font-weight:700;
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
    color:var(--muted); margin:56px 0 14px; padding-bottom:8px;
    border-bottom:1px solid var(--border);
  }
  h3 { font-size:19px; font-weight:600; margin:30px 0 6px; }
  p { margin:0 0 17px; }
  .lede { font-size:20px; line-height:1.55; color:var(--ink-2); margin-bottom:26px; }
  .answer {
    background:var(--wash); border-left:3px solid var(--s1); border-radius:0 10px 10px 0;
    padding:20px 24px; margin:30px 0;
  }
  .answer p:last-child { margin-bottom:0; }
  .byline {
    font-family:system-ui,sans-serif; font-size:13.5px; color:var(--muted);
    margin:0 0 34px; padding-bottom:20px; border-bottom:1px solid var(--border);
  }
  .tiles {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr)); gap:12px; margin:28px 0;
  }
  .tile {
    background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:15px 17px;
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  }
  .tile .label {
    font-size:11.5px; letter-spacing:.05em; text-transform:uppercase; color:var(--muted);
  }
  .tile .value { font-size:26px; font-weight:600; letter-spacing:-.02em; margin:6px 0 2px; }
  .tile .note { font-size:12.5px; color:var(--ink-2); line-height:1.4; }
  figure {
    margin:26px 0; background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:20px 22px 14px;
  }
  figure svg { display:block; margin:0 auto; }
  figcaption {
    font-family:system-ui,sans-serif; font-size:13.5px; color:var(--ink-2);
    margin-top:12px; line-height:1.5;
  }
  details { margin-top:10px; }
  summary {
    cursor:pointer; font-family:system-ui,sans-serif; font-size:12.5px; color:var(--muted);
  }
  table {
    border-collapse:collapse; width:100%; margin-top:12px;
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif; font-size:13.5px;
  }
  caption { text-align:left; color:var(--muted); font-size:12.5px; padding-bottom:8px; }
  th, td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--border); }
  th:first-child, td:first-child { text-align:left; }
  thead th { color:var(--muted); font-weight:600; font-size:12px; }
  tbody tr.pass td.mark { color:var(--good); font-weight:700; }
  tbody tr.fail td.mark { color:var(--bad); font-weight:700; }
  .scorecard td { text-align:left; }
  .scorecard td.mark { text-align:right; letter-spacing:.06em; font-size:12px; }
  .steps { counter-reset:step; list-style:none; padding:0; margin:22px 0; }
  .steps li {
    counter-increment:step; position:relative; padding:0 0 20px 46px; margin:0;
    border-left:1px solid var(--border); margin-left:14px;
  }
  .steps li:last-child { border-left-color:transparent; padding-bottom:0; }
  .steps li::before {
    content:counter(step); position:absolute; left:-14px; top:-2px;
    width:28px; height:28px; border-radius:50%; background:var(--s1); color:#fff;
    font:600 13px/28px system-ui,sans-serif; text-align:center;
  }
  .steps strong { display:block; font-size:17px; }
  .steps span {
    display:block; color:var(--ink-2); font-size:15.5px; line-height:1.55; margin-top:2px;
  }
  .links { display:flex; flex-wrap:wrap; gap:10px; margin:26px 0 0; }
  .links a {
    font-family:system-ui,sans-serif; font-size:14px; text-decoration:none;
    background:var(--surface); border:1px solid var(--border); border-radius:9px;
    padding:10px 16px; color:var(--ink);
  }
  .links a:hover { border-color:var(--s1); color:var(--s1); }
  a { color:var(--s1); }
  footer {
    margin-top:64px; padding-top:22px; border-top:1px solid var(--border);
    font-family:system-ui,sans-serif; font-size:13px; color:var(--muted);
  }
  .jargon { border-bottom:1px dotted var(--muted); cursor:help; }
  /* Deliberately loud. This is the one block on the page a reader must not
     skim past, and it sits above the answer for the same reason. */
  .demo-banner {
    margin:0 0 26px; padding:18px 20px; border-radius:8px;
    border:2px solid #b45309; border-left-width:8px;
    background:rgba(180,83,9,0.10); color:var(--ink);
    font-family:system-ui,sans-serif; font-size:15px; line-height:1.55;
  }
  .demo-banner strong { color:#b45309; }
  .demo-banner code { font-size:13px; }
  @media (max-width:640px) { h1 { font-size:27px; } .lede { font-size:18px; } body { font-size:16px; } }
  @media print { body { background:#fff; } .links { display:none; } }
</style>
</head>
<body>
<div class="wrap">

<h1>Does an earnings report tell you where the share price goes next?</h1>
{{synthetic_banner}}
<p class="lede">Four times a year, every large company publishes its results. Prices move
immediately. The interesting question is what happens over the <em>following weeks</em> — and
whether the contents of the report predict it. This is a study that tried to find out, and
reported what it found rather than what would have been nicer to find.</p>
<p class="byline">{{n_events}} earnings announcements &middot; {{universe_label}} &middot;
{{start}} to {{end}} &middot; {{n_years}} years of out-of-sample testing</p>

<div class="answer">
{{answer}}
</div>

<h2>What was actually being asked</h2>
<p>When a company reports earnings, the share price reacts within seconds. That instant jump
is not what this project is about — it is untradeable by the time anyone reads the report.
What it is about is a much older and stranger observation, first documented in 1968: after the
initial jump, prices tend to keep <em>drifting</em> in the same direction for weeks. Companies
that beat expectations keep rising. Companies that miss keep falling. Academics call this
<span class="jargon" title="Post-earnings-announcement drift">post-earnings-announcement
drift</span>, and if it is real and still present, it is money lying on the floor.</p>

<p>So the question this system tests is precise:</p>
<p><em>Using only information that was genuinely public before the market opened, can we rank
companies by how much they will subsequently out- or under-perform their sector over the next
month?</em></p>

<h3>Why "out-perform its sector" and not "go up"</h3>
<p>A stock that rose 3% in a month when the whole market rose 3% has told you nothing about
its earnings report. It told you the market went up. Every return in this study is therefore an
<strong>abnormal return</strong> — what the share did minus what the rest of its sector did over
exactly the same days. That single choice removes most of the ways a study like this fools
itself.</p>

<h2>How the system works</h2>
<ol class="steps">
<li><strong>Find out when each company reported</strong>
<span>Not the date a data vendor claims, but the timestamp the regulator recorded when the
announcement was filed — down to the minute, so we know whether it landed before the opening
bell or after the close.</span></li>
<li><strong>Read the financial statements</strong>
<span>Revenue, margins, earnings per share, cash flow and debt are pulled from the company's
own regulatory filings, and converted into <em>changes</em> — this quarter against the same
quarter a year ago. A 32% profit margin means nothing on its own; a margin three points above
last year's means something.</span></li>
{{text_step}}
<li><strong>Measure what happened next</strong>
<span>For each announcement, the share's abnormal return over the following 1, 5 and 20 trading
days — starting from the <em>next</em> day's open, because the overnight jump was never
available to trade.</span></li>
<li><strong>Train on the past, test on a future it has never seen</strong>
<span>The model learns from every year up to, say, 2018, then predicts 2019 with no knowledge of
it whatsoever. Then it re-learns through 2019 and predicts 2020. Six times over. This is the
step most amateur backtests skip, and it is the step that separates a real result from a
flattering one.</span></li>
<li><strong>Trade it, honestly</strong>
<span>Buy the top-ranked names, short the bottom-ranked ones, hold for a month, stay neutral
to every sector, and charge realistic costs for the spread, commission and the price impact of
your own trading.</span></li>
</ol>

<h2>How well does it predict?</h2>
<p>Three numbers matter, and they should be read together.</p>

<div class="tiles">
  <div class="tile"><div class="label">Ranking skill</div><div class="value">{{mean_ic}}</div>
    <div class="note">Average correlation between prediction and outcome. Zero is no skill;
    0.03&ndash;0.05 is a genuinely useful signal in this field.</div></div>
  <div class="tile"><div class="label">Is that real?</div><div class="value">t = {{ic_t}}</div>
    <div class="note">{{ic_t_note}}</div></div>
  <div class="tile"><div class="label">Consistency</div><div class="value">{{positive_years}}/{{n_years}}</div>
    <div class="note">Holdout years where the model ranked better than chance.</div></div>
  <div class="tile"><div class="label">After costs</div><div class="value">{{sharpe}}</div>
    <div class="note">Net Sharpe ratio, t = {{sharpe_t}}. Negative means the strategy lost
    money net of trading.</div></div>
</div>

<figure>
{{ic_chart}}
<figcaption><strong>Ranking skill, year by year.</strong> Each bar is one year the model had
never seen. Positive means its ranking pointed the right way. {{trend_caption}}
</figcaption>
<details><summary>Show the numbers</summary>{{ic_table}}</details>
</figure>

{{trend_paragraph}}

<figure>
{{spread_chart}}
<figcaption><strong>What the model promised against what it delivered.</strong> Blue is the gap
the model forecast between its best and worst picks; orange is the gap that actually appeared.
It over-promised in {{over_promised_words}} — an average slope of {{calib}}, where a perfectly
calibrated model would score 1.0.</figcaption>
<details><summary>Show the numbers</summary>{{spread_table}}</details>
</figure>

<figure>
{{returns_chart}}
<figcaption><strong>What it would have earned.</strong> Annualised return in each holdout year
after realistic trading costs. {{profitable_years}} of {{n_years}} years made money; the other
{{losing_years}} lost it. Trading costs alone removed {{cost_drag}} a year, which is small in
absolute terms but larger than the entire average return.</figcaption>
<details><summary>Show the numbers</summary>{{returns_table}}</details>
</figure>

<h2>The scorecard</h2>
<p>Before running any of this, five conditions were written down that the strategy would have
to meet to count as working. Publishing them in advance is the only defence against the
temptation to decide afterwards that whatever you found was what you were looking for. It
met <strong>{{passed}} of {{total}}</strong>.</p>

<table class="scorecard">
<thead><tr><th>Had to show</th><th>Threshold set in advance</th><th>What happened</th><th></th></tr></thead>
<tbody>{{verdict_rows}}</tbody>
</table>

<h2>Why a negative result is still worth reading</h2>
<p>A project that reports a working money-making strategy on free data should be treated with
suspicion, because the ways of accidentally producing one are numerous and well known. This one
is built to make them hard:</p>

<h3>It cannot see the future</h3>
<p>Every input carries the timestamp at which it became public, and the system refuses to build
a feature whose timestamp is later than the moment the trade would have been placed. This is
enforced as a check that fails the run, not as a convention someone remembers to follow.</p>

<h3>It does not quietly drop the failures</h3>
<p>If you build a study on the companies in the index <em>today</em>, you have silently excluded
every company that collapsed — and collapse is exactly what a bad earnings report predicts.
Membership here is tracked over time, so a company is in the sample for the years it was
actually in the index and no longer.</p>

<h3>It counts how many times it tried</h3>
<p>Test twenty strategies and one will look excellent by luck alone. Every specification tried
in this project was logged — {{n_trials}} of them — and the headline Sharpe ratio is then
<em>deflated</em> to account for the search. The deflated figure is <strong>{{dsr}}</strong>,
where 0.95 would be the threshold for confidence. Most published backtests never do this
arithmetic, and many would not survive it.</p>

<h3>It knows what "no signal" actually looks like</h3>
<p>A long-short book is not a neutral instrument. Holding twenty-day positions,
rebalancing daily, and capping how much can sit in any one name produces a small
negative drift even when the signal driving it is pure noise. So a negative Sharpe
ratio is not automatically evidence against the hypothesis — some of it is just
what this machinery does to anything.</p>
<p>Rather than argue about how much, the study measures it. The same book is run
{{perm_n}} times on the same events with the predictions <em>shuffled</em>, which
destroys the link between a forecast and the stock it belongs to while leaving
everything else identical. Shuffled, the book scores <strong>{{perm_mean}}</strong>
on average. The real one scored {{sharpe}} — the <strong>{{perm_pct}}th
percentile</strong> of that distribution, p = {{perm_p}}. Read the headline Sharpe
against {{perm_mean}}, not against zero.</p>

<h3>It checks the returns are not something ordinary in disguise</h3>
<p>A strategy can look clever while really just holding cheap stocks, or small ones, or
whatever has been rising. Measured against the standard academic risk factors, what is left
over is <strong>{{alpha}}</strong> a year with a t-statistic of {{alpha_t}} — which is to say,
nothing distinguishable from zero.</p>

<h2>What would change the answer</h2>
<p>Three limitations are load-bearing, and all three are documented rather than buried:</p>
<p><strong>Prices for companies that no longer exist.</strong> The free price source drops most
delisted companies. Two of the sample's most dramatic failures — Silicon Valley Bank and First
Republic — are missing for exactly that reason. A professional database would settle whether
this matters, and it would most likely make the results worse, not better, since failed
companies are where bad earnings news leads somewhere.</p>
<p><strong>Sample size.</strong> {{n_predictions}} predictions across {{n_years}} years is
enough to detect a strong signal, not a weak one. A weak signal may be here and undetected.</p>
{{text_caveat}}<p><strong>Costs are modelled, not measured.</strong> The spread, commission and market impact
are estimated. Real execution could be better or worse.</p>

<h2>Look at the detail</h2>
<p>Everything above is generated from the run artefacts, so nothing here can drift away from
the study it describes. The technical dashboard has the diagnostics, factor loadings and
cross-sectional regressions; the repository has the code, the tests and the full write-up.</p>
<div class="links">
  <a href="{{dashboard_href}}">Technical dashboard</a>
  <a href="{{repo_url}}">Source code on GitHub</a>
  <a href="{{repo_url}}/blob/main/docs/how-it-works.md">How it works, from scratch</a>
  <a href="{{repo_url}}/blob/main/docs/results.md">Full results write-up</a>
  <a href="{{repo_url}}/blob/main/docs/methodology.md">Methodology</a>
  <a href="{{repo_url}}/blob/main/docs/biases.md">Known biases</a>
</div>

<footer>
<p>Built as an independent research project. Not investment advice; nothing here is a
recommendation to buy or sell anything. Data from the SEC's public filing archive and free
market-data sources — see the repository for the full provenance and the licence conditions
under which each source is used.</p>
</footer>

</div>
</body>
</html>
"""
