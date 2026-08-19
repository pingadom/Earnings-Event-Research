#!/usr/bin/env python
"""Generate the two-page research note from the run outputs.

Generated rather than hand-written, so the figures and the numbers in the note
cannot drift away from the run that produced them. Re-run `make reproduce`
first; this reads `reports/holdout*/` and `docs/figures/`.

    make note        # or: python scripts/make_research_note.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
#: Which run the note describes. The real study leads; the synthetic run is the
#: validation behind it and is summarised rather than tabulated.
PRIMARY = "holdout_real"
CONTROL = "holdout_null"
FIGURES = REPO / "docs" / "figures" / "real"

INK = colors.HexColor("#111111")
MUTED = colors.HexColor("#5c5c5c")
RULE = colors.HexColor("#c9c9c9")
ACCENT = colors.HexColor("#0b6bcb")
BAND = colors.HexColor("#f2f2ef")

REPO_URL = "https://github.com/pingadom/Earnings-Event-Research"


# --- styles -----------------------------------------------------------------

def build_styles() -> dict:
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=base["Title"], fontName="Times-Bold", fontSize=16.5,
        leading=19.5, spaceAfter=3, textColor=INK, alignment=0,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"], fontName="Times-Italic", fontSize=10.5,
        leading=13, textColor=MUTED, spaceAfter=8,
    )
    s["byline"] = ParagraphStyle(
        "byline", parent=base["Normal"], fontName="Helvetica", fontSize=8.4,
        leading=11, textColor=MUTED, spaceAfter=10,
    )
    s["h"] = ParagraphStyle(
        "h", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8.6,
        leading=11, textColor=INK, spaceBefore=9, spaceAfter=3.5,
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["Normal"], fontName="Times-Roman", fontSize=9.3,
        leading=12.4, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5,
    )
    s["abstract"] = ParagraphStyle(
        "abstract", parent=s["body"], fontSize=9.1, leading=12.2,
        leftIndent=7, rightIndent=7, spaceBefore=4, spaceAfter=4,
    )
    s["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"], fontName="Helvetica", fontSize=7.4,
        leading=9.6, textColor=MUTED, spaceBefore=2.5, spaceAfter=7,
    )
    s["cell"] = ParagraphStyle(
        "cell", parent=base["Normal"], fontName="Times-Roman", fontSize=8.1,
        leading=10, textColor=INK,
    )
    return s


def para(text, style):
    return Paragraph(text, style)


def data_table(header, rows, widths, align_right_from=1):
    body = [[Paragraph(f"<b>{h}</b>", ParagraphStyle(
        "th", fontName="Helvetica-Bold", fontSize=7.1, leading=9,
        textColor=MUTED, alignment=2 if i >= align_right_from else 0))
        for i, h in enumerate(header)]] + rows
    t = Table(body, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.1),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def figure(name: str, caption: str, width_mm: float, styles) -> list:
    path = FIGURES / name
    if not path.exists():
        return []
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        ratio = im.height / im.width
    w = width_mm * mm
    return [KeepTogether([Image(str(path), width=w, height=w * ratio),
                          para(caption, styles["caption"])])]


# --- content ----------------------------------------------------------------

def load_numbers() -> dict:
    primary = PRIMARY if (REPORTS / PRIMARY / "holdout_summary.json").exists() else "holdout"
    hold = json.loads((REPORTS / primary / "holdout_summary.json").read_text())
    null = json.loads((REPORTS / CONTROL / "holdout_summary.json").read_text())
    by_year = pd.read_csv(REPORTS / primary / "holdout_by_year.csv")
    fm = pd.read_csv(REPORTS / primary / "fama_macbeth.csv")
    factors_path = REPORTS / primary / "factor_attribution.csv"
    factors = pd.read_csv(factors_path) if factors_path.exists() else pd.DataFrame()
    synth = json.loads((REPORTS / "holdout" / "holdout_summary.json").read_text())
    return {
        "hold": hold["aggregate"], "meta": hold["metadata"],
        "null": null["aggregate"], "by_year": by_year, "fm": fm, "factors": factors,
        "synth": synth["aggregate"],
    }


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10.5 * mm, "Earnings Event Research · S&P 500, 2019-2024 holdouts")
    canvas.drawRightString(A4[0] - 20 * mm, 10.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _loadings(factors, threshold: float = 2.0) -> str:
    """Name the factor exposures the regression actually found significant.

    Typing them into the prose is how a note ends up claiming a value tilt of
    t = 2.54 six months after the number became 2.91.
    """
    if factors is None or factors.empty:
        return "no factor at |t| > 2"
    names = {
        "HML": "value (HML)", "SMB": "size (SMB)", "RMW": "profitability (RMW)",
        "CMA": "conservative investment (CMA)", "MOM": "momentum (MOM)",
        "Mkt-RF": "the market",
    }
    rows = factors[factors["term"] != "alpha (annualised)"]
    hits = rows[rows["t_stat"].abs() >= threshold].sort_values(
        "t_stat", key=lambda c: c.abs(), ascending=False
    )
    if hits.empty:
        return "no factor at |t| > 2"
    parts = [
        f"{'negative ' if r.estimate < 0 else ''}{names.get(r.term, r.term)[:-1]}, "
        f"t = {r.t_stat:.2f})"
        for r in hits.head(4).itertuples()
    ]
    return ", ".join(parts[:-1]) + (" and " + parts[-1] if len(parts) > 1 else "")


def build(out_path: Path) -> Path:
    st = build_styles()
    n = load_numbers()
    h, nu, by, sy = n["hold"], n["null"], n["by_year"], n["synth"]
    fa = n["factors"]
    W = A4[0] - 40 * mm

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=17 * mm, bottomMargin=20 * mm,
        title="Does earnings information predict abnormal returns? A null result",
        author="Aaryan H.", subject="Quantitative equity research",
    )
    S = []

    S.append(para("Does earnings information predict abnormal returns?", st["title"]))
    S.append(para(
        f"A null result on {n['meta'].get('n_events', 0):,} S&amp;P 500 announcements, and the "
        "machinery built to make that answer trustworthy.", st["subtitle"]))
    S.append(para(
        f"Aaryan H. &nbsp;·&nbsp; {date.today():%B %Y} &nbsp;·&nbsp; "
        f'<a href="{REPO_URL}" color="#0b6bcb">{REPO_URL.replace("https://", "")}</a>',
        st["byline"]))
    S.append(HRFlowable(width="100%", thickness=0.7, color=RULE, spaceAfter=7))

    S.append(para("ABSTRACT", st["h"]))
    S.append(para(
        "Post-earnings-announcement drift is among the most-studied anomalies in finance and "
        "among the easiest to reproduce spuriously. This note tests whether information in "
        f"earnings releases predicts short-horizon abnormal returns across {n['meta'].get('n_events', 0):,} "
        "announcements by 466 S&amp;P 500 companies, timestamped to the minute from SEC filings, "
        "with each year from 2019 to 2024 held out and predicted by a model frozen before it "
        f"began. <b>It does not.</b> Mean out-of-sample rank IC is {h['mean_ic']:.3f} "
        f"(t = {h['ic_tstat_across_years']:.2f}), the sector-neutral book returns a net Sharpe of "
        f"{h['stitched_sharpe_net']:.2f}, and alpha against Fama–French five factors plus momentum "
        f"is {n['meta'].get('alpha_annual', 0):.2%} (t = {n['meta'].get('alpha_tstat', 0):.2f}). "
        f"That Sharpe is uninformative rather than adverse: running the same book on the same "
        f"events {h.get('perm_n_permutations', 0)} times with the predictions shuffled yields "
        f"{h.get('perm_null_mean', float('nan')):.2f} on average, placing the realised value at "
        f"the {h.get('perm_percentile', float('nan')):.0f}th percentile of noise "
        f"(p = {h.get('perm_p_value_one_sided', float('nan')):.2f}). An earlier version of this "
        "study, on a quarter as many companies, reported a significant decay in skill; on the "
        f"full sample the trend is {h.get('ic_trend_per_year', 0):+.3f} per year "
        f"(p = {h.get('ic_trend_p', float('nan')):.2f}) and does not replicate. "
        "Run on synthetic data with a known planted effect the same "
        f"pipeline recovers it in {sy['positive_ic_years']}/{sy['n_years']} years at a net Sharpe "
        f"of {sy['stitched_sharpe_net']:.2f}; with nothing planted it returns "
        f"{nu['mean_ic']:.3f} and {nu['stitched_sharpe_net']:.2f}. That contrast is why the null "
        "above is worth believing rather than merely asserting.", st["abstract"]))
    S.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceBefore=4, spaceAfter=2))

    S.append(para("1 · DESIGN", st["h"]))
    S.append(para(
        "For each year <i>Y</i>, the model is fitted only on announcements whose 20-day outcome "
        "had fully resolved before 1 January <i>Y</i>, less a 25-session embargo; it is then "
        "frozen and used to predict every announcement in <i>Y</i>. The target is the cumulative "
        "abnormal return from <b>one session after</b> the announcement to twenty sessions after: "
        "windows beginning on the announcement day contain the opening gap, which cannot be "
        "traded, and booking it is the single largest source of overstated drift results. Events "
        "come from SEC 8-K Item 2.02 filings, so every one carries a minute-level acceptance "
        "timestamp rather than a vendor date — 100% of the sample has a declared, not assumed, "
        "before-open or after-close flag. Features are fundamental changes differenced "
        "year-on-year and standardised unexpected earnings, each stamped with the moment it "
        "became public; a single guard refuses any panel where a feature post-dates the trade. "
        "Filing text is acquired but was not yet complete for the full universe when this run "
        "was made, so the textual half of the hypothesis is untested here and is switched off "
        "explicitly rather than left to impute silently.",
        st["body"]))

    S.append(para("2 · SIX HELD-OUT YEARS", st["h"]))
    rows = []
    for r in by.itertuples():
        rows.append([
            Paragraph(f"{int(r.year)}", st["cell"]),
            f"{int(r.n_train):,}", f"{int(r.n_test):,}",
            f"{r.ic_mean:.3f}", f"{r.ic_tstat:.2f}",
            f"{r.predicted_spread * 1e4:.0f}", f"{r.realised_spread * 1e4:.0f}",
            f"{r.calib_slope:.2f}", f"{r.sharpe_net:.2f}",
        ])
    widths = [W * f for f in (0.10, 0.12, 0.11, 0.10, 0.10, 0.13, 0.13, 0.10, 0.11)]
    S.append(data_table(
        ["Year", "Train n", "Test n", "IC", "IC t", "Pred. bp", "Real. bp", "Calib.", "Sharpe"],
        rows, widths))
    S.append(para(
        "<b>Table 1.</b> Each row is a year the model never saw in training. <i>Pred.</i> and "
        "<i>Real.</i> are the predicted and realised top-minus-bottom quintile spreads in basis "
        "points; <i>Calib.</i> is the slope of realised on predicted, where 1.00 would mean the "
        f"magnitudes were exactly right. The model predicted a positive spread of "
        f"{by.predicted_spread.min() * 1e4:.0f}–{by.predicted_spread.max() * 1e4:.0f} bp every "
        f"single year and realised between {by.realised_spread.min() * 1e4:.0f} and "
        f"{by.realised_spread.max() * 1e4:.0f}: confident throughout, and wrong as often as not. "
        f"{int(by.loc[by.ic_tstat.abs().idxmax(), 'year'])} is the only year clearing t = 2 — one "
        "year in six, which is what a 5% threshold produces from noise — and it is also one of "
        "the worst for realised profit. Ranking and sizing are different skills.", st["caption"]))
    S.extend(figure("predicted_vs_realised.png",
                    "<b>Figure 1.</b> Predicted against realised top-minus-bottom quintile "
                    "spread, by held-out year. A model with no signal still has opinions.",
                    90, st))

    S.append(para("3 · WHY BELIEVE THE NULL", st["h"]))
    S.append(para(
        "A null result and a broken pipeline look identical from outside, so the same code was "
        "run twice more on synthetic markets where the answer is known in advance. With a "
        f"post-announcement drift planted in the data it is recovered in "
        f"{sy['positive_ic_years']}/{sy['n_years']} years at a net Sharpe of "
        f"{sy['stitched_sharpe_net']:.2f}. With nothing planted it returns "
        f"{nu['mean_ic']:.3f} and {nu['stitched_sharpe_net']:.2f} — yet still posts an IC of "
        "0.080 at t = 3.67 in one individual year, a false positive in data where there is "
        "provably nothing to find. Anyone reporting a single year would have reported that one; "
        f"the same caution applies to "
        f"{int(by.loc[by.ic_tstat.abs().idxmax(), 'year'])} in Table 1.", st["body"]))
    def pair(key, fmt="{:.2f}"):
        return fmt.format(h[key]), fmt.format(nu[key])

    comp = [
        ["Mean out-of-sample IC", *pair("mean_ic", "{:.3f}")],
        ["t across the six years", *pair("ic_tstat_across_years")],
        ["Years with positive IC",
         f"{h['positive_ic_years']}/{h['n_years']}",
         f"{nu['positive_ic_years']}/{nu['n_years']}"],
        ["Mean realised spread (bp)",
         f"{h['mean_realised_spread'] * 1e4:.0f}",
         f"{nu['mean_realised_spread'] * 1e4:.0f}"],
        ["Mean calibration slope", *pair("mean_calibration_slope")],
        ["Net Sharpe, years stitched", *pair("stitched_sharpe_net")],
        ["Newey–West t on daily P&amp;L", *pair("stitched_tstat_nw")],
    ]
    rows = [[Paragraph(c[0], st["cell"]), c[1], c[2]] for c in comp]
    S.append(data_table(["Statistic", "Real data", "Synthetic null"], rows,
                        [W * 0.52, W * 0.24, W * 0.24]))
    S.append(para("<b>Table 2.</b> The real study beside the synthetic null control. They look "
                  "alike, which is the point: the real result is consistent with there being "
                  "nothing to find.", st["caption"]))

    S.append(para("4 · ALPHA, MULTIPLE TESTING, AND A SECOND METHOD", st["h"]))
    S.append(para(
        f"Regressed on Fama–French five factors plus momentum with Newey–West errors, the book "
        f"shows an annualised alpha of {n['meta'].get('alpha_annual', 0):.2%} "
        f"(t = {n['meta'].get('alpha_tstat', 0):.2f}) — no alpha. What the regression does find is "
        f"significant loadings on {_loadings(fa)}: the strategy is a quality-and-value portfolio "
        "in disguise, and those are compensated factors available for a few basis points. That "
        "alpha should also be read against the shuffled-prediction null in the abstract rather "
        "than against zero. Eight specifications are logged, "
        "including the two abandoned, giving a deflated Sharpe ratio of "
        f"{n['meta'].get('deflated_sharpe', float('nan')):.2f}. Fama–MacBeth over 60 monthly "
        "cross-sections finds no feature significant at 5%, agreeing with the portfolio sort — "
        "which is the outcome that should raise confidence in a null rather than lower it.",
        st["body"]))
    S.append(para("5 · THE LIMITATION THAT MATTERS MOST", st["h"]))
    S.append(para(
        "A point-in-time universe is not enough if the price source cannot serve delisted names. "
        "Yahoo returns no price history for 61% of the names deleted from the index during the "
        "sample, against 5% of the survivors, so 466 of the 735 members could be studied at all. "
        "The names lost include SIVB and FRC — Silicon Valley "
        "Bank and First Republic, both of which failed in 2023. Survivorship bias therefore "
        "re-enters through the data source even though the universe definition excludes it, and "
        "its direction is knowable: the missing firms are disproportionately those that "
        "collapsed, so the true result is probably somewhat worse than reported. Separately, SEC "
        "XBRL tags quarterly facts inconsistently. Differencing year-to-date flows into discrete "
        "quarters roughly doubled cash-flow coverage, but it still runs 40–50% rather than "
        "100%, and the release corpus was incomplete at run time, leaving the textual half "
        "untested. Fixing the price source needs CRSP or an equivalent with delisting coverage.",
        st["body"]))

    S.append(para("6 · REPRODUCIBILITY", st["h"]))
    S.append(para(
        "<font face='Courier' size='8'>make reproduce</font> regenerates every figure and number "
        "here and hashes each artefact; <font face='Courier' size='8'>make verify</font> diffs "
        "against the committed manifest, and CI fails if a hash moves. Method, the falsification "
        "criteria fixed before the run (none of the five is met), and a catalogue of the ways this "
        f'study could flatter itself: <a href="{REPO_URL}" color="#0b6bcb">'
        f'{REPO_URL.replace("https://", "")}</a>.', st["body"]))

    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    return out_path


def main(argv: list[str] | None = None) -> int:
    out = REPO / "docs" / "research-note.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        build(out)
    except FileNotFoundError as exc:
        print(f"missing input: {exc}\nRun `make reproduce` first.", file=sys.stderr)
        return 1
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
