"""Self-contained interactive HTML dashboard for a holdout study.

No build step, no CDN, no external assets: the data is embedded as JSON and the
charts are drawn as inline SVG by a few hundred lines of vanilla JavaScript, so
the file can be emailed, committed, or opened from disk in ten years.

Design notes, since they are deliberate rather than incidental:

* **Two categorical hues only** (blue / orange), taken from a palette validated
  for colour-vision deficiency against both the light and dark surfaces. Every
  chart here compares exactly two things, so two slots is the whole requirement.
* **One y-axis per chart, always.** Two measures on one plot with two scales
  invents a correlation that is not in the data.
* **Every chart has a table twin.** The per-year table at the bottom carries the
  same numbers, so nothing is reachable only by hovering.
* **Dark mode is selected, not inverted** -- the dark hues are their own steps,
  chosen against the dark surface rather than flipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_CURVE_POINTS = 900


def _clean(value):
    """JSON-safe: NaN and numpy scalars are not."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not np.isfinite(f) else round(f, 8)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def _calibration_bins(predictions: pd.DataFrame, target: str, bins: int = 20) -> list[dict]:
    df = predictions[["prediction", target]].dropna()
    if len(df) < bins * 5:
        return []
    df = df.assign(bucket=pd.qcut(df["prediction"].rank(method="first"), bins, labels=False))
    g = df.groupby("bucket")
    out = pd.DataFrame(
        {
            "predicted": g["prediction"].mean() * 1e4,
            "realised": g[target].mean() * 1e4,
            "n": g[target].count(),
            "se": (g[target].std() / np.sqrt(g[target].count())) * 1e4,
        }
    ).reset_index(drop=True)
    return _records(out)


def _equity(daily: pd.DataFrame | None) -> list[dict]:
    if daily is None or daily.empty:
        return []
    df = daily[["gross", "net"]].copy()
    df["gross_cum"] = (1 + df["gross"]).cumprod() - 1
    df["net_cum"] = (1 + df["net"]).cumprod() - 1
    if len(df) > MAX_CURVE_POINTS:
        step = int(np.ceil(len(df) / MAX_CURVE_POINTS))
        df = df.iloc[::step]
    return [
        {"date": str(pd.Timestamp(i).date()), "gross": _clean(r.gross_cum), "net": _clean(r.net_cum)}
        for i, r in df.iterrows()
    ]


def build_payload(result, metadata: dict) -> dict:
    by_year = result.by_year.copy()
    return {
        "metadata": {k: _clean(v) for k, v in metadata.items()},
        "aggregate": {k: _clean(v) for k, v in result.aggregate.items()},
        "byYear": _records(by_year),
        "calibration": _calibration_bins(result.predictions, result.target),
        "equity": _equity(getattr(result.backtest, "daily", None)),
        "target": result.target,
        "baselineFeature": result.baseline_feature,
    }


def write_dashboard(path: str | Path, result, metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_payload(result, metadata), indent=None, separators=(",", ":"))
    html = _TEMPLATE.replace("__PAYLOAD__", payload)
    path.write_text(html, encoding="utf-8")
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Earnings event engine — rolling annual holdouts</title>
<style>
  :root {
    color-scheme: light;
    --plane:#f9f9f7; --surface:#fcfcfb;
    --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
    --s1:#2a78d6; --s2:#eb6834; --good:#006300; --bad:#d03b3b;
    --wash:rgba(42,120,214,.08);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --plane:#0d0d0d; --surface:#1a1a19;
      --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
      --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
      --s1:#3987e5; --s2:#d95926; --good:#0ca30c; --bad:#e66767;
      --wash:rgba(57,135,229,.14);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --s1:#3987e5; --s2:#d95926; --good:#0ca30c; --bad:#e66767;
    --wash:rgba(57,135,229,.14);
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--plane); color:var(--ink);
    font:400 15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:1180px; margin:0 auto; padding:36px 24px 72px; }
  header { display:flex; align-items:flex-start; gap:20px; flex-wrap:wrap; margin-bottom:8px; }
  h1 { font-size:22px; font-weight:600; margin:0 0 4px; letter-spacing:-.01em; }
  .sub { color:var(--ink-2); font-size:14px; margin:0; }
  .spacer { flex:1 1 auto; }
  button.toggle {
    background:var(--surface); color:var(--ink-2); border:1px solid var(--border);
    border-radius:8px; padding:7px 13px; font:inherit; font-size:13px; cursor:pointer;
  }
  button.toggle:hover { color:var(--ink); }
  .banner {
    margin:20px 0 4px; padding:13px 16px; border-radius:10px;
    background:var(--wash); border:1px solid var(--border);
    color:var(--ink-2); font-size:13.5px; line-height:1.5;
  }
  .banner strong { color:var(--ink); font-weight:600; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:12px; margin:24px 0 8px; }
  .tile {
    background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px 18px;
    display:flex; flex-direction:column; justify-content:flex-start; min-height:112px;
  }
  .tile .value { margin-top:auto; }
  .tile .label { font-size:12px; color:var(--muted); letter-spacing:.02em; text-transform:uppercase; }
  .tile .value { font-size:27px; font-weight:600; letter-spacing:-.02em; }
  .tile .note { font-size:12px; color:var(--ink-2); margin-top:3px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:16px; margin-top:16px; }
  .card {
    background:var(--surface); border:1px solid var(--border); border-radius:12px;
    padding:18px 18px 12px; min-width:0;
  }
  .card h2 { font-size:14.5px; font-weight:600; margin:0 0 2px; }
  .card p.cap { font-size:12.5px; color:var(--ink-2); margin:0 0 12px; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin:2px 0 10px; font-size:12.5px; color:var(--ink-2); }
  .legend i { width:11px; height:11px; border-radius:3px; display:inline-block; margin-right:6px; vertical-align:-1px; }
  svg { display:block; width:100%; height:auto; overflow:visible; }
  .gridline { stroke:var(--grid); stroke-width:1; }
  .axisline { stroke:var(--axis); stroke-width:1; }
  .tick { fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }
  .alab { fill:var(--muted); font-size:11.5px; }
  .dlab { fill:var(--ink-2); font-size:11px; font-variant-numeric:tabular-nums; }
  .hit { fill:transparent; cursor:crosshair; }
  table { width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
  th, td { padding:8px 10px; text-align:right; border-bottom:1px solid var(--grid); white-space:nowrap; }
  th { color:var(--muted); font-weight:500; font-size:11.5px; text-transform:uppercase; letter-spacing:.02em; text-align:right; }
  th:first-child, td:first-child { text-align:left; }
  tbody tr:hover { background:var(--wash); }
  .pos { color:var(--good); } .neg { color:var(--bad); }
  .tablewrap { overflow-x:auto; margin-top:6px; }
  #tip {
    position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
    background:var(--surface); color:var(--ink); border:1px solid var(--border);
    border-radius:9px; padding:9px 11px; font-size:12.5px; line-height:1.5;
    box-shadow:0 6px 22px rgba(0,0,0,.14); z-index:50; max-width:260px;
  }
  #tip b { font-weight:600; }
  #tip .k { color:var(--muted); }
  footer { margin-top:32px; color:var(--muted); font-size:12.5px; line-height:1.6; }
  footer code { background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:1px 5px; font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Rolling annual holdouts</h1>
      <p class="sub" id="subtitle"></p>
    </div>
    <div class="spacer"></div>
    <button class="toggle" id="themeBtn" type="button">Dark mode</button>
  </header>

  <div class="banner" id="banner" hidden></div>
  <div class="tiles" id="tiles"></div>

  <div class="grid">
    <div class="card">
      <h2>Out-of-sample rank IC by held-out year</h2>
      <p class="cap">Each bar is a year the model never saw during training. The baseline is a single feature, so the gap is what the model adds.</p>
      <div class="legend" id="legIc"></div>
      <div id="chartIc"></div>
    </div>
    <div class="card">
      <h2>Predicted vs realised quintile spread</h2>
      <p class="cap">What the model said the top-minus-bottom spread would be, against what actually happened.</p>
      <div class="legend" id="legSpread"></div>
      <div id="chartSpread"></div>
    </div>
    <div class="card">
      <h2>Calibration</h2>
      <p class="cap">Predictions binned into twenties. On the dashed line, predicted magnitudes are exactly right; a flatter fit means the ranking holds but the scale is overconfident.</p>
      <div class="legend" id="legCal"></div>
      <div id="chartCal"></div>
    </div>
    <div class="card">
      <h2>Cumulative return across all held-out years</h2>
      <p class="cap">Sector-neutral long/short book, entered one session after each announcement and held 20 sessions.</p>
      <div class="legend" id="legEq"></div>
      <div id="chartEq"></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>Per-year detail</h2>
    <p class="cap">The table twin: every number in the charts above, plus the ones that did not fit.</p>
    <div class="tablewrap"><table id="table"></table></div>
  </div>

  <footer id="footer"></footer>
</div>
<div id="tip"></div>

<script>
const DATA = __PAYLOAD__;
const NS = "http://www.w3.org/2000/svg";
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const bp  = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (v * 1e4).toFixed(d) + " bp";
const pct = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (v * 100).toFixed(d) + "%";
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function el(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
function svg(host, w, h) {
  host.innerHTML = "";
  const s = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
  host.appendChild(s);
  return s;
}
function legend(host, items) {
  host.innerHTML = items.map(i =>
    `<span><i style="background:${i.color}"></i>${i.label}</span>`).join("");
}

/* tooltip ------------------------------------------------------------- */
const tip = document.getElementById("tip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.style.opacity = 1;
  const pad = 14, r = tip.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.style.opacity = 0; };
function hoverable(node, html) {
  node.addEventListener("pointermove", e => showTip(e, html));
  node.addEventListener("pointerleave", hideTip);
  node.setAttribute("tabindex", "0");
  node.addEventListener("focus", e => {
    const b = node.getBoundingClientRect();
    showTip({ clientX: b.left + b.width / 2, clientY: b.top }, html);
  });
  node.addEventListener("blur", hideTip);
}

/* scales & axes -------------------------------------------------------- */
function ticksIn(lo, hi, count = 5) {
  return niceTicks(lo, hi, count).filter(t => t >= lo - 1e-12 && t <= hi + 1e-12);
}
function domain(vals, { padTop = 0.12, padBottom = 0.12, includeZero = true, symmetric = false } = {}) {
  const clean = vals.filter(v => v !== null && v !== undefined && Number.isFinite(v));
  let lo = Math.min(...clean), hi = Math.max(...clean);
  if (symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
  else if (includeZero) { lo = Math.min(0, lo); hi = Math.max(0, hi); }
  const span = (hi - lo) || 1;
  return [lo - span * padBottom * (lo < 0 || symmetric ? 1 : 0), hi + span * padTop];
}
function niceTicks(lo, hi, count = 5) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.floor(lo / step) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
  return out;
}
function yAxis(s, ticks, y, x0, x1, label, fmtFn) {
  ticks.forEach(t => {
    s.appendChild(el("line", { class: "gridline", x1: x0, x2: x1, y1: y(t), y2: y(t) }));
    s.appendChild(el("text", { class: "tick", x: x0 - 8, y: y(t) + 4, "text-anchor": "end" },
      fmtFn ? fmtFn(t) : t));
  });
  if (label) {
    const tx = el("text", { class: "alab", x: 12, y: 0, transform: `translate(0,0)` }, label);
    tx.setAttribute("transform", `translate(11,${(y(ticks[0]) + y(ticks[ticks.length - 1])) / 2}) rotate(-90)`);
    tx.setAttribute("text-anchor", "middle");
    s.appendChild(tx);
  }
}

/* 1. IC by year -------------------------------------------------------- */
function chartIc() {
  const d = DATA.byYear;
  const host = document.getElementById("chartIc");
  const W = 560, H = 310, m = { t: 26, r: 12, b: 40, l: 56 };
  const s = svg(host, W, H);
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const vals = d.flatMap(r => [r.ic_mean, r.baseline_ic]).filter(v => v !== null);
  const [dlo, dhi] = domain(vals, { padTop: 0.16, padBottom: 0.22 });
  const ticks = ticksIn(dlo, dhi, 5);
  const y = v => m.t + ih - (v - dlo) / (dhi - dlo) * ih;
  yAxis(s, ticks, y, m.l, m.l + iw, "Spearman rank IC", t => t.toFixed(2));

  const bw = iw / d.length;
  const c1 = cssv("--s1"), c2 = cssv("--s2");
  const modelW = bw * 0.30, baseW = bw * 0.30, gap = 2;
  d.forEach((r, i) => {
    const cx = m.l + bw * (i + 0.5);
    [[r.baseline_ic, c2, -1, "Baseline (" + DATA.baselineFeature + ")"],
     [r.ic_mean, c1, 1, "Model"]].forEach(([v, col, side, name]) => {
      if (v === null) return;
      const w = side < 0 ? baseW : modelW;
      const x = side < 0 ? cx - gap / 2 - w : cx + gap / 2;
      const top = Math.min(y(v), y(0)), h = Math.max(1.5, Math.abs(y(v) - y(0)));
      s.appendChild(el("rect", { x, y: top, width: w, height: h, fill: col, rx: 3 }));
      const hit = el("rect", { class: "hit", x: x - 4, y: m.t, width: w + 8, height: ih });
      hoverable(hit, `<b>${r.year} — ${name}</b><br><span class="k">IC</span> ${fmt(v)}` +
        (name === "Model" ? `<br><span class="k">t-stat</span> ${fmt(r.ic_tstat, 2)}<br><span class="k">events</span> ${r.n_test}` : ""));
      s.appendChild(hit);
    });
    if (r.ic_tstat !== null) {
      const ly = Math.max(m.t + 9, y(r.ic_mean) - 6);
      s.appendChild(el("text", { class: "dlab", x: cx + gap / 2 + modelW / 2, y: ly, "text-anchor": "middle" },
        "t " + fmt(r.ic_tstat, 1)));
    }
    s.appendChild(el("text", { class: "tick", x: cx, y: m.t + ih + 20, "text-anchor": "middle" }, r.year));
  });
  s.appendChild(el("line", { class: "axisline", x1: m.l, x2: m.l + iw, y1: y(0), y2: y(0) }));
  legend(document.getElementById("legIc"), [
    { color: c1, label: "Model" },
    { color: c2, label: "Baseline: " + DATA.baselineFeature + " alone" }]);
}

/* 2. predicted vs realised -------------------------------------------- */
function chartSpread() {
  const d = DATA.byYear;
  const host = document.getElementById("chartSpread");
  const W = 560, H = 310, m = { t: 20, r: 12, b: 40, l: 60 };
  const s = svg(host, W, H);
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const vals = d.flatMap(r => [r.predicted_spread, r.realised_spread]).filter(v => v !== null).map(v => v * 1e4);
  const [dlo, dhi] = domain(vals, { padTop: 0.12, padBottom: 0.18 });
  const ticks = ticksIn(dlo, dhi, 5);
  const y = v => m.t + ih - (v - dlo) / (dhi - dlo) * ih;
  yAxis(s, ticks, y, m.l, m.l + iw, "Top-minus-bottom quintile (bp)", t => t.toFixed(0));

  const bw = iw / d.length, w = bw * 0.30, gap = 2;
  const c1 = cssv("--s1"), c2 = cssv("--s2");
  d.forEach((r, i) => {
    const cx = m.l + bw * (i + 0.5);
    [[r.predicted_spread, c1, -1, "Predicted"], [r.realised_spread, c2, 1, "Realised"]]
      .forEach(([v, col, side, name]) => {
        if (v === null) return;
        const val = v * 1e4;
        const x = side < 0 ? cx - gap / 2 - w : cx + gap / 2;
        const top = Math.min(y(val), y(0)), h = Math.max(1.5, Math.abs(y(val) - y(0)));
        s.appendChild(el("rect", { x, y: top, width: w, height: h, fill: col, rx: 3 }));
        const hit = el("rect", { class: "hit", x: x - 4, y: m.t, width: w + 8, height: ih });
        hoverable(hit, `<b>${r.year} — ${name}</b><br><span class="k">spread</span> ${val.toFixed(0)} bp` +
          `<br><span class="k">calibration slope</span> ${fmt(r.calib_slope, 2)}`);
        s.appendChild(hit);
      });
    s.appendChild(el("text", { class: "tick", x: cx, y: m.t + ih + 20, "text-anchor": "middle" }, r.year));
  });
  s.appendChild(el("line", { class: "axisline", x1: m.l, x2: m.l + iw, y1: y(0), y2: y(0) }));
  legend(document.getElementById("legSpread"), [
    { color: c1, label: "Predicted" }, { color: c2, label: "Realised" }]);
}

/* 3. calibration ------------------------------------------------------- */
function chartCal() {
  const d = DATA.calibration;
  const host = document.getElementById("chartCal");
  if (!d.length) { host.innerHTML = '<p class="cap">Not enough predictions to calibrate.</p>'; return; }
  const W = 560, H = 320, m = { t: 12, r: 14, b: 42, l: 60 };
  const s = svg(host, W, H);
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const all = d.flatMap(r => [r.predicted, r.realised, r.realised + 1.96 * (r.se || 0), r.realised - 1.96 * (r.se || 0)]);
  const [dlo, dhi] = domain(all, { symmetric: true, padTop: 0.1, padBottom: 0.1 });
  const ticks = ticksIn(dlo, dhi, 5);
  const sc = v => (v - dlo) / (dhi - dlo);
  const x = v => m.l + sc(v) * iw, y = v => m.t + ih - sc(v) * ih;
  yAxis(s, ticks, y, m.l, m.l + iw, "Realised (bp)", t => t.toFixed(0));
  ticks.forEach(t => {
    s.appendChild(el("line", { class: "gridline", x1: x(t), x2: x(t), y1: m.t, y2: m.t + ih }));
    s.appendChild(el("text", { class: "tick", x: x(t), y: m.t + ih + 18, "text-anchor": "middle" }, t.toFixed(0)));
  });
  s.appendChild(el("text", { class: "alab", x: m.l + iw / 2, y: H - 6, "text-anchor": "middle" }, "Predicted (bp)"));

  const c1 = cssv("--s1"), c2 = cssv("--s2"), surf = cssv("--surface");
  s.appendChild(el("line", {
    x1: x(dlo), y1: y(dlo), x2: x(dhi), y2: y(dhi),
    stroke: cssv("--axis"), "stroke-width": 1.5, "stroke-dasharray": "5 4"
  }));
  const n = d.length;
  const mx = d.reduce((a, r) => a + r.predicted, 0) / n, my = d.reduce((a, r) => a + r.realised, 0) / n;
  let num = 0, den = 0;
  d.forEach(r => { num += (r.predicted - mx) * (r.realised - my); den += (r.predicted - mx) ** 2; });
  const slope = den ? num / den : 0, icept = my - slope * mx;
  const fitAt = v => slope * v + icept;
  s.appendChild(el("line", {
    x1: x(dlo), y1: y(fitAt(dlo)), x2: x(dhi), y2: y(fitAt(dhi)),
    stroke: c2, "stroke-width": 2
  }));
  d.forEach(r => {
    if (r.se) s.appendChild(el("line", {
      x1: x(r.predicted), x2: x(r.predicted), y1: y(r.realised - 1.96 * r.se), y2: y(r.realised + 1.96 * r.se),
      stroke: c1, "stroke-width": 1, opacity: .55
    }));
    s.appendChild(el("circle", { cx: x(r.predicted), cy: y(r.realised), r: 5, fill: surf, stroke: c1, "stroke-width": 2 }));
    const hit = el("circle", { class: "hit", cx: x(r.predicted), cy: y(r.realised), r: 13 });
    hoverable(hit, `<b>Prediction bin</b><br><span class="k">predicted</span> ${r.predicted.toFixed(1)} bp` +
      `<br><span class="k">realised</span> ${r.realised.toFixed(1)} bp<br><span class="k">events</span> ${r.n}`);
    s.appendChild(hit);
  });
  legend(document.getElementById("legCal"), [
    { color: c1, label: "Prediction bins (95% CI)" },
    { color: c2, label: "Fitted, slope " + slope.toFixed(2) },
    { color: cssv("--axis"), label: "Perfect calibration" }]);
}

/* 4. equity curve ------------------------------------------------------ */
function chartEq() {
  const d = DATA.equity;
  const host = document.getElementById("chartEq");
  if (!d.length) { host.innerHTML = '<p class="cap">No backtest produced for this run.</p>'; return; }
  const W = 560, H = 300, m = { t: 12, r: 14, b: 38, l: 62 };
  const s = svg(host, W, H);
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const vals = d.flatMap(r => [r.gross, r.net]);
  const [dlo, dhi] = domain(vals, { padTop: 0.08, padBottom: 0.08 });
  const ticks = ticksIn(dlo, dhi, 5);
  const y = v => m.t + ih - (v - dlo) / (dhi - dlo) * ih;
  const x = i => m.l + (i / (d.length - 1)) * iw;
  yAxis(s, ticks, y, m.l, m.l + iw, "Cumulative return", t => (t * 100).toFixed(0) + "%");

  const c1 = cssv("--s1"), c2 = cssv("--s2");
  const line = (key, col, width) => {
    const pts = d.map((r, i) => `${x(i).toFixed(2)},${y(r[key]).toFixed(2)}`).join(" ");
    s.appendChild(el("polyline", { points: pts, fill: "none", stroke: col, "stroke-width": width,
      "stroke-linejoin": "round", "stroke-linecap": "round" }));
  };
  line("gross", c2, 1.6);
  line("net", c1, 2);

  const step = Math.max(1, Math.floor(d.length / 5));
  for (let i = 0; i < d.length; i += step) {
    s.appendChild(el("text", { class: "tick", x: x(i), y: m.t + ih + 18, "text-anchor": "middle" },
      d[i].date.slice(0, 7)));
  }
  const cross = el("line", { class: "axisline", y1: m.t, y2: m.t + ih, opacity: 0 });
  s.appendChild(cross);
  const dotN = el("circle", { r: 4.5, fill: cssv("--surface"), stroke: c1, "stroke-width": 2, opacity: 0 });
  const dotG = el("circle", { r: 4.5, fill: cssv("--surface"), stroke: c2, "stroke-width": 2, opacity: 0 });
  s.appendChild(dotG); s.appendChild(dotN);
  const hit = el("rect", { class: "hit", x: m.l, y: m.t, width: iw, height: ih });
  hit.addEventListener("pointermove", e => {
    const box = s.getBoundingClientRect();
    const px = (e.clientX - box.left) / box.width * W;
    let i = Math.round((px - m.l) / iw * (d.length - 1));
    i = Math.max(0, Math.min(d.length - 1, i));
    const r = d[i];
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 1);
    dotN.setAttribute("cx", x(i)); dotN.setAttribute("cy", y(r.net)); dotN.setAttribute("opacity", 1);
    dotG.setAttribute("cx", x(i)); dotG.setAttribute("cy", y(r.gross)); dotG.setAttribute("opacity", 1);
    showTip(e, `<b>${r.date}</b><br><span class="k">net</span> ${pct(r.net)}<br><span class="k">gross</span> ${pct(r.gross)}`);
  });
  hit.addEventListener("pointerleave", () => {
    hideTip(); cross.setAttribute("opacity", 0);
    dotN.setAttribute("opacity", 0); dotG.setAttribute("opacity", 0);
  });
  s.appendChild(hit);
  legend(document.getElementById("legEq"), [
    { color: c1, label: "Net of costs" }, { color: c2, label: "Gross" }]);
}

/* tiles, table, chrome ------------------------------------------------- */
function tiles() {
  const a = DATA.aggregate, host = document.getElementById("tiles");
  const sign = v => v === null ? "" : v > 0 ? "pos" : v < 0 ? "neg" : "";
  const items = [
    { label: "Mean out-of-sample IC", value: fmt(a.mean_ic), note: `pooled ${fmt(a.pooled_ic)}`, cls: sign(a.mean_ic) },
    { label: "IC t-stat across years", value: fmt(a.ic_tstat_across_years, 2), note: `${a.n_years} held-out years` },
    { label: "Years with positive IC", value: `${a.positive_ic_years}/${a.n_years}`, note: a.years },
    { label: "Net Sharpe", value: fmt(a.stitched_sharpe_net, 2), note: `t = ${fmt(a.stitched_tstat_nw, 2)}`, cls: sign(a.stitched_sharpe_net) },
    { label: "Calibration slope", value: fmt(a.mean_calibration_slope, 2), note: "1.00 = perfectly scaled" },
    { label: "Max drawdown", value: pct(a.stitched_max_drawdown), note: "net of costs" },
  ];
  host.innerHTML = items.map(i => `<div class="tile"><div class="label">${i.label}</div>` +
    `<div class="value ${i.cls || ""}">${i.value}</div><div class="note">${i.note}</div></div>`).join("");
}

function table() {
  const cols = [
    ["year", "Year", v => v], ["n_train", "Train n", v => v], ["n_test", "Test n", v => v],
    ["ic_mean", "IC", v => fmt(v)], ["ic_tstat", "IC t", v => fmt(v, 2)],
    ["ic_hit_rate", "IC hit", v => pct(v, 0)],
    ["predicted_spread", "Predicted", v => bp(v)], ["realised_spread", "Realised", v => bp(v)],
    ["calib_slope", "Calib slope", v => fmt(v, 2)],
    ["ann_return_net", "Net return", v => pct(v)], ["sharpe_net", "Sharpe", v => fmt(v, 2)],
    ["max_drawdown", "Max DD", v => pct(v)], ["turnover", "Turnover", v => fmt(v, 1)],
    ["baseline_ic", "Baseline IC", v => fmt(v)],
  ];
  const t = document.getElementById("table");
  t.innerHTML = "<thead><tr>" + cols.map(c => `<th>${c[1]}</th>`).join("") + "</tr></thead><tbody>" +
    DATA.byYear.map(r => "<tr>" + cols.map(([k, , f]) => {
      const v = r[k];
      const cls = ["ic_mean", "realised_spread", "ann_return_net", "sharpe_net"].includes(k) && v !== null
        ? (v > 0 ? "pos" : v < 0 ? "neg" : "") : "";
      return `<td class="${cls}">${f(v)}</td>`;
    }).join("") + "</tr>").join("") + "</tbody>";
}

function chrome() {
  const m = DATA.metadata;
  document.getElementById("subtitle").textContent =
    `${m.label} · trained through year Y−1, frozen, predicting year Y · target ${DATA.target} · ${m.model} · ${m.n_events} events`;
  if (String(m.label).includes("synthetic")) {
    const b = document.getElementById("banner");
    b.hidden = false;
    b.innerHTML = "<strong>Synthetic data.</strong> This market is generated from a known " +
      "data-generating process. These numbers show that the holdout machinery recovers an effect " +
      "that was deliberately planted and does not invent one that was not — they are not a claim " +
      "about real equities. Running <code>eee holdout --drift 0</code> plants no effect; every " +
      "number here should collapse toward zero in that run.";
  }
  document.getElementById("footer").innerHTML =
    `Generated by <code>eee holdout</code>. Training for each year uses only events whose outcome ` +
    `was already resolved before that year opened, with a ${m.embargo_days ?? 25}-session embargo. ` +
    `Positions open one session after the announcement, so the untradable opening gap is excluded. ` +
    `Returns are net of a square-root market-impact cost model.`;
}

function renderAll() { chrome(); tiles(); chartIc(); chartSpread(); chartCal(); chartEq(); table(); }

const btn = document.getElementById("themeBtn");
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  btn.textContent = t === "dark" ? "Light mode" : "Dark mode";
  renderAll();
}
btn.addEventListener("click", () => {
  const now = document.documentElement.getAttribute("data-theme");
  const dark = now ? now === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(dark ? "light" : "dark");
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (!document.documentElement.getAttribute("data-theme")) renderAll();
});
addEventListener("resize", () => { clearTimeout(window._rt); window._rt = setTimeout(renderAll, 120); });
renderAll();
</script>
</body>
</html>
"""
