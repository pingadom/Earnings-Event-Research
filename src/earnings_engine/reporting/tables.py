"""Markdown tables and the run report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def format_significance_table(table: pd.DataFrame, decimals: int = 4) -> str:
    """Significance grid as markdown, with CARs shown in basis points."""
    if table.empty:
        return "_no results_"
    out = table.copy()
    for col in ("mean", "std_error", "ci_low", "ci_high"):
        if col in out.columns:
            out[col] = (out[col] * 10_000).round(1)
    out = out.rename(
        columns={
            "mean": "mean (bp)",
            "std_error": "se (bp)",
            "ci_low": "ci low (bp)",
            "ci_high": "ci high (bp)",
            "t_stat": "t",
            "p_value": "p",
        }
    )
    for col in ("t", "p"):
        if col in out.columns:
            out[col] = out[col].round(decimals)
    return out.to_markdown(index=False)


def format_stats(stats: dict, decimals: int = 3) -> str:
    rows = [
        {"metric": k, "value": round(v, decimals) if isinstance(v, float) else v}
        for k, v in stats.items()
        if not isinstance(v, (pd.DataFrame, pd.Series))
    ]
    return pd.DataFrame(rows).to_markdown(index=False)


def write_report(
    out_dir: str | Path,
    *,
    title: str,
    sections: dict[str, str],
    figures: dict[str, str | Path] | None = None,
    metadata: dict | None = None,
) -> Path:
    """Write a self-contained markdown report next to the figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [f"# {title}", "", f"_Generated {stamp}_", ""]
    if metadata:
        blob = json.dumps(metadata, indent=2, default=str)
        lines += ["## Run metadata", "", "```json", blob, "```", ""]
    for heading, body in sections.items():
        lines += [f"## {heading}", "", body, ""]
    if figures:
        lines += ["## Figures", ""]
        for caption, path in figures.items():
            rel = Path(path).name
            lines += [f"**{caption}**", "", f"![{caption}]({rel})", ""]

    report = out / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
