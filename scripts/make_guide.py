#!/usr/bin/env python3
"""Render docs/how-it-works.md to a shareable PDF.

The markdown is the source of truth -- it renders on GitHub and is what anyone
reading the repository will see. This produces the version you can email to
someone who is never going to open GitHub.

Requires pandoc and wkhtmltopdf, both of which are packaged everywhere. If
either is missing the script says so and exits rather than producing a
half-rendered file.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs" / "how-it-works.md"
TARGET = REPO / "docs" / "how-it-works.pdf"

STYLE = """
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }
body { font: 10.5pt/1.5 "Times New Roman", Georgia, serif; color: #111; }
h1 { font: 700 20pt/1.25 Helvetica, Arial, sans-serif; margin: 0 0 6pt; letter-spacing: -.01em; }
h2 { font: 700 12.5pt/1.3 Helvetica, Arial, sans-serif; margin: 20pt 0 6pt;
     padding-bottom: 3pt; border-bottom: 1px solid #ccc; page-break-after: avoid; }
h3 { font: 700 10.5pt/1.3 Helvetica, Arial, sans-serif; margin: 13pt 0 3pt; page-break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt;
        font: 9pt/1.35 Helvetica, Arial, sans-serif; page-break-inside: avoid; }
th, td { border-bottom: 1px solid #ddd; padding: 3.5pt 6pt; text-align: left; vertical-align: top; }
thead th { border-bottom: 1px solid #888; color: #444; font-weight: 700; }
blockquote { margin: 8pt 0; padding: 6pt 12pt; border-left: 2.5pt solid #2a78d6; background: #f4f7fb; }
blockquote p { margin: 0; }
code { font: 9pt Consolas, monospace; background: #f2f2f0; padding: 0 2pt; }
hr { border: 0; border-top: 1px solid #ddd; margin: 16pt 0; }
li { margin-bottom: 3pt; }
"""

#: A cell that is a number, a percentage, or a basis-point figure.
_NUMERIC = re.compile(r"^[\s$£€]*[−\-+]?[\d.,]+\s*(?:%|bp|×|x|/yr)?\s*$")


def _align_numeric_columns(html: str) -> str:
    """Right-align columns that hold numbers, and only those.

    Markdown carries no alignment, and right-aligning everything puts prose
    against the wrong margin. A column counts as numeric when most of its body
    cells parse as numbers.
    """

    def fix(match: re.Match[str]) -> str:
        table = match.group(0)
        rows = re.findall(r"<tr>(.*?)</tr>", table, re.S)
        if not rows:
            return table
        columns = len(re.findall(r"<t[dh][^>]*>", rows[0]))
        numeric: set[int] = set()
        for column in range(columns):
            values = []
            for row in rows[1:]:
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
                if column < len(cells):
                    values.append(re.sub(r"<[^>]+>", "", cells[column]).strip())
            if values and sum(bool(_NUMERIC.match(v)) for v in values) >= max(
                1, int(0.7 * len(values))
            ):
                numeric.add(column)

        def fix_row(row: str) -> str:
            index = [-1]

            def replace(cell: re.Match[str]) -> str:
                index[0] += 1
                if index[0] in numeric:
                    return f'<{cell.group(1)} style="text-align:right">'
                return cell.group(0)

            return re.sub(r"<(t[dh])[^>]*>", replace, row)

        for row in rows:
            table = table.replace(f"<tr>{row}</tr>", f"<tr>{fix_row(row)}</tr>")
        return table

    return re.sub(r"<table>.*?</table>", fix, html, flags=re.S)


def build(source: Path = SOURCE, target: Path = TARGET) -> Path:
    for tool in ("pandoc", "wkhtmltopdf"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is required to build the guide; install it and re-run")
    if not source.exists():
        raise SystemExit(f"{source} not found")

    with tempfile.TemporaryDirectory() as scratch:
        body = Path(scratch) / "body.html"
        subprocess.run(
            ["pandoc", str(source), "-f", "gfm", "-t", "html5", "-o", str(body)], check=True
        )
        page = Path(scratch) / "guide.html"
        page.write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{source.stem}</title><style>{STYLE}</style></head><body>"
            f"{_align_numeric_columns(body.read_text(encoding='utf-8'))}</body></html>",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "wkhtmltopdf", "--quiet", "--enable-local-file-access", "--page-size", "A4",
                "--margin-top", "18mm", "--margin-bottom", "16mm",
                "--margin-left", "16mm", "--margin-right", "16mm",
                str(page), str(target),
            ],
            check=True,
        )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", default=str(TARGET))
    args = parser.parse_args(argv)
    print(f"wrote {build(Path(args.source), Path(args.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
