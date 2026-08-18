"""Fetch the text of the earnings release attached to an 8-K.

Which document to read
----------------------
The original question asks whether *management's language* at the announcement
predicts the subsequent drift. That language lives in the earnings press
release -- filed as exhibit 99.1 to a Form 8-K under Item 2.02, timestamped to
the minute, and public at exactly the moment the event window opens.

The 10-Q is the wrong document for this. It arrives days or weeks after the
release, so its text is not knowable at the announcement, and a similarity
measure computed across a mixture of press releases and quarterly reports
compares documents that were never meant to be compared.

Restricting the corpus to one document type per firm is also what makes the
Cohen-Malloy-Nguyen similarity meaningful: consecutive releases are written to
the same template, so a drop in similarity is a real editorial change rather
than a change of genre.

Extraction
----------
EDGAR exhibits are HTML written by a filing agent, not by anyone expecting a
reader. Scripts, styles and tables of XBRL tags are stripped, entities are
unescaped, and whitespace is collapsed. This is deliberately crude: the
downstream features are dictionary counts and TF-IDF cosines, both of which are
insensitive to layout and both of which would be corrupted by leaving markup in
place.
"""

from __future__ import annotations

import gzip
import html
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..utils.logging_utils import get_logger
from .http import HttpClient

log = get_logger(__name__)

INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{bare}/{accession}-index.htm"
DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

#: Item 2.02 is "Results of Operations and Financial Condition" -- the earnings
#: release itself, as opposed to the dozens of other things an 8-K can report.
EARNINGS_ITEM = "2.02"

_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_RE = re.compile(r"</(p|div|tr|table|h[1-6]|li|br)\s*>|<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")

#: Below this, the "exhibit" is a cover page or an exhibit index, not a release.
MIN_DOCUMENT_CHARS = 400

#: Some filing agents prepend the SGML document header to the exhibit body, so
#: the text opens with "EX-99.1 2 d453749dex991.htm TEXT OF PRESS RELEASE"
#: before the release itself. It is boilerplate, it is identical from quarter to
#: quarter, and leaving it in would inflate every similarity by a constant.
_EDGAR_HEADER_RE = re.compile(r"\AEX-\d+[^\n]{0,200}?(?=Exhibit\s+\d)", re.IGNORECASE | re.DOTALL)


def html_to_text(raw: str) -> str:
    """Reduce a filing-agent HTML exhibit to readable prose."""
    text = _COMMENT_RE.sub(" ", _SCRIPT_RE.sub(" ", raw))
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = _SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_RE.sub("\n\n", text).strip()


def is_earnings_release(items: object) -> bool:
    """Whether an 8-K's item list includes the earnings item."""
    if not isinstance(items, str):
        return False
    return any(part.strip().startswith(EARNINGS_ITEM) for part in items.split(","))


_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)


def parse_index(page: str) -> list[dict[str, object]]:
    """Extract (type, href, size) for each document listed on a filing index.

    The exhibit *type* is the only reliable way to find the press release:
    ``index.json`` omits it entirely, and filename conventions vary by filing
    agent. The index page carries it in a plain table.
    """
    rows: list[dict[str, object]] = []
    for row in _ROW_RE.findall(page):
        cells = [_TAG_RE.sub("", html.unescape(c)).strip() for c in _CELL_RE.findall(row)]
        href = _HREF_RE.search(row)
        if href is None or len(cells) < 4:
            continue
        try:
            size = int(cells[4]) if len(cells) > 4 and cells[4].isdigit() else 0
        except (TypeError, ValueError):
            size = 0
        rows.append({"type": cells[3].upper(), "href": href.group(1), "size": size})
    return rows


def choose_document(page: str, primary_document: str | None) -> str | None:
    """Pick the press release from a filing index page.

    Exhibit 99.1 is the release by convention; 99.2 and beyond are supplemental
    schedules or slide decks. Where a filer used no 99-series exhibit, the
    release *is* the primary document, which is the case for text-only 8-Ks.
    """
    readable = (".htm", ".html", ".txt")
    rows = [r for r in parse_index(page) if str(r["href"]).lower().endswith(readable)]
    exhibits = [r for r in rows if str(r["type"]).startswith("EX-99")]
    if exhibits:
        preferred = [r for r in exhibits if r["type"] == "EX-99.1"]
        chosen = preferred[0] if preferred else max(exhibits, key=lambda r: r["size"])
        return str(chosen["href"]).rsplit("/", 1)[-1]
    return primary_document


@dataclass
class FilingTextDownloader:
    """Download and cache the plain text of earnings releases."""

    client: HttpClient
    text_dir: Path

    def __post_init__(self) -> None:
        self.text_dir = Path(self.text_dir)
        self.text_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, accession: str) -> Path:
        return self.text_dir / f"{accession}.txt.gz"

    def fetch(self, cik: int, accession: str, primary_document: str | None = None) -> str:
        """Return the release text, downloading it only if it is not cached."""
        destination = self.path_for(accession)
        if destination.exists():
            with gzip.open(destination, "rt", encoding="utf-8") as handle:
                return handle.read()
        bare = accession.replace("-", "")
        page = self.client.get_text(INDEX_URL.format(cik=int(cik), bare=bare, accession=accession))
        document = choose_document(page, primary_document)
        if not document:
            raise ValueError(f"no readable document in {accession}")
        raw = self.client.get_text(
            DOCUMENT_URL.format(cik=int(cik), accession=bare, document=document)
        )
        text = _EDGAR_HEADER_RE.sub("", html_to_text(raw)).strip()
        if len(text) < MIN_DOCUMENT_CHARS:
            raise ValueError(f"{accession}: {document} is too short to be a release")
        temporary = destination.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            handle.write(text)
        temporary.replace(destination)
        return text

    def fetch_many(self, filings: pd.DataFrame, limit: int | None = None) -> dict[str, int]:
        """Download every earnings release in a filings frame.

        One issuer's malformed exhibit never aborts the job; the counts it
        returns are what the acquisition report prints.
        """
        wanted = filings.loc[filings["form"].isin(["8-K", "8-K/A"])].copy()
        wanted = wanted.loc[wanted["items"].map(is_earnings_release)]
        wanted = wanted.sort_values(["ticker", "filing_date"])
        if limit is not None:
            wanted = wanted.head(limit)
        counts = {"requested": len(wanted), "downloaded": 0, "cached": 0, "failed": 0}
        for position, row in enumerate(wanted.itertuples(index=False), 1):
            if self.path_for(row.accession).exists():
                counts["cached"] += 1
                continue
            try:
                self.fetch(int(row.cik), str(row.accession), getattr(row, "primary_document", None))
                counts["downloaded"] += 1
            except Exception as exc:
                counts["failed"] += 1
                log.warning("filing text: %s failed (%s)", row.accession, exc)
            if position % 250 == 0:
                log.info("filing text: %d/%d", position, len(wanted))
        return counts


def read_text(text_dir: Path, accession: str) -> str:
    """Read a cached release, tolerating both the gzipped and plain layouts."""
    gz = Path(text_dir) / f"{accession}.txt.gz"
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as handle:
            return handle.read()
    plain = Path(text_dir) / f"{accession}.txt"
    if plain.exists():
        return plain.read_text(encoding="utf-8", errors="replace")
    return ""
