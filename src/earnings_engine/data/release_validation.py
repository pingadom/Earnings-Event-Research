"""Measure how often the release parser is right, against an independent source.

A regular expression over prose written by thousands of different people cannot
be trusted on the strength of its author's confidence. It can, however, be
*measured*: the figure printed in an earnings release is the same figure the
issuer reports as diluted earnings per share in the 10-Q that follows weeks
later, so the XBRL history is an independent check on every extraction.

This is what makes the parsed figure usable in a study. Coverage is easy to
report and nearly meaningless -- a parser that returns a number every time and
is wrong a third of the time has 100% coverage. What matters is agreement, and
where it fails.

The comparison is deliberately not used to *correct* anything. Doing so would
reintroduce the look-ahead this whole exercise exists to remove, because the
10-Q is not public at the announcement. It measures the parser, and then the
parser stands on its own.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger
from .release_figures import extract_diluted_eps

log = get_logger(__name__)

#: Agreement to the cent. Releases print two decimals, so this is exact equality
#: for the reported figure rather than a tolerance chosen to flatter the result.
CENT = 0.005


@dataclass
class ParserReport:
    """How the parser did against the XBRL it can be checked against."""

    attempted: int
    parsed: int
    checkable: int
    exact: int
    sign_agreed: int
    errors: pd.Series

    @property
    def coverage(self) -> float:
        return self.parsed / self.attempted if self.attempted else float("nan")

    @property
    def accuracy(self) -> float:
        """Share of checkable extractions that match XBRL to the cent."""
        return self.exact / self.checkable if self.checkable else float("nan")

    @property
    def sign_accuracy(self) -> float:
        return self.sign_agreed / self.checkable if self.checkable else float("nan")

    def render(self) -> str:
        median = float(np.nanmedian(np.abs(self.errors))) if len(self.errors) else float("nan")
        return (
            f"parsed {self.parsed:,}/{self.attempted:,} releases ({self.coverage:.1%}); "
            f"{self.checkable:,} could be checked against XBRL, of which "
            f"{self.accuracy:.1%} match to the cent and {self.sign_accuracy:.1%} agree on sign; "
            f"median absolute error {median:.3f}"
        )


def parse_releases(filings: pd.DataFrame, text_dir: str | Path) -> pd.DataFrame:
    """Extract a diluted earnings figure from every release on disk.

    Returns one row per release that yielded a figure, carrying the accession,
    the value, and the sentence it came from so any number can be checked by
    eye without re-running anything.
    """
    directory = Path(text_dir)
    rows = []
    for row in filings.itertuples(index=False):
        path = directory / f"{row.accession}.txt.gz"
        if not path.exists():
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:  # pragma: no cover - corrupt cache entry
            log.warning("could not read %s (%s)", path.name, exc)
            continue
        figure = extract_diluted_eps(text)
        if figure is None:
            continue
        rows.append(
            {
                "ticker": row.ticker,
                "accession": row.accession,
                "eps_release": figure.value,
                "pattern_rank": figure.pattern_rank,
                "penalty": figure.penalty,
                "agreement": figure.agreement,
                "context": figure.context[:300],
            }
        )
    return pd.DataFrame(rows)


#: A release announces a quarter that ended roughly this long before it.
_TYPICAL_LAG_DAYS = 40
#: How far the nearest fiscal period may sit from that anchor and still be a
#: match. Wide enough for a 52/53-week retail calendar, narrow enough that the
#: neighbouring quarter cannot be picked up.
_MATCH_TOLERANCE_DAYS = 55


def validate_against_xbrl(
    parsed: pd.DataFrame,
    releases: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    eps_column: str = "eps",
) -> tuple[pd.DataFrame, ParserReport]:
    """Join each parsed figure to the XBRL value for the quarter it announced.

    The join is on the *nearest fiscal period*, not on an equality. An 8-K's
    ``period_end`` in the SEC submissions index is the date of the event, not
    the quarter being reported, and an issuer's fiscal quarters rarely end on
    calendar quarter ends -- Apple's first quarter ends in late December, most
    retailers' fourth ends in January. Keying on either would match almost
    nothing, which is exactly what a first attempt here did: fifty rows out of
    twenty thousand.

    So each release is anchored to roughly six weeks before it was filed, and
    matched to the issuer's own reporting period closest to that anchor, within
    a tolerance too narrow to reach the neighbouring quarter.

    Rows with no XBRL counterpart stay in the frame with a null comparison --
    they are extractions that cannot be checked, not extractions that failed.
    """
    if parsed.empty:
        return parsed, ParserReport(0, 0, 0, 0, 0, pd.Series(dtype="float64"))

    dates = releases[["accession", "ticker", "filing_date"]].copy()
    dates["filing_date"] = pd.to_datetime(dates["filing_date"], errors="coerce")
    merged = parsed.merge(dates, on=["accession", "ticker"], how="left")
    merged["anchor"] = merged["filing_date"] - pd.Timedelta(days=_TYPICAL_LAG_DAYS)

    facts = fundamentals.loc[fundamentals[eps_column].notna(), ["ticker", "period_end", eps_column]]
    facts = facts.copy()
    facts["period_end"] = pd.to_datetime(facts["period_end"], errors="coerce")
    facts = (
        facts.dropna(subset=["period_end"])
        .sort_values(["period_end", "ticker"])
        .drop_duplicates(["ticker", "period_end"], keep="first")
        .rename(columns={eps_column: "eps_xbrl"})
    )

    if facts.empty:
        # No reference to check against. Every extraction is uncheckable rather
        # than wrong, and saying so is the point of separating the two.
        merged["eps_xbrl"] = np.nan
        merged["period_end"] = pd.NaT
        left = merged
    else:
        left = merged.dropna(subset=["anchor"]).sort_values("anchor")
    merged = left if facts.empty else pd.merge_asof(
        left,
        facts.sort_values("period_end"),
        left_on="anchor",
        right_on="period_end",
        by="ticker",
        direction="nearest",
        tolerance=pd.Timedelta(days=_MATCH_TOLERANCE_DAYS),
    )

    merged["error"] = merged["eps_release"] - merged["eps_xbrl"]
    checkable = merged["eps_xbrl"].notna()
    exact = checkable & (merged["error"].abs() <= CENT)
    signs = checkable & (
        np.sign(merged["eps_release"]).eq(np.sign(merged["eps_xbrl"]))
        | (merged["eps_release"].abs() <= CENT)
    )
    report = ParserReport(
        attempted=int(len(releases)),
        parsed=int(len(parsed)),
        checkable=int(checkable.sum()),
        exact=int(exact.sum()),
        sign_agreed=int(signs.sum()),
        errors=merged.loc[checkable, "error"],
    )
    log.info("%s", report.render())
    return merged, report
