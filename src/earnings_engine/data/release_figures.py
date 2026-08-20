"""Read the headline earnings figure out of an earnings press release.

Why this is necessary
---------------------
The project's own diagnostic says the median feature is 83 days old at the
announcement it is attached to. That is structural: an issuer files the 8-K
announcing results within minutes, and files the XBRL financial statements for
that same quarter weeks later in the 10-Q. Any feature built from XBRL is
therefore describing the *previous* quarter -- point-in-time correct, and not
about the event.

The figure that *is* available at the announcement is printed in the release
itself. Extracting it is the only way the stated hypothesis gets tested rather
than approximated.

What makes this hard
--------------------
A press release is written for humans, and every issuer writes a different one.
The same number appears as "diluted earnings per share of $1.22", "diluted EPS
was $1.22", "net income per diluted share, continuing operations, of $0.10", and
inside a table as a bare column. Three failure modes matter more than coverage:

**Adjusted figures.** Nearly every release quotes both a GAAP figure and an
"adjusted", "non-GAAP", "core" or "pro forma" one, and the adjusted number is
usually the larger and more prominent. Taking it would silently substitute
management's preferred measure for the audited one, and would not be comparable
with the XBRL history the surprise is measured against.

**Prior-year comparatives.** "$1.22, compared to $(0.04) a year ago" contains
two numbers and only the first is this quarter's.

**Negative numbers.** Accounting convention prints losses in parentheses, so
``$(0.04)`` is minus four cents, and a parser that misses this turns the worst
quarters into the best ones.

The approach is therefore to gather every candidate, score it against those
failure modes, and take the best -- returning nothing when no candidate is
clean, because a wrong figure is worse than a missing one. Every extraction
carries the sentence it came from, so any value can be checked by eye.

Accuracy is not asserted. :mod:`earnings_engine.data.release_validation`
compares every parsed figure against the same quarter's XBRL value once the
10-Q eventually arrives, which is an independent measurement of how often this
module is right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

#: A per-share dollar amount. The decimal point is **required**: earnings per
#: share is always printed to at least one decimal, and demanding it is what
#: stops "the first fiscal quarter ended March 27, 2010" from being read as
#: twenty-seven dollars a share -- a real misparse this rule was written for.
_AMOUNT = (
    # The minus sign appears on either side of the currency symbol depending on
    # who typed it: both "-$0.43" and "$-0.43" occur.
    r"(?:-\s*)?\$?\s*\(?\s*-?\s*(\d{1,3}(?:,\d{3})*\.\d{1,3}|\.\d{1,3})\s*\)?"
)

#: A per-share label must be joined to its amount by one of these. A line in a
#: financial table -- "Diluted Loss Per Share from Discontinued Operations 0.00
#: 0.00" -- has no such word, and the columns of a table are the current and
#: prior period side by side, so reading one is a coin flip.
_LINK = r"(?:of|was|were|is|are|at|to|totall?ed|reached|:|\bequal(?:s|led)?\s+to)\s*"

#: Ways an issuer names the diluted per-share figure. Ordered by how specific
#: each phrase is: the first pattern cannot mean anything else, the last could.
_PATTERNS: tuple[tuple[str, int], ...] = (
    (rf"diluted\s+(?:earnings|net\s+income|income|loss|earnings\s*\(loss\)|"
     rf"net\s+income\s*\(loss\))\s*(?:\(loss\))?\s*per\s+(?:common\s+)?share"
     rf"[^.$\n]{{0,60}}?{_LINK}{_AMOUNT}", 0),
    (rf"(?:earnings|net\s+income|income|loss)\s*(?:\(loss\))?\s*per\s+diluted\s+"
     rf"(?:common\s+)?share[^.$\n]{{0,60}}?{_LINK}{_AMOUNT}", 1),
    (rf"diluted\s+EPS[^.$\n]{{0,40}}?{_LINK}{_AMOUNT}", 2),
    (rf"{_AMOUNT}\s+per\s+diluted\s+(?:common\s+)?share", 3),
    # Many issuers never write "diluted" in the headline. Under US GAAP the
    # per-share figure a company leads with is the diluted one -- basic is
    # reported alongside but is not the headline measure -- so these phrasings
    # are accepted at a lower rank, meaning an explicit "diluted" match always
    # wins when the release contains one.
    (rf"\bEPS\b[^.$\n]{{0,40}}?{_LINK}{_AMOUNT}", 4),
    (rf"(?:earnings|net\s+income|income|loss)\s*(?:\(loss\))?\s*per\s+"
     rf"(?:common\s+)?share[^.$\n]{{0,50}}?{_LINK}{_AMOUNT}", 5),
    # A bare "$X per share" with no mention of earnings is deliberately absent.
    # It was measured at 23% agreement with XBRL while accounting for half of
    # all matches: it reads dividends, buyback prices and book value per share
    # as though they were earnings.
)

#: Any of these near a candidate means it is not the GAAP figure. Issuers invent
#: their own names for the adjusted number -- "Economic EPS", "net operating
#: income per share", "distributable earnings" -- and each one that slips
#: through substitutes management's preferred measure for the audited one.
_ADJUSTED = re.compile(
    r"adjusted|non-?GAAP|core\b|pro\s*forma|excluding|ex-?items|ex-?amorti|"
    r"operating\s+(?:EPS|earnings|income)|cash\s+EPS|economic\s+EPS|"
    r"before\s+(?:special|one-?time|unusual)|underlying|normali[sz]ed|"
    r"comparable|distributable|\bFFO\b|\bAFFO\b|segment\s+(?:EPS|earnings)",
    re.IGNORECASE,
)

#: Phrases that mark the number *after* them as a comparative, not this quarter.
_COMPARATIVE_BEFORE = re.compile(
    r"compare[sd]?\s+(?:to|with)|versus|\bvs\.?\b|prior[-\s]year|year[-\s]ago|"
    r"a\s+year\s+earlier|same\s+(?:quarter|period)|up\s+from|down\s+from",
    re.IGNORECASE,
)

#: A period reference *after* a figure usually attaches the figure to that
#: period. "diluted loss per share of $0.47 in the fourth quarter of fiscal
#: 2009" is last year's number in this year's release, and nothing before the
#: amount says so.
_COMPARATIVE_AFTER = re.compile(
    r"in\s+the\s+(?:prior|preceding|same|year[-\s]ago|first|second|third|fourth)\s+"
    r"(?:quarter|period|year)|a\s+year\s+(?:ago|earlier)|in\s+(?:fiscal\s+)?\d{4}|"
    r"of\s+fiscal\s+\d{4}|last\s+year",
    re.IGNORECASE,
)

#: Forward-looking language. A release quotes next quarter's guidance in the
#: same breath as this quarter's result, and the two read almost identically.
_GUIDANCE = re.compile(
    r"outlook|guidance|expects?|expected|anticipat|forecast|project(?:s|ed|ion)|"
    r"\brange\s+of\b|\bto\s+be\s+in\s+the\b|full[-\s]year\s+\d{4}|"
    r"estimate|raise[sd]?|lower(?:s|ed|ing)|update[sd]?\s+its|\bsees\b|target|"
    r"reaffirm|initiat(?:es|ed)|\bfor\s+(?:the\s+)?(?:full\s+year|fiscal\s+\d{4})\b",
    re.IGNORECASE,
)

#: Full-year or year-to-date framing: the figure is not the quarter's.
_ANNUAL = re.compile(
    r"full[-\s]year|fiscal\s+year|twelve\s+months|nine\s+months|six\s+months|"
    r"year[-\s]to[-\s]date|first\s+half|second\s+half",
    re.IGNORECASE,
)

#: A diluted share figure outside this range is a misparse, not a company.
MIN_PLAUSIBLE, MAX_PLAUSIBLE = -100.0, 100.0

#: How much text either side of a match is inspected for the qualifiers above.
_WINDOW = 170


@dataclass(frozen=True)
class Figure:
    """One extracted number, with everything needed to check it by hand."""

    value: float
    context: str
    pattern_rank: int
    penalty: int
    #: How many *distinct* phrasings in the document produced this same value.
    #: An issuer states the headline figure more than once -- in the title, in a
    #: bullet, in the narrative -- and those are matched by different patterns.
    #: Two independent phrasings agreeing is far stronger evidence than one
    #: clean match, because the ways of being wrong (a table column, a
    #: comparative, an adjusted figure) rarely coincide on the same number.
    agreement: int = 1

    @property
    def confident(self) -> bool:
        """Nothing looked wrong, and at least two phrasings agree."""
        return self.penalty == 0 and self.agreement >= 2


def _to_float(raw: str, matched_text: str) -> float | None:
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return None
    # Accounting negatives: "(0.04)" and "$(0.04)" both mean minus four cents.
    negated = ("(" in matched_text and ")" in matched_text) or bool(
        re.search(r"-\s*\$?\s*" + re.escape(raw), matched_text)
    )
    if negated:
        value = -value
    if not MIN_PLAUSIBLE <= value <= MAX_PLAUSIBLE:
        return None
    return value


#: Where one statement ends and the next begins. "Reports Adjusted EPS of $2.59.
#: On a GAAP basis, diluted EPS was $1.22." must not let the first sentence
#: disqualify the second.
_CLAUSE_BREAK = re.compile(r"[.;\n•·]|\s{3,}")


def _clause_before(text: str, start: int) -> str:
    """The text between the start of the current statement and the figure."""
    window = text[max(0, start - _WINDOW) : start]
    breaks = list(_CLAUSE_BREAK.finditer(window))
    return window[breaks[-1].end() :] if breaks else window


def _clause_after(text: str, end: int) -> str:
    """Text following the figure, stopping at the next amount or statement.

    A period reference after an amount attaches to it -- unless another amount
    intervenes, in which case the reference belongs to that one. "$1.22,
    compared to $(0.04) in the prior year" is this quarter's figure, and reading
    the trailing period reference as its own would throw it away.
    """
    window = text[end : min(len(text), end + 90)]
    stop = re.search(r"\$|\d+\.\d{2}", window)
    if stop is not None:
        window = window[: stop.start()]
    breaks = list(_CLAUSE_BREAK.finditer(window))
    return window[: breaks[0].start()] if breaks else window


def _score(text: str, start: int, end: int) -> tuple[int, str]:
    """Penalise a candidate for each sign that it is the wrong number."""
    before = _clause_before(text, start)
    after = _clause_after(text, end)
    around = text[max(0, start - _WINDOW) : min(len(text), end + 90)]
    penalty = 0
    # The label itself carries the qualifier in "Economic EPS of $1.60", so the
    # matched text is searched along with what precedes it.
    labelled = before + text[start:end]
    if _ADJUSTED.search(labelled) or _ADJUSTED.search(after):
        penalty += 4
    if _GUIDANCE.search(labelled) or _GUIDANCE.search(after):
        penalty += 4
    if _ANNUAL.search(before):
        penalty += 3
    # A comparative phrase before the number means this number is the
    # comparison; a period reference after it attaches the number to that
    # period. Both appear constantly and neither is visible from the other side.
    if _COMPARATIVE_BEFORE.search(before):
        penalty += 2
    if _COMPARATIVE_AFTER.search(after):
        penalty += 2
    return penalty, " ".join(around.split())


def extract_diluted_eps(text: str) -> Figure | None:
    """Best GAAP diluted earnings per share in a release, or ``None``.

    Candidates are ranked by penalty first, then by how unambiguous the phrasing
    was, then by position -- releases lead with the current quarter, so an
    earlier match is more likely to be the headline figure. ``None`` is returned
    when every candidate looks like an adjusted or comparative number, because a
    confidently wrong figure is worse than a missing one.
    """
    if not text:
        return None
    best: Figure | None = None
    best_key: tuple[int, int, int] | None = None
    seen: dict[float, set[int]] = {}
    for pattern, rank in _PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = _to_float(match.group(1), match.group(0))
            if value is None:
                continue
            penalty, context = _score(text, match.start(), match.end())
            if penalty == 0:
                # Keyed by position, not by pattern: two regexes firing on one
                # sentence is one statement, not two independent ones.
                seen.setdefault(round(value, 2), set()).add(match.start() // 40)
            key = (penalty, rank, match.start())
            if best_key is None or key < best_key:
                best_key, best = key, Figure(value, context, rank, penalty)
    if best is not None:
        best = replace(best, agreement=len(seen.get(round(best.value, 2), {best.pattern_rank})))
    if best is not None and best.penalty >= 2:
        # Every candidate was adjusted, guidance, or a comparative. Report
        # nothing rather than substitute one of those for the figure announced.
        log.debug("no clean EPS candidate; returning none")
        return None
    return best
