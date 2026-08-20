"""Turn year-to-date XBRL flow facts into discrete quarters.

Why this module exists
----------------------
An issuer's 10-Q reports income and cash-flow items *cumulatively*: the Q2
filing states six months of operating cash flow, the Q3 filing nine months,
and the 10-K twelve. Only Q1 is naturally a single quarter. A naive reader
that keeps facts whose duration is roughly ninety days therefore throws away
three quarters in four -- which is exactly how operating cash flow ended up
with 20% coverage while the balance sheet had 40%.

Worse than the missing rows was the mixture that remained. Annual figures from
the 10-K were being accepted into the same column as quarterly figures from the
10-Q, so a level feature such as free cash flow was four times larger every
fourth row, and any feature differencing across adjacent rows compared a year
against a quarter.

The fix is the standard one: within a fiscal year, facts sharing a start date
form a nested sequence, and the discrete quarter is the difference between
consecutive members of that sequence.

    Q2 = H1 - Q1        Q3 = 9M - H1        Q4 = FY - 9M

Point-in-time discipline
------------------------
A derived quarter is only knowable once *both* of its inputs are public, so its
filing stamp is the later of the two. In practice the subtrahend was filed a
quarter earlier and the stamp is simply the current filing's, which keeps the
derived value attributable to a real accession number. When that ordering does
not hold -- a restatement filed out of sequence -- the quarter is dropped
rather than guessed at.

Restatements are excluded a second way: where the same (start, end) window has
been reported more than once, the *first* published value wins. A backtest may
only ever see what the market saw.

Additivity
----------
Differencing is exact for anything that sums over time: revenue, income, cash
flows. It is an approximation for per-share figures, because the weighted
average share count in the denominator drifts between quarters. Compustat
computes Q4 earnings per share this way regardless, and so does this module --
but the approximation is recorded in the provenance for every value it touches,
so nothing silently pretends to be an exact figure.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

#: A fact is already a single quarter if it spans roughly three months.
QUARTER_MIN_DAYS = 60
QUARTER_MAX_DAYS = 120
#: The longest cumulative window worth differencing: a fiscal year plus slack
#: for 52/53-week retail calendars.
CUMULATIVE_MAX_DAYS = 400


def _days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Inclusive day count, matching the convention used by XBRL durations."""
    return int((end - start).days) + 1


def parse_flow_facts(
    facts: list[dict[str, Any]],
    *,
    allowed_forms: frozenset[str] | set[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Normalise raw Company Facts entries into duration facts worth keeping."""
    parsed: list[dict[str, Any]] = []
    for fact in facts:
        form = str(fact.get("form", ""))
        if form not in allowed_forms or not fact.get("accn") or fact.get("val") is None:
            continue
        start = pd.to_datetime(fact.get("start"), errors="coerce")
        end = pd.to_datetime(fact.get("end"), errors="coerce")
        filed = pd.to_datetime(fact.get("filed"), errors="coerce")
        if pd.isna(start) or pd.isna(end) or pd.isna(filed):
            continue
        if not (window_start <= end <= window_end) or filed > window_end:
            continue
        span = _days(start, end)
        if not QUARTER_MIN_DAYS <= span <= CUMULATIVE_MAX_DAYS:
            continue
        try:
            value = float(fact["val"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        fiscal_year = fact.get("fy")
        try:
            fiscal_year = int(fiscal_year) if fiscal_year is not None else pd.NA
        except (TypeError, ValueError):
            fiscal_year = pd.NA
        parsed.append(
            {
                "accession": str(fact["accn"]),
                "period_end": end.normalize(),
                "period_start": start.normalize(),
                "span_days": span,
                "form": form,
                "fiscal_year": fiscal_year,
                "fiscal_period": str(fact.get("fp") or "unknown"),
                "filed": filed.normalize(),
                "value": value,
            }
        )
    return parsed


def _first_reported(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the earliest publication of each (start, end) window."""
    best: dict[tuple[pd.Timestamp, pd.Timestamp], dict[str, Any]] = {}
    for record in parsed:
        key = (record["period_start"], record["period_end"])
        incumbent = best.get(key)
        if incumbent is None or record["filed"] < incumbent["filed"]:
            best[key] = record
    return list(best.values())


def quarterly_flow_records(
    facts: list[dict[str, Any]],
    *,
    allowed_forms: frozenset[str] | set[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    additive: bool = True,
) -> list[dict[str, Any]]:
    """Return one record per discrete fiscal quarter the issuer has disclosed.

    Facts already spanning a single quarter pass through untouched. Cumulative
    facts are differenced against the longest shorter window sharing their start
    date. Anything that cannot be reduced to a quarter is discarded, so callers
    can rely on every returned value describing roughly ninety days.
    """
    parsed = _first_reported(
        parse_flow_facts(
            facts,
            allowed_forms=allowed_forms,
            window_start=window_start,
            window_end=window_end,
        )
    )
    by_start: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
    for record in parsed:
        by_start[record["period_start"]].append(record)

    out: list[dict[str, Any]] = []
    for start, group in by_start.items():
        group.sort(key=lambda item: item["period_end"])
        for position, record in enumerate(group):
            if record["span_days"] <= QUARTER_MAX_DAYS:
                out.append(_native(record))
                continue
            if not additive or position == 0:
                continue
            prior = group[position - 1]
            # The label must describe the quarter that was isolated, not the
            # cumulative window it came out of. A Q4 derived from a 10-K's
            # twelve-month figure inherited "FY" before this, and a downstream
            # adapter recognised it as an annual flow and differenced it a
            # second time -- silently, because the result is still a plausible
            # number. Facts here share a fiscal-year start, so position within
            # the sorted sequence is the quarter index.
            gap = int((record["period_end"] - prior["period_end"]).days)
            if not QUARTER_MIN_DAYS <= gap <= QUARTER_MAX_DAYS:
                continue
            if prior["filed"] > record["filed"]:
                # The subtrahend was published *after* the cumulative figure, so
                # the difference was not knowable at this filing. Drop it.
                continue
            out.append(_derived(record, prior, start, quarter=position + 1))
    return out


def _native(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "accession": record["accession"],
        "period_end": record["period_end"],
        "form": record["form"],
        "fiscal_year": record["fiscal_year"],
        "fiscal_period": record["fiscal_period"],
        "filed": record["filed"],
        "value": record["value"],
        "fact_start": str(record["period_start"].date()),
        # Prefer a fact that lands closest to a true quarter when an issuer
        # tags several overlapping windows.
        "duration_score": abs(record["span_days"] - 91),
        "derivation": None,
    }


def _derived(
    record: dict[str, Any], prior: dict[str, Any], start: pd.Timestamp, quarter: int
) -> dict[str, Any]:
    quarter_start = prior["period_end"] + pd.Timedelta(days=1)
    return {
        "accession": record["accession"],
        "period_end": record["period_end"],
        "form": record["form"],
        "fiscal_year": record["fiscal_year"],
        "fiscal_period": f"Q{quarter}" if 1 <= quarter <= 4 else record["fiscal_period"],
        # Knowable only once both inputs are public.
        "filed": max(record["filed"], prior["filed"]),
        "value": record["value"] - prior["value"],
        "fact_start": str(quarter_start.date()),
        # A derived quarter is a fallback: prefer a natively tagged one.
        "duration_score": 1000 + abs(int((record["period_end"] - quarter_start).days) + 1 - 91),
        "derivation": {
            "method": "cumulative_difference",
            "cumulative": {
                "start": str(start.date()),
                "end": str(record["period_end"].date()),
                "accession": record["accession"],
            },
            "subtrahend": {
                "start": str(start.date()),
                "end": str(prior["period_end"].date()),
                "accession": prior["accession"],
            },
        },
    }
