"""Cumulative year-to-date XBRL flows must reduce to honest single quarters."""

from __future__ import annotations

import pandas as pd
import pytest

from earnings_engine.data.quarterise import (
    QUARTER_MAX_DAYS,
    parse_flow_facts,
    quarterly_flow_records,
)

FORMS = frozenset({"10-K", "10-Q"})
WINDOW = {"window_start": pd.Timestamp("2015-01-01"), "window_end": pd.Timestamp("2026-01-01")}


def fact(start, end, val, filed, accn, form="10-Q", fp="Q1", fy=2020):
    return {
        "start": start,
        "end": end,
        "val": val,
        "filed": filed,
        "accn": accn,
        "form": form,
        "fp": fp,
        "fy": fy,
    }


def a_fiscal_year():
    """A filer that tags Q1 discretely and everything after it cumulatively."""
    return [
        fact("2020-01-01", "2020-03-31", 100.0, "2020-04-20", "q1", fp="Q1"),
        fact("2020-01-01", "2020-06-30", 250.0, "2020-07-20", "q2", fp="Q2"),
        fact("2020-01-01", "2020-09-30", 420.0, "2020-10-20", "q3", fp="Q3"),
        fact("2020-01-01", "2020-12-31", 600.0, "2021-02-15", "fy", form="10-K", fp="FY"),
    ]


def by_period(records):
    return {str(r["period_end"].date()): r for r in records}


def test_cumulative_facts_are_differenced_into_quarters():
    out = by_period(quarterly_flow_records(a_fiscal_year(), allowed_forms=FORMS, **WINDOW))
    assert out["2020-03-31"]["value"] == pytest.approx(100.0)
    assert out["2020-06-30"]["value"] == pytest.approx(150.0)
    assert out["2020-09-30"]["value"] == pytest.approx(170.0)
    assert out["2020-12-31"]["value"] == pytest.approx(180.0)


def test_every_returned_value_spans_one_quarter():
    """The point of the module: no annual figure may masquerade as a quarter."""
    for record in quarterly_flow_records(a_fiscal_year(), allowed_forms=FORMS, **WINDOW):
        span = (record["period_end"] - pd.Timestamp(record["fact_start"])).days + 1
        assert span <= QUARTER_MAX_DAYS


def test_derived_quarters_carry_their_derivation():
    out = by_period(quarterly_flow_records(a_fiscal_year(), allowed_forms=FORMS, **WINDOW))
    assert out["2020-03-31"]["derivation"] is None
    derivation = out["2020-09-30"]["derivation"]
    assert derivation["method"] == "cumulative_difference"
    assert derivation["cumulative"]["end"] == "2020-09-30"
    assert derivation["subtrahend"]["end"] == "2020-06-30"


def test_a_natively_tagged_quarter_outranks_a_derived_one():
    """Both are emitted; the caller keeps the lower score, which must be native."""
    facts = a_fiscal_year()
    facts.append(fact("2020-07-01", "2020-09-30", 999.0, "2020-10-20", "q3", fp="Q3"))
    records = [r for r in quarterly_flow_records(facts, allowed_forms=FORMS, **WINDOW)
               if str(r["period_end"].date()) == "2020-09-30"]
    best = min(records, key=lambda r: r["duration_score"])
    assert best["value"] == pytest.approx(999.0)
    assert best["derivation"] is None


def test_derived_quarter_is_stamped_when_both_inputs_are_public():
    """A difference is not knowable before its later input was filed."""
    out = by_period(quarterly_flow_records(a_fiscal_year(), allowed_forms=FORMS, **WINDOW))
    assert out["2020-12-31"]["filed"] == pd.Timestamp("2021-02-15")


def test_out_of_sequence_restatement_is_dropped_not_guessed():
    facts = [
        fact("2020-01-01", "2020-03-31", 100.0, "2020-11-01", "late"),
        fact("2020-01-01", "2020-06-30", 250.0, "2020-07-20", "q2", fp="Q2"),
    ]
    out = by_period(quarterly_flow_records(facts, allowed_forms=FORMS, **WINDOW))
    assert "2020-06-30" not in out


def test_first_reported_value_wins_over_a_restatement():
    facts = [
        fact("2020-01-01", "2020-03-31", 100.0, "2020-04-20", "orig"),
        fact("2020-01-01", "2020-03-31", 111.0, "2021-04-20", "amended"),
    ]
    out = by_period(quarterly_flow_records(facts, allowed_forms=FORMS, **WINDOW))
    assert out["2020-03-31"]["value"] == pytest.approx(100.0)


def test_non_additive_concepts_keep_only_natively_tagged_quarters():
    out = quarterly_flow_records(a_fiscal_year(), allowed_forms=FORMS, additive=False, **WINDOW)
    assert [str(r["period_end"].date()) for r in out] == ["2020-03-31"]


def test_a_gap_that_is_not_a_quarter_is_refused():
    """A filer skipping a period must not yield a six-month 'quarter'."""
    facts = [
        fact("2020-01-01", "2020-03-31", 100.0, "2020-04-20", "q1"),
        fact("2020-01-01", "2020-09-30", 420.0, "2020-10-20", "q3", fp="Q3"),
    ]
    out = by_period(quarterly_flow_records(facts, allowed_forms=FORMS, **WINDOW))
    assert "2020-09-30" not in out


def test_facts_outside_the_window_or_the_form_list_are_ignored():
    facts = [
        fact("2020-01-01", "2020-03-31", 100.0, "2020-04-20", "q1", form="S-1"),
        fact("2010-01-01", "2010-03-31", 5.0, "2010-04-20", "old"),
    ]
    assert parse_flow_facts(facts, allowed_forms=FORMS, **WINDOW) == []


def test_facts_missing_a_start_date_are_not_flows():
    assert parse_flow_facts(
        [{"end": "2020-03-31", "val": 1.0, "filed": "2020-04-20", "accn": "x", "form": "10-Q"}],
        allowed_forms=FORMS,
        **WINDOW,
    ) == []
