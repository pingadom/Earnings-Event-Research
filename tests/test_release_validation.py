"""Measuring the parser against XBRL, rather than trusting it."""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from earnings_engine.data.release_validation import (
    parse_releases,
    validate_against_xbrl,
)


@pytest.fixture
def corpus(tmp_path):
    releases = {
        "0-20-1": "GAAP diluted earnings per share of $1.22 for the quarter.",
        "0-20-2": "Diluted loss per share was $(0.43).",
        "0-20-3": "The board declared a dividend of $0.35 per share.",
    }
    for accession, text in releases.items():
        with gzip.open(tmp_path / f"{accession}.txt.gz", "wt", encoding="utf-8") as handle:
            handle.write(text)
    filings = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "accession": list(releases),
            "filing_date": pd.to_datetime(["2020-05-10", "2020-08-10", "2020-11-10"]),
        }
    )
    return tmp_path, filings


def test_only_releases_with_a_figure_are_returned(corpus):
    text_dir, filings = corpus
    parsed = parse_releases(filings, text_dir)
    assert list(parsed["accession"]) == ["0-20-1", "0-20-2"]
    assert parsed["eps_release"].tolist() == pytest.approx([1.22, -0.43])


def test_a_missing_file_is_skipped_not_invented(corpus):
    text_dir, filings = corpus
    filings = pd.concat(
        [filings, filings.head(1).assign(accession="not-on-disk")], ignore_index=True
    )
    assert len(parse_releases(filings, text_dir)) == 2


def test_the_join_finds_the_quarter_the_release_announced(corpus):
    """Not an equality: fiscal periods rarely end on calendar quarter ends."""
    text_dir, filings = corpus
    parsed = parse_releases(filings, text_dir)
    fundamentals = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            # Fiscal quarters ending a few days off the calendar quarter.
            "period_end": pd.to_datetime(["2020-03-28", "2020-06-27"]),
            "eps": [1.22, -0.40],
        }
    )
    merged, report = validate_against_xbrl(parsed, filings, fundamentals)
    assert report.checkable == 2
    assert report.exact == 1, "1.22 matches; -0.43 against -0.40 does not"
    assert report.sign_agreed == 2
    assert report.accuracy == pytest.approx(0.5)


def test_an_unmatched_release_is_reported_as_uncheckable_not_wrong(corpus):
    text_dir, filings = corpus
    parsed = parse_releases(filings, text_dir)
    empty = pd.DataFrame({"ticker": [], "period_end": [], "eps": []})
    merged, report = validate_against_xbrl(parsed, filings, empty)
    assert report.parsed == 2
    assert report.checkable == 0
    assert len(merged) == 2


def test_the_neighbouring_quarter_is_out_of_reach(corpus):
    """The tolerance must not let an adjacent quarter satisfy the match."""
    text_dir, filings = corpus
    parsed = parse_releases(filings, text_dir)
    far = pd.DataFrame(
        {"ticker": ["AAA"], "period_end": pd.to_datetime(["2019-06-30"]), "eps": [1.22]}
    )
    _merged, report = validate_against_xbrl(parsed, filings, far)
    assert report.checkable == 0


def test_the_report_renders_every_number_a_reader_needs(corpus):
    text_dir, filings = corpus
    parsed = parse_releases(filings, text_dir)
    fundamentals = pd.DataFrame(
        {"ticker": ["AAA"], "period_end": pd.to_datetime(["2020-03-28"]), "eps": [1.22]}
    )
    _merged, report = validate_against_xbrl(parsed, filings, fundamentals)
    rendered = report.render()
    for fragment in ("parsed", "checked against XBRL", "match to the cent", "median absolute"):
        assert fragment in rendered
