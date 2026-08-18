# ADR 0001 — US universe, EDGAR as the point-in-time backbone

**Status:** accepted · **Date:** 2026-08-18

## Context

The study needs (a) earnings announcement timestamps, (b) fundamentals as first
reported, and (c) filing text, over roughly a decade, with a verifiable
publication time for every fact. The alternative universe considered was the
FTSE 350.

## Decision

Target the S&P 500. Use SEC EDGAR as the reference source for filing
timestamps, first-reported XBRL fundamentals and filing text, with commercial
vendors layered on for consensus estimates and point-in-time index membership.

## Rationale

EDGAR's `acceptanceDateTime` is a minute-level public-availability stamp, and
`companyfacts` preserves the accession that first reported each fact. That
turns look-ahead prevention from an assumption into something testable. No
free UK equivalent exists; a FTSE 350 study would have to scrape RNS and would
have materially weaker point-in-time guarantees.

## Consequences

- The provider layer stays narrow so a UK adapter can be added later.
- US market microstructure conventions (BMO/AMC, NYSE calendar) are baked into
  `events/alignment.py` and `utils/calendar.py`; a UK version needs an LSE
  calendar and different announcement conventions.
- Results are not directly comparable to UK studies.
