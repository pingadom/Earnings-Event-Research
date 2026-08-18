# ADR 0003 — Enter one session after t0; target drift windows only

**Status:** accepted · **Date:** 2026-08-18

## Context

An initial implementation held positions from `t0` and targeted
`car_market_model_0_19`. On the synthetic fixture this produced a net Sharpe
near 3, and — importantly — *also* produced a Sharpe above 2 on the null
fixture with no planted drift.

## Decision

Positions open at `entry_offset = 1` sessions after `t0`. The default model
target is `car_market_model_1_20`. Windows beginning at day 0 are still computed
and reported, but are documented as measurement-only.

## Rationale

The diagnosis was not a coding error. The announcement reaction is realised in
the opening gap on `t0`, so it appears in `t0`'s close-to-close return. A
strategy entering at the `t0` open cannot capture it. Because the jump is an
order of magnitude larger than the drift, it dominated the P&L and made the
null fixture look profitable — which is exactly the failure this repository
exists to prevent.

With the offset applied, the fixtures separate as they should: planted drift
gives a positive net Sharpe; the null gives approximately zero.

## Consequences

- Measured performance is much lower, and correct.
- `tests/test_backtest.py::test_entry_offset_excludes_the_announcement_day`
  guards the regression.
- Anyone wanting the day-0 reaction must opt in deliberately and cannot do so by
  accident.
