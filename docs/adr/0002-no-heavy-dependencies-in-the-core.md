# ADR 0002 — Implement the statistical core on numpy/scipy

**Status:** accepted · **Date:** 2026-08-18

## Context

The natural choice for the market model, HAC standard errors and the
significance tests is `statsmodels`. It is well tested and widely trusted.

## Decision

Implement the market model (vectorised OLS across events), Newey–West, the
Patell/BMP standardisation and the cluster bootstrap directly on numpy/scipy.
Keep `statsmodels` as an optional `[stats]` extra used only by tests that
cross-check those implementations.

## Rationale

1. **Performance.** A per-event `statsmodels` OLS call over 50,000 events is a
   Python loop. The vectorised gather-and-regress implementation runs the same
   estimation in about a second.
2. **Hermetic CI.** Six lightweight dependencies means the test suite and the
   demo run anywhere, including environments with no package-index access.
3. **Transparency.** For a project whose whole claim is methodological care,
   having the estimator visible in the repository rather than behind an import
   is worth something.

## Consequences

- More code to maintain and test — mitigated by cross-checking against scipy
  closed forms in `tests/test_stats.py`.
- Advanced specifications (Fama–MacBeth with full covariance, panel GMM) would
  need to be added rather than imported. Acceptable: they are not needed for the
  stated hypotheses.
