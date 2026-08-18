# Methodology

This document states what the pipeline computes and why each choice was made.
It is written to be the basis of a write-up: every non-obvious decision has a
reason and, where it matters, an alternative that was rejected.

---

## 1. The hypothesis

> Information contained in earnings results and company filings predicts
> short-horizon abnormal equity returns.

Stated as something falsifiable:

**H1 (measurement).** Firms in the top quintile of an earnings-information
signal earn positive sector-adjusted abnormal returns over `[t0+1, t0+20]`, and
firms in the bottom quintile earn negative ones, with the spread significant
under a cluster-bootstrap test that treats event dates rather than events as the
independent unit.

**H2 (tradability).** That spread survives realistic transaction costs in a
sector-neutral long/short book with 20-day holding periods.

**H0.** Both spreads are zero. Nothing in the design should make it hard to
conclude this, and the synthetic null fixture exists to check that it does not.

H1 and H2 are separate because they usually give different answers. Most
documented anomalies pass H1 and fail H2.

---

## 2. Event definition and alignment

An *event* is one earnings announcement. It is characterised by:

- `announced_at_utc` — the instant the results became public;
- `timing` — `bmo` / `amc` / `during` / `unknown`;
- `t0` — the first NYSE session on which the information could be traded;
- `trade_open_ts` — the exact UTC instant of `t0`'s opening bell.

**Alignment rules.** Given a local announcement time `T` on calendar day `D`:

| Case | `t0` |
|---|---|
| `T` before 09:30 ET and `D` is a session | `D` |
| `T` after the close | next session |
| `T` during the session | next session (default), or `D` under `intraday_policy: same_day` |
| `T` unknown | treated per `unknown_time_policy` (default `amc`) and flagged |
| `D` is a weekend or holiday | next session |

The intraday default is conservative and deliberately so. A close-to-close
return on day `D` for an announcement made at 11:00 contains a pre-announcement
stub that was not tradable on the information. Choosing `same_day` is permitted
but sets `accepts_intraday_lookahead=True` on the event and moves
`trade_open_ts` to the announcement instant, so the decision is recorded rather
than hidden.

**Why the timing flag matters more than it looks.** Roughly half of US
announcements are before the open and half after the close. Mislabelling one as
the other shifts the entire event window by one session, which moves the
announcement jump into or out of the measured window. Because the jump is an
order of magnitude larger than the drift, a mislabelling rate of even a few per
cent contaminates the drift estimate. Events whose timing had to be inferred
carry `timing_source ∈ {declared, derived, assumed}` so the study can be re-run
on declared-only events as a robustness check.

---

## 3. Abnormal returns

For event *i* and relative day *τ*, abnormal return `AR_{i,τ} = r_{i,τ} − E[r_{i,τ}]`,
with three specifications of `E[·]`:

**Market-adjusted.** `E[r] = r_m`. No estimated parameters, so nothing to
overfit; adequate at horizons short enough that beta dispersion contributes
little.

**Market model.** `r_it = α_i + β_i r_mt + ε_it`, estimated by OLS over
`[t0−250, t0−31]`. The 30-session gap before the event is not decoration:
pre-announcement run-up and leakage would otherwise contaminate the parameters
that define "normal". The residual standard deviation `σ_i` from this regression
is retained — it is what makes the BMP test possible.

**Sector-neutral.** `E[r_{i}] = mean(r_j : j ∈ sector(i), j ≠ i)`, equal
weighted. The leave-one-out construction is essential: including the stock in
its own benchmark shrinks the measured abnormal return by roughly `1/n` of the
sector.

**Aggregation.** `CAR = Σ AR` over the window; `BHAR = Π(1+r_i) − Π(1+r_bench)`.
CAR is the right object for statistical testing (it is a sum of approximately
mean-zero terms, so its variance is tractable); BHAR is the right object for
"what would I have made". They diverge as the horizon grows and both are
reported.

Windows: `[0,0]`, `[0,4]`, `[0,19]` for measurement; `[1,5]`, `[1,20]` for the
tradable drift. Daily returns are winsorised at the 0.5%/99.5% quantiles before
estimation; this shrinks measured effects slightly and is preferred to letting
a handful of data errors drive the result.

---

## 4. Statistical inference

Three tests are reported for every estimator × window, because they disagree in
informative ways.

**Cross-sectional t-test.** The baseline. Reported so the reader can see how
much the corrections matter.

**Boehmer–Musumeci–Poulsen.** Each CAR is standardised by its own predicted
standard error,

```
s_i = σ_i · sqrt( L + L²/T + (Σ_{t∈W}(r_mt − r̄_m))² / S_xx )
```

(the Patell prediction-error correction: the last two terms account for α and β
having been estimated), and an ordinary t-test is run on the standardised
values. This is robust to the variance *jump* on announcement days, which is
severe — event-day variance is routinely several times estimation-window
variance — and which makes the unstandardised test over-reject.

**Cluster bootstrap over event dates.** Earnings cluster: a large fraction of
the S&P 500 reports within a three-week window each quarter, and those firms
share macro shocks. Treating events as independent inflates the effective
sample size. Resampling whole event dates with replacement preserves the
within-date correlation instead of assuming it away, and typically widens the
interval by a factor that is worth seeing.

For time-series quantities (portfolio returns, IC series) a Newey–West HAC
correction is applied, because overlapping 20-day holding periods induce
autocorrelation and the naive standard error is too small.

---

## 5. Point-in-time discipline

The invariant, enforced in exactly one place so it cannot be bypassed:

```
for every (event, feature):   feature.available_from_utc ≤ event.trade_open_ts
```

Consequences:

- fundamentals are joined by an **as-of merge in publication time**, never on
  `period_end`;
- a feature that depends on four quarters of history inherits the publication
  stamp of the *most recent* of those quarters;
- a feature block published after the trade open is blanked for that event
  entirely, rather than left half-populated;
- a fact with no publication timestamp is an error, not a warning: a feature
  whose publication time is unknown cannot be proven free of look-ahead.

`docs/biases.md` covers what to do when a data source genuinely cannot supply a
publication date.

---

## 6. Features

Levels are largely firm fixed effects; changes carry the information. All
differencing is year-on-year (`t−4`) rather than sequential, so seasonality
cancels — a retailer does not have a weak Q1 because Q4 was strong.

Cross-sectional standardisation is applied **within a reporting cohort** (by
default a calendar month of `t0`) rather than pooled across the sample. Two
reasons: pooling lets the model spend capacity learning which year it is, and a
same-day cross-section is too thin to rank meaningfully. Ranks are mapped
through the inverse normal CDF, which is robust to the fat tails that
fundamentals data is full of.

EPS growth uses `(x_t − x_{t−4}) / |x_{t−4}|`. The absolute value in the
denominator keeps the sign meaningful when a firm swings from a loss to a
profit, which the naive ratio gets backwards.

---

## 7. Model evaluation

**Expanding-window walk-forward.** Fold *k* trains on everything up to a cutoff
and tests on the following year; the cutoff then rolls forward. This mirrors how
the strategy would have been run.

**Purging.** A 20-day label observed at `t0` is not resolved until `t0+20`.
Training events whose label window reaches into the test period are removed,
because their labels contain test-period information.

**Embargo.** A further gap (default 25 sessions, ≥ the label horizon) is imposed
before the test window, to break serial correlation across the boundary.

**Metrics.** R² is close to useless here — a signal explaining 0.5% of the
variance of 20-day abnormal returns can be very profitable. What matters is
ordering:

- rank IC per cohort, and its Newey–West t-statistic across cohorts;
- IC hit rate (fraction of cohorts with positive IC);
- quantile spread and its monotonicity across quantiles.

A high mean IC with a low t-statistic means the signal worked in a few periods,
which is not a strategy.

---

## 8. Portfolio construction and costs

**Trailing breakpoints.** Quantile cut-offs come from a 252-day trailing window
of *past* events. Ranking a firm against others reporting later in the same
month is a look-ahead, and a surprisingly common one.

**Entry offset.** Positions open one session after `t0`. The announcement move
happens in the opening auction; booking it is the largest single source of
overstated PEAD backtests.

**Overlapping portfolios.** Positions are held 20 sessions and new ones open
daily, so the book is a mixture of up to 20 vintages (Jegadeesh–Titman).

**Sector neutrality at book level.** Because vintages overlap, entries do not
balance within sector by construction. Neutrality is therefore imposed on the
whole live book each day: weights are demeaned within `(date, sector)` before
the gross exposure is scaled. The resulting net sector exposure is zero to
machine precision, which is asserted in the tests.

**Costs.** One-way, in basis points of traded notional:

```
cost = half_spread + commission + impact_coef · sqrt(participation)
```

The square-root impact term is the standard functional form. Traded notional is
the day-on-day change in the weight vector summed in absolute value, which
captures entries, exits *and* the rebalancing implied by daily neutralisation —
a turnover figure based on entries alone would understate it.

Defaults (3.0 bp half-spread, 0.5 bp commission, 10 bp impact coefficient at 5%
participation) are assumptions about liquid US large caps, not measurements. A
sensitivity curve over cost multiples 0.5×–3× is reported so the conclusion can
be stated as "this survives costs up to N bp one-way" rather than as a single
unverifiable number.

---

## 9. What would falsify the result

Stated in advance, because deciding afterwards is how a result becomes a story:

- the `[1,20]` sector-neutral quintile spread is insignificant under the cluster
  bootstrap;
- the spread is significant pooled but the IC t-statistic across cohorts is not,
  i.e. it lives in a handful of periods;
- net Sharpe crosses zero below a 2× cost multiple;
- the effect disappears when the sample is restricted to events with a
  *declared* (not inferred) announcement time;
- the effect disappears in the second half of the sample, which is what the
  published decay of PEAD since roughly 2004 would predict.

Any of these is a finding worth reporting. None of them is a reason to search
for a specification that avoids it.
