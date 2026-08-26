# Biases: what they are, where they enter, what this repo does about them

Every one of these produces a *better-looking* result. That asymmetry is why
they survive: nobody investigates a backtest that looks disappointing.

---

## 1. Look-ahead bias

**What it is.** Using information in a decision before it was available.

**How it enters here.** Almost never dramatically. The usual route is joining
fundamentals on `period_end` instead of on the date they were published. A Q1
result dated 31 March is not public until early May; join it on 31 March and the
model gets a five-week head start on every observation in the sample.

Subtler routes:

- computing a rolling mean or standard deviation without shifting it, so the
  current observation contributes to its own expectation;
- fitting a scaler or imputer on the whole panel before splitting;
- restatements — a vendor's "2016 revenue" may be a figure published in 2019;
- ranking a firm against peers who had not yet reported.

**Defence.** Every fact carries `available_from_utc`. `restrict_to_known`
performs an as-of merge in publication time, and `assert_point_in_time` raises
on any violation. All preprocessing lives inside a scikit-learn `Pipeline` so it
is fitted per fold. Expanding-window statistics are explicitly shifted.
Portfolio breakpoints come from a trailing window of past events.
`tests/test_lookahead.py` plants each of these leaks and asserts it is caught.

**Residual risk.** `available_from_utc` is only as good as the source. EDGAR's
`acceptanceDateTime` is a real minute-level stamp. A vendor export with only a
period date is not, and the vendor adapter says so loudly.

---

## 2. Survivorship bias

**What it is.** Selecting a universe on a condition that is only knowable later
— usually "is in the index today".

**Magnitude.** Not small. Roughly 20–30 names leave the S&P 500 every year. Over
a decade, a "current constituents" universe is missing several hundred
company-years, and they are systematically the worst ones: firms deleted after
collapsing, being acquired at a discount, or falling out of the size threshold.
Every firm still in the list survived, and many were *added* precisely because
they had already performed.

**Defence.** `Universe` is an interval table `(ticker, start_date, end_date)`
queried as of a date. `Universe.static(...)` raises `SurvivorshipBiasError`
unless `acknowledge_bias=True`, and when acknowledged it sets
`static_membership=True`, which propagates into the run report.

**Getting a point-in-time membership file.** See
[`conf/universe/README.md`](../conf/universe/README.md). Capital IQ, LSEG
Datastream and Finaeon can all produce one; Finaeon is the most convenient for
long histories including delisted names.

**Related: delisting returns.** A stock that is acquired or goes to zero stops
having prices. If the position is simply dropped, the loss is never booked. Free
price sources generally do not carry delisting returns; CRSP does. Where they
are unavailable, the honest move is to say so and note the direction of the bias
(it flatters long portfolios).

**Measured, not asserted, in this repository.** The point-in-time universe is
built correctly, so the *membership* is unbiased. The prices are not. Of the
names the membership table says were deleted from the index during the sample,
61% could not be retrieved from Yahoo at all, against 5% of the names that
survived — a twelvefold difference, and precisely the wrong twelve. Silicon
Valley Bank and First Republic are both absent: two firms whose earnings
disclosures were followed by exactly the kind of move this study is trying to
detect.

Two escape routes were attempted and both are closed from a free source:

* Yahoo's chart endpoint returns `404 — symbol may be delisted` for every one
  of them, and for a handful of names that are demonstrably still trading, so
  its coverage is not even reliably a function of delisting.
* Stooq, the configured fallback, now answers automated requests with a
  JavaScript proof-of-work challenge. That is a deliberate statement about
  automated access, and this project does not defeat it.

So the bias is quantified, its direction is known — it flatters the results —
and the fix is named rather than improvised: a price source with delisting
coverage. CRSP through a university subscription, or Finaeon. Until then this
is the limitation that would most change the answer, and it is stated in the
results rather than left for a reader to discover.

**The fix, itemised.** Capital IQ retains delisted securities, and the licence
to hand is a Capital IQ one. Sheet 2 of
[`capiq-pull-specification.xlsx`](capiq-pull-specification.xlsx) (regenerate
with `make capiq`) lists every universe member the price store does not cover,
with the window each one needs. That turns "get a better price source" from an
intention into a request someone can actually run. Until it is run, the number
above stands.

---

## 3. The untradable announcement gap

**What it is.** Treating the announcement-day price move as capturable.

**Why it matters more than it sounds.** The same-day reaction to an earnings
surprise is roughly an order of magnitude larger than the subsequent 20-day
drift. It happens in the opening auction. A backtest that holds from `t0`
therefore books a large, entirely untradable return, and it will dominate
everything else in the P&L.

**Defence.** `backtest.entry_offset = 1` and a default model target of
`car_market_model_1_20`. `tests/test_backtest.py` asserts the daily book's
minimum relative day is 1. `[0,*]` windows are still computed and reported —
they are the right thing for *measuring* the reaction — they are simply not
tradable and are never model targets.

---

## 4. Data-snooping / multiple testing

**What it is.** Trying enough specifications that one works by chance. With
three estimators, five windows, five model families and a dozen feature subsets,
there are hundreds of implicit tests; at a 5% threshold, several will "work".

**Defence.** Partly structural, partly discipline:

- the significance table reports the **whole grid**, so a result that exists in
  one cell is visibly a result that exists in one cell;
- falsification criteria are stated in advance in
  [`methodology.md` §9](methodology.md#9-what-would-falsify-the-result);
- the walk-forward scheme means every hyperparameter choice made after seeing
  test-period results is a form of snooping — so record them in
  [`docs/adr/`](adr/) as they are made, and count how many you made.

**Honest framing for a write-up.** "We tested one pre-registered specification
and report all supporting analyses" is a strong claim. "We report the best of
many" is a weak one. Which of these is true is determined by what you did, not
by how you write it up.

---

## 5. Restatement bias

**What it is.** Using accounting figures as later corrected rather than as
originally reported.

**Why it biases upward.** Restatements are not random. Firms that restate
downward are disproportionately those that were doing badly. Using restated
figures gives the model a cleaner, more accurate view of fundamentals than
anyone had at the time — and one that correlates with the outcome.

**Defence.** The EDGAR provider keeps the **first-reported** value for each
`(ticker, period, item)` and ignores later revisions. The vendor adapter checks
whether the export carries a first-disclosure date; if not, it logs a warning,
stamps a conservative assumed filing lag, and marks the rows `_restated=True`.

---

## 6. Cross-sectional dependence

**What it is.** Treating clustered events as independent observations.

**Why it matters.** Earnings announcements bunch: a large fraction of the index
reports within a three-week window each quarter. Those firms share macro and
sector shocks, so the effective sample size is much closer to the number of
*event dates* than to the number of events. A naive t-test over 5,000 clustered
events can be badly overconfident.

**Defence.** Cluster bootstrap over event dates, reported next to the naive
t-test. Sector-neutral abnormal returns also remove a large share of the common
component before testing.

---

## 7. Transaction costs and capacity

**What it is.** Reporting gross performance, or netting an implausibly small
fixed cost.

**Defence.** Explicit cost model with a square-root impact term, traded notional
computed from the actual day-on-day weight changes, and a sensitivity curve over
cost multiples so the finding is stated as "survives up to N bp one-way".

**Capacity, which the cost model does not capture.** A signal that concentrates
in small, illiquid names has a low dollar capacity regardless of what the
backtest says. The universe filters (`min_price`, `min_dollar_volume`) and the
`participation` assumption bound this, but a proper capacity analysis means
re-running with a size-restricted universe and reporting how much of the edge
survives. Worth doing; not automated here.

---

## 8. Regime dependence and decay

**What it is.** Assuming an effect estimated over 2014–2024 is stable.

**Why it applies specifically to this study.** Post-earnings-announcement drift
has been documented since Ball and Brown (1968) and studied continuously since.
The published evidence is that it has *decayed substantially* since roughly
2004, as it became widely known and traded — which is what you would expect of a
real anomaly that got arbitraged. A study of 2014–2024 that finds a large PEAD
effect should be treated as suspicious rather than exciting.

**Defence.** Walk-forward evaluation exposes decay directly: the IC time series
and the per-fold results show whether the effect is fading. Report the second
half of the sample separately.

---

## Checklist before you believe a result

- [ ] Does it survive the cluster bootstrap, not just the t-test?
- [ ] Is it present under all three abnormal-return estimators?
- [ ] Is it present in `[1,20]`, not only in windows containing day 0?
- [ ] Is the IC t-statistic across cohorts significant, or is it a few periods?
- [ ] Does it survive a 2× cost multiple?
- [ ] Does it survive restricting to events with a declared announcement time?
- [ ] Is it present in the second half of the sample?
- [ ] Is the universe point-in-time, or is `static_membership` set?
- [ ] How many specifications did you try before this one?
