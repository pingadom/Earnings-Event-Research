# earnings-event-engine

[![CI](https://github.com/pingadom/Earnings-Event-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/pingadom/Earnings-Event-Research/actions/workflows/ci.yml)
[![Dashboard](https://img.shields.io/badge/dashboard-live-0b6bcb)](https://pingadom.github.io/Earnings-Event-Research/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Reproducible](https://img.shields.io/badge/results-hash--verified-brightgreen)](scripts/reproduce.py)


**Does the information in an earnings release predict the *abnormal* return that
follows it?**

Not "can a neural network predict stock prices". A narrower, answerable
question: after a company reports, is there information in the numbers and in
the language of the filing that predicts how the stock does relative to its
sector over the next one, five and twenty trading days — and does any of it
survive transaction costs, sector neutralisation, and an evaluation scheme that
never lets the model see the future?

This repository is the machinery for answering that properly.

```
universe → prices/events/fundamentals/filings
         → event alignment  → abnormal returns (CAR/BHAR, three estimators)
         → point-in-time features (fundamental Δ, SUE, NLP)
         → purged walk-forward model → signal
         → sector-neutral long/short book → transaction costs → evaluation
```

### [→ Results](docs/results.md)

**The headline is a null.** 114 S&P 500 companies, **3,323 real earnings
announcements** timestamped from SEC filings, held out one year at a time
2019–2024:

| | |
|---|---:|
| Mean out-of-sample IC | 0.024 (t = 1.17) |
| Years with positive IC | 4 / 6 |
| Net Sharpe after costs | −0.61 (t = −1.16) |
| Alpha vs FF5 + momentum | −0.81% (t = −1.36) |
| **IC trend per year** | **−0.023 (p = 0.033)** |

The hypothesis is **not supported on this sample**. The substantive finding is
the decay: positive 2019–2022, negative 2023–2024, significant at 5%. That is
what the literature predicts for an anomaly documented since 1968 and widely
traded since the mid-2000s.

Three of the five falsification criteria written down *before* the run are met.
The factor regression finds no alpha but does find unintended value and
investment tilts. Fama–MacBeth agrees with the portfolio sort. And
[§R5](docs/results.md#r5-the-limitation-that-matters-most) quantifies the bias
that survived: Yahoo serves no price history for **61% of the index-deleted
names** in the sample, against 5% of survivors — including SIVB and FRC.

The same pipeline run on synthetic data with a *known* planted effect recovers
it in 6/6 years at a net Sharpe of 4.8, and returns 0.02 / 0.37 when nothing is
planted. That contrast is why the null above is worth believing.

---

## Quickstart

```bash
git clone https://github.com/pingadom/Earnings-Event-Research.git
cd Earnings-Event-Research
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                                  # full suite, offline
eee demo    --out reports/demo          # one pass end to end
eee holdout --out reports/holdout       # six held-out years + dashboard
```

`eee demo` runs the entire pipeline end to end on a synthetic market with a
known data-generating process, with no network access and no data vendor. It
writes a markdown report, five figures and the intermediate CSVs to
`reports/demo/`. It takes about twenty seconds, and is deterministic — the same
command gives the same numbers on any machine.

```
eee demo                  →  out-of-sample IC 0.19,  net Sharpe  5.3
eee demo    --drift 0     →  out-of-sample IC 0.02,  net Sharpe  0.2

eee holdout               →  6/6 years positive, mean IC 0.197, net Sharpe 4.82
eee holdout --drift 0     →  4/6 years positive, mean IC 0.022, net Sharpe 0.37
```

`eee holdout` also writes a self-contained interactive `dashboard.html` —
per-year results, calibration, equity curve, dark mode, no dependencies.

> The demo's headline numbers are **not a research finding**. They are a
> statement that the machinery recovers an effect that was deliberately planted
> in the data. `eee demo --drift 0` plants no effect, and the reported Sharpe
> should collapse to approximately zero. Both cases are asserted in the test
> suite. That pair of tests is the point of the synthetic provider.
>
> Keep `--n-tickers` at 100 or above. Below that the daily book holds only a
> handful of names and the reported Sharpe is dominated by small-sample noise
> in both directions — which is itself a fair illustration of why thin
> cross-sections cannot support this kind of claim.

To use real data:

```bash
cp .env.example .env         # add your SEC_USER_AGENT
python scripts/download_data.py --tickers AAPL MSFT JPM --start 2015-01-01
eee event-study --provider local --tickers AAPL,MSFT,JPM --start 2015-01-01 --end 2026-08-18 --out reports/study
```

---

## What makes this different from a coding project

Most "predict stock returns" projects fail in the same four places. Each one has
a named defence here, and a test that fails if the defence is removed.

| Failure mode | What goes wrong | Defence |
|---|---|---|
| **Look-ahead bias** | Fundamentals are joined on the period they describe, not on the date they were published — so the model uses a Q1 result three weeks before anyone had it. | Every fact carries `available_from_utc`; `restrict_to_known` is an as-of join in *publication time*; `assert_point_in_time` raises on any violation. `tests/test_lookahead.py` plants leaks and asserts they are caught. |
| **The untradable gap** | The announcement move happens in the opening auction. Backtests that hold from `t0` book a jump nobody could transact. | Positions open `entry_offset=1` sessions after `t0` by default, and the model target is a drift window `[1, 20]`, not `[0, 19]`. |
| **Survivorship bias** | Building the universe from today's index constituents deletes every company that blew up. | `Universe` is an interval table queried *as of* a date. A static constituent list raises `SurvivorshipBiasError` unless explicitly acknowledged, and tags every downstream result as biased. |
| **Leakage in evaluation** | k-fold cross-validation trains on 2022 to predict 2016, and splits overlapping 20-day labels across the fold boundary. | Expanding-window walk-forward with label purging and an embargo (López de Prado). Every fold's training set ends before its test set begins, with a gap ≥ the label horizon. |

Two more that get less attention and matter just as much:

- **Cross-sectional correlation.** Hundreds of firms report in the same
  fortnight and share common shocks, so events are not independent draws.
  Significance is reported with a **cluster bootstrap over event dates**
  alongside the naive t-test, and the two are usually meaningfully different.
- **Event-induced variance.** Return variance *jumps* on announcement days, so
  the standard Patell-style test over-rejects. The
  **Boehmer–Musumeci–Poulsen** standardised cross-sectional test is reported
  alongside.

---

## Abnormal returns, three ways

"Abnormal return" is not one quantity — it is whatever is left after you
subtract a benchmark, and the benchmark is a modelling choice. Reporting a
single number hides how much of the result is the choice rather than the data.
All three are always computed:

| Estimator | Definition | What it controls for |
|---|---|---|
| `market_adjusted` | `r_i − r_m` | Nothing but the market. No parameters, nothing to overfit. |
| `market_model` | `r_i − (α_i + β_i·r_m)` | Market exposure, with α and β from a `[t0−250, t0−31]` estimation window. |
| `sector_neutral` | `r_i − mean(r_j : j ∈ sector, j ≠ i)` | Sector moves. **The one the hypothesis actually needs.** Excluding the stock from its own benchmark is essential; including it shrinks the measured effect mechanically. |

Each is reported as both CAR (summed) and BHAR (compounded), over `[0,0]`,
`[0,4]`, `[0,19]`, `[1,5]` and `[1,20]`. The `[0,*]` windows contain the
announcement gap and are for *measurement*; the `[1,*]` windows are the
tradable drift and are the only legitimate model targets.

---

## Features

**Fundamental** — changes, not levels, and differenced against the *same
quarter one year ago* so seasonality cancels: revenue growth and its
acceleration, gross/operating/net margin deltas, EPS growth (sign-safe when
earnings cross zero), free cash flow margin, accruals, net debt, asset growth,
share count change.

**Surprise** — standardised unexpected earnings. Analyst-based SUE when a
consensus history is available (Capital IQ / LSEG carry one); otherwise a
seasonal-random-walk-with-drift SUE (Foster–Olsen–Shevlin) computed from
strictly-past quarters, so the pipeline never silently degrades to nothing when
consensus data is missing.

**Text** — Loughran–McDonald dictionary tone, uncertainty and litigiousness;
the *change* in each against the firm's previous filing (the level is mostly a
firm fixed effect); and TF-IDF cosine similarity to the previous filing, the
"Lazy Prices" effect — firms rewrite their boilerplate when something is wrong.
A compact lexicon ships with the package so this works offline; point
`features.lm_dictionary_path` at the full Notre Dame master dictionary for
anything you intend to report.

All features are ranked within a reporting cohort and mapped through the
inverse normal CDF, so the model learns "how does this quarter compare to the
quarters reported around it" rather than "what year is it".

---

## Data

The provider layer is four narrow protocols (`PriceProvider`, `EventProvider`,
`FundamentalProvider`, `FilingProvider`), so swapping a source is a config
change.

| Provider | Cost | Notes |
|---|---|---|
| `synthetic` | free | Deterministic generated market. Hermetic tests, offline demo, known ground truth. |
| `edgar` | free | SEC filings and XBRL fundamentals. **`acceptanceDateTime` gives a genuine minute-level point-in-time stamp**, and `companyfacts` preserves first-reported values so restatements do not overwrite history. This is why the study is US-first. |
| `yahoo` | free | Prices and earnings dates. Fine for development; announcement *times* are unreliable, and index membership is current-only. |
| `capitaliq` | licensed | Excel plug-in exports dropped into `data/vendor/`. |
| `lseg` | licensed | Workspace / Datastream exports, same drop folder. |
| `finaeon` | licensed | Long-history exports; the best of the three for point-in-time index membership. |

Capital IQ, LSEG and Finaeon have no free programmatic API, so ingestion is
built around an explicit **vendor drop folder** with declared column mappings,
schema validation and a stated point-in-time policy — see
[`data/vendor/README.md`](data/vendor/README.md) for the export recipes, and
run `eee vendor-check` to validate what you have dropped.

⚠️ A standard vendor fundamentals export gives you the *latest restated*
figures, not what was on the tape. The vendor adapter detects this, logs it
loudly, and tags the data as restated so the caveat reaches the write-up. See
[`docs/biases.md`](docs/biases.md).

---

## Layout

```
conf/config.yaml            every knob, validated at load time
src/earnings_engine/
  config.py                 frozen dataclasses; unknown keys are load-time errors
  utils/calendar.py         NYSE sessions, holidays, half days, special closures
  utils/frames.py           schema contracts between pipeline stages
  data/                     provider protocols, providers, universe
  events/alignment.py       announcement → first tradable session
  events/pit.py             the point-in-time invariant
  returns/                  market model, CAR/BHAR, BMP, cluster bootstrap
  features/                 fundamentals, surprise, text, assembly
  models/walkforward.py     purged expanding-window splitter
  backtest/                 sector-neutral book, costs, performance
  reporting/                figures and the run report
  holdout.py                rolling annual holdouts: train Y-1, freeze, predict Y
  analysis/attribution.py   alpha vs Fama-French 5 + momentum, Newey-West
  analysis/multiple_testing.py  probabilistic & deflated Sharpe, trials log
  analysis/fama_macbeth.py  cross-sectional regressions, second methodology
  reporting/dashboard.py    self-contained interactive HTML, no dependencies
  cli.py                    eee demo | holdout | download | event-study | research
tests/                      168 tests, offline, including leak regressions
docs/results.md             the holdout evidence, with its null control
docs/manifest.json          SHA-256 of every published artefact
scripts/reproduce.py        make reproduce | make verify
docs/                       methodology, biases, data sources, decision records
```

The statistical core (OLS market model, Newey–West, BMP, bootstrap) is written
directly on numpy/scipy rather than pulling in statsmodels, so the package has
six dependencies and CI runs with no network at all. `pip install -e ".[stats]"`
adds statsmodels purely so the tests can cross-check those implementations.

---

## Reading the results honestly

The significance table reports **every estimator × every window × three tests**,
not the best cell. If an effect appears only under one estimator at one horizon,
that grid makes it obvious. `eee research` additionally reports:

- out-of-sample information coefficient per cohort, with a Newey–West t-stat —
  a high average IC driven by three good quarters is not a strategy;
- **calibration** — predicted magnitudes against realised ones. A signal can rank
  correctly and still be badly scaled, and position sizing depends on which;
- **factor attribution** — alpha and its HAC t-statistic against Fama–French 5
  plus momentum, so "is this just momentum?" has an answer in the repo rather
  than in the interview;
- **multiple testing** — a committed log of every specification tried, and the
  deflated Sharpe ratio it implies. When the trial dispersion cannot be
  estimated the code says so instead of guessing;
- **Fama–MacBeth** — the same question by cross-sectional regression. Two
  methods agreeing is much harder to dismiss than one; where they disagree,
  [`results.md` §5d](docs/results.md) explains why;
- gross **and** net equity curves;
- a cost-sensitivity curve answering "how wrong would my cost assumption have to
  be for this edge to disappear?", which converts an unverifiable assumption
  into a falsifiable statement.

Post-earnings-announcement drift is one of the most-studied anomalies in
finance, documented since Ball and Brown (1968) — and substantially decayed
since roughly 2004 as it became widely traded. A modest, cost-sensitive result
is the expected outcome. A Sharpe of 3 means you have found a bug, and the
first place to look is the four failure modes in the table above.

---

## Further reading

- Ball & Brown (1968), *An Empirical Evaluation of Accounting Income Numbers*
- Bernard & Thomas (1989), *Post-Earnings-Announcement Drift*
- Brown & Warner (1985), *Using Daily Stock Returns: The Case of Event Studies*
- Boehmer, Musumeci & Poulsen (1991), *Event-Study Methodology under Conditions of Event-Induced Variance*
- Loughran & McDonald (2011), *When Is a Liability Not a Liability?*
- Cohen, Malloy & Nguyen (2020), *Lazy Prices*
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7

---

## Real Data Acquisition

The production downloader writes consolidated, validated Parquet files under
`data/raw/` and durable per-symbol work under `data/cache/`. A stopped run can
be restarted without discarding completed symbols. The research provider named
`local` reads only these files: it never contacts the network and never falls
back to synthetic data.

### Windows PowerShell setup

From a fresh clone:

```powershell
git clone https://github.com/pingadom/Earnings-Event-Research.git
cd Earnings-Event-Research

py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env
```

Set a real identity in `.env`; SEC fair-access policy requires it:

```text
SEC_USER_AGENT=Your Name your.email@domain.com
DATA_START_DATE=2010-01-01
DATA_END_DATE=
```

Then acquire the full historical universe and validate every local output:

```powershell
python scripts/download_data.py
python scripts/download_data.py --validate-only
```

Useful resumable subsets:

```powershell
python scripts/download_data.py --start 2010-01-01 --end 2025-12-31
python scripts/download_data.py --tickers AAPL MSFT NVDA
python scripts/download_data.py --prices-only
python scripts/download_data.py --sec-only
python scripts/download_data.py --earnings-only
python scripts/download_data.py --force-refresh
```

The smallest end-to-end real-data smoke test is:

```powershell
python scripts/download_data.py --tickers AAPL MSFT NVDA JPM XOM --start 2020-01-01
```

Run the existing analysis fully offline after acquisition:

```powershell
eee event-study --provider local --tickers AAPL,MSFT,NVDA,JPM,XOM --start 2020-01-01 --end 2026-08-18 --out reports/real-smoke
```

### Outputs

- `prices.parquet`: equity OHLCV, raw close, adjusted close, and row-level source.
- `earnings.parquet`: Yahoo fields reconciled to exact SEC Item 2.02 acceptance timestamps where possible.
- `fundamentals.parquet`: standardized wide SEC facts with filing time, accession, and concept provenance.
- `sec_filings.parquet`: submission metadata and traceable EDGAR document URLs.
- `fama_french_daily.parquet`: FF3, RMW, CMA, momentum, and RF in decimal units.
- `benchmarks.parquet`: `SPY` and `^GSPC` stored separately from securities.
- `index_membership.parquet`: interval membership, including deleted and renamed historical symbols when the source has them.
- `acquisition_manifest.json` and `validation_report.json`: parameters, failures, coverage, and validation findings.

### Methodological safeguards

Fundamentals are joined in publication time, never on `period_end`. Raw SEC
amendments remain accession-level observations; the canonical research adapter
uses the earliest public filing for a period/item. An annual additive flow is
converted to Q4 only by the exact `FY - Q1 - Q2 - Q3` residual and only when all
three standalone quarters exist; missing components remain missing, and annual
EPS is never quarterised. Earnings with an exact time are mapped to the legal
next tradable open by the existing calendar logic.
Date-only events remain labelled unknown and use a conservative next-session
policy. Returns use Yahoo `Adj Close`; raw `Close` remains available for audit,
and no missing price is forward-filled.

The default ticker universe is every historical S&P 500 interval overlapping
the requested window, rather than today's members. This reduces survivorship
bias but does not eliminate it: the free membership reconstruction starts in
1996, Yahoo often lacks delisted histories, and current-sector metadata is not
a historical GICS series. Every unavailable ticker is recorded rather than
replaced with synthetic data. Stooq is used only as a logged, row-labelled
whole-symbol fallback and can be disabled with `--no-stooq-fallback`.

Free sources do not reliably provide complete historical revenue consensus,
precise Yahoo release times, all delisted-company prices, or licensed
point-in-time sector classifications. Filing bodies are not bulk-downloaded by
default because multi-decade 10-K/10-Q text is very large; offline real-data
runs therefore skip text features explicitly unless local filing text has been
provided. See [docs/data_sources.md](docs/data_sources.md) for endpoint-level coverage and
limitations.

## Licence

MIT. Not investment advice; nothing here is a recommendation to trade.
