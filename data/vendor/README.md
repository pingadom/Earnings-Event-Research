# Vendor drop folder

Export from Capital IQ / LSEG Workspace / Finaeon, drop the files here, and the
pipeline handles the rest: column mapping, schema validation, timezone
localisation and point-in-time stamping.

```
data/vendor/
  prices/          daily OHLCV + adjusted close
  events/          earnings announcement dates AND times
  fundamentals/    long form: one row per (ticker, period, line item)
  consensus/       analyst EPS estimates, each stamped with its as-of date
  filings/         filing metadata + a path to the document text
  universe/        point-in-time index membership
```

Any number of `.csv` / `.xlsx` / tab-delimited `.txt` files per folder; they are
concatenated. Nothing here is committed — vendor data is licensed.

Validate what you have:

```bash
eee vendor-check --provider capitaliq
eee vendor-check --provider lseg
```

`prices` and `events` are required; every other folder reports `--` when empty
and the pipeline runs without it.

---

## Capital IQ (Excel plug-in)

**Prices** — one row per ticker-date:

| Column | Mnemonic |
|---|---|
| `Ticker` | — |
| `Date` | — |
| `IQ_OPENPRICE` `IQ_HIGHPRICE` `IQ_LOWPRICE` `IQ_CLOSEPRICE` | daily OHLC |
| `IQ_CLOSEPRICE_ADJ` | **required** — split/dividend adjusted |
| `IQ_VOLUME` | shares |

**Events** — `IQ_EARNINGS_ANNOUNCE_DATE`, `IQ_EARNINGS_ANNOUNCE_TIME`,
`IQ_PERIODDATE`. Export the *time* field, not just the date: without it every
event falls back to the assumed-AMC default and the whole event window can be
off by a session.

**Fundamentals** — long form with `Item` and `Value`, plus `IQ_PERIODDATE` and
`IQ_FILINGDATE`. Use the **point-in-time** financials dataset if your licence
includes it; the standard fields are restated.

**Consensus** — the pull this project cares most about, and the one easiest to
get wrong.

| Column | Mnemonic |
|---|---|
| `Ticker` | — |
| `Period End` | `IQ_PERIODDATE` |
| `As Of Date` | **required** — see below |
| `Consensus EPS` | `IQ_EPS_EST` |
| `Estimate StdDev` | `IQ_EPS_EST_STDDEV` |
| `Num Estimates` | `IQ_EPS_NUM_EST` |

`IQ_EPS_EST` returns the consensus **as it stands today** unless you pass an
`asOfDate`. Pull it without one and every historical quarter gets a forecast
formed after the actual was known — a forecast that already contains the
answer. The result is a spectacular backtest and a worthless one.

So the as-of date is not metadata here, it is the data. The adapter **refuses**
to load a consensus export that has no as-of column, rather than assuming a lag
the way it does for fundamentals: no lag repairs a number that was computed in
2026 and labelled 2019.

`docs/capiq-pull-specification.xlsx` (regenerate with `make capiq`) pre-computes
an as-of date one session before each announcement, one row per event, and
carries live check formulas that flag a returned period end or actual that
disagrees with what was expected. Run its pilot batch first.

Downstream, `build_surprise_features` stamps each row with the **later** of the
snapshot date and the reported figure's date, and warns when snapshots post-date
the figure they forecast. If that warning fires across a material share of the
sample, the pull was run without `asOfDate` — fix the pull, do not silence the
warning.

**Universe** — `IQ_INDEX_CONSTITUENTS` looped over month-ends, collapsed into
`(ticker, start_date, end_date)` intervals.

---

## LSEG Workspace / Datastream

**Prices** — `Instrument`, `Date`, `Price Open/High/Low/Close`,
`Adjusted Close Price`, `Volume`.

**Events** — `Earnings Announcement Date`, `Announcement Time`,
`Period End Date`. I/B/E/S carries a proper **Original Announcement Date**;
export it, because it is exactly the point-in-time stamp this pipeline needs.

**Fundamentals** — long form with `Field` and `Value`, `Period End Date` and
`Original Announcement Date`.

**Consensus** — the single most valuable thing this licence gives you, and the
free stack cannot produce it at all. Export `Instrument`, `Period End Date`,
`Mean Estimate`, `Standard Deviation`, `Number of Estimates` and — the one that
matters — `Estimate Date`, into `consensus/`. I/B/E/S summary files are already
dated snapshots, so the as-of trap described under Capital IQ above is easier to
avoid here; export the statistical period date and it is avoided outright.

**Universe** — `Index Join Date` / `Index Leave Date` from Datastream's
constituent lists.

---

## Finaeon / GFD

`Symbol`, `Date`, OHLC, `Adjusted Close`, `Volume` for prices; `Start Date` /
`End Date` / `Sector` for universe. Its comparative advantage here is long-
history index membership **including delisted names** — the input that makes
the study survivorship-bias free.

---

## Column names not matching?

Edit the mapping in `src/earnings_engine/data/providers/vendor.py`
(`CAPITALIQ_MAP`, `LSEG_MAP`, `FINAEON_MAP`) rather than renaming columns in
Excel. Keeping the mapping in version control is what makes the ingestion
reproducible six months from now.

Already-canonical column names (`ticker`, `date`, `adj_close`, …) pass through
untouched, so a hand-built CSV works without any mapping.

---

## ⚠️ Restated vs point-in-time

A standard fundamentals export gives the **latest restated** figures, not what
was on the tape at the time. Restatements are not random — firms that restate
downward are disproportionately those that were doing badly — so using them
gives the model a cleaner view of the past than anyone had, and biases results
upward.

If the export has no first-disclosure date, the adapter logs a warning, stamps
a conservative assumed filing lag, and marks the rows `_restated=True`. Carry
that caveat into the write-up. See [`../../docs/biases.md`](../../docs/biases.md).
