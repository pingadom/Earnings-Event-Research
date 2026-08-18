# Data sources

## Why the S&P 500 rather than the FTSE 350

Not because US markets are more interesting — because of one property of SEC
EDGAR. The submissions feed publishes `acceptanceDateTime`, the minute at which
a filing became publicly available, and the `companyfacts` API returns every
XBRL fact with the accession that first reported it. Together those give a
genuine point-in-time view of both *what* was reported and *when* it became
knowable, for free, back to roughly 2009.

Look-ahead prevention stops being an assumption and becomes something you can
verify. Nothing comparable exists for UK issuers: RNS announcements are
scattered across the LSE site and individual IR pages, with no structured
financial data and no free bulk access. A FTSE 350 version of this study is
possible, but the NLP half becomes a scraping project and the point-in-time
guarantees get much weaker.

The provider layer is deliberately narrow so a UK adapter can be added later
without touching anything downstream.

---

## SEC EDGAR (free)

**What you get.** 10-K/10-Q metadata with acceptance timestamps, full filing
text, and XBRL fundamentals from 2009 onward.

**Conditions of use.** A descriptive `User-Agent` containing a contact address
is required, and requests are limited to 10 per second. Both are enforced by
the provider; set `SEC_USER_AGENT` in `.env`.

**Gotchas.**
- Issuers use different XBRL tags for the same economic quantity and change
  which one they use over time. `TAG_MAP` in `edgar.py` lists candidates in
  priority order per line item; expect to extend it.
- `Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax` shifted
  with ASC 606 adoption around 2018.
- Quarterly flow items in a 10-K are often reported as annual figures; check the
  `fp`/`frame` fields before treating a Q4 value as a quarter.
- The provider keeps the **earliest** filed value per period, so restatements do
  not overwrite what the market actually saw.

---

## Yahoo Finance via `yfinance` (free)

**What you get.** Daily adjusted OHLCV and earnings dates.

**Use it for development, not for results.**
- Unofficial, undocumented endpoint; it breaks without notice.
- Earnings *times of day* are frequently missing or wrong, and a wrong BMO/AMC
  flag shifts the entire event window by a session.
- Adjusted prices are recomputed from today's corporate-action history, so they
  are not what was observable at the time.
- No point-in-time index membership.

---

## Capital IQ (licensed)

**What you get.** Prices, earnings announcement dates *with times*, consensus
estimates, standardised fundamentals, and historical index constituents.

**How to extract it.** No free programmatic API. The practical route is the
Excel plug-in: build a template with `IQ_*` mnemonics across your ticker list
and date range, export to CSV/XLSX, drop into `data/vendor/<dataset>/`.

**Point-in-time.** The standard financials are *restated*. Capital IQ does
offer point-in-time financials — use that dataset if your licence includes it.
Otherwise set `available_from_policy="filing_lag"` and accept the caveat.

**Most valuable single field for this project:** consensus EPS with estimate
dispersion, which enables analyst-based SUE. The free stack cannot produce it at
all.

---

## LSEG Workspace / Datastream (licensed)

**What you get.** Prices with proper adjustment history, I/B/E/S consensus
estimates (the reference dataset for earnings surprise research), TRBC sector
classifications, and index join/leave dates.

**How to extract it.** Workspace's Excel add-in or the Screener; Datastream
Navigator for constituent lists with join/leave dates. Export and drop into
`data/vendor/`.

**Point-in-time.** I/B/E/S carries a proper "original announcement date", which
is exactly the stamp this pipeline wants. Prefer the `.PIT` variants of
fundamental fields where your licence exposes them.

---

## Finaeon / Global Financial Data (licensed)

**What you get.** Very long price histories and — the reason it is worth using
here — index membership including delisted names.

**Best use in this project:** generating the point-in-time universe file, which
is the input the other two make hardest to get.

---

## The vendor drop folder

```
data/vendor/
  prices/          any number of .csv / .xlsx exports
  events/
  fundamentals/
  filings/
  universe/
```

Column mappings for each vendor live in `data/providers/vendor.py`
(`CAPITALIQ_MAP`, `LSEG_MAP`, `FINAEON_MAP`) — adjust them there rather than
renaming columns by hand in Excel, so the mapping is version-controlled and the
ingestion stays reproducible.

Validate what you have dropped:

```bash
eee vendor-check --provider capitaliq
```

Nothing under `data/` is committed. Vendor data is licensed; check your terms
before it leaves your machine, and never push it to a public repository.
