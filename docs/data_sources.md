# Data sources

Two halves. **[§A](#a-what-the-pipeline-actually-downloads)** is what the
acquisition layer downloads today, with each source's known limitations and
the point-in-time policy applied to it. **[§B](#b-choosing-a-source)** is the
reasoning behind those choices, and what the licensed terminal routes give you
that the free ones cannot.

---

# A. What the pipeline actually downloads

Everything below is produced by `eee download` and validated by
`earnings_engine.data.validation` before it lands in `data/raw/`.

| Dataset | Source | Coverage | Update method | Known limitations |
|---|---|---|---|---|
| Equity prices | Yahoo Finance via `yfinance` | Requested window; availability varies by symbol | Per-symbol incremental Parquet cache | Unofficial API; delisted/renamed symbols are often unavailable; adjusted close reflects the currently known corporate-action history |
| Price fallback | Stooq CSV endpoint | Requested window when Yahoo returns no usable symbol history | Whole-symbol fallback, logged and labelled `source=stooq` | The endpoint has no separate total-return adjusted-close field, so `adjusted_close=close` is explicit and must not be treated as verified dividend adjustment |
| Earnings | Yahoo `get_earnings_dates` plus SEC 8-K Item 2.02 | Yahoo usually exposes up to 100 events; SEC coverage depends on 8-K tagging | Per-symbol Yahoo cache; SEC acceptance-time reconciliation | Yahoo dates/times and analyst fields are incomplete; revenue estimates are usually absent; an Item 2.02 filing can lag the actual release |
| Fundamentals | SEC Company Facts/XBRL | Primarily 2009 onward; issuer/concept dependent | Sequential `data.sec.gov` pulls with cache, retries, and 8 req/s cap | Concept heterogeneity; quarterly cash-flow facts are often year-to-date; foreign/private issuers vary |
| Filing provenance | SEC submissions API and EDGAR Archives metadata | SEC filing history available through current and older submission shards | Sequential cached pull | Filing bodies are not bulk-downloaded by default; `sec_filings.parquet` stores URLs/accessions and precise acceptance timestamps where supplied |
| Fama-French | Kenneth French Data Library daily ZIP files | Library history through its latest published date | Cached ZIP download; `--force-refresh` updates | Publication can lag the latest equity session; factor values are converted from percent to decimal |
| Benchmarks | Yahoo Finance (`SPY`, `^GSPC`) | Requested window | Same resumable price mechanism | `SPY` is the practical total-return proxy; `^GSPC` is a price index, not a total-return series |
| S&P 500 membership | `fja05680/sp500` on raw.githubusercontent.com | Reconstructed intervals from 1996 | Download `sp500_ticker_start_end.csv`; current metadata from `sp500.csv` | Community reconstruction, not an official S&P product; early years may be incomplete; current sector metadata is not a point-in-time sector history |

## Exact endpoints/hosts

- `query1.finance.yahoo.com` / `query2.finance.yahoo.com` (indirectly through `yfinance`)
- `https://stooq.com/q/d/l/`
- `https://data.sec.gov/submissions/`
- `https://data.sec.gov/api/xbrl/companyfacts/`
- `https://www.sec.gov/files/company_tickers.json`
- `https://www.sec.gov/Archives/edgar/data/` (provenance URLs only)
- `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/`
- `https://raw.githubusercontent.com/fja05680/sp500/master/`

## Point-in-time interpretation

`period_end` says what interval a financial statement describes. It is never
the date the market knew the statement. `filing_date` is retained separately,
and `available_from_utc` uses SEC `acceptanceDateTime` where supplied. Each
accession remains a distinct raw row, so an amendment/restatement does not
overwrite the originally filed observation. The offline research adapter
selects the earliest public value for each period/item.

The research adapter converts an annual additive flow to a standalone Q4 value
only when Q1, Q2, and Q3 standalone values are all present, using the exact
residual `FY - Q1 - Q2 - Q3`. If any component is unavailable, Q4 remains
missing. Annual EPS is never quarterised because weighted-share denominators
make subtraction invalid. This policy does not interpolate financial statements.

For date-only Yahoo earnings events, `announced_at_utc` remains null in
`earnings.parquet`. The offline adapter applies an auditable, conservative
end-of-day timestamp, retains `timing=unknown`, and the existing event aligner
moves the tradable event to the next session. It never pretends the derived
timestamp was observed.

---

# B. Choosing a source

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
