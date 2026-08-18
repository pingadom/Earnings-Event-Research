# Data sources

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
