# Canonical columns

What every vendor export is mapped onto. A CSV using these names directly needs
no mapping at all.

## prices
`ticker, date, open, high, low, close, adj_close, volume`
Unique on `(ticker, date)`. `adj_close` must be split- and dividend-adjusted.

## events
`ticker, event_id, announced_at_utc, timing, period_end, fiscal_quarter`
Unique on `event_id`. `announced_at_utc` is timezone-aware UTC.
`timing ∈ {bmo, amc, during, unknown}`.

## fundamentals
`ticker, period_end, available_from_utc, item, value`
Long form, unique on `(ticker, period_end, item)`.
`available_from_utc` is **when the figure became public**, not the period end.

Recognised `item` values: `revenue, gross_profit, operating_income, net_income,
eps_diluted, cfo, capex, total_debt, cash, total_assets, shares_diluted`.

## filings
`ticker, accession, form, filed_at_utc, period_end, path`
Unique on `accession`.

## universe
`ticker, start_date, end_date, sector`
Interval membership. `end_date` far-future for current constituents; **deleted
names must be present with their real end date**.

## consensus (optional, for analyst SUE)
`ticker, period_end, consensus_eps, consensus_std, n_estimates, available_from_utc`
