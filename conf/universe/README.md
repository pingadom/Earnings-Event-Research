# Point-in-time universe files

Drop a CSV here with these columns and point `universe.membership_file` at it:

```csv
ticker,start_date,end_date,sector
AAPL,1982-11-30,2100-01-01,Information Technology
FRC,2010-12-20,2023-05-01,Financials
```

`end_date` is the last date the company was a constituent — for names still in
the index, use a far-future date. Deleted names **must** be present with their
real `end_date`; that is the entire point of the file.

## Getting one

**Capital IQ** — `IQ_INDEX_CONSTITUENTS` with a historical as-of date, looped
over month-ends, then collapsed into intervals. The Excel plug-in's screening
tool will do the loop for you.

**LSEG Workspace** — the index chain (`0#.SPX`) supports historical dates;
Datastream's `LS&PCOMP` list constituents with join/leave dates is cleaner if
you have Datastream access.

**Finaeon / GFD** — carries long-history index membership including delisted
names, which is its main advantage over the other two for this purpose.

## If you cannot get one

Set `universe.allow_static_membership: true` and accept that every result is
survivorship-biased. The code will let you, and it will label every output as
biased so the caveat reaches the write-up. See `docs/biases.md`.
