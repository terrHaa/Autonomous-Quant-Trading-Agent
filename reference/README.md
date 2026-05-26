# Reference data

Static reference data that the platform reads at startup. Unlike `data/`
(gitignored — caches and run outputs), this directory **is** version-controlled
because the files here are part of the platform's definition of truth: change
one of them and every future backtest result changes.

## What's here

- `universe/sp500.csv` — S&P 500 membership history (starter set). Used by
  `quant.data.universe` to give point-in-time membership and prevent
  survivorship bias in backtests.

## Extending `sp500.csv`

The shipped file has ~15 well-known names — enough to demo the loader and
the survivorship-bias correction, but not exhaustive. Two paths to grow it:

1. **Manual:** look up additions and removals on Wikipedia ("List of S&P 500
   companies" → the "Selected changes" table), add rows to the CSV, keep the
   header columns the same.
2. **Scripted (TODO):** write `scripts/scrape_sp500.py` that parses the
   Wikipedia table. Coverage is good post-1976; pre-1976 the data thins out.

### Schema

```
symbol,added,removed
AAPL,1982-11-30,
LEH,1957-03-04,2008-09-15
```

- `added` is **inclusive** — the first day the symbol IS a member.
- `removed` is **exclusive** — the first day the symbol is NOT a member.
  (Lehman's row above means it's a member through 2008-09-14, not on 9-15.)
- Empty `removed` = still a member today.
- Symbols must be unique. The loader doesn't yet support multiple membership
  intervals per symbol (e.g., AIG was removed in 2008 and re-added in 2012).
  Names like that need a schema extension before they can be modeled honestly.
