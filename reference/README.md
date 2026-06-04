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
2. **Scripted:** `tools/curate_sp500_membership.py` builds this CSV from two
   Wikipedia-format inputs. See the script's docstring for the full workflow.
   Quick start:

       # 1. Paste Wikipedia tables into spreadsheets, save as CSV.
       #    https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
       # 2. Run the curator:
       uv run python tools/curate_sp500_membership.py \
           --current wikipedia_current.csv \
           --changes wikipedia_changes.csv \
           --out reference/universe/sp500.csv
       # 3. Diff, sanity-check, commit.

   Coverage: Wikipedia's "Selected changes" table is reliable back to ~2000.
   That's enough history for any realistic backtest window.

   **Quarterly refresh:** S&P announces index changes ~10-20x per year.
   After each batch, update `wikipedia_changes.csv`, re-run, diff, commit.

### How the agent uses this CSV

`quant.data.universe.load_active_universe(as_of)` returns the point-in-time
S&P 500 membership for any date. The live agent calls it daily; the weekly
HRP refit + monthly grid search use it for their backtest windows.

**Fallback behavior:** if the CSV has < 50 active members on `as_of`, the
loader falls back to `load_top50_snapshot()` (the static, survivorship-biased
list) and logs a warning. The monthly AI analyst's pipeline self-audit
also flags the fallback, so you'll see it in the email until the CSV is
comprehensive enough to support the live universe.

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
