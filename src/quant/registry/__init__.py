"""registry — strategy registry and promotion gates.

The registry is the audit trail of every strategy variant we've ever tested.
Two reasons this matters:

1. **Honest DSR.** The Deflated Sharpe Ratio correction needs the *true*
   number of trials. If we silently discard losing variants, we deceive
   ourselves about how lucky a winner is. The registry records every variant,
   passed or failed.
2. **Promotion gates.** A strategy moves through stages:
       research -> backtest -> walk-forward -> paper -> live
   Each transition requires meeting specific criteria (e.g. positive OOS
   Sharpe after DSR, max drawdown under threshold, 90 days of paper trading
   with tracking error below tolerance). The registry enforces these gates;
   nothing flips to live capital by accident or by a developer's say-so.

Backing store: a versioned SQLite/Parquet file under data/registry/ (TBD).
"""
