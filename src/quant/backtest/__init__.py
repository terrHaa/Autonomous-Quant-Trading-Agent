"""backtest — event-driven backtest engine.

Design principles:
- **No leakage.** The engine advances time bar-by-bar and exposes to the
  strategy only data that would have been known at that bar's close. A
  strategy that asks for tomorrow's price gets an exception, not a number.
- **Realistic fills.** Orders submitted at bar close fill at the *next* bar's
  open with configurable slippage. Market-on-close orders fill at the same
  bar's close minus impact.
- **Deterministic.** Same config + same data + same strategy => same results.
  Any randomness must be seeded from config.
- **Costs are mandatory.** There is no zero-cost backtest mode. If you want
  to see gross returns, run with realistic costs and decompose afterward.

The engine emits a structured `BacktestResult` that `quant.evaluation` consumes.
"""
