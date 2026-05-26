"""strategies — individual strategy implementations.

Each strategy is a self-contained module that exposes a class implementing the
strategy protocol (defined in `quant.backtest`). The protocol roughly looks
like: on each bar, the engine hands the strategy the latest data; the strategy
returns desired target weights; the engine handles the order generation.

Strategies must:
- Be pure functions of the data they receive — no hidden state from disk, no
  network calls, no clock reads.
- Declare their universe requirements (e.g. "needs daily OHLCV for past 252
  days") so the engine can validate inputs.
- Be cheap to instantiate; expensive precomputation belongs in a fit/setup
  step that runs once per walk-forward fold.
"""
