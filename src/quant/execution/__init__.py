"""execution — broker interface.

Wraps Alpaca's API (via the `alpaca-py` SDK) behind a thin internal interface,
so the rest of the codebase doesn't depend on Alpaca-specific types. If we
ever switch brokers, only this module changes.

Key responsibilities:
- Map our internal "target weights" -> concrete orders (size, side, type).
- Submit orders, track fills, reconcile against the broker's book on startup.
- Honor the `ALPACA_ENV` switch (paper vs live). Live mode requires an
  explicit, deliberate config flag — there is no accidental path to it.
- Surface errors clearly. A failed order is *not* the same as a filled-zero
  order; collapsing the two is how live PnL goes mysteriously sideways.
"""
