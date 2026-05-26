"""backtest — event-driven backtest engine.

Design principles (see ``docs/specs/backtest-engine.md`` for the full spec):

- **No leakage.** The engine advances time bar-by-bar and exposes to the
  strategy only data that would have been known at that bar's close. A
  strategy that tries to peek at tomorrow gets an exception, not a number.
- **Realistic fills.** Orders submitted at close T fill at the bar of T+1.
  Market orders fill at next open with bp-based slippage; limit and stop
  orders fill only when the next bar's range crosses the trigger.
- **Deterministic.** Same config + same data + same strategy => same results.
  Any randomness must be seeded from config.
- **Costs are mandatory.** There is no zero-cost backtest mode. If you want
  to see gross returns, run with realistic costs and decompose afterward.

The engine emits a structured ``BacktestResult`` that ``quant.evaluation``
consumes.

Public surface (strategy authors mostly need these two):
"""

from quant.backtest.engine import BacktestResult, run_backtest
from quant.backtest.portfolio import Portfolio
from quant.backtest.types import (
    Fill,
    Order,
    OrderIntent,
    OrderType,
    Side,
    Snapshot,
    Strategy,
    TimeInForce,
)

__all__ = [
    "BacktestResult",
    "Fill",
    "Order",
    "OrderIntent",
    "OrderType",
    "Portfolio",
    "Side",
    "Snapshot",
    "Strategy",
    "TimeInForce",
    "run_backtest",
]
