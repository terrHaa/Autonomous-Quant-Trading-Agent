"""strategies — individual strategy implementations.

Each strategy is a self-contained module that exposes a class implementing the
``Strategy`` protocol (defined in ``quant.backtest.types``). On each bar, the
engine hands the strategy a ``Snapshot``; the strategy returns desired target
weights; the engine handles the order generation.

Strategies must:
- Be pure functions of the snapshot they receive — no hidden state from disk,
  no network calls, no clock reads.
- Be cheap to instantiate; expensive precomputation belongs in a fit/setup
  step that runs once per walk-forward fold (deferred for now).

Currently available:
- ``SmaCrossover`` — 50/200-day moving-average crossover (classic; usually
  doesn't make money, but exercises the engine on realistic mechanics).
"""

from quant.strategies.sma_crossover import SmaCrossover

__all__ = ["SmaCrossover"]
