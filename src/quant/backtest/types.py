"""types.py — data classes for the backtest engine.

This module holds the *immutable* pieces of the engine:
  - ``Snapshot``    — a no-leak view of historical bars at a moment in time
  - ``OrderIntent`` — what a strategy returns when it wants more than a bare weight
  - ``Order``       — the engine's internal order record
  - ``Fill``        — the record of an executed trade

The mutable state (cash + positions over time) lives in ``portfolio.py``.
Keeping the immutable types here means tests can build them freely without
worrying about cross-contamination between fixtures.

Every class is frozen so an accidental mutation (e.g. ``fill.qty = -fill.qty``)
fails loudly instead of silently corrupting the audit trail downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol

import pandas as pd


# ---------------------------------------------------------------------------
# Type aliases. Centralizing these makes it easy to change the order-side
# representation (e.g., add 'short_open' / 'short_close' for borrow tracking)
# in one place when we need it.
# ---------------------------------------------------------------------------

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop"]
TimeInForce = Literal["DAY", "GTC"]


# ---------------------------------------------------------------------------
# Snapshot — the no-leak primitive.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """A point-in-time view of historical bars.

    The platform's most important invariant: a strategy receiving a snapshot
    for ``as_of`` cannot see any data with a timestamp later than ``as_of``.
    We enforce this two ways:

    1. ``from_full_bars`` pre-slices the input frame, so the future is not
       present in the returned object.
    2. ``__post_init__`` validates that the ``bars`` field doesn't contain
       any post-``as_of`` rows — catches misuse if someone constructs a
       ``Snapshot`` directly rather than via the factory.

    The check at (2) costs a single pandas comparison per bar; cheap
    insurance against the entire platform's most catastrophic bug class.
    """

    as_of: date
    # Already-sliced bars (MultiIndex(symbol, timestamp), OHLCV columns).
    # repr=False because pandas frames are huge and clutter test output.
    bars: pd.DataFrame = field(repr=False)

    def __post_init__(self) -> None:
        # No-leak guard. An empty frame is fine — common when the first bar
        # in the test window hasn't arrived yet.
        if not self.bars.empty:
            ts = self.bars.index.get_level_values("timestamp")
            if (ts.date > self.as_of).any():
                raise ValueError(
                    f"Snapshot.bars contains rows after as_of={self.as_of}; "
                    f"max timestamp seen = {ts.max()}. This would leak future "
                    f"data to the strategy."
                )

    @classmethod
    def from_full_bars(cls, full_bars: pd.DataFrame, as_of: date) -> Snapshot:
        """Build a Snapshot by slicing a full bars frame at ``as_of``.

        This is the engine's canonical way to construct snapshots — using
        it guarantees the no-leak invariant. Direct ``Snapshot(...)``
        construction works too (with the post-init validation), but the
        factory makes the slicing intent obvious to readers.
        """
        if full_bars.empty:
            return cls(as_of=as_of, bars=full_bars)
        ts = full_bars.index.get_level_values("timestamp")
        sliced = full_bars[ts.date <= as_of]
        return cls(as_of=as_of, bars=sliced)

    def close(self, symbol: str) -> float:
        """Most recent close price for ``symbol`` on or before ``as_of``.

        Raises KeyError if the symbol has no bars yet. Strategies that
        trade illiquid names should guard against this.
        """
        try:
            sym_bars = self.bars.loc[symbol]
        except KeyError as e:
            raise KeyError(
                f"No bars for {symbol!r} on or before {self.as_of}"
            ) from e
        if sym_bars.empty:
            raise KeyError(f"No bars for {symbol!r} on or before {self.as_of}")
        # `.iloc[-1]` because the index is timestamp-sorted within a symbol
        # (the cache guarantees this). The most recent close is the last row.
        return float(sym_bars["close"].iloc[-1])

    def symbols(self) -> list[str]:
        """All symbols that have at least one bar on or before ``as_of``.

        Sorted for deterministic iteration order — matters for reproducibility.
        """
        if self.bars.empty:
            return []
        return sorted(self.bars.index.get_level_values("symbol").unique().tolist())


# ---------------------------------------------------------------------------
# OrderIntent — the rich-form output from a strategy's on_bar().
# Strategies that just want market-on-open behavior return bare floats;
# OrderIntent is the escape hatch for limit/stop semantics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderIntent:
    """What a strategy returns when a bare float weight is too coarse.

    Strategies typically return ``dict[str, float]`` where each float is the
    target portfolio weight (market-on-open at the next bar). When a strategy
    needs control over the order type — e.g., a mean-reversion strategy that
    only wants to buy on a pullback to a limit price — it returns this object
    instead of a float for the relevant symbol(s).
    """

    target_weight: float
    order_type: OrderType = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = "DAY"

    def __post_init__(self) -> None:
        # Validate that the price fields required by the order type are present.
        # Catching this here means the engine doesn't have to special-case
        # 'malformed intent' deep inside the order-generation logic.
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit order requires limit_price")
        if self.order_type == "stop" and self.stop_price is None:
            raise ValueError("stop order requires stop_price")


# ---------------------------------------------------------------------------
# Order — the engine's internal record of a queued order.
# Generated from OrderIntents (or bare weights) by the engine; consumed by
# the fill logic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Order:
    """A concrete order generated by the engine, waiting to be filled."""

    submitted_date: date          # close of the bar where the decision was made
    symbol: str
    side: Side
    qty: int                      # always positive; ``side`` carries direction
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = "DAY"

    def __post_init__(self) -> None:
        if self.qty <= 0:
            # qty must be positive — direction is in `side`. A negative qty
            # would be ambiguous (buy -5 = sell 5? cover 5?).
            raise ValueError(f"Order.qty must be positive, got {self.qty}")


# ---------------------------------------------------------------------------
# Fill — record of an executed trade.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fill:
    """Record of an executed trade.

    All cost components are separated for accurate attribution downstream:
    the reports module can decompose strategy PnL into "trading frictions"
    by summing each column.
    """

    date: date
    symbol: str
    side: Side
    qty: int
    fill_price: float

    # Derived but recorded explicitly so downstream consumers don't have to
    # recompute (and accidentally use a different definition than the engine).
    notional: float               # |qty * fill_price|

    # Cost breakdown (all in dollars). The total cost above the modeled open
    # is spread_cost + slippage_cost + commission.
    spread_cost: float            # half-spread component
    slippage_cost: float          # impact / adverse-selection component
    commission: float             # broker commission


# ---------------------------------------------------------------------------
# Strategy Protocol — the contract any strategy must satisfy to be runnable
# by the engine. Defined here because it's strategy-facing (alongside
# Snapshot and OrderIntent); the engine itself just imports it.
# ---------------------------------------------------------------------------


class Strategy(Protocol):
    """What the engine expects from any strategy.

    Implement as a class (so ``name`` can be a class attribute and on_bar
    can carry state) or a plain object with the matching shape.

    Example::

        class BuyAndHoldSPY:
            name = "buy_and_hold_spy"

            def on_bar(self, snapshot):
                return {"SPY": 1.0}  # 100% in SPY at all times
    """

    name: str

    def on_bar(self, snapshot: Snapshot) -> dict[str, float | OrderIntent]:
        """Return desired per-symbol intent for the next bar's orders.

        - Float values become market-on-open orders at the next bar.
        - OrderIntent values can specify limit/stop semantics.
        - Symbols held but not in the returned dict are flatted at market.

        Implementation must be pure: no side effects, no network/file IO,
        no time-of-day reads. Otherwise the engine can't promise determinism.
        """
