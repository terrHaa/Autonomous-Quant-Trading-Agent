"""portfolio.py — mutable account state.

The ``Portfolio`` class holds cash + positions and the mark-to-market prices
needed to compute equity and weights. It is the only mutable thing in the
backtest engine — everything else (Snapshot, Order, Fill) is frozen.

The split matters: by isolating mutation to this one class, every other
module can be reasoned about as pure data. When something goes wrong with
account state, you know exactly where to look.
"""

from __future__ import annotations

from quant.backtest.types import Fill


class Portfolio:
    """Cash + signed positions + last-known prices for held names.

    State that evolves through a backtest:
      - ``cash``       — float dollars. Can go negative briefly during a
                          buy round-trip (cash drops at fill, equity stays
                          ~constant because the position offsets it).
      - ``positions``  — dict[symbol, signed_qty]. Negative = short. We
                          drop the entry when qty hits zero so iteration
                          never includes flat names.
      - ``_marks``     — dict[symbol, last_close]. Updated by
                          ``mark_to_market``; used in ``equity()`` and
                          ``weights()`` calculations.
    """

    def __init__(self, starting_equity: float) -> None:
        if starting_equity <= 0:
            raise ValueError(
                f"starting_equity must be positive, got {starting_equity}"
            )
        self.cash: float = float(starting_equity)
        self.positions: dict[str, int] = {}
        self._marks: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Mutation: apply a fill, mark to market.
    # ------------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> None:
        """Update cash and positions for a single fill.

        The math handles all four cases uniformly:
          - long open  (buy from flat)        → -cash, +qty
          - long close (sell to flat)         → +cash, -qty
          - short open (sell from flat)       → +cash, -qty
          - short close (buy to flat)         → -cash, +qty

        Plus commission, which always reduces cash regardless of side.
        """
        # Position change: +qty for buy, -qty for sell. The signed `delta`
        # works the same whether the existing position is long, short, or zero.
        delta = fill.qty if fill.side == "buy" else -fill.qty
        old_qty = self.positions.get(fill.symbol, 0)
        new_qty = old_qty + delta

        if new_qty == 0:
            # Drop the entry so callers iterating over positions never see
            # zero-share holdings (would clutter reports and waste compute).
            self.positions.pop(fill.symbol, None)
            # The mark is no longer relevant to equity, but keep it — the
            # next entry into this symbol will refresh it. Cheap memory.
        else:
            self.positions[fill.symbol] = new_qty

        # Cash change: notional moves the opposite way to the position.
        # Commission always reduces cash (it's a flat cost).
        if fill.side == "buy":
            self.cash -= fill.notional
        else:
            self.cash += fill.notional
        self.cash -= fill.commission

    def mark_to_market(self, marks: dict[str, float]) -> None:
        """Update last-known prices for all held positions.

        We only store marks for symbols we hold — other entries in the
        ``marks`` dict (e.g., the full universe's closes) are ignored.
        That keeps ``_marks`` small and ``equity()`` proportional to the
        position count rather than universe size.
        """
        for sym in self.positions:
            if sym in marks:
                self._marks[sym] = marks[sym]

    # ------------------------------------------------------------------
    # Queries: equity, position value, weights. All derived from state.
    # ------------------------------------------------------------------

    def positions_value(self) -> float:
        """Sum of position values at last marks.

        Symbols held but never marked contribute zero — that's the safe
        default; if we don't know the price, we don't credit it. In
        practice the engine marks every held name on every bar.
        """
        return sum(
            qty * self._marks.get(sym, 0.0)
            for sym, qty in self.positions.items()
        )

    def equity(self) -> float:
        """Cash + sum of position values at last marks."""
        return self.cash + self.positions_value()

    def weights(self) -> dict[str, float]:
        """Per-symbol portfolio weight at the latest marks.

        weight[sym] = (qty[sym] * mark[sym]) / equity

        Returns an empty dict if equity is non-positive — weights are
        meaningless when you're at or below zero (drawdown kill territory).
        """
        e = self.equity()
        if e <= 0:
            return {}
        return {
            sym: (qty * self._marks.get(sym, 0.0)) / e
            for sym, qty in self.positions.items()
        }
