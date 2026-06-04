"""sma_crossover.py — classic 50/200-day moving-average crossover strategy.

What it does
------------
For each symbol it tracks: go fully long when the fast SMA crosses above the
slow SMA ("golden cross"); go flat when the fast crosses below ("death
cross"). When multiple symbols are simultaneously long, they share equal
weight (so 3 longs = 33% each).

Why we build this first
-----------------------
SMA crossover **usually does not make money** after costs on liquid US
equities — it's been one of the most-tested and most-overfit ideas in
public domain quant. So why is it our first strategy? Three reasons:

1. **Hand-verifiable.** When the equity curve does something weird, we can
   inspect the SMA values on that bar with a calculator and confirm whether
   the engine did what the strategy asked. Simple signals = easy debugging
   of the *engine*, which is what we actually care about right now.
2. **Realistic mechanics.** Unlike buy-and-hold, this strategy actually
   trades — entries, exits, re-entries — exercising the order generation,
   fill, and cost paths on real data. Confidence in mechanics first;
   chasing alpha later.
3. **A benchmark for absurdity.** Future strategies should beat or at
   least match SMA crossover. Anything that loses to a hand-crafted
   classroom example needs a hard look.

The strategy itself
-------------------
Per bar, for each tracked symbol:
  - Pull the symbol's close series from the snapshot.
  - Skip if there's less than `slow` bars of history.
  - Compute fast_sma = mean of last `fast` closes.
  - Compute slow_sma = mean of last `slow` closes.
  - If fast > slow → mark symbol as "long this bar".

Then split 100% equity equally across the long set. (Empty long set =
return {} which the engine interprets as 'flat everything'.)
"""

from __future__ import annotations

from quant.backtest.types import Snapshot


class SmaCrossover:
    """50/200-day SMA crossover, long-or-flat, equal-weight across longs.

    Parameters
    ----------
    symbols
        List of tickers to track. Case-normalized to upper.
    fast
        Window for the fast SMA. Default 50 days — the standard.
    slow
        Window for the slow SMA. Default 200 days — also standard.

    Raises
    ------
    ValueError
        If fast >= slow (a crossover needs fast and slow to actually differ).
    """

    def __init__(
        self,
        symbols: list[str],
        *,
        fast: int = 50,
        slow: int = 200,
    ) -> None:
        if fast <= 0 or slow <= 0:
            raise ValueError(f"fast and slow must be positive (got {fast}, {slow})")
        if fast >= slow:
            raise ValueError(
                f"fast ({fast}) must be strictly less than slow ({slow}); "
                f"otherwise there's no meaningful crossover."
            )
        self._symbols = [s.upper() for s in symbols]
        self._fast = fast
        self._slow = slow
        # name includes the parameters so the registry can distinguish
        # SmaCrossover(fast=20, slow=100) from SmaCrossover(fast=50, slow=200)
        # as distinct variants in the multi-testing correction.
        self.name = f"sma_crossover_{fast}_{slow}"

    def on_bar(self, snapshot: Snapshot) -> dict[str, float]:
        """Return target weights per the SMA rule.

        Side effects: none — pure function of the snapshot. The strategy
        holds no state between bars, so re-running the backtest on the
        same data produces the same orders.
        """
        # Decide which symbols pass the crossover filter, AND compute
        # each one's CONVICTION = (fast - slow) / slow. A name where
        # the fast SMA is 5% above slow has stronger trend than one
        # where it's 0.5% above; we give the strong-trend name more
        # capital instead of equal-weighting both. This is the
        # "weight ∝ signal strength" principle that turns a binary
        # filter into a graded one.
        long_strengths: list[tuple[str, float]] = []

        for sym in self._symbols:
            # snapshot.bars is MultiIndex(symbol, timestamp). Slicing by
            # the symbol level returns a single-symbol frame indexed by
            # timestamp. If the symbol has no rows yet (recent IPO, future
            # universe addition), .loc raises KeyError — skip cleanly.
            try:
                sym_bars = snapshot.bars.loc[sym]
            except KeyError:
                continue

            closes = sym_bars["close"]
            if len(closes) < self._slow:
                # Not enough history to compute the slow SMA → stay out.
                continue

            # `.iloc[-N:]` takes the last N rows. Cheap on a Series.
            fast_sma = float(closes.iloc[-self._fast:].mean())
            slow_sma = float(closes.iloc[-self._slow:].mean())

            if fast_sma > slow_sma and slow_sma > 0:
                strength = (fast_sma - slow_sma) / slow_sma
                long_strengths.append((sym, strength))

        if not long_strengths:
            # Empty dict → engine flats every held position. Cash on the side.
            return {}

        # Conviction-weighted: weight ∝ strength / sum(strengths).
        total = sum(s for _, s in long_strengths)
        if total <= 0:
            # Pathological fallback (all strengths zero somehow) → equal.
            w = 1.0 / len(long_strengths)
            return {sym: w for sym, _ in long_strengths}
        return {sym: s / total for sym, s in long_strengths}
