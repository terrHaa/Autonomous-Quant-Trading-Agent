"""mean_reversion.py — short-term mean reversion strategy.

What it does
------------
For each tracked symbol: compare today's close to the N-day moving average.
If price is more than ``threshold_pct`` BELOW the MA, expect a bounce → go
long. If ``allow_short=True`` and price is more than ``threshold_pct`` ABOVE
the MA, expect a pullback → go short.

This is the academic single-name version of short-term reversal (Lehmann
1990, Lo & MacKinlay 1990). The well-documented cross-sectional version
(long the bottom-decile by recent return, short the top decile) often
beats it on Sharpe, but the single-name threshold version composes
naturally with our per-symbol Strategy interface and is enough to
demonstrate diversification against momentum.

Why we build this second
------------------------
SMA crossover (trend-following) and mean reversion are textbook opposites
— momentum buys strength, reversion buys weakness. Their returns aren't
perfectly anti-correlated (timing and costs blur the picture), but the
correlation is low enough that combining them helps. With only one
strategy, the HRP allocator (Step 17) has nothing to allocate — the
whole point of HRP is to weight uncorrelated streams.

The rule
--------
Per bar, for each tracked symbol:
  - Pull the close series from the snapshot.
  - Skip if there are fewer than ``lookback`` bars of history.
  - Compute ma = mean of the last ``lookback`` closes.
  - Compute deviation = (current_close - ma) / ma.
  - If deviation < -threshold_pct → mark as "long this bar".
  - Elif ``allow_short`` and deviation > +threshold_pct → mark as "short".
  - Otherwise → no position (flat).

Then split exposure equal-weight across the long set, equal-weight across
the short set if shorting. When both sides are populated, each side gets
50% of gross exposure (so total gross stays at 1.0 — no leverage).
"""

from __future__ import annotations

from quant.backtest.types import Snapshot


class MeanReversion:
    """Short-term mean reversion, long-or-flat (optionally short).

    Parameters
    ----------
    symbols
        Tickers to track. Case-normalized to upper.
    lookback
        Window for the moving average. Default 5 days — the classic
        short-term reversal horizon.
    threshold_pct
        How far below (or above) the MA price must be before a signal
        fires. Default 0.02 = 2%, which catches roughly 1-sigma 5-day
        moves on liquid US equities.
    allow_short
        If True, fires SHORT signals when price is above the MA by
        threshold_pct. If False (default), the strategy is long-or-flat —
        symmetric to the SMA crossover variant we ship.

    Raises
    ------
    ValueError
        For non-sensical parameters (lookback < 2, threshold_pct <= 0).
    """

    def __init__(
        self,
        symbols: list[str],
        *,
        lookback: int = 5,
        threshold_pct: float = 0.02,
        allow_short: bool = False,
    ) -> None:
        if lookback < 2:
            # A 1-bar lookback gives MA == current → deviation always 0,
            # signal never fires. 2 is the minimum where the rule has
            # information.
            raise ValueError(f"lookback must be >= 2 (got {lookback})")
        if threshold_pct <= 0:
            raise ValueError(f"threshold_pct must be > 0 (got {threshold_pct})")
        self._symbols = [s.upper() for s in symbols]
        self._lookback = lookback
        self._threshold = threshold_pct
        self._allow_short = allow_short
        # Name encodes ALL parameters so the registry treats e.g.
        # MeanReversion(lookback=5) and MeanReversion(lookback=10) as
        # distinct variants — important for honest trial counting in DSR.
        short_tag = "_short" if allow_short else ""
        self.name = (
            f"mean_reversion_{lookback}_"
            f"{int(threshold_pct * 10_000)}bp"
            f"{short_tag}"
        )

    def on_bar(self, snapshot: Snapshot) -> dict[str, float]:
        """Return target weights per the mean-reversion rule.

        Pure function — no state between bars. Re-running on the same
        snapshot returns the same dict.
        """
        long_syms: list[str] = []
        short_syms: list[str] = []

        for sym in self._symbols:
            # As in SMA crossover: missing symbols (recent IPOs, etc.)
            # are skipped silently rather than crashing.
            try:
                sym_bars = snapshot.bars.loc[sym]
            except KeyError:
                continue

            closes = sym_bars["close"]
            if len(closes) < self._lookback:
                continue

            window = closes.iloc[-self._lookback:]
            ma = float(window.mean())
            if ma <= 0:
                # Defensive: would yield divide-by-zero / nonsense
                # deviation. Equities shouldn't trade at 0, but if our
                # data ever lies, we'd rather skip than NaN-poison.
                continue
            current = float(closes.iloc[-1])
            deviation = (current - ma) / ma

            if deviation < -self._threshold:
                long_syms.append(sym)
            elif self._allow_short and deviation > self._threshold:
                short_syms.append(sym)

        if not long_syms and not short_syms:
            return {}  # all flat — engine will exit any held positions

        # Equal-weight within each side. When both sides populated, split
        # gross 50/50 so the net stays under 1.0 leverage (matches v1
        # engine assumption that target weights sum to <= 1.0 in absolute).
        if long_syms and short_syms:
            long_weight = 0.5 / len(long_syms)
            short_weight = 0.5 / len(short_syms)
        elif long_syms:
            long_weight = 1.0 / len(long_syms)
            short_weight = 0.0
        else:
            long_weight = 0.0
            short_weight = 1.0 / len(short_syms)

        intents: dict[str, float] = {}
        for sym in long_syms:
            intents[sym] = long_weight
        for sym in short_syms:
            # Negative weight = short. The engine handles this via its
            # signed target_qty = int(target_dollars / signal_price) path.
            intents[sym] = -short_weight
        return intents
