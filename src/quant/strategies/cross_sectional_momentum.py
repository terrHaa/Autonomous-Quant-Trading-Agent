"""cross_sectional_momentum.py — daily-rebalanced top-K momentum.

What it does
------------
Each bar, ranks the universe by total return over the last `lookback`
days, EXCLUDING the most recent `skip` days. Goes equal-weight long the
top `top_k` names. Everyone else is flat (the engine implicitly exits
positions not in the returned dict).

Why "skip the recent week"
--------------------------
At short horizons (1-5 days) US equities exhibit reversal, not
momentum — winners over the last week tend to underperform the next
week. Jegadeesh & Titman (1993) and a wall of subsequent research:
*sustained* momentum lives at the 3–12 month horizon, but the most
recent few days contaminates that signal with reversal. Excluding the
last 5 days when ranking is the standard fix.

How it composes with the rest of the platform
----------------------------------------------
- The strategy is stateless, pure function of the snapshot. Same data
  on the same bar → same signal. (Required for engine determinism.)
- top_k=10 with 1/top_k weighting gives 10% per name — comfortably
  under the operator's 20% per-trade cap. The risk overlay will not
  bind on these signals.
- Position sizing happens at the engine layer (target_qty = int(weight
  * equity / signal_price)). The strategy just emits weights.

Why not multi-factor / fancy
----------------------------
This is the simplest defensible cross-sectional signal that benchmarks
agree is real. Anything fancier (momentum + quality + low-vol composite,
neutral to sector exposures, etc.) is a rabbit hole worth descending
LATER, once we've measured this baseline's true Sharpe through the DSR
gate. Build the simple thing first; only complicate when you have
evidence the simple thing leaves money on the table.
"""

from __future__ import annotations

from quant.backtest.types import Snapshot


class CrossSectionalMomentum:
    """Long the top-K names by total return over [t-lookback, t-skip].

    Parameters
    ----------
    symbols
        Universe to rank. Typically loaded from
        ``reference/universe/sp500_top100.csv``.
    lookback
        Total return window length, in trading days. Default 60 (≈ 3 mo).
    skip
        How many of the most recent days to EXCLUDE from the lookback,
        to avoid the short-term reversal effect. Default 5 (≈ 1 week).
    top_k
        How many top-ranked names to hold, equal-weighted. Default 10.

    Raises
    ------
    ValueError
        For non-sensical parameters (zero/negative windows, top_k ≥ len(symbols)).
    """

    def __init__(
        self,
        symbols: list[str],
        *,
        lookback: int = 60,
        skip: int = 5,
        top_k: int = 10,
    ) -> None:
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2 (got {lookback})")
        if skip < 0:
            raise ValueError(f"skip must be >= 0 (got {skip})")
        if skip >= lookback:
            raise ValueError(
                f"skip ({skip}) must be < lookback ({lookback}); otherwise "
                f"the signal window has zero or negative length."
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1 (got {top_k})")
        if top_k > len(symbols):
            raise ValueError(
                f"top_k ({top_k}) cannot exceed universe size ({len(symbols)})"
            )

        # Normalize: uppercase, dedupe in input order.
        self._symbols = list(dict.fromkeys(s.upper() for s in symbols))
        self._lookback = lookback
        self._skip = skip
        self._top_k = top_k
        self.name = f"xsec_momo_{lookback}_{skip}_{top_k}"

    def on_bar(self, snapshot: Snapshot) -> dict[str, float]:
        """Return target weights for the top-K momentum names."""
        # Compute signal per symbol that has enough history.
        # We need `lookback` bars of close data; the last `skip` bars get
        # excluded, so the signal window is closes[-lookback:-skip] (or
        # the whole tail if skip=0).
        signals: dict[str, float] = {}
        for sym in self._symbols:
            try:
                sym_bars = snapshot.bars.loc[sym]
            except KeyError:
                continue   # universe contains symbol with no data yet

            closes = sym_bars["close"]
            if len(closes) < self._lookback:
                continue   # not enough history for a fair rank

            # Window: the lookback days, EXCLUDING the most recent `skip`.
            # E.g., lookback=60 skip=5 → use bars [t-60 .. t-5].
            if self._skip == 0:
                window = closes.iloc[-self._lookback:]
            else:
                window = closes.iloc[-self._lookback:-self._skip]

            if len(window) < 2:
                continue
            start = float(window.iloc[0])
            end = float(window.iloc[-1])
            if start <= 0:
                continue
            # Total return over the window. Pure ranking signal; magnitudes
            # don't matter for cross-sectional sorting, only order.
            signals[sym] = (end / start) - 1.0

        if not signals:
            return {}

        # Sort descending by return, take top K.
        ranked = sorted(signals.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[: self._top_k]

        weight_each = 1.0 / len(top)
        return {sym: weight_each for sym, _ in top}
