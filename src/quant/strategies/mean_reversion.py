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
        vol_normalize: bool = True,
        vol_window: int = 20,
        vol_multiplier: float = 1.5,
    ) -> None:
        if lookback < 2:
            # A 1-bar lookback gives MA == current → deviation always 0,
            # signal never fires. 2 is the minimum where the rule has
            # information.
            raise ValueError(f"lookback must be >= 2 (got {lookback})")
        if threshold_pct <= 0:
            raise ValueError(f"threshold_pct must be > 0 (got {threshold_pct})")
        if vol_window < lookback:
            raise ValueError(
                f"vol_window ({vol_window}) must be >= lookback ({lookback}) "
                "so the vol estimate has enough bars."
            )
        if vol_multiplier <= 0:
            raise ValueError(f"vol_multiplier must be > 0 (got {vol_multiplier})")
        self._symbols = [s.upper() for s in symbols]
        self._lookback = lookback
        self._threshold = threshold_pct
        self._allow_short = allow_short
        # Vol-normalization: when True, the entry threshold is
        # `vol_multiplier * rolling_std(returns)` instead of the static
        # `threshold_pct`. A 2% move on KO (daily vol ~1%) is a 2-sigma
        # event — real signal. A 2% move on NVDA (daily vol ~3%) is
        # noise. Vol-normalizing gives a CONSISTENT statistical signal
        # across names instead of over-trading high-vol noise + under-
        # trading low-vol real moves. The legacy static-threshold path
        # is kept for backward compat / backtests; the live agent
        # defaults to vol-normalized.
        self._vol_normalize = vol_normalize
        self._vol_window = vol_window
        self._vol_multiplier = vol_multiplier
        # Name encodes ALL parameters so the registry treats e.g.
        # MeanReversion(lookback=5) and MeanReversion(lookback=10) as
        # distinct variants — important for honest trial counting in DSR.
        short_tag = "_short" if allow_short else ""
        vol_tag = (
            f"_vol{vol_window}x{vol_multiplier:g}" if vol_normalize else ""
        )
        self.name = (
            f"mean_reversion_{lookback}_"
            f"{int(threshold_pct * 10_000)}bp"
            f"{vol_tag}{short_tag}"
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
                # Need the MA window at minimum. If we have lookback bars
                # but not enough for vol estimation, the vol branch below
                # falls back to the static threshold for THIS symbol.
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

            # Pick the effective threshold for this name.
            if self._vol_normalize:
                returns = closes.pct_change().dropna().tail(self._vol_window)
                if len(returns) < 2:
                    eff_threshold = self._threshold
                else:
                    realized_vol = float(returns.std(ddof=1))
                    eff_threshold = self._vol_multiplier * realized_vol
                    # Floor at half the static threshold so we never
                    # over-trade ultra-low-vol names; cap at 2× so we
                    # don't under-trade extreme cases.
                    eff_threshold = max(
                        self._threshold * 0.5,
                        min(self._threshold * 2.0, eff_threshold),
                    )
            else:
                eff_threshold = self._threshold

            if deviation < -eff_threshold:
                long_syms.append((sym, deviation))
            elif self._allow_short and deviation > eff_threshold:
                short_syms.append((sym, deviation))

        if not long_syms and not short_syms:
            return {}  # all flat — engine will exit any held positions

        # Conviction-weighted within each side: a name with deviation
        # -5% (deeper oversold) gets more capital than one at -2%. The
        # signal magnitude IS the strength of the bet. Weight is
        # proportional to |deviation| within each side; total per side
        # sums to the side's portion of the gross book (50/50 when both
        # sides populated, 100% when only one side fires).
        gross_long = 0.5 if (long_syms and short_syms) else 1.0
        gross_short = 0.5 if (long_syms and short_syms) else 1.0

        intents: dict[str, float] = {}
        if long_syms:
            total_long = sum(abs(dev) for _, dev in long_syms)
            if total_long > 0:
                for sym, dev in long_syms:
                    intents[sym] = gross_long * abs(dev) / total_long
            else:
                # Pathological (all deviations exactly 0 somehow) → fallback
                # to equal-weight so we still emit a signal.
                w = gross_long / len(long_syms)
                for sym, _ in long_syms:
                    intents[sym] = w
        if short_syms:
            total_short = sum(abs(dev) for _, dev in short_syms)
            if total_short > 0:
                for sym, dev in short_syms:
                    # Negative = short. Engine handles via signed target_qty.
                    intents[sym] = -gross_short * abs(dev) / total_short
            else:
                w = gross_short / len(short_syms)
                for sym, _ in short_syms:
                    intents[sym] = -w
        return intents
