"""deployment.py — when to deploy how much: regime filter + drawdown ladder.

Deployment policy, decided with the operator 2026-06-12 (Sharpe target
>1.5, kill line -15%). Three layers, applied multiplicatively AFTER
vol-targeting in the daily pipeline:

1. **Vol targeting** (in ``quant.allocator``) answers "how much, given
   current market risk" — the engine, not in this module.
2. **Regime filter** (here): full gross when SPY closes at/above its
   200-day SMA, half gross below. Long-only momentum earns its Sharpe
   in uptrends and gives it back in regime breaks; an index trend
   filter has cut drawdowns roughly in half across a century of data
   while costing little return. Hysteresis is deliberate: the scale
   flips only on the SMA cross, and whipsaw cost near the line is
   bounded by the 0.5 floor (we never go to zero on the filter alone).
3. **Drawdown ladder** (here): de-risk INTO a drawdown instead of
   running full size until the -15% kill cliff executes the book.
   Cutting gross as the drawdown deepens compresses the left tail and
   makes reaching the kill line much less likely — which is what makes
   the 12% vol target coherent with a 15% kill line.

Both functions are pure and fail-open by design: deployment scaling is
an optimization, not a safety requirement (the hard rules in
daily_runner are the safety layer). If SPY data is missing, the filter
returns 1.0 with a diagnostic — a data outage shouldn't halve the book.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Regime filter parameters. 200-day SMA is the canonical trend gauge —
# deliberately NOT tunable per-strategy or by the AI analyst: a regime
# filter you re-tune is a regime filter you've overfit.
REGIME_SMA_WINDOW = 200
REGIME_BELOW_SCALE = 0.50

# Drawdown ladder: (drawdown threshold, gross multiplier), checked
# deepest-first. Drawdowns are negative numbers. The -15% kill switch
# (MAX_DRAWDOWN_KILL in daily_runner) remains the operator hard rule
# and is NOT part of this ladder — the ladder's job is to make sure
# we arrive at -15% with a quarter of the book, not all of it.
DRAWDOWN_LADDER: tuple[tuple[float, float], ...] = (
    (-0.125, 0.25),
    (-0.10, 0.50),
    (-0.05, 0.75),
)


def regime_scale(
    spy_closes: pd.Series | None,
    *,
    window: int = REGIME_SMA_WINDOW,
    below_scale: float = REGIME_BELOW_SCALE,
) -> tuple[float, dict[str, Any]]:
    """Gross multiplier from the SPY trend regime.

    Returns ``(scale, diagnostic)``. Scale is 1.0 when the latest SPY
    close is at/above its ``window``-day SMA, ``below_scale`` when
    below, and 1.0 (fail-open, with reason in the diagnostic) when the
    series is missing or shorter than ``window``.
    """
    if spy_closes is None or len(spy_closes) == 0:
        return 1.0, {"regime": "unknown", "reason": "no SPY data"}
    closes = spy_closes.dropna()
    if len(closes) < window:
        return 1.0, {
            "regime": "unknown",
            "reason": f"only {len(closes)} SPY closes < {window} needed",
        }
    sma = float(closes.tail(window).mean())
    last = float(closes.iloc[-1])
    diag: dict[str, Any] = {
        "spy_close": round(last, 2),
        "spy_sma200": round(sma, 2),
        "regime": "risk_on" if last >= sma else "risk_off",
    }
    return (1.0, diag) if last >= sma else (below_scale, diag)


def drawdown_scale(
    drawdown: float,
    *,
    ladder: tuple[tuple[float, float], ...] = DRAWDOWN_LADDER,
) -> float:
    """Gross multiplier from the current peak-to-trough drawdown.

    ``drawdown`` is negative (e.g. -0.07 = 7% below peak); 0.0 or
    positive means at/above the peak → full size. Rungs are checked
    deepest-first so the worst applicable multiplier wins.
    """
    for threshold, scale in sorted(ladder):
        if drawdown <= threshold:
            return scale
    return 1.0
