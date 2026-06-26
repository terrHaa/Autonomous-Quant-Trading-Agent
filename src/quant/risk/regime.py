"""regime.py — market-regime classification and regime-aware allocation.

Phase 3 of the Sharpe-target build. The signal-health tracker showed
that strategy edges are regime-dependent (mean-reversion's IC is +0.28
in down-trends but ~0 in up-trends). A static HRP allocation ignores
this and keeps paying for a sleeve that's dead in the current regime.

This module provides the mechanism to fix that, in two pieces:

1. ``classify_regime`` — label today's market by trend (price vs its
   200-day average) and volatility (realized vs its own median). Four
   labels: ``trend_up_calm``, ``trend_up_stormy``, ``trend_dn_calm``,
   ``trend_dn_stormy``. Coarse on purpose — finer regimes overfit.

2. ``apply_regime_policy`` — scale each strategy's HRP weight by a
   per-strategy, per-regime multiplier (the "regime policy"), then
   renormalize. The policy is RESEARCH OUTPUT: the monthly review
   computes regime-conditional IC and writes the multipliers; the daily
   path just classifies the regime and applies them. That keeps the
   daily run cheap and the heavy analysis monthly.

Plus a diversification guard:

3. ``average_pairwise_correlation`` + ``correlation_degross_factor`` —
   when the book's names start moving together (correlations spike, as
   they do in selloffs), diversification is failing and the realized
   risk is higher than the per-name vols imply. The factor ramps gross
   exposure DOWN as correlation rises.

DESIGN NOTE: nothing here is wired into the live trading path yet —
these are pure functions. Wiring them in changes live risk behavior and
must go through a backtest + operator sign-off (the project's rule for
the sizing/risk layer). ``tools/preview_regime_policy.py`` shows what
they would do before anything goes live.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_LABELS: tuple[str, ...] = (
    "trend_up_calm",
    "trend_up_stormy",
    "trend_dn_calm",
    "trend_dn_stormy",
)

_TREND_LOOKBACK = 200       # SMA window for the trend axis
_VOL_LOOKBACK = 20          # realized-vol window for the storm axis
_VOL_MEDIAN_LOOKBACK = 200  # window the current vol is compared against


def classify_regime(
    market_close: pd.Series,
    *,
    trend_lookback: int = _TREND_LOOKBACK,
    vol_lookback: int = _VOL_LOOKBACK,
    vol_median_lookback: int = _VOL_MEDIAN_LOOKBACK,
) -> str:
    """Classify the latest bar of ``market_close`` into a regime label.

    ``market_close`` is a date-indexed close series for the market proxy
    (SPY, or an equal-weighted universe index). Returns one of
    ``REGIME_LABELS``. Falls back to ``trend_up_calm`` (the most benign,
    "do nothing special" label) when there's insufficient history — a
    regime classifier must never block trading on missing data.
    """
    s = market_close.dropna()
    if len(s) < max(trend_lookback, vol_median_lookback) // 2:
        logger.info("classify_regime: short history (%d bars); default benign", len(s))
        return "trend_up_calm"

    sma = s.rolling(trend_lookback, min_periods=trend_lookback // 2).mean()
    trend = "trend_up" if s.iloc[-1] >= sma.iloc[-1] else "trend_dn"

    rets = s.pct_change()
    cur_vol = rets.tail(vol_lookback).std()
    med_vol = rets.rolling(vol_lookback).std().tail(vol_median_lookback).median()
    storm = "stormy" if (pd.notna(cur_vol) and pd.notna(med_vol)
                         and cur_vol > med_vol) else "calm"
    return f"{trend}_{storm}"


def apply_regime_policy(
    hrp_weights: dict[str, float],
    regime: str,
    policy: dict[str, dict[str, float]] | None,
    *,
    floor: float = 0.0,
) -> dict[str, float]:
    """Scale strategy HRP weights by their regime multiplier, renormalize.

    ``policy`` is ``{strategy_name: {regime_label: multiplier}}``. A
    strategy/regime not present in the policy gets multiplier 1.0 (no
    change) — so a partial or empty policy degrades gracefully to the
    unmodified HRP weights. After scaling, weights are renormalized to
    preserve the original gross (this only RE-DISTRIBUTES across sleeves;
    total deployment stays the vol-target's job).

    ``floor`` clamps each multiplier from below (e.g. 0.0 lets a sleeve
    be switched fully off; 0.25 keeps a residual allocation).
    """
    if not policy or not hrp_weights:
        return dict(hrp_weights)

    scaled: dict[str, float] = {}
    for name, w in hrp_weights.items():
        mult = policy.get(name, {}).get(regime, 1.0)
        scaled[name] = w * max(mult, floor)

    gross_in = sum(hrp_weights.values())
    gross_out = sum(scaled.values())
    if gross_out <= 0:
        # Policy zeroed everything — refuse to return an empty book;
        # fall back to the original weights.
        logger.warning(
            "apply_regime_policy: regime '%s' zeroed all sleeves; "
            "falling back to unmodified HRP weights.", regime,
        )
        return dict(hrp_weights)
    norm = gross_in / gross_out
    return {name: w * norm for name, w in scaled.items()}


def average_pairwise_correlation(
    returns: pd.DataFrame,
    *,
    lookback: int = 60,
    min_names: int = 5,
) -> float:
    """Mean off-diagonal pairwise correlation of recent name returns.

    ``returns`` is a date × symbol returns frame. Returns NaN if there
    aren't enough names/observations to estimate it.
    """
    recent = returns.tail(lookback).dropna(axis=1, how="any")
    if recent.shape[1] < min_names or len(recent) < min_names:
        return float("nan")
    corr = recent.corr().to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    vals = corr[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def correlation_degross_factor(
    avg_corr: float,
    *,
    normal: float = 0.30,
    high: float = 0.60,
    min_factor: float = 0.50,
) -> float:
    """Gross-exposure multiplier that falls as correlation rises.

    At or below ``normal`` correlation → 1.0 (full exposure). At or above
    ``high`` → ``min_factor``. Linear in between. NaN input → 1.0 (don't
    de-gross on a measurement we couldn't make).

    Rationale: when names co-move, the book's realized vol exceeds what
    its per-name vols predict, so the vol-target overlay UNDER-estimates
    risk. This trims gross to compensate during correlation spikes
    (typically selloffs), an extra layer beyond the drawdown ladder.
    """
    if not np.isfinite(avg_corr):
        return 1.0
    if avg_corr <= normal:
        return 1.0
    if avg_corr >= high:
        return min_factor
    frac = (avg_corr - normal) / (high - normal)
    return float(1.0 - frac * (1.0 - min_factor))
