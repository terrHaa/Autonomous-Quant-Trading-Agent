"""Tests for the per-strategy signal-health tracker (Phase 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.evaluation.signal_health import compute_signal_health


def _trending_bars(n_days=400, n_syms=40, seed=0):
    """Bars where each name has a persistent drift (momentum predicts fwd)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=n_days)
    drifts = np.linspace(-0.001, 0.001, n_syms)
    rows, idx = [], []
    for j in range(n_syms):
        p = 100.0
        for ts in days:
            p *= 1 + drifts[j] + rng.normal(0, 0.005)
            rows.append({"open": p, "high": p, "low": p, "close": p, "volume": 1})
            idx.append((f"S{j:02d}", ts))
    return pd.DataFrame(rows,
                        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
                        columns=["open", "high", "low", "close", "volume"])


class _MomentumStrat:
    """Weights names by trailing 20-day return — should have positive IC
    on trending bars."""
    name = "fake_momentum"

    def on_bar(self, snapshot):
        bars = snapshot.bars
        close = bars["close"].unstack(level=0)
        if len(close) < 25:
            return {}
        trail = close.iloc[-1] / close.iloc[-21] - 1.0
        return {s: float(v) for s, v in trail.items() if np.isfinite(v) and v > 0}


class _RandomStrat:
    name = "fake_random"

    def __init__(self, seed=1):
        self._rng = np.random.default_rng(seed)

    def on_bar(self, snapshot):
        return {s: float(self._rng.random()) for s in snapshot.symbols()}


def test_momentum_strategy_has_positive_ic_on_trending_bars() -> None:
    h = compute_signal_health(_MomentumStrat(), _trending_bars(), lookback_days=600)
    assert h.n_periods > 10
    assert h.mean_ic > 0.05        # real, persistent edge
    assert h.hit_rate > 0.6


def test_random_strategy_has_near_zero_ic() -> None:
    h = compute_signal_health(_RandomStrat(), _trending_bars(seed=2), lookback_days=600)
    assert h.n_periods > 10
    assert abs(h.mean_ic) < 0.05   # no edge


def test_regime_split_is_reported() -> None:
    bars = _trending_bars()
    close = bars["close"].unstack(level=0).mean(axis=1)
    dates = [t.date() for t in close.index]
    # Alternate a synthetic regime label by index parity.
    regime = {d: ("A" if i % 2 == 0 else "B") for i, d in enumerate(dates)}
    h = compute_signal_health(_MomentumStrat(), bars, regime=regime, lookback_days=600)
    assert set(h.regime_ic).issubset({"A", "B"})
    assert len(h.regime_ic) >= 1


def test_insufficient_history_warns_not_raises() -> None:
    tiny = _trending_bars(n_days=8, n_syms=6)
    h = compute_signal_health(_MomentumStrat(), tiny, lookback_days=600)
    assert h.n_periods == 0
    assert h.warnings
