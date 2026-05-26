"""Tests for vol targeting and Kelly fraction.

Hand-computable cases: construct returns with known mean/std, scale to a
known target, assert the result has the predicted vol/leverage. Same
pattern as test_metrics.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant.allocator import apply_vol_target, kelly_leverage, vol_target_scale


def _gaussian_returns(
    n: int,
    *,
    mean: float = 0.0,
    std: float = 0.01,
    seed: int = 42,
) -> pd.Series:
    """Deterministic Gaussian returns for tests."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


# ---------------------------------------------------------------------------
# vol_target_scale
# ---------------------------------------------------------------------------


def test_vol_target_scale_brings_realized_vol_to_target() -> None:
    """After multiplying returns by scale, realized vol should hit target.

    Sanity check: std(c * X) == c * std(X). So scale = target / realized
    makes the scaled series have the requested vol exactly.
    """
    returns = _gaussian_returns(2520, std=0.01)  # ~16% annualized
    target_vol = 0.10  # 10% annualized

    scale = vol_target_scale(returns, target_vol_annual=target_vol)
    scaled = returns * scale
    realized_vol_after = float(scaled.std(ddof=1)) * math.sqrt(252)

    assert realized_vol_after == pytest.approx(target_vol, rel=1e-10)


def test_vol_target_scale_lower_when_underlying_is_volatile() -> None:
    """A noisier strategy needs MORE de-leveraging to hit a fixed target."""
    quiet = _gaussian_returns(2520, std=0.005)   # ~8% vol
    noisy = _gaussian_returns(2520, std=0.02)    # ~32% vol
    target = 0.10

    scale_quiet = vol_target_scale(quiet, target_vol_annual=target)
    scale_noisy = vol_target_scale(noisy, target_vol_annual=target)

    # Quiet strategy needs LEVER UP (scale > 1); noisy needs DELEVERAGE (< 1).
    assert scale_quiet > 1.0 > scale_noisy


def test_vol_target_scale_returns_zero_on_no_signal() -> None:
    """Constant returns → vol = 0 → can't scale → stay flat."""
    returns = pd.Series([0.001] * 252)
    scale = vol_target_scale(returns, target_vol_annual=0.10)
    assert scale == 0.0


def test_vol_target_scale_rejects_non_positive_target() -> None:
    returns = _gaussian_returns(252)
    with pytest.raises(ValueError, match="positive"):
        vol_target_scale(returns, target_vol_annual=0)
    with pytest.raises(ValueError):
        vol_target_scale(returns, target_vol_annual=-0.05)


# ---------------------------------------------------------------------------
# apply_vol_target
# ---------------------------------------------------------------------------


def test_apply_vol_target_preserves_relative_weights() -> None:
    """The scaling is uniform — all weights multiplied by the same scalar.

    So the ratio between any two weights is preserved.
    """
    rng = np.random.default_rng(7)
    strat_returns = pd.DataFrame(rng.normal(0, 0.01, (1000, 3)),
                                  columns=["A", "B", "C"])
    base_weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

    scaled = apply_vol_target(
        base_weights, strat_returns, target_vol_annual=0.10,
    )

    # Ratios unchanged.
    assert scaled["A"] / scaled["B"] == pytest.approx(0.5 / 0.3, rel=1e-10)
    assert scaled["B"] / scaled["C"] == pytest.approx(0.3 / 0.2, rel=1e-10)


def test_apply_vol_target_caps_at_max_leverage() -> None:
    """When the unconstrained scale exceeds the leverage cap, clip down."""
    quiet_returns = pd.DataFrame(
        {"A": _gaussian_returns(2520, std=0.002),  # extremely quiet
         "B": _gaussian_returns(2520, std=0.002, seed=7)},
    )
    base = pd.Series({"A": 0.5, "B": 0.5})  # gross = 1.0

    scaled = apply_vol_target(
        base, quiet_returns,
        target_vol_annual=0.10,
        max_gross_leverage=1.5,
    )

    gross_after = float(scaled.abs().sum())
    # The cap should bind: gross is exactly the cap (within float).
    assert gross_after == pytest.approx(1.5, abs=1e-10)


def test_apply_vol_target_mismatched_index_raises() -> None:
    """Catch bad inputs at the boundary, not deep in numpy land."""
    rets = pd.DataFrame({"A": [0.01, -0.005], "B": [0.02, 0.01]})
    weights = pd.Series({"A": 0.5, "X": 0.5})  # X not in rets
    with pytest.raises(ValueError, match="index"):
        apply_vol_target(weights, rets, target_vol_annual=0.10)


# ---------------------------------------------------------------------------
# kelly_leverage
# ---------------------------------------------------------------------------


def test_kelly_formula_matches_hand_computation() -> None:
    """leverage = fraction * mean_annual / variance_annual."""
    # Daily mean 0.001 (≈ 25% annualized), daily std 0.01 (≈ 16% annualized).
    # variance_annual = (0.01 * sqrt(252))^2 = 0.0252.
    # mean_annual = 0.001 * 252 = 0.252.
    # pure Kelly = 0.252 / 0.0252 = 10.0.  (Yes — pure Kelly is wild.)
    returns = _gaussian_returns(50_000, mean=0.001, std=0.01)
    full_kelly = kelly_leverage(returns, fraction=1.0)
    half_kelly = kelly_leverage(returns, fraction=0.5)

    # With a finite sample mean/var may drift; allow 5% tolerance.
    assert full_kelly == pytest.approx(10.0, rel=0.05)
    assert half_kelly == pytest.approx(5.0, rel=0.05)


def test_kelly_negative_mean_implies_negative_leverage() -> None:
    """If the strategy has negative drift, Kelly recommends a short — i.e.,
    negative leverage. Caller can choose to honor or floor at 0."""
    returns = _gaussian_returns(2520, mean=-0.0005, std=0.01)
    lev = kelly_leverage(returns, fraction=1.0)
    assert lev < 0


def test_kelly_zero_variance_returns_zero() -> None:
    """Defending against div-by-zero: no variance → no Kelly bet."""
    returns = pd.Series([0.001] * 252)
    assert kelly_leverage(returns, fraction=0.5) == 0.0


def test_kelly_fraction_must_be_positive() -> None:
    returns = _gaussian_returns(252)
    with pytest.raises(ValueError):
        kelly_leverage(returns, fraction=0)
    with pytest.raises(ValueError):
        kelly_leverage(returns, fraction=-0.5)
