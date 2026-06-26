"""Tests for regime classification + regime-aware allocation (Phase 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.risk.regime import (
    REGIME_LABELS,
    apply_regime_policy,
    average_pairwise_correlation,
    classify_regime,
    correlation_degross_factor,
)


def _series(values):
    idx = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=len(values))
    return pd.Series(values, index=idx)


def test_classify_uptrend_vs_downtrend() -> None:
    up = _series(100 + np.arange(300) * 0.3)        # steadily rising
    down = _series(200 - np.arange(300) * 0.3)      # steadily falling
    assert classify_regime(up).startswith("trend_up")
    assert classify_regime(down).startswith("trend_dn")


def test_classify_returns_valid_label_and_is_benign_on_short_history() -> None:
    assert classify_regime(_series([100, 101, 102])) == "trend_up_calm"
    label = classify_regime(_series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 300))))
    assert label in REGIME_LABELS


def test_regime_policy_scales_and_renormalizes() -> None:
    hrp = {"sma": 0.6, "mr": 0.4}
    # In an up-trend, switch MR off (its IC is ~0 there).
    policy = {"mr": {"trend_up_calm": 0.0}}
    out = apply_regime_policy(hrp, "trend_up_calm", policy)
    assert out["mr"] == pytest.approx(0.0)
    assert out["sma"] == pytest.approx(1.0)            # renormalized to gross 1.0
    assert sum(out.values()) == pytest.approx(sum(hrp.values()))


def test_regime_policy_passthrough_when_no_policy() -> None:
    hrp = {"sma": 0.6, "mr": 0.4}
    assert apply_regime_policy(hrp, "trend_dn_calm", None) == hrp
    assert apply_regime_policy(hrp, "trend_dn_calm", {}) == hrp


def test_regime_policy_falls_back_when_everything_zeroed() -> None:
    hrp = {"sma": 0.5, "mr": 0.5}
    policy = {"sma": {"x": 0.0}, "mr": {"x": 0.0}}
    # Refuses to return an empty book.
    assert apply_regime_policy(hrp, "x", policy) == hrp


def test_correlation_degross_ramps() -> None:
    assert correlation_degross_factor(0.2) == 1.0          # calm
    assert correlation_degross_factor(0.6) == pytest.approx(0.5)  # crisis
    mid = correlation_degross_factor(0.45)
    assert 0.5 < mid < 1.0                                  # linear middle
    assert correlation_degross_factor(float("nan")) == 1.0  # fail-open


def test_average_pairwise_correlation_detects_comovement() -> None:
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.01, 80)
    # Highly correlated: each name = common factor + tiny idiosyncratic noise.
    corr_df = pd.DataFrame({f"S{i}": common + rng.normal(0, 0.001, 80) for i in range(8)})
    indep_df = pd.DataFrame({f"S{i}": rng.normal(0, 0.01, 80) for i in range(8)})
    assert average_pairwise_correlation(corr_df) > 0.8
    assert abs(average_pairwise_correlation(indep_df)) < 0.3
