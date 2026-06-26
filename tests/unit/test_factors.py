"""Tests for the factor library + attribution engine (Phase 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.factors import (
    FACTOR_NAMES,
    attribute_returns,
    compute_factor_returns,
)
from quant.factors.attribution import AttributionResult


def _synthetic_bars(n_days: int = 400, n_syms: int = 60, seed: int = 0) -> pd.DataFrame:
    """OHLCV bars with a PLANTED momentum structure.

    Each symbol has a persistent drift; high-drift names keep winning, so
    a 12-1 momentum long-short must earn a positive return. Returns the
    platform's MultiIndex (symbol, timestamp) frame.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n_days)
    drifts = np.linspace(-0.0008, 0.0008, n_syms)  # persistent per-name drift
    rows, idx = [], []
    for j in range(n_syms):
        price = 100.0
        for ts in days:
            ret = drifts[j] + rng.normal(0, 0.01)
            price *= (1 + ret)
            rows.append({"open": price, "high": price, "low": price,
                         "close": price, "volume": 1000})
            idx.append((f"S{j:02d}", ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=["open", "high", "low", "close", "volume"],
    )


def test_factor_panel_has_expected_columns_and_no_lookahead_nans() -> None:
    fr = compute_factor_returns(_synthetic_bars())
    assert list(fr.columns) == list(FACTOR_NAMES)
    assert not fr.isna().any().any()      # dropna(how=any) leaves a clean panel
    # MOM needs ~252d runway, so the panel must start well after day 0.
    assert len(fr) > 50


def test_momentum_factor_captures_planted_drift() -> None:
    # With monotonic persistent drift, 12-1 momentum should be net positive.
    fr = compute_factor_returns(_synthetic_bars(seed=1))
    assert fr["MOM"].mean() > 0


def test_empty_bars_returns_empty_panel() -> None:
    empty = pd.DataFrame(
        [], index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
        columns=["open", "high", "low", "close", "volume"],
    )
    assert compute_factor_returns(empty).empty


def test_attribution_recovers_planted_alpha_and_beta() -> None:
    rng = np.random.default_rng(7)
    n = 500
    idx = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=n)
    f = pd.DataFrame(
        {"MKT": rng.normal(0, 0.01, n), "MOM": rng.normal(0, 0.015, n)},
        index=idx,
    )
    true_alpha_daily = 0.0004
    port = (true_alpha_daily + 0.8 * f["MKT"] + 0.3 * f["MOM"]
            + rng.normal(0, 0.002, n))
    res = attribute_returns(port, f)
    assert isinstance(res, AttributionResult)
    assert res.betas["MKT"] == pytest.approx(0.8, abs=0.08)
    assert res.betas["MOM"] == pytest.approx(0.3, abs=0.08)
    assert res.alpha_daily == pytest.approx(true_alpha_daily, abs=0.0002)
    assert res.alpha_tstat > 2          # 500 obs → significant
    assert res.r_squared > 0.8
    assert not res.warnings             # 500 obs → not underpowered


def test_attribution_flags_underpowered_short_history() -> None:
    rng = np.random.default_rng(3)
    n = 20
    idx = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=n)
    f = pd.DataFrame({"MKT": rng.normal(0, 0.01, n), "MOM": rng.normal(0, 0.01, n)},
                     index=idx)
    port = 0.5 * f["MKT"] + rng.normal(0, 0.003, n)
    res = attribute_returns(port, f)
    assert res.n_obs == n
    assert any("underpowered" in w for w in res.warnings)


def test_attribution_raises_when_too_few_obs() -> None:
    idx = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=3)
    f = pd.DataFrame({"MKT": [0.01, -0.01, 0.0], "MOM": [0.0, 0.01, -0.01]}, index=idx)
    port = pd.Series([0.01, 0.0, -0.01], index=idx)
    with pytest.raises(ValueError):
        attribute_returns(port, f)

