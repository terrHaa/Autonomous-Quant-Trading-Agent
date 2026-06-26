"""Tests for the regime-policy auto-apply gate."""
from __future__ import annotations

import pandas as pd

from quant.agent.regime_gate import (
    gate_regime_policy,
    validate_regime_policy,
)

_NAMES = {"sma", "mr"}


def test_validate_accepts_well_formed_policy() -> None:
    ok, _ = validate_regime_policy(
        {"mr": {"trend_up_calm": 0.0, "trend_up_stormy": 0.5}}, _NAMES
    )
    assert ok


def test_validate_rejects_unknown_strategy() -> None:
    ok, reason = validate_regime_policy({"ghost": {"trend_up_calm": 0.5}}, _NAMES)
    assert not ok and "unknown strategy" in reason


def test_validate_rejects_unknown_regime_label() -> None:
    ok, reason = validate_regime_policy({"mr": {"bull_market": 0.5}}, _NAMES)
    assert not ok and "unknown regime" in reason


def test_validate_rejects_out_of_bounds_multiplier() -> None:
    ok, reason = validate_regime_policy({"mr": {"trend_up_calm": 9.0}}, _NAMES)
    assert not ok and "out of" in reason


def test_validate_rejects_empty_policy() -> None:
    ok, reason = validate_regime_policy({}, _NAMES)
    assert not ok


class _FakeStrat:
    def __init__(self, name: str) -> None:
        self.name = name

    def on_bar(self, snapshot):  # noqa: ANN001
        return {s: 1.0 for s in snapshot.symbols()[:5]}


def _tiny_bars() -> pd.DataFrame:
    days = pd.bdate_range(end=pd.Timestamp("2024-06-01"), periods=10)
    rows, idx = [], []
    for sym in ("AAA", "BBB", "CCC", "DDD", "EEE"):
        for k, ts in enumerate(days):
            c = 100.0 + k
            rows.append({"open": c, "high": c, "low": c, "close": c, "volume": 1})
            idx.append((sym, ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=["open", "high", "low", "close", "volume"],
    )


def test_gate_rejects_structurally_invalid_before_backtesting() -> None:
    strats = [_FakeStrat("sma"), _FakeStrat("mr")]
    g = gate_regime_policy(
        {"ghost": {"trend_up_calm": 0.5}}, strats,
        {"sma": 0.6, "mr": 0.4}, _tiny_bars(),
    )
    assert not g.passed
    assert "unknown strategy" in g.reason


def test_gate_rejects_on_insufficient_history() -> None:
    strats = [_FakeStrat("sma"), _FakeStrat("mr")]
    g = gate_regime_policy(
        {"mr": {"trend_up_calm": 0.0}}, strats,
        {"sma": 0.6, "mr": 0.4}, _tiny_bars(),
    )
    assert not g.passed
    assert "insufficient" in g.reason.lower()
