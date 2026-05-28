"""Tests for the multi-strategy ensemble.

State persistence + the per-symbol combination math + the HRP refit
mechanics. The refit test uses synthetic strategies that emit
predictable, distinct return streams so HRP's allocation is testable
without running real backtests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.agent import ensemble as ensemble_mod
from quant.agent.ensemble import (
    EnsembleState,
    build_strategies,
    compute_ensemble_targets,
    load_ensemble_state,
    refit_hrp_weights,
    save_ensemble_state,
)
from quant.backtest.types import Snapshot
from quant.data.alpaca_client import BAR_COLUMNS

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_load_returns_defaults_when_no_file(tmp_path: Path) -> None:
    """Fresh install: empty disk → safe defaults with equal HRP weights."""
    state = load_ensemble_state(path=tmp_path / "nope.json")
    assert state.sma_fast == 50
    assert state.sma_slow == 200
    assert state.mr_lookback == 5
    assert state.xsec_top_k == 10
    # Three strategies, equal HRP weight.
    assert len(state.hrp_weights) == 3
    assert sum(state.hrp_weights.values()) == pytest.approx(1.0)


def test_save_and_load_round_trips(tmp_path: Path) -> None:
    state = EnsembleState(
        sma_fast=20, sma_slow=100,
        mr_lookback=10, mr_threshold_pct=0.03, mr_allow_short=True,
        xsec_top_k=5, xsec_lookback=120, xsec_skip=10,
        hrp_weights={"a": 0.5, "b": 0.3, "c": 0.2},
        last_hrp_refit_date="2024-06-01",
    )
    fp = tmp_path / "state.json"
    save_ensemble_state(state, path=fp)
    loaded = load_ensemble_state(path=fp)
    assert loaded == state


# ---------------------------------------------------------------------------
# Strategy construction
# ---------------------------------------------------------------------------


def test_build_strategies_returns_three_with_distinct_names() -> None:
    """Three strategies, three distinct names — needed for HRP keying."""
    state = EnsembleState()
    # CrossSectionalMomentum requires top_k (10) <= universe size.
    universe = [f"SYM{i}" for i in range(10)]
    strats = build_strategies(state, universe)
    assert len(strats) == 3
    names = {s.name for s in strats}
    assert len(names) == 3
    # The default HRP weights' keys must MATCH the actual strategy names.
    assert set(state.hrp_weights.keys()) == names


# ---------------------------------------------------------------------------
# compute_ensemble_targets — the combination math
# ---------------------------------------------------------------------------


class _ConstantStrategy:
    """Returns a fixed dict every call. For testing combination math
    without depending on the real strategies' signal logic."""

    def __init__(self, name: str, targets: dict[str, float]):
        self.name = name
        self._targets = targets

    def on_bar(self, snapshot):
        return dict(self._targets)


def _dummy_snapshot() -> Snapshot:
    empty = pd.DataFrame(
        columns=list(BAR_COLUMNS),
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )
    from datetime import date
    return Snapshot.from_full_bars(empty, as_of=date(2024, 6, 1))


def test_combination_is_weighted_inner_product() -> None:
    """final[sym] = Σ_strategy h_strategy × strategy_target[sym]."""
    strategies = [
        _ConstantStrategy("A", {"AAPL": 1.0}),
        _ConstantStrategy("B", {"AAPL": 0.5, "MSFT": 0.5}),
        _ConstantStrategy("C", {"NVDA": 1.0}),
    ]
    hrp = {"A": 0.4, "B": 0.3, "C": 0.3}
    out = compute_ensemble_targets(strategies, hrp, _dummy_snapshot())
    # AAPL: 0.4*1.0 + 0.3*0.5 = 0.55
    # MSFT: 0.3*0.5 = 0.15
    # NVDA: 0.3*1.0 = 0.30
    assert out["AAPL"] == pytest.approx(0.55)
    assert out["MSFT"] == pytest.approx(0.15)
    assert out["NVDA"] == pytest.approx(0.30)


def test_strategy_not_in_hrp_weights_gets_zero() -> None:
    """If a strategy isn't keyed in hrp_weights, its targets are ignored."""
    strategies = [
        _ConstantStrategy("KNOWN", {"AAPL": 1.0}),
        _ConstantStrategy("UNKNOWN", {"NVDA": 1.0}),
    ]
    hrp = {"KNOWN": 1.0}   # UNKNOWN missing
    out = compute_ensemble_targets(strategies, hrp, _dummy_snapshot())
    assert "AAPL" in out
    assert "NVDA" not in out


def test_dust_positions_dropped() -> None:
    """Positions below 0.5% (the agent's stop-loss noise floor) are
    pruned to keep the order book clean."""
    # Strategy A picks 200 names equal-weighted (0.5%/each); strategy B is
    # concentrated. HRP weight on A is 0.5, on B is 0.5.
    # Contribution of A to each name = 0.5 * 0.005 = 0.0025 = 25 bps. Dust.
    a_targets = {f"S{i}": 1 / 200 for i in range(200)}
    strategies = [
        _ConstantStrategy("A", a_targets),
        _ConstantStrategy("B", {"AAPL": 1.0}),
    ]
    hrp = {"A": 0.5, "B": 0.5}
    out = compute_ensemble_targets(strategies, hrp, _dummy_snapshot())
    # AAPL gets 0.5 * 1.0 = 0.5; kept.
    assert out["AAPL"] == pytest.approx(0.5)
    # The 200 dust names from A each got 0.5 * 0.005 = 0.0025; under threshold.
    assert not any(s.startswith("S") for s in out)


def test_zero_weight_strategy_is_short_circuited() -> None:
    """A strategy with HRP weight 0 should be skipped (perf optimization)."""
    # If we DON'T skip, the test still passes; but we can verify the
    # output doesn't accidentally include the zero-weight strategy's
    # picks (which would happen if 0 * something rounded to non-zero).
    strategies = [
        _ConstantStrategy("ZERO", {"DOOMED": 1.0}),
        _ConstantStrategy("ALIVE", {"AAPL": 1.0}),
    ]
    hrp = {"ZERO": 0.0, "ALIVE": 1.0}
    out = compute_ensemble_targets(strategies, hrp, _dummy_snapshot())
    assert "DOOMED" not in out
    assert out["AAPL"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Safety: per-strategy exception isolation
# ---------------------------------------------------------------------------


class _RaisingStrategy:
    """on_bar always raises — simulates a buggy AI-generated strategy."""

    def __init__(self, name: str, exc: Exception):
        self.name = name
        self._exc = exc

    def on_bar(self, snapshot):
        raise self._exc


def test_one_strategy_raising_does_not_break_the_others() -> None:
    """If one strategy raises during on_bar, the others must still contribute.

    This is the most important operational protection — a bug in
    AI-generated code cannot crash the entire daily trade routine.
    """
    strategies = [
        _ConstantStrategy("STABLE_A", {"AAPL": 1.0}),
        _RaisingStrategy("BUGGY", ZeroDivisionError("division by zero")),
        _ConstantStrategy("STABLE_B", {"MSFT": 1.0}),
    ]
    hrp = {"STABLE_A": 0.4, "BUGGY": 0.3, "STABLE_B": 0.3}
    out = compute_ensemble_targets(strategies, hrp, _dummy_snapshot())
    # BUGGY contributed nothing. STABLE_A and STABLE_B still applied.
    assert out["AAPL"] == pytest.approx(0.4)
    assert out["MSFT"] == pytest.approx(0.3)


def test_strategy_raising_does_not_propagate_to_caller() -> None:
    """compute_ensemble_targets must never raise even if every strategy fails."""
    strategies = [
        _RaisingStrategy("A", RuntimeError("a")),
        _RaisingStrategy("B", ValueError("b")),
    ]
    hrp = {"A": 0.5, "B": 0.5}
    # Should return an empty dict, NOT raise.
    out = compute_ensemble_targets(strategies, hrp, _dummy_snapshot())
    assert out == {}


# ---------------------------------------------------------------------------
# Safety: shadow mode
# ---------------------------------------------------------------------------


def test_shadow_strategies_do_not_contribute_to_targets() -> None:
    """A strategy in shadow_strategies must have ZERO impact on combined targets."""
    strategies = [
        _ConstantStrategy("ACTIVE", {"AAPL": 1.0}),
        _ConstantStrategy("SHADOW", {"NVDA": 1.0}),
    ]
    hrp = {"ACTIVE": 0.5, "SHADOW": 0.5}  # even though SHADOW has weight,
    out = compute_ensemble_targets(
        strategies, hrp, _dummy_snapshot(),
        shadow_strategies={"SHADOW"},
    )
    # SHADOW's targets are NOT in the combined output.
    assert out["AAPL"] == pytest.approx(0.5)
    assert "NVDA" not in out


def test_shadow_targets_are_recorded() -> None:
    """If record_shadow_targets dict is passed, shadow strategies' targets
    are captured for later analysis (so we can see what they'd have traded)."""
    strategies = [
        _ConstantStrategy("ACTIVE", {"AAPL": 1.0}),
        _ConstantStrategy("SHADOW", {"NVDA": 0.7, "AMD": 0.3}),
    ]
    hrp = {"ACTIVE": 1.0, "SHADOW": 0.0}
    sink: dict[str, dict[str, float]] = {}
    compute_ensemble_targets(
        strategies, hrp, _dummy_snapshot(),
        shadow_strategies={"SHADOW"},
        record_shadow_targets=sink,
    )
    assert "SHADOW" in sink
    assert sink["SHADOW"]["NVDA"] == pytest.approx(0.7)
    assert sink["SHADOW"]["AMD"] == pytest.approx(0.3)
    # Active strategy is NOT recorded as shadow.
    assert "ACTIVE" not in sink


def test_shadow_state_field_defaults_to_empty() -> None:
    """New EnsembleState has empty shadow map (backwards-compatible)."""
    state = EnsembleState()
    assert state.ai_strategy_shadow_until == {}


# ---------------------------------------------------------------------------
# refit_hrp_weights — the weekly self-improvement
# ---------------------------------------------------------------------------


def test_refit_returns_weights_summing_to_one(monkeypatch) -> None:
    """The refit's output must be valid HRP weights (non-negative, sum to 1)."""
    # Stub out run_backtest to return predictable equity curves so we
    # don't run a real backtest in this unit test.
    import yaml

    from quant.config import DEFAULT_CONFIG_PATH, Config

    config = Config.model_validate(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))

    # Synthesize equity curves with different vols / correlation.
    rng = np.random.default_rng(0)
    n = 500
    dates = list(pd.bdate_range("2023-01-02", periods=n).date)

    # Three streams with deliberately different vols so HRP will give
    # the lower-vol ones higher weight.
    rets_a = rng.normal(0.0005, 0.005, n)
    rets_b = rng.normal(0.0005, 0.010, n)
    rets_c = rng.normal(0.0005, 0.020, n)
    counter = [0]

    class _FakeResult:
        def __init__(self, rets):
            eq = 1_000_000 * np.cumprod(1 + rets)
            self.equity_curve = pd.Series(eq, index=dates)

    def fake_backtest(*, config, strategy, bars):
        ret_streams = [rets_a, rets_b, rets_c]
        result = _FakeResult(ret_streams[counter[0]])
        counter[0] += 1
        return result

    monkeypatch.setattr(ensemble_mod, "run_backtest", fake_backtest)

    state = EnsembleState()
    new_weights, diag = refit_hrp_weights(
        state, universe=[f"SYM{i}" for i in range(10)],
        bars=pd.DataFrame(),   # ignored in our stub
        config=config,
    )

    assert sum(new_weights.values()) == pytest.approx(1.0)
    assert all(w >= 0 for w in new_weights.values())
    # The strategy keyed to the lowest-vol stream (rets_a, SMA) should
    # get a meaningfully larger HRP weight than the highest-vol (XsecMomentum).
    assert new_weights["sma_crossover_50_200"] > new_weights["xsec_momo_60_5_10"]
    # Diagnostic contains the before/after for the report.
    assert "hrp_weights_before" in diag
    assert "hrp_weights_after" in diag


def test_refit_falls_back_to_equal_weight_when_strategies_degenerate(monkeypatch) -> None:
    """If most strategies produce flat equity (never traded), HRP can't
    run; fall back to equal weights rather than zero out everything."""
    import yaml

    from quant.config import DEFAULT_CONFIG_PATH, Config

    config = Config.model_validate(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))

    # All three strategies return a perfectly flat equity curve.
    n = 500
    dates = list(pd.bdate_range("2023-01-02", periods=n).date)

    class _FlatResult:
        equity_curve = pd.Series([1_000_000.0] * n, index=dates)

    monkeypatch.setattr(
        ensemble_mod, "run_backtest",
        lambda *, config, strategy, bars: _FlatResult(),
    )

    state = EnsembleState()
    new_weights, diag = refit_hrp_weights(
        state, universe=[f"SYM{i}" for i in range(10)],
        bars=pd.DataFrame(), config=config,
    )
    # Equal-weight fallback: each strategy gets 1/3.
    assert all(abs(w - 1/3) < 1e-9 for w in new_weights.values())
    assert "fallback" in diag
