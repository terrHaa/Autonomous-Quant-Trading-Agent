"""Tests for the strategy registry.

Each test uses a fresh ``tmp_path / "registry.db"`` so state doesn't leak.
We construct minimal-but-valid BacktestResults synthetically to avoid
running the full engine on every test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestResult
from quant.config import DEFAULT_CONFIG_PATH, Config
from quant.registry import STAGES, Registry

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _load_test_config() -> Config:
    """Pin the shipped default config so all tests use the same conventions."""
    import yaml

    return Config.model_validate(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))


def _make_result(
    strategy_name: str,
    *,
    mean_daily: float = 0.0005,
    std_daily: float = 0.01,
    n_days: int = 252,
    seed: int = 42,
) -> BacktestResult:
    """Synthesize a BacktestResult with a controllable equity curve.

    Uses a Gaussian return process so we can tune the resulting Sharpe.
    Only the fields the registry actually reads are filled in.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(mean_daily, std_daily, n_days)
    equity = 1_000_000 * np.cumprod(1 + returns)
    dates = list(pd.bdate_range("2020-01-02", periods=n_days).date)

    eq_series = pd.Series(equity, index=dates)
    config = _load_test_config()

    return BacktestResult(
        config=config,
        strategy_name=strategy_name,
        equity_curve=eq_series,
        positions=pd.DataFrame(),
        weights=pd.DataFrame(),
        orders=pd.DataFrame(),
        fills=pd.DataFrame(),
        costs=pd.DataFrame(),
        metadata={
            "n_bars": n_days,
            "n_orders": 0,
            "n_fills": 0,
            "start_date": dates[0],
            "end_date": dates[-1],
            "starting_equity": 1_000_000.0,
            "ending_equity": float(equity[-1]),
        },
    )


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "registry.db")


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_record_persists_and_get_retrieves(registry: Registry) -> None:
    """A recorded run is queryable by its returned ID."""
    result = _make_result("test_strat")
    run_id = registry.record(result, parameters={"foo": 1, "bar": "two"})

    row = registry.get(run_id)
    assert row is not None
    assert row["strategy_name"] == "test_strat"
    assert row["stage"] == "research"
    # parameters_json is stored as a string; the registry doesn't
    # auto-parse it back. Caller decodes when they need it.
    assert '"foo"' in row["parameters_json"]


def test_get_returns_none_for_missing_id(registry: Registry) -> None:
    assert registry.get("nonexistent-uuid") is None


def test_list_runs_returns_dataframe_with_recorded_rows(registry: Registry) -> None:
    registry.record(_make_result("A", seed=1))
    registry.record(_make_result("B", seed=2))
    registry.record(_make_result("A", seed=3))

    all_runs = registry.list_runs()
    assert len(all_runs) == 3
    assert set(all_runs["strategy_name"]) == {"A", "B"}


def test_list_runs_filters_by_strategy_name(registry: Registry) -> None:
    registry.record(_make_result("A", seed=1))
    registry.record(_make_result("B", seed=2))
    just_a = registry.list_runs(strategy_name="A")
    assert len(just_a) == 1
    assert just_a.iloc[0]["strategy_name"] == "A"


def test_list_runs_filters_by_stage(registry: Registry) -> None:
    rid1 = registry.record(_make_result("A", seed=1))
    # rid2 stays in "research" — exists so list_runs(stage="research")
    # has something to filter to (covered by other tests).
    registry.record(_make_result("B", seed=2))
    registry.promote(rid1, to_stage="walk_forward")

    wf = registry.list_runs(stage="walk_forward")
    assert len(wf) == 1
    assert wf.iloc[0]["id"] == rid1


def test_delete_removes_run(registry: Registry) -> None:
    rid = registry.record(_make_result("A"))
    assert registry.delete(rid) is True
    assert registry.get(rid) is None
    # Second delete returns False — idempotent-ish.
    assert registry.delete(rid) is False


def test_record_rejects_invalid_stage(registry: Registry) -> None:
    with pytest.raises(ValueError, match="stage"):
        registry.record(_make_result("A"), stage="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Trial accounting (the DSR plug)
# ---------------------------------------------------------------------------


def test_n_trials_counts_recorded_runs(registry: Registry) -> None:
    assert registry.n_trials() == 0
    for i in range(5):
        registry.record(_make_result(f"variant_{i}", seed=i))
    assert registry.n_trials() == 5


def test_trial_sharpes_returns_all_recorded_sharpes(registry: Registry) -> None:
    """Used by check_promotion_gate to compute V[SR] for DSR."""
    for i in range(4):
        registry.record(_make_result(f"variant_{i}", seed=i))
    sharpes = registry.trial_sharpes()
    assert len(sharpes) == 4
    assert all(isinstance(s, float) for s in sharpes)


# ---------------------------------------------------------------------------
# Promotion + gates
# ---------------------------------------------------------------------------


def test_promote_advances_stage(registry: Registry) -> None:
    rid = registry.record(_make_result("A"))
    registry.promote(rid, to_stage="walk_forward")
    assert registry.get(rid)["stage"] == "walk_forward"


def test_promote_rejects_backward_move(registry: Registry) -> None:
    """Once a strategy is in paper, you can't quietly demote it to research.

    Backward moves are how a sloppy researcher hides a previous promotion
    they wished hadn't happened. Force them to be explicit (delete + re-record).
    """
    rid = registry.record(_make_result("A"))
    registry.promote(rid, to_stage="walk_forward")
    with pytest.raises(ValueError, match="strictly later"):
        registry.promote(rid, to_stage="research")


def test_promote_rejects_same_stage(registry: Registry) -> None:
    """No-ops at least look like no-ops in the API."""
    rid = registry.record(_make_result("A"))
    with pytest.raises(ValueError, match="strictly later"):
        registry.promote(rid, to_stage="research")


def test_promote_rejects_unknown_stage(registry: Registry) -> None:
    rid = registry.record(_make_result("A"))
    with pytest.raises(ValueError, match="unknown stage"):
        registry.promote(rid, to_stage="staging")  # type: ignore[arg-type]


def test_promotion_gate_requires_returns_for_paper(registry: Registry) -> None:
    """No returns → can't compute DSR → can't gate-check paper promotion."""
    rid = registry.record(_make_result("A"))
    ok, reason = registry.check_promotion_gate(rid, to_stage="paper")
    assert ok is False
    assert "returns" in reason


def test_promotion_gate_blocks_low_dsr(registry: Registry) -> None:
    """A clearly-bad strategy should fail the DSR gate."""
    # Seed the registry with several decent strategies so V[SR] is meaningful.
    for i in range(20):
        registry.record(_make_result(f"variant_{i}", mean_daily=0.0005, seed=i))

    # The candidate: barely-positive Sharpe.
    candidate = _make_result(
        "weak", mean_daily=0.00005, std_daily=0.01, seed=999, n_days=500,
    )
    rid = registry.record(candidate)
    candidate_returns = candidate.equity_curve.pct_change().dropna()

    ok, reason = registry.check_promotion_gate(
        rid, to_stage="paper", returns=candidate_returns,
    )
    assert ok is False
    assert "DSR" in reason


def test_promotion_gate_allows_strong_strategy(registry: Registry) -> None:
    """A clearly-strong strategy with few trials should pass.

    Setup: only 3 trials in the registry, with one a real winner.
    Low n_trials → minimal deflation → easy to pass.
    """
    # Seed with two unrelated trials.
    registry.record(_make_result("A", mean_daily=0.0001, seed=1))
    registry.record(_make_result("B", mean_daily=0.0002, seed=2))

    # The candidate: Sharpe ~3 (very high), long sample.
    candidate = _make_result(
        "winner", mean_daily=0.002, std_daily=0.01, seed=3, n_days=2000,
    )
    rid = registry.record(candidate)
    candidate_returns = candidate.equity_curve.pct_change().dropna()

    # Move to walk_forward first (so we're testing the wf→paper hop).
    registry.promote(rid, to_stage="walk_forward")
    ok, reason = registry.check_promotion_gate(
        rid, to_stage="paper", returns=candidate_returns,
    )
    assert ok is True
    assert "DSR" in reason


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------


def test_registry_persists_across_instances(tmp_path: Path) -> None:
    """Re-opening the same path sees the prior session's data.

    This is the whole reason for the SQLite backend — research is a
    multi-day process, the registry must survive process restarts.
    """
    path = tmp_path / "reg.db"
    reg1 = Registry(path)
    rid = reg1.record(_make_result("persistent"))

    # Brand-new Registry instance on the same path.
    reg2 = Registry(path)
    assert reg2.n_trials() == 1
    assert reg2.get(rid)["strategy_name"] == "persistent"


def test_known_stages_constant_is_immutable() -> None:
    """STAGES tuple defines the ladder; downstream code may iterate it."""
    assert STAGES == ("research", "walk_forward", "paper", "live")
