"""Tests for the global trial ledger (research substrate, phase A1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research import TrialLedger


def _ledger(tmp_path) -> TrialLedger:
    return TrialLedger(path=tmp_path / "ledger.jsonl")


def test_log_and_fold_trial(tmp_path) -> None:
    led = _ledger(tmp_path)
    tid = led.log_trial(kind="strategy", name="rsi_revert_14", backtest_sharpe=1.2)
    trials = led.trials()
    assert len(trials) == 1
    t = trials[0]
    assert t.trial_id == tid
    assert t.name == "rsi_revert_14"
    assert t.outcome == "proposed"           # no outcome yet
    assert not t.is_terminal


def test_outcome_transitions_fold_to_latest(tmp_path) -> None:
    led = _ledger(tmp_path)
    tid = led.log_trial(kind="strategy", name="s1", backtest_sharpe=0.9)
    led.log_outcome(tid, "shadow", "entered paper")
    led.log_outcome(tid, "promoted", "cleared gate")
    t = led.trials()[0]
    assert t.outcome == "promoted"
    assert t.is_terminal
    assert "cleared gate" in t.outcome_reason


def test_rejects_unknown_outcome(tmp_path) -> None:
    led = _ledger(tmp_path)
    tid = led.log_trial(kind="factor", name="f1", backtest_sharpe=0.5)
    with pytest.raises(ValueError):
        led.log_outcome(tid, "deployed_to_mars")


def test_trial_population_filters_by_kind(tmp_path) -> None:
    led = _ledger(tmp_path)
    led.log_trial(kind="strategy", name="s1", backtest_sharpe=1.0)
    led.log_trial(kind="strategy", name="s2", backtest_sharpe=1.5)
    led.log_trial(kind="factor", name="f1", backtest_sharpe=0.3)
    assert led.n_trials() == 3
    assert led.n_trials(kind="strategy") == 2
    assert sorted(led.trial_sharpes(kind="strategy")) == [1.0, 1.5]


def test_global_dsr_deflates_harder_as_trials_grow(tmp_path) -> None:
    """The keystone: the SAME candidate looks less impressive once the
    ledger knows we ran many trials to find it (multiple-testing)."""
    rng = np.random.default_rng(0)
    # A modestly-good candidate return series (Sharpe ~1).
    idx = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=252)
    cand = pd.Series(0.001 + rng.normal(0, 0.01, 252), index=idx)

    led_few = _ledger(tmp_path)
    led_few.log_trial(kind="strategy", name="a", backtest_sharpe=1.0)
    led_few.log_trial(kind="strategy", name="b", backtest_sharpe=1.1)
    dsr_few = led_few.deflated_sharpe(cand, kind="strategy", include_self_sharpe=1.0)

    led_many = TrialLedger(path=tmp_path / "many.jsonl")
    for i in range(60):
        led_many.log_trial(kind="strategy", name=f"v{i}",
                           backtest_sharpe=float(rng.normal(0.8, 0.4)))
    dsr_many = led_many.deflated_sharpe(cand, kind="strategy", include_self_sharpe=1.0)

    # More trials searched → same candidate is deflated to a lower PSR.
    assert dsr_many < dsr_few


def test_deflated_sharpe_handles_empty_ledger(tmp_path) -> None:
    led = _ledger(tmp_path)
    idx = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=60)
    cand = pd.Series(np.full(60, 0.0005), index=idx)
    # No trials logged, candidate not self-counted → n_trials falls back to 1.
    val = led.deflated_sharpe(cand)
    assert 0.0 <= val <= 1.0


def test_summary_counts(tmp_path) -> None:
    led = _ledger(tmp_path)
    t1 = led.log_trial(kind="strategy", name="s1", backtest_sharpe=1.0)
    led.log_trial(kind="strategy", name="s2", backtest_sharpe=1.0)
    led.log_outcome(t1, "promoted")
    s = led.summary()
    assert s["total_trials"] == 2
    assert s["reached_decision"] == 1
    assert s["by_kind"]["strategy"] == 2


def test_append_only_survives_reload(tmp_path) -> None:
    p = tmp_path / "ledger.jsonl"
    led1 = TrialLedger(path=p)
    tid = led1.log_trial(kind="strategy", name="s1", backtest_sharpe=1.0)
    led1.log_outcome(tid, "shadow")
    # Fresh instance reads the same file.
    led2 = TrialLedger(path=p)
    assert led2.trials()[0].outcome == "shadow"
