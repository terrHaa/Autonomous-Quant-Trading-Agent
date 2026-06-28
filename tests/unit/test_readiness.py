"""Tests for the phase-gate readiness reporter (research substrate, A5)."""
from __future__ import annotations

from quant.research import (
    ShadowQueue,
    TrialLedger,
    evaluate_readiness,
    render_readiness_md,
)


def test_empty_substrate_is_not_ready(tmp_path) -> None:
    led = TrialLedger(path=tmp_path / "l.jsonl")
    q = ShadowQueue(path=tmp_path / "q.json")
    rep = evaluate_readiness(ledger=led, shadow_queue=q)
    assert rep.gate.startswith("A→B")
    assert not rep.ready
    assert rep.n_passed == 0
    assert len(rep.checks) == 3


def test_all_conditions_met_reports_ready(tmp_path) -> None:
    led = TrialLedger(path=tmp_path / "l.jsonl")
    q = ShadowQueue(path=tmp_path / "q.json")
    # 3 candidates that reached a decision (promote/reject).
    for i in range(3):
        tid = led.log_trial(kind="strategy", name=f"s{i}", backtest_sharpe=0.9)
        led.log_outcome(tid, "rejected" if i else "promoted", "decided")
    # ≥1 structural A/B trial.
    led.log_trial(kind="structural", name="ab1", backtest_sharpe=0.5)
    # backtest-good / shadow-bad already covered (s1/s2 had Sharpe 0.9 and
    # were rejected) → condition 3 satisfied.
    rep = evaluate_readiness(ledger=led, shadow_queue=q)
    assert rep.ready, render_readiness_md(rep)
    assert rep.n_passed == 3


def test_backtest_good_shadow_bad_condition(tmp_path) -> None:
    led = TrialLedger(path=tmp_path / "l.jsonl")
    q = ShadowQueue(path=tmp_path / "q.json")
    # A weak strategy that was rejected does NOT count (backtest not good).
    tid = led.log_trial(kind="strategy", name="weak", backtest_sharpe=0.2)
    led.log_outcome(tid, "rejected")
    rep = evaluate_readiness(ledger=led, shadow_queue=q)
    cond3 = rep.checks[2]
    assert not cond3.passed          # 0.2 Sharpe doesn't meet "backtest-good"
    # Now a strong one that failed live → counts.
    tid2 = led.log_trial(kind="strategy", name="strong", backtest_sharpe=1.4)
    led.log_outcome(tid2, "killed")
    rep2 = evaluate_readiness(ledger=led, shadow_queue=q)
    assert rep2.checks[2].passed


def test_render_contains_gate_and_checkboxes(tmp_path) -> None:
    led = TrialLedger(path=tmp_path / "l.jsonl")
    q = ShadowQueue(path=tmp_path / "q.json")
    md = render_readiness_md(evaluate_readiness(ledger=led, shadow_queue=q))
    assert "phase-gate readiness" in md
    assert "A→B" in md
    assert "[ ]" in md               # unmet conditions render as empty boxes
