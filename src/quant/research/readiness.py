"""readiness.py — phase-gate readiness reporter (research substrate, phase A5).

Answers "are we ready to start the next phase of the research desk?" so the
operator never has to track it by hand. It evaluates the OPEN phase gate's
conditions (design doc §7) against live artifacts — the trial ledger and
the shadow queue — and renders a checklist. That block is appended to the
monthly review email, so phase-readiness becomes a number that lands in the
inbox every month, the same way deployment % or entry fidelity do.

The substrate (phase A) is what's just been built, so the open gate is
A→B: "build the falsifier when the substrate has demonstrably done its
job." Later phases' gates live in the design doc and get wired here as each
phase lands. Conditions that are genuine judgment calls are shown with
their evidence and flagged as the operator's call — the reporter informs
the go/no-go, it doesn't pretend to make it.

Never raises; a missing ledger/queue just reads as "0 so far".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quant.research.ledger import TrialLedger
from quant.research.shadow_queue import ShadowQueue

# A candidate whose backtest Sharpe was at least this but failed live is the
# "backtest-good / shadow-bad" evidence that proves the falsifier is needed.
_GOOD_BACKTEST_SHARPE = 0.70


@dataclass(frozen=True)
class GateCheck:
    label: str
    passed: bool
    detail: str
    operator_call: bool = False   # true when this is a judgment call, not a hard count


@dataclass(frozen=True)
class PhaseGateReport:
    gate: str
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def ready(self) -> bool:
        return all(c.passed for c in self.checks)


def evaluate_readiness(
    *,
    ledger: TrialLedger | None = None,
    shadow_queue: ShadowQueue | None = None,
) -> PhaseGateReport:
    """Evaluate the currently-open phase gate (A→B). See design doc §7."""
    ledger = ledger or TrialLedger()
    shadow_queue = shadow_queue or ShadowQueue()

    summary = ledger.summary()
    trials = ledger.trials()
    n_structural = ledger.n_trials(kind="structural")
    reached_decision = summary.get("reached_decision", 0)

    # backtest-good / shadow-bad: a strategy trial that looked good in
    # backtest but ended killed/rejected — proves naive backtest-passing
    # is insufficient and an adversary is warranted.
    good_but_failed = [
        t for t in trials
        if t.kind == "strategy"
        and t.backtest_sharpe >= _GOOD_BACKTEST_SHARPE
        and t.outcome in {"killed", "rejected"}
    ]

    checks = [
        GateCheck(
            "≥3 candidates reached a promote/reject decision",
            reached_decision >= 3,
            f"now: {reached_decision}",
        ),
        GateCheck(
            "A/B harness scored ≥1 structural change",
            n_structural >= 1,
            f"now: {n_structural} structural trials",
        ),
        GateCheck(
            "≥1 backtest-good / shadow-bad candidate observed",
            len(good_but_failed) >= 1,
            f"now: {len(good_but_failed)} "
            f"(strategy backtest Sharpe ≥ {_GOOD_BACKTEST_SHARPE} then killed/rejected)",
            operator_call=True,
        ),
    ]
    return PhaseGateReport(gate="A→B (build the falsifier agent)", checks=checks)


def render_readiness_md(report: PhaseGateReport) -> str:
    """Markdown block for the monthly review email."""
    lines = ["## 🧭 Research desk — phase-gate readiness", ""]
    lines.append(f"**Next gate: {report.gate}** — "
                 f"{report.n_passed}/{len(report.checks)} conditions met"
                 + ("  ✅ ready" if report.ready else "  — not yet") + ".")
    lines.append("")
    for c in report.checks:
        box = "x" if c.passed else " "
        tail = "  _(operator call)_" if c.operator_call and not c.passed else ""
        lines.append(f"- [{box}] {c.label} — {c.detail}{tail}")
    if not report.ready:
        lines.append("")
        lines.append("_Build the next phase only when all conditions hold "
                     "(see docs/design/research-desk.md §7)._")
    return "\n".join(lines)
