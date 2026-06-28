"""ledger.py — the global trial ledger (research substrate, phase A1).

Every hypothesis the system ever tests — a proposed strategy, a factor, a
structural A/B variant, a parameter sweep — is logged here, once, forever.
Why this is the non-negotiable foundation of the research desk:

Continuous + parallel research makes the multiple-testing problem WORSE,
not better. The more variants you try, the more likely one clears any bar
by luck (the look-elsewhere effect). The only defense is to judge every
candidate's Sharpe *deflated for the total number of trials* — and that
total has to be cumulative across all agents and all time, not reset per
run. This ledger is what makes the trial count global, so
``deflated_sharpe`` here is the honest, multiple-testing-corrected DSR that
the shadow queue and promotion gate consume. Without it, a continuous
research loop is a machine for harvesting false positives.

Design: event-sourced, append-only JSONL. Each line is one immutable
event. A trial is a sequence of events sharing a ``trial_id`` — a
``proposed`` event carrying the backtest stats, then ``outcome`` events as
it moves (killed / shadow / promoted / rejected / retired). The current
state of a trial is the fold of its events. Append-only means the full
audit trail survives — we can always ask "what did we know when we
deployed this, and how many trials had we run by then."

Never raises on read; a telemetry substrate must not break its consumers.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/agent/research/trial_ledger.jsonl")

# Lifecycle a trial can move through. "proposed" is logged once at creation;
# the rest are outcome transitions.
OUTCOMES: tuple[str, ...] = (
    "proposed", "killed", "shadow", "promoted", "rejected", "retired",
)
# Terminal outcomes — a trial that reached one of these is done moving.
_TERMINAL = {"killed", "promoted", "rejected", "retired"}


@dataclass(frozen=True)
class TrialView:
    """The folded current state of one trial (its proposed event + latest outcome)."""

    trial_id: str
    kind: str               # "strategy" | "factor" | "structural" | "param"
    name: str
    family: str
    generator: str          # who proposed it (e.g. "monthly_analyst")
    backtest_sharpe: float
    proposed_at: str
    outcome: str            # latest lifecycle state
    outcome_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.outcome in _TERMINAL


class TrialLedger:
    """Append-only, event-sourced record of every experiment tested."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else _DEFAULT_PATH

    # ---- writes -----------------------------------------------------------

    def _append(self, event: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

    def log_trial(
        self,
        *,
        kind: str,
        name: str,
        backtest_sharpe: float,
        family: str = "",
        generator: str = "monthly_analyst",
        trial_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a newly-proposed trial. Returns its ``trial_id``.

        ``backtest_sharpe`` is the annualized in/out-of-sample Sharpe from
        the candidate's backtest — this is what enters the trial population
        used to deflate future candidates.
        """
        tid = trial_id or uuid.uuid4().hex[:12]
        self._append({
            "event": "proposed",
            "trial_id": tid,
            "ts": datetime.now(UTC).isoformat(),
            "kind": kind,
            "name": name,
            "family": family,
            "generator": generator,
            "backtest_sharpe": float(backtest_sharpe),
            "metadata": metadata or {},
        })
        return tid

    def log_outcome(self, trial_id: str, outcome: str, reason: str = "") -> None:
        """Record a lifecycle transition for an existing trial."""
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}; must be one of {OUTCOMES}")
        self._append({
            "event": "outcome",
            "trial_id": trial_id,
            "ts": datetime.now(UTC).isoformat(),
            "outcome": outcome,
            "reason": reason,
        })

    # ---- reads ------------------------------------------------------------

    def _events(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    logger.warning("trial ledger: skipping malformed line")
        except OSError as e:
            logger.warning("trial ledger read failed: %s", e)
        return out

    def trials(self) -> list[TrialView]:
        """Fold the event log into current per-trial state."""
        proposed: dict[str, dict[str, Any]] = {}
        latest_outcome: dict[str, tuple[str, str]] = {}
        for ev in self._events():
            tid = ev.get("trial_id")
            if not tid:
                continue
            if ev.get("event") == "proposed":
                proposed[tid] = ev
            elif ev.get("event") == "outcome":
                latest_outcome[tid] = (ev.get("outcome", ""), ev.get("reason", ""))
        views: list[TrialView] = []
        for tid, ev in proposed.items():
            outcome, reason = latest_outcome.get(tid, ("proposed", ""))
            views.append(TrialView(
                trial_id=tid,
                kind=ev.get("kind", ""),
                name=ev.get("name", ""),
                family=ev.get("family", ""),
                generator=ev.get("generator", ""),
                backtest_sharpe=float(ev.get("backtest_sharpe", 0.0)),
                proposed_at=ev.get("ts", ""),
                outcome=outcome,
                outcome_reason=reason,
                metadata=ev.get("metadata", {}),
            ))
        return views

    def trial_sharpes(self, *, kind: str | None = None) -> list[float]:
        """Backtest Sharpes across the trial population (optionally one kind).

        This is the multiple-testing population: ``len`` is the trial count
        and the variance feeds the deflation. Filter by ``kind`` to deflate a
        strategy only against other strategies, etc.
        """
        return [
            t.backtest_sharpe for t in self.trials()
            if kind is None or t.kind == kind
        ]

    def n_trials(self, *, kind: str | None = None) -> int:
        return len(self.trial_sharpes(kind=kind))

    def deflated_sharpe(
        self,
        returns,
        *,
        kind: str | None = None,
        include_self_sharpe: float | None = None,
        trading_days_per_year: int = 252,
    ) -> float:
        """Globally-deflated Sharpe for a candidate `returns` series.

        Deflates against the WHOLE trial population in the ledger — this is
        the multiple-testing correction that makes continuous research
        honest. ``n_trials`` and the trial-Sharpe variance both come from
        the ledger, not from a single run.

        ``include_self_sharpe`` optionally adds the candidate's own Sharpe to
        the trial population (use when the candidate isn't logged yet, so it
        still counts as one of the trials). With <2 trials the variance is
        undefined, so we fall back to var=0 (DSR collapses to PSR vs 0).
        """
        from quant.evaluation.dsr import (
            deflated_sharpe_ratio,
            estimate_var_sr_from_trials,
        )

        sharpes = self.trial_sharpes(kind=kind)
        if include_self_sharpe is not None:
            sharpes = [*sharpes, float(include_self_sharpe)]
        n_trials = max(1, len(sharpes))
        try:
            var_sr = estimate_var_sr_from_trials(sharpes) if len(sharpes) >= 2 else 0.0
        except ValueError:
            var_sr = 0.0
        return float(deflated_sharpe_ratio(
            returns,
            n_trials=n_trials,
            var_sr_trials_annual=var_sr,
            trading_days_per_year=trading_days_per_year,
        ))

    def summary(self) -> dict[str, Any]:
        """Counts by outcome and kind — for the readiness reporter."""
        views = self.trials()
        by_outcome: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        for v in views:
            by_outcome[v.outcome] = by_outcome.get(v.outcome, 0) + 1
            by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
        decided = sum(1 for v in views if v.outcome in {"promoted", "rejected"})
        return {
            "total_trials": len(views),
            "by_outcome": by_outcome,
            "by_kind": by_kind,
            "reached_decision": decided,
        }


def iter_trial_sharpes(views: Iterable[TrialView]) -> list[float]:
    """Helper: pull the Sharpe population out of a set of trial views."""
    return [v.backtest_sharpe for v in views]
