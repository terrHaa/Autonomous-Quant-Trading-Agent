"""reliability.py — Pillar 5: is the live system capturing the edge?

A backtest's Sharpe is a promise. This module measures how much of it
survives contact with the live broker and the operator's flaky China
network. Two views:

  compute_implementation_shortfall(daily_runs)
    The alpha leak between intent and execution: what fraction of planned
    entries actually got placed, how much intended exposure was lost to
    failures, and WHY entries failed (categorized). Recurrent failure
    buckets are the signal — e.g. the stale-stop-anchor deadlock showed
    up here as a wall of "insufficient qty" before it was diagnosed.

  compute_reliability_scorecard(runs_dir, logs_dir, audits_dir)
    Operational health from the artifacts the agent already writes:
    missed trading days, SMTP delivery success, audit pass rate, kill-
    switch trips. The June 2026 incidents (battery-sleep missed trade,
    SMTP outages, the never-generated sector map) were all silent
    degradation; this turns them into a standing number the monthly
    review reads instead of discovering them ad hoc.

Price slippage (fill vs signal price) is deliberately NOT here: the run
records don't persist fill prices yet. Capturing fills is a separate
data task, flagged so the consumer knows shortfall is measured on
intent-fidelity, not realized slippage.

Everything is best-effort and never raises — a telemetry module must not
be able to break the review that consumes it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _classify_failure(error: str) -> str:
    """Bucket an order error string into a recurring failure category."""
    e = (error or "").lower()
    if "insufficient qty" in e or "available" in e:
        return "insufficient_qty (stale anchor / sizing)"
    if "already" in e and "filled" in e:
        return "race: already filled"
    if "wash" in e:
        return "wash-trade block"
    if "selective cancel" in e:
        return "stop-cancel race"
    if "buying power" in e or "insufficient buying" in e:
        return "insufficient buying power"
    if not e:
        return "unspecified"
    return "other"


def compute_implementation_shortfall(daily_runs: list[dict]) -> dict[str, Any]:
    """Intent-vs-execution fidelity over a set of daily run records."""
    if not daily_runs:
        return {}

    intended = placed = failed = 0
    leaked_weights: list[float] = []
    fail_causes: Counter[str] = Counter()

    for run in daily_runs:
        er = run.get("execution_report", {})
        tw = run.get("target_weights", {})
        failed_syms_today: set[str] = set()
        for o in er.get("submitted_orders", []):
            if o.get("role") != "entry":
                continue
            status = o.get("status")
            if status in ("submitted", "kept"):
                placed += 1
                intended += 1
            elif status == "failed":
                failed += 1
                intended += 1
                fail_causes[_classify_failure(o.get("error", ""))] += 1
                sym = o.get("symbol")
                if sym:
                    failed_syms_today.add(sym)
        # Exposure that was supposed to go on but didn't (failed entries).
        leaked = sum(
            w for s, w in tw.items() if s in failed_syms_today and w > 0
        )
        leaked_weights.append(leaked)

    fidelity = round(placed / intended * 100, 1) if intended else None
    return {
        "entries_intended": intended,
        "entries_placed": placed,
        "entries_failed": failed,
        "entry_fidelity_pct": fidelity,
        "avg_leaked_exposure_pct": (
            round(sum(leaked_weights) / len(leaked_weights) * 100, 2)
            if leaked_weights else 0.0
        ),
        "failure_causes": dict(fail_causes.most_common()),
        "note": "slippage not measured (fill prices not persisted)",
    }


# US market full-closure holidays. Ground truth is the bars cache (a bar
# exists ⟺ the market was open); this hardcoded set is only the offline
# fallback when no ``trading_days`` is passed, so holidays aren't
# miscounted as missed trades. Extend per year.
_US_MARKET_HOLIDAYS: frozenset[date] = frozenset({
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
})


def _is_trading_day(d: date) -> bool:
    """Weekday and not a known US market holiday (offline fallback)."""
    return d.weekday() < 5 and d not in _US_MARKET_HOLIDAYS


def compute_reliability_scorecard(
    *,
    runs_dir: Path | None = None,
    logs_dir: Path | None = None,
    audits_dir: Path | None = None,
    days: int = 35,
    trading_days: set[date] | None = None,
) -> dict[str, Any]:
    """Operational-health rollup over the last ``days`` calendar days.

    ``trading_days`` (e.g. derived from the bars cache index) is the
    authoritative set of market-open days; when given it overrides the
    weekday/holiday heuristic so holidays are never miscounted as missed.
    """
    runs_dir = runs_dir or Path("data/agent/runs")
    logs_dir = logs_dir or Path("data/agent/launchd-logs")
    audits_dir = audits_dir or Path("data/agent/audits")
    today = date.today()
    cutoff = today - timedelta(days=days)

    # --- trade completion: expected trading days vs run records present ---
    present: set[date] = set()
    if runs_dir.exists():
        for f in runs_dir.glob("*.json"):
            try:
                d = date.fromisoformat(f.stem)
            except ValueError:
                continue
            if cutoff <= d <= today:
                present.add(d)
    if trading_days is not None:
        expected = {d for d in trading_days if cutoff <= d < today}
    else:
        expected = {
            cutoff + timedelta(days=i)
            for i in range((today - cutoff).days + 1)
            if _is_trading_day(cutoff + timedelta(days=i))
        }
    # Don't count today (may not have traded yet) or future.
    expected = {d for d in expected if d < today}
    missed = sorted(d for d in expected if d not in present)

    # --- SMTP delivery: count exhausted-retry events in the logs ---
    smtp_failures = 0
    if logs_dir.exists():
        for f in logs_dir.glob("*.out"):
            try:
                txt = f.read_text(errors="ignore")
            except OSError:
                continue
            mtime = datetime.fromtimestamp(f.stat().st_mtime).date()
            if mtime >= cutoff:
                smtp_failures += txt.count("all 4 retries exhausted")

    # --- audit pass rate ---
    audits_total = audits_passed = 0
    if audits_dir.exists():
        for f in audits_dir.glob("*.json"):
            try:
                d = date.fromisoformat(f.stem)
            except ValueError:
                continue
            if d < cutoff:
                continue
            try:
                payload = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            audits_total += 1
            if payload.get("passed"):
                audits_passed += 1

    return {
        "window_days": days,
        "trading_days_expected": len(expected),
        "trading_days_traded": len(expected) - len(missed),
        "missed_trade_days": [d.isoformat() for d in missed],
        "trade_completion_pct": (
            round((len(expected) - len(missed)) / len(expected) * 100, 1)
            if expected else None
        ),
        "smtp_delivery_failures": smtp_failures,
        "audit_pass_rate_pct": (
            round(audits_passed / audits_total * 100, 1) if audits_total else None
        ),
        "audits_run": audits_total,
    }


def render_reliability_md(
    shortfall: dict[str, Any], scorecard: dict[str, Any]
) -> str:
    """Markdown Pillar-5 section for the monthly email."""
    lines = ["## ⚙️ Pillar 5 — Pipeline effectiveness & reliability", ""]
    if shortfall:
        lines.append(
            f"**Implementation shortfall:** entry fidelity "
            f"{shortfall.get('entry_fidelity_pct')}% "
            f"({shortfall.get('entries_failed')} of "
            f"{shortfall.get('entries_intended')} entries failed); "
            f"avg leaked exposure {shortfall.get('avg_leaked_exposure_pct')}%."
        )
        causes = shortfall.get("failure_causes", {})
        if causes:
            top = "; ".join(f"{k} ×{v}" for k, v in list(causes.items())[:5])
            lines.append(f"- Failure causes: {top}")
        lines.append("")
    if scorecard:
        lines.append(
            f"**Operational reliability ({scorecard.get('window_days')}d):** "
            f"trade completion {scorecard.get('trade_completion_pct')}% "
            f"({scorecard.get('trading_days_traded')}/"
            f"{scorecard.get('trading_days_expected')} days), "
            f"audit pass rate {scorecard.get('audit_pass_rate_pct')}%, "
            f"SMTP delivery failures {scorecard.get('smtp_delivery_failures')}."
        )
        missed = scorecard.get("missed_trade_days", [])
        if missed:
            lines.append(f"- ⚠ Missed trade days: {', '.join(missed)}")
    return "\n".join(lines)
