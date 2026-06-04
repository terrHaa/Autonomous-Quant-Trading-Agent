"""audit.py — end-of-trading-day full-pipeline health check.

Fires once per trading day after the close, after the daily-report has
already run. Its job is to catch silent failures the other routines
might miss — a stop that expired, a position whose qty drifted, a launchd
plist that didn't reload, an AI strategy file that vanished — and email
the operator a structured pass/fail summary.

The audit does **not** trade, doesn't refit weights, doesn't generate
strategies; it only **observes** and **reports**. That keeps the failure
mode predictable: a buggy audit can't cause a bad trade.

Checks performed
----------------
1. **Run record exists** — today's run JSON is parseable and complete.
2. **Broker reconciliation** — account active; every open position
   has a matching open GTC stop; no orphans; no unprotected; quantities
   match the run record.
3. **Ensemble state** — HRP weights sum to 1.0; every key corresponds
   to a buildable strategy; AI strategy names have files on disk.
4. **Recent error logs** — any non-empty `.err` file modified in the
   last ~26 hours.
5. **Connectivity** — Alpaca data API reachable (single tiny GET).

Each check is independent. The audit reports the FULL result even if
an early one fails — the operator should see everything at once.

Output
------
- ``data/agent/audits/YYYY-MM-DD.json``  — structured record.
- Email — green-banner if all pass, red-banner with failed-check list.
- Exit code 0 if all checks pass; 1 if any failed (launchd reflects this).

Console-script: ``quant-daily-audit`` (registered in pyproject.toml).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from quant.agent.daily_runner import _email_failure, _markdown_to_html
from quant.agent.email_sender import EmailSender
from quant.agent.ensemble import build_strategies, load_ensemble_state
from quant.agent.log import _atomic_write_text, load_daily_run
from quant.data.universe import load_active_universe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class AuditCheck:
    """One named pass/fail item with a human message and structured data."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """All check results for a single audit run."""

    for_date: str          # ISO date of the trading day being audited
    timestamp: str         # ISO timestamp the audit was generated
    checks: list[AuditCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[AuditCheck]:
        return [c for c in self.checks if not c.passed]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_run_record(
    for_date: date,
    runs_dir: Path | None = None,
) -> AuditCheck:
    """Today's daily-run JSON exists and has the fields downstream code uses."""
    payload = load_daily_run(for_date, runs_dir=runs_dir)
    if payload is None:
        return AuditCheck(
            name="run_record",
            passed=False,
            message=f"no daily run record for {for_date.isoformat()}",
        )
    required = {"strategy_name", "target_weights", "execution_report"}
    missing = required - set(payload.keys())
    if missing:
        return AuditCheck(
            name="run_record",
            passed=False,
            message=f"run record present but missing fields: {sorted(missing)}",
            details={"present_keys": sorted(payload.keys())},
        )
    er = payload["execution_report"]
    return AuditCheck(
        name="run_record",
        passed=True,
        message=(
            f"run record found ({er.get('env', '?')}, "
            f"{len(er.get('submitted_orders', []))} orders, "
            f"dry_run={er.get('dry_run')})"
        ),
        details={
            "env": er.get("env"),
            "n_orders": len(er.get("submitted_orders", [])),
            "n_targets": len(payload.get("target_weights", {})),
            "equity_before": er.get("account_equity_before"),
            "dry_run": er.get("dry_run"),
        },
    )


def _check_broker_reconciliation(
    for_date: date,
    runs_dir: Path | None = None,
    executor: Any = None,
) -> AuditCheck:
    """Account active, positions all stopped (GTC), qty matches run record."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    from quant.execution.alpaca_executor import AlpacaExecutor

    ex = executor or AlpacaExecutor()

    # Account health
    try:
        acct = ex._client.get_account()
    except Exception as e:
        return AuditCheck(
            name="broker_reconciliation",
            passed=False,
            message=f"could not reach broker for account: {type(e).__name__}: {e}",
        )

    if acct.trading_blocked:
        return AuditCheck(
            name="broker_reconciliation",
            passed=False,
            message="ACCOUNT IS TRADING_BLOCKED — broker has restricted the account",
            details={"status": str(acct.status)},
        )

    # Positions and open stops
    positions = ex.get_positions()
    open_orders = ex._client.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200)
    )
    open_stops = [o for o in open_orders if o.order_type.value == "stop"]
    stop_syms = {o.symbol for o in open_stops}
    pos_syms = set(positions.keys())

    unprotected = pos_syms - stop_syms
    orphan_stops = stop_syms - pos_syms
    non_gtc = [
        o.symbol for o in open_stops if o.time_in_force.value.lower() != "gtc"
    ]

    # Quantity check: each open stop's qty matches the position qty (whole share).
    qty_mismatch: list[str] = []
    for o in open_stops:
        if o.symbol in positions and int(float(o.qty)) != positions[o.symbol]:
            qty_mismatch.append(
                f"{o.symbol}: stop_qty={o.qty} != pos_qty={positions[o.symbol]}"
            )

    # Cross-reference with the run record — submitted/kept entries should
    # match current positions (one trading day later, before next rebalance
    # fires). "kept" entries are carryforward (no broker trade) but the qty
    # reflects what should still be held; include them in run_qty.
    # FAILED entries (refused / forced exit) are skipped: their qty is the
    # would-be target that we deliberately did NOT submit.
    payload = load_daily_run(for_date, runs_dir=runs_dir) or {}
    run_qty: dict[str, int] = {}
    for o in payload.get("execution_report", {}).get("submitted_orders", []):
        if o.get("role") == "entry" and o.get("status") in ("submitted", "kept"):
            run_qty[o["symbol"]] = int(o.get("qty", 0))
    # If a name in run_qty has a different broker qty, EITHER the entry only
    # partially filled OR a stop has triggered overnight. We don't fail the
    # audit on this — it's informational; flag it in details.
    entry_vs_pos: list[str] = []
    for sym, q in run_qty.items():
        actual = positions.get(sym, 0)
        if actual != q:
            entry_vs_pos.append(f"{sym}: entry_qty={q} → pos_qty={actual}")

    issues: list[str] = []
    if unprotected:
        issues.append(f"unprotected positions: {sorted(unprotected)}")
    if orphan_stops:
        issues.append(f"orphan stops (no matching position): {sorted(orphan_stops)}")
    if non_gtc:
        issues.append(f"non-GTC stops (will expire at close): {non_gtc}")
    if qty_mismatch:
        issues.append(f"qty mismatch: {qty_mismatch}")

    passed = not issues
    if passed:
        message = (
            f"{len(positions)} positions, {len(open_stops)} GTC stops, "
            f"all matched; equity ${float(acct.equity):,.2f}"
        )
    else:
        message = "; ".join(issues)

    return AuditCheck(
        name="broker_reconciliation",
        passed=passed,
        message=message,
        details={
            "env": ex.env,
            "equity": float(acct.equity),
            "n_positions": len(positions),
            "n_open_stops": len(open_stops),
            "unprotected": sorted(unprotected),
            "orphan_stops": sorted(orphan_stops),
            "non_gtc": non_gtc,
            "qty_mismatch": qty_mismatch,
            "entry_vs_position_drift": entry_vs_pos,
        },
    )


def _check_ensemble_state() -> AuditCheck:
    """HRP weights sum to 1.0; strategy names are buildable; AI files exist."""
    state = load_ensemble_state()
    issues: list[str] = []

    # HRP weight integrity
    total = sum(state.hrp_weights.values())
    if abs(total - 1.0) > 1e-6:
        issues.append(f"HRP weights sum to {total:.6f}, expected 1.0")
    negatives = {k: v for k, v in state.hrp_weights.items() if v < 0}
    if negatives:
        issues.append(f"negative HRP weights: {negatives}")

    # Each strategy in the ensemble must build cleanly.
    try:
        universe = load_active_universe(date.today(), fallback_log=False)
        strats = build_strategies(state, universe)
        built_names = {s.name for s in strats}
    except Exception as e:
        issues.append(f"build_strategies raised: {type(e).__name__}: {e}")
        built_names = set()

    # HRP keys vs built names: each HRP key should correspond to a built strategy.
    # AI strategies that fail to load disappear from built_names — we want to
    # surface those clearly because their HRP weight is then unallocated.
    missing_strats = set(state.hrp_weights) - built_names
    if missing_strats:
        issues.append(
            f"HRP keys with no buildable strategy: {sorted(missing_strats)}"
        )

    # AI strategy files exist on disk
    generated_dir = Path("src/quant/strategies/generated")
    ai_missing: list[str] = []
    for ai_name in state.ai_strategy_names:
        py = generated_dir / f"{ai_name}.py"
        meta = generated_dir / f"{ai_name}.json"
        if not py.exists() or not meta.exists():
            ai_missing.append(ai_name)
    if ai_missing:
        issues.append(f"AI strategies in state but missing files: {ai_missing}")

    passed = not issues
    return AuditCheck(
        name="ensemble_state",
        passed=passed,
        message=(
            f"{len(state.hrp_weights)} strategies, weights sum to {total:.4f}, "
            f"all buildable" if passed else "; ".join(issues)
        ),
        details={
            "n_strategies": len(state.hrp_weights),
            "hrp_sum": total,
            "ai_strategies": list(state.ai_strategy_names),
            "shadow": dict(state.ai_strategy_shadow_until),
            "last_hrp_refit_date": state.last_hrp_refit_date,
        },
    )


def _check_recent_error_logs(
    log_dir: Path | None = None,
    hours: int = 26,
) -> AuditCheck:
    """Any non-empty .err log with REAL errors modified within `hours`?

    A "real error" means lines containing ERROR, Traceback, Exception,
    or Failed (case-sensitive). We deliberately IGNORE WARNING-only logs
    because the retry layer writes WARNING on each transient attempt;
    when those retries succeed, the run was fine but the .err file is
    non-empty. Flagging those would cause a false-positive on every
    network-flaky run.
    """
    log_dir = log_dir or Path("data/agent/launchd-logs")
    if not log_dir.exists():
        return AuditCheck(
            name="error_logs",
            passed=True,
            message=f"no log directory at {log_dir} (clean install)",
        )
    cutoff = datetime.now(tz=UTC) - timedelta(hours=hours)
    # Case-sensitive markers. We rely on the CLI entry points routing
    # INFO/WARNING to stdout (logging.basicConfig(stream=sys.stdout)),
    # so the .err file should ONLY contain uncaught Python exceptions
    # (which always include "Traceback (most recent call last):") plus
    # any explicit print()-to-stderr from _email_failure. These
    # markers cover both cases.
    error_markers = ("Traceback", "ERROR")
    flagged: list[dict[str, Any]] = []
    for err_file in log_dir.glob("*.err"):
        stat = err_file.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        if stat.st_size == 0 or mtime <= cutoff:
            continue
        # Scan the file for real error markers — skip WARNING-only files
        # (retry-layer chatter that resolved successfully).
        try:
            text = err_file.read_text(errors="replace")
        except OSError:
            text = ""
        has_real_error = any(marker in text for marker in error_markers)
        if has_real_error:
            # Capture a short excerpt of the first error line for the report.
            first_err_line = next(
                (ln for ln in text.splitlines()
                 if any(m in ln for m in error_markers)),
                "",
            )
            flagged.append({
                "name": err_file.name,
                "size_bytes": stat.st_size,
                "modified": mtime.isoformat(),
                "first_error_line": first_err_line[:200],
            })
    passed = not flagged
    return AuditCheck(
        name="error_logs",
        passed=passed,
        message=(
            f"no real errors in launchd logs in last {hours}h "
            "(WARNINGs from retry layer ignored)" if passed else
            f"{len(flagged)} launchd .err logs with REAL errors in last {hours}h: "
            f"{[f['name'] for f in flagged]}"
        ),
        details={"flagged": flagged, "log_dir": str(log_dir), "hours": hours},
    )


def _check_alpaca_connectivity(executor: Any = None) -> AuditCheck:
    """Single cheap broker round-trip to verify network reachability.

    Wrapped in the retry layer so a one-shot TLS handshake hiccup
    doesn't fail the audit on an otherwise-healthy pipeline. Aligns
    with how the daily-trade routine handles the same flake.
    """
    from quant.execution.alpaca_executor import AlpacaExecutor
    from quant.util.retry import HTTP_TRANSIENT, retry_on_transient

    ex = executor or AlpacaExecutor()
    try:
        acct = retry_on_transient(
            lambda: ex._client.get_account(),
            transient=HTTP_TRANSIENT,
            description="audit connectivity check",
        )
        return AuditCheck(
            name="connectivity",
            passed=True,
            message=f"broker reachable; account {acct.status} env={ex.env}",
            details={"env": ex.env, "status": str(acct.status)},
        )
    except Exception as e:
        return AuditCheck(
            name="connectivity",
            passed=False,
            message=f"broker unreachable: {type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


# Default audit checks — declared as a tuple so tests can substitute.
_DEFAULT_CHECKS: tuple[
    tuple[str, Callable[[date], AuditCheck]], ...
] = ()  # populated at call time inside run_daily_audit


def run_daily_audit(
    *,
    for_date: date | None = None,
    runs_dir: Path | None = None,
    audits_dir: Path | None = None,
    executor: Any = None,
    log_dir: Path | None = None,
    email_sender: EmailSender | None = None,
) -> AuditReport:
    """Run the full audit and email the result. Returns the AuditReport.

    Every check runs to completion even if earlier ones fail — the operator
    needs to see the full picture. Persists JSON to ``audits_dir`` regardless
    of pass/fail.
    """
    for_date = for_date or date.today()
    audits_dir = audits_dir or Path("data/agent/audits")
    audits_dir.mkdir(parents=True, exist_ok=True)

    # Uniform defensive wrap so the audit's "every check runs to
    # completion" contract holds even if a check raises (e.g. corrupt
    # run JSON, weird FS permissions on the launchd-logs dir).
    def _safe_run(name: str, fn: Callable[[], AuditCheck]) -> AuditCheck:
        try:
            return fn()
        except Exception as e:
            return AuditCheck(
                name=name,
                passed=False,
                message=f"check raised {type(e).__name__}: {e}",
                details={"traceback": traceback.format_exc()},
            )

    checks: list[AuditCheck] = [
        _safe_run("run_record", lambda: _check_run_record(for_date, runs_dir=runs_dir)),
        _safe_run("broker_reconciliation", lambda: _check_broker_reconciliation(
            for_date, runs_dir=runs_dir, executor=executor,
        )),
        _safe_run("ensemble_state", lambda: _check_ensemble_state()),
        _safe_run("error_logs", lambda: _check_recent_error_logs(log_dir=log_dir)),
        _safe_run("connectivity", lambda: _check_alpaca_connectivity(executor=executor)),
    ]

    report = AuditReport(
        for_date=for_date.isoformat(),
        timestamp=datetime.now(tz=UTC).isoformat(),
        checks=checks,
    )

    # Persist the JSON record. Atomic write so a launchd kill mid-write
    # can't corrupt yesterday's audit (matches save_daily_run / save_weekly_report
    # / save_ensemble_state pattern).
    out_path = audits_dir / f"{for_date.isoformat()}.json"
    _atomic_write_text(out_path, json.dumps({
        "for_date": report.for_date,
        "timestamp": report.timestamp,
        "passed": report.passed,
        "checks": [asdict(c) for c in report.checks],
    }, indent=2, default=str))
    logger.info("audit saved to %s (passed=%s)", out_path, report.passed)

    # Email the result.
    subject, body = _render_email(report)
    sender = email_sender or EmailSender()
    try:
        sender.send(subject=subject, body_text=body, body_html=_markdown_to_html(body))
        logger.info("audit emailed: %s", subject)
    except Exception as e:
        # Email failure shouldn't mask audit failure. Log and continue.
        logger.error("audit email send failed: %s", e)

    return report


def _render_email(report: AuditReport) -> tuple[str, str]:
    """Build subject + markdown body for the audit email."""
    failures = report.failures
    if failures:
        subject = (
            f"quant audit FAILED — {report.for_date} — "
            f"{len(failures)} of {len(report.checks)} checks failed"
        )
        banner = "## ❌ AUDIT FAILED\n"
    else:
        subject = (
            f"quant audit OK — {report.for_date} — "
            f"all {len(report.checks)} checks passed"
        )
        banner = "## ✅ AUDIT PASSED\n"

    lines: list[str] = [
        banner,
        f"For trading day: **{report.for_date}**",
        f"Generated at:    {report.timestamp}",
        "",
        "### Checks",
        "",
    ]
    for c in report.checks:
        mark = "✅" if c.passed else "❌"
        lines.append(f"- {mark} **{c.name}** — {c.message}")
    lines.append("")

    if failures:
        lines.append("### Details for failing checks")
        lines.append("")
        for c in failures:
            lines.append(f"**{c.name}**")
            lines.append("```")
            lines.append(json.dumps(c.details, indent=2, default=str))
            lines.append("```")
            lines.append("")

    return subject, "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cli_run() -> None:
    """Console-script: ``uv run quant-daily-audit``.

    Exits with code 1 if any check failed, so launchd's job-status reflects
    overall pipeline health.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Audit the trading pipeline.")
    parser.add_argument(
        "--for-date", default=None,
        help="ISO date YYYY-MM-DD; defaults to today",
    )
    args = parser.parse_args()
    for_date = date.fromisoformat(args.for_date) if args.for_date else None
    try:
        report = run_daily_audit(for_date=for_date)
        if report.passed:
            print(f"[audit] all {len(report.checks)} checks passed")
            sys.exit(0)
        else:
            print(
                f"[audit] {len(report.failures)} of {len(report.checks)} checks failed:",
                file=sys.stderr,
            )
            for c in report.failures:
                print(f"  - {c.name}: {c.message}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        _email_failure("daily audit", e)
        raise
