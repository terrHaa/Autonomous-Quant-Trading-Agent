"""position_gate.py — refuse to trade when the broker's position feed lies.

T-incident 2026-07-07: Alpaca's paper engine desynced its positions
ledger from its cash ledger — shares the account truly owned vanished
from ``get_all_positions()`` and several longs showed as shorts. Every
downstream system trusted that feed: stops sized to TRUE holdings fired
against the broken ledger and were booked as naked sells (phantom
shorts), close-outs sold ghosts, and the dashboard showed -20.6% on a
book that was actually flat. The failure wasn't our logic — it was that
we ACTED on a lying input.

This gate is the defense, same philosophy as the bar-freshness guard
(T3.18) for price data: verify the input before acting on it.

  reconcile_positions(): rebuild what the positions SHOULD be from the
  last run's snapshot + every fill since (the broker's own fill stream —
  fills settle disputes; the positions ledger is the derived view that
  drifts). Compare to the live ledger. Any per-symbol qty mismatch is a
  desync.

  gate_positions(): called by run_daily_trade before planning. On
  mismatch → REFUSE to trade (fail-closed: trading on poison mints real
  damage; skipping a day costs ~nothing). launchd KeepAlive keeps
  retrying through the trade window, so if the desync heals intraday the
  trade proceeds on a later attempt automatically. On infrastructure
  error (can't fetch orders) → fail-open with a warning, consistent with
  the kill-switch check: a network blip must not block trading, and
  without fills we can't compute a verdict anyway.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionReconciliation:
    """Outcome of a fills-vs-ledger position reconciliation."""

    ok: bool
    reason: str
    mismatches: dict[str, tuple[int, int]] = field(default_factory=dict)
    # {symbol: (reconstructed_qty, ledger_qty)}
    checked: int = 0

    def summary(self) -> str:
        if self.ok:
            return f"position gate: OK — {self.reason} ({self.checked} names)"
        rows = ", ".join(
            f"{s}: expect {r} ledger {lq}" for s, (r, lq) in
            sorted(self.mismatches.items())
        )
        return f"position gate: DESYNC — {self.reason}: {rows}"


def reconcile_positions(
    executor: Any,
    baseline_positions: dict[str, int],
    baseline_ts: datetime,
) -> PositionReconciliation:
    """Rebuild expected positions from baseline + fills; compare to ledger.

    ``baseline_positions``/``baseline_ts`` come from the previous run
    record (its ``positions_before`` snapshot and timestamp). ALL fills
    since — agent and manual alike — are replayed on top; the fill stream
    is the ground truth the positions ledger is supposed to derive from.
    """
    from alpaca.trading.enums import QueryOrderStatus  # noqa: PLC0415
    from alpaca.trading.requests import GetOrdersRequest  # noqa: PLC0415

    recon: dict[str, int] = {s: int(q) for s, q in baseline_positions.items()}
    orders = executor._client.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=baseline_ts - timedelta(seconds=1),
        limit=500,
    ))
    n_fills = 0
    for o in orders:
        filled = float(getattr(o, "filled_qty", 0) or 0)
        if filled <= 0:
            continue
        n_fills += 1
        q = int(filled)
        sym = o.symbol
        recon[sym] = recon.get(sym, 0) + (
            q if str(o.side).upper().endswith("BUY") else -q
        )
    recon = {s: q for s, q in recon.items() if q != 0}

    ledger = executor.get_positions()
    ledger = {s: q for s, q in ledger.items() if q != 0}

    mismatches = {
        s: (recon.get(s, 0), ledger.get(s, 0))
        for s in set(recon) | set(ledger)
        if recon.get(s, 0) != ledger.get(s, 0)
    }
    if mismatches:
        return PositionReconciliation(
            ok=False,
            reason=f"{len(mismatches)} of {len(set(recon) | set(ledger))} "
                   f"names disagree (baseline+{n_fills} fills vs ledger)",
            mismatches=mismatches,
            checked=len(set(recon) | set(ledger)),
        )
    return PositionReconciliation(
        ok=True,
        reason=f"ledger matches baseline+{n_fills} fills",
        checked=len(ledger),
    )


# Operator-verified baseline override. After a desync heals BY FIAT
# (Alpaca rewrites the ledger instead of replaying fills), reconstruction
# from any pre-heal run record can never reconcile — the fix is a human
# (or the audit) verifying the healed state once and pinning it here as
# the new baseline: {"timestamp": iso, "positions": {sym: qty}}. The gate
# prefers this file over the previous run record when it's newer.
_BASELINE_OVERRIDE = Path("data/agent/position_gate_baseline.json")


def pin_verified_baseline(positions: dict[str, int], path: Path | None = None) -> Path:
    """Persist an operator-verified position baseline (post-desync reset)."""
    p = path or _BASELINE_OVERRIDE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "positions": {s: int(q) for s, q in positions.items()},
    }, indent=2))
    return p


def gate_positions(
    executor: Any,
    prev_run: dict[str, Any] | None,
    override_path: Path | None = None,
) -> PositionReconciliation:
    """Trade-or-refuse verdict for run_daily_trade. Never raises.

    Baseline preference: the pinned operator-verified baseline (if newer
    than the previous run record), else the previous run record's
    snapshot, else pass (fresh install — nothing to reconcile against).
    Infrastructure errors fail OPEN with a warning; only a computed
    mismatch fails closed.
    """
    baseline: dict[str, Any] | None = None
    ts_raw: str | None = None

    er = (prev_run or {}).get("execution_report", {})
    baseline, ts_raw = er.get("positions_before"), er.get("timestamp")

    ov = override_path or _BASELINE_OVERRIDE
    try:
        if ov.exists():
            data = json.loads(ov.read_text())
            if not ts_raw or str(data.get("timestamp", "")) > str(ts_raw):
                baseline = data.get("positions", {})
                ts_raw = data.get("timestamp")
    except (OSError, ValueError) as e:
        logger.warning("position gate: bad override file (%s) — ignored", e)

    if baseline is None or not ts_raw:
        return PositionReconciliation(
            ok=True, reason="no baseline available — nothing to reconcile",
        )
    try:
        ts = datetime.fromisoformat(str(ts_raw))
        recon = reconcile_positions(executor, baseline, ts)
    except Exception as e:
        logger.warning(
            "position gate: reconciliation errored (%s: %s) — failing OPEN",
            type(e).__name__, e,
        )
        return PositionReconciliation(
            ok=True, reason=f"gate errored ({type(e).__name__}) — failed open",
        )
    return _classify_severity(recon)


# A halt is only justified for the DANGEROUS signature — not any mismatch.
# T-incident 2026-07-15: the gate blocked ALL trading for days over a
# 2-of-37 stray-share drift (GLW 2, WDC 1 — leftovers the reconstruction
# missed exiting). Fill-reconstruction ALWAYS drifts by a few shares over
# time (pagination limits, partial fills, corporate actions), so a
# fail-closed-on-any-mismatch gate is guaranteed to eventually halt
# everything. That did more damage than the July-7 desync it guards
# against. Halt only when the mismatch looks like that real event:
#   • a PHANTOM SHORT — the ledger shows a name short that we don't hold
#     short (the dangerous case that mints naked sells), OR
#   • a WHOLESALE desync — a large FRACTION of the book disagrees.
# Everything else is minor drift: log it and TRADE; the daily audit's
# reconciliation is the backstop that surfaces stragglers next morning.
_MATERIAL_FRACTION = 0.34


def _classify_severity(recon: PositionReconciliation) -> PositionReconciliation:
    if recon.ok or not recon.mismatches:
        return recon
    mm = recon.mismatches
    phantom_shorts = {
        s: (r, lq) for s, (r, lq) in mm.items() if lq < 0 <= r
    }
    frac = len(mm) / recon.checked if recon.checked else 1.0
    if phantom_shorts:
        return PositionReconciliation(
            ok=False,
            reason=f"phantom short(s) in ledger: {sorted(phantom_shorts)}",
            mismatches=mm, checked=recon.checked,
        )
    if frac > _MATERIAL_FRACTION:
        return PositionReconciliation(
            ok=False,
            reason=f"wholesale desync — {len(mm)}/{recon.checked} names "
                   f"({frac:.0%}) disagree",
            mismatches=mm, checked=recon.checked,
        )
    # Minor drift — trade anyway; the daily audit will surface stragglers.
    logger.warning(
        "position gate: minor drift (%d/%d names: %s) — trading; "
        "daily audit will reconcile.",
        len(mm), recon.checked,
        ", ".join(f"{s}(exp {r}/led {lq})" for s, (r, lq) in sorted(mm.items())),
    )
    return PositionReconciliation(
        ok=True,
        reason=f"minor drift on {len(mm)}/{recon.checked} names — traded",
        mismatches=mm, checked=recon.checked,
    )
