"""reports.py — markdown report renderers for the agent's emails.

Three audiences, three reports:

- ``render_daily_report``  — sent every weekday after the close. Summarizes
  TODAY'S trading: positions held, entries, stops, equity change.
- ``render_weekly_report`` — sent every Friday after close. Aggregates the
  week's daily runs; surfaces winning/losing names, hit rate, cost drag.
- ``render_monthly_report`` — sent end of month. Same shape as weekly but
  longer window, plus a "recommendations" section if the auto-improver
  has candidates.

Markdown output. The email sender attaches an HTML version too — for now
we use the raw markdown both as text and (via plain pre-wrapping) as HTML.
Markdown-to-HTML conversion is a v2 nicety.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from quant.execution.alpaca_executor import ExecutionReport


def render_daily_report(
    *,
    run_date: date,
    strategy_name: str,
    target_weights: dict[str, float],
    execution_report: ExecutionReport,
    account_equity_after: float | None = None,
    benchmarks: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Render today's report. Returns (subject, markdown_body).

    The subject is short (~70 chars max so it doesn't wrap in inbox lists).

    Parameters
    ----------
    account_equity_after
        End-of-session equity (post-close). When provided alongside
        ``execution_report.account_equity_before`` (which captures the
        pre-trade equity at 09:35 ET), the report computes today's
        portfolio return and shows it next to the benchmarks. Optional
        because some test paths don't have it.
    benchmarks
        ``{"SPY": pct_return, "QQQ": pct_return}`` for the trade day's
        close-to-close move. Either key may be absent if the cache
        couldn't price it; the renderer just omits that row.
    """
    # Count both NEW entries (status=submitted in OTO bracket) AND KEPT
    # entries (carryforward positions where only the stop was re-armed).
    # Both represent positions that are now ON THE BOOKS — the subject
    # line should reflect total active positions, not just new ones.
    n_entries = sum(
        1 for o in execution_report.submitted_orders
        if o.role == "entry"
        and o.status in ("submitted", "skipped_dry_run", "kept")
    )
    n_stops = sum(
        1 for o in execution_report.submitted_orders
        if o.role == "stop_loss" and o.status in ("submitted", "skipped_dry_run")
    )
    n_failed = sum(
        1 for o in execution_report.submitted_orders if o.status == "failed"
    )
    subject = (
        f"quant agent — {run_date.isoformat()} — "
        f"{n_entries} entries, {n_stops} stops"
        + (f", {n_failed} FAILED" if n_failed else "")
    )

    lines: list[str] = []
    lines.append(f"# Quant agent — daily report — {run_date.isoformat()}")
    lines.append("")
    lines.append(f"**Strategy:** `{strategy_name}`  ")
    lines.append(f"**Environment:** `{execution_report.env}`  ")
    lines.append(
        f"**Account equity (pre-trade):** ${execution_report.account_equity_before:,.2f}  "
    )
    if account_equity_after is not None:
        lines.append(
            f"**Account equity (post-close):** ${account_equity_after:,.2f}  "
        )
    lines.append(f"**Dry run:** {execution_report.dry_run}")
    lines.append("")

    # --- Today's performance vs benchmarks ---
    # Compares the portfolio's session return (pre-trade open → post-close)
    # to SPY and QQQ close-to-close on the same trading day. Without this
    # context a +0.4% day looks fine in isolation but is actually -0.6%
    # vs a benchmark that did +1.0%.
    portfolio_ret: float | None = None
    if (
        account_equity_after is not None
        and execution_report.account_equity_before > 0
    ):
        portfolio_ret = (
            account_equity_after / execution_report.account_equity_before - 1.0
        )

    if portfolio_ret is not None or benchmarks:
        lines.append("## Today vs benchmarks")
        lines.append("")
        lines.append("| Book | Return |")
        lines.append("|---|---|")
        if portfolio_ret is not None:
            lines.append(f"| **Portfolio** | **{portfolio_ret:+.2%}** |")
        if benchmarks:
            for sym in ("SPY", "QQQ"):
                if sym in benchmarks:
                    label = (
                        "SPY (S&P 500)" if sym == "SPY"
                        else "QQQ (Nasdaq 100)"
                    )
                    bench_ret = benchmarks[sym]
                    line = f"| {label} | {bench_ret:+.2%} |"
                    if portfolio_ret is not None:
                        delta = portfolio_ret - bench_ret
                        sign = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                        line = (
                            f"| {label} | {bench_ret:+.2%} "
                            f"({sign} {abs(delta):.2%} vs portfolio) |"
                        )
                    lines.append(line)
        if portfolio_ret is None and benchmarks:
            lines.append("")
            lines.append(
                "_(Portfolio return not available — post-close equity wasn't "
                "captured. Benchmarks shown as market context.)_"
            )
        lines.append("")

    # --- Target book ---
    lines.append("## Target weights")
    lines.append("")
    if target_weights:
        lines.append("| Symbol | Weight |")
        lines.append("|---|---|")
        for sym, w in sorted(target_weights.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {sym} | {w:+.2%} |")
    else:
        lines.append("_No target positions today (strategy returned no longs)._")
    lines.append("")

    # --- Pre-trade positions ---
    lines.append("## Positions at start of session")
    lines.append("")
    if execution_report.positions_before:
        lines.append("| Symbol | Qty |")
        lines.append("|---|---|")
        for sym, qty in sorted(execution_report.positions_before.items()):
            lines.append(f"| {sym} | {qty:+,d} |")
    else:
        lines.append("_Account was flat._")
    lines.append("")

    # --- Submitted orders ---
    lines.append("## Orders submitted")
    lines.append("")
    if execution_report.submitted_orders:
        lines.append("| Role | Symbol | Side | Qty | Stop price | Status | Alpaca ID |")
        lines.append("|---|---|---|---|---|---|---|")
        for o in execution_report.submitted_orders:
            stop = f"${o.stop_price:.2f}" if o.stop_price is not None else "—"
            err = f" *(err: {o.error})*" if o.error else ""
            oid = o.alpaca_order_id or "—"
            lines.append(
                f"| {o.role} | {o.symbol} | {o.side} | {o.qty:,d} | "
                f"{stop} | {o.status}{err} | `{oid}` |"
            )
    else:
        lines.append("_No orders were submitted (no rebalance needed)._")
    lines.append("")

    # --- Failures need their own emphasis. Buried in a long table they
    #     can be missed; surface them at the top of the agent's attention. ---
    failures = [o for o in execution_report.submitted_orders if o.status == "failed"]
    if failures:
        lines.append("## ⚠ Failures")
        lines.append("")
        for o in failures:
            lines.append(f"- **{o.symbol}** {o.side} {o.qty}: {o.error}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Generated by `quant.agent.daily_runner`._")

    return subject, "\n".join(lines) + "\n"


def compute_deployment_fidelity(daily_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Deployment + execution-fidelity diagnostics from a week's run records.

    These three numbers exist because of the June 2026 "flat performance"
    incident, where the book silently ran ~25% deployed for a week and
    nothing in the weekly email surfaced it:

      • ensemble gross   — what the strategies asked for (pre vol-target;
        sum of the record's top-level ``target_weights``). A value well
        under 100% means weight is leaking before vol-targeting (e.g.
        the dust-filter bug).
      • submitted gross  — what was actually sent to the broker (post
        vol-target; sum of ``execution_report.target_weights``). The
        true deployment level.
      • entry fidelity   — of the entry orders the executor planned,
        how many were actually placed vs failed. Repeat failers (same
        symbol failing 2+ days) get named: that pattern is how the
        trail-anchor re-entry deadlock stayed invisible for a week.

    Returns {} when ``daily_runs`` is empty. Pure function — safe in tests.
    """
    if not daily_runs:
        return {}

    ens_gross: list[float] = []
    sub_gross: list[float] = []
    n_entries_placed = 0
    n_entries_failed = 0
    fail_days: dict[str, int] = {}
    for run in sorted(daily_runs, key=lambda r: r.get("date", "")):
        tw = run.get("target_weights", {})
        er = run.get("execution_report", {})
        etw = er.get("target_weights", {})
        if tw:
            ens_gross.append(sum(tw.values()))
        if etw:
            sub_gross.append(sum(etw.values()))
        failed_today: set[str] = set()
        for o in er.get("submitted_orders", []):
            if o.get("role") != "entry":
                continue
            if o.get("status") in ("submitted", "kept"):
                n_entries_placed += 1
            elif o.get("status") == "failed":
                n_entries_failed += 1
                failed_today.add(o.get("symbol", "?"))
        for sym in failed_today:
            fail_days[sym] = fail_days.get(sym, 0) + 1

    n_intended = n_entries_placed + n_entries_failed
    out: dict[str, Any] = {
        "ensemble_gross_pct_latest": round(ens_gross[-1] * 100, 1) if ens_gross else None,
        "ensemble_gross_pct_week_avg": (
            round(sum(ens_gross) / len(ens_gross) * 100, 1) if ens_gross else None
        ),
        "submitted_gross_pct_latest": round(sub_gross[-1] * 100, 1) if sub_gross else None,
        "submitted_gross_pct_week_avg": (
            round(sum(sub_gross) / len(sub_gross) * 100, 1) if sub_gross else None
        ),
        "entries_intended_week": n_intended,
        "entries_failed_week": n_entries_failed,
        "entry_fidelity_pct": (
            round(n_entries_placed / n_intended * 100, 1) if n_intended else None
        ),
        # Symbols whose entries failed on 2+ days, worst first, top 10.
        "repeat_entry_failers": dict(
            sorted(
                ((s, n) for s, n in fail_days.items() if n >= 2),
                key=lambda kv: -kv[1],
            )[:10]
        ),
    }
    return out


def _deployment_fidelity_lines(daily_runs: list[dict[str, Any]]) -> list[str]:
    """Markdown section for the weekly email. Empty list if no data."""
    df = compute_deployment_fidelity(daily_runs)
    if not df:
        return []
    lines: list[str] = []
    lines.append("## Deployment & execution fidelity")
    lines.append("")
    lines.append("| Metric | Latest run | Week avg |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Ensemble gross (strategies asked) "
        f"| {df['ensemble_gross_pct_latest']}% "
        f"| {df['ensemble_gross_pct_week_avg']}% |"
    )
    lines.append(
        f"| Submitted gross (sent to broker) "
        f"| {df['submitted_gross_pct_latest']}% "
        f"| {df['submitted_gross_pct_week_avg']}% |"
    )
    if df["entry_fidelity_pct"] is not None:
        lines.append(
            f"| Entry fidelity (placed / planned) "
            f"| {df['entry_fidelity_pct']}% "
            f"({df['entries_failed_week']} of {df['entries_intended_week']} failed) | |"
        )
    lines.append("")
    sub = df.get("submitted_gross_pct_week_avg")
    if sub is not None and sub < 50:
        lines.append(
            f"⚠️ **Under-deployed:** the book averaged {sub}% gross this week "
            "— most of the account sat in cash. Check the dust filter, "
            "vol-target scale, and entry failures below."
        )
        lines.append("")
    failers = df.get("repeat_entry_failers", {})
    if failers:
        named = ", ".join(f"`{s}` ×{n}d" for s, n in failers.items())
        lines.append(
            f"⚠️ **Repeat entry failures (2+ days):** {named}. "
            "The same name failing day after day usually means a stale "
            "stop anchor or a sizing refusal — not market noise."
        )
        lines.append("")
    return lines


def render_weekly_report(
    *,
    week_ending: date,
    daily_runs: list[dict[str, Any]],
    equity_curve: dict[date, float] | None = None,
    notes: str = "",
    benchmarks: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Render a weekly summary.

    ``daily_runs`` is a list of the per-day payloads loaded from
    ``log.load_daily_run``. ``equity_curve`` is optional; if provided
    we add a returns line.

    Parameters
    ----------
    benchmarks
        ``{"SPY": pct, "QQQ": pct}`` close-to-close returns over the
        same window as the equity curve. Surfaced as a side-by-side
        comparison row so the operator can see relative performance
        at a glance.
    """
    subject = f"quant agent — weekly review — week ending {week_ending.isoformat()}"

    lines: list[str] = []
    lines.append(f"# Quant agent — weekly review — week ending {week_ending.isoformat()}")
    lines.append("")
    lines.append(f"**Runs included:** {len(daily_runs)} trading days  ")
    portfolio_ret: float | None = None
    if equity_curve and len(equity_curve) >= 2:
        eq_dates = sorted(equity_curve.keys())
        start_eq = equity_curve[eq_dates[0]]
        end_eq = equity_curve[eq_dates[-1]]
        pnl = end_eq - start_eq
        portfolio_ret = (end_eq / start_eq - 1) if start_eq > 0 else 0.0
        lines.append(f"**Equity:** ${start_eq:,.2f} → ${end_eq:,.2f}  ")
        lines.append(f"**P&L this week:** ${pnl:+,.2f} ({portfolio_ret:+.2%})  ")
    lines.append("")

    # --- Vs benchmarks ---
    # Side-by-side comparison: portfolio return vs SPY (S&P 500 proxy) and
    # QQQ (Nasdaq 100 proxy) over the SAME date window. Includes the
    # relative (out/under)performance vs each benchmark — that's the
    # number that actually tells the operator whether the strategy is
    # adding value vs just riding the market up.
    if benchmarks or portfolio_ret is not None:
        lines.append("## Vs benchmarks (same window)")
        lines.append("")
        lines.append("| Book | Return |")
        lines.append("|---|---|")
        if portfolio_ret is not None:
            lines.append(f"| **Portfolio** | **{portfolio_ret:+.2%}** |")
        if benchmarks:
            for sym in ("SPY", "QQQ"):
                if sym in benchmarks:
                    label = (
                        "SPY (S&P 500)" if sym == "SPY"
                        else "QQQ (Nasdaq 100)"
                    )
                    bench_ret = benchmarks[sym]
                    line = f"| {label} | {bench_ret:+.2%} |"
                    if portfolio_ret is not None:
                        delta = portfolio_ret - bench_ret
                        sign = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                        line = (
                            f"| {label} | {bench_ret:+.2%} "
                            f"({sign} {abs(delta):.2%} vs portfolio) |"
                        )
                    lines.append(line)
        lines.append("")

    # Aggregate stats from the daily runs. Count only NEW broker entries
    # ("submitted") — "kept" rows are carryforward and don't represent
    # new trading activity. This number is "how many new entries did
    # the system make this week", which is the operator-facing metric.
    n_entries = sum(
        1
        for run in daily_runs
        for o in run.get("execution_report", {}).get("submitted_orders", [])
        if o.get("role") == "entry" and o.get("status") == "submitted"
    )
    n_stops = sum(
        1
        for run in daily_runs
        for o in run.get("execution_report", {}).get("submitted_orders", [])
        if o.get("role") == "stop_loss" and o.get("status") == "submitted"
    )
    n_failed = sum(
        1
        for run in daily_runs
        for o in run.get("execution_report", {}).get("submitted_orders", [])
        if o.get("status") == "failed"
    )

    lines.append("## Activity")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Entry orders | {n_entries} |")
    lines.append(f"| Stop-loss orders | {n_stops} |")
    lines.append(f"| Failed orders | {n_failed} |")
    lines.append("")

    lines.extend(_deployment_fidelity_lines(daily_runs))

    # Per-day equity if available.
    if equity_curve:
        lines.append("## Daily equity")
        lines.append("")
        lines.append("| Date | Equity |")
        lines.append("|---|---|")
        for d in sorted(equity_curve.keys()):
            lines.append(f"| {d.isoformat()} | ${equity_curve[d]:,.2f} |")
        lines.append("")

    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(notes)
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Generated by `quant.agent.weekly_review`._")
    return subject, "\n".join(lines) + "\n"


def render_monthly_report(
    *,
    month_ending: date,
    daily_runs: list[dict[str, Any]],
    equity_curve: dict[date, float] | None = None,
    recommendations: list[str] | None = None,
    benchmarks: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Render a monthly summary. Like the weekly but with a Recommendations
    section populated by the auto-improver.

    ``benchmarks`` follows the same shape as in ``render_weekly_report``;
    the comparison covers the full monthly window.
    """
    subject = (
        f"quant agent — monthly review — "
        f"month ending {month_ending.isoformat()}"
    )

    # Reuse the weekly renderer for the core stats, then bolt on recs.
    _, body = render_weekly_report(
        week_ending=month_ending,
        daily_runs=daily_runs,
        equity_curve=equity_curve,
        benchmarks=benchmarks,
    )
    body = body.replace(
        "quant agent — weekly review",
        "quant agent — monthly review",
    ).replace(
        "_Generated by `quant.agent.weekly_review`._",
        "",
    )

    extra: list[str] = []
    if recommendations:
        extra.append("## Recommendations")
        extra.append("")
        for r in recommendations:
            extra.append(f"- {r}")
        extra.append("")
    extra.append("---")
    extra.append("")
    extra.append("_Generated by `quant.agent.monthly_review`._")

    return subject, body.rstrip() + "\n\n" + "\n".join(extra) + "\n"
