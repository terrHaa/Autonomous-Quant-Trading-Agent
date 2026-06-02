"""weekly_review.py — Friday-after-close summary email + HRP refit.

Two jobs in one weekly run:

1. **Aggregate**: load the week's daily run JSONs, build a markdown
   summary of activity (orders, equity progression, fills).

2. **Refit HRP weights**: backtest each strategy on the last ~252 days,
   compute new HRP weights across them, persist to the ensemble state.
   This is the agent's "self-improve as you go" loop for the
   strategy-mix layer. Per-strategy parameter changes still happen
   monthly (in monthly_review).

If the refit fails or produces degenerate output, the previous weights
remain in effect — fail-safe behavior. The email reports either way.

Console-script: ``quant-weekly-review`` (registered in pyproject.toml).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from quant.agent.daily_runner import _email_failure, _markdown_to_html
from quant.agent.email_sender import EmailSender
from quant.agent.ensemble import (
    load_ensemble_state,
    refit_hrp_weights,
    save_ensemble_state,
)
from quant.agent.log import (
    DEFAULT_RUNS_DIR,
    load_daily_run,
    load_recent_weekly_reports,
    save_weekly_report,
)
from quant.agent.reports import render_weekly_report
from quant.config import Config, load_config
from quant.data.alpaca_client import AlpacaDataClient
from quant.data.cache import BarsCache
from quant.data.universe import load_top50_snapshot
from quant.util.equity_stats import equity_series_stats, top_movers

logger = logging.getLogger(__name__)


# Window for the HRP refit's per-strategy backtest. 252 trading days
# = ~1 year. Long enough for stable correlation estimates, short enough
# to weight recent regime more than ancient history.
HRP_REFIT_LOOKBACK_DAYS = 365   # calendar days; ~252 trading days


def _compute_weekly_metrics(
    daily_runs: list[dict],
    equity_curve: dict[date, float],
) -> dict:
    """Compute the headline numbers the weekly AI analyst needs.

    Aims for SIGNAL not VOLUME — the metrics here are what a human PM would
    glance at first. Each value is also passed verbatim to the AI so it
    can quote magnitudes in the narrative.

    Shared base math (Sharpe, max-DD, daily returns) lives in
    ``util.equity_stats`` so the monthly review uses the same formulas;
    weekly adds top-5 attribution + concentration on top.
    """
    if not equity_curve:
        return {"n_days": 0, "note": "no equity data this week"}

    metrics = equity_series_stats(equity_curve)

    # Position-attribution: top 5 movers over the week.
    top_gainers, top_losers = top_movers(daily_runs, n=5)
    metrics["top_gainers_week"] = top_gainers
    metrics["top_losers_week"] = top_losers

    # Concentration: top-3 weights as % of equity from latest run.
    if daily_runs:
        last_run = max(daily_runs, key=lambda r: r.get("date", ""))
        targets = last_run.get("target_weights", {})
        sorted_w = sorted(targets.values(), reverse=True)
        top3 = sum(sorted_w[:3]) if sorted_w else 0.0
        metrics["top3_concentration_pct"] = round(top3 * 100, 2)
        metrics["n_positions_latest_run"] = len(targets)
    else:
        metrics["top3_concentration_pct"] = 0.0
        metrics["n_positions_latest_run"] = 0
    return metrics


def run_weekly_review(
    *,
    for_date: date | None = None,
    runs_dir: Path | None = None,
    email_sender: EmailSender | None = None,
    config: Config | None = None,
    state_path: Path | None = None,
    refit_hrp: bool = True,
    enable_ai_analyst: bool = True,
    weekly_dir: Path | None = None,
    n_past_reports: int = 4,
    # Test-injection points:
    cache: BarsCache | None = None,
    universe: list[str] | None = None,
) -> str:
    """Load week's runs, refit HRP weights, email summary. Returns subject."""
    for_date = for_date or date.today()
    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    config = config or load_config()

    # -------- 1. Aggregate the week's daily runs --------
    week_dates: list[date] = []
    daily_runs: list[dict] = []
    equity_curve: dict[date, float] = {}
    for offset in range(7):
        d = for_date - timedelta(days=offset)
        payload = load_daily_run(d, runs_dir=runs_dir)
        if payload is None:
            continue
        week_dates.append(d)
        daily_runs.append(payload)
        eq = payload.get("execution_report", {}).get("account_equity_before")
        if eq is not None:
            equity_curve[d] = float(eq)

    # -------- 2. HRP refit (best-effort; failure surfaces in the email) --------
    refit_notes: list[str] = []
    hrp_diag: dict = {}     # captured here so the AI analyst can reference it
    if refit_hrp:
        try:
            state = load_ensemble_state(path=state_path)
            universe = universe or load_top50_snapshot()
            cache = cache or BarsCache(
                client=AlpacaDataClient(), root=Path("data/bars/daily"),
            )
            end = for_date - timedelta(days=1)
            start = end - timedelta(days=HRP_REFIT_LOOKBACK_DAYS)
            bars = cache.get_daily_bars(universe, start, end)
            if bars.empty:
                refit_notes.append(
                    "HRP refit skipped: no bars fetched. Cache or Alpaca issue."
                )
            else:
                new_hrp, diag = refit_hrp_weights(
                    state,
                    universe=universe,
                    bars=bars,
                    config=config,
                )
                hrp_diag = diag   # surface for the AI analyst
                # Persist the new weights + bump the refit date.
                updated_state = replace(
                    state,
                    hrp_weights=new_hrp,
                    last_hrp_refit_date=for_date.isoformat(),
                )
                save_ensemble_state(updated_state, path=state_path)
                refit_notes.append("**HRP weights refit:**")
                for sname, w in sorted(
                    new_hrp.items(), key=lambda kv: -kv[1]
                ):
                    prev = state.hrp_weights.get(sname, 0.0)
                    arrow = "↑" if w > prev else "↓" if w < prev else "→"
                    refit_notes.append(
                        f"- `{sname}`: {prev:.3f} {arrow} {w:.3f}"
                    )
                # Surface per-strategy backtest stats from the diagnostic.
                per = diag.get("per_strategy", {})
                if per:
                    refit_notes.append("")
                    refit_notes.append("**Per-strategy 1-year backtest:**")
                    for sname, s in per.items():
                        if s.get("skipped"):
                            refit_notes.append(
                                f"- `{sname}`: skipped ({s.get('reason')})"
                            )
                        else:
                            refit_notes.append(
                                f"- `{sname}`: total {s['total_return']:+.2%}, "
                                f"Sharpe {s['sharpe']:+.2f}"
                            )
        except Exception as e:
            refit_notes.append(f"HRP refit failed: {type(e).__name__}: {e}")
            logger.exception("HRP refit failed")

    # -------- 3. AI deep-dive (Claude writes the qualitative analysis) ------
    # Pre-compute metrics so the analyst has structured numbers to anchor
    # claims; pass the HRP diagnostic so it can reason about weight shifts.
    # Disabled in tests via enable_ai_analyst=False to avoid hitting the API.
    deep_dive_md = ""
    metrics: dict = {}
    if enable_ai_analyst and daily_runs:
        try:
            from quant.agent.ai_analyst import AIAnalyst  # lazy: optional dep
            metrics = _compute_weekly_metrics(daily_runs, equity_curve)
            # Self-improvement: feed the past N weeks' narratives so this
            # week's analysis can reference continuity, self-critique, and
            # escalate persistent issues (per WEEKLY_ANALYST.md §5).
            past_reports = load_recent_weekly_reports(
                before=for_date, n=n_past_reports, weekly_dir=weekly_dir,
            )
            logger.info(
                "weekly_review: loaded %d past weekly reports for self-improvement",
                len(past_reports),
            )
            analyst = AIAnalyst()
            weekly_report = analyst.analyze_weekly(
                daily_runs=daily_runs,
                weekly_metrics=metrics,
                hrp_diagnostic=hrp_diag,
                past_weekly_reports=past_reports,
            )
            deep_dive_md = weekly_report.narrative
            # Persist THIS week's narrative so next week (and the monthly
            # review) can read it.
            saved_path = save_weekly_report(
                week_ending=for_date,
                narrative=deep_dive_md,
                metrics=metrics,
                hrp_diagnostic=hrp_diag,
                weekly_dir=weekly_dir,
            )
            logger.info("weekly_review: saved deep-dive to %s", saved_path)
        except Exception as e:
            # AI failure must NOT prevent the rest of the report from sending.
            # Surface it inline so the operator notices.
            deep_dive_md = (
                f"_AI deep-dive failed: {type(e).__name__}: {e}. "
                "The numeric summary and HRP refit below are still valid._"
            )
            logger.exception("weekly AI analyst call failed")

    # -------- 4. Render + email --------
    if not daily_runs and not refit_notes:
        notes = (
            f"No daily run records found between "
            f"{(for_date - timedelta(days=6)).isoformat()} and "
            f"{for_date.isoformat()}, and HRP refit was skipped."
        )
    else:
        # Order: deep-dive first (top of email = highest signal), then
        # mechanical refit notes below.
        parts: list[str] = []
        if deep_dive_md:
            parts.append("## AI Weekly Deep-Dive\n\n" + deep_dive_md)
        if refit_notes:
            parts.append("## HRP Refit & Per-Strategy Stats\n\n" + "\n".join(refit_notes))
        notes = "\n\n---\n\n".join(parts)
        if not daily_runs:
            notes = (
                "No daily run records this week. (Agent likely wasn't running.)\n\n"
                + notes
            )

    subject, body = render_weekly_report(
        week_ending=for_date,
        daily_runs=daily_runs,
        equity_curve=equity_curve,
        notes=notes,
    )
    sender = email_sender or EmailSender()
    sender.send(subject=subject, body_text=body, body_html=_markdown_to_html(body))
    logger.info("weekly review emailed: %s", subject)
    return subject


def cli_run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Send the agent's weekly review.")
    parser.add_argument("--for-date", default=None, help="ISO date; defaults to today")
    args = parser.parse_args()
    for_date = date.fromisoformat(args.for_date) if args.for_date else None
    try:
        subject = run_weekly_review(for_date=for_date)
        print(f"[agent] weekly review sent: {subject}")
    except Exception as e:
        _email_failure("weekly review", e)
        raise
