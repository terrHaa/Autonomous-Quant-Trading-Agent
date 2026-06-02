"""monthly_review.py — end-of-month review + safety-railed auto-apply.

Runs on the last trading day of each month after the close. Does what
the weekly review does, PLUS:

- Backtests a small grid of candidate strategy parameters.
- Applies three safety gates (Sharpe up, drawdown not worse,
  DSR ≥ 0.95 vs trial population).
- If a candidate passes all gates, **automatically writes** the new
  xsec params into the ``EnsembleState`` on disk (preserving the
  SMA + MR + HRP layers). The daily runner picks it up next session.
- Emails the operator the full result (candidates considered, gates
  passed/failed, what (if anything) was applied).

This is the operator's "full auto-apply with safety rails" choice. The
gates are deliberately conservative so most monthly runs don't change
anything; that's the goal — only switch when the evidence is strong.

Console-script: ``quant-monthly-review`` (registered in pyproject.toml).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path

from quant.agent.ai_analyst import (
    AIAnalyst,
    AnalysisReport,
    append_accepted_strategy_to_library,
    append_memory_entry,
)
from quant.agent.daily_runner import _email_failure, _markdown_to_html
from quant.agent.email_sender import EmailSender
from quant.agent.ensemble import (
    load_ensemble_state,
    save_ensemble_state,
)
from quant.agent.improver import (
    ImprovementResult,
    search_improvements,
)
from quant.agent.log import (
    DEFAULT_RUNS_DIR,
    load_daily_run,
    load_recent_weekly_reports,
)
from quant.agent.params import StrategyParams
from quant.agent.reports import render_monthly_report
from quant.agent.strategy_sandbox import (
    SandboxResult,
    save_generated_strategy,
    validate_and_test_strategy,
)
from quant.config import Config, load_config
from quant.data.alpaca_client import AlpacaDataClient
from quant.data.cache import BarsCache
from quant.data.universe import load_top50_snapshot
from quant.util.equity_stats import daily_returns, equity_series_stats, top_movers

logger = logging.getLogger(__name__)


# How much history to feed the improver's backtests. 2 years is enough
# for ~500 trading days of returns — enough to estimate Sharpe with some
# stability, not so much that ancient regimes dominate.
IMPROVER_BACKTEST_YEARS = 2


def _compute_monthly_metrics(
    daily_runs: list[dict],
    equity_curve: dict[date, float],
) -> dict:
    """30-day statistical view for the monthly analyst.

    Builds on the same base stats as the weekly review (Sharpe, max DD,
    daily returns — shared via ``util.equity_stats``), then layers on
    longer-horizon patterns a 4-week window of weekly summaries would
    miss:
      • Lag-1 autocorrelation of returns (trend vs MR regime)
      • Day-of-week breakdown (catches calendar effects)
      • Position persistence (book stability)
      • HRP weight drift over the full month
      • Streak analysis (psychology + tail risk)
      • Top-10 movers (vs top-5 in weekly — broader attribution)
      • Raw daily-return series (analyst can run its own stats)
    The monthly analyst is directed (ANALYST.md §6 step 5b) to combine
    these statistical metrics with the weekly narratives and the raw
    daily-runs table for triangulated analysis.
    """
    if not equity_curve or len(equity_curve) < 2:
        return {"insufficient_data": True, "n_days": len(equity_curve)}

    # Base stats (Sharpe, vol, max DD, etc.) from the shared core.
    metrics = equity_series_stats(equity_curve)
    return_dates, rets = daily_returns(equity_curve)

    # Day-of-week breakdown: Mon=0..Fri=4. Mean return + win rate per dow.
    dow_buckets: dict[int, list[float]] = {k: [] for k in range(5)}
    for d, r in zip(return_dates, rets, strict=True):
        if d.weekday() < 5:   # exclude weekends defensively (shouldn't occur)
            dow_buckets[d.weekday()].append(r)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    dow_summary: dict[str, dict] = {}
    for k, rs in dow_buckets.items():
        if rs:
            mean = sum(rs) / len(rs)
            wins = sum(1 for r in rs if r > 0)
            dow_summary[dow_names[k]] = {
                "n": len(rs),
                "mean_return_pct": round(mean * 100, 3),
                "win_rate_pct": round(wins / len(rs) * 100, 1),
            }

    # Lag-1 autocorrelation — positive = trending/momentum regime,
    # negative = mean-reverting regime, ~0 = noise.
    if len(rets) >= 3:
        r_prev = rets[:-1]
        r_next = rets[1:]
        m_prev = sum(r_prev) / len(r_prev)
        m_next = sum(r_next) / len(r_next)
        cov = sum(
            (a - m_prev) * (b - m_next)
            for a, b in zip(r_prev, r_next, strict=True)
        ) / (len(r_prev) - 1)
        s_prev = math.sqrt(sum((a - m_prev) ** 2 for a in r_prev) / (len(r_prev) - 1))
        s_next = math.sqrt(sum((b - m_next) ** 2 for b in r_next) / (len(r_next) - 1))
        autocorr1 = cov / (s_prev * s_next) if s_prev > 0 and s_next > 0 else 0.0
    else:
        autocorr1 = 0.0

    # Position persistence — what fraction of yesterday's target names
    # survive into today's targets? High = stable book; low = high churn.
    sorted_runs = sorted(daily_runs, key=lambda r: r.get("date", ""))
    persistence_rates: list[float] = []
    for i in range(1, len(sorted_runs)):
        prev_tgts = set(sorted_runs[i - 1].get("target_weights", {}).keys())
        curr_tgts = set(sorted_runs[i].get("target_weights", {}).keys())
        if prev_tgts:
            persistence_rates.append(len(prev_tgts & curr_tgts) / len(prev_tgts))
    avg_persistence = (
        sum(persistence_rates) / len(persistence_rates) if persistence_rates else 0.0
    )

    # HRP weight drift over the month
    hrp_drift: dict[str, float] = {}
    if len(sorted_runs) >= 2:
        first_hrp = (
            sorted_runs[0]
            .get("strategy_params", {})
            .get("ensemble_state", {})
            .get("hrp_weights", {})
        )
        last_hrp = (
            sorted_runs[-1]
            .get("strategy_params", {})
            .get("ensemble_state", {})
            .get("hrp_weights", {})
        )
        for k in set(first_hrp) | set(last_hrp):
            drift = last_hrp.get(k, 0.0) - first_hrp.get(k, 0.0)
            hrp_drift[k] = round(drift, 4)

    # Streak analysis — longest positive and negative runs of consecutive
    # daily returns. Useful for psychology + tail risk reasoning.
    longest_win = longest_loss = 0
    cur_run = cur_sign = 0
    for r in rets:
        sign = 1 if r > 0 else (-1 if r < 0 else 0)
        if sign == cur_sign and sign != 0:
            cur_run += 1
        else:
            cur_sign = sign
            cur_run = 1 if sign != 0 else 0
        if sign > 0:
            longest_win = max(longest_win, cur_run)
        elif sign < 0:
            longest_loss = max(longest_loss, cur_run)

    # Top-10 movers over the full month — shared helper, same logic as weekly.
    top10_gainers, top10_losers = top_movers(daily_runs, n=10)

    metrics.update({
        "longest_winning_streak_days": longest_win,
        "longest_losing_streak_days": longest_loss,
        "lag1_autocorrelation": round(autocorr1, 3),
        "avg_position_persistence_pct": round(avg_persistence * 100, 1),
        "hrp_weight_drift_over_month": hrp_drift,
        "day_of_week_breakdown": dow_summary,
        "top10_gainers_month": top10_gainers,
        "top10_losers_month": top10_losers,
        # Raw daily-return series — empowers the analyst to compute its
        # own statistics (e.g., rolling Sharpe, regime breaks).
        "daily_returns_pct": [round(r * 100, 3) for r in rets],
    })
    return metrics


def run_monthly_review(
    *,
    for_date: date | None = None,
    runs_dir: Path | None = None,
    email_sender: EmailSender | None = None,
    config: Config | None = None,
    auto_apply: bool = True,
    params_path: Path | None = None,
    # Test-injection points:
    cache: BarsCache | None = None,
    universe: list[str] | None = None,
    enable_ai_analyst: bool = True,
) -> tuple[str, ImprovementResult | None]:
    """Run the month's review and return (subject, improvement_result).

    Flow
    ----
    1. Aggregate the month's daily run JSONs.
    2. Grid-search for better xsec-momentum params (math-based).
    3. Call the AI analyst for a qualitative narrative + optional new strategy.
    4. Validate + backtest the proposed strategy through the sandbox.
    5. If it passes all gates, persist it and update the EnsembleState.
    6. Render and email the full report (grid results + AI analysis + decision).

    ``improvement_result`` is None when the improver step was skipped
    (e.g., test mode where no cache is configured).
    """
    for_date = for_date or date.today()
    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    config = config or load_config()

    # -------- 1. Aggregate the month's daily runs ----------------------------
    daily_runs = []
    equity_curve: dict[date, float] = {}
    for offset in range(31):
        d = for_date - timedelta(days=offset)
        payload = load_daily_run(d, runs_dir=runs_dir)
        if payload is None:
            continue
        daily_runs.append(payload)
        eq = payload.get("execution_report", {}).get("account_equity_before")
        if eq is not None:
            equity_curve[d] = float(eq)

    # -------- 2. Math-based grid search (xsec momentum params) ---------------
    current_state = load_ensemble_state(path=params_path)
    current_xsec_params = StrategyParams(
        top_k=current_state.xsec_top_k,
        lookback=current_state.xsec_lookback,
        skip=current_state.xsec_skip,
    )
    universe = universe or load_top50_snapshot()
    improvement_result: ImprovementResult | None = None
    recommendations: list[str] = []
    grid_search_summary = "Grid search was not run this month."
    bars = None  # will be set below; reused for AI sandbox

    try:
        cache = cache or BarsCache(
            client=AlpacaDataClient(), root=Path("data/bars/daily"),
        )
        end = for_date - timedelta(days=1)
        start = end - timedelta(days=IMPROVER_BACKTEST_YEARS * 365)
        bars = cache.get_daily_bars(universe, start, end)
        if bars.empty:
            recommendations.append(
                "Improver skipped: no bars fetched. Cache or Alpaca issue."
            )
            grid_search_summary = "No bars available — grid search skipped."
        else:
            improvement_result = search_improvements(
                current_xsec_params,
                universe=universe,
                bars=bars,
                config=config,
            )
            n_cands = len(improvement_result.candidates)
            recommendations.append(
                f"Grid search evaluated {n_cands} candidates on {len(bars)} bars."
            )
            best = improvement_result.best_passing
            if best is None:
                recommendations.append(f"No parameter change: {improvement_result.reason}")
                grid_search_summary = (
                    f"Evaluated {n_cands} candidates. "
                    f"No improvement found: {improvement_result.reason}"
                )
            else:
                grid_search_summary = (
                    f"Evaluated {n_cands} candidates. "
                    f"Best: top_k={best.params.top_k}, "
                    f"lookback={best.params.lookback}, skip={best.params.skip}. "
                    f"Sharpe {best.sharpe:.2f} vs current {improvement_result.current.sharpe:.2f}. "
                    f"{improvement_result.reason}"
                )
                if auto_apply:
                    new_state = replace(
                        current_state,
                        xsec_top_k=best.params.top_k,
                        xsec_lookback=best.params.lookback,
                        xsec_skip=best.params.skip,
                    )
                    save_ensemble_state(new_state, path=params_path)
                    # Update current_state so the AI step sees the new params.
                    current_state = new_state
                    recommendations.append(
                        f"**APPLIED** xsec params: "
                        f"top_k={best.params.top_k}, "
                        f"lookback={best.params.lookback}, "
                        f"skip={best.params.skip} "
                        f"(was top_k={current_xsec_params.top_k}, "
                        f"lookback={current_xsec_params.lookback}, "
                        f"skip={current_xsec_params.skip})"
                    )
                else:
                    recommendations.append(
                        f"Candidate passes all gates but auto-apply is off. "
                        f"Run without --no-apply to manually apply: "
                        f"top_k={best.params.top_k}, "
                        f"lookback={best.params.lookback}, "
                        f"skip={best.params.skip}."
                    )
    except Exception as e:
        recommendations.append(f"Grid search failed: {type(e).__name__}: {e}")
        logger.exception("grid search failed")

    # -------- 3. AI analysis + strategy generation ---------------------------
    if not enable_ai_analyst:
        # Tests + dry-run paths can disable the API call entirely.
        recommendations.append("\n---\n_AI analyst disabled (enable_ai_analyst=False)._")
        subject, body = render_monthly_report(
            month_ending=for_date,
            daily_runs=daily_runs,
            equity_curve=equity_curve,
            recommendations=recommendations,
        )
        sender = email_sender or EmailSender()
        sender.send(subject=subject, body_text=body, body_html=_markdown_to_html(body))
        logger.info("monthly review emailed (AI disabled): %s", subject)
        return subject, improvement_result

    recommendations.append("\n---\n")
    ai_report: AnalysisReport | None = None
    try:
        # Load the past ~4 weekly reports so the monthly analyst builds
        # on the weekly analyst's curated observations (esp. items the
        # weekly side flagged with ai_analyst.ESCALATION_MARKER). Empty
        # list when no weekly reports exist yet — analyst handles that
        # case explicitly per ANALYST.md §6 step 5a.
        recent_weeklies = load_recent_weekly_reports(before=for_date, n=4)
        if recent_weeklies:
            recommendations.append(
                f"_Loaded {len(recent_weeklies)} prior weekly report(s) into "
                "analyst context (week endings: "
                f"{', '.join(r.get('week_ending', '?') for r in recent_weeklies)})._\n"
            )
        # Pre-compute monthly statistical metrics. Different signal from
        # weekly digests — these are 30-day patterns (autocorrelation,
        # day-of-week effects, position persistence, HRP weight drift,
        # streak runs) that 4 weekly summaries condensed into narratives
        # would obscure. The analyst is told to triangulate THREE sources:
        # weekly narratives + raw daily table + these metrics.
        monthly_metrics = _compute_monthly_metrics(daily_runs, equity_curve)
        analyst = AIAnalyst()
        ai_report = analyst.analyze(
            daily_runs=daily_runs,
            current_state=asdict(current_state),
            ai_strategy_names=list(current_state.ai_strategy_names),
            grid_search_summary=grid_search_summary,
            recent_weekly_reports=recent_weeklies,
            monthly_metrics=monthly_metrics,
        )
        # Always include the qualitative analysis in the email.
        recommendations.append(f"## AI Analysis\n\n{ai_report.analysis}")

        # -------- 3b. Surface any proposed state-change (trail_pct) ---------
        # We do NOT auto-apply state changes — risk-management tuning is
        # too important to handle silently. The operator reviews and edits
        # ensemble_state.json manually if they agree with the proposal.
        sc = ai_report.proposed_state_changes
        if sc is not None and sc.trail_pct is not None:
            new_val = sc.trail_pct
            cur_val = current_state.trail_pct
            # Light validation — surface a warning if the proposal violates
            # the operator's hard ceiling. Don't suppress it; the operator
            # might still want to see what the analyst was thinking.
            from quant.agent.daily_runner import STOP_LOSS_PCT
            within_bounds = 0 < new_val <= STOP_LOSS_PCT
            verdict = (
                "VALID — within bounds; operator may apply by editing "
                "`trail_pct` in `data/agent/ensemble_state.json`."
                if within_bounds else
                f"OUT OF BOUNDS — must be in (0, {STOP_LOSS_PCT}]. "
                "Operator should NOT apply this value verbatim."
            )
            arrow = "↓" if new_val < cur_val else "↑" if new_val > cur_val else "→"
            recommendations.append(
                f"## 🎯 AI proposed `trail_pct` change\n\n"
                f"**Current**: `{cur_val:.4f}` {arrow} **Proposed**: `{new_val:.4f}` "
                f"({verdict})\n\n"
                f"**Reasoning** (verbatim from analyst):\n\n"
                f"> {sc.reasoning}\n"
            )

        # -------- 4. Sandbox + gate the proposed strategy --------------------
        if ai_report.proposed_strategy is None:
            recommendations.append(
                "_AI analyst: no new strategy proposed this month._"
            )
        elif bars is None or bars.empty:
            recommendations.append(
                "_AI analyst proposed a strategy but bars unavailable — "
                "cannot backtest. Strategy not applied._"
            )
        else:
            proposal = ai_report.proposed_strategy
            # Skip if this strategy name is already in the ensemble.
            if proposal.name in current_state.ai_strategy_names:
                recommendations.append(
                    f"_AI proposed '{proposal.name}' which is already active — skipped._"
                )
            else:
                recommendations.append(
                    f"**AI proposed strategy**: `{proposal.name}` ({proposal.class_name})\n\n"
                    f"{proposal.reasoning}\n\n"
                    "_Running sandbox validation…_"
                )
                sandbox: SandboxResult = validate_and_test_strategy(
                    code=proposal.code,
                    class_name=proposal.class_name,
                    strategy_name=proposal.name,
                    universe=universe,
                    bars=bars,
                    config=config,
                )

                if sandbox.passed_gates:
                    # Save code to disk and update EnsembleState.
                    save_generated_strategy(
                        name=proposal.name,
                        class_name=proposal.class_name,
                        code=proposal.code,
                    )
                    # Accept into ensemble in SHADOW MODE:
                    # - Added to ai_strategy_names (so build_strategies loads it)
                    # - HRP weight stays at 0 during shadow (no real allocation)
                    # - ai_strategy_shadow_until[name] = today + 10 business days
                    # The daily runner graduates it at that date by removing
                    # the entry and giving it an equal-split initial weight.
                    import pandas as _pd  # noqa: PLC0415
                    shadow_end = (
                        _pd.Timestamp(for_date) + _pd.tseries.offsets.BDay(10)
                    ).date()

                    new_ai_names = list(current_state.ai_strategy_names) + [proposal.name]
                    new_shadow_map = dict(current_state.ai_strategy_shadow_until)
                    new_shadow_map[proposal.name] = shadow_end.isoformat()

                    # hrp_weights unchanged — no allocation until graduation.
                    final_state = replace(
                        current_state,
                        ai_strategy_names=new_ai_names,
                        ai_strategy_shadow_until=new_shadow_map,
                    )
                    save_ensemble_state(final_state, path=params_path)
                    recommendations.append(
                        f"✅ **AI STRATEGY ACCEPTED — ENTERING 10-DAY SHADOW**: `{proposal.name}`\n\n"
                        f"- Sharpe: {sandbox.sharpe:.2f}\n"
                        f"- Max drawdown: {sandbox.max_drawdown:.1%}\n"
                        f"- DSR: {sandbox.dsr:.3f}\n\n"
                        f"Shadow period ends: **{shadow_end.isoformat()}** "
                        f"(10 business days from today).\n\n"
                        f"During shadow the strategy will be called every day and its "
                        f"targets logged for analysis, but allocation remains zero. "
                        f"On {shadow_end.isoformat()} the daily runner graduates it to "
                        f"equal-split HRP weight; the weekly refit takes over from there.\n\n"
                        f"```python\n{proposal.code}\n```"
                    )
                    logger.info(
                        "monthly review: AI strategy '%s' applied. "
                        "Sharpe=%.2f DSR=%.3f",
                        proposal.name, sandbox.sharpe, sandbox.dsr,
                    )
                    # Persist to STRATEGY_LIBRARY.md so next month's analyst
                    # sees this strategy in its canonical catalog.
                    append_accepted_strategy_to_library(
                        review_date=for_date,
                        proposal=proposal,
                        sharpe=sandbox.sharpe,
                        max_drawdown=sandbox.max_drawdown,
                        dsr=sandbox.dsr,
                    )
                else:
                    recommendations.append(
                        f"❌ **AI strategy rejected**: {sandbox.rejection_reason}\n\n"
                        f"- Sharpe: {sandbox.sharpe:.2f} (min {0.30:.2f})\n"
                        f"- Max drawdown: {sandbox.max_drawdown:.1%} (cap 35%)\n"
                        f"- DSR: {sandbox.dsr:.3f} (threshold 0.95)\n\n"
                        f"The strategy code is logged below for reference:\n\n"
                        f"```python\n{proposal.code}\n```"
                    )
                    logger.info(
                        "monthly review: AI strategy '%s' rejected — %s",
                        proposal.name, sandbox.rejection_reason,
                    )

        # -------- Always append to MEMORY.md (success, rejection, or no proposal)
        try:
            if ai_report.proposed_strategy is None:
                outcome_label = "no_proposal"
                sandbox_summary = "(no proposal made)"
            elif ai_report.proposed_strategy.name in (current_state.ai_strategy_names or []):
                outcome_label = "duplicate_skipped"
                sandbox_summary = "Name already active — skipped sandbox."
            elif bars is None or bars.empty:
                outcome_label = "no_bars"
                sandbox_summary = "Bars unavailable — sandbox not run."
            elif sandbox.passed_gates:  # type: ignore[possibly-undefined]
                outcome_label = "accepted"
                sandbox_summary = (
                    f"Sharpe={sandbox.sharpe:.3f}, "
                    f"MaxDD={sandbox.max_drawdown:.1%}, "
                    f"DSR={sandbox.dsr:.3f} — passed all gates."
                )
            else:
                outcome_label = "rejected"
                sandbox_summary = (
                    f"Sharpe={sandbox.sharpe:.3f}, "
                    f"MaxDD={sandbox.max_drawdown:.1%}, "
                    f"DSR={sandbox.dsr:.3f}. "
                    f"Reason: {sandbox.rejection_reason}"
                )

            append_memory_entry(
                review_date=for_date,
                analysis=ai_report.analysis,
                proposal=ai_report.proposed_strategy,
                outcome=outcome_label,
                sandbox_details=sandbox_summary,
                grid_search_summary=grid_search_summary,
            )
        except Exception as mem_err:
            # MEMORY.md write failure must not break the review.
            logger.warning("ai_analyst: failed to append MEMORY.md entry: %s", mem_err)

    except RuntimeError as e:
        # ANTHROPIC_API_KEY not set — not a bug, just not configured yet.
        recommendations.append(
            f"_AI analyst skipped: {e}_\n\n"
            "Add `ANTHROPIC_API_KEY=sk-ant-...` to your `.env` file to enable it."
        )
        logger.info("ai analyst skipped (no API key): %s", e)
    except Exception as e:
        # Any other error — log and continue so the email still goes out.
        recommendations.append(f"_AI analyst failed: {type(e).__name__}: {e}_")
        logger.exception("ai analyst failed")

    # -------- 5. Render + send -----------------------------------------------
    subject, body = render_monthly_report(
        month_ending=for_date,
        daily_runs=daily_runs,
        equity_curve=equity_curve,
        recommendations=recommendations,
    )
    sender = email_sender or EmailSender()
    sender.send(subject=subject, body_text=body, body_html=_markdown_to_html(body))
    logger.info("monthly review emailed: %s", subject)
    return subject, improvement_result


def cli_run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Send the agent's monthly review.")
    parser.add_argument("--for-date", default=None, help="ISO date; defaults to today")
    parser.add_argument(
        "--no-apply", action="store_true",
        help="run the improver but never apply (review only)",
    )
    args = parser.parse_args()
    for_date = date.fromisoformat(args.for_date) if args.for_date else None
    try:
        subject, _ = run_monthly_review(
            for_date=for_date, auto_apply=not args.no_apply,
        )
        print(f"[agent] monthly review sent: {subject}")
    except Exception as e:
        _email_failure("monthly review", e)
        raise
