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
from quant.data.universe import load_active_universe
from quant.util.equity_stats import daily_returns, equity_series_stats, top_movers

logger = logging.getLogger(__name__)


# How much history to feed the improver's backtests. 2 years is enough
# for ~500 trading days of returns — enough to estimate Sharpe with some
# stability, not so much that ancient regimes dominate.
IMPROVER_BACKTEST_YEARS = 2


def _pit_universe_is_active() -> bool:
    """True iff the point-in-time S&P 500 loader has enough names to
    be the live universe (no fallback to the static top-50 snapshot).

    Computed at monthly-review time so the wiring_status flag in the
    pipeline snapshot reflects the CURRENT state of the data file.
    Flips True after the operator runs (or the cron auto-runs)
    quant-sp500-refresh and the CSV grows past the minimum viability
    threshold; flips False if someone truncates the file.
    """
    from datetime import date as _date

    from quant.data.universe import _MIN_VIABLE_UNIVERSE_SIZE, load_universe
    try:
        members = load_universe("sp500").members(_date.today())
        return len(members) >= _MIN_VIABLE_UNIVERSE_SIZE
    except Exception:
        return False


def _build_pipeline_snapshot(config: Config) -> dict:
    """Bundle every hardcoded knob + config value the AI analyst should
    cross-check each month.

    Why this exists: the analyst is great at proposing strategies from
    daily data, but it CANNOT see the codebase. So when a hardcoded
    constant drifts from the config (e.g., MAX_POSITION_WEIGHT was 20%
    in code but 5% in config for many commits — a real bug we just
    found), the analyst has no way to notice. This snapshot exposes
    the knobs as data so the analyst's pipeline self-audit (per
    ANALYST.md §6 step 5c) can flag the drift.

    Includes:
      - All operator hard-rule constants (from daily_runner.py)
      - All sandbox gate thresholds (from strategy_sandbox.py)
      - The relevant config.yaml risk + cost values
      - Whether known risk features are actively wired ("dead code"
        detection: if the kill switch is configured but never called,
        we want the analyst to surface that).

    The analyst is instructed to compare these against industry norms
    AND against each other (drift detection), and to emit findings
    via `proposed_state_changes.pipeline_findings`.
    """
    from quant.agent.daily_runner import (
        MAX_DRAWDOWN_KILL,
        MAX_POSITION_WEIGHT,
        STOP_LOSS_PCT,
    )
    from quant.agent.strategy_sandbox import (
        _BACKTEST_TIMEOUT_SECONDS,
        _DSR_THRESHOLD,
        _MAX_DRAWDOWN_ABS,
        _MIN_SHARPE,
    )
    return {
        "operator_hard_rules_in_code": {
            "STOP_LOSS_PCT": STOP_LOSS_PCT,
            "MAX_POSITION_WEIGHT": MAX_POSITION_WEIGHT,
            "MAX_DRAWDOWN_KILL": MAX_DRAWDOWN_KILL,
            "_source": "src/quant/agent/daily_runner.py",
        },
        "sandbox_gates_in_code": {
            "min_sharpe": _MIN_SHARPE,
            "max_drawdown_abs": _MAX_DRAWDOWN_ABS,
            "dsr_threshold": _DSR_THRESHOLD,
            "backtest_timeout_seconds": _BACKTEST_TIMEOUT_SECONDS,
            "_source": "src/quant/agent/strategy_sandbox.py",
        },
        "config_yaml_values": {
            "risk_max_position_weight": config.risk.max_position_weight,
            "risk_max_drawdown_kill": config.risk.max_drawdown_kill,
            "risk_vol_target_annual": config.risk.vol_target_annual,
            "risk_max_gross_leverage": config.risk.max_gross_leverage,
            "risk_max_net_exposure": config.risk.max_net_exposure,
            "backtest_costs_spread_bps": config.backtest.costs.spread_bps,
            "backtest_costs_slippage_bps": config.backtest.costs.slippage_bps,
            "backtest_costs_commission_bps": config.backtest.costs.commission_bps,
            "universe_min_avg_dollar_volume_20d": config.universe.min_avg_dollar_volume_20d,
            "universe_min_price": config.universe.min_price,
            "_source": "configs/default.yaml",
        },
        "wiring_status": {
            "drawdown_kill_switch_active_in_daily_trade": True,
            "vol_targeting_active_in_daily_trade": True,           # wired in T1.3
            "atr_normalized_stops_active": True,                   # wired in T1.5
            "fill_anchored_stops_active": True,                    # wired in T1.4 (post-fill repair)
            "sector_concentration_cap_active": True,               # wired in T2.9
            "conviction_weighted_strategy_outputs": True,          # wired in T2.8 (all 3 strategies)
            "trail_high_uses_intraday_high": True,                 # wired in T2.10
            "mean_reversion_vol_normalized": True,                 # wired in T2.11
            "improver_uses_cost_aware_backtest": True,             # verified in T2.12 — was already correct
            "universe_uses_point_in_time_membership": _pit_universe_is_active(),  # Computed
                                                                   # at snapshot-build time — checks whether
                                                                   # load_universe('sp500').members(today)
                                                                   # returns the minimum viable count. Flips
                                                                   # True automatically after the quarterly
                                                                   # quant-sp500-refresh cron populates the
                                                                   # CSV; flips False if someone truncates
                                                                   # the file.
            "_notes": (
                "These flags indicate whether 'advertised' risk-management features are "
                "actually called in the live trading path. A False here is a dead-code or "
                "missing-feature signal — the analyst should flag it via pipeline_findings."
            ),
        },
        "industry_norms_for_comparison": {
            "max_position_weight_institutional": "0.03 to 0.05 (3-5%)",
            "max_drawdown_kill_institutional": "0.10 to 0.15 (10-15%)",
            "min_sharpe_institutional_floor": "0.7 to 1.0",
            "max_strategy_drawdown_institutional": "0.15 to 0.20 (15-20%)",
            "sector_concentration_typical_cap": "0.20 to 0.30 (20-30%)",
            "top3_concentration_typical_cap": "0.25 to 0.30 (25-30%)",
        },
    }


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
    ai_only: bool = False,
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

    Parameters
    ----------
    ai_only
        Recovery path after a transient AI failure (HTTP 403, connection
        reset, etc.). When True, ALL state mutations are gated:
          • xsec auto-apply is forced off (regardless of ``auto_apply``)
          • Accepted AI strategies are NOT persisted to disk / state
          • MEMORY.md and STRATEGY_LIBRARY.md are NOT appended
        Effect: re-running with ``--ai-only`` re-renders the analyst's
        narrative for the operator's inbox without polluting state with
        duplicate entries. Used by ``quant-monthly-review --ai-only``.
    """
    # T-fix B: --ai-only forces no-mutation mode.
    if ai_only:
        auto_apply = False
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

    # -------- 1b. Benchmarks (best-effort) -----------------------------------
    # SPY + QQQ close-to-close over the month so the email surfaces relative
    # performance vs the broad market and Nasdaq. Best-effort: failures do
    # not block the email.
    benchmarks: dict[str, float] = {}
    if equity_curve:
        try:
            from quant.util.benchmarks import fetch_benchmark_returns

            bc_for_bench = cache or BarsCache(
                client=AlpacaDataClient(), root=Path("data/bars/daily"),
            )
            eq_dates = sorted(equity_curve.keys())
            benchmarks = fetch_benchmark_returns(
                bc_for_bench, eq_dates[0], eq_dates[-1],
            )
        except Exception as e:
            logger.warning(
                "monthly_review: benchmark fetch failed (%s: %s)",
                type(e).__name__, e,
            )

    # -------- 2. Math-based grid search (xsec momentum params) ---------------
    current_state = load_ensemble_state(path=params_path)
    current_xsec_params = StrategyParams(
        top_k=current_state.xsec_top_k,
        lookback=current_state.xsec_lookback,
        skip=current_state.xsec_skip,
    )
    universe = universe or load_active_universe(for_date)
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
            benchmarks=benchmarks,
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
        # Pipeline self-audit snapshot — exposes hardcoded constants +
        # config values + wiring status so the analyst can flag drift
        # between them, dead-wired risk features, and gates that fall
        # below institutional norms. See ai_analyst._build_user_message
        # and ANALYST.md §6 step 5c.
        pipeline_snapshot = _build_pipeline_snapshot(config)

        # Quant diagnostics bundle (Phase 4): factor attribution (alpha vs
        # beta), per-strategy signal health (IC/decay/regime), and the
        # current regime + candidate regime policy. Rendered in the email
        # and fed to the analyst so the monthly is grounded in numbers, not
        # narrative. Guarded — a diagnostics failure must not block review.
        quant_diagnostics: dict | None = None
        try:
            from quant.agent.monthly_diagnostics import (
                build_quant_diagnostics,
                render_diagnostics_md,
            )
            dcache = cache or BarsCache(
                client=AlpacaDataClient(), root=Path("data/bars/daily"),
            )
            quant_diagnostics = build_quant_diagnostics(
                equity_curve=equity_curve,
                state=current_state,
                universe=universe,
                cache=dcache,
                as_of=for_date,
            )
            recommendations.append(render_diagnostics_md(quant_diagnostics))
        except Exception as e:
            logger.warning(
                "monthly quant diagnostics failed (%s: %s) — continuing",
                type(e).__name__, e,
            )

        analyst = AIAnalyst()
        ai_report = analyst.analyze(
            daily_runs=daily_runs,
            current_state=asdict(current_state),
            ai_strategy_names=list(current_state.ai_strategy_names),
            grid_search_summary=grid_search_summary,
            recent_weekly_reports=recent_weeklies,
            monthly_metrics=monthly_metrics,
            pipeline_snapshot=pipeline_snapshot,
            quant_diagnostics=quant_diagnostics,
        )
        # Always include the qualitative analysis in the email.
        recommendations.append(f"## AI Analysis\n\n{ai_report.analysis}")

        # -------- 3a. Pipeline self-audit findings (surfaced PROMINENTLY) ---
        # These are infrastructure issues — config drift, dead-wired risk
        # features, gates below institutional norms. They're rendered at
        # the TOP of recommendations (above strategy proposals) because
        # they affect every position the agent ever takes, not just one
        # new strategy. The operator decides what to act on.
        if ai_report.pipeline_findings:
            sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            sorted_findings = sorted(
                ai_report.pipeline_findings,
                key=lambda f: sev_order.get(f.severity.lower(), 4),
            )
            sev_emoji = {
                "critical": "🚨", "high": "🔴", "medium": "🟠", "low": "🟡",
            }
            lines = ["## 🛠️ Pipeline Self-Audit Findings", ""]
            lines.append(
                f"_{len(ai_report.pipeline_findings)} infrastructure issue(s) "
                "the analyst found this month. These are NOT strategy ideas — "
                "they're config drift / dead code / weak risk thresholds. "
                "Operator review required (no auto-apply for safety-critical changes)._\n"
            )
            for f in sorted_findings:
                emoji = sev_emoji.get(f.severity.lower(), "")
                lines.append(
                    f"### {emoji} {f.severity.upper()} — {f.category}\n\n"
                    f"**Issue**: {f.description}\n\n"
                    f"**Recommendation**: {f.recommendation}\n"
                )
            # INSERT at position 0 so it lands at the TOP of the email,
            # above the AI Analysis section.
            recommendations.insert(0, "\n".join(lines))
        else:
            recommendations.append(
                "_AI analyst pipeline self-audit: no infrastructure findings this month._"
            )

        # -------- 3b. Surface any proposed state-change ---------------------
        # We do NOT auto-apply state changes — risk-management and strategy
        # tuning is too important to handle silently. The operator reviews
        # and edits ensemble_state.json manually if they agree with the
        # proposal. Five knobs are tunable via this channel: trail_pct
        # (exit risk), sma_fast / sma_slow (SMA crossover), mr_lookback /
        # mr_threshold_pct (mean-reversion). See ANALYST.md §6.5.
        sc = ai_report.proposed_state_changes
        if sc is not None:
            from quant.agent.daily_runner import STOP_LOSS_PCT

            # (proposed_value, current_value, field_name, fmt, validator)
            # validator returns (within_bounds: bool, hint: str)
            def _trail_valid(v: float) -> tuple[bool, str]:
                return (0 < v <= STOP_LOSS_PCT,
                        f"must be in (0, {STOP_LOSS_PCT}]")

            def _sma_fast_valid(v: int) -> tuple[bool, str]:
                slow = sc.sma_slow if sc.sma_slow is not None else current_state.sma_slow
                return (2 <= v < slow, f"must be in [2, sma_slow={slow})")

            def _sma_slow_valid(v: int) -> tuple[bool, str]:
                fast = sc.sma_fast if sc.sma_fast is not None else current_state.sma_fast
                return (v > fast, f"must be > sma_fast={fast}")

            def _mr_lookback_valid(v: int) -> tuple[bool, str]:
                return (v >= 2, "must be >= 2")

            def _mr_threshold_valid(v: float) -> tuple[bool, str]:
                return (v > 0, "must be > 0")

            knob_specs = [
                ("trail_pct", sc.trail_pct, current_state.trail_pct,
                 "{:.4f}", _trail_valid),
                ("sma_fast", sc.sma_fast, current_state.sma_fast,
                 "{}", _sma_fast_valid),
                ("sma_slow", sc.sma_slow, current_state.sma_slow,
                 "{}", _sma_slow_valid),
                ("mr_lookback", sc.mr_lookback, current_state.mr_lookback,
                 "{}", _mr_lookback_valid),
                ("mr_threshold_pct", sc.mr_threshold_pct, current_state.mr_threshold_pct,
                 "{:.4f}", _mr_threshold_valid),
            ]

            proposed_lines: list[str] = []
            for name, new_val, cur_val, fmt, validator in knob_specs:
                if new_val is None:
                    continue
                within, hint = validator(new_val)
                verdict = (
                    "VALID — operator may apply by editing "
                    f"`{name}` in `data/agent/ensemble_state.json`."
                    if within else
                    f"OUT OF BOUNDS — {hint}. "
                    "Operator should NOT apply this value verbatim."
                )
                arrow = "↓" if new_val < cur_val else "↑" if new_val > cur_val else "→"
                proposed_lines.append(
                    f"- **`{name}`**: `{fmt.format(cur_val)}` {arrow} "
                    f"`{fmt.format(new_val)}` ({verdict})"
                )

            if proposed_lines:
                recommendations.append(
                    "## 🎯 AI proposed state-change(s)\n\n"
                    + "\n".join(proposed_lines)
                    + f"\n\n**Reasoning** (verbatim from analyst):\n\n"
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
                    if ai_only:
                        # T-fix B: re-run mode. Show what WOULD have been
                        # accepted but mutate nothing on disk — the
                        # original run already handled persistence.
                        recommendations.append(
                            f"✅ **AI strategy would have been accepted** "
                            f"(`{proposal.name}`) but `--ai-only` re-run "
                            f"skips state mutation. Sharpe {sandbox.sharpe:.2f}, "
                            f"max DD {sandbox.max_drawdown:.1%}, "
                            f"DSR {sandbox.dsr:.3f}.\n\n"
                            "If this is a duplicate, the original monthly "
                            "run already persisted it; check "
                            "`data/agent/ensemble_state.json` and "
                            "`src/quant/strategies/generated/`."
                        )
                    else:
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

        # -------- Append to MEMORY.md (success, rejection, or no proposal)
        # ai_only re-runs skip this — the original monthly run already
        # appended its entry; a re-run would duplicate it and confuse
        # next month's analyst.
        if ai_only:
            logger.info(
                "monthly review (--ai-only): skipping MEMORY.md append "
                "(original run already wrote its entry)"
            )
        else:
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
        # format_ai_error classifies the failure (403 / connection / other)
        # and includes the --ai-only re-run recipe so the operator can
        # recover with one command.
        from quant.agent.ai_analyst import format_ai_error
        recommendations.append(format_ai_error(e))
        logger.exception("ai analyst failed")

    # -------- 5. Render + send -----------------------------------------------
    subject, body = render_monthly_report(
        month_ending=for_date,
        daily_runs=daily_runs,
        equity_curve=equity_curve,
        recommendations=recommendations,
        benchmarks=benchmarks,
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
    parser.add_argument(
        "--ai-only", action="store_true",
        help=(
            "Re-run the AI analyst without mutating any state. Use this "
            "to recover from a transient Anthropic failure (HTTP 403, "
            "connection reset, etc.): switch VPN region, then "
            "`quant-monthly-review --for-date=YYYY-MM-DD --ai-only`. "
            "Skips xsec auto-apply, AI strategy acceptance, MEMORY.md "
            "and STRATEGY_LIBRARY.md appends — the original monthly "
            "run already handled all of those, so a re-run would "
            "duplicate entries. The analyst's narrative IS re-rendered "
            "and emailed."
        ),
    )
    args = parser.parse_args()
    for_date = date.fromisoformat(args.for_date) if args.for_date else None
    try:
        subject, _ = run_monthly_review(
            for_date=for_date,
            auto_apply=not args.no_apply,
            ai_only=args.ai_only,
        )
        print(f"[agent] monthly review sent: {subject}")
    except Exception as e:
        _email_failure("monthly review", e)
        raise
