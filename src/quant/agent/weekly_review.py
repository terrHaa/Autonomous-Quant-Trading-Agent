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
from datetime import date, timedelta
from pathlib import Path

from quant.agent.daily_runner import _email_failure, _markdown_to_html
from quant.agent.email_sender import EmailSender
from quant.agent.ensemble import (
    EnsembleState,
    load_ensemble_state,
    refit_hrp_weights,
    save_ensemble_state,
)
from quant.agent.log import DEFAULT_RUNS_DIR, load_daily_run
from quant.agent.reports import render_weekly_report
from quant.config import Config, load_config
from quant.data.alpaca_client import AlpacaDataClient
from quant.data.cache import BarsCache
from quant.data.universe import load_top50_snapshot

logger = logging.getLogger(__name__)


# Window for the HRP refit's per-strategy backtest. 252 trading days
# = ~1 year. Long enough for stable correlation estimates, short enough
# to weight recent regime more than ancient history.
HRP_REFIT_LOOKBACK_DAYS = 365   # calendar days; ~252 trading days


def run_weekly_review(
    *,
    for_date: date | None = None,
    runs_dir: Path | None = None,
    email_sender: EmailSender | None = None,
    config: Config | None = None,
    state_path: Path | None = None,
    refit_hrp: bool = True,
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
                # Persist the new weights + bump the refit date.
                updated_state = EnsembleState(
                    sma_fast=state.sma_fast, sma_slow=state.sma_slow,
                    mr_lookback=state.mr_lookback,
                    mr_threshold_pct=state.mr_threshold_pct,
                    mr_allow_short=state.mr_allow_short,
                    xsec_top_k=state.xsec_top_k,
                    xsec_lookback=state.xsec_lookback,
                    xsec_skip=state.xsec_skip,
                    hrp_weights=new_hrp,
                    last_hrp_refit_date=for_date.isoformat(),
                    ai_strategy_names=list(state.ai_strategy_names),
                    ai_strategy_shadow_until=dict(state.ai_strategy_shadow_until),
                    trail_high=dict(state.trail_high),
                    trail_pct=state.trail_pct,
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

    # -------- 3. Render + email --------
    if not daily_runs and not refit_notes:
        notes = (
            f"No daily run records found between "
            f"{(for_date - timedelta(days=6)).isoformat()} and "
            f"{for_date.isoformat()}, and HRP refit was skipped."
        )
    else:
        notes = "\n".join(refit_notes) if refit_notes else ""
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
