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
from datetime import date, timedelta
from pathlib import Path

from quant.agent.daily_runner import _email_failure, _markdown_to_html
from quant.agent.email_sender import EmailSender
from quant.agent.ensemble import (
    EnsembleState,
    load_ensemble_state,
    save_ensemble_state,
)
from quant.agent.improver import (
    ImprovementResult,
    search_improvements,
)
from quant.agent.log import DEFAULT_RUNS_DIR, load_daily_run
from quant.agent.params import StrategyParams
from quant.agent.reports import render_monthly_report
from quant.config import Config, load_config
from quant.data.alpaca_client import AlpacaDataClient
from quant.data.cache import BarsCache
from quant.data.universe import load_top50_snapshot

logger = logging.getLogger(__name__)


# How much history to feed the improver's backtests. 2 years is enough
# for ~500 trading days of returns — enough to estimate Sharpe with some
# stability, not so much that ancient regimes dominate.
IMPROVER_BACKTEST_YEARS = 2


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
) -> tuple[str, ImprovementResult | None]:
    """Run the month's review and return (subject, improvement_result).

    ``improvement_result`` is None when the improver step was skipped
    (e.g., test mode where no cache is configured).
    """
    for_date = for_date or date.today()
    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    config = config or load_config()

    # -------- 1. Aggregate the month's daily runs --------
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

    # -------- 2. Improver step --------
    # Load EnsembleState. The improver currently tunes only the
    # cross-sectional momentum sub-strategy; SMA + MR tuning is a future
    # extension. The weekly HRP refit handles the strategy-mix layer.
    current_state = load_ensemble_state(path=params_path)
    current_xsec_params = StrategyParams(
        top_k=current_state.xsec_top_k,
        lookback=current_state.xsec_lookback,
        skip=current_state.xsec_skip,
    )
    universe = universe or load_top50_snapshot()
    improvement_result: ImprovementResult | None = None
    recommendations: list[str] = []

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
        else:
            improvement_result = search_improvements(
                current_xsec_params,
                universe=universe,
                bars=bars,
                config=config,
            )
            recommendations.append(
                f"Improver evaluated {len(improvement_result.candidates)} candidates "
                f"on {len(bars)} bars."
            )
            best = improvement_result.best_passing
            if best is None:
                recommendations.append(
                    f"No change: {improvement_result.reason}"
                )
            else:
                if auto_apply:
                    # Persist the winning xsec params back into the
                    # ensemble state (keep the SMA, MR, and HRP weights
                    # untouched — those are managed elsewhere).
                    new_state = EnsembleState(
                        sma_fast=current_state.sma_fast,
                        sma_slow=current_state.sma_slow,
                        mr_lookback=current_state.mr_lookback,
                        mr_threshold_pct=current_state.mr_threshold_pct,
                        mr_allow_short=current_state.mr_allow_short,
                        xsec_top_k=best.params.top_k,
                        xsec_lookback=best.params.lookback,
                        xsec_skip=best.params.skip,
                        hrp_weights=current_state.hrp_weights,
                        last_hrp_refit_date=current_state.last_hrp_refit_date,
                    )
                    save_ensemble_state(new_state, path=params_path)
                    recommendations.append(
                        f"**APPLIED** xsec momentum params: "
                        f"top_k={best.params.top_k}, "
                        f"lookback={best.params.lookback}, "
                        f"skip={best.params.skip}. "
                        f"Previous: top_k={current_xsec_params.top_k}, "
                        f"lookback={current_xsec_params.lookback}, "
                        f"skip={current_xsec_params.skip}. "
                        f"Gate reason: {improvement_result.reason}"
                    )
                else:
                    recommendations.append(
                        f"Candidate passes all gates but auto-apply is off. "
                        f"Candidate: top_k={best.params.top_k}, "
                        f"lookback={best.params.lookback}, "
                        f"skip={best.params.skip}."
                    )
    except Exception as e:
        # The improver is best-effort; failure doesn't block the email.
        recommendations.append(f"Improver failed: {type(e).__name__}: {e}")
        logger.exception("improver failed")

    # -------- 3. Render + send --------
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
