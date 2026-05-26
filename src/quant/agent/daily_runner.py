"""daily_runner.py — the agent's once-a-day trade routine and report routine.

The two entry points (both also exposed via console-scripts in
pyproject.toml so launchd can invoke them by name):

- ``run_daily_trade()`` — at ~09:35 ET, compute target weights from
  yesterday's close, submit market entries with atomic stop-losses,
  persist the full record.
- ``run_daily_report(for_date=...)`` — at ~16:05 ET, load the day's
  persisted record, render markdown, email it.

Both are deliberately small: most logic lives in the existing modules
(strategy, executor, log, reports, email_sender). This file is the
orchestration glue.

Failure handling
----------------
If anything raises inside the trade routine, the CLI wrapper catches it,
tries to email the operator with the traceback, then re-raises so
launchd notices the non-zero exit and (per OS log) the operator can
investigate. Email failure on top of trade failure is logged to stderr
and does NOT loop into a new error email — that's how cascades start.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quant.agent.email_sender import EmailSender
from quant.agent.log import (
    DEFAULT_RUNS_DIR,
    load_daily_run,
    save_daily_run,
)
from quant.agent.reports import render_daily_report
from quant.backtest.types import Snapshot
from quant.config import Config, load_config
from quant.data.alpaca_client import AlpacaDataClient
from quant.data.cache import BarsCache
from quant.data.universe import load_top100_snapshot
from quant.execution.alpaca_executor import AlpacaExecutor, ExecutionReport
from quant.strategies import CrossSectionalMomentum


logger = logging.getLogger(__name__)


# Agent-level constants — operator's rules, hard-coded. These are NOT in
# the YAML config because they're the *agent's* contract, not the
# backtester's. The YAML config covers backtest defaults; the agent
# enforces these regardless of what gets put in YAML.
STOP_LOSS_PCT = 0.05            # 5% stop on every entry
MAX_POSITION_WEIGHT = 0.20      # 20% per-trade cap
TOP_K = 10                      # 10 names = 10% each, well under the 20% cap
LOOKBACK_DAYS = 60
SKIP_DAYS = 5
LOOKBACK_BUFFER_DAYS = 30       # extra bars fetched to cover non-trading days


# ---------------------------------------------------------------------------
# run_daily_trade — the morning routine
# ---------------------------------------------------------------------------


def run_daily_trade(
    *,
    dry_run: bool = False,
    today: date | None = None,
    config: Config | None = None,
    universe: list[str] | None = None,
    cache: BarsCache | None = None,
    executor: AlpacaExecutor | None = None,
    runs_dir: Path | None = None,
) -> Path:
    """Execute today's trade cycle. Returns the path of the saved JSON log.

    Every dependency can be injected for tests; in production they
    default to the real components.

    Parameters
    ----------
    dry_run
        If True, the executor reports what WOULD be submitted but
        doesn't actually send orders. Used for first install / debug.
    today
        Override "today's date" (default: system date). Useful for
        re-running a missed day or simulating ahead.
    """
    today = today or date.today()
    config = config or load_config()
    universe = universe or load_top100_snapshot()
    cache = cache or BarsCache(client=AlpacaDataClient(), root=Path("data/bars/daily"))
    executor = executor or AlpacaExecutor()

    logger.info(
        "run_daily_trade: today=%s dry_run=%s universe_size=%d env=%s",
        today, dry_run, len(universe), executor.env,
    )

    # --- 1. Fetch bars covering the signal window ---
    # We need at least LOOKBACK_DAYS + SKIP_DAYS bars of history. Fetch
    # extra calendar days to be safe (non-trading days, holidays).
    end = today - timedelta(days=1)   # don't request today's bar (not closed yet)
    start = end - timedelta(
        days=LOOKBACK_DAYS + SKIP_DAYS + LOOKBACK_BUFFER_DAYS + 30
    )
    bars = cache.get_daily_bars(universe, start, end)
    if bars.empty:
        raise RuntimeError(
            f"no bars fetched for universe of {len(universe)} symbols "
            f"between {start} and {end}. Cache or Alpaca issue?"
        )

    # --- 2. Build snapshot, run the strategy ---
    # as_of = the LATEST date with bars in the frame (typically end, but
    # could be earlier if today is right after a long weekend etc.)
    ts = bars.index.get_level_values("timestamp")
    as_of = ts.max().date()
    snapshot = Snapshot.from_full_bars(bars, as_of=as_of)
    strategy = CrossSectionalMomentum(
        universe,
        lookback=LOOKBACK_DAYS,
        skip=SKIP_DAYS,
        top_k=TOP_K,
    )
    target_weights = strategy.on_bar(snapshot)
    logger.info(
        "strategy emitted %d target positions (as_of=%s)",
        len(target_weights), as_of,
    )

    # --- 3. Signal prices = the last available close per target name. ---
    signal_prices: dict[str, float] = {}
    for sym in target_weights:
        try:
            sym_closes = bars.loc[sym]["close"]
            signal_prices[sym] = float(sym_closes.iloc[-1])
        except KeyError:
            # Strategy targeted a name with no bar data — shouldn't
            # happen since the strategy only ranks names it can see,
            # but defensive.
            logger.warning("no signal price for %s; skipping", sym)
    # ALSO include any currently-held names not in targets so the
    # executor can compute their close-out notionals.
    held = executor.get_positions()
    for sym in held:
        if sym in signal_prices:
            continue
        try:
            sym_closes = bars.loc[sym]["close"]
            signal_prices[sym] = float(sym_closes.iloc[-1])
        except KeyError:
            # Held name not in the universe (manual prior trade, etc.).
            # The executor will skip closing it; warn so the operator
            # notices.
            logger.warning(
                "held name %s has no bars in current universe; agent "
                "won't auto-close it. Liquidate manually if desired.",
                sym,
            )

    # --- 4. Submit via the executor's agent flow ---
    report = executor.submit_daily_rebalance(
        target_weights=target_weights,
        signal_prices=signal_prices,
        stop_loss_pct=STOP_LOSS_PCT,
        max_position_weight=MAX_POSITION_WEIGHT,
        dry_run=dry_run,
        notes=f"daily trade {today.isoformat()}"
              + (" (dry-run)" if dry_run else ""),
    )

    # --- 5. Persist the full record. ---
    path = save_daily_run(
        run_date=today,
        strategy_name=strategy.name,
        strategy_params={
            "lookback": LOOKBACK_DAYS,
            "skip": SKIP_DAYS,
            "top_k": TOP_K,
            "stop_loss_pct": STOP_LOSS_PCT,
            "max_position_weight": MAX_POSITION_WEIGHT,
        },
        target_weights=target_weights,
        signal_prices=signal_prices,
        execution_report=report,
        runs_dir=runs_dir,
    )
    logger.info("saved daily run to %s", path)
    return path


# ---------------------------------------------------------------------------
# run_daily_report — the close-of-day email
# ---------------------------------------------------------------------------


def run_daily_report(
    *,
    for_date: date | None = None,
    runs_dir: Path | None = None,
    email_sender: EmailSender | None = None,
) -> str:
    """Render today's report and email it. Returns the subject sent.

    Loads the persisted JSON from earlier in the day, renders markdown,
    emails to the configured REPORT_TO recipient. Raises if no run was
    persisted for ``for_date``.
    """
    for_date = for_date or date.today()
    payload = load_daily_run(for_date, runs_dir=runs_dir)
    if payload is None:
        raise RuntimeError(
            f"no daily run record for {for_date.isoformat()} at "
            f"{runs_dir or DEFAULT_RUNS_DIR}. Did the morning trade routine run?"
        )

    # Reconstitute just enough of ExecutionReport for the renderer.
    # The renderer reads only a subset — order rows, equity, env, etc.
    # We could fully deserialize but a SimpleNamespace is easier.
    er = payload["execution_report"]
    report_view = _ExecutionReportView(
        env=er.get("env", "paper"),
        timestamp=er.get("timestamp", ""),
        account_equity_before=float(er.get("account_equity_before", 0.0)),
        positions_before=er.get("positions_before", {}),
        target_weights=er.get("target_weights", {}),
        proposed_orders=[],   # not rendered in daily report
        submitted_orders=[
            _SubmittedOrderView(**o) for o in er.get("submitted_orders", [])
        ],
        dry_run=bool(er.get("dry_run", False)),
        notes=er.get("notes", ""),
    )

    subject, body = render_daily_report(
        run_date=for_date,
        strategy_name=payload.get("strategy_name", "(unknown)"),
        target_weights=payload.get("target_weights", {}),
        execution_report=report_view,
    )

    sender = email_sender or EmailSender()
    sender.send(subject=subject, body_text=body, body_html=_markdown_to_html(body))
    logger.info("daily report emailed: %s", subject)
    return subject


# ---------------------------------------------------------------------------
# View shims — let the renderer treat the deserialized dict like an
# ExecutionReport without re-running the dataclass machinery.
# ---------------------------------------------------------------------------


class _SubmittedOrderView:
    """Minimal duck-type matching SubmittedOrder for the renderer's needs."""

    def __init__(self, **kw: Any) -> None:
        self.symbol = kw.get("symbol", "")
        self.side = kw.get("side", "buy")
        self.qty = int(kw.get("qty", 0))
        self.status = kw.get("status", "")
        self.alpaca_order_id = kw.get("alpaca_order_id")
        self.role = kw.get("role", "entry")
        self.stop_price = kw.get("stop_price")
        self.error = kw.get("error")


class _ExecutionReportView:
    """Minimal duck-type matching ExecutionReport for the renderer."""

    def __init__(
        self, *, env, timestamp, account_equity_before, positions_before,
        target_weights, proposed_orders, submitted_orders, dry_run, notes,
    ) -> None:
        self.env = env
        self.timestamp = timestamp
        self.account_equity_before = account_equity_before
        self.positions_before = positions_before
        self.target_weights = target_weights
        self.proposed_orders = proposed_orders
        self.submitted_orders = submitted_orders
        self.dry_run = dry_run
        self.notes = notes


# ---------------------------------------------------------------------------
# Markdown -> HTML (trivial — wrap in <pre> for now; v2 can use markdown lib)
# ---------------------------------------------------------------------------


def _markdown_to_html(md: str) -> str:
    """Wrap markdown in <pre> for HTML clients that don't render md natively.

    For v2 we'd use the ``markdown`` package; for now, monospaced preformatted
    text is readable in every mail client without adding a dependency.
    """
    # Minimal HTML-escape to avoid raw markdown breaking the HTML.
    escaped = (
        md.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        '<html><body style="font-family: -apple-system, Segoe UI, sans-serif">'
        f"<pre style='font-size: 13px; line-height: 1.4'>{escaped}</pre>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# CLI entry points (registered as console_scripts in pyproject.toml)
# ---------------------------------------------------------------------------


def _email_failure(routine: str, exc: Exception) -> None:
    """Best-effort failure notification. Never raises."""
    try:
        sender = EmailSender()
        sender.send(
            subject=f"quant agent FAILURE — {routine}",
            body_text=(
                f"The {routine} routine raised an exception.\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                f"Traceback:\n{traceback.format_exc()}\n\n"
                f"Check the agent's logs for context. The launchd job exit "
                f"code will also be non-zero."
            ),
        )
    except Exception as send_err:
        # Never let an email-on-error failure mask the real error.
        # launchd will surface the real one via the non-zero exit.
        print(
            f"[agent] failed to send failure email: {send_err}",
            file=sys.stderr,
        )


def cli_run_trade() -> None:
    """Console-script entry point: ``uv run quant-daily-trade``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the agent's daily trade.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="don't submit orders to Alpaca; report what would happen",
    )
    args = parser.parse_args()
    try:
        path = run_daily_trade(dry_run=args.dry_run)
        print(f"[agent] daily trade complete; log: {path}")
    except Exception as e:
        _email_failure("daily trade", e)
        raise


def cli_run_report() -> None:
    """Console-script entry point: ``uv run quant-daily-report``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Send the agent's daily report.")
    parser.add_argument(
        "--for-date", default=None,
        help="ISO date YYYY-MM-DD; defaults to today",
    )
    args = parser.parse_args()
    for_date = date.fromisoformat(args.for_date) if args.for_date else None
    try:
        subject = run_daily_report(for_date=for_date)
        print(f"[agent] daily report sent: {subject}")
    except Exception as e:
        _email_failure("daily report", e)
        raise
