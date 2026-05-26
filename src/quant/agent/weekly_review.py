"""weekly_review.py — Friday-after-close summary email.

Aggregates this week's daily run JSONs and emails a markdown summary.
No auto-apply: the weekly review is observational. Parameter changes
happen monthly.

Console-script: ``quant-weekly-review`` (registered in pyproject.toml).
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

from quant.agent.daily_runner import _email_failure, _markdown_to_html
from quant.agent.email_sender import EmailSender
from quant.agent.log import DEFAULT_RUNS_DIR, list_recent_runs, load_daily_run
from quant.agent.reports import render_weekly_report


logger = logging.getLogger(__name__)


def run_weekly_review(
    *,
    for_date: date | None = None,
    runs_dir: Path | None = None,
    email_sender: EmailSender | None = None,
) -> str:
    """Load the week's daily runs, render summary, email it. Returns subject."""
    for_date = for_date or date.today()
    runs_dir = runs_dir or DEFAULT_RUNS_DIR

    # Walk back 7 calendar days to capture Mon-Fri of the current week.
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

    if not daily_runs:
        # No data this week — still send an email so the operator knows
        # the review job ran, but with a clear "no data" note.
        notes = (
            f"No daily run records found between "
            f"{(for_date - timedelta(days=6)).isoformat()} and "
            f"{for_date.isoformat()}. Either the agent didn't run, or "
            f"the runs directory is misconfigured."
        )
        subject, body = render_weekly_report(
            week_ending=for_date, daily_runs=[], notes=notes,
        )
    else:
        subject, body = render_weekly_report(
            week_ending=for_date,
            daily_runs=daily_runs,
            equity_curve=equity_curve,
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
