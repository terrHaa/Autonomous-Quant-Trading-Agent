"""Tests for the weekly and monthly review jobs.

Both inject a fake email sender + temp dirs so no network or files
outside ``tmp_path`` are touched.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant.agent import monthly_review as monthly_mod
from quant.agent import weekly_review as weekly_mod
from quant.agent.log import save_daily_run
from quant.agent.params import StrategyParams, load_params, save_params
from quant.execution.alpaca_executor import (
    ExecutionReport,
    SubmittedOrder,
)


class _RecordingEmail:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, *, subject, body_text, body_html=None, recipient=None):
        self.sent.append({
            "subject": subject, "body_text": body_text,
            "body_html": body_html, "recipient": recipient,
        })


def _save_daily(tmp_path: Path, d: date, equity: float = 100_000.0) -> None:
    """Helper: persist a minimal daily run record."""
    rep = ExecutionReport(
        env="paper", timestamp=datetime.now(timezone.utc),
        account_equity_before=equity, positions_before={},
        target_weights={}, proposed_orders=[],
        submitted_orders=[
            SubmittedOrder(symbol="X", side="buy", qty=1,
                           status="submitted", role="entry"),
        ],
        dry_run=False, notes="",
    )
    save_daily_run(
        run_date=d, strategy_name="xsec_momo_60_5_10",
        strategy_params={}, target_weights={}, signal_prices={},
        execution_report=rep, runs_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# Weekly review
# ---------------------------------------------------------------------------


def test_weekly_review_emails_when_runs_exist(tmp_path: Path) -> None:
    """Five days of records → weekly email goes out with aggregate stats."""
    base = date(2024, 6, 7)   # Friday
    for offset in range(5):   # Mon-Fri
        _save_daily(tmp_path, base - pd.Timedelta(days=offset).to_pytimedelta())

    sender = _RecordingEmail()
    subject = weekly_mod.run_weekly_review(
        for_date=base, runs_dir=tmp_path, email_sender=sender,
    )
    assert len(sender.sent) == 1
    assert "weekly review" in subject
    # Body should mention the agent.
    assert "quant agent" in sender.sent[0]["body_text"].lower()


def test_weekly_review_still_emails_when_no_runs(tmp_path: Path) -> None:
    """No data → still ship an email so the operator knows the job ran."""
    sender = _RecordingEmail()
    weekly_mod.run_weekly_review(
        for_date=date(2024, 6, 7), runs_dir=tmp_path, email_sender=sender,
    )
    assert len(sender.sent) == 1
    # No-data note should appear in the body.
    assert "no daily run" in sender.sent[0]["body_text"].lower()


# ---------------------------------------------------------------------------
# Monthly review
# ---------------------------------------------------------------------------


def test_monthly_review_runs_improver_and_emails(monkeypatch, tmp_path: Path) -> None:
    """Monthly review must:
    1. Load the month's runs.
    2. Invoke the improver.
    3. Email a report.
    With a mocked improver (no real backtest), end-to-end.
    """
    # Pre-populate a few daily runs.
    for offset in range(20):
        _save_daily(tmp_path, date(2024, 6, 28) - pd.Timedelta(days=offset).to_pytimedelta())

    # Mock the cache (return empty bars to short-circuit the improver path).
    class _EmptyCache:
        def get_daily_bars(self, *a, **kw):
            return pd.DataFrame()
    sender = _RecordingEmail()

    subject, _improvement = monthly_mod.run_monthly_review(
        for_date=date(2024, 6, 28),
        runs_dir=tmp_path,
        email_sender=sender,
        cache=_EmptyCache(),
        universe=["AAPL"],
        params_path=tmp_path / "params.json",
    )
    assert len(sender.sent) == 1
    assert "monthly review" in subject
    # The recommendations block should mention either "skipped" or "no change".
    body = sender.sent[0]["body_text"].lower()
    assert "improver" in body or "recommendation" in body


def test_monthly_review_auto_apply_persists_new_params_when_gates_pass(
    monkeypatch, tmp_path: Path,
) -> None:
    """When the improver returns a winner AND auto_apply=True, the
    new params must be saved to disk."""
    from quant.agent.improver import (
        ImprovementCandidate,
        ImprovementResult,
    )

    new_params = StrategyParams(top_k=5, lookback=120, skip=5)
    winner = ImprovementCandidate(
        params=new_params, sharpe=1.2, max_drawdown=-0.08,
        total_return=0.40, n_days=500,
    )
    current = ImprovementCandidate(
        params=StrategyParams(top_k=10, lookback=60, skip=5),
        sharpe=0.3, max_drawdown=-0.10, total_return=0.10, n_days=500,
    )
    fake_result = ImprovementResult(
        current=current, candidates=[current, winner],
        best_passing=winner, reason="DSR 0.99 >= 0.95",
    )

    monkeypatch.setattr(
        monthly_mod, "search_improvements",
        lambda *a, **kw: fake_result,
    )

    # Cache must return non-empty so the improver is invoked.
    class _FakeCache:
        def get_daily_bars(self, *a, **kw):
            return pd.DataFrame({"close": [1, 2, 3]})   # non-empty
    sender = _RecordingEmail()
    params_path = tmp_path / "params.json"

    _save_daily(tmp_path, date(2024, 6, 28))   # at least one run for content

    monthly_mod.run_monthly_review(
        for_date=date(2024, 6, 28),
        runs_dir=tmp_path,
        email_sender=sender,
        cache=_FakeCache(),
        universe=["AAPL"],
        auto_apply=True,
        params_path=params_path,
    )

    # The new params should have been persisted.
    loaded = load_params(path=params_path)
    assert loaded == new_params
    # Email body should mention the apply.
    assert "APPLIED" in sender.sent[0]["body_text"]


def test_monthly_review_no_apply_does_not_persist_even_if_gate_passes(
    monkeypatch, tmp_path: Path,
) -> None:
    """auto_apply=False → improver runs, gate may pass, but params stay."""
    from quant.agent.improver import (
        ImprovementCandidate,
        ImprovementResult,
    )

    winner = ImprovementCandidate(
        params=StrategyParams(top_k=5, lookback=120, skip=5),
        sharpe=1.2, max_drawdown=-0.08, total_return=0.40, n_days=500,
    )
    current = ImprovementCandidate(
        params=StrategyParams(top_k=10, lookback=60, skip=5),
        sharpe=0.3, max_drawdown=-0.10, total_return=0.10, n_days=500,
    )
    fake_result = ImprovementResult(
        current=current, candidates=[current, winner],
        best_passing=winner, reason="gates pass",
    )
    monkeypatch.setattr(
        monthly_mod, "search_improvements",
        lambda *a, **kw: fake_result,
    )

    class _FakeCache:
        def get_daily_bars(self, *a, **kw):
            return pd.DataFrame({"close": [1, 2, 3]})
    sender = _RecordingEmail()
    params_path = tmp_path / "params.json"
    _save_daily(tmp_path, date(2024, 6, 28))

    monthly_mod.run_monthly_review(
        for_date=date(2024, 6, 28),
        runs_dir=tmp_path,
        email_sender=sender,
        cache=_FakeCache(),
        universe=["AAPL"],
        auto_apply=False,
        params_path=params_path,
    )

    # No params file should exist — we didn't save.
    assert not params_path.exists()
    # Email body should explain the manual step.
    body = sender.sent[0]["body_text"]
    assert "auto-apply is off" in body or "manually" in body
