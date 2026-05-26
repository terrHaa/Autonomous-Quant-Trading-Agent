"""Tests for the weekly and monthly review jobs.

Both inject a fake email sender + temp dirs so no network or files
outside ``tmp_path`` are touched.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from quant.agent import monthly_review as monthly_mod
from quant.agent import weekly_review as weekly_mod
from quant.agent.log import save_daily_run
from quant.agent.params import StrategyParams
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
        env="paper", timestamp=datetime.now(UTC),
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
    # refit_hrp=False so we don't hit Alpaca during unit tests.
    subject = weekly_mod.run_weekly_review(
        for_date=base, runs_dir=tmp_path, email_sender=sender,
        refit_hrp=False,
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
        refit_hrp=False,
    )
    assert len(sender.sent) == 1
    # No-data note should appear in the body.
    assert "no daily run" in sender.sent[0]["body_text"].lower()


def test_weekly_review_refits_hrp_when_enabled(monkeypatch, tmp_path: Path) -> None:
    """With refit_hrp=True and a mocked refit, the new weights must be
    persisted to disk and the email body must surface the change."""
    from quant.agent import weekly_review as weekly_module

    # Pre-save baseline ensemble state.
    from quant.agent.ensemble import (
        EnsembleState,
        load_ensemble_state,
        save_ensemble_state,
    )
    state_path = tmp_path / "state.json"
    save_ensemble_state(EnsembleState(), path=state_path)

    # Mock refit_hrp_weights to return a deliberately new allocation.
    new_weights = {
        "sma_crossover_50_200": 0.6,
        "mean_reversion_5_200bp": 0.1,
        "xsec_momo_60_5_10": 0.3,
    }
    monkeypatch.setattr(
        weekly_module, "refit_hrp_weights",
        lambda *a, **kw: (
            new_weights,
            {
                "per_strategy": {
                    "sma_crossover_50_200": {
                        "total_return": 0.12, "sharpe": 0.8, "n_days": 250,
                    },
                    "mean_reversion_5_200bp": {
                        "total_return": -0.03, "sharpe": -0.2, "n_days": 250,
                    },
                    "xsec_momo_60_5_10": {
                        "total_return": 0.20, "sharpe": 1.1, "n_days": 250,
                    },
                },
                "hrp_weights_before": {},
                "hrp_weights_after": new_weights,
            },
        ),
    )

    # Provide a cache stub that returns non-empty bars (so the refit branch runs).
    class _FakeCache:
        def get_daily_bars(self, *a, **kw):
            return pd.DataFrame({"close": [1, 2, 3]})

    _save_daily(tmp_path, date(2024, 6, 7))   # at least one daily run

    sender = _RecordingEmail()
    weekly_module.run_weekly_review(
        for_date=date(2024, 6, 7),
        runs_dir=tmp_path,
        email_sender=sender,
        state_path=state_path,
        cache=_FakeCache(),
        universe=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )

    # The new weights must have landed on disk.
    loaded = load_ensemble_state(path=state_path)
    assert loaded.hrp_weights == new_weights
    assert loaded.last_hrp_refit_date == "2024-06-07"
    # The email body must surface the change.
    body = sender.sent[0]["body_text"]
    assert "HRP weights" in body
    assert "sma_crossover_50_200" in body


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
    new xsec params must be saved into the EnsembleState file (and the
    other strategies' params plus the HRP weights must be PRESERVED)."""
    from quant.agent.ensemble import load_ensemble_state
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

    class _FakeCache:
        def get_daily_bars(self, *a, **kw):
            return pd.DataFrame({"close": [1, 2, 3]})

    sender = _RecordingEmail()
    state_path = tmp_path / "ensemble_state.json"
    _save_daily(tmp_path, date(2024, 6, 28))

    monthly_mod.run_monthly_review(
        for_date=date(2024, 6, 28),
        runs_dir=tmp_path,
        email_sender=sender,
        cache=_FakeCache(),
        universe=[f"S{i}" for i in range(10)],
        auto_apply=True,
        params_path=state_path,
    )

    # The new xsec params should be on the ensemble state; other fields
    # should be at their default values (untouched).
    loaded = load_ensemble_state(path=state_path)
    assert loaded.xsec_top_k == 5
    assert loaded.xsec_lookback == 120
    assert loaded.xsec_skip == 5
    # Untouched strategies' params remain at defaults.
    assert loaded.sma_fast == 50
    assert loaded.sma_slow == 200
    assert loaded.mr_lookback == 5
    # HRP weights preserved (defaults, since no refit ran in this test).
    assert abs(sum(loaded.hrp_weights.values()) - 1.0) < 1e-9
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
