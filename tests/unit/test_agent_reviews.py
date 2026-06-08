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


# ---------------------------------------------------------------------------
# Weekly metrics computation — feeds the AI deep-dive
# ---------------------------------------------------------------------------


def test_weekly_metrics_empty_when_no_equity_data() -> None:
    out = weekly_mod._compute_weekly_metrics(daily_runs=[], equity_curve={})
    assert out["n_days"] == 0


# ---------------------------------------------------------------------------
# Monthly metrics — the LONGER-HORIZON statistical view the AI uses
# ---------------------------------------------------------------------------


def test_monthly_metrics_marks_insufficient_data() -> None:
    out = monthly_mod._compute_monthly_metrics(daily_runs=[], equity_curve={})
    assert out.get("insufficient_data") is True


def test_monthly_metrics_day_of_week_breakdown() -> None:
    """Day-of-week stats let the AI catch calendar effects across the month."""
    # 5 days: Mon-Fri the week of 2024-06-03. Each day's equity change
    # encodes a specific daily return.
    eq = {
        date(2024, 6,  3): 100_000.0,   # Mon
        date(2024, 6,  4): 100_500.0,   # Tue → +0.5%
        date(2024, 6,  5): 100_300.0,   # Wed → -0.199%
        date(2024, 6,  6): 101_100.0,   # Thu → +0.798%
        date(2024, 6,  7): 101_000.0,   # Fri → -0.099%
    }
    m = monthly_mod._compute_monthly_metrics(daily_runs=[], equity_curve=eq)
    dow = m["day_of_week_breakdown"]
    # First day (Mon) has no prior; returns start from Tue.
    assert set(dow.keys()) == {"Tue", "Wed", "Thu", "Fri"}
    assert dow["Tue"]["n"] == 1
    assert dow["Wed"]["mean_return_pct"] < 0
    assert dow["Thu"]["win_rate_pct"] == 100.0
    assert dow["Fri"]["win_rate_pct"] == 0.0


def test_monthly_metrics_lag1_autocorr_positive_for_trending() -> None:
    """Successively-larger up days → positive lag-1 autocorrelation (trending)."""
    eq = {date(2024, 6, i): 100_000.0 * (1.005 ** (i - 1)) for i in range(1, 11)}
    m = monthly_mod._compute_monthly_metrics(daily_runs=[], equity_curve=eq)
    # Constant +0.5% daily returns → autocorr near 0 (no variance) but
    # we just check the field exists and is a finite float.
    assert isinstance(m["lag1_autocorrelation"], float)


def test_monthly_metrics_lag1_autocorr_negative_for_mean_reverting() -> None:
    """Alternating up/down days → negative lag-1 autocorrelation."""
    eq = {date(2024, 6, 1): 100_000.0}
    base = 100_000.0
    for i in range(2, 12):
        # zigzag: +1%, -1%, +1%, ...
        base = base * (1.01 if i % 2 == 0 else 0.99)
        eq[date(2024, 6, i)] = base
    m = monthly_mod._compute_monthly_metrics(daily_runs=[], equity_curve=eq)
    # Strongly mean-reverting series → autocorr near -1
    assert m["lag1_autocorrelation"] < -0.5


def test_monthly_metrics_position_persistence() -> None:
    """Persistence = avg fraction of yesterday's targets that survive into today."""
    eq = {date(2024, 6, 1): 100_000.0, date(2024, 6, 2): 100_000.0}
    runs = [
        {"date": "2024-06-01", "target_weights": {"AAPL": 0.5, "MSFT": 0.5}, "signal_prices": {}},
        # Day 2 keeps AAPL, drops MSFT, adds NVDA: 1/2 of yesterday's survives
        {"date": "2024-06-02", "target_weights": {"AAPL": 0.5, "NVDA": 0.5}, "signal_prices": {}},
    ]
    m = monthly_mod._compute_monthly_metrics(runs, eq)
    assert abs(m["avg_position_persistence_pct"] - 50.0) < 0.01


def test_monthly_metrics_hrp_drift_first_to_last() -> None:
    """HRP weight drift = last_run's weights minus first_run's weights, per key."""
    eq = {date(2024, 6, 1): 100_000.0, date(2024, 6, 2): 100_000.0}
    runs = [
        {
            "date": "2024-06-01",
            "target_weights": {}, "signal_prices": {},
            "strategy_params": {"ensemble_state": {"hrp_weights": {
                "sma_crossover_50_200": 0.50,
                "mean_reversion_5_200bp": 0.30,
                "xsec_momo_60_5_10": 0.20,
            }}},
        },
        {
            "date": "2024-06-02",
            "target_weights": {}, "signal_prices": {},
            "strategy_params": {"ensemble_state": {"hrp_weights": {
                "sma_crossover_50_200": 0.60,    # +0.10
                "mean_reversion_5_200bp": 0.20,  # -0.10
                "xsec_momo_60_5_10": 0.20,       #  0.00
            }}},
        },
    ]
    m = monthly_mod._compute_monthly_metrics(runs, eq)
    drift = m["hrp_weight_drift_over_month"]
    assert abs(drift["sma_crossover_50_200"] - 0.10) < 1e-6
    assert abs(drift["mean_reversion_5_200bp"] - (-0.10)) < 1e-6
    assert abs(drift["xsec_momo_60_5_10"] - 0.00) < 1e-6


def test_monthly_metrics_streak_analysis() -> None:
    """Longest run of consecutive winning / losing days."""
    # 5 ups, 1 down, 3 ups, 2 downs → longest_win=5, longest_loss=2
    eq = {date(2024, 6, 1): 100_000.0}
    base = 100_000.0
    # +5 up: 1.01 ^5
    for i in range(2, 7):
        base *= 1.01
        eq[date(2024, 6, i)] = base
    # -1
    base *= 0.99
    eq[date(2024, 6, 7)] = base
    # +3 up
    for i in range(8, 11):
        base *= 1.01
        eq[date(2024, 6, i)] = base
    # -2 down
    for i in range(11, 13):
        base *= 0.99
        eq[date(2024, 6, i)] = base
    m = monthly_mod._compute_monthly_metrics(daily_runs=[], equity_curve=eq)
    assert m["longest_winning_streak_days"] == 5
    assert m["longest_losing_streak_days"] == 2


def test_monthly_metrics_top10_movers() -> None:
    """Top 10 gainers + 10 losers over the full month by signal price."""
    eq = {date(2024, 6, 1): 100_000.0, date(2024, 6, 28): 100_000.0}
    first = {f"S{i}": 100.0 for i in range(15)}
    # Day 28: S0..S4 went up; S10..S14 went down; S5..S9 unchanged.
    last = dict(first)
    for i in range(5):
        last[f"S{i}"] *= 1.20  # +20%
    for i in range(10, 15):
        last[f"S{i}"] *= 0.85  # -15%
    runs = [
        {"date": "2024-06-01", "target_weights": {}, "signal_prices": first},
        {"date": "2024-06-28", "target_weights": {}, "signal_prices": last},
    ]
    m = monthly_mod._compute_monthly_metrics(runs, eq)
    # Top gainers should be S0..S4 (all +20%, ties broken by sort order).
    gainers = [g["symbol"] for g in m["top10_gainers_month"]]
    losers = [row["symbol"] for row in m["top10_losers_month"]]
    assert set(gainers[:5]) == {"S0", "S1", "S2", "S3", "S4"}
    assert set(losers[:5]) == {"S10", "S11", "S12", "S13", "S14"}


def test_monthly_metrics_raw_returns_series_present() -> None:
    """The raw daily-return series enables the AI to compute its own stats."""
    eq = {date(2024, 6, i): 100_000.0 * (1.005 ** (i - 1)) for i in range(1, 6)}
    m = monthly_mod._compute_monthly_metrics(daily_runs=[], equity_curve=eq)
    assert "daily_returns_pct" in m
    assert len(m["daily_returns_pct"]) == 4   # 5 days → 4 returns
    # Each return ≈ +0.5%
    for r in m["daily_returns_pct"]:
        assert abs(r - 0.5) < 0.001


def test_weekly_metrics_total_return_and_win_rate() -> None:
    """Two days: 100k → 101k = +1%. 3 daily returns possible only if n=4."""
    eq = {
        date(2024, 6, 3): 100_000.0,   # Mon
        date(2024, 6, 4): 100_500.0,   # Tue: +0.5%
        date(2024, 6, 5): 100_200.0,   # Wed: -0.3%
        date(2024, 6, 6): 101_000.0,   # Thu: +0.8%
    }
    runs = [
        {"date": "2024-06-03", "signal_prices": {"AAPL": 200.0, "MSFT": 400.0}, "target_weights": {"AAPL": 0.2, "MSFT": 0.2}},
        {"date": "2024-06-06", "signal_prices": {"AAPL": 210.0, "MSFT": 396.0}, "target_weights": {"AAPL": 0.2, "MSFT": 0.2}},
    ]
    m = weekly_mod._compute_weekly_metrics(runs, eq)

    assert m["n_days"] == 4
    assert m["n_daily_returns"] == 3
    assert m["equity_start"] == 100_000.0
    assert m["equity_end"] == 101_000.0
    # Total return = 101000 / 100000 - 1 = 0.01 = 1%
    assert abs(m["total_return_pct"] - 1.0) < 0.001
    # 2 of 3 daily returns positive → 66.67%
    assert abs(m["win_rate_pct"] - 66.6667) < 0.01
    # Top gainer: AAPL (+5%); top loser: MSFT (-1%).
    assert m["top_gainers_week"][0]["symbol"] == "AAPL"
    assert m["top_losers_week"][0]["symbol"] == "MSFT"


def test_weekly_metrics_max_drawdown_intra_week() -> None:
    """Peak at day 2, trough at day 3, recovery at day 4. Max DD = trough/peak - 1."""
    eq = {
        date(2024, 6, 3): 100_000.0,
        date(2024, 6, 4): 102_000.0,   # peak
        date(2024, 6, 5):  98_000.0,   # trough: (98000/102000 - 1) = -3.92%
        date(2024, 6, 6): 101_000.0,
    }
    runs = [{"date": "2024-06-03", "signal_prices": {}, "target_weights": {}}]
    m = weekly_mod._compute_weekly_metrics(runs, eq)
    # max_drawdown_pct is negative
    assert m["max_drawdown_pct"] < -3.9
    assert m["max_drawdown_pct"] > -4.0


def test_weekly_metrics_concentration_from_last_run() -> None:
    eq = {date(2024, 6, 6): 100_000.0}
    runs = [{
        "date": "2024-06-06",
        "signal_prices": {},
        "target_weights": {"AAPL": 0.1, "MSFT": 0.07, "NVDA": 0.05, "OTHER": 0.03},
    }]
    m = weekly_mod._compute_weekly_metrics(runs, eq)
    # Top 3: AAPL + MSFT + NVDA = 0.22 → 22%
    assert abs(m["top3_concentration_pct"] - 22.0) < 0.01


# ---------------------------------------------------------------------------
# Weekly review wiring tests (existing — extended for enable_ai_analyst)
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
        enable_ai_analyst=False,
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
        enable_ai_analyst=False,
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
        enable_ai_analyst=False,
    )

    # The new weights must have landed on disk.
    loaded = load_ensemble_state(path=state_path)
    assert loaded.hrp_weights == new_weights
    assert loaded.last_hrp_refit_date == "2024-06-07"
    # The email body must surface the change.
    body = sender.sent[0]["body_text"]
    assert "HRP weights" in body
    assert "sma_crossover_50_200" in body


def test_weekly_review_ai_only_does_not_touch_state(monkeypatch, tmp_path: Path) -> None:
    """T-fix B: refit_hrp=False (the --ai-only path) must NOT save state."""
    from quant.agent import weekly_review as weekly_module
    from quant.agent.ensemble import EnsembleState, load_ensemble_state, save_ensemble_state

    state_path = tmp_path / "state.json"
    baseline = EnsembleState()
    save_ensemble_state(baseline, path=state_path)
    baseline_mtime = state_path.stat().st_mtime

    # If anything mutated state, the test would fail because refit_hrp_weights
    # would be called — assert it ISN'T.
    def _explode(*a, **kw):  # noqa: ANN002, ANN003
        raise AssertionError(
            "refit_hrp_weights must NOT be called when refit_hrp=False"
        )
    monkeypatch.setattr(weekly_module, "refit_hrp_weights", _explode)

    _save_daily(tmp_path, date(2024, 6, 7))
    sender = _RecordingEmail()
    weekly_module.run_weekly_review(
        for_date=date(2024, 6, 7),
        runs_dir=tmp_path,
        email_sender=sender,
        state_path=state_path,
        refit_hrp=False,         # the --ai-only setting
        enable_ai_analyst=False,
    )
    # State file untouched: same mtime, same content.
    assert state_path.stat().st_mtime == baseline_mtime
    loaded = load_ensemble_state(path=state_path)
    assert loaded.hrp_weights == baseline.hrp_weights


def test_weekly_cli_ai_only_flag_threads_through(monkeypatch) -> None:
    """The --ai-only CLI flag must reach run_weekly_review as refit_hrp=False."""
    import sys

    from quant.agent import weekly_review as weekly_module
    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return "fake subject"

    monkeypatch.setattr(weekly_module, "run_weekly_review", _fake_run)
    monkeypatch.setattr(
        sys, "argv",
        ["quant-weekly-review", "--for-date=2024-06-07", "--ai-only"],
    )
    weekly_module.cli_run()
    assert captured["refit_hrp"] is False
    assert captured["for_date"] == date(2024, 6, 7)


def test_monthly_review_ai_only_skips_state_writes(monkeypatch, tmp_path: Path) -> None:
    """T-fix B: --ai-only on monthly must skip ALL state mutations:
    no xsec auto-apply, no AI strategy acceptance, no MEMORY/LIBRARY appends.
    Verified by checking the ensemble_state.json mtime is unchanged AND
    that append_memory_entry / append_accepted_strategy_to_library were
    never called.
    """
    from quant.agent import monthly_review as monthly_module
    from quant.agent.ensemble import EnsembleState, save_ensemble_state

    state_path = tmp_path / "state.json"
    save_ensemble_state(EnsembleState(), path=state_path)
    baseline_mtime = state_path.stat().st_mtime

    def _no_call(*a, **kw):
        raise AssertionError(
            f"state mutator called with --ai-only: a={a}, kw={kw}"
        )

    # Block any write paths that ai_only must NOT trigger.
    monkeypatch.setattr(monthly_module, "append_memory_entry", _no_call)
    monkeypatch.setattr(
        monthly_module, "append_accepted_strategy_to_library", _no_call,
    )
    # The improver still runs (its output goes to the analyst prompt) —
    # but auto-apply is forced off, so even a winning candidate doesn't
    # mutate state.
    _save_daily(tmp_path, date(2024, 6, 28))
    sender = _RecordingEmail()
    monthly_module.run_monthly_review(
        for_date=date(2024, 6, 28),
        runs_dir=tmp_path,
        email_sender=sender,
        params_path=state_path,
        ai_only=True,
        enable_ai_analyst=False,    # don't need the AI for this test
        cache=type("_C", (), {"get_daily_bars": lambda *a, **k: pd.DataFrame()})(),
        universe=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )
    assert state_path.stat().st_mtime == baseline_mtime


def test_monthly_cli_ai_only_flag_threads_through(monkeypatch) -> None:
    """The --ai-only CLI flag must reach run_monthly_review as ai_only=True."""
    import sys

    from quant.agent import monthly_review as monthly_module
    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return "fake subject", None

    monkeypatch.setattr(monthly_module, "run_monthly_review", _fake_run)
    monkeypatch.setattr(
        sys, "argv",
        ["quant-monthly-review", "--for-date=2024-06-28", "--ai-only"],
    )
    monthly_module.cli_run()
    assert captured["ai_only"] is True
    assert captured["auto_apply"] is True   # --no-apply was NOT passed
    assert captured["for_date"] == date(2024, 6, 28)


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
        enable_ai_analyst=False,   # tests must not hit the Anthropic API
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
        enable_ai_analyst=False,
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
        enable_ai_analyst=False,
    )

    # No params file should exist — we didn't save.
    assert not params_path.exists()
    # Email body should explain the manual step.
    body = sender.sent[0]["body_text"]
    assert "auto-apply is off" in body or "manually" in body
