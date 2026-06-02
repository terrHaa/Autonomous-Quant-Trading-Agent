"""Tests for the agent's daily trade + report routines.

These mock every external dependency (cache, executor, email sender, log
dir) so the tests never touch Alpaca or SMTP. The goal is to verify the
orchestration: the right pieces called in the right order with the right
arguments.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant.agent import daily_runner
from quant.agent.log import save_daily_run
from quant.data.alpaca_client import BAR_COLUMNS
from quant.execution.alpaca_executor import (
    ExecutionReport,
    ProposedOrder,
    SubmittedOrder,
)


@pytest.fixture
def _in_trade_window(monkeypatch):
    """Mark the test as 'inside the trade window' so the 09:00-15:35 ET
    guard doesn't trip on fixed historical test dates.

    NOT autouse — tests that directly call `_outside_trade_window()`
    (the helper unit tests) need the real function. Tests that go
    through `run_daily_trade()` request this fixture explicitly.
    """
    monkeypatch.setattr(daily_runner, "_outside_trade_window", lambda _d: False)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeCache:
    """BarsCache stand-in that returns a hand-crafted bars frame."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars
        self.calls: list = []

    def get_daily_bars(self, symbols, start, end):
        self.calls.append((tuple(symbols), start, end))
        return self._bars


class _FakeExecutor:
    """AlpacaExecutor stand-in that records args and returns a canned report."""

    env = "paper"

    def __init__(self, *, current_positions: dict[str, int] | None = None) -> None:
        self._positions = current_positions or {}
        self.last_call: dict[str, Any] = {}

    def get_positions(self):
        return dict(self._positions)

    def submit_daily_rebalance(self, **kwargs) -> ExecutionReport:
        self.last_call = kwargs
        # Build a fake report capturing the inputs so tests can verify.
        proposed: list[ProposedOrder] = []
        submitted: list[SubmittedOrder] = []
        for sym, weight in kwargs["target_weights"].items():
            price = kwargs["signal_prices"][sym]
            qty = int(weight * 100_000 / price)
            proposed.append(ProposedOrder(
                symbol=sym, side="buy", qty=qty,
                rationale=f"weight {weight}",
            ))
            submitted.append(SubmittedOrder(
                symbol=sym, side="buy", qty=qty,
                status="skipped_dry_run" if kwargs.get("dry_run") else "submitted",
                role="entry",
                alpaca_order_id=f"fake-{sym}",
            ))
            submitted.append(SubmittedOrder(
                symbol=sym, side="sell", qty=qty,
                status="skipped_dry_run" if kwargs.get("dry_run") else "submitted",
                role="stop_loss",
                stop_price=round(price * (1 - kwargs["stop_loss_pct"]), 2),
            ))
        return ExecutionReport(
            env="paper",
            timestamp=datetime.now(UTC),
            account_equity_before=100_000.0,
            positions_before=dict(self._positions),
            target_weights=dict(kwargs["target_weights"]),
            proposed_orders=proposed,
            submitted_orders=submitted,
            dry_run=bool(kwargs.get("dry_run")),
            notes=kwargs.get("notes", ""),
        )


def _make_bars(symbols: list[str], n_days: int = 100, *, trend: float = 0.5) -> pd.DataFrame:
    """Build a bars frame with n_days of business-day data.

    All symbols ramp UP by `trend` per day starting from base price 100.
    Since they all ramp at the same rate, momentum ranking is determined
    by base price differences (which we make distinct per symbol).
    """
    days = pd.bdate_range("2024-01-02", periods=n_days, tz="UTC")
    rows, idx = [], []
    for i, sym in enumerate(symbols):
        base = 100.0 + i  # unique base so ranking is deterministic
        for k, ts in enumerate(days):
            c = base + k * trend
            rows.append({
                "open": c, "high": c + 0.01, "low": c, "close": c, "volume": 1,
            })
            idx.append((sym, ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=list(BAR_COLUMNS),
    )


# ---------------------------------------------------------------------------
# run_daily_trade
# ---------------------------------------------------------------------------


def test_run_daily_trade_is_idempotent_when_record_exists(tmp_path: Path, _in_trade_window) -> None:
    """Re-firing on a day whose run JSON already exists must be a no-op.

    This is what makes launchd KeepAlive safe: after a SUCCESSFUL trade,
    subsequent retries (until tomorrow's scheduled fire) detect the
    existing record and exit cleanly without re-trading. Saturday's
    audit will then see one clean run.
    """
    universe = [f"SYM{i}" for i in range(10)]
    bars = _make_bars(universe, n_days=100)
    cache = _FakeCache(bars)
    executor = _FakeExecutor()

    # First call: produces the JSON.
    first = daily_runner.run_daily_trade(
        today=date(2024, 6, 3),
        universe=universe, cache=cache, executor=executor,
        runs_dir=tmp_path,
    )
    assert first is not None and first.exists()
    first_call_snapshot = dict(executor.last_call)
    # Mutate last_call so we can detect a 2nd invocation by it being overwritten.
    executor.last_call = {"_idempotency_marker": True}

    # Second call (simulates launchd KeepAlive retry).
    second = daily_runner.run_daily_trade(
        today=date(2024, 6, 3),
        universe=universe, cache=cache, executor=executor,
        runs_dir=tmp_path,
    )
    # Idempotent skip → no new path, no new orders.
    assert second is None
    assert executor.last_call == {"_idempotency_marker": True}, (
        "executor should NOT have been called again on the second run"
    )
    assert first_call_snapshot, "first call should have hit the executor"


def test_run_daily_trade_skips_outside_window(
    tmp_path: Path, monkeypatch
) -> None:
    """If wall-clock is outside 09:00-15:35 ET on the trade day, the routine
    exits early without trading. Covers both ends:
      - early off-schedule launchd KeepAlive fire (e.g. plist reload at 23:00 ET)
      - late KeepAlive-loop kill switch (after 15:35 ET).
    """
    monkeypatch.setattr(
        daily_runner, "_outside_trade_window", lambda _d: True,
    )
    universe = [f"SYM{i}" for i in range(5)]
    cache = _FakeCache(_make_bars(universe, n_days=10))
    executor = _FakeExecutor()
    result = daily_runner.run_daily_trade(
        today=date(2024, 6, 3),
        universe=universe, cache=cache, executor=executor,
        runs_dir=tmp_path,
    )
    assert result is None
    # No persistence happened.
    assert not (tmp_path / "2024-06-03.json").exists()
    # No broker calls.
    assert executor.last_call == {}


def test_outside_trade_window_inside_returns_false(monkeypatch) -> None:
    """Helper unit test: 10:00 ET on the trade date is inside the window."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class _FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 6, 3, 10, 0, tzinfo=ZoneInfo("America/New_York"))

    import datetime as _dt
    real_dt = _dt.datetime
    _dt.datetime = _FakeDatetime   # type: ignore[misc]
    try:
        assert daily_runner._outside_trade_window(date(2024, 6, 3)) is False
    finally:
        _dt.datetime = real_dt   # type: ignore[misc]


def test_outside_trade_window_too_early_returns_true(monkeypatch) -> None:
    """05:00 ET on the trade date is BEFORE the 09:00 window open → out."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class _FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 6, 3, 5, 0, tzinfo=ZoneInfo("America/New_York"))

    import datetime as _dt
    real_dt = _dt.datetime
    _dt.datetime = _FakeDatetime   # type: ignore[misc]
    try:
        assert daily_runner._outside_trade_window(date(2024, 6, 3)) is True
    finally:
        _dt.datetime = real_dt   # type: ignore[misc]


def test_outside_trade_window_too_late_returns_true(monkeypatch) -> None:
    """16:00 ET on the trade date is AFTER the 15:35 deadline → out."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class _FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 6, 3, 16, 0, tzinfo=ZoneInfo("America/New_York"))

    import datetime as _dt
    real_dt = _dt.datetime
    _dt.datetime = _FakeDatetime   # type: ignore[misc]
    try:
        assert daily_runner._outside_trade_window(date(2024, 6, 3)) is True
    finally:
        _dt.datetime = real_dt   # type: ignore[misc]


def test_run_daily_trade_persists_a_json_record(tmp_path: Path, _in_trade_window) -> None:
    """The runner must write data/agent/runs/<date>.json so the report
    routine can find it later in the day."""
    universe = [f"SYM{i}" for i in range(20)]
    bars = _make_bars(universe, n_days=100)
    cache = _FakeCache(bars)
    executor = _FakeExecutor()

    path = daily_runner.run_daily_trade(
        today=date(2024, 6, 1),
        universe=universe,
        cache=cache,
        executor=executor,
        runs_dir=tmp_path,
        dry_run=True,
    )
    assert path.exists()
    assert path.name == "2024-06-01.json"

    # File contents have the expected top-level keys.
    payload = json.loads(path.read_text())
    for key in ("date", "strategy_name", "target_weights", "signal_prices",
                "execution_report"):
        assert key in payload


def test_run_daily_trade_passes_operator_constants_to_executor(tmp_path: Path, _in_trade_window) -> None:
    """20% cap and 5% stop-loss are HARD-CODED at the agent level —
    verify they flow through to the executor call regardless of
    upstream config defaults."""
    universe = [f"SYM{i}" for i in range(20)]
    bars = _make_bars(universe, n_days=100)
    cache = _FakeCache(bars)
    executor = _FakeExecutor()

    daily_runner.run_daily_trade(
        today=date(2024, 6, 1),
        universe=universe,
        cache=cache,
        executor=executor,
        runs_dir=tmp_path,
        dry_run=True,
    )
    assert executor.last_call["stop_loss_pct"] == daily_runner.STOP_LOSS_PCT == 0.05
    assert executor.last_call["max_position_weight"] == daily_runner.MAX_POSITION_WEIGHT == 0.20


def test_run_daily_trade_includes_held_names_in_signal_prices(tmp_path: Path, _in_trade_window) -> None:
    """If we currently hold a name not in the target book, the runner
    must still pass its signal price so the executor can size the
    close-out order."""
    universe = [f"SYM{i}" for i in range(20)]
    bars = _make_bars(universe, n_days=100)
    cache = _FakeCache(bars)
    # Hold a name the strategy WON'T target (SYM0 is the lowest-momentum).
    # Actually the strategy picks the TOP 10 by momentum; SYM19 is highest.
    # We hold SYM5 which is below the median, so it won't make the cut.
    executor = _FakeExecutor(current_positions={"SYM5": 100})

    daily_runner.run_daily_trade(
        today=date(2024, 6, 1),
        universe=universe,
        cache=cache,
        executor=executor,
        runs_dir=tmp_path,
        dry_run=True,
    )
    # SYM5 should be in the signal_prices passed to the executor.
    assert "SYM5" in executor.last_call["signal_prices"]


def test_run_daily_trade_raises_on_empty_bars(tmp_path: Path, _in_trade_window) -> None:
    """Empty bars (cache or Alpaca outage) → hard failure, no silent
    'we just won't trade' behavior."""
    empty = pd.DataFrame(
        columns=list(BAR_COLUMNS),
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )
    cache = _FakeCache(empty)
    executor = _FakeExecutor()
    with pytest.raises(RuntimeError, match="no bars"):
        daily_runner.run_daily_trade(
            today=date(2024, 6, 1),
            universe=["AAPL", "MSFT"],
            cache=cache,
            executor=executor,
            runs_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# run_daily_report
# ---------------------------------------------------------------------------


class _RecordingEmailSender:
    """EmailSender stand-in that captures send() calls."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, *, subject, body_text, body_html=None, recipient=None):
        self.sent.append({
            "subject": subject,
            "body_text": body_text,
            "body_html": body_html,
            "recipient": recipient,
        })


def test_run_daily_report_emails_the_persisted_log(tmp_path: Path) -> None:
    """Render-and-email loads the saved JSON and produces a subject + body."""
    # Pre-populate one day's record via the same save function.
    fake_exec_report = ExecutionReport(
        env="paper",
        timestamp=datetime.now(UTC),
        account_equity_before=100_000.0,
        positions_before={},
        target_weights={"AAPL": 0.1, "MSFT": 0.1},
        proposed_orders=[],
        submitted_orders=[
            SubmittedOrder(symbol="AAPL", side="buy", qty=50,
                           status="submitted", role="entry"),
            SubmittedOrder(symbol="AAPL", side="sell", qty=50,
                           status="submitted", role="stop_loss",
                           stop_price=185.0),
        ],
        dry_run=False,
        notes="test",
    )
    save_daily_run(
        run_date=date(2024, 6, 1),
        strategy_name="xsec_momo_60_5_10",
        strategy_params={},
        target_weights={"AAPL": 0.1, "MSFT": 0.1},
        signal_prices={"AAPL": 195.0, "MSFT": 405.0},
        execution_report=fake_exec_report,
        runs_dir=tmp_path,
    )

    sender = _RecordingEmailSender()
    subject = daily_runner.run_daily_report(
        for_date=date(2024, 6, 1),
        runs_dir=tmp_path,
        email_sender=sender,  # type: ignore[arg-type]
    )
    assert len(sender.sent) == 1
    assert subject.startswith("quant agent")
    assert "AAPL" in sender.sent[0]["body_text"]


def test_run_daily_report_raises_when_no_log_exists(tmp_path: Path) -> None:
    """Missing log = the morning routine failed. Surface it loudly."""
    sender = _RecordingEmailSender()
    with pytest.raises(RuntimeError, match="no daily run record"):
        daily_runner.run_daily_report(
            for_date=date(2024, 6, 1),
            runs_dir=tmp_path,
            email_sender=sender,  # type: ignore[arg-type]
        )
