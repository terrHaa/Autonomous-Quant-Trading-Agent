"""Tests for the end-of-day pipeline audit.

We unit-test each check function in isolation with fakes for the broker
and disk, plus an integration test of the orchestrator with all checks
mocked. The actual broker round-trip is covered separately by the live
broker-reconciliation script the operator can run manually.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from quant.agent.audit import (
    AuditCheck,
    AuditReport,
    _check_alpaca_connectivity,
    _check_broker_reconciliation,
    _check_ensemble_state,
    _check_recent_error_logs,
    _check_run_record,
    _render_email,
    run_daily_audit,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_position(symbol: str, qty: int) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, qty=qty)


def _fake_order(
    symbol: str,
    *,
    qty: int = 1,
    order_type: str = "stop",
    tif: str = "gtc",
    side: str = "sell",
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        qty=qty,
        order_type=SimpleNamespace(value=order_type),
        time_in_force=SimpleNamespace(value=tif),
        side=SimpleNamespace(value=side),
        status=SimpleNamespace(value="accepted"),
        stop_price=0.0,
    )


class _FakeBrokerClient:
    """Minimal fake of alpaca TradingClient for the reconciliation check."""

    def __init__(
        self,
        *,
        equity: float = 100_000.0,
        status: str = "ACTIVE",
        trading_blocked: bool = False,
        positions: dict[str, int] | None = None,
        open_orders: list[SimpleNamespace] | None = None,
        raise_on_account: Exception | None = None,
    ):
        self._equity = equity
        self._status = status
        self._trading_blocked = trading_blocked
        self._positions = positions or {}
        self._orders = open_orders or []
        self._raise = raise_on_account

    def get_account(self):
        if self._raise is not None:
            raise self._raise
        return SimpleNamespace(
            equity=self._equity,
            status=SimpleNamespace(value=self._status),
            trading_blocked=self._trading_blocked,
        )

    def get_all_positions(self):
        return [_fake_position(s, q) for s, q in self._positions.items()]

    def get_orders(self, **kwargs):
        return list(self._orders)


class _FakeExecutor:
    """Stand-in for AlpacaExecutor in tests."""

    def __init__(self, *, env: str = "paper", client: _FakeBrokerClient | None = None):
        self.env = env
        self._client = client or _FakeBrokerClient()

    def get_positions(self) -> dict[str, int]:
        return dict(self._client._positions)


# ---------------------------------------------------------------------------
# _check_run_record
# ---------------------------------------------------------------------------


def test_run_record_missing_returns_failure(tmp_path: Path) -> None:
    """No JSON for the date → failure."""
    check = _check_run_record(date(2026, 1, 5), runs_dir=tmp_path)
    assert not check.passed
    assert "no daily run record" in check.message


def test_run_record_present_returns_pass(tmp_path: Path) -> None:
    """Valid record → pass and details include order/target counts."""
    fp = tmp_path / "2026-01-05.json"
    fp.write_text(json.dumps({
        "run_date": "2026-01-05",
        "strategy_name": "ensemble(3)",
        "target_weights": {"AAPL": 0.5, "MSFT": 0.5},
        "signal_prices": {"AAPL": 200.0, "MSFT": 400.0},
        "execution_report": {
            "env": "paper",
            "account_equity_before": 100_000.0,
            "submitted_orders": [
                {"symbol": "AAPL", "role": "entry", "qty": 250},
                {"symbol": "AAPL", "role": "stop_loss", "qty": 250},
            ],
            "dry_run": False,
        },
    }))
    check = _check_run_record(date(2026, 1, 5), runs_dir=tmp_path)
    assert check.passed
    assert "paper" in check.message
    assert check.details["n_orders"] == 2
    assert check.details["n_targets"] == 2


def test_run_record_missing_fields_fails(tmp_path: Path) -> None:
    """Record present but missing required keys → fail with clear reason."""
    fp = tmp_path / "2026-01-05.json"
    fp.write_text(json.dumps({"run_date": "2026-01-05"}))   # almost empty
    check = _check_run_record(date(2026, 1, 5), runs_dir=tmp_path)
    assert not check.passed
    assert "missing fields" in check.message


# ---------------------------------------------------------------------------
# _check_broker_reconciliation
# ---------------------------------------------------------------------------


def test_broker_clean_state_passes(tmp_path: Path) -> None:
    """Every position has matching GTC stop; no orphans → pass."""
    positions = {"AAPL": 10, "MSFT": 5}
    orders = [
        _fake_order("AAPL", qty=10, tif="gtc"),
        _fake_order("MSFT", qty=5, tif="gtc"),
    ]
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))
    check = _check_broker_reconciliation(date.today(), runs_dir=tmp_path, executor=ex)
    assert check.passed, check.message
    assert "2 positions, 2 GTC stops" in check.message


def test_broker_unprotected_position_fails(tmp_path: Path) -> None:
    """A position with no matching stop → audit failure."""
    positions = {"AAPL": 10, "MSFT": 5}
    orders = [_fake_order("AAPL", qty=10, tif="gtc")]  # MSFT has no stop
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))
    check = _check_broker_reconciliation(date.today(), runs_dir=tmp_path, executor=ex)
    assert not check.passed
    assert "unprotected" in check.message
    assert "MSFT" in check.message


def test_broker_orphan_stop_fails(tmp_path: Path) -> None:
    """A stop with no matching position → audit failure."""
    positions = {"AAPL": 10}
    orders = [
        _fake_order("AAPL", qty=10, tif="gtc"),
        _fake_order("GHOST", qty=1, tif="gtc"),    # orphan
    ]
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))
    check = _check_broker_reconciliation(date.today(), runs_dir=tmp_path, executor=ex)
    assert not check.passed
    assert "orphan" in check.message
    assert "GHOST" in check.message


def test_broker_non_gtc_stop_fails(tmp_path: Path) -> None:
    """A DAY-TIF stop will expire at close — audit must flag it."""
    positions = {"AAPL": 10}
    orders = [_fake_order("AAPL", qty=10, tif="day")]  # DAY, not GTC
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))
    check = _check_broker_reconciliation(date.today(), runs_dir=tmp_path, executor=ex)
    assert not check.passed
    assert "non-GTC" in check.message


def test_broker_trading_blocked_fails(tmp_path: Path) -> None:
    """If the broker has frozen the account, audit must scream."""
    ex = _FakeExecutor(client=_FakeBrokerClient(trading_blocked=True))
    check = _check_broker_reconciliation(date.today(), runs_dir=tmp_path, executor=ex)
    assert not check.passed
    assert "TRADING_BLOCKED" in check.message


def test_broker_quantity_mismatch_fails(tmp_path: Path) -> None:
    """Stop qty differs from position qty (e.g., partial fill) → failure."""
    positions = {"AAPL": 10}
    orders = [_fake_order("AAPL", qty=7, tif="gtc")]  # stop only covers 7
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))
    check = _check_broker_reconciliation(date.today(), runs_dir=tmp_path, executor=ex)
    assert not check.passed
    assert "qty mismatch" in check.message


# ---------------------------------------------------------------------------
# _check_ensemble_state
# ---------------------------------------------------------------------------


def test_ensemble_state_defaults_pass() -> None:
    """Fresh-install defaults pass: 3 strategies, equal weights, no AI files."""
    check = _check_ensemble_state()
    assert check.passed, check.message
    assert check.details["n_strategies"] == 3
    assert abs(check.details["hrp_sum"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# _check_recent_error_logs
# ---------------------------------------------------------------------------


def test_no_log_dir_is_pass(tmp_path: Path) -> None:
    """A missing logs directory means fresh install, not an error."""
    check = _check_recent_error_logs(log_dir=tmp_path / "nope")
    assert check.passed
    assert "no log directory" in check.message


def test_empty_err_logs_pass(tmp_path: Path) -> None:
    """Empty .err files (everything ran clean) → pass."""
    (tmp_path / "daily-trade.err").touch()
    (tmp_path / "daily-report.err").touch()
    check = _check_recent_error_logs(log_dir=tmp_path)
    assert check.passed


def test_recent_non_empty_err_log_fails(tmp_path: Path) -> None:
    """A non-empty .err modified recently → flag for operator attention."""
    fp = tmp_path / "daily-trade.err"
    fp.write_text("Traceback (most recent call last)...\nValueError: nope\n")
    check = _check_recent_error_logs(log_dir=tmp_path, hours=24)
    assert not check.passed
    assert "daily-trade.err" in check.message


def test_old_err_log_ignored(tmp_path: Path) -> None:
    """An old error log (outside the window) should NOT trigger a failure."""
    fp = tmp_path / "ancient.err"
    fp.write_text("Traceback: very old error\n")   # has marker
    # Set mtime to 48h ago.
    import os
    old_ts = (datetime.now(tz=UTC) - timedelta(hours=48)).timestamp()
    os.utime(fp, (old_ts, old_ts))
    check = _check_recent_error_logs(log_dir=tmp_path, hours=26)
    assert check.passed, check.message


def test_warning_only_err_log_does_not_flag(tmp_path: Path) -> None:
    """A .err containing only WARNING lines (e.g. successful retries
    that ended up succeeding) must NOT be flagged — that was the
    false-positive that caused every flaky-but-OK run to fail the audit.
    """
    fp = tmp_path / "daily-trade.err"
    fp.write_text(
        "2026-05-30 21:35:11,000 WARNING quant.util.retry: Alpaca bars fetch "
        "(1 symbols): transient error on attempt 1/4 (ConnectionError); "
        "retrying in 1.0s\n"
        "2026-05-30 21:35:14,000 WARNING quant.util.retry: Alpaca bars fetch "
        "(1 symbols): transient error on attempt 2/4 (ConnectionError); "
        "retrying in 3.0s\n"
    )
    check = _check_recent_error_logs(log_dir=tmp_path, hours=24)
    assert check.passed, (
        f"WARNING-only log should not trip the audit (msg: {check.message})"
    )


def test_err_log_with_real_error_still_flags(tmp_path: Path) -> None:
    """A real Python traceback or ERROR-level log must flag.
    Belt-and-suspenders test for the marker-based detection."""
    for marker_line, fname in [
        ("ERROR something went wrong", "a.err"),
        ("Traceback (most recent call last):", "b.err"),
    ]:
        fp = tmp_path / fname
        fp.write_text(marker_line + "\n")
        check = _check_recent_error_logs(log_dir=tmp_path, hours=24)
        assert not check.passed, (
            f"Log containing '{marker_line}' should have flagged"
        )


# ---------------------------------------------------------------------------
# _check_alpaca_connectivity
# ---------------------------------------------------------------------------


def test_connectivity_pass_with_fake() -> None:
    ex = _FakeExecutor(client=_FakeBrokerClient())
    check = _check_alpaca_connectivity(executor=ex)
    assert check.passed


def test_connectivity_fail_when_broker_raises() -> None:
    ex = _FakeExecutor(client=_FakeBrokerClient(
        raise_on_account=ConnectionError("dns flake"),
    ))
    check = _check_alpaca_connectivity(executor=ex)
    assert not check.passed
    assert "unreachable" in check.message


# ---------------------------------------------------------------------------
# run_daily_audit orchestrator
# ---------------------------------------------------------------------------


class _FakeEmailSender:
    """Captures sent messages instead of touching SMTP."""
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
    def send(self, *, subject: str, body_text: str, body_html: str = "") -> None:  # noqa: ARG002
        self.sent.append((subject, body_text))


def test_run_audit_persists_json_and_sends_email(tmp_path: Path) -> None:
    """The orchestrator writes a JSON record and emails the result."""
    # Provide a fake broker so the audit can complete without network.
    positions = {"AAPL": 10}
    orders = [_fake_order("AAPL", qty=10, tif="gtc")]
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))

    # Create a minimal valid run record for today.
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    today = date.today()
    (runs_dir / f"{today.isoformat()}.json").write_text(json.dumps({
        "run_date": today.isoformat(),
        "strategy_name": "ensemble(3)",
        "target_weights": {"AAPL": 1.0},
        "signal_prices": {"AAPL": 200.0},
        "execution_report": {
            "env": "paper",
            "account_equity_before": 100_000.0,
            "submitted_orders": [
                {"symbol": "AAPL", "role": "entry", "qty": 10},
                {"symbol": "AAPL", "role": "stop_loss", "qty": 10},
            ],
            "dry_run": False,
        },
    }))

    audits_dir = tmp_path / "audits"
    mailbox = _FakeEmailSender()

    report = run_daily_audit(
        for_date=today,
        runs_dir=runs_dir,
        audits_dir=audits_dir,
        executor=ex,
        log_dir=tmp_path / "empty-logs",  # nonexistent → counts as clean
        email_sender=mailbox,
    )

    # Persisted JSON exists.
    audit_file = audits_dir / f"{today.isoformat()}.json"
    assert audit_file.exists()
    saved = json.loads(audit_file.read_text())
    assert saved["passed"] == report.passed
    assert len(saved["checks"]) == len(report.checks)

    # Email was sent with appropriate subject.
    assert len(mailbox.sent) == 1
    subject, body = mailbox.sent[0]
    if report.passed:
        assert "OK" in subject and "all" in subject
        assert "✅" in body
    else:
        assert "FAILED" in subject
        assert "❌" in body


def test_run_audit_returns_failed_report_on_problem(tmp_path: Path) -> None:
    """Any one check failing flips the overall report to passed=False."""
    # Give the broker an unprotected position → reconciliation will fail.
    positions = {"AAPL": 10, "MSFT": 5}
    orders = [_fake_order("AAPL", qty=10, tif="gtc")]  # MSFT unprotected
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    today = date.today()
    (runs_dir / f"{today.isoformat()}.json").write_text(json.dumps({
        "run_date": today.isoformat(),
        "strategy_name": "ensemble(3)",
        "target_weights": {},
        "signal_prices": {},
        "execution_report": {
            "env": "paper", "account_equity_before": 0.0,
            "submitted_orders": [], "dry_run": False,
        },
    }))

    report = run_daily_audit(
        for_date=today,
        runs_dir=runs_dir,
        audits_dir=tmp_path / "audits",
        executor=ex,
        log_dir=tmp_path / "empty-logs",
        email_sender=_FakeEmailSender(),
    )
    assert not report.passed
    failed_names = {c.name for c in report.failures}
    assert "broker_reconciliation" in failed_names


def test_run_audit_email_failure_does_not_mask_audit(tmp_path: Path) -> None:
    """If the email sender raises, the audit should still complete & persist."""
    positions = {"AAPL": 10}
    orders = [_fake_order("AAPL", qty=10, tif="gtc")]
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    today = date.today()
    (runs_dir / f"{today.isoformat()}.json").write_text(json.dumps({
        "run_date": today.isoformat(),
        "strategy_name": "ensemble(3)",
        "target_weights": {"AAPL": 1.0},
        "signal_prices": {"AAPL": 200.0},
        "execution_report": {
            "env": "paper", "account_equity_before": 100_000.0,
            "submitted_orders": [
                {"symbol": "AAPL", "role": "entry", "qty": 10},
                {"symbol": "AAPL", "role": "stop_loss", "qty": 10},
            ],
            "dry_run": False,
        },
    }))

    class _BrokenSender:
        def send(self, **kwargs):
            raise RuntimeError("SMTP down")

    # Should NOT raise — email failures are logged, not propagated.
    report = run_daily_audit(
        for_date=today,
        runs_dir=runs_dir,
        audits_dir=tmp_path / "audits",
        executor=ex,
        log_dir=tmp_path / "empty-logs",
        email_sender=_BrokenSender(),
    )
    # Audit JSON should still be on disk.
    assert (tmp_path / "audits" / f"{today.isoformat()}.json").exists()


# ---------------------------------------------------------------------------
# Email rendering
# ---------------------------------------------------------------------------


def test_render_email_pass_banner() -> None:
    report = AuditReport(
        for_date="2026-05-27",
        timestamp="2026-05-28T11:00:00+00:00",
        checks=[AuditCheck(name="x", passed=True, message="ok")],
    )
    subject, body = _render_email(report)
    assert "OK" in subject
    assert "all 1 checks passed" in subject
    assert "AUDIT PASSED" in body


def test_render_email_fail_banner_with_details() -> None:
    report = AuditReport(
        for_date="2026-05-27",
        timestamp="2026-05-28T11:00:00+00:00",
        checks=[
            AuditCheck(name="x", passed=True, message="ok"),
            AuditCheck(name="y", passed=False, message="bad", details={"why": "z"}),
        ],
    )
    subject, body = _render_email(report)
    assert "FAILED" in subject
    assert "1 of 2 checks failed" in subject
    assert "AUDIT FAILED" in body
    assert "Details for failing checks" in body
    assert "\"why\"" in body  # JSON-rendered detail


# ---------------------------------------------------------------------------
# T-bug 2026-06-09: intent-aware direction-mismatch check (Fix 4)
# ---------------------------------------------------------------------------


def _save_run_with_target_weights(
    tmp_path: Path, for_date: date, target_weights: dict[str, float],
) -> None:
    """Persist a minimal run JSON the audit can read for target weights."""
    import json
    payload = {
        "date": for_date.isoformat(),
        "strategy_name": "test-ensemble",
        "strategy_params": {},
        "target_weights": target_weights,
        "signal_prices": {},
        "execution_report": {
            "env": "paper",
            "account_equity_before": 100_000.0,
            "positions_before": {},
            "target_weights": target_weights,
            "proposed_orders": [],
            "submitted_orders": [],
            "dry_run": False,
            "timestamp": "2026-06-09T13:35:00+00:00",
            "notes": "",
        },
    }
    (tmp_path / f"{for_date.isoformat()}.json").write_text(json.dumps(payload))


def test_broker_direction_mismatch_flags_short_when_target_is_long(tmp_path: Path) -> None:
    """The 2026-06-09 AMD bug: broker says -9 short, intent was long.
    Audit must hard-fail on this so the operator catches it on day 1."""
    today = date(2024, 6, 3)
    _save_run_with_target_weights(tmp_path, today, {"AMD": 0.05})  # long intent
    positions = {"AMD": -9}                                          # but actually short
    orders = []   # no stops to worry about
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))
    check = _check_broker_reconciliation(today, runs_dir=tmp_path, executor=ex)
    assert not check.passed
    assert "direction mismatch" in check.message.lower()
    assert "AMD" in check.message
    # Details surface it for the operator email.
    assert any("AMD" in row for row in check.details["direction_mismatch"])


def test_broker_direction_mismatch_flags_long_when_target_is_short(tmp_path: Path) -> None:
    """Symmetric: intent was short but broker shows long. Same alarm.
    This branch is what makes the check forward-compatible with future
    shorting strategies — it catches both directions."""
    today = date(2024, 6, 3)
    _save_run_with_target_weights(tmp_path, today, {"NVDA": -0.05})  # short intent
    positions = {"NVDA": 10}                                          # but actually long
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=[
        _fake_order("NVDA", qty=10, tif="gtc"),
    ]))
    check = _check_broker_reconciliation(today, runs_dir=tmp_path, executor=ex)
    assert not check.passed
    assert "direction mismatch" in check.message.lower()
    assert "NVDA" in check.message


def test_broker_short_with_matching_short_intent_passes(tmp_path: Path) -> None:
    """Forward-compatibility: when a strategy emits a short signal,
    the broker holding a matching short must NOT trip the check.
    Without this, enabling shorting would force ripping out the check."""
    today = date(2024, 6, 3)
    _save_run_with_target_weights(tmp_path, today, {"AAPL": -0.10})   # short intent
    positions = {"AAPL": -50}                                          # matches: short held
    # Pretend we have a buy-stop on the short (audit doesn't care which side).
    orders = [_fake_order("AAPL", qty=50, tif="gtc")]
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))
    check = _check_broker_reconciliation(today, runs_dir=tmp_path, executor=ex)
    # Direction-mismatch should be EMPTY.
    assert check.details["direction_mismatch"] == []


def test_broker_long_with_long_intent_passes(tmp_path: Path) -> None:
    """Baseline sanity: standard long position with long target → pass."""
    today = date(2024, 6, 3)
    _save_run_with_target_weights(tmp_path, today, {"AAPL": 0.10})
    positions = {"AAPL": 50}
    orders = [_fake_order("AAPL", qty=50, tif="gtc")]
    ex = _FakeExecutor(client=_FakeBrokerClient(positions=positions, open_orders=orders))
    check = _check_broker_reconciliation(today, runs_dir=tmp_path, executor=ex)
    assert check.details["direction_mismatch"] == []


# ---------------------------------------------------------------------------
# T-bug 2026-06-09: SMTP-only error log is transient, must not flag (Fix 5)
# ---------------------------------------------------------------------------


def test_smtp_only_traceback_is_treated_as_transient(tmp_path: Path) -> None:
    """A .err that contains ONLY SMTP disconnect tracebacks must NOT
    flag the audit. The retry layer already gave SMTP 4 attempts; if
    all fail, the next cron fire tries fresh. Flagging it daily is
    noise, not signal.
    """
    fp = tmp_path / "daily-report.err"
    fp.write_text(
        "Traceback (most recent call last):\n"
        '  File "/path/smtplib.py", line 261, in __init__\n'
        '    (code, msg) = self.connect(host, port)\n'
        "smtplib.SMTPServerDisconnected: Connection unexpectedly closed\n"
    )
    check = _check_recent_error_logs(log_dir=tmp_path, hours=24)
    assert check.passed, (
        f"SMTP-only flake should be ignored; got: {check.message}"
    )
    # But the operator should still see it noted.
    assert "daily-report.err" in check.message
    assert check.details["skipped_transient"] == ["daily-report.err"]


def test_connection_reset_only_traceback_is_treated_as_transient(tmp_path: Path) -> None:
    """TLS handshake / connection reset traces are also transient
    (typical VPN re-establishment behavior)."""
    fp = tmp_path / "daily-trade.err"
    fp.write_text(
        "Traceback (most recent call last):\n"
        '  File "/path/adapters.py", line 711, in send\n'
        "    raise ConnectionError(err, request=request)\n"
        "ConnectionResetError: [Errno 54] Connection reset by peer\n"
    )
    check = _check_recent_error_logs(log_dir=tmp_path, hours=24)
    assert check.passed


def test_mixed_transient_and_real_error_still_flags(tmp_path: Path) -> None:
    """If a file contains BOTH a transient AND a genuine bug, the
    transient must not mask the bug. The audit flags it."""
    fp = tmp_path / "daily-trade.err"
    fp.write_text(
        "Traceback (most recent call last):\n"
        '  File "/path/foo.py", line 1, in <module>\n'
        "    1/0\n"
        "ZeroDivisionError: division by zero\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "/path/smtplib.py", line 261, in __init__\n'
        "smtplib.SMTPServerDisconnected: Connection unexpectedly closed\n"
    )
    check = _check_recent_error_logs(log_dir=tmp_path, hours=24)
    assert not check.passed
    assert "daily-trade.err" in check.message


def test_timeout_error_only_is_transient(tmp_path: Path) -> None:
    """`TimeoutError: timed out` is the classic SMTP-during-send leaf;
    treat as transient."""
    fp = tmp_path / "daily-report.err"
    fp.write_text(
        "Traceback (most recent call last):\n"
        '  File "/path/socket.py", line 859, in create_connection\n'
        "    sock.connect(sa)\n"
        "TimeoutError: timed out\n"
    )
    check = _check_recent_error_logs(log_dir=tmp_path, hours=24)
    assert check.passed
