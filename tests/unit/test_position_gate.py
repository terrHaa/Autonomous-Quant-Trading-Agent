"""Tests for the position-feed sanity gate (T-incident 2026-07-07)."""
from __future__ import annotations

from datetime import UTC, datetime

from quant.agent.position_gate import gate_positions, reconcile_positions


class _FakeOrder:
    def __init__(self, symbol, side, filled_qty):
        self.symbol = symbol
        self.side = side          # "BUY" / "SELL"
        self.filled_qty = filled_qty


class _FakeExec:
    def __init__(self, ledger, fills):
        self._ledger = ledger
        self._fills = fills
        self._client = self

    def get_orders(self, req):
        return self._fills

    def get_positions(self):
        return dict(self._ledger)


_TS = datetime(2026, 7, 6, 13, 35, tzinfo=UTC)


def test_reconcile_ok_when_ledger_matches_fills() -> None:
    baseline = {"AAPL": 10}
    fills = [_FakeOrder("AAPL", "BUY", 5), _FakeOrder("MSFT", "BUY", 3)]
    ex = _FakeExec(ledger={"AAPL": 15, "MSFT": 3}, fills=fills)
    r = reconcile_positions(ex, baseline, _TS)
    assert r.ok, r.summary()


def test_reconcile_flags_desync_lost_shares() -> None:
    """The July 7 pattern: ledger LOST most of a position."""
    baseline = {"APA": 25}
    ex = _FakeExec(ledger={"APA": 2}, fills=[])   # ledger says 2, truth is 25
    r = reconcile_positions(ex, baseline, _TS)
    assert not r.ok
    assert r.mismatches["APA"] == (25, 2)


def test_reconcile_flags_phantom_short() -> None:
    """Stop fill applied against a ledger that lost the shares → -8."""
    baseline = {"MRNA": 8}
    fills = [_FakeOrder("MRNA", "SELL", 8)]        # stop sold the real 8 → 0
    ex = _FakeExec(ledger={"MRNA": -8}, fills=fills)  # ledger claims short
    r = reconcile_positions(ex, baseline, _TS)
    assert not r.ok
    assert r.mismatches["MRNA"] == (0, -8)


def test_gate_passes_without_previous_record(tmp_path) -> None:
    v = gate_positions(_FakeExec({}, []), prev_run=None,
                       override_path=tmp_path / "missing.json")
    assert v.ok


def test_gate_fails_open_on_infrastructure_error(tmp_path) -> None:
    class _Boom(_FakeExec):
        def get_orders(self, req):
            raise ConnectionError("network down")
    prev = {"execution_report": {
        "positions_before": {"AAPL": 1},
        "timestamp": _TS.isoformat(),
    }}
    v = gate_positions(_Boom({}, []), prev_run=prev,
                       override_path=tmp_path / "missing.json")
    assert v.ok                       # infra error must not block trading
    assert "failed open" in v.reason


def test_gate_halts_on_phantom_short(tmp_path) -> None:
    """Ledger shows a name SHORT that we hold long → the dangerous July-7
    signature → halt."""
    prev = {"execution_report": {
        "positions_before": {"AMD": 5},
        "timestamp": _TS.isoformat(),
    }}
    # Reconstruction expects long 5, ledger claims short -5.
    v = gate_positions(_FakeExec({"AMD": -5}, []), prev_run=prev,
                       override_path=tmp_path / "missing.json")
    assert not v.ok
    assert "phantom short" in v.reason


def test_gate_halts_on_wholesale_desync(tmp_path) -> None:
    """Most of the book disagrees → wholesale desync → halt."""
    baseline = {f"S{i}": 1 for i in range(10)}
    prev = {"execution_report": {
        "positions_before": baseline, "timestamp": _TS.isoformat(),
    }}
    ledger = {f"S{i}": 1 for i in range(3)}   # 7 of 10 vanished
    v = gate_positions(_FakeExec(ledger, []), prev_run=prev,
                       override_path=tmp_path / "missing.json")
    assert not v.ok
    assert "wholesale" in v.reason


def test_gate_trades_through_minor_drift(tmp_path) -> None:
    """2-of-37 stray-share drift (the 2026-07-15 false positive) must NOT
    halt the whole book — trade, and let the audit reconcile."""
    baseline = {f"S{i}": 1 for i in range(37)}
    prev = {"execution_report": {
        "positions_before": baseline, "timestamp": _TS.isoformat(),
    }}
    # 2 names show 0 at the broker (stopped out; reconstruction missed exit).
    ledger = {f"S{i}": 1 for i in range(37) if i not in (5, 12)}
    v = gate_positions(_FakeExec(ledger, []), prev_run=prev,
                       override_path=tmp_path / "missing.json")
    assert v.ok                       # <-- trades, does not halt
    assert "minor drift" in v.reason


def test_pinned_baseline_overrides_stale_run_record(tmp_path) -> None:
    """Post-desync: Alpaca healed by fiat, so the pre-heal run record can
    never reconcile. A pinned operator-verified baseline takes over."""
    from quant.agent.position_gate import pin_verified_baseline

    # Stale run record claims we hold FSLR 1 (pre-heal snapshot).
    prev = {"execution_report": {
        "positions_before": {"FSLR": 1, "AAPL": 2},
        "timestamp": _TS.isoformat(),
    }}
    # Operator verifies healed state: no FSLR, AAPL 2.
    ov = tmp_path / "baseline.json"
    pin_verified_baseline({"AAPL": 2}, path=ov)
    ex = _FakeExec(ledger={"AAPL": 2}, fills=[])
    v = gate_positions(ex, prev_run=prev, override_path=ov)
    assert v.ok, v.summary()   # would FAIL on the stale record without the pin
