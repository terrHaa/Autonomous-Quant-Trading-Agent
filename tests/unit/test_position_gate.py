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


def test_gate_fails_closed_on_desync(tmp_path) -> None:
    prev = {"execution_report": {
        "positions_before": {"APA": 25},
        "timestamp": _TS.isoformat(),
    }}
    v = gate_positions(_FakeExec({"APA": 2}, []), prev_run=prev,
                       override_path=tmp_path / "missing.json")
    assert not v.ok


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
