"""Tests for the Alpaca executor.

Pattern: a fake TradingClient stub satisfies the executor's Protocol so we
can drive submit/get_positions/get_account paths without network. Tests
focus on the business logic (delta computation, safety gates, dry-run vs
live-submit branches) rather than alpaca-py specifics.
"""

from __future__ import annotations

import pytest

from quant.execution.alpaca_executor import (
    AlpacaExecutor,
    _compute_proposed_orders,
)


# ---------------------------------------------------------------------------
# Test stubs
# ---------------------------------------------------------------------------


class _FakeAccount:
    def __init__(self, equity: float) -> None:
        self.equity = equity


class _FakePosition:
    def __init__(self, symbol: str, qty: int) -> None:
        self.symbol = symbol
        self.qty = str(qty)  # mirror Alpaca's string-typed return


class _FakeOrderResponse:
    def __init__(self, order_id: str) -> None:
        self.id = order_id


class _FakeTradingClient:
    """Stub trading client. Records every submit_order call."""

    def __init__(
        self,
        *,
        equity: float = 1_000_000.0,
        positions: dict[str, int] | None = None,
        raise_on_submit: bool = False,
    ) -> None:
        self._equity = equity
        self._positions = positions or {}
        self.submitted: list = []
        self._raise_on_submit = raise_on_submit
        self._next_order_id = 0

    def get_account(self) -> _FakeAccount:
        return _FakeAccount(equity=self._equity)

    def get_all_positions(self) -> list[_FakePosition]:
        return [_FakePosition(s, q) for s, q in self._positions.items()]

    def submit_order(self, request) -> _FakeOrderResponse:
        if self._raise_on_submit:
            raise RuntimeError("simulated broker rejection")
        self._next_order_id += 1
        self.submitted.append(request)
        return _FakeOrderResponse(order_id=f"fake-{self._next_order_id}")


# ---------------------------------------------------------------------------
# Safety gates
# ---------------------------------------------------------------------------


def test_live_env_without_flag_raises_permission_error() -> None:
    """The whole point: typing env='live' alone shouldn't put money in motion."""
    with pytest.raises(PermissionError, match="i_mean_it_live"):
        AlpacaExecutor(env="live", trading_client=_FakeTradingClient())


def test_live_env_with_flag_constructs_ok() -> None:
    """If the operator says i_mean_it_live=True, allow it."""
    exec_ = AlpacaExecutor(
        env="live",
        i_mean_it_live=True,
        trading_client=_FakeTradingClient(),
    )
    assert exec_.env == "live"


def test_paper_env_default_is_safe() -> None:
    """Default construction is paper, no flag needed."""
    exec_ = AlpacaExecutor(trading_client=_FakeTradingClient())
    assert exec_.env == "paper"


def test_unknown_env_rejected() -> None:
    with pytest.raises(ValueError, match="env"):
        AlpacaExecutor(env="sandbox", trading_client=_FakeTradingClient())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pure delta computation
# ---------------------------------------------------------------------------


def test_no_existing_positions_one_target_generates_buy() -> None:
    """Fresh account, weight 0.5 of $1M @ $200 → buy 2500 shares."""
    orders = _compute_proposed_orders(
        target_weights={"AAPL": 0.5},
        signal_prices={"AAPL": 200.0},
        current_positions={},
        equity=1_000_000.0,
    )
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].side == "buy"
    assert orders[0].qty == 2500


def test_negative_target_weight_generates_sell_from_flat() -> None:
    """Short opens via a sell from flat. The engine and executor agree on this."""
    orders = _compute_proposed_orders(
        target_weights={"AAPL": -0.25},
        signal_prices={"AAPL": 100.0},
        current_positions={},
        equity=1_000_000.0,
    )
    assert len(orders) == 1
    assert orders[0].side == "sell"
    # target_qty = int(-0.25 * 1_000_000 / 100) = -2500. current = 0. delta = -2500.
    assert orders[0].qty == 2500


def test_unmentioned_held_symbol_gets_flattened() -> None:
    """We hold MSFT but the new target weights don't mention it → sell all MSFT."""
    orders = _compute_proposed_orders(
        target_weights={"AAPL": 0.5},
        signal_prices={"AAPL": 100.0, "MSFT": 400.0},
        current_positions={"MSFT": 100},
        equity=1_000_000.0,
    )
    msft_orders = [o for o in orders if o.symbol == "MSFT"]
    assert len(msft_orders) == 1
    assert msft_orders[0].side == "sell"
    assert msft_orders[0].qty == 100


def test_partial_rebalance_only_emits_the_delta() -> None:
    """Already at the target → no order. Slight target change → small order."""
    # Already long 5000 AAPL at $200. Target weight 1.0 of $1M / $200 = 5000.
    # Delta = 0. No order.
    orders = _compute_proposed_orders(
        target_weights={"AAPL": 1.0},
        signal_prices={"AAPL": 200.0},
        current_positions={"AAPL": 5000},
        equity=1_000_000.0,
    )
    assert orders == []


def test_symbol_without_price_is_skipped() -> None:
    """Can't size without a price — silent skip rather than crash."""
    orders = _compute_proposed_orders(
        target_weights={"AAPL": 0.5, "FOO": 0.5},
        signal_prices={"AAPL": 100.0},  # FOO missing
        current_positions={},
        equity=1_000_000.0,
    )
    assert {o.symbol for o in orders} == {"AAPL"}


# ---------------------------------------------------------------------------
# End-to-end (with stub client) — dry run and submission
# ---------------------------------------------------------------------------


def test_dry_run_does_not_call_submit_order() -> None:
    """The whole point of dry_run: see the diff, submit nothing."""
    client = _FakeTradingClient(equity=1_000_000)
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_to_match_targets(
        target_weights={"AAPL": 0.5},
        signal_prices={"AAPL": 200.0},
        dry_run=True,
    )
    assert client.submitted == []
    # The proposed orders are still computed and reported.
    assert len(report.proposed_orders) == 1
    # All submitted_orders are "skipped_dry_run" status.
    assert all(o.status == "skipped_dry_run" for o in report.submitted_orders)


def test_real_submit_calls_alpaca_per_order() -> None:
    """With dry_run=False, every proposed order results in a submit_order call."""
    client = _FakeTradingClient(equity=1_000_000)
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_to_match_targets(
        target_weights={"AAPL": 0.4, "MSFT": 0.4},
        signal_prices={"AAPL": 200.0, "MSFT": 400.0},
        dry_run=False,
    )
    assert len(client.submitted) == 2
    assert all(o.status == "submitted" for o in report.submitted_orders)
    # Each report row should have an Alpaca order id.
    assert all(o.alpaca_order_id is not None for o in report.submitted_orders)


def test_broker_rejection_recorded_as_failed_not_raised() -> None:
    """A single bad symbol shouldn't drop the whole batch.

    The executor records the failure on the report row and keeps going.
    """
    client = _FakeTradingClient(equity=1_000_000, raise_on_submit=True)
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_to_match_targets(
        target_weights={"AAPL": 0.5},
        signal_prices={"AAPL": 200.0},
        dry_run=False,
    )
    assert len(report.submitted_orders) == 1
    assert report.submitted_orders[0].status == "failed"
    assert report.submitted_orders[0].error is not None
    assert "simulated broker rejection" in report.submitted_orders[0].error


def test_report_includes_snapshot_of_inputs() -> None:
    """The report is the audit trail — must capture inputs verbatim."""
    client = _FakeTradingClient(
        equity=500_000,
        positions={"AAPL": 100},
    )
    exec_ = AlpacaExecutor(trading_client=client)
    targets = {"AAPL": 0.5, "MSFT": 0.5}
    report = exec_.submit_to_match_targets(
        target_weights=targets,
        signal_prices={"AAPL": 100.0, "MSFT": 200.0},
        dry_run=True,
        notes="test run",
    )
    assert report.env == "paper"
    assert report.account_equity_before == 500_000
    assert report.positions_before == {"AAPL": 100}
    assert report.target_weights == targets
    assert report.notes == "test run"
    assert report.dry_run is True
