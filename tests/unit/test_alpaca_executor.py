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
        self.cancel_calls = 0

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

    def cancel_orders(self) -> None:
        self.cancel_calls += 1


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
        i_understand_no_stops=True,
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
        i_understand_no_stops=True,
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


# ---------------------------------------------------------------------------
# submit_daily_rebalance — the agent's flow
# ---------------------------------------------------------------------------


def test_submit_to_match_targets_refuses_live_without_explicit_optin() -> None:
    """The bare-entry path bypasses the 5% stop rule; live calls without
    the i_understand_no_stops opt-in must raise rather than silently
    place naked orders."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    with pytest.raises(PermissionError, match="bypasses the 5% stop"):
        exec_.submit_to_match_targets(
            target_weights={"AAPL": 0.1},
            signal_prices={"AAPL": 200.0},
            dry_run=False,        # live submit
            # i_understand_no_stops omitted on purpose
        )
    # No orders were submitted; the broker remained pristine.
    assert client.submitted == []


def test_submit_to_match_targets_dry_run_does_not_require_optin() -> None:
    """Dry-run is read-only; safe to call without the opt-in flag."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_to_match_targets(
        target_weights={"AAPL": 0.1},
        signal_prices={"AAPL": 200.0},
        dry_run=True,
    )
    # No exception, no broker calls, but the report still computed.
    assert client.submitted == []
    assert len(report.submitted_orders) == 1


def test_daily_rebalance_cancels_open_orders_first() -> None:
    """Every daily run starts by clearing yesterday's stops."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.1},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
    )
    assert client.cancel_calls == 1


def test_daily_rebalance_dry_run_does_not_call_cancel_or_submit() -> None:
    """Dry run is fully read-only — no broker mutation at all."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.1},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        dry_run=True,
    )
    assert client.cancel_calls == 0
    assert client.submitted == []
    # Report still shows what WOULD have happened.
    statuses = {o.status for o in report.submitted_orders}
    assert statuses == {"skipped_dry_run"}


def test_daily_rebalance_emits_entry_and_stop_loss_rows() -> None:
    """For each new long, the report has TWO rows: entry + stop_loss audit."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.1},   # $10k / $200 = 50 shares
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        dry_run=True,
    )
    roles = sorted(o.role for o in report.submitted_orders)
    assert roles == ["entry", "stop_loss"]
    stop_row = next(o for o in report.submitted_orders if o.role == "stop_loss")
    # 5% below the signal price of $200 = $190.
    assert stop_row.stop_price == 190.0


def test_daily_rebalance_refuses_oversize_orders() -> None:
    """The 20%-of-equity per-trade cap is enforced as a HARD refusal,
    even if the upstream target weight implies a larger notional."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    # Target 30% in AAPL — exceeds 20% default cap.
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.30},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        max_position_weight=0.20,
        dry_run=False,
    )
    failed = [o for o in report.submitted_orders if o.status == "failed"]
    assert len(failed) >= 1
    assert "max_position_weight" in failed[0].error


def test_daily_rebalance_rejects_negative_target_weight() -> None:
    """This method is long-only; shorts would need OTO with a buy-stop child."""
    exec_ = AlpacaExecutor(trading_client=_FakeTradingClient())
    with pytest.raises(ValueError, match="long-only"):
        exec_.submit_daily_rebalance(
            target_weights={"AAPL": -0.1},
            signal_prices={"AAPL": 200.0},
            stop_loss_pct=0.05,
        )


def test_daily_rebalance_rejects_bad_stop_pct() -> None:
    exec_ = AlpacaExecutor(trading_client=_FakeTradingClient())
    with pytest.raises(ValueError, match="stop_loss_pct"):
        exec_.submit_daily_rebalance(
            target_weights={"AAPL": 0.1},
            signal_prices={"AAPL": 200.0},
            stop_loss_pct=0,
        )
    with pytest.raises(ValueError, match="stop_loss_pct"):
        exec_.submit_daily_rebalance(
            target_weights={"AAPL": 0.1},
            signal_prices={"AAPL": 200.0},
            stop_loss_pct=1.5,
        )


def test_daily_rebalance_closes_stale_positions_not_in_targets() -> None:
    """Yesterday's longs that aren't on today's list get sold to flat."""
    client = _FakeTradingClient(
        equity=100_000,
        positions={"OLDPOS": 50},      # held but not in target
    )
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.1},  # OLDPOS not mentioned → flatten
        signal_prices={"AAPL": 200.0, "OLDPOS": 80.0},
        stop_loss_pct=0.05,
        dry_run=True,
    )
    # Should have a sell of OLDPOS in the report.
    sells = [o for o in report.submitted_orders
             if o.symbol == "OLDPOS" and o.side == "sell"]
    assert len(sells) == 1
    assert sells[0].qty == 50


def test_daily_rebalance_uses_trail_highs_for_stop_anchor() -> None:
    """When a trail_high is provided, the stop anchors to it, not the signal price.

    AAPL signal price today is $200 (yesterday's close). Trail high from
    a prior up-leg is $250. With a 5% stop, the trailing stop sits at
    $250 * 0.95 = $237.50 — protecting most of the gain from the move
    up from $200 to $250.

    Note: $237.50 > signal $200, so the rebalance's stop_price>=signal
    guard fires and the entry is refused — the position would be
    immediately stopped out if we re-entered. That's the correct
    trailing-stop behavior: if the stock has retraced enough that the
    trailing stop would fire, exit and stay flat.
    """
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        trail_highs={"AAPL": 250.0},
        dry_run=False,
    )
    failed = [o for o in report.submitted_orders if o.status == "failed"]
    assert len(failed) >= 1
    assert "stop_price" in failed[0].error


def test_daily_rebalance_trail_high_below_signal_uses_it() -> None:
    """When trail_high*0.95 < signal_price, the entry proceeds with trailing stop."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    # AAPL signal = $200, trail_high = $210 → trailing stop = $210*0.95 = $199.50
    # $199.50 < $200 (signal), so the entry passes the >=guard.
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        trail_highs={"AAPL": 210.0},
        dry_run=True,
    )
    stop_row = next(o for o in report.submitted_orders if o.role == "stop_loss")
    # 210 * 0.95 = 199.5  (NOT 200 * 0.95 = 190 — the trail anchors the stop)
    assert stop_row.stop_price == 199.50


def test_daily_rebalance_tighter_trail_pct_locks_in_more_gain() -> None:
    """trail_pct < stop_loss_pct → trailing stop sits closer to the running high."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    # Trail high = $250, trail_pct = 0.03 → stop = $250 * 0.97 = $242.50.
    # Signal price = $245 (slight retrace) → stop $242.50 < $245, entry OK.
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 245.0},
        stop_loss_pct=0.05,
        trail_highs={"AAPL": 250.0},
        trail_pct=0.03,                 # tighter than the 5% initial stop
        dry_run=True,
    )
    stop_row = next(o for o in report.submitted_orders if o.role == "stop_loss")
    # 250 * 0.97 = 242.5 (tighter than 250 * 0.95 = 237.5)
    assert stop_row.stop_price == 242.50


def test_daily_rebalance_rejects_trail_pct_wider_than_stop_loss() -> None:
    """trail_pct > stop_loss_pct violates the operator's per-trade floor; raise."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    with pytest.raises(ValueError, match="trail_pct must be in"):
        exec_.submit_daily_rebalance(
            target_weights={"AAPL": 0.10},
            signal_prices={"AAPL": 200.0},
            stop_loss_pct=0.05,
            trail_pct=0.07,            # > 0.05 ceiling
            dry_run=True,
        )


def test_daily_rebalance_rejects_zero_or_negative_trail_pct() -> None:
    """trail_pct must be strictly positive."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    for bad in [0.0, -0.01]:
        with pytest.raises(ValueError, match="trail_pct must be in"):
            exec_.submit_daily_rebalance(
                target_weights={"AAPL": 0.10},
                signal_prices={"AAPL": 200.0},
                stop_loss_pct=0.05,
                trail_pct=bad,
                dry_run=True,
            )


def test_daily_rebalance_fresh_entry_uses_stop_loss_pct_even_with_trail_pct() -> None:
    """For names without a trail_high, trail_pct is ignored — fresh entry uses
    stop_loss_pct (operator's 5% floor). Only existing trails use trail_pct."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        trail_highs={},               # no trail for AAPL → fresh entry
        trail_pct=0.03,                # would be tighter, but doesn't apply
        dry_run=True,
    )
    stop_row = next(o for o in report.submitted_orders if o.role == "stop_loss")
    # 200 * 0.95 = 190 (NOT 200 * 0.97 = 194; trail_pct ignored on fresh entry)
    assert stop_row.stop_price == 190.0


def test_daily_rebalance_signal_anchored_stop_when_no_trail_high() -> None:
    """No trail_high provided → behaves exactly like the legacy entry-stop."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        trail_highs=None,                  # legacy path
        dry_run=True,
    )
    stop_row = next(o for o in report.submitted_orders if o.role == "stop_loss")
    # 200 * 0.95 = 190 (signal-anchored)
    assert stop_row.stop_price == 190.0


def test_daily_rebalance_trail_highs_only_applies_when_sym_present() -> None:
    """Names in target_weights but NOT in trail_highs fall back to signal-anchored."""
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    # AAPL has trail_high; MSFT doesn't (e.g., MSFT is a fresh entry).
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.05, "MSFT": 0.05},
        signal_prices={"AAPL": 200.0, "MSFT": 400.0},
        stop_loss_pct=0.05,
        trail_highs={"AAPL": 210.0},   # AAPL only
        dry_run=True,
    )
    aapl_stop = next(o for o in report.submitted_orders
                     if o.role == "stop_loss" and o.symbol == "AAPL").stop_price
    msft_stop = next(o for o in report.submitted_orders
                     if o.role == "stop_loss" and o.symbol == "MSFT").stop_price
    # AAPL uses trail; MSFT uses signal.
    assert aapl_stop == 199.50            # 210 * 0.95
    assert msft_stop == 380.0             # 400 * 0.95


def test_daily_rebalance_closes_held_target_even_when_qty_unchanged() -> None:
    """Held position with target_qty == current_qty MUST still be closed-and-reopened.

    Step 1 cancels yesterday's GTC stops, so any in-target position needs
    a fresh stop attached via the OTO bracket. If we skipped the close
    when qty matched, step 3's unconditional buy would DOUBLE the position
    (broker has 50; step 3 buys 50 more → 100). This test pins down the
    correct close-and-reopen invariant.
    """
    # Equity 100k, weight 0.10, price $200 → target_qty = 50.
    # Current_qty also 50 → bug case.
    client = _FakeTradingClient(equity=100_000, positions={"AAPL": 50})
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        dry_run=True,
    )
    # We expect: 1 sell (close), 1 buy (re-open with bracket), 1 stop_loss audit row.
    by_role = {}
    for o in report.submitted_orders:
        by_role.setdefault((o.role, o.side), []).append(o)
    # The close-out sell goes through the bare-entry path with role='entry'.
    assert ("entry", "sell") in by_role, (
        f"expected a sell row for the close; got roles: {sorted(by_role)}"
    )
    assert by_role[("entry", "sell")][0].qty == 50
    # The re-entry buy + stop_loss audit row.
    assert ("entry", "buy") in by_role
    assert by_role[("entry", "buy")][0].qty == 50
    assert ("stop_loss", "sell") in by_role
    assert by_role[("stop_loss", "sell")][0].qty == 50
