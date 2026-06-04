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

    # --- Protocol methods used by the post-fill stop-repair phase ---
    # `_orders_db` lets a test pre-stage what get_order_by_id returns,
    # so we can simulate gap-up fills without a real broker.
    _orders_db: dict[str, object] = {}        # class attribute as default
    _repair_cancels: list[str] = []            # set per-instance via setattr

    def get_order_by_id(self, order_id: str) -> object:
        return self._orders_db.get(order_id, None)

    def cancel_order_by_id(self, order_id: str) -> None:
        # Per-instance log (override _repair_cancels in tests).
        if hasattr(self, "_instance_repair_cancels"):
            self._instance_repair_cancels.append(order_id)

    def get_orders(self, *, filter=None) -> list:
        return []


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


def test_selective_cancel_preserves_manual_orders() -> None:
    """T3.16 — orders WITHOUT our agent tag, on symbols we don't hold,
    must NOT be cancelled by the daily rebalance."""
    from types import SimpleNamespace

    # Pre-stage open orders: one ours-tagged, one manual on a foreign symbol.
    open_orders_db = [
        SimpleNamespace(
            id="agent-order-1",
            client_order_id="qagent-oto-entry-AAPL-12345",
            symbol="AAPL",
            order_type=SimpleNamespace(value="market"),
        ),
        SimpleNamespace(
            id="manual-order-1",
            client_order_id="manual-trade-by-operator",
            symbol="GOLD",   # symbol we don't trade
            order_type=SimpleNamespace(value="limit"),
        ),
    ]
    client = _FakeTradingClient(equity=100_000)
    client.get_orders = lambda **kw: list(open_orders_db)
    cancelled_ids: list[str] = []
    client.cancel_order_by_id = lambda oid: cancelled_ids.append(oid)

    exec_ = AlpacaExecutor(trading_client=client)
    n, err = exec_._cancel_agent_orders()
    assert err is None
    # Our tagged order cancelled; manual order preserved.
    assert "agent-order-1" in cancelled_ids
    assert "manual-order-1" not in cancelled_ids
    assert n == 1


def test_selective_cancel_catches_oto_child_stops_via_position_match() -> None:
    """OTO child stops get broker-generated client_order_ids that we
    can't tag. The fallback heuristic is: if it's a stop order on a
    symbol we currently hold, treat it as one of ours."""
    from types import SimpleNamespace

    open_orders_db = [
        SimpleNamespace(
            id="oto-child-stop-1",
            client_order_id="broker-generated-xyz",   # no agent tag
            symbol="AAPL",                            # but we hold AAPL
            order_type=SimpleNamespace(value="stop"),
        ),
        SimpleNamespace(
            id="manual-stop-on-foreign-symbol",
            client_order_id="manual-xyz",
            symbol="TSLA",                            # we don't hold TSLA
            order_type=SimpleNamespace(value="stop"),
        ),
    ]
    client = _FakeTradingClient(equity=100_000, positions={"AAPL": 10})
    client.get_orders = lambda **kw: list(open_orders_db)
    cancelled_ids: list[str] = []
    client.cancel_order_by_id = lambda oid: cancelled_ids.append(oid)

    exec_ = AlpacaExecutor(trading_client=client)
    n, err = exec_._cancel_agent_orders()
    assert err is None
    # AAPL stop cancelled (we hold AAPL → it's our OTO child); TSLA
    # stop preserved (we don't hold TSLA → it's the operator's).
    assert "oto-child-stop-1" in cancelled_ids
    assert "manual-stop-on-foreign-symbol" not in cancelled_ids
    assert n == 1


def test_kept_stop_failure_triggers_emergency_close() -> None:
    """T3.14 — if the standalone stop submission fails for a kept
    position, the executor MUST emergency-close the position rather
    than leave it unprotected for 24 hours."""
    # Simulate: AAPL is held + in targets at same qty; the stop
    # submission raises; we expect a follow-up market sell.
    client = _FakeTradingClient(equity=100_000, positions={"AAPL": 50})
    # Make the FIRST submit raise (the stop arming). The SECOND submit
    # (the emergency close) should succeed.
    submit_count = [0]
    original_submit = client.submit_order
    def _submit_with_first_failure(req):
        submit_count[0] += 1
        if submit_count[0] == 1:
            raise RuntimeError("simulated stop-arm rejection")
        return original_submit(req)
    client.submit_order = _submit_with_first_failure

    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},    # → qty 50 (kept)
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
    )
    # The stop_loss row should be marked failed; an entry row should
    # show the emergency close as submitted (or failed, but trying).
    stop_rows = [o for o in report.submitted_orders if o.role == "stop_loss"]
    entry_close_rows = [
        o for o in report.submitted_orders
        if o.role == "entry" and o.side == "sell"
        and o.error and "stop failed" in (o.error or "")
    ]
    # At least one failed stop-arm and one corresponding emergency close.
    assert any(o.status == "failed" for o in stop_rows)
    assert len(entry_close_rows) == 1, (
        "Expected exactly one emergency close after stop-arm failure; "
        f"got {len(entry_close_rows)}: {entry_close_rows}"
    )


def test_daily_rebalance_calls_selective_cancel_not_broad_cancel() -> None:
    """T3.16 — the executor must use SELECTIVE cancellation (filter by
    agent tag / position symbol) rather than the indiscriminate
    cancel_orders() which would kill operator's manual orders too.

    Regression guard: if anyone reverts to self._client.cancel_orders(),
    this test screams.
    """
    client = _FakeTradingClient(equity=100_000)
    # Track which broad-cancel calls happen.
    cancel_orders_calls = 0
    orig_cancel = client.cancel_orders
    def _wrap():
        nonlocal cancel_orders_calls
        cancel_orders_calls += 1
        orig_cancel()
    client.cancel_orders = _wrap

    exec_ = AlpacaExecutor(trading_client=client)
    exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.1},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
    )
    # Selective path: get_orders() + cancel_order_by_id() per match.
    # Broad cancel_orders() must NOT be called.
    assert cancel_orders_calls == 0, (
        f"Broad cancel_orders() called {cancel_orders_calls}x — "
        "would kill operator's manual orders. Use _cancel_agent_orders."
    )


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


def test_daily_rebalance_trims_oversize_orders_to_cap() -> None:
    """T-audit fix H6: the 20%-of-equity per-trade cap TRIMS the order
    quantity rather than refusing it. The original behaviour silently
    dropped high-conviction signals (e.g., xsec's top-momentum name ×
    HRP > 20% cap) → strongest bets got zero allocation. Now we cap
    the qty and the bet still ships, just bounded by policy.
    """
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    # Target 30% in AAPL — exceeds 20% default cap; gets trimmed to 20%.
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.30},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        max_position_weight=0.20,
        dry_run=False,
    )
    # No "failed" entry rows — the order ships at the trimmed qty.
    failed_entries = [
        o for o in report.submitted_orders
        if o.status == "failed" and o.role == "entry"
    ]
    assert failed_entries == [], (
        f"expected zero failed entry rows after cap-trim, got: {failed_entries}"
    )
    # The entry was submitted at the capped qty.
    entries = [
        o for o in report.submitted_orders
        if o.role == "entry" and o.status == "submitted"
    ]
    assert len(entries) == 1
    # 20% × $100k / $200 = 100 shares.
    assert entries[0].qty == 100


def test_daily_rebalance_refuses_only_when_single_share_exceeds_cap() -> None:
    """When even ONE share's notional exceeds the cap (e.g., super-high
    price on a tiny account), we still refuse — there's no trim path."""
    # $1000 share, $5000 cap → can't even buy 1 share at $1000 ≤ $5000? wait,
    # $1000 < $5000 so 1 share fits. Build the impossible case explicitly.
    client = _FakeTradingClient(equity=10_000)
    exec_ = AlpacaExecutor(trading_client=client)
    # 20% × $10k = $2k cap. A $3000/share name can't fit even 1 share.
    report = exec_.submit_daily_rebalance(
        target_weights={"BRKA": 1.0},
        signal_prices={"BRKA": 3_000.0},
        stop_loss_pct=0.05,
        max_position_weight=0.20,
        dry_run=False,
    )
    failed = [o for o in report.submitted_orders if o.status == "failed"]
    assert any("single share" in (o.error or "").lower() for o in failed), (
        f"expected single-share refusal, got: {[o.error for o in failed]}"
    )


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
    # New refusal message phrasing: mentions "trail-anchored stop ... >= signal"
    assert "trail-anchored stop" in failed[0].error
    assert ">= signal" in failed[0].error


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


def test_post_fill_stop_repair_reanchors_on_gap_up() -> None:
    """When the OTO entry fills well above the signal price (gap-up), the
    repair phase cancels the auto-attached stop and submits a new one
    anchored to the actual fill price.

    Without this, a 10% gap up means the stop is at signal*0.95 = 13.6%
    below fill — silently violating the 5% per-trade floor.
    """
    from types import SimpleNamespace
    client = _FakeTradingClient(equity=100_000)
    # Pre-stage what get_order_by_id will return: filled at $220 vs signal $200.
    # The OTO parent order ID is generated by the fake client during submit.
    client._instance_repair_cancels = []  # type: ignore[attr-defined]

    # Wrap submit_order to capture the parent ID for the next get_order_by_id call.
    real_submit = client.submit_order
    def _spy_submit(req):
        resp = real_submit(req)
        # If this is the OTO parent (has order_class=OTO), stage the fill response.
        oc = getattr(req, "order_class", None)
        if oc is not None and getattr(oc, "value", "") == "oto":
            # Build a fake parent order: filled at $220, with a stop child.
            child = SimpleNamespace(
                id="child-stop-123",
                order_type=SimpleNamespace(value="stop"),
            )
            client._orders_db[resp.id] = SimpleNamespace(
                id=resp.id,
                status=SimpleNamespace(value="filled"),
                filled_avg_price="220.00",
                legs=[child],
            )
        return resp
    client.submit_order = _spy_submit

    exec_ = AlpacaExecutor(trading_client=client)
    exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        repair_stops_after_fill=True,
        fill_wait_seconds=0.0,    # synchronous
    )

    # The original OTO bracket fired with stop at $200 * 0.95 = $190.
    # After the repair: the child ("child-stop-123") should be cancelled,
    # AND a new StopOrderRequest submitted at $220 * 0.95 = $209.
    assert client._instance_repair_cancels == ["child-stop-123"]   # type: ignore[attr-defined]
    # Find the newly-submitted stop order (the last one after the OTO).
    from alpaca.trading.requests import StopOrderRequest
    new_stops = [
        r for r in client.submitted
        if isinstance(r, StopOrderRequest) and r.symbol == "AAPL"
    ]
    assert len(new_stops) == 1
    assert new_stops[0].stop_price == 209.0


def test_post_fill_repair_skipped_when_fill_close_to_signal() -> None:
    """Fill within 1% of signal → don't bother re-anchoring; the auto-attached
    stop is good enough."""
    from types import SimpleNamespace
    client = _FakeTradingClient(equity=100_000)
    client._instance_repair_cancels = []  # type: ignore[attr-defined]

    real_submit = client.submit_order
    def _spy_submit(req):
        resp = real_submit(req)
        oc = getattr(req, "order_class", None)
        if oc is not None and getattr(oc, "value", "") == "oto":
            # Fill at $200.50 — only 0.25% above signal.
            child = SimpleNamespace(
                id="child-stop-NA",
                order_type=SimpleNamespace(value="stop"),
            )
            client._orders_db[resp.id] = SimpleNamespace(
                id=resp.id,
                status=SimpleNamespace(value="filled"),
                filled_avg_price="200.50",
                legs=[child],
            )
        return resp
    client.submit_order = _spy_submit

    exec_ = AlpacaExecutor(trading_client=client)
    exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        repair_stops_after_fill=True,
        fill_wait_seconds=0.0,
    )
    # No cancellations; no new standalone stops.
    assert client._instance_repair_cancels == []   # type: ignore[attr-defined]
    from alpaca.trading.requests import StopOrderRequest
    new_stops = [r for r in client.submitted if isinstance(r, StopOrderRequest)]
    assert new_stops == []


def test_per_symbol_stop_pcts_override_flat_default() -> None:
    """When ``stop_pcts`` is supplied, per-symbol stop overrides apply.

    Used to wire ATR-normalized stops: low-vol KO might get 3% while
    high-vol NVDA stays at the 5% flat floor.
    """
    client = _FakeTradingClient(equity=100_000)
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"KO": 0.10, "NVDA": 0.10},
        signal_prices={"KO": 100.0, "NVDA": 200.0},
        stop_loss_pct=0.05,           # flat fallback
        stop_pcts={"KO": 0.03},       # tighter override on KO
        dry_run=True,
    )
    by_sym = {(o.symbol, o.role): o for o in report.submitted_orders}
    # KO uses the per-symbol override (3% → stop at $100 * 0.97 = $97).
    assert by_sym[("KO", "stop_loss")].stop_price == 97.0
    # NVDA falls back to the flat default (5% → stop at $200 * 0.95 = $190).
    assert by_sym[("NVDA", "stop_loss")].stop_price == 190.0


def test_unchanged_position_makes_no_buy_and_no_sell() -> None:
    """CRITICAL REGRESSION GUARD: an in-target position with target_qty ==
    current_qty must NOT trigger any buy or sell — only a standalone GTC
    stop re-arm.

    If this test ever breaks because someone adds an unconditional close
    or unconditional buy in submit_daily_rebalance, the result is wash
    trading on every unchanged position — paying the bid-ask spread
    twice per day on every name we're already holding correctly. On
    a real-capital account that's a meaningful return drag.

    Failure modes this catches:
      - the original close-and-reopen-always design (sell + buy + stop)
      - the doubling bug fixed in commit 07a92a0 (no close + unconditional buy)
      - any future "let's just re-cancel and re-open everything" shortcut

    The correct flow for an unchanged position is exactly ONE order:
    a standalone GTC stop at the (possibly higher) trail level. No buy
    at the broker, no sell at the broker.
    """
    # Equity 100k, weight 0.10, price $200 → target_qty = 50.
    # Current_qty also 50 → "kept" case.
    client = _FakeTradingClient(equity=100_000, positions={"AAPL": 50})
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
    )

    # --- Hard invariants ---
    real_submissions = [
        o for o in report.submitted_orders if o.status == "submitted"
    ]
    # Exactly ONE real broker order: the standalone stop.
    assert len(real_submissions) == 1, (
        f"unchanged position should submit exactly 1 order (the stop); "
        f"got {len(real_submissions)}: {real_submissions}"
    )
    assert real_submissions[0].role == "stop_loss"
    assert real_submissions[0].side == "sell"
    assert real_submissions[0].qty == 50

    # NO sells of AAPL at the broker (would be a wash trade).
    aapl_sells = [
        o for o in real_submissions
        if o.symbol == "AAPL" and o.side == "sell" and o.role == "entry"
    ]
    assert aapl_sells == [], (
        f"WASH TRADE REGRESSION: unchanged AAPL position generated a "
        f"sell order: {aapl_sells}. This is exactly what the signal-driven "
        "rebalance refactor exists to prevent."
    )

    # NO buys of AAPL at the broker either.
    aapl_buys = [
        o for o in real_submissions
        if o.symbol == "AAPL" and o.side == "buy"
    ]
    assert aapl_buys == [], (
        f"WASH TRADE REGRESSION: unchanged AAPL position generated a "
        f"buy order: {aapl_buys}."
    )

    # The audit-trail row still shows the position carried forward, with
    # status="kept", so the daily report and audit qty-checks still work.
    kept_rows = [o for o in report.submitted_orders if o.status == "kept"]
    assert len(kept_rows) == 1
    assert kept_rows[0].symbol == "AAPL"
    assert kept_rows[0].qty == 50


def test_resized_position_does_close_and_reopen() -> None:
    """Held position with target_qty != current_qty SHOULD close + re-open.

    Counterpart to the "no wash trade" test: when qty actually needs to
    change, the close-and-reopen path still fires (sell, then OTO).
    """
    # Equity 100k, weight 0.20, price $200 → target_qty = 100.
    # Current_qty 50 → resize case.
    client = _FakeTradingClient(equity=100_000, positions={"AAPL": 50})
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.20},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        dry_run=True,
    )
    by_role = {}
    for o in report.submitted_orders:
        by_role.setdefault((o.role, o.side), []).append(o)
    # Close-out sell of the existing 50.
    assert by_role[("entry", "sell")][0].qty == 50
    # New OTO bracket buy of 100.
    assert by_role[("entry", "buy")][0].qty == 100
    # New stop on the 100 shares.
    assert by_role[("stop_loss", "sell")][0].qty == 100


def test_dropped_from_targets_flattens_position() -> None:
    """A name held yesterday but absent from today's target_weights is
    flatted — sell to 0, no re-open."""
    client = _FakeTradingClient(equity=100_000, positions={"OLDPOS": 30})
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 200.0, "OLDPOS": 50.0},
        stop_loss_pct=0.05,
    )
    real = [o for o in report.submitted_orders if o.status == "submitted"]
    # OLDPOS sell-to-flat + AAPL OTO bracket entry + AAPL stop child = 3.
    syms_sold = [o.symbol for o in real if o.side == "sell" and o.role == "entry"]
    assert syms_sold == ["OLDPOS"]


def test_forced_exit_when_trail_stop_above_signal() -> None:
    """If the trail-anchored stop would fire immediately (stock has
    retraced past the trail level), close the position and DON'T re-enter.

    Example: AAPL was held with trail_high=$250. Today signal is $200,
    trail_pct=0.05 → stop would be $237.50. Stop above signal means stop
    would fire on entry. Correct behavior: sell, no re-buy.
    """
    client = _FakeTradingClient(equity=100_000, positions={"AAPL": 50})
    exec_ = AlpacaExecutor(trading_client=client)
    report = exec_.submit_daily_rebalance(
        target_weights={"AAPL": 0.10},
        signal_prices={"AAPL": 200.0},
        stop_loss_pct=0.05,
        trail_highs={"AAPL": 250.0},
    )
    real = [o for o in report.submitted_orders if o.status == "submitted"]
    # Exactly one real submission: the sell-to-flat.
    assert len(real) == 1
    assert real[0].side == "sell" and real[0].role == "entry"
    assert real[0].symbol == "AAPL"
    # The "buy" row exists but as a failed/refused audit entry, not a
    # real broker submission.
    refused = [
        o for o in report.submitted_orders
        if o.status == "failed" and o.symbol == "AAPL" and o.side == "buy"
    ]
    assert len(refused) == 1
    assert "stop" in refused[0].error.lower()
