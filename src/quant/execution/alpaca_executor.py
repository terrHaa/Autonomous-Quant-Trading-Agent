"""alpaca_executor.py — submit orders to Alpaca to match target weights.

This is where the platform stops being a research tool and starts being a
trading tool. Everything before this module is pure: data, backtests,
metrics. This module *makes the broker do things*.

Two design priorities:

1. **It's nearly impossible to accidentally trade live.** The constructor
   only opens a live connection if you pass ``env="live"`` AND
   ``i_mean_it_live=True``. The default config ships with paper. Even an
   honest typo doesn't put real money in motion.

2. **The diff is computed once and shown to you in dry-run before any
   network call modifies state.** Caller passes target weights and the
   most recent signal prices; we fetch current Alpaca positions, compute
   the share-count deltas, and return an ``ExecutionReport``. With
   ``dry_run=True``, no orders go out — you get the report and can
   eyeball before the real run.

What this is NOT (yet)
----------------------
- **Pre-trade gating.** Real desks check market hours, halted symbols,
  position-limit rules with the broker, etc., before submitting. We
  don't yet. Run during market hours; expect rejections from Alpaca to
  show up as failed-submission rows in the report.
- **Smart routing.** Every order is a market order, qty-based. No
  limits, no IOC, no algo orders. The engine's spec supports those for
  backtests; live execution will need a separate iteration.
- **Reconciliation across sessions.** Each call is one-shot — submits
  orders, returns the report, done. If Alpaca's view of positions
  drifts from yours between sessions, that's on the caller to detect
  (usually by snapshotting on session start and comparing).
- **Partial-fill handling.** We submit market orders and assume they
  fill at the broker's discretion. The report says "submitted"; it
  doesn't poll for final fill status. Add that when you have a strategy
  that genuinely cares about post-submission state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from quant.data.alpaca_client import AlpacaCredentials

logger = logging.getLogger(__name__)

Env = Literal["paper", "live"]


@dataclass(frozen=True)
class ProposedOrder:
    """One row of what the executor wants to submit, before submission."""

    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    rationale: str   # human-readable explanation: "delta from 0 to 100", "flatten"


@dataclass(frozen=True)
class SubmittedOrder:
    """One row of what the executor actually submitted (or attempted to)."""

    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    # 'status' values:
    #   submitted       — order went to the broker successfully
    #   failed          — broker rejected (or local guard refused)
    #   skipped_dry_run — dry_run=True; no broker contact
    #   kept            — position carried forward from yesterday at the
    #                     SAME qty. NO buy was placed at the broker; only
    #                     a fresh standalone stop. The "kept" entry row
    #                     exists so the daily report can render carried-
    #                     forward positions and the audit's qty cross-
    #                     check still passes.
    status: Literal["submitted", "failed", "skipped_dry_run", "kept"]
    alpaca_order_id: str | None = None
    # 'role' tells reports the difference between the entry order and
    # its protective stop-loss child. ``"entry"`` is the default for
    # the existing submit_to_match_targets flow.
    role: Literal["entry", "stop_loss"] = "entry"
    stop_price: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class ExecutionReport:
    """Structured record of one executor invocation.

    Goes back to the caller as the source of truth for "what we did this
    session." Frozen — the registry / reports module will pin reports by
    hash so any post-hoc mutation invalidates the audit trail.
    """

    env: Env
    timestamp: datetime
    account_equity_before: float
    positions_before: dict[str, int]
    target_weights: dict[str, float]
    proposed_orders: list[ProposedOrder]
    submitted_orders: list[SubmittedOrder]
    dry_run: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Trading-client Protocol — minimal surface, easy to mock for tests.
# Real implementation: alpaca.trading.client.TradingClient.
# ---------------------------------------------------------------------------


class _TradingClientLike(Protocol):
    """The bits of alpaca.trading.client.TradingClient we actually use.

    Defined as a Protocol (duck-typed) so test stubs don't need to
    inherit anything — they just need methods of these shapes.
    """

    def get_account(self) -> Any: ...
    def get_all_positions(self) -> list[Any]: ...
    def submit_order(self, request: Any) -> Any: ...
    def cancel_orders(self) -> Any: ...
    # Used by the post-fill stop-repair phase to re-anchor stops to actual
    # fill prices when the broker gaps between the signal close and the
    # market open. See _repair_oto_stops_to_fill_price at module bottom.
    def get_order_by_id(self, order_id: str) -> Any: ...
    def cancel_order_by_id(self, order_id: str) -> Any: ...
    def get_orders(self, *, filter: Any = None) -> list[Any]: ...


# ---------------------------------------------------------------------------
# AlpacaExecutor — the main class.
# ---------------------------------------------------------------------------


class AlpacaExecutor:
    """Submits orders to Alpaca to align live positions with target weights.

    Usage:
        executor = AlpacaExecutor()   # defaults to paper from .env
        report = executor.submit_to_match_targets(
            target_weights={"AAPL": 0.5, "MSFT": 0.5},
            signal_prices={"AAPL": 185.00, "MSFT": 400.00},
            dry_run=True,             # SEE the diff before submitting
        )
        print(report)
        # If happy:
        report = executor.submit_to_match_targets(..., dry_run=False)
    """

    def __init__(
        self,
        credentials: AlpacaCredentials | None = None,
        *,
        env: Env = "paper",
        i_mean_it_live: bool = False,
        trading_client: _TradingClientLike | None = None,
    ) -> None:
        """Connect to Alpaca's paper or live trading API.

        Parameters
        ----------
        credentials
            Optional ``AlpacaCredentials``. If None, loads from .env using
            the requested ``env`` (so ``env="paper"`` picks paper keys).
        env
            "paper" (default) or "live". The paper environment is a
            separate sandbox with fake money; live trades real capital.
        i_mean_it_live
            **Must** be True if env="live". A trip-wire against typos
            and dev-environment slip-ups.
        trading_client
            Inject a pre-built (or mock) trading client. Mostly for
            tests. Production code leaves this None.
        """
        if env not in ("paper", "live"):
            raise ValueError(f"env must be 'paper' or 'live'; got {env!r}")
        if env == "live" and not i_mean_it_live:
            raise PermissionError(
                "Live trading requires i_mean_it_live=True. This is "
                "deliberate friction to prevent accidental real-money runs. "
                "Pass i_mean_it_live=True only if you've genuinely audited "
                "the code path for this session."
            )

        if trading_client is not None:
            # Test path: caller supplies a stub/mock that satisfies the
            # _TradingClientLike Protocol.
            self._client: _TradingClientLike = trading_client
        else:
            # Real path: build an alpaca-py TradingClient. Import inside
            # __init__ so tests that pass a mock don't need alpaca's
            # network deps at import time.
            from alpaca.trading.client import TradingClient

            creds = credentials or AlpacaCredentials.from_env(env=env)
            self._client = TradingClient(
                api_key=creds.api_key,
                secret_key=creds.api_secret,
                paper=(env == "paper"),
            )
        self._env: Env = env

    # ------------------------------------------------------------------
    # Read-only views of broker state
    # ------------------------------------------------------------------

    @property
    def env(self) -> Env:
        return self._env

    def get_equity(self) -> float:
        """Account equity (cash + mark-to-market positions) in USD."""
        return float(self._client.get_account().equity)

    def get_positions(self) -> dict[str, int]:
        """Current positions as ``{symbol: signed_qty}``.

        Returns whole-share integer quantities. Alpaca supports fractional
        shares; we floor here for consistency with the rest of the
        platform (the engine also works in integer shares).
        """
        positions: dict[str, int] = {}
        for p in self._client.get_all_positions():
            # Alpaca returns signed strings for quantities ("100", "-50").
            qty = int(float(p.qty))
            if qty != 0:
                positions[p.symbol] = qty
        return positions

    # ------------------------------------------------------------------
    # The main entry point
    # ------------------------------------------------------------------

    def submit_to_match_targets(
        self,
        target_weights: dict[str, float],
        signal_prices: dict[str, float],
        *,
        dry_run: bool = False,
        notes: str = "",
        i_understand_no_stops: bool = False,
    ) -> ExecutionReport:
        """Bring current positions to match the target weight book.

        .. warning::

           **THIS METHOD BYPASSES THE 5% STOP-LOSS RULE.** It submits
           bare market orders without atomic protective stops. The
           autonomous agent uses :meth:`submit_daily_rebalance` instead,
           which attaches a GTC stop to every entry. This method exists
           for the original pre-agent walk-forward research flow and
           ad-hoc operator scripts that explicitly do their own risk
           management.

           Pass ``i_understand_no_stops=True`` to actually submit live;
           otherwise a ``PermissionError`` is raised before any order
           hits the broker. ``dry_run=True`` still works without the
           flag (no orders submitted, just the diff report).

        Parameters
        ----------
        target_weights
            Per-symbol target weight (fraction of equity). Positive =
            long, negative = short. Symbols absent from the dict are
            treated as "go flat" if currently held.
        signal_prices
            Per-symbol price used to convert weights to share counts.
            Typically the latest cached close. The actual fill happens
            at the broker's market price, so this is only for sizing.
        dry_run
            If True, computes the diff and returns the report with
            ``submitted_orders`` showing ``status="skipped_dry_run"``.
            No network mutation. Always run dry-run once before a real
            submit.
        notes
            Free-text note saved on the report (e.g., "session 2024-12-31
            after walk-forward pass").
        i_understand_no_stops
            Explicit opt-in to bypass the 5% stop rule. Required for any
            non-dry-run submission. Catches the case where someone wires
            this method into a production flow by mistake.

        Returns
        -------
        ExecutionReport
            Full structured record of what was proposed and submitted.
        """
        if not dry_run and not i_understand_no_stops:
            raise PermissionError(
                "submit_to_match_targets bypasses the 5% stop-loss rule. "
                "Pass i_understand_no_stops=True to acknowledge, or use "
                "submit_daily_rebalance for the agent's safe flow with "
                "atomic GTC stops."
            )
        # ---- 1. Snapshot account + positions ---------------------------
        equity = self.get_equity()
        positions = self.get_positions()
        now = datetime.now(UTC)

        # ---- 2. Compute proposed orders --------------------------------
        proposed = _compute_proposed_orders(
            target_weights=target_weights,
            signal_prices=signal_prices,
            current_positions=positions,
            equity=equity,
        )

        # ---- 3. Submit (or skip if dry_run) ----------------------------
        submitted: list[SubmittedOrder] = []
        for order in proposed:
            if dry_run:
                submitted.append(SubmittedOrder(
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    status="skipped_dry_run",
                ))
                continue
            try:
                # Import lazily so the import isn't required for dry-run-only
                # users (e.g., during tests that don't touch real submit).
                from alpaca.trading.enums import OrderSide, TimeInForce
                from alpaca.trading.requests import MarketOrderRequest

                req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                resp = self._client.submit_order(req)
                submitted.append(SubmittedOrder(
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    status="submitted",
                    alpaca_order_id=str(getattr(resp, "id", None)),
                ))
            except Exception as e:
                # Broker errors (insufficient buying power, halted name,
                # etc.) surface here. We keep going so a single bad order
                # doesn't drop the whole batch.
                submitted.append(SubmittedOrder(
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    status="failed",
                    error=str(e),
                ))

        return ExecutionReport(
            env=self._env,
            timestamp=now,
            account_equity_before=equity,
            positions_before=positions,
            target_weights=dict(target_weights),
            proposed_orders=proposed,
            submitted_orders=submitted,
            dry_run=dry_run,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Daily rebalance for the autonomous agent: cancel old orders,
    # close stale positions, open new positions with atomic stop-losses.
    # ------------------------------------------------------------------

    def submit_daily_rebalance(
        self,
        target_weights: dict[str, float],
        signal_prices: dict[str, float],
        *,
        stop_loss_pct: float,
        max_position_weight: float = 0.20,
        dry_run: bool = False,
        notes: str = "",
        trail_highs: dict[str, float] | None = None,
        trail_pct: float | None = None,
        stop_pcts: dict[str, float] | None = None,
        repair_stops_after_fill: bool = False,
        fill_wait_seconds: float = 30.0,
    ) -> ExecutionReport:
        """Daily-cadence rebalance with hard per-trade cap and stop-loss.

        Workflow (agent-style; opinionated):
          1. Cancel all open orders at the broker (kills yesterday's
             stops that didn't trigger; also kills stale pending entries).
          2. For each currently-held name that is NOT in the target book
             (target weight 0 or absent): close the position at market.
          3. For each target-symbol with weight > 0: submit an Alpaca
             OTO order — market buy + atomic protective stop. The stop
             price is ``signal_price * (1 - stop_loss_pct)``.
          4. Refuse to submit any order whose notional exceeds
             ``max_position_weight × equity``. Hard defense-in-depth
             against the operator's 20% per-trade rule, in case the
             upstream allocator failed to enforce it.

        For partial position changes (already long X, want to be long Y,
        with X ≠ Y), the v1 simplification is "close-and-reopen": we
        liquidate the old position fully, then open a fresh one with a
        new stop. Higher turnover than incrementally adjusting + restops,
        but completely unambiguous about position+stop state at any
        moment. Optimize in v2 if costs become noticeable.

        Long-only. Negative target weights are rejected (they'd require
        OTO with a buy-stop, which is a different child structure;
        out of scope for the cross-sectional momentum agent).

        Parameters
        ----------
        target_weights
            Per-symbol target weight (fraction of equity). Must be >= 0
            for this method; negative weights → ValueError.
        signal_prices
            Per-symbol reference price used for sizing AND for the stop
            level. Typically yesterday's close.
        stop_loss_pct
            Decimal stop distance. ``0.05`` = stop sells at -5% from
            ``signal_price``. The agent picks 0.05.
        max_position_weight
            Per-trade notional cap as a fraction of equity. Defaults to
            0.20 (the operator's 20% rule).
        dry_run
            If True, no network mutation; report shows what WOULD happen.
        notes
            Free-text saved on the report.
        trail_highs
            Optional per-symbol all-time-high price since position opened.
            When provided, the stop level for ``sym`` is computed as
            ``trail_highs[sym] * (1 - trail_pct)`` (or ``stop_loss_pct``
            if trail_pct is None) instead of
            ``signal_prices[sym] * (1 - stop_loss_pct)``. This is the
            trailing-stop mechanism: as a winner ratchets up, its stop
            ratchets up with it; on a flat or down day, the stop stays
            at the prior high. Symbols absent from ``trail_highs`` fall
            back to the signal-price stop (identical to legacy behavior).
        trail_pct
            Optional trailing-stop distance (e.g. 0.03 = 3% below the
            running high). When None, falls back to ``stop_loss_pct``
            (identical behavior to no trailing-stop tuning). Must satisfy
            ``0 < trail_pct <= stop_loss_pct`` if given — a trail wider
            than the initial entry stop would violate the operator's
            single-trade loss floor on a fresh-entry's down day.
        """
        if stop_loss_pct <= 0 or stop_loss_pct >= 1:
            raise ValueError(
                f"stop_loss_pct must be in (0, 1); got {stop_loss_pct}"
            )
        if trail_pct is not None and (trail_pct <= 0 or trail_pct > stop_loss_pct):
            raise ValueError(
                f"trail_pct must be in (0, {stop_loss_pct}]; got {trail_pct}. "
                "A trailing stop wider than the entry stop would violate the "
                "operator's per-trade loss floor on a fresh entry's down day."
            )
        if any(w < 0 for w in target_weights.values()):
            raise ValueError(
                "submit_daily_rebalance is long-only; negative target "
                "weights aren't supported. Got: "
                f"{ {s: w for s, w in target_weights.items() if w < 0} }"
            )

        equity = self.get_equity()
        positions = self.get_positions()
        now = datetime.now(UTC)
        submitted: list[SubmittedOrder] = []
        max_notional = max_position_weight * equity

        # ---- 1. Cancel all open orders. -----------------------------
        # Without this, yesterday's GTC stops would linger and double up
        # with today's. cancel_orders() is idempotent — no-op if nothing
        # is pending.
        if not dry_run:
            try:
                self._client.cancel_orders()
            except Exception as e:
                # Cancellation failure is bad but recoverable — log it
                # on the report and proceed; the broker will refuse
                # duplicate orders rather than silently double-fill.
                submitted.append(SubmittedOrder(
                    symbol="(all)",
                    side="sell",
                    qty=0,
                    status="failed",
                    role="entry",
                    error=f"cancel_orders failed: {e}",
                ))

        # ---- 2. PLAN the day's actions, signal-driven ---------------
        # For each in-target symbol, decide one of four outcomes:
        #   - "kept":    target_qty == current_qty > 0 → no buy, no sell;
        #                just re-arm a standalone GTC stop at the new
        #                (possibly higher) trail level. Saves the spread
        #                round-trip the old close-and-reopen incurred.
        #   - "resize":  current > 0 but target_qty differs → close
        #                current, then re-open at new size via OTO bracket.
        #   - "new":     current == 0 → fresh OTO bracket entry.
        #   - "exit":    trail-anchored stop would fire immediately
        #                (stop_price >= signal_price) → close, don't
        #                re-enter today. Trail logic is saying "we should
        #                already be out".
        #   - "refused": notional cap violated (defense-in-depth on the
        #                20% rule) → emit a failure row, do nothing.
        # Plus: every held name NOT in target_symbols → close-out.
        #
        # The CRITICAL invariant this enforces: NEVER buy and sell the
        # same name in the same run when the position is unchanged. The
        # old "close-and-reopen-always" cost ~50 wash trades/day in
        # steady state — fine on zero-commission paper, real money on a
        # live account. Test
        # `test_unchanged_position_makes_no_buy_and_no_sell` is the
        # explicit regression guard.
        target_symbols = {s for s, w in target_weights.items() if w > 0}
        stale_positions: dict[str, int] = {
            sym: qty for sym, qty in positions.items()
            if sym not in target_symbols and qty != 0
        }
        # plans[sym] is consumed in step 3. Three relevant fields:
        #   action: "kept" | "resize" | "new" | "exit" | "refused"
        #   target_qty, stop_price, signal_price
        # For "refused" the error field is also set.
        plans: dict[str, dict] = {}

        for sym in sorted(target_symbols):
            weight = target_weights[sym]
            price = signal_prices.get(sym)
            if price is None or price <= 0:
                continue
            target_qty = int(weight * equity / price)
            current_qty = positions.get(sym, 0)
            if target_qty <= 0:
                # 0 target after rounding (e.g. tiny weight on a high-price
                # name). If we hold it, close out; otherwise nothing.
                if current_qty > 0:
                    stale_positions[sym] = current_qty
                continue

            # Effective stop distance for this symbol:
            #   1. trail_pct if there's a trail_high (winning position
            #      being maintained)
            #   2. else stop_pcts[sym] if the caller supplied a per-symbol
            #      override (ATR-normalized stops — wider on high-vol
            #      names, tighter on low-vol). Always ≤ stop_loss_pct
            #      since that's the operator's hard cap; the caller is
            #      responsible for enforcing it.
            #   3. else stop_loss_pct (the operator's flat-rate fallback).
            if trail_highs is not None and sym in trail_highs:
                stop_anchor = trail_highs[sym]
                stop_dist = trail_pct if trail_pct is not None else stop_loss_pct
            else:
                stop_anchor = price
                stop_dist = (
                    stop_pcts.get(sym, stop_loss_pct)
                    if stop_pcts is not None
                    else stop_loss_pct
                )
            stop_price = round(stop_anchor * (1.0 - stop_dist), 2)

            # Notional-cap defense (operator's 20% rule).
            entry_notional = target_qty * price
            if entry_notional > max_notional:
                plans[sym] = {
                    "action": "refused",
                    "target_qty": target_qty,
                    "stop_price": stop_price,
                    "signal_price": price,
                    "error": (
                        f"refusing entry: notional ${entry_notional:,.2f} "
                        f"exceeds max_position_weight × equity = "
                        f"${max_notional:,.2f}"
                    ),
                }
                # If we currently hold it, also close (don't maintain).
                if current_qty > 0:
                    stale_positions[sym] = current_qty
                continue

            # Forced exit: trail-stop would fire immediately. Close existing
            # position if any; don't re-enter.
            if stop_price >= price:
                plans[sym] = {
                    "action": "exit",
                    "target_qty": target_qty,
                    "stop_price": stop_price,
                    "signal_price": price,
                }
                if current_qty > 0:
                    stale_positions[sym] = current_qty
                continue

            # Kept: same qty as currently held → just re-arm the stop.
            if current_qty == target_qty:
                plans[sym] = {
                    "action": "kept",
                    "target_qty": target_qty,
                    "stop_price": stop_price,
                    "signal_price": price,
                }
                continue

            # Resize: held but at wrong qty → close + reopen.
            if current_qty > 0:
                stale_positions[sym] = current_qty
                plans[sym] = {
                    "action": "resize",
                    "target_qty": target_qty,
                    "stop_price": stop_price,
                    "signal_price": price,
                }
                continue

            # New entry: not held → OTO bracket.
            plans[sym] = {
                "action": "new",
                "target_qty": target_qty,
                "stop_price": stop_price,
                "signal_price": price,
            }

        for sym, qty in sorted(stale_positions.items()):
            if dry_run:
                submitted.append(SubmittedOrder(
                    symbol=sym, side="sell", qty=abs(qty),
                    status="skipped_dry_run", role="entry",
                ))
                continue
            try:
                from alpaca.trading.enums import OrderSide, TimeInForce
                from alpaca.trading.requests import MarketOrderRequest

                req = MarketOrderRequest(
                    symbol=sym,
                    qty=abs(qty),
                    side=OrderSide.SELL if qty > 0 else OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                resp = self._client.submit_order(req)
                submitted.append(SubmittedOrder(
                    symbol=sym, side="sell" if qty > 0 else "buy",
                    qty=abs(qty), status="submitted",
                    role="entry",
                    alpaca_order_id=str(getattr(resp, "id", None)),
                ))
            except Exception as e:
                submitted.append(SubmittedOrder(
                    symbol=sym, side="sell" if qty > 0 else "buy",
                    qty=abs(qty), status="failed", role="entry",
                    error=str(e),
                ))

        # ---- 3. Execute the plan: OTO for new/resize, standalone stop for kept.
        for sym in sorted(plans):
            p = plans[sym]
            action = p["action"]

            # --- Refused: notional cap blocked the entry. Emit a failure row.
            if action == "refused":
                submitted.append(SubmittedOrder(
                    symbol=sym, side="buy", qty=p["target_qty"],
                    status="failed", role="entry",
                    error=p["error"],
                ))
                continue

            # --- Forced exit: position was already added to stale_positions
            # and sold above. Emit an entry row marked as exit for audit clarity.
            if action == "exit":
                submitted.append(SubmittedOrder(
                    symbol=sym, side="buy", qty=p["target_qty"],
                    status="failed", role="entry",
                    error=(
                        f"trail-anchored stop {p['stop_price']} >= signal "
                        f"{p['signal_price']} — closed without re-entry"
                    ),
                ))
                continue

            target_qty = p["target_qty"]
            stop_price = p["stop_price"]

            # --- Kept: no buy, no sell. Re-arm a standalone GTC stop at the
            # (possibly higher) trail level. This is THE optimization — it
            # replaces the old close-and-reopen wash trade with a single
            # stop-refresh order. Emits a "kept" status entry row so the
            # daily report shows the position carried forward and the audit's
            # qty cross-check still passes.
            if action == "kept":
                if dry_run:
                    submitted.append(SubmittedOrder(
                        symbol=sym, side="buy", qty=target_qty,
                        status="kept", role="entry",
                    ))
                    submitted.append(SubmittedOrder(
                        symbol=sym, side="sell", qty=target_qty,
                        status="skipped_dry_run", role="stop_loss",
                        stop_price=stop_price,
                    ))
                    continue
                try:
                    from alpaca.trading.enums import OrderSide, TimeInForce
                    from alpaca.trading.requests import StopOrderRequest

                    req = StopOrderRequest(
                        symbol=sym,
                        qty=target_qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                        stop_price=stop_price,
                    )
                    resp = self._client.submit_order(req)
                    # No buy row: the position was not touched. Emit a
                    # "kept" audit row so the daily report can render
                    # carried-forward positions and the audit's qty
                    # cross-check (run_qty vs broker pos_qty) still works.
                    submitted.append(SubmittedOrder(
                        symbol=sym, side="buy", qty=target_qty,
                        status="kept", role="entry",
                    ))
                    submitted.append(SubmittedOrder(
                        symbol=sym, side="sell", qty=target_qty,
                        status="submitted", role="stop_loss",
                        stop_price=stop_price,
                        alpaca_order_id=str(getattr(resp, "id", None)),
                    ))
                except Exception as e:
                    submitted.append(SubmittedOrder(
                        symbol=sym, side="sell", qty=target_qty,
                        status="failed", role="stop_loss",
                        stop_price=stop_price, error=str(e),
                    ))
                continue

            # --- New entry OR resize: full OTO bracket (existing path).
            if dry_run:
                submitted.append(SubmittedOrder(
                    symbol=sym, side="buy", qty=target_qty,
                    status="skipped_dry_run", role="entry",
                ))
                submitted.append(SubmittedOrder(
                    symbol=sym, side="sell", qty=target_qty,
                    status="skipped_dry_run", role="stop_loss",
                    stop_price=stop_price,
                ))
                continue

            try:
                from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
                from alpaca.trading.requests import (
                    MarketOrderRequest,
                    StopLossRequest,
                )

                # TIF must be GTC for bracket orders so the stop-loss leg
                # persists overnight. Alpaca's bracket spec requires GTC; if
                # we use DAY, the stop child silently expires at 16:00 ET
                # and tomorrow's open finds the position unprotected. The
                # parent market order fills immediately regardless of TIF,
                # so promoting to GTC has no downside for the entry.
                req = MarketOrderRequest(
                    symbol=sym,
                    qty=target_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.OTO,
                    stop_loss=StopLossRequest(stop_price=stop_price),
                )
                resp = self._client.submit_order(req)
                parent_id = str(getattr(resp, "id", None))
                submitted.append(SubmittedOrder(
                    symbol=sym, side="buy", qty=target_qty,
                    status="submitted", role="entry",
                    alpaca_order_id=parent_id,
                ))
                # The stop-loss child is auto-created by Alpaca. We
                # record an audit row with the stop_price; the child's
                # actual order id is on the parent's `.legs` attribute
                # and discoverable via get_order_by_id later.
                submitted.append(SubmittedOrder(
                    symbol=sym, side="sell", qty=target_qty,
                    status="submitted", role="stop_loss",
                    stop_price=stop_price,
                    alpaca_order_id=f"(child of {parent_id})",
                ))
            except Exception as e:
                submitted.append(SubmittedOrder(
                    symbol=sym, side="buy", qty=target_qty,
                    status="failed", role="entry",
                    error=str(e),
                ))

        # Proposed orders aren't separately built for this flow — the
        # submitted_orders list IS the audit trail. We mirror the entry
        # rows into proposed_orders for downstream consistency with the
        # other submit method.
        proposed = [
            ProposedOrder(symbol=o.symbol, side=o.side, qty=o.qty,
                          rationale=f"daily rebalance ({o.role})")
            for o in submitted
            if o.role == "entry"
        ]

        # ---- 4. Post-fill stop repair ----------------------------------
        # OTO brackets attach stops anchored to SIGNAL price (yesterday's
        # close). If the stock gaps significantly between the signal close
        # and our market-buy fill, the stop is in the wrong place:
        #   - Gap UP: stop is effectively wider than stop_pct (e.g., 13%
        #             instead of 5% — silent floor violation).
        #   - Gap DOWN: stop is above the fill → fires immediately at
        #             whatever the broker can get.
        # This phase polls each entry order's actual fill price and, if
        # drift exceeds the threshold, replaces the auto-attached stop
        # with a new one anchored to the actual fill price.
        #
        # Why post-submission and not pre: we can't know the fill price
        # until the order executes. The OTO at signal-anchored stop is
        # the best "first pass" guarantee — never unprotected — and the
        # repair tightens it to the correct level seconds later.
        if (
            repair_stops_after_fill
            and not dry_run
            and any(o.status == "submitted" and o.role == "entry" for o in submitted)
        ):
            _repair_oto_stops_to_fill_price(
                client=self._client,
                submitted_orders=submitted,
                signal_prices=signal_prices,
                stop_pcts=stop_pcts,
                default_stop_pct=stop_loss_pct,
                wait_seconds=fill_wait_seconds,
            )

        return ExecutionReport(
            env=self._env,
            timestamp=now,
            account_equity_before=equity,
            positions_before=positions,
            target_weights=dict(target_weights),
            proposed_orders=proposed,
            submitted_orders=submitted,
            dry_run=dry_run,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Pure helper — testable in isolation, no network.
# ---------------------------------------------------------------------------


def _compute_proposed_orders(
    *,
    target_weights: dict[str, float],
    signal_prices: dict[str, float],
    current_positions: dict[str, int],
    equity: float,
) -> list[ProposedOrder]:
    """Compute the share-count deltas needed to align positions with weights.

    Logic:
      - For each target symbol with a valid price:
          target_qty = int(weight * equity / price)
        Delta from current; if non-zero, generate a buy or sell order.
      - For each currently-held symbol NOT in target_weights: flatten.
      - Symbols missing from signal_prices are skipped (can't size).
        Caller should notice they're absent — silent skip is intentional
        (matches engine behavior) but logged via the rationale string.

    Returns proposals sorted by symbol for deterministic execution order.
    """
    proposals: list[ProposedOrder] = []

    # Symbols of interest = anything we target OR currently hold.
    symbols = set(target_weights.keys()) | set(current_positions.keys())

    for sym in sorted(symbols):
        target_w = target_weights.get(sym, 0.0)
        current_qty = current_positions.get(sym, 0)

        if target_w == 0.0:
            target_qty = 0
            rationale = (
                f"flatten {current_qty} → 0" if current_qty != 0
                else "no change"
            )
        else:
            price = signal_prices.get(sym)
            if price is None or price <= 0:
                # Skip silently — can't size without a usable price.
                # Caller can see absence by comparing proposed vs targets.
                continue
            target_qty = int(target_w * equity / price)
            rationale = (
                f"target weight {target_w:+.3f} × equity {equity:,.0f} "
                f"/ price {price:.2f} = {target_qty} (currently {current_qty})"
            )

        delta = target_qty - current_qty
        if delta == 0:
            continue

        proposals.append(ProposedOrder(
            symbol=sym,
            side="buy" if delta > 0 else "sell",
            qty=abs(delta),
            rationale=rationale,
        ))
    return proposals


# ---------------------------------------------------------------------------
# Post-fill stop repair — fixes the OTO-bracket gap-day exposure
# ---------------------------------------------------------------------------


def _repair_oto_stops_to_fill_price(
    *,
    client: Any,
    submitted_orders: list[SubmittedOrder],
    signal_prices: dict[str, float],
    stop_pcts: dict[str, float] | None,
    default_stop_pct: float,
    wait_seconds: float = 30.0,
    drift_threshold: float = 0.01,
) -> None:
    """After OTO submissions, re-anchor stops to actual fill prices.

    For each entry order this run submitted:
      1. Wait ``wait_seconds`` for the fill to land at the broker.
      2. Fetch the parent order from Alpaca and read filled_avg_price.
      3. If |filled_avg_price - signal_price| / signal_price >
         drift_threshold (default 1%), the OTO's auto-attached stop is
         in the wrong place. Cancel it and submit a standalone stop
         anchored to filled_avg_price.

    Cases handled gracefully (do nothing):
      - Order still pending after wait_seconds (not yet filled)
      - Order rejected / cancelled / expired
      - Drift within threshold (signal price was close enough)
      - Broker API errors (logged, not raised)

    The repair phase mutates the broker state but does NOT update the
    ``submitted_orders`` list (the original audit trail is preserved).
    A separate "stop_repair" SubmittedOrder row could be added later if
    operators want richer reporting; the current behaviour is to log
    each repair to the standard logger so the .out file captures it.

    Skipped entirely when ``wait_seconds <= 0`` — tests use zero to
    keep the executor synchronous-only.
    """
    import time as _time

    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import StopOrderRequest

    if wait_seconds > 0:
        logger.info(
            "stop repair: waiting %.1fs for fills before re-anchoring",
            wait_seconds,
        )
        _time.sleep(wait_seconds)

    # Collect entry orders that were ACTUALLY submitted (not dry-run,
    # not failed, not "kept") — those are the candidates whose stops
    # might need re-anchoring.
    entries = [
        o for o in submitted_orders
        if o.role == "entry" and o.status == "submitted" and o.alpaca_order_id
    ]
    if not entries:
        return

    for entry in entries:
        sym = entry.symbol
        signal_price = signal_prices.get(sym, 0.0)
        if signal_price <= 0:
            continue
        per_sym_pct = (
            stop_pcts.get(sym, default_stop_pct)
            if stop_pcts is not None
            else default_stop_pct
        )

        # Fetch the parent order to read filled_avg_price + child IDs.
        try:
            order = client.get_order_by_id(entry.alpaca_order_id)
        except Exception as e:
            logger.warning(
                "stop repair: could not fetch order %s for %s: %s",
                entry.alpaca_order_id, sym, e,
            )
            continue

        # Skip if not filled yet — the OTO's signal-anchored stop still
        # protects (just at the wrong level). Next day's run will catch
        # any stragglers via the regular signal-price stop.
        status = getattr(getattr(order, "status", None), "value", "")
        if status != "filled":
            continue

        filled_avg_price = getattr(order, "filled_avg_price", None)
        if filled_avg_price is None:
            continue
        try:
            fill_px = float(filled_avg_price)
        except (TypeError, ValueError):
            continue
        if fill_px <= 0:
            continue

        drift = abs(fill_px - signal_price) / signal_price
        if drift < drift_threshold:
            # Fill close enough to signal that the auto-stop is fine.
            continue

        # Drift > threshold: replace the stop. New stop level anchored
        # to actual fill price using the per-symbol stop_pct.
        new_stop_price = round(fill_px * (1.0 - per_sym_pct), 2)
        if new_stop_price <= 0:
            continue

        # Find the OTO child stop order via the parent's `legs` attribute.
        legs = getattr(order, "legs", None) or []
        child_stop_id = None
        for leg in legs:
            leg_type = getattr(getattr(leg, "order_type", None), "value", "")
            if leg_type == "stop":
                child_stop_id = getattr(leg, "id", None)
                break

        try:
            if child_stop_id:
                client.cancel_order_by_id(child_stop_id)
            new_stop_req = StopOrderRequest(
                symbol=sym,
                qty=int(entry.qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=new_stop_price,
            )
            client.submit_order(new_stop_req)
            logger.info(
                "stop repair: %s filled at %.2f (signal %.2f, drift %.1f%%) — "
                "stop moved to %.2f (was anchored at %.2f)",
                sym, fill_px, signal_price, drift * 100,
                new_stop_price, round(signal_price * (1.0 - per_sym_pct), 2),
            )
        except Exception as e:
            logger.warning(
                "stop repair: failed to replace stop for %s: %s. "
                "Original OTO stop remains active (anchored to signal price).",
                sym, e,
            )
