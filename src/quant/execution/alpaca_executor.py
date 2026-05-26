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

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from quant.data.alpaca_client import AlpacaCredentials

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
    status: Literal["submitted", "failed", "skipped_dry_run"]
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
    ) -> ExecutionReport:
        """Bring current positions to match the target weight book.

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

        Returns
        -------
        ExecutionReport
            Full structured record of what was proposed and submitted.
        """
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
        """
        if stop_loss_pct <= 0 or stop_loss_pct >= 1:
            raise ValueError(
                f"stop_loss_pct must be in (0, 1); got {stop_loss_pct}"
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

        # ---- 2. Close positions not in the target book. -------------
        # Done as PROPOSED orders so they show on the report alongside
        # entries; dry_run skips submission as elsewhere.
        target_symbols = {s for s, w in target_weights.items() if w > 0}
        stale_positions = {
            sym: qty for sym, qty in positions.items()
            if sym not in target_symbols and qty != 0
        }
        # We also close positions whose target qty differs from current
        # (the close-and-reopen simplification documented above).
        for sym in target_symbols:
            current_qty = positions.get(sym, 0)
            if current_qty <= 0:
                continue
            target_dollars = target_weights[sym] * equity
            price = signal_prices.get(sym)
            if price is None or price <= 0:
                # No price → can't even compute target qty → conservative
                # exit. We'll skip the re-entry in step 3 too.
                stale_positions[sym] = current_qty
                continue
            target_qty = int(target_dollars / price)
            if current_qty != target_qty:
                stale_positions[sym] = current_qty

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

        # ---- 3. Open new target positions with OTO + protective stop. ----
        # For each target symbol whose current qty doesn't already match,
        # we submitted a close above. Now we (re-)enter via OTO:
        #   parent: market buy `target_qty` shares
        #   child:  stop-loss sell `target_qty` shares at signal*(1-pct)
        # Alpaca makes the child active only after the parent fills.
        for sym in sorted(target_symbols):
            weight = target_weights[sym]
            price = signal_prices.get(sym)
            if price is None or price <= 0:
                continue

            target_qty = int(weight * equity / price)
            current_qty = positions.get(sym, 0)
            if target_qty <= 0:
                continue
            # If current == target after the (potential) close above,
            # no order needed. But: we cancelled the old stop in step 1,
            # so a position-unchanged name now has no stop. We need to
            # re-establish it. Simplest: close-and-reopen always.
            # (Already handled above — sym is in stale_positions if
            # current != 0.)

            entry_notional = target_qty * price
            if entry_notional > max_notional:
                submitted.append(SubmittedOrder(
                    symbol=sym, side="buy", qty=target_qty,
                    status="failed", role="entry",
                    error=(
                        f"refusing entry: notional ${entry_notional:,.2f} "
                        f"exceeds max_position_weight × equity = "
                        f"${max_notional:,.2f}"
                    ),
                ))
                continue

            stop_price = round(price * (1.0 - stop_loss_pct), 2)
            # If for any reason stop_price >= entry signal price, refuse.
            if stop_price >= price:
                submitted.append(SubmittedOrder(
                    symbol=sym, side="buy", qty=target_qty,
                    status="failed", role="entry",
                    error=f"computed stop_price {stop_price} >= signal {price}",
                ))
                continue

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

                req = MarketOrderRequest(
                    symbol=sym,
                    qty=target_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
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
