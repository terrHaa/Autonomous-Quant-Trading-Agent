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

from dataclasses import dataclass, field
from datetime import datetime, timezone
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
        now = datetime.now(timezone.utc)

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
