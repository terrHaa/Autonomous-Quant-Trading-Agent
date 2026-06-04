"""engine.py — the event-driven backtest engine.

Walks forward bar-by-bar, asks the strategy what to do, queues orders, fills
them against the next bar, marks to market, and records everything into a
``BacktestResult``. See ``docs/specs/backtest-engine.md`` for the full design
and rationale; this file is the implementation.

The whole loop is in one function (``run_backtest``) so the control flow is
readable top-to-bottom. Helpers handle the per-bar mechanics: order
generation, fill attempts, mark-to-market.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from quant.backtest.portfolio import Portfolio
from quant.backtest.types import (
    Fill,
    Order,
    OrderIntent,
    Side,
    Snapshot,
    Strategy,
    TimeInForce,
)
from quant.config import Config

# ---------------------------------------------------------------------------
# Output: the structured result of one backtest run.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestResult:
    """Everything one backtest produces. Frozen — part of the audit trail.

    Use ``equity_curve`` for plotting and Sharpe-family metrics;
    ``positions`` / ``weights`` for exposure analysis;
    ``orders`` / ``fills`` / ``costs`` for trade-level attribution;
    ``metadata`` for run-level provenance (config snapshot, durations, etc).
    """

    config: Config
    strategy_name: str
    equity_curve: pd.Series        # index = date, value = end-of-bar equity
    positions: pd.DataFrame        # index = date, cols = symbol, values = shares
    weights: pd.DataFrame          # index = date, cols = symbol, values = weight after mark
    orders: pd.DataFrame           # one row per queued order
    fills: pd.DataFrame            # one row per executed fill
    costs: pd.DataFrame            # index = date, cols = spread/slippage/commission/total
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run_backtest(
    *,
    config: Config,
    strategy: Strategy,
    bars: pd.DataFrame,
    stop_loss_pct: float | None = 0.05,
) -> BacktestResult:
    """Run a full backtest.

    Parameters
    ----------
    config
        Loaded ``Config`` object (typically from ``quant.config.load_config``).
        The engine reads from ``config.backtest`` (starting_equity, costs).
    strategy
        Any object satisfying the ``Strategy`` Protocol — has ``name`` and
        ``on_bar(snapshot)``.
    bars
        MultiIndex(symbol, timestamp) OHLCV DataFrame, as produced by the
        cache. Must satisfy the standard contract — run ``check_daily_bars``
        on it before calling this if you don't trust the source.
    stop_loss_pct
        T-audit fix C2: model a hard stop-loss at this distance below the
        position's weighted-average entry price. Default 0.05 matches the
        live agent's STOP_LOSS_PCT. Pass ``None`` to disable (legacy
        behaviour — useful for testing strategies in isolation from the
        risk overlay). Gap-through is modelled honestly: if today's open
        is already below the stop, the fill is at the open (the gap is
        not magically protected).

    Returns
    -------
    BacktestResult
        Frozen container with the equity curve, positions, fills, costs,
        and metadata for downstream analysis.
    """
    t_start = time.perf_counter()

    # ---- Precompute wide-format OHLC for fast per-bar lookup ----------
    # `bars` is MultiIndex(symbol, timestamp). For fills/marking we need
    # quick scalar access by (date, symbol). Wide format makes this O(1).
    opens, highs, lows, closes = _wide_ohlc(bars)
    trading_dates = closes.index.tolist()   # sorted ascending by unstack

    # ---- Initialize state ---------------------------------------------
    portfolio = Portfolio(starting_equity=config.backtest.starting_equity)
    queued_orders: list[Order] = []
    # T-audit fix C2: weighted-average entry price per held symbol, used
    # to compute the per-position stop level. Updated on every buy fill
    # (add-on positions get a proper weighted average) and cleared when
    # the symbol goes flat.
    entry_price: dict[str, float] = {}
    n_stop_outs = 0   # surfaced in metadata so the operator sees if stops fire often

    # Recording accumulators. Each list grows by one row per matching event.
    equity_by_date: dict[date, float] = {}
    weights_by_date: dict[date, dict[str, float]] = {}
    positions_by_date: dict[date, dict[str, int]] = {}
    order_records: list[dict[str, Any]] = []
    fill_records: list[dict[str, Any]] = []

    # ---- Main loop: walk the trading calendar -------------------------
    for bar_date in trading_dates:
        # ---- 1. FILL: attempt to execute yesterday's queued orders ----
        # against TODAY's bar. Orders that don't fill (limit not crossed,
        # stop not triggered) are dropped (DAY) or re-queued (GTC).
        still_queued: list[Order] = []
        for order in queued_orders:
            fill = _try_fill(
                order=order,
                bar_date=bar_date,
                opens=opens,
                highs=highs,
                lows=lows,
                costs_cfg=config.backtest.costs,
            )
            if fill is not None:
                # T-audit fix C2: maintain weighted-average entry price
                # for the stop check below. Buy fills add to the basis;
                # sell fills that flat the symbol clear it.
                prev_qty = portfolio.positions.get(fill.symbol, 0)
                portfolio.apply_fill(fill)
                new_qty = portfolio.positions.get(fill.symbol, 0)
                if fill.side == "buy" and new_qty > 0:
                    if prev_qty <= 0:
                        # Fresh entry (or short cover into long).
                        entry_price[fill.symbol] = fill.fill_price
                    else:
                        # Add-on — weight by share count.
                        prev_basis = entry_price.get(fill.symbol, fill.fill_price)
                        entry_price[fill.symbol] = (
                            (prev_basis * prev_qty + fill.fill_price * fill.qty)
                            / new_qty
                        )
                elif new_qty == 0:
                    entry_price.pop(fill.symbol, None)
                fill_records.append(_fill_to_dict(fill))
            else:
                # No fill today. Decide based on time-in-force.
                if order.time_in_force == "GTC":
                    still_queued.append(order)
                # DAY: drop the order silently.
        queued_orders = still_queued

        # ---- 1b. STOP-LOSS CHECK (T-audit fix C2) ----------------------
        # For each long position with a known entry price, the stop fires
        # if today's LOW touches stop_price = entry × (1 - stop_loss_pct).
        # Fill at the stop_price unless today's OPEN is already below
        # (gap-through), in which case fill at the open — the gap was
        # not magically protected. Models the real-world worst case.
        if stop_loss_pct is not None and stop_loss_pct > 0:
            stopped_out: list[tuple[str, int, float]] = []
            for sym, qty in portfolio.positions.items():
                if qty <= 0:
                    continue   # shorts use a different stop direction; skip
                basis = entry_price.get(sym)
                if basis is None:
                    continue
                try:
                    lo = lows.at[bar_date, sym]
                    op = opens.at[bar_date, sym]
                except KeyError:
                    continue
                if pd.isna(lo) or pd.isna(op):
                    continue
                stop_level = basis * (1.0 - stop_loss_pct)
                if float(lo) <= stop_level:
                    # Fill at stop_level OR open if the open gapped through.
                    fill_px = min(float(op), stop_level)
                    stopped_out.append((sym, qty, fill_px))
            for sym, qty, fill_px in stopped_out:
                stop_fill = _synthesize_stop_fill(
                    bar_date=bar_date,
                    symbol=sym,
                    qty=qty,
                    fill_price=fill_px,
                    costs_cfg=config.backtest.costs,
                )
                portfolio.apply_fill(stop_fill)
                fill_records.append(_fill_to_dict(stop_fill))
                entry_price.pop(sym, None)
                n_stop_outs += 1

        # ---- 2. MARK: revalue held positions at today's close ---------
        # We only mark held names — other entries in `closes` row are
        # noise (the universe is wider than the portfolio).
        todays_closes = closes.loc[bar_date].dropna().to_dict()
        portfolio.mark_to_market(todays_closes)

        # ---- 3. RECORD end-of-bar state -------------------------------
        equity_by_date[bar_date] = portfolio.equity()
        weights_by_date[bar_date] = portfolio.weights()
        positions_by_date[bar_date] = dict(portfolio.positions)

        # ---- 4. SIGNAL: hand strategy a snapshot through today's close
        # Snapshot pre-slices to ``timestamp <= bar_date``. The strategy
        # CAN see today's bar (decisions are made post-close), but cannot
        # see tomorrow's.
        snapshot = Snapshot.from_full_bars(bars, as_of=bar_date)
        try:
            intents = strategy.on_bar(snapshot)
        except Exception as e:
            # Strategy bugs shouldn't crash the engine silently — re-raise
            # with the bar date so the user knows which day broke it.
            raise RuntimeError(
                f"strategy {strategy.name!r} raised on bar {bar_date}: {e}"
            ) from e

        # ---- 5. ORDER: turn intents into orders for tomorrow's bar ----
        new_orders = _build_orders(
            intents=intents,
            portfolio=portfolio,
            signal_closes=todays_closes,
            submitted_date=bar_date,
        )
        for order in new_orders:
            order_records.append(_order_to_dict(order))
        # Carry GTC orders forward AND add the new ones.
        queued_orders = queued_orders + new_orders

    # ---- Build the BacktestResult -------------------------------------
    return _assemble_result(
        config=config,
        strategy_name=strategy.name,
        equity_by_date=equity_by_date,
        positions_by_date=positions_by_date,
        weights_by_date=weights_by_date,
        order_records=order_records,
        fill_records=fill_records,
        trading_dates=trading_dates,
        run_time_s=time.perf_counter() - t_start,
        n_stop_outs=n_stop_outs,
        stop_loss_pct=stop_loss_pct,
    )


# ---------------------------------------------------------------------------
# Helpers (private). Each does one thing so the main loop reads as the spec.
# ---------------------------------------------------------------------------


def _wide_ohlc(
    bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert the long-form bars frame into wide (date × symbol) per field.

    Why: the loop needs O(1) lookups for (date, symbol) → price. Wide
    format gives us that via ``.at[date, symbol]``, vs O(N) per slice on
    the original MultiIndex.

    The index gets normalized to plain ``date`` objects (not timestamps),
    matching how the engine iterates and how callers will read the result.
    """
    def _wide(field: str) -> pd.DataFrame:
        # unstack symbol level → index=timestamp, columns=symbol
        w = bars[field].unstack(level="symbol")
        # Drop the time-of-day; we work in date-resolution
        w = w.copy()
        w.index = pd.Index([ts.date() for ts in w.index], name="date")
        # Stable column order so iteration is deterministic.
        return w.sort_index().sort_index(axis=1)

    return _wide("open"), _wide("high"), _wide("low"), _wide("close")


def _build_orders(
    *,
    intents: dict[str, float | OrderIntent],
    portfolio: Portfolio,
    signal_closes: dict[str, float],
    submitted_date: date,
) -> list[Order]:
    """Turn a strategy's ``on_bar`` output into concrete ``Order``s.

    Logic:
      1. For each (symbol, intent) the strategy returned: compute target qty
         from target weight and today's close, diff against current qty,
         generate an order for the delta.
      2. For each symbol the portfolio holds that the strategy DID NOT
         mention: liquidate at market-on-open (implicit flat).
      3. Symbols with no signal-day close get skipped (can't size). The
         engine doesn't trade what it can't price.
    """
    orders: list[Order] = []
    equity = portfolio.equity()
    mentioned: set[str] = set()

    for symbol, intent in intents.items():
        mentioned.add(symbol)

        # Normalize: a bare float means "market order at this target weight".
        if isinstance(intent, OrderIntent):
            target_weight = intent.target_weight
            order_type = intent.order_type
            limit_price = intent.limit_price
            stop_price = intent.stop_price
            tif: TimeInForce = intent.time_in_force
        else:
            target_weight = float(intent)
            order_type = "market"
            limit_price = None
            stop_price = None
            tif = "DAY"

        signal_price = signal_closes.get(symbol)
        if signal_price is None or signal_price <= 0:
            # Can't size what we can't price; document by skipping silently
            # (an explicit warning would be noisy in normal operation).
            continue

        # Target $ → target shares (truncate toward zero so we never over-buy).
        target_dollars = target_weight * equity
        target_qty = int(target_dollars / signal_price)
        current_qty = portfolio.positions.get(symbol, 0)
        delta = target_qty - current_qty

        if delta == 0:
            continue

        side: Side = "buy" if delta > 0 else "sell"
        orders.append(
            Order(
                submitted_date=submitted_date,
                symbol=symbol,
                side=side,
                qty=abs(delta),
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=tif,
            )
        )

    # Implicit flats: held symbols the strategy didn't mention → exit at market.
    for symbol, qty in list(portfolio.positions.items()):
        if symbol in mentioned:
            continue
        if qty > 0:
            side = "sell"
        else:
            side = "buy"   # covering a short
        orders.append(
            Order(
                submitted_date=submitted_date,
                symbol=symbol,
                side=side,
                qty=abs(qty),
                order_type="market",
                time_in_force="DAY",
            )
        )

    return orders


def _try_fill(
    *,
    order: Order,
    bar_date: date,
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    costs_cfg,
) -> Fill | None:
    """Attempt to fill one order against the bar of ``bar_date``.

    Returns the Fill if executed, None if not (limit not crossed, stop not
    triggered, or the symbol has no bar on this date).
    """
    # Pull today's OHLC for the order's symbol. Missing → can't fill.
    try:
        o = opens.at[bar_date, order.symbol]
        h = highs.at[bar_date, order.symbol]
        lo = lows.at[bar_date, order.symbol]
    except KeyError:
        return None
    if pd.isna(o) or pd.isna(h) or pd.isna(lo):
        return None

    # Branch by order type. Each branch sets `fill_price`, `spread_cost`,
    # and `slippage_cost`; commission is computed uniformly at the bottom.
    if order.order_type == "market":
        half_spread_bp = costs_cfg.spread_bps / 2
        slip_bp = costs_cfg.slippage_bps
        total_bp = half_spread_bp + slip_bp
        if order.side == "buy":
            fill_price = float(o) * (1 + total_bp / 10_000)
        else:
            fill_price = float(o) * (1 - total_bp / 10_000)
        # Decompose costs vs the modeled open.
        spread_cost = abs(order.qty) * float(o) * half_spread_bp / 10_000
        slippage_cost = abs(order.qty) * float(o) * slip_bp / 10_000

    elif order.order_type == "limit":
        # Buy limit fills if today's low reached down to the limit;
        # sell limit fills if today's high reached up to the limit.
        if order.side == "buy" and float(lo) <= order.limit_price:
            fill_price = float(order.limit_price)
        elif order.side == "sell" and float(h) >= order.limit_price:
            fill_price = float(order.limit_price)
        else:
            return None
        # No spread or slippage on limits — by definition you got your price.
        spread_cost = 0.0
        slippage_cost = 0.0

    elif order.order_type == "stop":
        # Buy stop triggers if high >= stop (price moved up through level);
        # sell stop triggers if low <= stop (price moved down through level).
        # Fills at the stop price — see gap-through caveat in the spec.
        if order.side == "buy" and float(h) >= order.stop_price:
            fill_price = float(order.stop_price)
        elif order.side == "sell" and float(lo) <= order.stop_price:
            fill_price = float(order.stop_price)
        else:
            return None
        spread_cost = 0.0
        slippage_cost = 0.0

    else:
        # Defensive — Literal type should make this unreachable.
        raise ValueError(f"unknown order_type {order.order_type!r}")

    notional = order.qty * fill_price
    commission = notional * costs_cfg.commission_bps / 10_000

    return Fill(
        date=bar_date,
        symbol=order.symbol,
        side=order.side,
        qty=order.qty,
        fill_price=fill_price,
        notional=notional,
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        commission=commission,
    )


# ---------------------------------------------------------------------------
# Result assembly. Pure-conversion code; segregated to keep the main loop
# focused on the simulation.
# ---------------------------------------------------------------------------


def _order_to_dict(o: Order) -> dict[str, Any]:
    return {
        "submitted_date": o.submitted_date,
        "symbol": o.symbol,
        "side": o.side,
        "qty": o.qty,
        "order_type": o.order_type,
        "limit_price": o.limit_price,
        "stop_price": o.stop_price,
        "time_in_force": o.time_in_force,
    }


def _fill_to_dict(f: Fill) -> dict[str, Any]:
    return {
        "date": f.date,
        "symbol": f.symbol,
        "side": f.side,
        "qty": f.qty,
        "fill_price": f.fill_price,
        "notional": f.notional,
        "spread_cost": f.spread_cost,
        "slippage_cost": f.slippage_cost,
        "commission": f.commission,
    }


def _synthesize_stop_fill(
    *,
    bar_date: date,
    symbol: str,
    qty: int,
    fill_price: float,
    costs_cfg,
) -> Fill:
    """T-audit fix C2: build a Fill that matches what the live executor's
    GTC stop-loss order would have produced.

    Costs: we charge commission_bps (paid on every fill) but NO spread or
    slippage — a real stop triggers and converts to a market-on-touch
    order; the spread is real but it's already implicit in the fact that
    fill_price is the stop level (operator gets the bid, not the mid).
    Modelling extra slippage here would double-count.
    """
    notional = qty * fill_price
    commission = notional * costs_cfg.commission_bps / 10_000
    return Fill(
        date=bar_date,
        symbol=symbol,
        side="sell",
        qty=qty,
        fill_price=fill_price,
        notional=notional,
        spread_cost=0.0,
        slippage_cost=0.0,
        commission=commission,
    )


def _assemble_result(
    *,
    config: Config,
    strategy_name: str,
    equity_by_date: dict[date, float],
    positions_by_date: dict[date, dict[str, int]],
    weights_by_date: dict[date, dict[str, float]],
    order_records: list[dict[str, Any]],
    fill_records: list[dict[str, Any]],
    trading_dates: list[date],
    run_time_s: float,
    n_stop_outs: int = 0,
    stop_loss_pct: float | None = None,
) -> BacktestResult:
    """Convert per-day accumulators into the final ``BacktestResult``."""

    equity_curve = pd.Series(
        equity_by_date, name="equity"
    ).reindex(trading_dates)

    # Wide DataFrames for positions and weights. fillna(0) so missing names
    # show explicit zero on days they weren't held — easier to reason about.
    positions_df = (
        pd.DataFrame.from_dict(positions_by_date, orient="index")
        .reindex(trading_dates)
        .fillna(0)
        .astype(int)
        .sort_index(axis=1)
    )
    weights_df = (
        pd.DataFrame.from_dict(weights_by_date, orient="index")
        .reindex(trading_dates)
        .fillna(0.0)
        .sort_index(axis=1)
    )

    orders_df = pd.DataFrame(order_records) if order_records else pd.DataFrame(
        columns=[
            "submitted_date", "symbol", "side", "qty", "order_type",
            "limit_price", "stop_price", "time_in_force",
        ]
    )

    fills_df = pd.DataFrame(fill_records) if fill_records else pd.DataFrame(
        columns=[
            "date", "symbol", "side", "qty", "fill_price", "notional",
            "spread_cost", "slippage_cost", "commission",
        ]
    )

    # Daily cost breakdown: aggregate fills by date. Reindex so every
    # trading day has a row (zero on no-trade days) — makes plots clean.
    if not fills_df.empty:
        costs_df = (
            fills_df.groupby("date")[["spread_cost", "slippage_cost", "commission"]]
            .sum()
            .reindex(trading_dates)
            .fillna(0.0)
        )
    else:
        costs_df = pd.DataFrame(
            0.0,
            index=trading_dates,
            columns=["spread_cost", "slippage_cost", "commission"],
        )
    costs_df["total"] = costs_df.sum(axis=1)

    metadata = {
        "n_bars": len(trading_dates),
        "n_orders": len(order_records),
        "n_fills": len(fill_records),
        "run_time_s": round(run_time_s, 4),
        "start_date": trading_dates[0] if trading_dates else None,
        "end_date": trading_dates[-1] if trading_dates else None,
        "starting_equity": config.backtest.starting_equity,
        "ending_equity": (
            float(equity_curve.iloc[-1])
            if len(equity_curve) and pd.notna(equity_curve.iloc[-1])
            else config.backtest.starting_equity
        ),
        # T-audit fix C2: surface how the stop overlay behaved. Operators
        # can compare across candidates — a strategy with >20% of fills
        # being forced stop-outs is fundamentally less reliable than one
        # whose stops rarely fire even after the same Sharpe.
        "n_stop_outs": n_stop_outs,
        "stop_loss_pct": stop_loss_pct,
    }

    return BacktestResult(
        config=config,
        strategy_name=strategy_name,
        equity_curve=equity_curve,
        positions=positions_df,
        weights=weights_df,
        orders=orders_df,
        fills=fills_df,
        costs=costs_df,
        metadata=metadata,
    )
