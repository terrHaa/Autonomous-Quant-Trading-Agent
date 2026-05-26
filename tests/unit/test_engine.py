"""Tests for the event-driven backtest engine.

Each test exercises one engine behavior on synthetic, hand-crafted bars
so we can predict the exact result and assert on it. The two foundational
tests are:

  - ``test_noop_strategy_keeps_equity_flat`` — basic plumbing works.
  - ``test_strategy_cannot_see_future`` — the no-leak rule holds end-to-end.

Without those two passing, nothing else in this file matters.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant.backtest import BacktestResult, OrderIntent, Snapshot, run_backtest
from quant.config import DEFAULT_CONFIG_PATH, Config
from quant.data.alpaca_client import BAR_COLUMNS

# ----------------------------------------------------------------------------
# Helpers — build small bars frames and configs for deterministic tests.
# ----------------------------------------------------------------------------


def _bars(
    symbols: list[str],
    closes: dict[str, list[float]],
    *,
    start: str = "2024-01-02",
    spread_pct: float = 0.01,
) -> pd.DataFrame:
    """Build a MultiIndex bars frame with deterministic OHLCV.

    open == low == close (1-pct spread); volume constant. Lets tests reason
    about expected fills without OHLC complications.
    """
    bdays = pd.bdate_range(start, periods=len(next(iter(closes.values()))), tz="UTC")
    rows = []
    idx = []
    for sym in symbols:
        for i, ts in enumerate(bdays):
            c = closes[sym][i]
            rows.append({
                "open": c,
                "high": c * (1 + spread_pct),
                "low": c * (1 - spread_pct),
                "close": c,
                "volume": 1_000_000,
            })
            idx.append((sym, ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=list(BAR_COLUMNS),
    )


def _zero_cost_config() -> Config:
    """A config with zero trading costs. Strips noise from PnL assertions."""
    import yaml

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["backtest"]["costs"] = {
        "commission_bps": 0.0,
        "spread_bps": 0.0,
        "slippage_bps": 0.0,
    }
    return Config.model_validate(raw)


# ----------------------------------------------------------------------------
# Strategy stubs
# ----------------------------------------------------------------------------


class _NoOp:
    """Holds nothing, ever. Used to verify plumbing without trade noise."""
    name = "noop"

    def on_bar(self, snapshot: Snapshot) -> dict:
        return {}


class _BuyAndHold:
    """100% long the named symbol, every bar."""

    def __init__(self, symbol: str, name: str = "buy_and_hold"):
        self.name = name
        self._symbol = symbol

    def on_bar(self, snapshot: Snapshot) -> dict:
        return {self._symbol: 1.0}


class _PeekerRecorder:
    """No-op strategy that records the max timestamp it sees each bar.

    The recorded list lets a test verify the no-leak rule end-to-end:
    on bar i, the strategy's snapshot must not contain any data beyond i.
    """

    name = "peeker"

    def __init__(self):
        self.max_dates_seen: list[date] = []

    def on_bar(self, snapshot: Snapshot) -> dict:
        if snapshot.bars.empty:
            self.max_dates_seen.append(snapshot.as_of)
        else:
            ts = snapshot.bars.index.get_level_values("timestamp")
            self.max_dates_seen.append(ts.date.max())
        return {}


# ----------------------------------------------------------------------------
# Foundational tests — the engine and the no-leak rule
# ----------------------------------------------------------------------------


def test_noop_strategy_keeps_equity_exactly_flat() -> None:
    """A strategy that trades nothing should produce flat equity == start.

    If this fails, the engine is doing *something* on idle — maybe marking
    against a missing symbol, accidentally generating phantom orders, etc.
    """
    bars = _bars(["AAPL"], {"AAPL": [100.0, 101.0, 102.0, 103.0, 104.0]})
    config = _zero_cost_config()

    result = run_backtest(config=config, strategy=_NoOp(), bars=bars)

    # Equity equals starting equity on every bar; no orders generated.
    assert (result.equity_curve == config.backtest.starting_equity).all()
    assert len(result.orders) == 0
    assert len(result.fills) == 0


def test_strategy_cannot_see_future_via_snapshot() -> None:
    """End-to-end no-leak: on bar i, the strategy's snapshot maxes out at i.

    This is the platform's most important invariant. The Snapshot class
    enforces it structurally; this test confirms the engine actually
    constructs snapshots correctly (i.e., it isn't bypassing the factory).
    """
    bars = _bars(["AAPL"], {"AAPL": [100.0, 101.0, 102.0, 103.0, 104.0]})
    recorder = _PeekerRecorder()

    run_backtest(config=_zero_cost_config(), strategy=recorder, bars=bars)

    # pd.unique on a numpy date array; sort for deterministic comparison.
    ts = bars.index.get_level_values("timestamp")
    expected_dates = sorted(pd.unique(ts.date).tolist())
    assert recorder.max_dates_seen == expected_dates


# ----------------------------------------------------------------------------
# Market-on-open fills
# ----------------------------------------------------------------------------


def test_market_buy_fills_at_next_open() -> None:
    """A buy intent on bar 0 fills at the OPEN of bar 1, not bar 0's close.

    Note: BuyAndHold targets weight=1.0 every bar, so the engine will
    *rebalance* on subsequent bars (target_qty drifts as price moves).
    That's correct behavior — we just check the FIRST fill is the buy
    at bar 1's open.
    """
    bars = _bars(["AAPL"], {"AAPL": [100.0, 110.0, 120.0]})
    bnh = _BuyAndHold("AAPL")
    config = _zero_cost_config()

    result = run_backtest(config=config, strategy=bnh, bars=bars)

    buys = result.fills[result.fills["side"] == "buy"]
    assert len(buys) >= 1, "buy-and-hold should produce at least one buy fill"

    first_buy = buys.iloc[0]
    assert first_buy["fill_price"] == 110.0, "buy must fill at next bar's open"
    expected_date = bars.index.get_level_values("timestamp")[1].date()
    assert first_buy["date"] == expected_date


def test_implicit_flat_exits_unmentioned_holdings() -> None:
    """A symbol held in t-1 but not mentioned in t's intents should be sold."""
    bars = _bars(["AAPL"], {"AAPL": [100.0, 110.0, 120.0, 130.0]})
    config = _zero_cost_config()

    class _HoldThenDrop:
        """Bar 0: buy AAPL. Bar 1: drop it from intents → engine should sell."""
        name = "hold_then_drop"

        def __init__(self):
            self._bar = 0

        def on_bar(self, snapshot):
            self._bar += 1
            if self._bar == 1:
                return {"AAPL": 1.0}     # buy
            return {}                     # implicit flat from bar 2 onward

    result = run_backtest(config=config, strategy=_HoldThenDrop(), bars=bars)
    # Expect: 1 buy fill (bar 1's open) + 1 sell fill (bar 2's open).
    sides = result.fills["side"].tolist()
    assert sides == ["buy", "sell"]


# ----------------------------------------------------------------------------
# Limit orders
# ----------------------------------------------------------------------------


def test_buy_limit_fills_when_low_crosses_limit() -> None:
    """A buy-limit triggers iff the next bar's low <= limit."""
    # Day 0 close=100. Day 1: low=99, so a limit at 99.5 fills at 99.5.
    bars = _bars(["AAPL"], {"AAPL": [100.0, 100.0]}, spread_pct=0.01)
    # spread_pct=0.01 → on day 1 the low = 100 * 0.99 = 99.0 (well below 99.5).

    class _LimitBuyer:
        name = "limit_buyer"

        def on_bar(self, snapshot):
            return {"AAPL": OrderIntent(
                target_weight=1.0,
                order_type="limit",
                limit_price=99.5,
            )}

    result = run_backtest(config=_zero_cost_config(), strategy=_LimitBuyer(), bars=bars)
    assert len(result.fills) >= 1
    # Filled at the limit price, not the open.
    assert result.fills.iloc[0]["fill_price"] == 99.5


def test_limit_DAY_drops_unfilled_orders_after_one_bar() -> None:
    """A DAY limit that doesn't fill on the next bar shouldn't carry over."""
    # All prices flat at 100. A buy limit at 90 will never trigger.
    bars = _bars(["AAPL"], {"AAPL": [100.0, 100.0, 100.0]}, spread_pct=0.0)

    class _UnreachableLimit:
        name = "unreachable"

        def on_bar(self, snapshot):
            return {"AAPL": OrderIntent(
                target_weight=1.0,
                order_type="limit",
                limit_price=90.0,
                time_in_force="DAY",
            )}

    result = run_backtest(config=_zero_cost_config(), strategy=_UnreachableLimit(), bars=bars)
    # New order each bar, but none fills, none carry over → zero fills.
    assert len(result.fills) == 0
    # Each bar regenerated the same intent → orders > 0.
    assert len(result.orders) >= 1


# ----------------------------------------------------------------------------
# Stop orders
# ----------------------------------------------------------------------------


def test_sell_stop_exit_triggers_when_low_crosses_stop() -> None:
    """A stop-loss EXIT (target_weight=0 via stop order) fires when low <= stop.

    Realistic stop-loss pattern: enter long at market, then on the next bar
    place a sell-stop to exit if price drops to a level. The engine only
    generates an order when the target qty differs from current — so 'stop'
    intents are paired with weight=0 (or another non-equilibrium weight)
    to trigger order generation.

    Bars used:
      day 1: close = 100  (signal day for the initial buy)
      day 2: open = 100   (buy fills here; also signal day for stop intent)
      day 3: low = 94.05  (stop at 95 triggers on day 3)
    """
    bars = _bars(
        ["AAPL"],
        {"AAPL": [100.0, 100.0, 95.0]},
        spread_pct=0.01,
    )

    class _BuyThenStopExit:
        name = "buy_then_stop_exit"

        def __init__(self):
            self._bar = 0

        def on_bar(self, snapshot):
            self._bar += 1
            if self._bar == 1:
                return {"AAPL": 1.0}     # market buy at next open
            # From bar 2: a sell-stop to exit on a drop to 95.
            return {"AAPL": OrderIntent(
                target_weight=0.0,        # weight=0 → engine generates sell order
                order_type="stop",
                stop_price=95.0,
            )}

    result = run_backtest(config=_zero_cost_config(), strategy=_BuyThenStopExit(), bars=bars)

    # The stop fill should appear at the stop price = 95.
    stop_fills = result.fills[result.fills["fill_price"] == 95.0]
    assert len(stop_fills) >= 1, "stop should have triggered on day 3 (low=94.05 <= 95)"


# ----------------------------------------------------------------------------
# BacktestResult shape
# ----------------------------------------------------------------------------


def test_result_shape_is_correct() -> None:
    """The BacktestResult must contain the promised structures."""
    bars = _bars(["AAPL"], {"AAPL": [100.0, 110.0, 120.0]})
    result = run_backtest(config=_zero_cost_config(), strategy=_BuyAndHold("AAPL"), bars=bars)

    assert isinstance(result, BacktestResult)
    assert result.strategy_name == "buy_and_hold"
    # Equity curve, positions, weights all indexed by the trading dates.
    ts = bars.index.get_level_values("timestamp")
    expected_dates = sorted(pd.unique(ts.date).tolist())
    assert list(result.equity_curve.index) == expected_dates
    assert list(result.positions.index) == expected_dates
    assert list(result.weights.index) == expected_dates
    # Costs frame has the right columns regardless of trades.
    assert set(result.costs.columns) >= {"spread_cost", "slippage_cost", "commission", "total"}
    # Metadata has expected keys.
    for key in ("n_bars", "n_orders", "n_fills", "starting_equity", "ending_equity"):
        assert key in result.metadata


# ----------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------


def test_engine_is_deterministic() -> None:
    """Same config + same data + same strategy must give identical results."""
    bars = _bars(["AAPL"], {"AAPL": [100.0, 110.0, 105.0, 115.0, 120.0]})
    config = _zero_cost_config()

    r1 = run_backtest(config=config, strategy=_BuyAndHold("AAPL"), bars=bars.copy())
    r2 = run_backtest(config=config, strategy=_BuyAndHold("AAPL"), bars=bars.copy())

    # Equity curves bit-identical.
    pd.testing.assert_series_equal(r1.equity_curve, r2.equity_curve)
    pd.testing.assert_frame_equal(r1.positions, r2.positions)
    pd.testing.assert_frame_equal(r1.fills, r2.fills)
