"""Tests for the backtest engine's data classes.

Snapshot tests focus on the no-leak guarantee — the platform's most
load-bearing invariant. OrderIntent/Order/Fill tests focus on construction
validation: most of the engine assumes these objects are well-formed, so
catching malformed ones at construction (not deep inside the engine) is
where the work belongs.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pandas as pd
import pytest

from quant.backtest.types import (
    Fill,
    Order,
    OrderIntent,
    Snapshot,
)
from quant.data.alpaca_client import BAR_COLUMNS


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _bars(*, symbols: list[str], dates: list[str]) -> pd.DataFrame:
    """Construct a minimal MultiIndex bars frame for tests.

    Prices are deterministic so test assertions can be explicit about
    "close on day 3 of AAPL".
    """
    rows = []
    idx = []
    for sym in symbols:
        for i, d in enumerate(dates):
            ts = pd.Timestamp(d, tz="UTC")
            rows.append({
                "open": 100.0 + i,
                "high": 102.0 + i,
                "low": 99.0 + i,
                "close": 101.0 + i,
                "volume": 1_000_000,
            })
            idx.append((sym, ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=list(BAR_COLUMNS),
    )


# ----------------------------------------------------------------------------
# Snapshot — the no-leak primitive
# ----------------------------------------------------------------------------


def test_snapshot_factory_slices_through_as_of() -> None:
    """from_full_bars should drop every row with timestamp > as_of."""
    full = _bars(symbols=["AAPL"], dates=["2024-01-02", "2024-01-03", "2024-01-04"])
    snap = Snapshot.from_full_bars(full, as_of=date(2024, 1, 3))

    seen_dates = snap.bars.index.get_level_values("timestamp").date.tolist()
    assert max(seen_dates) == date(2024, 1, 3)
    assert date(2024, 1, 4) not in seen_dates


def test_snapshot_constructor_rejects_future_data() -> None:
    """Even hand-constructed snapshots can't smuggle in future data.

    This is the belt-and-suspenders on the no-leak rule: even if someone
    bypasses the factory, the __post_init__ check fires.
    """
    full = _bars(symbols=["AAPL"], dates=["2024-01-02", "2024-01-03", "2024-01-04"])
    # Constructed with the full frame but a too-early as_of — must raise.
    with pytest.raises(ValueError, match="leak future data"):
        Snapshot(as_of=date(2024, 1, 3), bars=full)


def test_snapshot_close_returns_most_recent_close() -> None:
    """close(symbol) returns the most recent close on or before as_of."""
    full = _bars(symbols=["AAPL"], dates=["2024-01-02", "2024-01-03", "2024-01-04"])
    snap = Snapshot.from_full_bars(full, as_of=date(2024, 1, 3))

    # Day 2 close = 101.0 + 1 (i=1) = 102.0 in our helper's scheme.
    assert snap.close("AAPL") == 102.0


def test_snapshot_close_unknown_symbol_raises() -> None:
    """Asking for a name not in the snapshot must raise, not silently return 0."""
    full = _bars(symbols=["AAPL"], dates=["2024-01-02"])
    snap = Snapshot.from_full_bars(full, as_of=date(2024, 1, 2))

    with pytest.raises(KeyError, match="MSFT"):
        snap.close("MSFT")


def test_snapshot_symbols_lists_all_present() -> None:
    """symbols() should return every symbol with data on or before as_of, sorted."""
    full = _bars(symbols=["MSFT", "AAPL"], dates=["2024-01-02"])
    snap = Snapshot.from_full_bars(full, as_of=date(2024, 1, 2))

    # Sorted, not in insertion order.
    assert snap.symbols() == ["AAPL", "MSFT"]


def test_empty_snapshot_works() -> None:
    """An empty frame must round-trip — common case at the very start of a run."""
    empty = pd.DataFrame(
        columns=list(BAR_COLUMNS),
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )
    snap = Snapshot.from_full_bars(empty, as_of=date(2024, 1, 2))
    assert snap.symbols() == []


def test_snapshot_is_frozen() -> None:
    full = _bars(symbols=["AAPL"], dates=["2024-01-02"])
    snap = Snapshot.from_full_bars(full, as_of=date(2024, 1, 2))
    with pytest.raises(FrozenInstanceError):
        snap.as_of = date(2024, 1, 3)  # type: ignore[misc]


# ----------------------------------------------------------------------------
# OrderIntent
# ----------------------------------------------------------------------------


def test_order_intent_market_needs_no_prices() -> None:
    """A bare market intent is the common case; nothing else required."""
    intent = OrderIntent(target_weight=0.05)
    assert intent.order_type == "market"
    assert intent.limit_price is None
    assert intent.stop_price is None


def test_order_intent_limit_requires_limit_price() -> None:
    """Limit orders without a limit price are nonsense — fail at construction."""
    with pytest.raises(ValueError, match="limit_price"):
        OrderIntent(target_weight=0.05, order_type="limit")


def test_order_intent_stop_requires_stop_price() -> None:
    """Same for stops."""
    with pytest.raises(ValueError, match="stop_price"):
        OrderIntent(target_weight=0.05, order_type="stop")


def test_order_intent_is_frozen() -> None:
    intent = OrderIntent(target_weight=0.05)
    with pytest.raises(FrozenInstanceError):
        intent.target_weight = 0.10  # type: ignore[misc]


# ----------------------------------------------------------------------------
# Order
# ----------------------------------------------------------------------------


def test_order_qty_must_be_positive() -> None:
    """Direction lives in `side`; negative qty would be ambiguous."""
    with pytest.raises(ValueError, match="positive"):
        Order(
            submitted_date=date(2024, 1, 2),
            symbol="AAPL",
            side="buy",
            qty=-5,
            order_type="market",
        )


def test_order_is_frozen() -> None:
    order = Order(
        submitted_date=date(2024, 1, 2),
        symbol="AAPL",
        side="buy",
        qty=100,
        order_type="market",
    )
    with pytest.raises(FrozenInstanceError):
        order.qty = 200  # type: ignore[misc]


# ----------------------------------------------------------------------------
# Fill
# ----------------------------------------------------------------------------


def test_fill_is_frozen() -> None:
    """A fill is part of the audit trail; mutating it would corrupt history."""
    fill = Fill(
        date=date(2024, 1, 3),
        symbol="AAPL",
        side="buy",
        qty=100,
        fill_price=185.00,
        notional=18_500.00,
        spread_cost=1.85,
        slippage_cost=5.55,
        commission=0.00,
    )
    with pytest.raises(FrozenInstanceError):
        fill.qty = 200  # type: ignore[misc]
