"""Tests for the Portfolio class — the only mutable state in the engine.

Pattern: each test builds a fresh Portfolio, applies one or two fills, and
asserts on cash/positions/equity/weights. Small focused cases > one giant
narrative test.
"""

from __future__ import annotations

from datetime import date

import pytest

from quant.backtest.portfolio import Portfolio
from quant.backtest.types import Fill


def _fill(
    side: str,
    qty: int,
    price: float,
    *,
    symbol: str = "AAPL",
    commission: float = 0.0,
    spread: float = 0.0,
    slippage: float = 0.0,
    when: date = date(2024, 1, 3),
) -> Fill:
    """Shorthand fill constructor for tests — most fields default to zero."""
    return Fill(
        date=when,
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        qty=qty,
        fill_price=price,
        notional=qty * price,
        spread_cost=spread,
        slippage_cost=slippage,
        commission=commission,
    )


# ----------------------------------------------------------------------------
# Construction & initial state
# ----------------------------------------------------------------------------


def test_starting_state_is_only_cash() -> None:
    p = Portfolio(starting_equity=1_000_000)
    assert p.cash == 1_000_000
    assert p.positions == {}
    assert p.equity() == 1_000_000
    assert p.weights() == {}


def test_zero_or_negative_starting_equity_raises() -> None:
    """A backtest with zero starting cash is meaningless."""
    with pytest.raises(ValueError):
        Portfolio(starting_equity=0)
    with pytest.raises(ValueError):
        Portfolio(starting_equity=-1)


# ----------------------------------------------------------------------------
# apply_fill — long side
# ----------------------------------------------------------------------------


def test_buy_decreases_cash_and_creates_position() -> None:
    p = Portfolio(starting_equity=1_000_000)
    p.apply_fill(_fill("buy", 100, 185.00))

    assert p.cash == 1_000_000 - 18_500
    assert p.positions == {"AAPL": 100}


def test_sell_full_position_increases_cash_and_drops_entry() -> None:
    """Closing a position completely should remove it from the positions dict
    so callers iterating over positions don't see flat names."""
    p = Portfolio(starting_equity=1_000_000)
    p.apply_fill(_fill("buy", 100, 185.00))
    p.apply_fill(_fill("sell", 100, 190.00))

    assert p.positions == {}
    assert p.cash == 1_000_000 - 18_500 + 19_000  # net +$500


def test_partial_close_keeps_remainder() -> None:
    p = Portfolio(starting_equity=1_000_000)
    p.apply_fill(_fill("buy", 100, 185.00))
    p.apply_fill(_fill("sell", 30, 190.00))

    assert p.positions == {"AAPL": 70}


def test_commission_reduces_cash() -> None:
    """Commission is a separate flat cost regardless of side."""
    p = Portfolio(starting_equity=1_000_000)
    p.apply_fill(_fill("buy", 100, 185.00, commission=1.00))

    assert p.cash == 1_000_000 - 18_500 - 1.00


# ----------------------------------------------------------------------------
# apply_fill — short side
# ----------------------------------------------------------------------------


def test_short_sell_creates_negative_position_and_increases_cash() -> None:
    """Opening a short via a sell from flat: cash goes UP, position goes negative."""
    p = Portfolio(starting_equity=1_000_000)
    p.apply_fill(_fill("sell", 50, 185.00))  # short open

    assert p.positions == {"AAPL": -50}
    assert p.cash == 1_000_000 + 9_250


def test_cover_short_decreases_cash_and_closes_position() -> None:
    """Closing a short: buy from negative → flat. Cash goes DOWN."""
    p = Portfolio(starting_equity=1_000_000)
    p.apply_fill(_fill("sell", 50, 185.00))
    p.apply_fill(_fill("buy", 50, 180.00))   # cover at lower price

    assert p.positions == {}
    # Net cash: started with 1M, +9_250 on short open, -9_000 on cover = 1_000_250.
    assert p.cash == 1_000_000 + 9_250 - 9_000


# ----------------------------------------------------------------------------
# mark_to_market, equity, weights
# ----------------------------------------------------------------------------


def test_equity_reflects_mark_to_market() -> None:
    """After marking, equity should equal cash + position * mark."""
    p = Portfolio(starting_equity=1_000_000)
    p.apply_fill(_fill("buy", 100, 185.00))   # cash = 981_500, position = 100 AAPL

    p.mark_to_market({"AAPL": 190.00})        # AAPL up 5
    # equity = 981_500 + 100*190 = 981_500 + 19_000 = 1_000_500
    assert p.equity() == 1_000_500


def test_mark_to_market_ignores_unheld_symbols() -> None:
    """Marks for names we don't hold don't leak into _marks (keeps memory tidy)."""
    p = Portfolio(starting_equity=1_000_000)
    p.apply_fill(_fill("buy", 100, 185.00))
    p.mark_to_market({"AAPL": 190.00, "MSFT": 400.00, "GOOGL": 150.00})

    # Equity should only include AAPL; MSFT/GOOGL marks are noise.
    assert p.equity() == 981_500 + 19_000


def test_weights_sum_to_equity_fraction() -> None:
    """Each weight = (qty * mark) / equity."""
    p = Portfolio(starting_equity=1_000_000)
    p.apply_fill(_fill("buy", 100, 185.00, symbol="AAPL"))
    p.apply_fill(_fill("buy", 25, 400.00, symbol="MSFT"))
    p.mark_to_market({"AAPL": 200.00, "MSFT": 410.00})

    w = p.weights()
    e = p.equity()

    # Verify each weight matches the manual computation.
    assert w["AAPL"] == pytest.approx((100 * 200.00) / e)
    assert w["MSFT"] == pytest.approx((25 * 410.00) / e)


def test_weights_empty_when_equity_nonpositive() -> None:
    """If equity goes to zero or below (catastrophic loss), weights are undefined."""
    p = Portfolio(starting_equity=1_000_000)
    # Borrow $1M, lose it all on a position then mark to zero.
    p.apply_fill(_fill("buy", 1_000_000, 1.00))   # cash = 0, position = 1M sh @ $1
    p.mark_to_market({"AAPL": 0.0})               # position now worthless

    assert p.equity() == 0
    assert p.weights() == {}


def test_unmarked_position_contributes_zero_value() -> None:
    """A held name we've never marked is conservatively valued at zero.

    In practice the engine marks every held name on every bar, so this is
    a defensive default — never knowingly book PnL we can't price.
    """
    p = Portfolio(starting_equity=1_000_000)
    p.apply_fill(_fill("buy", 100, 185.00))
    # Skip mark_to_market entirely.
    # equity = cash (981_500) + position value (0 because no mark) = 981_500
    assert p.equity() == 981_500
