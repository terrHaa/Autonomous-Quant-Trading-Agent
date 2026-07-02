"""Regression tests for top_movers (the CSCO-in-both-lists artifact)."""
from __future__ import annotations

from quant.util.equity_stats import top_movers


def _runs(prices_first: dict, prices_last: dict) -> list[dict]:
    return [
        {"date": "2026-06-01", "signal_prices": prices_first},
        {"date": "2026-06-30", "signal_prices": prices_last},
    ]


def test_gainers_and_losers_are_disjoint_with_few_movers() -> None:
    """< 2n movers used to make the slices overlap — CSCO showed up in
    BOTH lists on the 2026-07-02 monthly. Lists must be disjoint."""
    first = {f"S{i}": 100.0 for i in range(6)}
    last = {f"S{i}": 100.0 + (i - 2) for i in range(6)}  # moves: -2..+3
    gainers, losers = top_movers(_runs(first, last), n=10)
    gsyms = {g["symbol"] for g in gainers}
    lsyms = {x["symbol"] for x in losers}
    assert not (gsyms & lsyms), f"overlap: {gsyms & lsyms}"


def test_gainers_are_up_and_losers_are_down() -> None:
    first = {"UP": 100.0, "DOWN": 100.0, "FLAT": 100.0}
    last = {"UP": 110.0, "DOWN": 90.0, "FLAT": 100.0}
    gainers, losers = top_movers(_runs(first, last), n=5)
    assert [g["symbol"] for g in gainers] == ["UP"]
    assert [x["symbol"] for x in losers] == ["DOWN"]
    # FLAT appears in neither.
    assert all(g["move_pct"] > 0 for g in gainers)
    assert all(x["move_pct"] < 0 for x in losers)


def test_normal_case_orders_most_extreme_first() -> None:
    first = {f"S{i}": 100.0 for i in range(30)}
    last = {f"S{i}": 100.0 + (i - 15) for i in range(30)}
    gainers, losers = top_movers(_runs(first, last), n=5)
    assert len(gainers) == 5 and len(losers) == 5
    assert gainers[0]["move_pct"] == max(g["move_pct"] for g in gainers)
    assert losers[0]["move_pct"] == min(x["move_pct"] for x in losers)
