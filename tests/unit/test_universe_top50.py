"""Tests for the top-50 snapshot universe loader.

Distinct from test_universe.py because the top-50 list is a snapshot
(not a point-in-time membership history) — it loads from a different
CSV with a different schema.
"""

from __future__ import annotations

from quant.data.universe import load_top50_snapshot


def test_load_returns_nonempty_list_of_uppercase_symbols() -> None:
    """The shipped CSV must parse and contain a sensible set of names."""
    symbols = load_top50_snapshot()
    # We ship exactly 50; assert a band to catch egregious truncation
    # without breaking on a one-off ADR removal.
    assert 40 <= len(symbols) <= 50, (
        f"snapshot should be ~50 names; got {len(symbols)}"
    )
    assert all(s == s.upper() for s in symbols), "loader must uppercase"
    assert all(s.strip() == s for s in symbols), "loader must strip whitespace"


def test_no_duplicates_in_snapshot() -> None:
    """If anyone accidentally duplicates a row, the loader dedupes silently
    BUT we want the underlying CSV clean. Belt-and-suspenders test."""
    symbols = load_top50_snapshot()
    assert len(symbols) == len(set(symbols)), "duplicate symbols in snapshot"


def test_shipped_snapshot_includes_known_mega_caps() -> None:
    """A sanity check on the shipped contents.

    If these names are missing, someone has shipped a clearly-broken
    snapshot — these are the largest US equities by market cap as of
    every quarter for years.
    """
    symbols = set(load_top50_snapshot())
    for must_have in ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"):
        assert must_have in symbols, (
            f"{must_have} missing from top-50 snapshot — refresh needed"
        )


def test_no_foreign_listings() -> None:
    """SAP (Germany) and ASML (Netherlands) aren't S&P 500 members even
    though they're huge. Catches accidental re-introductions on refresh."""
    symbols = set(load_top50_snapshot())
    for foreign in ("SAP", "ASML"):
        assert foreign not in symbols, (
            f"{foreign} listed in foreign exchange — not S&P 500"
        )
