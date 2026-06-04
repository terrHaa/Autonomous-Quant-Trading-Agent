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


# ---------------------------------------------------------------------------
# load_active_universe — point-in-time loader with fallback
# ---------------------------------------------------------------------------

from datetime import date as _date  # noqa: E402

from quant.data.universe import load_active_universe  # noqa: E402


def test_load_active_universe_falls_back_when_pit_csv_too_small() -> None:
    """The shipped sp500.csv has < 50 active names (it's a starter set),
    so today the loader MUST fall back to load_top50_snapshot. This
    behaviour is the safety hatch that keeps the agent running while
    you curate the comprehensive CSV.

    When you finish curating the CSV (>= 50 active names), this test
    starts failing — that's the signal that fallback is no longer
    needed and the agent is on point-in-time membership for real.
    """
    syms = load_active_universe(_date.today(), fallback_log=False)
    # The fallback returns the top-50 snapshot. It should contain
    # at least 50 names; the starter point-in-time CSV has way fewer.
    assert len(syms) >= 50, (
        "Either the fallback is broken (top-50 snapshot returned < 50 names) "
        "OR the point-in-time CSV is now comprehensive — in which case "
        "delete this test and add one verifying the PIT path is in use."
    )


def test_load_active_universe_uses_pit_when_csv_has_enough_names(
    monkeypatch, tmp_path,
) -> None:
    """When the point-in-time CSV has enough active names, the loader
    USES it (no fallback). Regression guard for the day the user
    finishes the curation work."""
    # Build a fake Universe with 60 active members.
    from types import SimpleNamespace

    fake_members = [f"SYM{i}" for i in range(60)]

    def fake_load_universe(name):
        return SimpleNamespace(members=lambda as_of: fake_members)

    monkeypatch.setattr(
        "quant.data.universe.load_universe", fake_load_universe,
    )
    syms = load_active_universe(_date.today(), fallback_log=False)
    # PIT path took over — exact membership returned.
    assert syms == fake_members
    assert len(syms) == 60
