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


def test_load_active_universe_uses_pit_when_csv_is_populated() -> None:
    """After quant-sp500-refresh runs, the CSV holds ~500 active names
    and load_active_universe MUST use the point-in-time path (not the
    survivorship-biased top-50 fallback).

    Regression guard. If this test fails, either:
      - reference/universe/sp500.csv was truncated (revert it), or
      - the loader's fallback threshold was raised above ~500
    Both are bugs.
    """
    syms = load_active_universe(_date.today(), fallback_log=False)
    # PIT returns the full S&P 500 (~500); fallback returns only 50.
    # > 100 is well above the fallback ceiling — proves we're on PIT.
    assert len(syms) > 100, (
        f"Expected ~500 active S&P 500 names from PIT loader; got {len(syms)}. "
        "Either reference/universe/sp500.csv was truncated, or the loader "
        "fell back to the survivorship-biased top-50 snapshot. Re-run "
        "quant-sp500-refresh to repopulate."
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
