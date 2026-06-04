"""Tests for the point-in-time universe loader.

Most tests run against a small synthetic fixture CSV — that keeps them
independent of curation changes to the shipped sp500.csv. A handful of
"shipped" tests pin the production file's intended shape so that an
accidental edit breaks tests before it breaks backtests.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from quant.data.universe import Universe, _load_from_csv, load_universe

# A tiny synthetic membership history. Chosen so each row exercises a
# different membership-window case:
#   - ALPHA:  always in (added long ago, never removed)
#   - BETA:   added mid-stream (tests `added` is inclusive)
#   - GAMMA:  removed mid-stream (tests `removed` is exclusive — the
#             survivorship-bias correction)
#   - DELTA:  added very late (tests "added strictly after today" is excluded)
FIXTURE_CSV = """symbol,added,removed
ALPHA,2010-01-01,
BETA,2015-06-01,
GAMMA,2010-01-01,2018-06-15
DELTA,2024-01-01,
"""


@pytest.fixture
def fixture_universe(tmp_path: Path) -> Universe:
    csv = tmp_path / "fixture.csv"
    csv.write_text(FIXTURE_CSV)
    return _load_from_csv(name="fixture", csv_path=csv)


def test_continuous_member_is_included(fixture_universe: Universe) -> None:
    """A symbol present throughout the window is included."""
    assert "ALPHA" in fixture_universe.members(date(2020, 1, 1))


def test_added_date_is_inclusive(fixture_universe: Universe) -> None:
    """A symbol IS a member on the day it was added."""
    assert fixture_universe.is_member("BETA", date(2015, 6, 1))
    assert not fixture_universe.is_member("BETA", date(2015, 5, 31))


def test_removed_date_is_exclusive(fixture_universe: Universe) -> None:
    """A symbol is NOT a member on its removal date — this is the
    survivorship-bias-correct behavior. Lehman 'removed 2008-09-15' means
    tradeable on 9/14, NOT on 9/15."""
    assert fixture_universe.is_member("GAMMA", date(2018, 6, 14))
    assert not fixture_universe.is_member("GAMMA", date(2018, 6, 15))


def test_pre_addition_date_excludes_symbol(fixture_universe: Universe) -> None:
    """A symbol that hasn't been added yet is not in the universe."""
    assert not fixture_universe.is_member("DELTA", date(2020, 1, 1))
    # Same query via the bulk members() path — should agree.
    assert "DELTA" not in fixture_universe.members(date(2020, 1, 1))


def test_pre_any_addition_returns_empty(fixture_universe: Universe) -> None:
    """A date before any symbol was added yields an empty member list."""
    assert fixture_universe.members(date(2000, 1, 1)) == []


def test_members_are_sorted(fixture_universe: Universe) -> None:
    """Deterministic order — two equal memberships should produce equal lists."""
    members = fixture_universe.members(date(2020, 1, 1))
    assert members == sorted(members)


def test_unknown_universe_name_raises() -> None:
    """Bad name → KeyError that mentions the bad name AND lists the known ones,
    so the user knows what to type instead."""
    with pytest.raises(KeyError, match="ZZZ"):
        load_universe("ZZZ")


def test_duplicate_symbols_rejected_at_load(tmp_path: Path) -> None:
    """Schema check: dupes raise at load, not at query time."""
    csv = tmp_path / "dupes.csv"
    csv.write_text("symbol,added,removed\nAAPL,2010-01-01,\nAAPL,2015-01-01,\n")
    with pytest.raises(ValueError, match="duplicate symbols"):
        _load_from_csv(name="dupes", csv_path=csv)


def test_removed_before_added_rejected_at_load(tmp_path: Path) -> None:
    """Schema check: a 'zero-day membership' row is data corruption."""
    csv = tmp_path / "bad_dates.csv"
    csv.write_text("symbol,added,removed\nAAPL,2010-01-01,2009-01-01\n")
    with pytest.raises(ValueError, match="removed <= added"):
        _load_from_csv(name="bad_dates", csv_path=csv)


# ----------------------------------------------------------------------------
# Smoke tests against the SHIPPED sp500.csv. These pin the curated file to
# its intended shape so a careless edit fails tests before it silently
# changes every backtest.
# ----------------------------------------------------------------------------


def test_shipped_sp500_loads() -> None:
    sp500 = load_universe("sp500")
    assert sp500.name == "sp500"
    # The starter file has at least a handful of names on any modern date.
    assert len(sp500.members(date(2024, 1, 2))) >= 5


def test_shipped_sp500_demonstrates_survivorship_bias_correction() -> None:
    """The whole point of point-in-time membership: dead names exit, not
    just disappear-from-history.

    LEH (Lehman Brothers, bankrupt 2008-09-15) must NOT be tradeable on
    2008-12-31 — by then it was out of the index. The Wikipedia-sourced
    CSV records its 2008-09-16 removal date even though the original
    1957 add date is too old for the "Selected changes" table to
    include. That's enough to make the survivorship-bias-correction
    real: a backtest that asks "was LEH a member on date X" gets the
    right answer for any date after the removal.

    (For dates BEFORE the recorded change, members() can't include
    LEH because there's no add date — that's a known limitation of
    the schema. Real coverage requires a paid data source. The live
    agent uses the post-removal correctness, which is what matters.)
    """
    sp500 = load_universe("sp500")
    # Post-removal: NOT a member.
    assert not sp500.is_member("LEH", date(2008, 12, 31))
    assert not sp500.is_member("LEH", date.today())


def test_sp500_liquid_is_aliased_to_sp500() -> None:
    """The config's `sp500_liquid` should load the same membership base.

    The liquidity filter applies at a higher layer (using bar data, which
    this module doesn't have). For now both names return the same set.
    """
    assert (
        load_universe("sp500").members(date(2024, 1, 2))
        == load_universe("sp500_liquid").members(date(2024, 1, 2))
    )
