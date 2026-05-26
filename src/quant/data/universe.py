"""universe.py — point-in-time membership for named investment universes.

Why this exists
---------------
A backtest on "today's S&P 500" silently throws away every stock that was
booted from the index over the test window — Lehman in 2008, GE in 2018,
Enron in 2001, and so on. Your strategy looks great because it never owned
a name that went to zero. This is **survivorship bias**, and it's the most
common source of inflated retail-grade backtest performance.

The fix is point-in-time membership: for each trading day, ask "which
symbols were actually in the index on THIS day?" and trade only those.

How it works
------------
Membership history per universe lives in a CSV with three columns:

    symbol, added, removed

- ``added``   = first day IN the index (INCLUSIVE).
- ``removed`` = first day NOT in the index (EXCLUSIVE), blank if still in.

Example: Lehman row ``LEH,1957-03-04,2008-09-15`` means Lehman was a member
from 1957-03-04 through 2008-09-14, and NOT a member on 2008-09-15 or after.
This convention makes the membership test clean:

    added <= as_of < removed   (or removed is null)

Limitations of the current schema
---------------------------------
- One contiguous membership interval per symbol. Symbols that were removed
  then re-added (e.g., AIG: removed Sept 2008, re-added Sept 2012) need a
  schema extension before they can be modeled honestly. For now they'd
  need to be split into two rows with different ticker suffixes, which is
  a hack — better to extend the schema when the time comes.
- The shipped ``sp500.csv`` is a hand-curated starter, not exhaustive.
  See ``reference/README.md`` for how to grow it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

# Project root — same trick as config.py. This file lives at
# src/quant/data/universe.py, so the root is FOUR parents up:
#   universe.py -> data -> quant -> src -> <project root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REFERENCE_DIR = _PROJECT_ROOT / "reference" / "universe"


# Mapping from universe name (as used in configs/*.yaml) to its CSV file.
#
# Multiple names can point to the same membership file. For example,
# "sp500_liquid" is the same underlying membership as "sp500" — the
# "_liquid" suffix is a filter applied at a higher layer (BarsLoader,
# not here) using actual bar data (avg dollar volume, etc.). Splitting
# "name resolution" from "filter application" keeps this module focused.
_UNIVERSE_FILES: dict[str, str] = {
    "sp500": "sp500.csv",
    "sp500_liquid": "sp500.csv",
}


@dataclass(frozen=True)
class Universe:
    """A named investment universe with point-in-time membership.

    Frozen — once loaded, the membership table is immutable. If you need
    a different universe, load a different one; don't mutate this object.
    """

    name: str
    # The full membership table — keeping it on the object means callers
    # who want extra detail (when was AAPL added? show me every removal in
    # 2008) can poke at it without a re-read.
    _table: pd.DataFrame

    def members(self, as_of: date) -> list[str]:
        """Return symbols that were in the universe on ``as_of``.

        Result is sorted alphabetically for deterministic output (so equal
        membership on two dates produces equal lists). Empty if nothing
        qualifies — e.g. asking for a date before any symbol was added.
        """
        as_of_ts = pd.Timestamp(as_of)
        df = self._table

        # The contract: added <= as_of AND (removed is NaT OR removed > as_of).
        # `isna()` on a datetime column catches the "still in the index" rows.
        in_after_add = df["added"] <= as_of_ts
        not_yet_removed = df["removed"].isna() | (df["removed"] > as_of_ts)
        mask = in_after_add & not_yet_removed

        return sorted(df.loc[mask, "symbol"].tolist())

    def is_member(self, symbol: str, as_of: date) -> bool:
        """Was ``symbol`` in the universe on ``as_of``?"""
        return symbol.upper() in self.members(as_of)


def load_universe(name: str) -> Universe:
    """Load a named universe from the reference data directory.

    Parameters
    ----------
    name
        Universe name as used in configs/*.yaml (e.g., ``"sp500"``).

    Raises
    ------
    KeyError
        If ``name`` isn't in the known-universes mapping.
    FileNotFoundError
        If the membership CSV is missing — usually means you're running
        outside the source tree where ``reference/`` doesn't exist.
    """
    if name not in _UNIVERSE_FILES:
        raise KeyError(
            f"Unknown universe {name!r}. Known: {sorted(_UNIVERSE_FILES)}. "
            f"Add a mapping in quant.data.universe._UNIVERSE_FILES if you've "
            f"added a new membership CSV."
        )
    csv_path = _REFERENCE_DIR / _UNIVERSE_FILES[name]
    return _load_from_csv(name=name, csv_path=csv_path)


def _load_from_csv(name: str, csv_path: Path) -> Universe:
    """Read a membership CSV and validate the contract.

    Internal helper — tests use it directly with fixture CSVs so they
    don't depend on (or break when we edit) the shipped sp500.csv.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Universe CSV not found at {csv_path}. "
            f"Are you running from the project source tree? See reference/README.md."
        )

    df = pd.read_csv(
        csv_path,
        # The CSV header includes `#` provenance comments — tell pandas to
        # skip them rather than parsing them as data.
        comment="#",
        # parse_dates yields NaT (Not a Time) for empty `removed` cells,
        # which is exactly what we want — "still in the index" = NaT.
        parse_dates=["added", "removed"],
    )

    # Validate once at load time. The point-in-time query is called many
    # times per backtest; we don't want to re-validate on every call.
    _validate_membership_table(df, csv_path)

    return Universe(name=name, _table=df)


def load_top100_snapshot() -> list[str]:
    """Load the top-100 S&P 500 snapshot used by the autonomous agent.

    Unlike :func:`load_universe`, this is NOT a point-in-time membership
    history — it's a *static snapshot* of "the names we're allowed to
    trade this quarter." Refresh the CSV at ``reference/universe/
    sp500_top100.csv`` from a current market-cap ranking each quarter.

    Returns the symbols as a list, uppercased and deduped (insertion
    order preserved).
    """
    csv_path = _REFERENCE_DIR / "sp500_top100.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Top-100 snapshot CSV not found at {csv_path}. "
            "Run from the source tree, or refresh the snapshot per "
            "reference/README.md."
        )
    df = pd.read_csv(csv_path, comment="#")
    if "symbol" not in df.columns:
        raise ValueError(f"{csv_path.name} must have a 'symbol' column")
    # dict.fromkeys preserves order while dropping duplicates.
    return list(dict.fromkeys(s.upper().strip() for s in df["symbol"]))


def _validate_membership_table(df: pd.DataFrame, csv_path: Path) -> None:
    """Fail loudly if the membership CSV is structurally wrong."""
    required = {"symbol", "added", "removed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {sorted(missing)}"
        )

    # Symbols must be unique — our schema doesn't support multiple
    # membership intervals per symbol (see module docstring).
    dupes = df["symbol"][df["symbol"].duplicated()].tolist()
    if dupes:
        raise ValueError(
            f"{csv_path.name} has duplicate symbols: {dupes}. "
            f"Multiple membership intervals per symbol aren't supported yet."
        )

    # If removed is present it must be strictly after added; otherwise the
    # symbol was 'a member for zero days', which is always data corruption.
    has_removal = df["removed"].notna()
    bad = df[has_removal & (df["removed"] <= df["added"])]
    if len(bad):
        raise ValueError(
            f"{csv_path.name} has rows where removed <= added: "
            f"{bad['symbol'].tolist()}. `removed` must be strictly after `added`."
        )
