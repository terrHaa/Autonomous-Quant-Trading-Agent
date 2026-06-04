"""Tests for tools/curate_sp500_membership.py — verifies the Wikipedia
ingest produces a valid sp500.csv that the universe loader can consume.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _import_curate_module():
    """Import the tool module from disk (it's in tools/, not the package)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    path = repo_root / "tools" / "curate_sp500_membership.py"
    spec = importlib.util.spec_from_file_location("curate_sp500_membership", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Module-level singleton so importing once per test session is enough.
curate = _import_curate_module()


def test_parse_date_handles_common_wikipedia_formats() -> None:
    """Wikipedia mixes ISO dates, "Month DD, YYYY", and the occasional
    short form. Parser should hit them all."""
    assert curate._parse_date("2024-01-15") == "2024-01-15"
    assert curate._parse_date("January 15, 2024") == "2024-01-15"
    assert curate._parse_date("Jan 15, 2024") == "2024-01-15"
    assert curate._parse_date("01/15/2024") == "2024-01-15"
    assert curate._parse_date("") is None
    assert curate._parse_date(None) is None
    assert curate._parse_date(float("nan")) is None


def test_pick_column_is_case_and_space_insensitive() -> None:
    """Wikipedia column headers vary by case and spacing — the picker
    normalises both."""
    df = pd.DataFrame(columns=["Date Added", "Symbol"])
    norm = curate._normalise_columns(df)
    assert curate._pick_column(norm, "date_added", "date_first_added") == "date_added"
    assert curate._pick_column(norm, "ticker", "symbol") == "symbol"


def test_pick_column_raises_when_no_match() -> None:
    df = pd.DataFrame(columns=["Foo"])
    norm = curate._normalise_columns(df)
    with pytest.raises(ValueError, match="Could not find"):
        curate._pick_column(norm, "bar", "baz")


def test_build_membership_csv_end_to_end(tmp_path: Path) -> None:
    """Full integration: two synthetic input CSVs → output sp500.csv that
    the universe loader can parse."""
    current = tmp_path / "wikipedia_current.csv"
    current.write_text(
        "Symbol,Date added\n"
        "AAPL,1982-11-30\n"
        "MSFT,1994-06-01\n"
        "NEWCO,2024-03-15\n"
    )
    changes = tmp_path / "wikipedia_changes.csv"
    changes.write_text(
        "Date,Added,Removed\n"
        "2024-03-15,NEWCO,OLDCO\n"
        "2023-06-01,GOOGL,WORN\n"   # removed-only event for WORN
    )
    out = tmp_path / "sp500.csv"
    curate.build_membership_csv(current, changes, out)

    # Output must be a valid CSV the universe loader can read.
    assert out.exists()
    df = pd.read_csv(out, comment="#")
    assert set(df.columns) == {"symbol", "added", "removed"}
    symbols = set(df["symbol"])
    assert "AAPL" in symbols                  # active member
    assert "OLDCO" in symbols                 # historical, removed
    assert "WORN" in symbols                  # historical, removed by changes table
    # OLDCO should have removed=2024-03-15 (no added because not in current).
    oldco_row = df[df["symbol"] == "OLDCO"].iloc[0]
    assert oldco_row["removed"] == "2024-03-15"


def test_build_membership_csv_output_works_with_universe_loader(
    tmp_path: Path, monkeypatch,
) -> None:
    """End-to-end: build a CSV with this tool, then have load_universe
    successfully parse it. Closes the loop between the curation tool
    and the production loader."""
    from quant.data.universe import load_universe

    current = tmp_path / "current.csv"
    current.write_text(
        "Symbol,Date added\n"
        "AAPL,1982-11-30\n"
        "MSFT,1994-06-01\n"
    )
    changes = tmp_path / "changes.csv"
    changes.write_text("Date,Added,Removed\n")   # no changes
    out = tmp_path / "sp500.csv"
    curate.build_membership_csv(current, changes, out)

    # Redirect the universe loader's reference dir to our tmp_path.
    monkeypatch.setattr(
        "quant.data.universe._REFERENCE_DIR", tmp_path,
    )
    u = load_universe("sp500")
    from datetime import date
    members = u.members(date(2024, 1, 1))
    assert set(members) >= {"AAPL", "MSFT"}
