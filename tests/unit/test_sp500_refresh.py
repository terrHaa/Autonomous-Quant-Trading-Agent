"""Tests for the quarterly S&P 500 auto-refresh.

We do NOT hit the live Wikipedia URL in tests. The fetch path is
exercised indirectly via the build_membership + validate + diff
helpers, with constructed inputs that simulate what Wikipedia
returns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant.agent.sp500_refresh import (
    _build_membership,
    _diff_against_existing,
    _normalise_changes_table,
    _normalise_current_table,
    _parse_date,
    _render_email,
    _validate_membership,
    _write_csv,
    refresh_sp500_universe,
)


def _wiki_style_current() -> pd.DataFrame:
    """Synthetic version of Wikipedia's current-constituents table."""
    return pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "NVDA", "FAKE", "NEWCO"],
        "Date added": [
            "1982-11-30", "1994-06-01", "2001-11-30",
            "January 15, 2023", "Mar 1, 2024",
        ],
    })


def _wiki_style_changes() -> pd.DataFrame:
    """Synthetic version of Wikipedia's selected-changes table."""
    return pd.DataFrame({
        "Date": ["2024-03-01", "2023-09-15"],
        "Added Ticker": ["NEWCO", ""],
        "Removed Ticker": ["OLDCO", "ANCIENT"],
    })


# ---------------------------------------------------------------------------
# Parser robustness
# ---------------------------------------------------------------------------


def test_parse_date_handles_iso() -> None:
    assert _parse_date("2024-03-15") == "2024-03-15"


def test_parse_date_handles_long_form() -> None:
    assert _parse_date("January 15, 2024") == "2024-01-15"


def test_parse_date_blank_returns_none() -> None:
    assert _parse_date("") is None
    assert _parse_date(None) is None


def test_normalise_current_table_lowercases_and_renames() -> None:
    out = _normalise_current_table(_wiki_style_current())
    assert set(out.columns) == {"symbol", "date_added"}
    assert "AAPL" in out["symbol"].values


def test_normalise_changes_table_collapses_multilevel_headers() -> None:
    df = pd.DataFrame({
        "Date": ["2024-03-15"],
        "Added Ticker": ["NEWCO"],
        "Removed Ticker": ["OLDCO"],
    })
    out = _normalise_changes_table(df)
    assert set(out.columns) == {"date", "added", "removed"}


# ---------------------------------------------------------------------------
# Membership build
# ---------------------------------------------------------------------------


def test_build_membership_marks_active_when_no_removal() -> None:
    cur = _normalise_current_table(_wiki_style_current())
    chg = _normalise_changes_table(_wiki_style_changes())
    members = _build_membership(cur, chg)
    # AAPL is active (no removal in changes table).
    assert members["AAPL"] == ("1982-11-30", None)


def test_build_membership_records_historical_exits() -> None:
    cur = _normalise_current_table(_wiki_style_current())
    chg = _normalise_changes_table(_wiki_style_changes())
    members = _build_membership(cur, chg)
    # OLDCO + ANCIENT only appear in the changes table — record them as
    # historical (removed dates, no added date).
    assert "OLDCO" in members
    assert members["OLDCO"][1] == "2024-03-01"
    assert "ANCIENT" in members
    assert members["ANCIENT"][1] == "2023-09-15"


# ---------------------------------------------------------------------------
# Validation safety gates
# ---------------------------------------------------------------------------


def test_validate_flags_too_few_active_members() -> None:
    members = {"AAPL": ("1982-11-30", None)}
    errors = _validate_membership(members, min_active=400)
    assert any("only 1 active members" in e for e in errors)


def test_validate_accepts_full_universe() -> None:
    members = {f"SYM{i}": ("2020-01-01", None) for i in range(500)}
    errors = _validate_membership(members, min_active=400)
    assert errors == []


def test_validate_catches_inverted_dates() -> None:
    # Removed BEFORE added — corrupt entry.
    members = {f"SYM{i}": ("2020-01-01", None) for i in range(500)}
    members["BUG"] = ("2020-01-01", "2019-01-01")
    errors = _validate_membership(members, min_active=400)
    assert any("BUG" in e and "not before" in e for e in errors)


# ---------------------------------------------------------------------------
# Diff vs existing CSV
# ---------------------------------------------------------------------------


def test_diff_returns_change_fraction_for_universe_replacement(tmp_path: Path) -> None:
    """Symmetric diff measured as fraction of OLD active count."""
    # Old CSV: 100 active names
    old_csv = tmp_path / "old.csv"
    old_rows = ["symbol,added,removed"]
    for i in range(100):
        old_rows.append(f"SYM{i},2020-01-01,")
    old_csv.write_text("\n".join(old_rows))

    # New: 95 of the old + 5 new = 5 added, 5 removed = 10/100 = 10%
    new_members = {f"SYM{i}": ("2020-01-01", None) for i in range(5, 100)}
    new_members.update({f"NEW{i}": ("2024-01-01", None) for i in range(5)})
    diff = _diff_against_existing(new_members, old_csv)
    assert diff["change_fraction"] == 0.10
    assert len(diff["added_symbols"]) == 5
    assert len(diff["removed_symbols"]) == 5
    assert diff["old_active_count"] == 100
    assert diff["new_active_count"] == 100


def test_diff_handles_missing_existing_csv(tmp_path: Path) -> None:
    """First-run case: no existing CSV → everything is 'added', change=100%."""
    new_members = {"AAPL": ("1982-11-30", None)}
    diff = _diff_against_existing(new_members, tmp_path / "nope.csv")
    assert diff["change_fraction"] == 1.0
    assert diff["added_symbols"] == ["AAPL"]
    assert diff["removed_symbols"] == []


# ---------------------------------------------------------------------------
# End-to-end with mocked fetch
# ---------------------------------------------------------------------------


class _RecordingSender:
    """EmailSender stand-in that captures sent messages."""
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, *, subject: str, body_text: str, body_html: str = "") -> None:  # noqa: ARG002
        self.sent.append((subject, body_text))


def test_refresh_rejects_when_too_few_active_members(tmp_path, monkeypatch) -> None:
    """End-to-end safety gate: a parse that returns < 400 active names
    is REJECTED, the existing CSV is untouched, FAILED email goes out."""
    # Pre-populate the "live" CSV with valid content.
    live_csv = tmp_path / "sp500.csv"
    live_csv.write_text("symbol,added,removed\nAAPL,1982-11-30,\nMSFT,1994-06-01,\n")
    original = live_csv.read_text()

    # Mock the fetch to return a tiny universe (3 names — far below 400).
    monkeypatch.setattr(
        "quant.agent.sp500_refresh._fetch_wikipedia_tables",
        lambda: (_wiki_style_current(), _wiki_style_changes()),
    )

    sender = _RecordingSender()
    ok = refresh_sp500_universe(
        csv_path=live_csv, email_sender=sender, min_active=400,
    )
    # Refresh rejected.
    assert ok is False
    # Live CSV untouched.
    assert live_csv.read_text() == original
    # FAILED email sent.
    assert len(sender.sent) == 1
    assert "FAILED" in sender.sent[0][0]


def test_refresh_rejects_when_change_too_large(tmp_path, monkeypatch) -> None:
    """If symmetric diff > threshold, refuse to replace — Wikipedia
    parse likely broke."""
    live_csv = tmp_path / "sp500.csv"
    old_rows = ["symbol,added,removed"]
    for i in range(500):
        old_rows.append(f"SYM{i},2020-01-01,")
    live_csv.write_text("\n".join(old_rows))
    original = live_csv.read_text()

    # Mock fetch: return 500 totally different names (100% churn).
    def _fake_fetch():
        cur = pd.DataFrame({
            "Symbol": [f"DIFF{i}" for i in range(500)],
            "Date added": ["2020-01-01"] * 500,
        })
        chg = pd.DataFrame({"Date": [], "Added Ticker": [], "Removed Ticker": []})
        return cur, chg
    monkeypatch.setattr(
        "quant.agent.sp500_refresh._fetch_wikipedia_tables", _fake_fetch,
    )

    sender = _RecordingSender()
    ok = refresh_sp500_universe(
        csv_path=live_csv, email_sender=sender,
        min_active=400, max_change_fraction=0.10,
    )
    assert ok is False
    assert live_csv.read_text() == original   # untouched
    assert "FAILED" in sender.sent[0][0]
    assert "symmetric diff" in sender.sent[0][1].lower()


def test_refresh_succeeds_with_reasonable_diff(tmp_path, monkeypatch) -> None:
    """Happy path: 500 names current, 5 names added, 5 removed → 2%
    diff, under the 10% threshold. Write succeeds, success email sent."""
    live_csv = tmp_path / "sp500.csv"
    old_rows = ["symbol,added,removed"]
    for i in range(500):
        old_rows.append(f"SYM{i},2020-01-01,")
    live_csv.write_text("\n".join(old_rows))

    def _fake_fetch():
        # 495 surviving + 5 new = 500 active. 5 of original 500 dropped.
        cur = pd.DataFrame({
            "Symbol": [f"SYM{i}" for i in range(5, 500)] + [f"NEW{i}" for i in range(5)],
            "Date added": ["2020-01-01"] * 495 + ["2024-06-01"] * 5,
        })
        # Removals show up in the changes table.
        chg = pd.DataFrame({
            "Date": ["2024-06-01"] * 5,
            "Added Ticker": [f"NEW{i}" for i in range(5)],
            "Removed Ticker": [f"SYM{i}" for i in range(5)],
        })
        return cur, chg
    monkeypatch.setattr(
        "quant.agent.sp500_refresh._fetch_wikipedia_tables", _fake_fetch,
    )

    sender = _RecordingSender()
    ok = refresh_sp500_universe(
        csv_path=live_csv, email_sender=sender,
        min_active=400, max_change_fraction=0.10,
    )
    assert ok is True
    # Live CSV WAS replaced.
    new_content = live_csv.read_text()
    assert new_content != old_rows[0] + "\n"
    assert "NEW0" in new_content
    assert "SYM499" in new_content
    # Success email sent.
    assert len(sender.sent) == 1
    assert "refreshed" in sender.sent[0][0].lower()
    body = sender.sent[0][1]
    assert "Added" in body
    assert "Removed" in body


# ---------------------------------------------------------------------------
# Email rendering
# ---------------------------------------------------------------------------


def test_render_email_success_includes_added_and_removed() -> None:
    diff = {
        "added_symbols": ["NEWCO"], "removed_symbols": ["OLDCO"],
        "date_corrections": [],
        "change_fraction": 0.01,
        "old_active_count": 500, "new_active_count": 500,
    }
    subj, body = _render_email(success=True, diff=diff)
    assert "refreshed" in subj.lower()
    assert "NEWCO" in body
    assert "OLDCO" in body


def test_render_email_failure_includes_errors() -> None:
    subj, body = _render_email(success=False, errors=["bad parse"])
    assert "FAILED" in subj
    assert "bad parse" in body
    # Use "not modified" — body explicitly says "The existing CSV was NOT modified."
    assert "not modified" in body.lower()


def test_write_csv_round_trips_via_pandas(tmp_path: Path) -> None:
    """The written CSV must be parseable by the universe loader."""
    members = {
        "AAPL": ("1982-11-30", None),
        "OLDCO": (None, "2024-01-01"),
    }
    out = tmp_path / "sp500.csv"
    _write_csv(members, out)
    df = pd.read_csv(out, comment="#")
    assert "symbol" in df.columns and "added" in df.columns and "removed" in df.columns
    assert set(df["symbol"]) == {"AAPL", "OLDCO"}
