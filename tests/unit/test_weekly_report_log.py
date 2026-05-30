"""Tests for the weekly-report persistence layer in log.py.

This is the storage that lets the weekly analyst self-improve (week N+1
reads N) and the monthly analyst build on weekly observations.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from quant.agent.log import (
    load_recent_weekly_reports,
    save_weekly_report,
)


def test_save_round_trips_through_load(tmp_path: Path) -> None:
    """Save → load returns the same payload."""
    save_weekly_report(
        week_ending=date(2026, 5, 29),
        narrative="## Headline\n\nThe week was +1.2%.",
        metrics={"total_return_pct": 1.2, "ann_sharpe": 0.7},
        hrp_diagnostic={"per_strategy": {"sma": {"sharpe": 0.5}}},
        weekly_dir=tmp_path,
    )
    loaded = load_recent_weekly_reports(
        before=date(2026, 6, 1), n=10, weekly_dir=tmp_path,
    )
    assert len(loaded) == 1
    r = loaded[0]
    assert r["week_ending"] == "2026-05-29"
    assert "+1.2%" in r["narrative"]
    assert r["metrics"]["total_return_pct"] == 1.2
    assert r["hrp_diagnostic"]["per_strategy"]["sma"]["sharpe"] == 0.5


def test_load_returns_empty_when_no_dir(tmp_path: Path) -> None:
    """Fresh install: no weekly_reports dir → returns empty list, not error."""
    out = load_recent_weekly_reports(weekly_dir=tmp_path / "nope")
    assert out == []


def test_load_strictly_before_cutoff(tmp_path: Path) -> None:
    """A report dated == before should NOT be included; only strictly earlier."""
    for d in [date(2026, 5, 15), date(2026, 5, 22), date(2026, 5, 29)]:
        save_weekly_report(
            week_ending=d, narrative=f"week {d}",
            metrics={}, weekly_dir=tmp_path,
        )
    # Cutoff exactly equals the latest report — should exclude it.
    out = load_recent_weekly_reports(
        before=date(2026, 5, 29), n=10, weekly_dir=tmp_path,
    )
    weeks = [r["week_ending"] for r in out]
    assert "2026-05-29" not in weeks
    assert weeks == ["2026-05-15", "2026-05-22"]


def test_load_returns_oldest_first(tmp_path: Path) -> None:
    """Chronological order = oldest first (analyst reads in time order)."""
    for d in [date(2026, 5, 29), date(2026, 5, 8), date(2026, 5, 22), date(2026, 5, 15)]:
        save_weekly_report(
            week_ending=d, narrative=f"week {d}",
            metrics={}, weekly_dir=tmp_path,
        )
    out = load_recent_weekly_reports(
        before=date(2026, 6, 1), n=10, weekly_dir=tmp_path,
    )
    weeks = [r["week_ending"] for r in out]
    assert weeks == ["2026-05-08", "2026-05-15", "2026-05-22", "2026-05-29"]


def test_load_caps_at_n_keeps_newest(tmp_path: Path) -> None:
    """N=2 → return the 2 newest (still oldest-first within that window)."""
    for d in [date(2026, 5, 1), date(2026, 5, 8), date(2026, 5, 15),
              date(2026, 5, 22), date(2026, 5, 29)]:
        save_weekly_report(
            week_ending=d, narrative=f"week {d}",
            metrics={}, weekly_dir=tmp_path,
        )
    out = load_recent_weekly_reports(
        before=date(2026, 6, 1), n=2, weekly_dir=tmp_path,
    )
    weeks = [r["week_ending"] for r in out]
    # Newest 2 are 5-22 and 5-29; oldest of that pair listed first.
    assert weeks == ["2026-05-22", "2026-05-29"]


def test_save_overwrites_existing_report(tmp_path: Path) -> None:
    """Re-running a week's review (e.g., after SMTP failure) overwrites in place."""
    save_weekly_report(
        week_ending=date(2026, 5, 29), narrative="v1",
        metrics={"total_return_pct": 1.0}, weekly_dir=tmp_path,
    )
    save_weekly_report(
        week_ending=date(2026, 5, 29), narrative="v2 (corrected)",
        metrics={"total_return_pct": 1.5}, weekly_dir=tmp_path,
    )
    # One file, latest content.
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["narrative"] == "v2 (corrected)"
    assert payload["metrics"]["total_return_pct"] == 1.5


def test_save_is_atomic_no_orphan_tmp_files(tmp_path: Path) -> None:
    """save_weekly_report uses tempfile+rename → no partial-write artifacts.

    After a successful save, the directory contains exactly the final
    file (no .tmp orphans). A future audit on this directory shouldn't
    trip over half-written state.
    """
    save_weekly_report(
        week_ending=date(2026, 5, 29), narrative="x",
        metrics={}, weekly_dir=tmp_path,
    )
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "2026-05-29.json"


def test_load_ignores_non_iso_filenames(tmp_path: Path) -> None:
    """Stray files like .DS_Store or .gitignore shouldn't crash the loader."""
    (tmp_path / ".DS_Store").write_text("junk")
    (tmp_path / "summary.txt").write_text("not json")
    save_weekly_report(
        week_ending=date(2026, 5, 29), narrative="real",
        metrics={}, weekly_dir=tmp_path,
    )
    out = load_recent_weekly_reports(
        before=date(2026, 6, 1), n=10, weekly_dir=tmp_path,
    )
    assert len(out) == 1
    assert out[0]["week_ending"] == "2026-05-29"
