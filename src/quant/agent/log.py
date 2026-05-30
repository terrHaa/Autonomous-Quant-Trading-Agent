"""log.py — JSON-per-day persistence for the agent's daily runs.

One file per trading day at ``data/agent/runs/YYYY-MM-DD.json``,
containing the full ``ExecutionReport`` plus the strategy's target
weights and the signal prices used for sizing. That's enough
information to reconstruct what the agent did and why on any past day.

Why JSON-per-day and not the SQLite registry?
---------------------------------------------
The registry is for BACKTEST runs — the strategy variant being recorded
is a hypothesis. Live paper runs are operational records, not
hypotheses, and they have a different lifecycle (one entry per day
forever rather than one per parameter combination). Mixing them in one
table makes both messier. The weekly/monthly review loads from these
JSON files; future trial-count-in-paper tracking can be a separate
sidecar if/when it matters.

Schema (the serialized JSON top-level fields):
    date            — ISO date string YYYY-MM-DD
    timestamp_utc   — ISO datetime when the run executed
    strategy_name   — for filtering / strategy-rotation later
    strategy_params — the dict used to construct the strategy
    target_weights  — what the strategy asked for
    signal_prices   — closes used for sizing
    execution_report — the broker-side record
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from quant.execution.alpaca_executor import ExecutionReport

# Project root → resolved from this file's location, mirroring config.py.
# log.py at src/quant/agent/log.py → 4 parents up = repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RUNS_DIR = _PROJECT_ROOT / "data" / "agent" / "runs"
DEFAULT_WEEKLY_DIR = _PROJECT_ROOT / "data" / "agent" / "weekly_reports"


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text to ``path`` atomically: tempfile in same dir, then rename.

    Prevents partial-file corruption if the process is killed mid-write —
    a real risk for state files that the daily-trade routine clobbers on
    every fire. ``os.replace`` is atomic on POSIX when source and dest
    are on the same filesystem (guaranteed here since the tempfile lives
    in the same parent directory).
    """
    import os
    import tempfile
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file…
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        # …then atomically swap into place. On crash before this point,
        # the original file is untouched; the tmp file is orphaned but
        # harmless (next run cleans it up via tempfile's own logic).
        os.replace(tmp_path, str(path))
    except BaseException:
        # Best-effort cleanup of the tmp file if rename never happened.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def save_daily_run(
    *,
    run_date: date,
    strategy_name: str,
    strategy_params: dict[str, Any],
    target_weights: dict[str, float],
    signal_prices: dict[str, float],
    execution_report: ExecutionReport,
    runs_dir: Path | None = None,
) -> Path:
    """Serialize one day's run to JSON. Returns the file path."""
    out_dir = runs_dir or DEFAULT_RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_date.isoformat()}.json"

    payload = {
        "date": run_date.isoformat(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "strategy_name": strategy_name,
        "strategy_params": strategy_params,
        "target_weights": dict(target_weights),
        "signal_prices": dict(signal_prices),
        "execution_report": _serialize_report(execution_report),
    }
    # default=str lets datetime, date, and other non-JSON-native types
    # serialize cleanly without us hand-writing a converter for each.
    _atomic_write_text(out_path, json.dumps(payload, default=str, indent=2))
    return out_path


def load_daily_run(
    run_date: date,
    *,
    runs_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Load a previously-saved run as a plain dict. ``None`` if no file."""
    out_dir = runs_dir or DEFAULT_RUNS_DIR
    path = out_dir / f"{run_date.isoformat()}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Weekly report persistence — feeds self-improvement (weekly N+1 reads
# past N weekly reports) and the monthly analyst's comprehensive analysis.
# ---------------------------------------------------------------------------


def save_weekly_report(
    *,
    week_ending: date,
    narrative: str,
    metrics: dict[str, Any],
    hrp_diagnostic: dict[str, Any] | None = None,
    weekly_dir: Path | None = None,
) -> Path:
    """Persist one week's AI deep-dive to JSON. Returns the file path.

    Stored at ``data/agent/weekly_reports/YYYY-MM-DD.json`` keyed by the
    Friday ending the week. Subsequent weekly + monthly reviews load these
    so the analyst has its own history to refer to.
    """
    out_dir = weekly_dir or DEFAULT_WEEKLY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{week_ending.isoformat()}.json"
    payload = {
        "week_ending": week_ending.isoformat(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "narrative": narrative,
        "metrics": metrics,
        "hrp_diagnostic": hrp_diagnostic or {},
    }
    _atomic_write_text(out_path, json.dumps(payload, default=str, indent=2))
    return out_path


def load_recent_weekly_reports(
    *,
    before: date | None = None,
    n: int = 4,
    weekly_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load up to N most-recent weekly reports STRICTLY BEFORE ``before``.

    Ordering: oldest → newest, so the analyst reads them in chronological
    order (matching how a human would catch up on history). Empty list
    when no reports exist (fresh install).
    """
    out_dir = weekly_dir or DEFAULT_WEEKLY_DIR
    if not out_dir.exists():
        return []
    cutoff = before or date.today()
    dated: list[tuple[date, Path]] = []
    for p in out_dir.glob("*.json"):
        try:
            d = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if d < cutoff:
            dated.append((d, p))
    dated.sort(key=lambda kv: kv[0])  # oldest → newest
    selected = dated[-n:] if n > 0 else []
    return [json.loads(p.read_text()) for _, p in selected]


# ---------------------------------------------------------------------------
# ExecutionReport serialization
# ---------------------------------------------------------------------------


def _serialize_report(report: ExecutionReport) -> dict[str, Any]:
    """Convert an ExecutionReport to a JSON-safe dict.

    Frozen dataclasses already serialize through ``asdict``; the only
    fiddly fields are the ``datetime`` (use isoformat) and the
    enum-typed Literals which stringify naturally.
    """
    data = asdict(report)
    data["timestamp"] = report.timestamp.isoformat()
    return data
