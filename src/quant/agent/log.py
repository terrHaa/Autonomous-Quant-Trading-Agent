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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from quant.execution.alpaca_executor import ExecutionReport


# Project root → resolved from this file's location, mirroring config.py.
# log.py at src/quant/agent/log.py → 4 parents up = repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RUNS_DIR = _PROJECT_ROOT / "data" / "agent" / "runs"


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
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "strategy_name": strategy_name,
        "strategy_params": strategy_params,
        "target_weights": dict(target_weights),
        "signal_prices": dict(signal_prices),
        "execution_report": _serialize_report(execution_report),
    }
    # default=str lets datetime, date, and other non-JSON-native types
    # serialize cleanly without us hand-writing a converter for each.
    out_path.write_text(json.dumps(payload, default=str, indent=2))
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


def list_recent_runs(
    *,
    runs_dir: Path | None = None,
    limit: int | None = None,
) -> list[date]:
    """List run dates in the directory, newest first.

    ``limit=10`` returns the 10 most recent. ``None`` returns all.
    """
    out_dir = runs_dir or DEFAULT_RUNS_DIR
    if not out_dir.exists():
        return []
    dates: list[date] = []
    for p in out_dir.glob("*.json"):
        try:
            dates.append(date.fromisoformat(p.stem))
        except ValueError:
            # Stray file with a non-date stem — ignore rather than crash.
            continue
    dates.sort(reverse=True)
    return dates[:limit] if limit else dates


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
