"""registry.py — SQLite-backed audit trail of every backtest run.

What this module is for
-----------------------
Two jobs, one component:

1. **The honest trial count for DSR.** Every strategy variant you've ever
   tested becomes a row in this database. When you ask "is the deflated
   Sharpe of my best strategy significant?", the registry tells you how
   many trials to deflate against. Cheating on the trial count is the
   single most common way to fool yourself with backtested Sharpe — the
   registry exists to make cheating require deliberate action.

2. **Promotion-stage tracking.** Every strategy variant has a stage:
   ``research → walk_forward → paper → live``. Promotion is an explicit
   API call that validates the transition and (for the critical
   walk_forward → paper hop) enforces a DSR gate.

The registry intentionally stores ONLY SUMMARY statistics — Sharpe, max
drawdown, etc. — not the full equity curves. That keeps the file tiny
(under a MB even after thousands of runs) and the queries snappy. When
you need the actual returns (e.g. for DSR), pass them in as an argument;
the registry computes the deflation using its persistent trial count.

What's NOT in here (yet)
------------------------
- Code/config provenance (git SHA, config hash). Useful, but adds complexity.
  Until that's wired in, treat the registry as a "what numbers came out"
  log, not a perfect-reproducibility log.
- Soft-delete and version migration. Schema changes will require manual SQL.
- Automated paper → live gate. That needs paper-trading telemetry from
  Step 21's execution module.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from quant.backtest.engine import BacktestResult
from quant.evaluation.dsr import (
    deflated_sharpe_ratio,
    estimate_var_sr_from_trials,
)
from quant.evaluation.metrics import metrics_for

# The promotion ladder. A run starts at the leftmost stage and can only
# move rightward; "research → live" without intermediate stages is forbidden.
Stage = Literal["research", "walk_forward", "paper", "live"]
STAGES: tuple[Stage, ...] = ("research", "walk_forward", "paper", "live")


# Schema applied on first open. CREATE TABLE IF NOT EXISTS makes this
# idempotent — safe to call every time we open the DB.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    parameters_json TEXT,
    start_date TEXT,
    end_date TEXT,
    n_bars INTEGER,
    starting_equity REAL,
    ending_equity REAL,
    sharpe REAL,
    sortino REAL,
    max_drawdown REAL,
    total_return REAL,
    n_fills INTEGER,
    stage TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy_name);
CREATE INDEX IF NOT EXISTS idx_runs_stage ON runs(stage);
"""


class Registry:
    """SQLite-backed registry of backtest runs.

    Concurrency: SQLite's default mode handles single-process multi-thread
    fine but multi-process writes can collide. For now we assume one
    research process writing at a time. WAL mode would be the next step
    if that changes.
    """

    def __init__(self, path: Path | str) -> None:
        """Open (or create) the registry at ``path``.

        Creates parent directories and applies the schema if needed.
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize schema. Idempotent.
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Connection helper. One short-lived connection per operation —
    # wasteful but trivially correct. Optimize if it ever shows up in
    # profiles.
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._path))
        # Row factory: rows come back like dicts (row["sharpe"]) not tuples.
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        result: BacktestResult,
        *,
        parameters: dict | None = None,
        stage: Stage = "research",
        notes: str = "",
    ) -> str:
        """Record one backtest run. Returns the new run's UUID.

        ``parameters`` is a JSON-serializable dict of the strategy's
        constructor kwargs (or anything you want to remember about how
        this variant was configured). Defaults to ``{}``.
        """
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}; got {stage!r}")

        metrics = metrics_for(result)
        run_id = str(uuid.uuid4())
        now_iso = datetime.now(UTC).isoformat()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    id, strategy_name, parameters_json,
                    start_date, end_date, n_bars,
                    starting_equity, ending_equity,
                    sharpe, sortino, max_drawdown, total_return,
                    n_fills, stage, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    result.strategy_name,
                    # default=str so date / datetime objects serialize cleanly.
                    json.dumps(parameters or {}, default=str),
                    str(result.metadata.get("start_date")),
                    str(result.metadata.get("end_date")),
                    metrics.n_days,
                    metrics.starting_equity,
                    metrics.ending_equity,
                    metrics.sharpe,
                    metrics.sortino,
                    metrics.max_drawdown,
                    metrics.total_return,
                    int(result.metadata.get("n_fills", 0)),
                    stage,
                    notes,
                    now_iso,
                ),
            )
        return run_id

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list_runs(
        self,
        *,
        strategy_name: str | None = None,
        stage: Stage | None = None,
    ) -> pd.DataFrame:
        """Return matching runs as a DataFrame, newest first."""
        sql = "SELECT * FROM runs"
        params: list[str] = []
        conds: list[str] = []
        if strategy_name is not None:
            conds.append("strategy_name = ?")
            params.append(strategy_name)
        if stage is not None:
            conds.append("stage = ?")
            params.append(stage)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY created_at DESC"

        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get(self, run_id: str) -> dict | None:
        """Fetch one run by ID. Returns None if no such run."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def delete(self, run_id: str) -> bool:
        """Delete a run. Returns True if a row was deleted, False if not found.

        Use sparingly — deleting a run silently lies to DSR's trial count.
        Prefer to leave failed variants in place (they ARE trials).
        """
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Trial accounting (the bit DSR plugs into)
    # ------------------------------------------------------------------

    def n_trials(self) -> int:
        """Total number of recorded runs. THE trial count for DSR.

        Honest counting matters: if you tested 100 variants and only
        recorded the winners, your DSR will be over-stated. The default
        behavior of ``record()`` is to capture every run, including
        losers — keep it that way.
        """
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def trial_sharpes(self) -> list[float]:
        """All recorded annualized Sharpes (in insertion order).

        Pass into ``estimate_var_sr_from_trials`` for the V[SR] term of DSR.
        """
        with self._conn() as conn:
            return [float(r[0]) for r in conn.execute("SELECT sharpe FROM runs")]

    # ------------------------------------------------------------------
    # Stage management + promotion gates
    # ------------------------------------------------------------------

    def promote(self, run_id: str, *, to_stage: Stage) -> None:
        """Move a run to a later stage.

        Validates that the destination stage is strictly later than the
        current stage on the ladder ``research → walk_forward → paper → live``.

        This is the *plumbing* — it does NOT enforce DSR or other
        promotion criteria. Use ``check_promotion_gate`` to test those
        before calling ``promote``.
        """
        if to_stage not in STAGES:
            raise ValueError(
                f"unknown stage {to_stage!r}; valid stages: {STAGES}"
            )
        current = self.get(run_id)
        if current is None:
            raise KeyError(f"no run with id {run_id!r}")

        current_idx = STAGES.index(current["stage"])
        new_idx = STAGES.index(to_stage)
        if new_idx <= current_idx:
            raise ValueError(
                f"cannot promote run {run_id!r} from {current['stage']!r} "
                f"to {to_stage!r}: target stage must be strictly later"
            )

        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET stage = ? WHERE id = ?",
                (to_stage, run_id),
            )

    def check_promotion_gate(
        self,
        run_id: str,
        *,
        to_stage: Stage,
        returns: pd.Series | None = None,
        dsr_threshold: float = 0.95,
    ) -> tuple[bool, str]:
        """Test whether a run is eligible for the requested promotion.

        Returns (eligible, reason). ``reason`` is human-readable and
        explains the outcome regardless of pass/fail.

        Gates currently enforced:
        - **research → walk_forward**: no automated gate (manual review;
          typically "you actually ran walk-forward on this variant").
        - **walk_forward → paper**: DSR ≥ ``dsr_threshold`` against the
          full trial population in the registry. THIS is the critical
          "is the strategy real?" gate.
        - **paper → live**: no automated gate yet. Will require
          paper-trading telemetry once execution is wired (Step 21).
        """
        current = self.get(run_id)
        if current is None:
            return False, f"no run with id {run_id!r}"

        if to_stage == "paper":
            if returns is None:
                return False, (
                    "DSR gate requires the candidate's returns series; "
                    "pass it via the `returns=` argument"
                )
            sharpes = self.trial_sharpes()
            if len(sharpes) < 2:
                return False, (
                    f"need ≥ 2 trials in the registry to estimate V[SR]; "
                    f"got {len(sharpes)}"
                )
            n_trials = self.n_trials()
            var_sr = estimate_var_sr_from_trials(sharpes)
            dsr = deflated_sharpe_ratio(
                returns,
                n_trials=n_trials,
                var_sr_trials_annual=var_sr,
            )
            if dsr >= dsr_threshold:
                return True, f"DSR = {dsr:.3f} ≥ {dsr_threshold} ({n_trials} trials)"
            return False, f"DSR = {dsr:.3f} < {dsr_threshold} ({n_trials} trials)"

        # Other transitions: no automated criteria yet. The caller is
        # promising they've done the manual review.
        return True, f"no automated gate for {current['stage']} → {to_stage} yet"
