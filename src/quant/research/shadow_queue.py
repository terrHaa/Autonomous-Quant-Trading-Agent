"""shadow_queue.py — multi-candidate shadow queue (research substrate, phase A2).

Anything that clears the sandbox doesn't go live — it enters here and
accrues an out-of-sample *live paper* track record with zero real
allocation, until the promotion criteria (phase A4) say it's earned a
seat. This is the wall-clock-bound validation step that no amount of
backtesting can replace: a strategy can be gorgeous in-sample and fall
apart the moment it meets unseen data, and the only way to know is to
watch it forward.

Today's live ensemble already has a single-strategy shadow flag
(``EnsembleState.ai_strategy_shadow_until``). This is the generalization:
a *multi-candidate* registry that holds many candidates at once, each with
its own accruing OOS return series, decoupled from the live ensemble so
testing more things never touches the trading path. A candidate links back
to its ``trial_id`` in the global ledger, so its whole life — backtest →
shadow → promote/reject — is one auditable thread.

Mutable working state (unlike the append-only ledger): a JSON dict keyed
by candidate id, rewritten atomically. Never raises on read.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/agent/research/shadow_queue.json")

STATUSES: tuple[str, ...] = ("shadow", "promoted", "rejected")


@dataclass
class ShadowCandidate:
    """One candidate accruing an out-of-sample paper record."""

    candidate_id: str
    name: str
    backtest_sharpe: float
    entered_at: str                          # iso date the shadow period began
    status: str = "shadow"
    shadow_returns: dict[str, float] = field(default_factory=dict)  # iso date -> ret
    metadata: dict[str, Any] = field(default_factory=dict)

    def returns_series(self) -> pd.Series:
        if not self.shadow_returns:
            return pd.Series(dtype=float)
        s = pd.Series(self.shadow_returns)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    @property
    def shadow_days(self) -> int:
        return len(self.shadow_returns)


class ShadowQueue:
    """Persistent registry of candidates in their OOS shadow period."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else _DEFAULT_PATH

    # ---- persistence ------------------------------------------------------

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("shadow queue read failed: %s", e)
            return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-write can't corrupt the queue.
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, default=str, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ---- writes -----------------------------------------------------------

    def add(
        self,
        *,
        candidate_id: str,
        name: str,
        backtest_sharpe: float,
        entered_at: date | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Enter a candidate into the shadow period."""
        data = self._load()
        data[candidate_id] = {
            "candidate_id": candidate_id,
            "name": name,
            "backtest_sharpe": float(backtest_sharpe),
            "entered_at": (entered_at or datetime.now(UTC).date()).isoformat(),
            "status": "shadow",
            "shadow_returns": {},
            "metadata": metadata or {},
        }
        self._save(data)

    def record_return(self, candidate_id: str, d: date, ret: float) -> None:
        """Append one day's OOS paper return for a shadowing candidate."""
        data = self._load()
        c = data.get(candidate_id)
        if c is None:
            logger.warning("shadow queue: record_return for unknown %s", candidate_id)
            return
        c["shadow_returns"][d.isoformat()] = float(ret)
        self._save(data)

    def set_status(self, candidate_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; must be one of {STATUSES}")
        data = self._load()
        if candidate_id in data:
            data[candidate_id]["status"] = status
            self._save(data)

    # ---- reads ------------------------------------------------------------

    def get(self, candidate_id: str) -> ShadowCandidate | None:
        c = self._load().get(candidate_id)
        return self._to_candidate(c) if c else None

    def candidates(self, *, status: str | None = None) -> list[ShadowCandidate]:
        out = [self._to_candidate(c) for c in self._load().values()]
        return [c for c in out if status is None or c.status == status]

    @staticmethod
    def _to_candidate(c: dict[str, Any]) -> ShadowCandidate:
        return ShadowCandidate(
            candidate_id=c["candidate_id"],
            name=c.get("name", ""),
            backtest_sharpe=float(c.get("backtest_sharpe", 0.0)),
            entered_at=c.get("entered_at", ""),
            status=c.get("status", "shadow"),
            shadow_returns={k: float(v) for k, v in c.get("shadow_returns", {}).items()},
            metadata=c.get("metadata", {}),
        )
