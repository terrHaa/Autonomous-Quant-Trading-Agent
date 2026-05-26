"""params.py — persistent strategy parameters for the agent.

The agent's HARD operator rules (5% stop, 20% per-trade cap) are constants
in ``daily_runner.py`` — the auto-improver is forbidden from touching
them. But the strategy's OWN parameters (top_k, lookback, skip) can be
tuned by the monthly review process, so they live in a small persistent
JSON file rather than as code constants.

If the file doesn't exist, ``load_params()`` returns the v1 defaults
(top_k=10, lookback=60, skip=5). The monthly improver may then
``save_params()`` a new tuple after gating it on DSR + drawdown. The
daily runner picks the new values up on its next morning run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


# Same project-root trick used elsewhere. params.py at
# src/quant/agent/params.py → 4 parents up = repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_PARAMS_PATH = _PROJECT_ROOT / "data" / "agent" / "strategy_params.json"


@dataclass(frozen=True)
class StrategyParams:
    """Tunable strategy parameters. NOT the operator's hard rules."""

    top_k: int = 10
    lookback: int = 60
    skip: int = 5

    def __post_init__(self) -> None:
        # Same validation as CrossSectionalMomentum but applied at the
        # config-load boundary so the daily runner doesn't blow up
        # halfway through a session.
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {self.top_k}")
        if self.lookback < 2:
            raise ValueError(f"lookback must be >= 2; got {self.lookback}")
        if self.skip < 0:
            raise ValueError(f"skip must be >= 0; got {self.skip}")
        if self.skip >= self.lookback:
            raise ValueError(
                f"skip ({self.skip}) must be < lookback ({self.lookback})"
            )


def load_params(path: Path | None = None) -> StrategyParams:
    """Load params from JSON; return defaults if no file exists."""
    p = path or DEFAULT_PARAMS_PATH
    if not p.exists():
        return StrategyParams()
    data = json.loads(p.read_text())
    return StrategyParams(
        top_k=int(data.get("top_k", 10)),
        lookback=int(data.get("lookback", 60)),
        skip=int(data.get("skip", 5)),
    )


def save_params(params: StrategyParams, path: Path | None = None) -> Path:
    """Persist params to JSON. Creates the parent directory if missing."""
    p = path or DEFAULT_PARAMS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(params), indent=2))
    return p
