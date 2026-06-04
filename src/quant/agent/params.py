"""params.py — the StrategyParams data shape used by the monthly improver.

This module USED to hold the agent's persistent strategy parameters
(via ``load_params`` / ``save_params`` writing to a JSON file). Once the
ensemble landed, parameters moved into ``EnsembleState`` (see
``quant.agent.ensemble``), and the JSON persistence functions became
dead code — every production caller now reads/writes EnsembleState.

What remains here is the ``StrategyParams`` dataclass itself, which is
still the natural data shape the improver uses to express a candidate
parameter tuple (top_k, lookback, skip) for the cross-sectional
momentum sub-strategy. Keeping it in its own module avoids cluttering
the improver with the validation / constructor logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParams:
    """Tunable parameters for the cross-sectional momentum sub-strategy.

    NOT the operator's hard rules (5% stop, 20% per-trade cap) — those
    are constants in ``daily_runner.py``. SMA + MR params have parallel
    dataclasses below (``SmaParams``, ``MrParams``) — the monthly grid
    search exercises all three.

    The improver builds a grid of these and the monthly review writes
    the winning tuple back into the surrounding ``EnsembleState``.
    """

    top_k: int = 10
    lookback: int = 60
    skip: int = 5

    def __post_init__(self) -> None:
        # Validate at construction so a malformed candidate can never
        # reach the engine. Mirrors the strategy class's own checks.
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


@dataclass(frozen=True)
class SmaParams:
    """Tunable parameters for the SMA crossover sub-strategy."""

    fast: int = 50
    slow: int = 200

    def __post_init__(self) -> None:
        if self.fast < 2:
            raise ValueError(f"fast must be >= 2; got {self.fast}")
        if self.slow <= self.fast:
            raise ValueError(
                f"slow ({self.slow}) must be > fast ({self.fast})"
            )


@dataclass(frozen=True)
class MrParams:
    """Tunable parameters for the mean-reversion sub-strategy.

    Vol-normalize defaults match the strategy class's v2 defaults.
    The improver may flip vol_normalize off to test the static-threshold
    path — useful for backtest comparison.
    """

    lookback: int = 5
    threshold_pct: float = 0.02
    vol_normalize: bool = True
    vol_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError(f"lookback must be >= 2; got {self.lookback}")
        if self.threshold_pct <= 0:
            raise ValueError(
                f"threshold_pct must be > 0; got {self.threshold_pct}"
            )
        if self.vol_multiplier <= 0:
            raise ValueError(
                f"vol_multiplier must be > 0; got {self.vol_multiplier}"
            )
