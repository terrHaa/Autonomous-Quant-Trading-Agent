"""config.py — typed, validated configuration loader.

The whole platform reads its parameters from `configs/*.yaml`. Rather than
sprinkling `cfg["risk"]["vol_target_annual"]` dict lookups everywhere (which
silently return None for typos and then break in math code far from the
actual bug), we parse YAML into Pydantic models once at startup. After that:

- Every config access is a typed attribute: `cfg.risk.vol_target_annual`.
- Typos in the YAML raise immediately with a clear error pointing at the field.
- Numeric ranges and string enums are validated at load time.
- The config object is frozen, so nothing downstream can mutate it.
- Your editor / type checker knows the shape of `cfg`.

If you change a *value* in default.yaml, you don't need to touch this file.
If you *add* a new field to default.yaml, add the matching attribute here too
— otherwise `extra="forbid"` will reject the new field.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Path to the project root, computed from this file's location.
# This file lives at `src/quant/config.py`, so the project root is three
# parents up: src/quant -> src -> <project root>.
# `.resolve()` turns the path into an absolute path with symlinks expanded,
# so this works no matter where you run Python from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"


# A small base class so every section gets the same strict settings.
#
# `extra="forbid"`: a typo in the YAML (e.g. `vol_taregt_annual:` instead of
# `vol_target_annual:`) raises a ValidationError instead of being silently
# dropped. Without this, you'd think you set the vol target and actually be
# running with the schema default — one of the most insidious config bugs.
#
# `frozen=True`: the loaded config is immutable. If someone tries to do
# `cfg.risk.vol_target_annual = 0.20` at runtime, Python raises. This forces
# overrides to go through "create a new config" rather than mutating shared
# state, which is the only sane behavior in a system that mixes backtests
# and live trading.
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Per-section models. Each one mirrors a top-level key in default.yaml.
# Field(...) attaches validation constraints — see Pydantic docs for the full
# vocabulary. The ones we use here:
#   gt = greater than       ge = greater than or equal
#   lt = less than          le = less than or equal
# ---------------------------------------------------------------------------


class UniverseConfig(_StrictModel):
    """The set of tickers a strategy is allowed to consider."""

    # Named universe (e.g. "sp500_liquid"). The data layer resolves this name
    # into an actual list of tickers — using a name instead of a hard-coded
    # list is what lets us reconstruct historical membership without
    # survivorship bias.
    name: str

    # Liquidity filter applied on top of the named universe. USD.
    # `ge=0` says "must be non-negative" — a negative volume floor is nonsense.
    min_avg_dollar_volume_20d: float = Field(ge=0)

    # Price filter to dodge penny-stock dynamics.
    min_price: float = Field(ge=0)


class DatesConfig(_StrictModel):
    """Date range covered by the backtest.

    Pydantic auto-parses ISO date strings (`"2015-01-01"`) into Python `date`
    objects, so we get real date arithmetic, not string compare.
    """

    start: date
    end: date
    # Held-out OOS window: training/research uses data up to train_end;
    # final evaluation uses (train_end, end].
    train_end: date

    @model_validator(mode="after")
    def _check_ordering(self) -> DatesConfig:
        # Cross-field validator: runs after all individual fields parse.
        # If someone swaps these by accident, every downstream split is wrong
        # in subtle ways — better to fail loudly at config load.
        if not (self.start <= self.train_end <= self.end):
            raise ValueError(
                f"Dates must satisfy start <= train_end <= end, got "
                f"start={self.start} train_end={self.train_end} end={self.end}"
            )
        return self


class RiskConfig(_StrictModel):
    """Hard risk limits enforced by the risk overlay (see quant.risk)."""

    # Annualized vol target, decimal. 0.10 = 10%.
    # le=1.0 (max 100% vol) catches the classic typo of writing 10 instead of
    # 0.10 — that would be 1000% target vol, almost certainly a mistake.
    vol_target_annual: float = Field(gt=0, le=1.0)

    # Gross leverage cap. ge=1.0 because a sub-1 gross cap would prevent any
    # meaningful long+short book.
    max_gross_leverage: float = Field(ge=1.0)

    # Net exposure cap, decimal. 1.0 = fully long allowed; 0.0 = market-neutral.
    max_net_exposure: float = Field(ge=0.0, le=2.0)

    # Per-name cap as a fraction of equity. le=0.5 is a safety rail — no
    # systematic equity book should put half the portfolio in one name.
    max_position_weight: float = Field(gt=0, le=0.5)

    # Drawdown kill switch threshold. 0.15 = flatten at 15% peak-to-trough.
    max_drawdown_kill: float = Field(gt=0, lt=1.0)


class CostsConfig(_StrictModel):
    """Per-trade cost model, all in basis points (1 bp = 0.01% = 0.0001)."""

    commission_bps: float = Field(ge=0)
    spread_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)


class BacktestConfig(_StrictModel):
    """Backtest engine settings."""

    starting_equity: float = Field(gt=0)

    # `Literal[...]` restricts the value to a fixed set of strings. Writing
    # `rebalance: weakly` instead of `weekly` will fail at config load with a
    # message listing the allowed values.
    rebalance: Literal["daily", "weekly", "monthly"]

    # Nested model: the YAML's `costs:` sub-block becomes a CostsConfig.
    costs: CostsConfig


class WalkForwardConfig(_StrictModel):
    """Rolling train/test windows for walk-forward analysis."""

    train_years: int = Field(ge=1)
    test_years: int = Field(ge=1)
    step_years: int = Field(ge=1)


class EvaluationConfig(_StrictModel):
    """Settings for metrics + walk-forward."""

    risk_free_annual: float = Field(ge=0)

    # 252 is the US equity convention. Some literatures use 250 or 260.
    trading_days_per_year: int = Field(gt=0)

    walk_forward: WalkForwardConfig


class ExecutionConfig(_StrictModel):
    """Live/paper execution settings (ignored in backtests)."""

    # Restricted to brokers we actually implement. Add more here when we
    # support more brokers — keeps the YAML honest about what's available.
    broker: Literal["alpaca"]

    # Which Alpaca environment to use by default. The runtime can still
    # override via ALPACA_ENV in `.env`.
    alpaca_env: Literal["paper", "live"]


class Config(_StrictModel):
    """Top-level config — the object `load_config()` returns."""

    universe: UniverseConfig
    dates: DatesConfig
    risk: RiskConfig
    backtest: BacktestConfig
    evaluation: EvaluationConfig
    execution: ExecutionConfig


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate a YAML config file.

    Parameters
    ----------
    path
        Path to a YAML file. If None (the default), loads
        `configs/default.yaml` from the project root.

    Returns
    -------
    A fully-validated `Config` object. Any missing field, typo'd key,
    out-of-range number, or wrong type raises `pydantic.ValidationError`
    with a clear message pointing to the offending field.

    Example
    -------
    >>> from quant.config import load_config
    >>> cfg = load_config()
    >>> cfg.risk.vol_target_annual
    0.1
    >>> cfg.execution.alpaca_env
    'paper'
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    # `yaml.safe_load` is the *safe* variant. `yaml.load` can execute
    # arbitrary Python embedded in the YAML — historically the source of a
    # whole class of remote code execution bugs. Always use safe_load for
    # config files; never use load.
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    # Pydantic does the heavy lifting from here: parses types, enforces
    # numeric ranges, runs the cross-field validators (e.g. date ordering),
    # and rejects unknown keys via extra="forbid".
    return Config.model_validate(raw)
