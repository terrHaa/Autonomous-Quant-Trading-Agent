"""Tests for the typed config loader.

Each test is named after the failure mode it prevents. If a test name doesn't
explain *why* we care, the test is probably noise — kill it or rewrite it.
"""

from copy import deepcopy
from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from quant.config import DEFAULT_CONFIG_PATH, Config, load_config


@pytest.fixture
def default_dict() -> dict:
    """Return the default config as a raw Python dict.

    Tests that want to corrupt one field at a time start from this fixture
    rather than re-spelling the entire config — keeps tests focused on the
    one thing they're actually checking.
    """
    with DEFAULT_CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def test_default_config_loads() -> None:
    """The shipped default.yaml must always parse cleanly.

    If this fails, someone changed default.yaml without updating the
    Pydantic schema (or vice versa). Either side can drift; this test
    pins them together.
    """
    cfg = load_config()

    assert isinstance(cfg, Config)
    # Spot-check one value from each top-level section so we'd notice if
    # the YAML or the schema started returning the wrong shape.
    assert cfg.universe.name == "sp500_liquid"
    assert cfg.dates.start == date(2015, 1, 1)
    assert cfg.risk.vol_target_annual == 0.10
    assert cfg.backtest.rebalance == "daily"
    assert cfg.evaluation.trading_days_per_year == 252
    assert cfg.execution.alpaca_env == "paper"


def test_typo_in_yaml_key_raises(default_dict: dict) -> None:
    """`extra="forbid"` should reject unknown keys.

    Without this, a typo like `vol_taregt_annual` would be silently dropped
    and the strategy would run with whatever default the schema had — one of
    the most insidious config bugs.
    """
    bad = deepcopy(default_dict)
    bad["risk"]["vol_taregt_annual"] = 0.10  # deliberately misspelled

    # `match="vol_taregt_annual"` asserts that the error message actually
    # names the bad field — a generic "validation failed" wouldn't help a
    # user fix their YAML.
    with pytest.raises(ValidationError, match="vol_taregt_annual"):
        Config.model_validate(bad)


def test_date_ordering_enforced(default_dict: dict) -> None:
    """train_end after end is nonsense and must raise.

    If we let this through, every walk-forward split downstream would put
    test data inside the training window — silent look-ahead leakage.
    """
    bad = deepcopy(default_dict)
    bad["dates"]["train_end"] = "2025-12-31"  # after end (2024-12-31)

    with pytest.raises(ValidationError, match="start <= train_end <= end"):
        Config.model_validate(bad)


def test_out_of_range_vol_target_raises(default_dict: dict) -> None:
    """A vol target of 10.0 is almost certainly a typo for 0.10.

    The `le=1.0` constraint on RiskConfig.vol_target_annual catches this at
    load time, before we wire up a backtest sized for 1000% annualized vol.
    """
    bad = deepcopy(default_dict)
    bad["risk"]["vol_target_annual"] = 10.0  # forgot the decimal point

    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_invalid_literal_value_raises(default_dict: dict) -> None:
    """A misspelled rebalance frequency should fail loudly.

    `Literal["daily", "weekly", "monthly"]` means "weakly" can't sneak in.
    """
    bad = deepcopy(default_dict)
    bad["backtest"]["rebalance"] = "weakly"  # misspelled "weekly"

    with pytest.raises(ValidationError, match="rebalance"):
        Config.model_validate(bad)


def test_config_is_frozen() -> None:
    """The loaded config must be immutable.

    Live trading and backtesting share the same `Config` shape; allowing
    mutation invites a bug where one component changes a parameter and
    another component sees the change at an unexpected moment.
    """
    cfg = load_config()

    # Pydantic raises ValidationError on assignment when frozen=True.
    with pytest.raises(ValidationError):
        cfg.risk.vol_target_annual = 0.20  # type: ignore[misc]
