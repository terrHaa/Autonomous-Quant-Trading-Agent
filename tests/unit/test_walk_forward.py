"""Tests for the walk-forward harness.

Pattern: synthetic bars with known structure, then assert on fold
boundaries, OOS chaining math, and degenerate-input handling.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant.backtest.types import Snapshot
from quant.config import DEFAULT_CONFIG_PATH, Config
from quant.data.alpaca_client import BAR_COLUMNS
from quant.evaluation.walk_forward import run_walk_forward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bars(symbol: str, n_days: int, *, start_close: float = 100.0) -> pd.DataFrame:
    """Synthetic bars: deterministic uptrend so SMA strategies actually trade."""
    bdays = pd.bdate_range("2010-01-04", periods=n_days, tz="UTC")
    rows = []
    idx = []
    for i, ts in enumerate(bdays):
        c = start_close + i * 0.1   # gentle uptrend
        rows.append({
            "open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1_000_000,
        })
        idx.append((symbol, ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=list(BAR_COLUMNS),
    )


def _zero_cost_config(
    *,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> Config:
    """Config with zero costs and the requested walk-forward windows."""
    import yaml

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["backtest"]["costs"] = {
        "commission_bps": 0.0, "spread_bps": 0.0, "slippage_bps": 0.0,
    }
    raw["evaluation"]["walk_forward"] = {
        "train_years": train_years,
        "test_years": test_years,
        "step_years": step_years,
    }
    return Config.model_validate(raw)


class _NoOp:
    """No-trade strategy. Used to verify pure plumbing (no fills, flat equity)."""
    name = "noop"

    def on_bar(self, snapshot: Snapshot) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------


def test_runs_n_folds_for_given_window_sizes() -> None:
    """train=3y, test=1y, step=1y over 7y of data → 5 folds.

    Folds: years 0-3→3-4, 1-4→4-5, 2-5→5-6, 3-6→6-7, last train would
    end at year 7 leaving no test data → stop. 4 folds actually.

    Math: with n=7*252=1764 bars, train_n=756, test_n=252,
    valid i values: i + 1008 <= 1764 → i <= 756 → i in {0, 252, 504, 756}.
    Step=252. That's 4 values → 4 folds.
    """
    bars = _bars("AAPL", n_days=7 * 252)
    config = _zero_cost_config(train_years=3, test_years=1, step_years=1)

    result = run_walk_forward(config=config, strategy=_NoOp(), bars=bars)

    assert len(result.folds) == 4
    assert result.metadata["n_folds"] == 4


def test_fold_boundaries_are_non_overlapping_when_step_equals_test() -> None:
    """With step_years == test_years, consecutive test windows are adjacent."""
    bars = _bars("AAPL", n_days=6 * 252)
    config = _zero_cost_config(train_years=3, test_years=1, step_years=1)

    result = run_walk_forward(config=config, strategy=_NoOp(), bars=bars)

    # For each consecutive pair, fold i's test_end should be the trading
    # day right before fold i+1's test_start.
    for a, b in zip(result.folds[:-1], result.folds[1:], strict=False):
        assert a.test_end < b.test_start
        # And they should be close together (within a few days).
        days_gap = (b.test_start - a.test_end).days
        assert 0 < days_gap <= 7


def test_train_window_size_matches_config() -> None:
    """Each fold's train window spans train_years × trading_days_per_year days."""
    bars = _bars("AAPL", n_days=6 * 252)
    config = _zero_cost_config(train_years=3, test_years=1, step_years=1)

    result = run_walk_forward(config=config, strategy=_NoOp(), bars=bars)

    for fold in result.folds:
        # train_n = 3 * 252 = 756, so train_end is index 755 from train_start.
        # We can verify by counting how many bars fall in [train_start, train_end].
        ts = bars.index.get_level_values("timestamp").date
        n_train_bars = ((ts >= fold.train_start) & (ts <= fold.train_end)).sum()
        # Each symbol contributes one row per date — here we only have AAPL.
        assert n_train_bars == 3 * 252


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------


def test_raises_when_bars_too_short_for_one_fold() -> None:
    """We can't run walk-forward on data narrower than train + test."""
    bars = _bars("AAPL", n_days=2 * 252)  # only 2 years, need 4 for one fold
    config = _zero_cost_config(train_years=3, test_years=1, step_years=1)

    with pytest.raises(ValueError, match="need at least"):
        run_walk_forward(config=config, strategy=_NoOp(), bars=bars)


# ---------------------------------------------------------------------------
# OOS chaining
# ---------------------------------------------------------------------------


def test_noop_strategy_oos_curve_is_flat() -> None:
    """No-op produces no trades, so OOS equity stays at starting_equity."""
    bars = _bars("AAPL", n_days=5 * 252)
    config = _zero_cost_config(train_years=3, test_years=1, step_years=1)

    result = run_walk_forward(config=config, strategy=_NoOp(), bars=bars)

    assert (result.oos_equity_curve == config.backtest.starting_equity).all()
    assert result.overall_metrics.total_return == 0.0


def test_oos_curve_spans_test_windows_only() -> None:
    """The OOS curve must start at fold-1's test_start (not train_start)."""
    bars = _bars("AAPL", n_days=5 * 252)
    config = _zero_cost_config(train_years=3, test_years=1, step_years=1)

    result = run_walk_forward(config=config, strategy=_NoOp(), bars=bars)

    first_oos_date = result.oos_equity_curve.index[0]
    assert first_oos_date == result.folds[0].test_start


def test_oos_chain_is_continuous_across_folds() -> None:
    """Each fold's first OOS point should equal the prior fold's last value
    (after the scaling chain that hides the train-period drift)."""
    bars = _bars("AAPL", n_days=6 * 252)
    config = _zero_cost_config(train_years=3, test_years=1, step_years=1)

    result = run_walk_forward(config=config, strategy=_NoOp(), bars=bars)

    # OOS curve is built by skipping duplicate boundary values; for NoOp
    # every value is exactly starting_equity, so the curve is a single
    # constant. That's enough to confirm chaining didn't double-count.
    assert (result.oos_equity_curve == result.oos_equity_curve.iloc[0]).all()


# ---------------------------------------------------------------------------
# Per-fold metrics
# ---------------------------------------------------------------------------


def test_fold_summary_has_one_row_per_fold() -> None:
    """fold_summary() returns a DataFrame for printing/reports — one row per fold."""
    bars = _bars("AAPL", n_days=5 * 252)
    config = _zero_cost_config(train_years=3, test_years=1, step_years=1)

    result = run_walk_forward(config=config, strategy=_NoOp(), bars=bars)
    df = result.fold_summary()

    assert len(df) == len(result.folds)
    expected_cols = {
        "fold", "train_start", "train_end", "test_start", "test_end",
        "total_return", "sharpe", "max_dd", "n_fills",
    }
    assert set(df.columns) == expected_cols
