"""Tests for the tearsheet renderer.

The tearsheet is presentation code — the test surface is "does the output
contain the right things and reflect the inputs?" rather than exact string
equality, which would break on harmless whitespace/format tweaks.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestResult
from quant.config import DEFAULT_CONFIG_PATH, Config
from quant.reports import render_tearsheet, write_tearsheet


def _load_config() -> Config:
    import yaml
    return Config.model_validate(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))


def _synthetic_result(
    *,
    strategy_name: str = "test_strategy",
    n_days: int = 252,
    n_fills: int = 5,
    seed: int = 42,
) -> BacktestResult:
    """Build a minimal BacktestResult with predictable contents."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, n_days)
    equity = 1_000_000 * np.cumprod(1 + returns)
    dates = list(pd.bdate_range("2020-01-02", periods=n_days).date)

    # Synthesize a small fills frame.
    fill_rows = []
    for i in range(n_fills):
        fill_rows.append({
            "date": dates[i * (n_days // (n_fills + 1))],
            "symbol": f"SYM{i % 3}",
            "side": "buy" if i % 2 == 0 else "sell",
            "qty": 100 * (i + 1),
            "fill_price": 50.0 + i,
            "notional": (50.0 + i) * 100 * (i + 1),
            "spread_cost": 0.5 * (i + 1),
            "slippage_cost": 1.5 * (i + 1),
            "commission": 0.0,
        })
    fills_df = pd.DataFrame(fill_rows)

    # Build a tiny per-day costs frame.
    costs_df = pd.DataFrame({
        "spread_cost":    [0.5 * (i + 1) for i in range(n_fills)],
        "slippage_cost":  [1.5 * (i + 1) for i in range(n_fills)],
        "commission":     [0.0] * n_fills,
        "total":          [2.0 * (i + 1) for i in range(n_fills)],
    })

    return BacktestResult(
        config=_load_config(),
        strategy_name=strategy_name,
        equity_curve=pd.Series(equity, index=dates),
        positions=pd.DataFrame(),
        weights=pd.DataFrame(),
        orders=pd.DataFrame(),
        fills=fills_df,
        costs=costs_df,
        metadata={
            "n_bars": n_days,
            "n_orders": n_fills,
            "n_fills": n_fills,
            "run_time_s": 0.123,
            "start_date": dates[0],
            "end_date": dates[-1],
            "starting_equity": 1_000_000.0,
            "ending_equity": float(equity[-1]),
        },
    )


# ---------------------------------------------------------------------------
# Structural / content checks
# ---------------------------------------------------------------------------


def test_render_returns_a_string() -> None:
    """API basics: returns a str, isn't empty, ends with a newline."""
    md = render_tearsheet(_synthetic_result())
    assert isinstance(md, str)
    assert len(md) > 100
    assert md.endswith("\n")


def test_tearsheet_includes_strategy_name() -> None:
    """The header must surface the strategy's identity."""
    md = render_tearsheet(_synthetic_result(strategy_name="my_great_idea"))
    assert "my_great_idea" in md


def test_tearsheet_title_overrides_strategy_name() -> None:
    """The optional `title` arg replaces the H1 heading."""
    md = render_tearsheet(
        _synthetic_result(strategy_name="raw_name"),
        title="Pretty Title for Reports",
    )
    assert "# Pretty Title for Reports" in md
    # The strategy name still appears (in the metadata line below the title).
    assert "raw_name" in md


def test_tearsheet_has_required_sections() -> None:
    """Every section header must be present and in the right order."""
    md = render_tearsheet(_synthetic_result())
    headings = [
        "## Headline metrics",
        "## Trading activity",
        "## Costs paid",
        "## Top",  # "Top N fills by notional" — N varies
    ]
    last_index = -1
    for h in headings:
        idx = md.find(h)
        assert idx > last_index, f"section {h!r} missing or out of order"
        last_index = idx


def test_tearsheet_metrics_table_has_each_metric() -> None:
    md = render_tearsheet(_synthetic_result())
    for label in (
        "Total return", "CAGR", "Annualized vol", "Sharpe", "Sortino",
        "Calmar", "Max drawdown", "Max DD days", "Hit rate",
    ):
        assert label in md, f"metric {label!r} missing"


def test_tearsheet_omits_top_fills_section_when_no_fills() -> None:
    """An empty fills frame shouldn't render a 'Top 0 fills' section."""
    result = _synthetic_result(n_fills=0)
    md = render_tearsheet(result)
    assert "Top" not in md.split("## Costs")[1]


def test_top_n_fills_parameter_caps_table_rows() -> None:
    """top_n_fills=3 → only 3 data rows (plus header + separator)."""
    md = render_tearsheet(_synthetic_result(n_fills=10), top_n_fills=3)
    # Find the top-fills table and count its pipe-delimited rows.
    section = md.split("## Top")[1]
    # Each data row starts with '| <date>'; the header row starts with '| Date'.
    data_rows = [line for line in section.splitlines() if line.startswith("| 20")]
    assert len(data_rows) == 3


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------


def test_write_tearsheet_creates_file(tmp_path: Path) -> None:
    """write_tearsheet should round-trip: file contents == render output."""
    result = _synthetic_result()
    out = write_tearsheet(result, tmp_path / "reports" / "test.md")
    assert out.exists()
    assert out == tmp_path / "reports" / "test.md"
    # Content matches render output exactly.
    assert out.read_text() == render_tearsheet(result)


def test_write_tearsheet_creates_parent_directory(tmp_path: Path) -> None:
    """Nested paths should auto-create their parents."""
    target = tmp_path / "deep" / "nested" / "dir" / "tear.md"
    write_tearsheet(_synthetic_result(), target)
    assert target.exists()
