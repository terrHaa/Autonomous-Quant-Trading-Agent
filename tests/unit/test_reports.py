"""Tests for the markdown renderers — focused on the new benchmark section.

The renderers were untested before (only end-to-end coverage via review
test files). The benchmark feature touches all three so we add direct
tests for the new behaviour.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from quant.agent.reports import (
    render_daily_report,
    render_monthly_report,
    render_weekly_report,
)
from quant.execution.alpaca_executor import ExecutionReport


def _empty_report() -> ExecutionReport:
    return ExecutionReport(
        env="paper",
        timestamp=datetime.now(tz=UTC),
        account_equity_before=100_000.0,
        positions_before={},
        target_weights={},
        proposed_orders=[],
        submitted_orders=[],
        dry_run=False,
        notes="",
    )


# ---------------------------------------------------------------------------
# Daily — benchmark section
# ---------------------------------------------------------------------------


def test_daily_renders_benchmark_row_when_data_provided() -> None:
    """SPY +0.5%, QQQ +1.2% should appear in the email body."""
    _, body = render_daily_report(
        run_date=date(2024, 6, 3),
        strategy_name="ensemble(3)",
        target_weights={},
        execution_report=_empty_report(),
        account_equity_after=101_000.0,    # +1% session return
        benchmarks={"SPY": 0.005, "QQQ": 0.012},
    )
    assert "Today vs benchmarks" in body
    assert "Portfolio" in body
    assert "+1.00%" in body
    assert "+0.50%" in body
    assert "+1.20%" in body
    # Out-performance vs SPY (+1.00% - +0.50% = +0.50%) should show ↑.
    assert "↑" in body


def test_daily_omits_benchmark_section_when_nothing_provided() -> None:
    """No benchmarks AND no post-close equity → section is fully omitted."""
    _, body = render_daily_report(
        run_date=date(2024, 6, 3),
        strategy_name="ensemble(3)",
        target_weights={},
        execution_report=_empty_report(),
    )
    assert "Today vs benchmarks" not in body


def test_daily_renders_benchmarks_only_when_post_close_equity_missing() -> None:
    """We can show market context without computing a portfolio return."""
    _, body = render_daily_report(
        run_date=date(2024, 6, 3),
        strategy_name="ensemble(3)",
        target_weights={},
        execution_report=_empty_report(),
        account_equity_after=None,
        benchmarks={"SPY": 0.005, "QQQ": 0.012},
    )
    assert "Today vs benchmarks" in body
    assert "+0.50%" in body
    assert "Portfolio return not available" in body


def test_daily_handles_only_one_benchmark() -> None:
    """If the cache only priced SPY, QQQ row is skipped — no crash."""
    _, body = render_daily_report(
        run_date=date(2024, 6, 3),
        strategy_name="ensemble(3)",
        target_weights={},
        execution_report=_empty_report(),
        account_equity_after=101_000.0,
        benchmarks={"SPY": 0.005},
    )
    assert "+0.50%" in body
    assert "Nasdaq" not in body   # QQQ wasn't priced


# ---------------------------------------------------------------------------
# Weekly — benchmark section
# ---------------------------------------------------------------------------


def test_weekly_renders_benchmarks_with_portfolio_compare() -> None:
    equity = {
        date(2024, 6, 3): 100_000.0,
        date(2024, 6, 7): 102_000.0,   # +2% week
    }
    _, body = render_weekly_report(
        week_ending=date(2024, 6, 7),
        daily_runs=[],
        equity_curve=equity,
        benchmarks={"SPY": 0.01, "QQQ": 0.03},
    )
    assert "Vs benchmarks (same window)" in body
    assert "+2.00%" in body   # portfolio
    assert "+1.00%" in body   # SPY
    assert "+3.00%" in body   # QQQ
    # Portfolio out-performs SPY (+1pp) but underperforms QQQ (-1pp).
    # Both deltas should appear with directional markers.
    assert "↑" in body and "↓" in body


def test_weekly_omits_section_when_no_data() -> None:
    _, body = render_weekly_report(
        week_ending=date(2024, 6, 7),
        daily_runs=[],
        equity_curve=None,
        benchmarks=None,
    )
    assert "Vs benchmarks" not in body


def test_weekly_renders_benchmarks_alone_without_equity() -> None:
    """Even with no equity curve, benchmarks alone are useful context."""
    _, body = render_weekly_report(
        week_ending=date(2024, 6, 7),
        daily_runs=[],
        equity_curve=None,
        benchmarks={"SPY": 0.005, "QQQ": 0.012},
    )
    assert "Vs benchmarks" in body
    assert "+0.50%" in body
    assert "+1.20%" in body
    # No portfolio row.
    assert "Portfolio" not in body


# ---------------------------------------------------------------------------
# Monthly — inherits benchmark plumbing from weekly via render_weekly_report
# ---------------------------------------------------------------------------


def test_monthly_renders_benchmarks_via_weekly_renderer() -> None:
    equity = {
        date(2024, 5, 1): 100_000.0,
        date(2024, 5, 31): 105_000.0,   # +5% month
    }
    _, body = render_monthly_report(
        month_ending=date(2024, 5, 31),
        daily_runs=[],
        equity_curve=equity,
        recommendations=["test rec"],
        benchmarks={"SPY": 0.03, "QQQ": 0.07},
    )
    assert "Vs benchmarks" in body
    assert "+5.00%" in body
    assert "+3.00%" in body
    assert "+7.00%" in body
    # Recommendations section still works.
    assert "test rec" in body
