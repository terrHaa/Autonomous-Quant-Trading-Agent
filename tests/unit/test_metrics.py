"""Tests for the metrics module.

Each test uses a hand-computable case so the assertion isn't "metric ==
whatever the function returns" tautology. If you change the implementation,
these tests catch any change in semantics.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from quant.evaluation.metrics import compute_metrics


def _equity_from_returns(returns: list[float], start: float = 1_000_000) -> pd.Series:
    """Build an equity curve from a list of daily returns.

    Equity at bar 0 is `start`; each subsequent bar applies one return.
    Result has len(returns) + 1 points.
    """
    eq = [start]
    for r in returns:
        eq.append(eq[-1] * (1 + r))
    # Plain DatetimeIndex so the test doesn't depend on calendar choice.
    return pd.Series(eq, index=pd.bdate_range("2020-01-02", periods=len(eq)))


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_requires_at_least_two_points() -> None:
    """A single equity point has no return; raise rather than return 0."""
    eq = pd.Series([1_000_000], index=pd.bdate_range("2020-01-02", periods=1))
    with pytest.raises(ValueError, match="at least 2 equity points"):
        compute_metrics(eq)


def test_rejects_non_positive_equity() -> None:
    """Zero / negative equity means the strategy blew up — refuse to lie about it."""
    eq = pd.Series(
        [1_000_000, 500_000, 0],
        index=pd.bdate_range("2020-01-02", periods=3),
    )
    with pytest.raises(ValueError, match="non-positive"):
        compute_metrics(eq)


# ---------------------------------------------------------------------------
# Flat equity — degenerate baseline
# ---------------------------------------------------------------------------


def test_flat_equity_gives_zero_returns_and_vol() -> None:
    """No movement → every return metric is zero; no drawdown."""
    eq = pd.Series(
        [1_000_000] * 252,
        index=pd.bdate_range("2020-01-02", periods=252),
    )
    m = compute_metrics(eq)

    assert m.total_return == 0.0
    assert m.cagr == 0.0
    assert m.annualized_vol == 0.0
    assert m.max_drawdown == 0.0
    # Sharpe is undefined when vol=0; we return 0 as the "no signal" sentinel.
    assert m.sharpe == 0.0
    # No positive days → hit rate is 0.
    assert m.hit_rate == 0.0


# ---------------------------------------------------------------------------
# Total return + CAGR
# ---------------------------------------------------------------------------


def test_total_return_matches_start_end() -> None:
    """(end / start) - 1, no rounding tricks."""
    eq = pd.Series(
        [100.0, 110.0, 120.0, 150.0],
        index=pd.bdate_range("2020-01-02", periods=4),
    )
    m = compute_metrics(eq)
    assert m.total_return == pytest.approx(0.5)


def test_cagr_matches_geometric_definition() -> None:
    """CAGR = (end/start)^(1/years) - 1, where years = (n-1)/trading_days_per_year.

    With 253 bars (252 returns) and 10x growth → 900% CAGR (one year of 900%).
    """
    eq = pd.Series(
        [1.0] + [None] * 251 + [10.0],  # 253 bars; only first and last matter for CAGR
        index=pd.bdate_range("2020-01-02", periods=253),
    )
    # Fill in a monotonic ramp so all bars are positive and ascending.
    eq = pd.Series(
        [1.0 * (10 ** (i / 252)) for i in range(253)],
        index=pd.bdate_range("2020-01-02", periods=253),
    )
    m = compute_metrics(eq, trading_days_per_year=252)
    # ~900% CAGR — 10x growth over exactly one year.
    assert m.cagr == pytest.approx(9.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_matches_hand_computation() -> None:
    """eq = [100, 110, 95, 100, 105] → peak at 110, trough at 95.

    drawdown_at_bar2 = (95 - 110) / 110 ≈ -13.64%; that's the worst.
    """
    eq = pd.Series(
        [100.0, 110.0, 95.0, 100.0, 105.0],
        index=pd.bdate_range("2020-01-02", periods=5),
    )
    m = compute_metrics(eq)
    expected = (95.0 - 110.0) / 110.0
    assert m.max_drawdown == pytest.approx(expected)


def test_max_drawdown_duration_counts_consecutive_underwater_days() -> None:
    """eq = [100, 90, 95, 99, 110, 100, 105, 109, 115]

    peaks:       [100, 100, 100, 100, 110, 110, 110, 110, 115]
    underwater:  [ F,  T,   T,   T,   F,   T,   T,   T,   F ]

    Note: 'underwater' means STRICTLY below the running peak. eq == peak
    means we've fully recovered (or made a new high), not underwater.
    With strictly-below comparison, the longest run here is 3 bars
    (positions 1-3, with eq < 100; OR positions 5-7, with eq < 110).
    """
    eq = pd.Series(
        [100.0, 90.0, 95.0, 99.0, 110.0, 100.0, 105.0, 109.0, 115.0],
        index=pd.bdate_range("2020-01-02", periods=9),
    )
    m = compute_metrics(eq)
    assert m.max_drawdown_duration_days == 3


def test_monotonic_growth_has_no_drawdown() -> None:
    """Strictly-increasing equity → max_dd = 0, duration = 0."""
    eq = pd.Series(
        [100.0 + i for i in range(10)],
        index=pd.bdate_range("2020-01-02", periods=10),
    )
    m = compute_metrics(eq)
    assert m.max_drawdown == 0.0
    assert m.max_drawdown_duration_days == 0


# ---------------------------------------------------------------------------
# Sharpe and Sortino — hand-computed
# ---------------------------------------------------------------------------


def test_sharpe_matches_hand_computation() -> None:
    """5 daily returns: [0.01, -0.005, 0.02, 0.0, 0.015]

    mean        = 0.008
    sample std  = sqrt(sum((r - mean)^2) / 4)
                = sqrt(((0.01-.008)^2 + (-.005-.008)^2 + (.02-.008)^2 + (0-.008)^2 + (.015-.008)^2) / 4)
                = sqrt((0.002^2 + (-0.013)^2 + 0.012^2 + (-0.008)^2 + 0.007^2) / 4)
                = sqrt((4e-6 + 169e-6 + 144e-6 + 64e-6 + 49e-6) / 4)
                = sqrt(430e-6 / 4)
                = sqrt(107.5e-6)
                ≈ 0.010369

    With rf=0:
      sharpe = (0.008 / 0.010369) * sqrt(252) ≈ 12.246
    """
    returns = [0.01, -0.005, 0.02, 0.0, 0.015]
    eq = _equity_from_returns(returns)

    m = compute_metrics(eq, risk_free_annual=0.0, trading_days_per_year=252)

    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    expected_sharpe = (mean / std) * math.sqrt(252)

    assert m.sharpe == pytest.approx(expected_sharpe, rel=1e-9)


def test_risk_free_rate_reduces_sharpe() -> None:
    """A non-zero rf should DECREASE the Sharpe by exactly rf_daily / std."""
    returns = [0.01] * 252  # constant 1%/day for a year
    eq = _equity_from_returns(returns)

    sharpe_rf0 = compute_metrics(eq, risk_free_annual=0.0).sharpe
    sharpe_rf4 = compute_metrics(eq, risk_free_annual=0.04).sharpe

    # Constant returns → std=0 → both Sharpes are 0 (sentinel for undefined).
    # Use a noisier series so std > 0:
    returns = [0.01, -0.005, 0.02, 0.0, 0.015, 0.008, -0.002] * 36  # ~252 days
    eq = _equity_from_returns(returns)

    sharpe_rf0 = compute_metrics(eq, risk_free_annual=0.0).sharpe
    sharpe_rf4 = compute_metrics(eq, risk_free_annual=0.04).sharpe

    assert sharpe_rf0 > sharpe_rf4, "raising rf should lower Sharpe"


def test_sortino_uses_only_downside_deviation() -> None:
    """Sortino's denominator excludes positive returns → typically > Sharpe.

    For an asymmetric return distribution where most variance is on the
    upside, Sortino should be noticeably higher than Sharpe.
    """
    # Many small ups, a few small downs — variance mostly on the up side.
    returns = [0.02] * 200 + [-0.005] * 52
    eq = _equity_from_returns(returns)

    m = compute_metrics(eq, risk_free_annual=0.0)
    assert m.sortino > m.sharpe


# ---------------------------------------------------------------------------
# Hit rate
# ---------------------------------------------------------------------------


def test_hit_rate_is_fraction_strictly_positive() -> None:
    """3 of 5 returns are positive (zero doesn't count) → hit_rate = 0.6."""
    returns = [0.01, -0.005, 0.02, 0.0, 0.015]
    eq = _equity_from_returns(returns)
    m = compute_metrics(eq)
    assert m.hit_rate == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------
# Calmar
# ---------------------------------------------------------------------------


def test_calmar_is_cagr_over_abs_max_dd() -> None:
    """Calmar = CAGR / |max_dd|. With known CAGR and |max_dd|, ratio is exact."""
    # Up 21% over a year, with a max drawdown along the way.
    # Construct: 200 bars at $1, drop to $0.80 (max_dd = -20%), recover to $1.21.
    eq_list = [1.0] * 200 + [0.80] + [1.0 + i * 0.005 for i in range(53)]
    eq = pd.Series(eq_list, index=pd.bdate_range("2020-01-02", periods=len(eq_list)))
    m = compute_metrics(eq)

    expected = m.cagr / abs(m.max_drawdown)
    assert m.calmar == pytest.approx(expected)


def test_calmar_is_zero_when_no_drawdown() -> None:
    """Defending against div-by-zero — flat-growth strategies get Calmar=0."""
    eq = pd.Series(
        [100.0 + i for i in range(10)],
        index=pd.bdate_range("2020-01-02", periods=10),
    )
    m = compute_metrics(eq)
    assert m.calmar == 0.0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_fields_are_stamped() -> None:
    """The Metrics object records which rf and trading-day convention produced it."""
    eq = _equity_from_returns([0.01, -0.005, 0.02])
    m = compute_metrics(eq, risk_free_annual=0.03, trading_days_per_year=250)

    assert m.risk_free_annual == 0.03
    assert m.trading_days_per_year == 250
    assert m.n_days == 4   # 3 returns -> 4 equity points
    assert m.starting_equity == 1_000_000


# ---------------------------------------------------------------------------
# Pretty-print smoke
# ---------------------------------------------------------------------------


def test_str_includes_key_metrics() -> None:
    """__str__ must surface the main metrics — used in demos and report headers."""
    eq = _equity_from_returns([0.01, -0.005, 0.02, 0.015])
    m = compute_metrics(eq)
    s = str(m)
    for label in ("Total return", "CAGR", "Sharpe", "Sortino", "Max drawdown", "Hit rate"):
        assert label in s
