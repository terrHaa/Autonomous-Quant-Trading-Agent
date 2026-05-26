"""Tests for the Probabilistic and Deflated Sharpe Ratio implementations.

The tests target *behavior* rather than exact numerical equivalence with a
reference implementation, because subtle differences in skew/kurtosis
formulas across libraries (scipy vs statsmodels vs hand) give different
last-decimal values. Behavioral tests are robust to those differences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.evaluation.dsr import (
    _EULER_MASCHERONI,
    deflated_sharpe_ratio,
    estimate_var_sr_from_trials,
    probabilistic_sharpe_ratio,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_returns(
    n: int,
    *,
    mean: float = 0.0,
    std: float = 0.01,
    seed: int = 42,
) -> pd.Series:
    """Deterministic Gaussian return series for tests.

    Fixed seed → identical results across runs → no flakiness in CI.
    """
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


# ---------------------------------------------------------------------------
# PSR — happy paths and edge cases
# ---------------------------------------------------------------------------


def test_psr_at_zero_sharpe_equals_half() -> None:
    """Observed mean of zero → Sharpe of zero → PSR(0) is exactly 0.5.

    This is the "no information" anchor: a strategy with no edge gives 50/50
    when tested against a zero-Sharpe benchmark.
    """
    # Symmetric returns with mean exactly 0.
    returns = pd.Series([0.01, -0.01] * 100)
    psr = probabilistic_sharpe_ratio(returns, benchmark_sharpe_annual=0.0)
    assert psr == pytest.approx(0.5, abs=1e-9)


def test_psr_above_half_when_observed_exceeds_benchmark() -> None:
    """Positive mean returns → PSR(0) > 0.5."""
    returns = _gaussian_returns(252, mean=0.001, std=0.01)  # SR ≈ 1.6 annualized
    psr = probabilistic_sharpe_ratio(returns, benchmark_sharpe_annual=0.0)
    assert psr > 0.5


def test_psr_below_half_when_observed_below_benchmark() -> None:
    """Same returns vs a much higher benchmark → PSR < 0.5.

    The strategy "looks worse" than the benchmark with high confidence.
    """
    returns = _gaussian_returns(252, mean=0.001, std=0.01)  # SR ≈ 1.6
    psr = probabilistic_sharpe_ratio(returns, benchmark_sharpe_annual=3.0)
    assert psr < 0.5


def test_psr_monotonic_in_sample_size() -> None:
    """Same Sharpe, more bars → higher confidence (PSR increases).

    More data = tighter confidence bounds. The PSR formula encodes this
    via the sqrt(N-1) numerator term.
    """
    short = _gaussian_returns(60, mean=0.001, std=0.01, seed=7)
    long = _gaussian_returns(2520, mean=0.001, std=0.01, seed=7)

    psr_short = probabilistic_sharpe_ratio(short)
    psr_long = probabilistic_sharpe_ratio(long)

    assert psr_long > psr_short


def test_psr_rejects_too_few_returns() -> None:
    """A handful of points can't support skew/kurtosis estimation."""
    with pytest.raises(ValueError, match="at least 4"):
        probabilistic_sharpe_ratio(pd.Series([0.01, -0.01, 0.005]))


def test_psr_zero_variance_returns_half() -> None:
    """Constant returns → no signal, no noise → 50/50 by convention."""
    returns = pd.Series([0.005] * 100)
    psr = probabilistic_sharpe_ratio(returns)
    assert psr == 0.5


# ---------------------------------------------------------------------------
# DSR — deflation behavior
# ---------------------------------------------------------------------------


def test_dsr_with_one_trial_equals_psr_zero() -> None:
    """With n_trials=1, DSR collapses to PSR(SR* = 0).

    No selection bias when there's nothing to select among. The var_sr_trials
    argument is ignored in this branch.
    """
    returns = _gaussian_returns(252, mean=0.001, std=0.01)
    psr_val = probabilistic_sharpe_ratio(returns, benchmark_sharpe_annual=0.0)
    dsr_val = deflated_sharpe_ratio(
        returns, n_trials=1, var_sr_trials_annual=0.5,  # ignored
    )
    assert dsr_val == pytest.approx(psr_val, abs=1e-9)


def test_dsr_decreases_as_n_trials_increases() -> None:
    """The core promise: same observed Sharpe, more trials → lower DSR.

    With 1 trial, a 1.6-Sharpe is impressive. With 1000 trials, the same
    1.6-Sharpe is roughly what null + selection alone would produce.
    """
    returns = _gaussian_returns(2520, mean=0.001, std=0.01)
    # A plausible V[SR] across trials.
    var_sr = 0.5  # std(annual Sharpes) ≈ 0.71 across trials

    dsr_1 = deflated_sharpe_ratio(returns, n_trials=1, var_sr_trials_annual=var_sr)
    dsr_10 = deflated_sharpe_ratio(returns, n_trials=10, var_sr_trials_annual=var_sr)
    dsr_100 = deflated_sharpe_ratio(returns, n_trials=100, var_sr_trials_annual=var_sr)
    dsr_10000 = deflated_sharpe_ratio(returns, n_trials=10_000, var_sr_trials_annual=var_sr)

    assert dsr_1 > dsr_10 > dsr_100 > dsr_10000, (
        f"DSR should monotonically decrease with n_trials, got "
        f"{dsr_1:.3f} → {dsr_10:.3f} → {dsr_100:.3f} → {dsr_10000:.3f}"
    )


def test_dsr_decreases_as_var_sr_increases() -> None:
    """Higher variance across trials → wider null distribution → harder to beat.

    If your N trials all gave wildly different Sharpes, the expected max
    is further out; the deflation is more aggressive.
    """
    returns = _gaussian_returns(2520, mean=0.001, std=0.01)

    dsr_narrow = deflated_sharpe_ratio(
        returns, n_trials=100, var_sr_trials_annual=0.01,  # tight trial cluster
    )
    dsr_wide = deflated_sharpe_ratio(
        returns, n_trials=100, var_sr_trials_annual=1.0,   # wild trial cluster
    )
    assert dsr_narrow > dsr_wide


def test_dsr_rejects_invalid_arguments() -> None:
    """Bad inputs raise early — better than silently returning garbage."""
    returns = _gaussian_returns(252, mean=0.001, std=0.01)

    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(returns, n_trials=0, var_sr_trials_annual=0.5)
    with pytest.raises(ValueError, match="var_sr"):
        deflated_sharpe_ratio(returns, n_trials=10, var_sr_trials_annual=-0.1)


def test_dsr_with_large_n_trials_pushes_to_zero_for_marginal_sharpe() -> None:
    """A barely-positive Sharpe under a million trials shouldn't survive deflation.

    Concrete: a strategy with SR ≈ 0.5 looks fine vs zero (PSR ≈ 0.85),
    but vs the expected best-of-million it's nothing.
    """
    # Build a return series with annualized Sharpe ≈ 0.5.
    # mean/std * sqrt(252) = 0.5  →  mean/std = 0.5/sqrt(252) ≈ 0.0315
    # With std = 0.01, mean ≈ 0.000315.
    returns = _gaussian_returns(2520, mean=0.000315, std=0.01, seed=1)
    dsr_one_million = deflated_sharpe_ratio(
        returns, n_trials=1_000_000, var_sr_trials_annual=0.5,
    )
    # Should be far below 0.95 — the strategy doesn't survive massive selection.
    assert dsr_one_million < 0.5


# ---------------------------------------------------------------------------
# estimate_var_sr_from_trials
# ---------------------------------------------------------------------------


def test_estimate_var_sr_matches_sample_variance() -> None:
    """Just a thin wrapper around pandas .var(ddof=1) — sanity check."""
    sharpes = [0.5, 1.0, 1.5, 2.0, 1.2]
    expected = pd.Series(sharpes).var(ddof=1)
    assert estimate_var_sr_from_trials(sharpes) == pytest.approx(expected)


def test_estimate_var_sr_requires_two_trials() -> None:
    """One trial has no variance to estimate."""
    with pytest.raises(ValueError, match="at least 2"):
        estimate_var_sr_from_trials([1.5])


# ---------------------------------------------------------------------------
# Sanity: the formula's constants are what we documented
# ---------------------------------------------------------------------------


def test_euler_mascheroni_constant_is_correct_to_machine_precision() -> None:
    """We use this in the SR* expected-max-of-N formula; drift here
    silently shifts every DSR.
    """
    # Known value to 16 digits.
    assert _EULER_MASCHERONI == pytest.approx(0.5772156649015329, abs=1e-16)
