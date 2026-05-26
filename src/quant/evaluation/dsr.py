"""dsr.py — Probabilistic and Deflated Sharpe Ratios.

PSR (Probabilistic Sharpe Ratio): probability the strategy's true Sharpe
exceeds a benchmark, accounting for finite-sample skew and kurtosis.

DSR (Deflated Sharpe Ratio): PSR with the benchmark set to the expected
*maximum* Sharpe under the null of zero skill across N trials — corrects
for selection bias from multiple testing.

References: Bailey & López de Prado (2014). See ``docs/specs/dsr.md`` for
the full design, math, and how this plugs into the registry's promotion
gates.

API
---
- ``probabilistic_sharpe_ratio(returns, ...)`` — pure function on returns.
- ``deflated_sharpe_ratio(returns, n_trials, var_sr_trials_annual, ...)``.
- ``psr_for(result, ...)`` / ``dsr_for(result, ...)`` — convenience around
  a ``BacktestResult`` (pulls trading_days_per_year from its config).
"""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import kurtosis, norm, skew

from quant.backtest.engine import BacktestResult


# Euler-Mascheroni constant — used in the expected-max-of-N asymptotic.
# Defined here once with full precision to avoid drift.
_EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    *,
    benchmark_sharpe_annual: float = 0.0,
    trading_days_per_year: int = 252,
) -> float:
    """Probability that the strategy's true Sharpe exceeds the benchmark.

    The output is a value in [0, 1]:
      - 0.5 = observed Sharpe equals benchmark (50/50).
      - > 0.95 = ~95% confident the true Sharpe is above benchmark.
      - < 0.5 = observed Sharpe BELOW benchmark; the strategy looks worse
        than the benchmark even after smoothing for noise.

    Parameters
    ----------
    returns
        Per-period (daily) return series — NOT cumulative equity.
        Get this from ``equity_curve.pct_change().dropna()``.
    benchmark_sharpe_annual
        Annualized Sharpe to test against. The function de-annualizes
        internally. Default 0 = "is the strategy better than zero skill?".
    trading_days_per_year
        Annualization factor. 252 is the US equity convention.

    Returns
    -------
    PSR in [0, 1].

    Raises
    ------
    ValueError
        If returns has fewer than 4 observations (skew/kurtosis are
        meaningless on a tiny sample).

    Notes
    -----
    Uses the Bailey-López de Prado formula (2014):

        PSR(SR*) = Φ( (SR_hat - SR*) * sqrt(N-1)
                       / sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2) )

    where SR_hat is per-period, g3 is sample skewness, g4 is sample
    kurtosis (Pearson, not excess), and Φ is the standard-normal CDF.
    """
    n = len(returns)
    if n < 4:
        # Need at least a few points for skew/kurtosis to be meaningful.
        # scipy itself doesn't error on n=3 but the result is uninformative.
        raise ValueError(f"need at least 4 returns; got {n}")

    mean_r = float(returns.mean())
    std_r = float(returns.std(ddof=1))

    # Use a small-epsilon check, not == 0. Pandas's std on a constant
    # series (e.g., [0.005] * N) can return a tiny non-zero value due to
    # catastrophic cancellation inside Welford's algorithm. Without the
    # epsilon, we'd fall through and hand scipy a near-constant series,
    # which then NaN-poisons skew/kurtosis. Real returns have std on the
    # order of 0.01; 1e-15 is far below anything meaningful.
    if std_r < 1e-15:
        # Zero variance → no signal, no noise, no inference possible.
        # Returning 0.5 (the "no information" outcome) is the sensible
        # default rather than NaN, which would propagate confusingly.
        return 0.5

    sr_period = mean_r / std_r

    # De-annualize the benchmark to the same per-period units as SR_hat.
    sr_benchmark_period = benchmark_sharpe_annual / math.sqrt(trading_days_per_year)

    # Sample skew and kurtosis. `bias=False` applies the small-sample
    # correction; `fisher=False` returns Pearson kurtosis (3 for normal,
    # which is what Bailey's formula expects), not excess (0 for normal).
    g3 = float(skew(returns, bias=False))
    g4 = float(kurtosis(returns, fisher=False, bias=False))

    # Denominator of the PSR formula — the higher-moment correction to
    # the standard t-test on a Sharpe ratio.
    denom_var = 1.0 - g3 * sr_period + (g4 - 1.0) / 4.0 * sr_period ** 2
    if denom_var <= 0:
        # Catastrophic skew/kurtosis combo — formula is undefined here.
        # NaN signals to the caller that the PSR can't be computed for
        # this distribution; better than silently returning 0 or 1.
        return float("nan")

    numerator = (sr_period - sr_benchmark_period) * math.sqrt(n - 1)
    return float(norm.cdf(numerator / math.sqrt(denom_var)))


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    n_trials: int,
    var_sr_trials_annual: float,
    trading_days_per_year: int = 252,
) -> float:
    """PSR with the benchmark set to the expected-max-of-N Sharpe under null.

    This is the actual multiple-testing correction. With ``n_trials=1``,
    DSR == PSR(benchmark=0). With ``n_trials=100``, the benchmark rises
    to roughly the 99th-percentile Sharpe of the null distribution — a
    Sharpe of 1.5 looks much less impressive against that.

    Parameters
    ----------
    returns
        Per-period return series. As with PSR.
    n_trials
        How many variants you tested to arrive at this one. Honest
        accounting matters — the registry (Step 20) will track this
        automatically; for now, count carefully by hand.
    var_sr_trials_annual
        Sample variance of ANNUALIZED Sharpe ratios across your trial
        population. Use ``statistics.variance(trial_sharpes)`` or
        ``estimate_var_sr_from_trials``.
    trading_days_per_year
        Annualization factor.

    Returns
    -------
    DSR in [0, 1]. Same interpretation as PSR.

    Notes
    -----
    Bailey-López de Prado expected-max formula:

        SR* = sqrt(V[SR]) * ((1-γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N·e)))

    where γ is the Euler-Mascheroni constant (~0.5772) and the two
    Φ^{-1} terms encode the expected-max of N standard normals.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1; got {n_trials}")
    if var_sr_trials_annual < 0:
        raise ValueError(
            f"var_sr_trials_annual must be non-negative; got {var_sr_trials_annual}"
        )

    if n_trials == 1:
        # No selection bias when only one strategy was ever tried.
        # DSR collapses to PSR(SR* = 0).
        return probabilistic_sharpe_ratio(
            returns,
            benchmark_sharpe_annual=0.0,
            trading_days_per_year=trading_days_per_year,
        )

    # Expected max Sharpe under the null. The threshold grows roughly as
    # sqrt(2 log N) — slowly at first (n=2 → ~0.56), faster at large N.
    norm_inv_1 = norm.ppf(1 - 1 / n_trials)
    norm_inv_2 = norm.ppf(1 - 1 / (n_trials * math.e))
    sd_sr_annual = math.sqrt(var_sr_trials_annual)
    sr_threshold_annual = sd_sr_annual * (
        (1 - _EULER_MASCHERONI) * norm_inv_1 + _EULER_MASCHERONI * norm_inv_2
    )

    return probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe_annual=sr_threshold_annual,
        trading_days_per_year=trading_days_per_year,
    )


def estimate_var_sr_from_trials(sharpe_ratios_annual: list[float]) -> float:
    """Sample variance of annualized Sharpe ratios across a trial population.

    Use this when you've run N variants and gathered each one's Sharpe;
    the result is what to pass as ``var_sr_trials_annual`` to ``deflated_sharpe_ratio``.

    Raises ValueError if fewer than 2 trials — variance requires at least 2.
    """
    if len(sharpe_ratios_annual) < 2:
        raise ValueError(
            f"need at least 2 trial sharpes to estimate variance; got {len(sharpe_ratios_annual)}"
        )
    # ddof=1 → sample variance (the unbiased estimator). Same convention
    # used elsewhere in metrics.py.
    series = pd.Series(sharpe_ratios_annual)
    return float(series.var(ddof=1))


# ---------------------------------------------------------------------------
# BacktestResult convenience wrappers.
# ---------------------------------------------------------------------------


def psr_for(
    result: BacktestResult,
    *,
    benchmark_sharpe_annual: float = 0.0,
) -> float:
    """PSR computed from a BacktestResult, using its config's trading-days convention."""
    returns = result.equity_curve.pct_change().dropna()
    return probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe_annual=benchmark_sharpe_annual,
        trading_days_per_year=result.config.evaluation.trading_days_per_year,
    )


def dsr_for(
    result: BacktestResult,
    *,
    n_trials: int,
    var_sr_trials_annual: float,
) -> float:
    """DSR computed from a BacktestResult, using its config's trading-days convention."""
    returns = result.equity_curve.pct_change().dropna()
    return deflated_sharpe_ratio(
        returns,
        n_trials=n_trials,
        var_sr_trials_annual=var_sr_trials_annual,
        trading_days_per_year=result.config.evaluation.trading_days_per_year,
    )
