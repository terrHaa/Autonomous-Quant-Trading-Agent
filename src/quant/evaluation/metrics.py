"""metrics.py — standard performance metrics for a backtest equity curve.

These are the bog-standard quant ratios you'd see on any fund tear sheet:
total return, CAGR, annualized vol, Sharpe, Sortino, Calmar, max drawdown,
hit rate.

**These are necessary but NOT sufficient.** A backtest with a great Sharpe is
not evidence of a real edge — see ``docs/specs/dsr.md`` (Step 14) for the
Deflated Sharpe correction that controls for multiple-testing inflation and
the higher moments. Use these metrics for first-pass screening; trust them
only after DSR confirms the trial-count-adjusted Sharpe is positive.

Two entry points:
  - ``compute_metrics(equity_curve, ...)`` — pure function on a Series.
  - ``metrics_for(result)`` — convenience wrapper that reads risk_free and
    trading-days from a ``BacktestResult``'s config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestResult


@dataclass(frozen=True)
class Metrics:
    """Standard performance metrics for an equity curve.

    Sign conventions matter:
      - ``max_drawdown`` is NEGATIVE (e.g., ``-0.34`` = 34% peak-to-trough drop).
      - ``hit_rate`` is in [0, 1] (e.g., ``0.55`` = 55% of days were positive).
      - Returns are decimal (e.g., ``0.12`` = 12%), not percent.

    Provenance fields (``risk_free_annual``, ``trading_days_per_year``)
    are stamped so the consumer can tell which convention produced these
    numbers — important when comparing results across configs.
    """

    # ----- Returns -----
    total_return: float                # (end/start) - 1
    cagr: float                        # geometric annualized return
    annualized_return: float           # arithmetic mean of daily returns * trading_days

    # ----- Risk -----
    annualized_vol: float              # std(daily) * sqrt(trading_days)
    downside_vol: float                # std of NEGATIVE daily returns only * sqrt(trading_days)
    max_drawdown: float                # most negative (peak-to-trough), e.g. -0.34
    max_drawdown_duration_days: int

    # ----- Risk-adjusted -----
    sharpe: float                      # (annual_return - rf_annual) / annual_vol
    sortino: float                     # (annual_return - rf_annual) / downside_vol
    calmar: float                      # CAGR / |max_dd|

    # ----- Trading -----
    hit_rate: float                    # fraction of strictly-positive daily returns

    # ----- Provenance -----
    n_days: int                        # bars in the equity curve
    starting_equity: float
    ending_equity: float
    risk_free_annual: float
    trading_days_per_year: int

    def __str__(self) -> str:
        """Pretty-print summary for terminal / report use."""
        return (
            f"Metrics ({self.n_days} days, ${self.starting_equity:,.0f} -> ${self.ending_equity:,.0f}):\n"
            f"  Total return:   {self.total_return:+8.2%}\n"
            f"  CAGR:           {self.cagr:+8.2%}\n"
            f"  Ann. vol:       {self.annualized_vol:8.2%}\n"
            f"  Sharpe:         {self.sharpe:8.2f}\n"
            f"  Sortino:        {self.sortino:8.2f}\n"
            f"  Max drawdown:   {self.max_drawdown:+8.2%}\n"
            f"  Max DD days:    {self.max_drawdown_duration_days:>8d}\n"
            f"  Calmar:         {self.calmar:8.2f}\n"
            f"  Hit rate:       {self.hit_rate:8.2%}\n"
        )


def compute_metrics(
    equity_curve: pd.Series,
    *,
    risk_free_annual: float = 0.04,
    trading_days_per_year: int = 252,
) -> Metrics:
    """Compute standard performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve
        End-of-bar equity values, ordered chronologically. NaN values are
        dropped (the engine fills these for warmup bars).
    risk_free_annual
        Annualized risk-free rate, decimal (e.g., 0.04 = 4%). Used in
        Sharpe and Sortino. Default matches the shipped config.
    trading_days_per_year
        Annualization factor. 252 is the US equity convention; some
        literatures use 250 or 260.

    Raises
    ------
    ValueError
        If the curve has fewer than 2 points (no returns to compute) or
        any non-positive equity (would cause infinite/negative returns).
    """
    eq = equity_curve.dropna()
    if len(eq) < 2:
        raise ValueError(
            f"need at least 2 equity points to compute returns; got {len(eq)}"
        )
    if (eq <= 0).any():
        # A zero or negative equity means the strategy blew up. We can't
        # compute meaningful returns past that point — caller should slice
        # to the pre-blowup region or accept the strategy is bust.
        raise ValueError(
            "equity curve has non-positive values; can't compute returns. "
            "Either the strategy blew up, or there's a bug upstream."
        )

    start = float(eq.iloc[0])
    end = float(eq.iloc[-1])
    n = len(eq)

    # ---- Returns -------------------------------------------------------
    total_return = (end / start) - 1.0

    # Years used for CAGR is based on number of *returns* (n-1), which is
    # the elapsed time in trading days. (e.g., 253 bars = 252 returns = 1 yr.)
    years = (n - 1) / trading_days_per_year
    cagr = (end / start) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    daily_returns = eq.pct_change().dropna()
    mean_daily = float(daily_returns.mean())
    annualized_return = mean_daily * trading_days_per_year

    # ---- Risk ----------------------------------------------------------
    # Sample standard deviation (ddof=1) is the unbiased estimator for the
    # population std; matches what pandas/numpy default to and what every
    # quant text means by "std".
    std_daily = float(daily_returns.std(ddof=1))
    annualized_vol = std_daily * math.sqrt(trading_days_per_year)

    # Downside vol: only the negative-return days enter the std. Returns
    # an empty Series if no negatives → downside_vol stays at 0.
    neg_returns = daily_returns[daily_returns < 0]
    downside_std_daily = (
        float(neg_returns.std(ddof=1)) if len(neg_returns) > 1 else 0.0
    )
    downside_vol = downside_std_daily * math.sqrt(trading_days_per_year)

    # Drawdown: at each bar, (equity - running max) / running max.
    # Always <= 0. The min is the worst peak-to-trough.
    peak = eq.expanding().max()
    drawdown = (eq - peak) / peak
    max_drawdown = float(drawdown.min())

    # Drawdown duration: longest consecutive run of bars below the high-water mark.
    underwater = (eq < peak).to_numpy()
    max_dd_duration = _longest_true_run(underwater)

    # ---- Risk-adjusted -------------------------------------------------
    # Simple linear de-annualization of the risk-free rate. (1+rf)^(1/252)-1
    # is more precise but differs by ~bps for typical rf values.
    rf_daily = risk_free_annual / trading_days_per_year
    excess_daily_mean = mean_daily - rf_daily

    # Sharpe: undefined when std is zero (constant-return strategy). We
    # return 0 in that case rather than NaN — a "no signal" strategy
    # should sort to the bottom of any ranking, not the top.
    sharpe = (
        (excess_daily_mean / std_daily) * math.sqrt(trading_days_per_year)
        if std_daily > 0
        else 0.0
    )
    sortino = (
        (excess_daily_mean / downside_std_daily) * math.sqrt(trading_days_per_year)
        if downside_std_daily > 0
        else 0.0
    )
    # Calmar undefined when no drawdown — same "0 not NaN" treatment.
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    # ---- Trading -------------------------------------------------------
    # Strictly positive days; zero-return days don't count as wins.
    hit_rate = float((daily_returns > 0).mean()) if len(daily_returns) > 0 else 0.0

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        downside_vol=downside_vol,
        max_drawdown=max_drawdown,
        max_drawdown_duration_days=max_dd_duration,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        hit_rate=hit_rate,
        n_days=n,
        starting_equity=start,
        ending_equity=end,
        risk_free_annual=risk_free_annual,
        trading_days_per_year=trading_days_per_year,
    )


def metrics_for(
    result: BacktestResult,
    *,
    risk_free_annual: float | None = None,
    trading_days_per_year: int | None = None,
) -> Metrics:
    """Compute metrics from a ``BacktestResult``, using its config's defaults.

    Convenience wrapper around ``compute_metrics``. If you want to override
    the risk-free rate or trading-day convention, pass it explicitly;
    otherwise we read from ``result.config.evaluation``.
    """
    rf = (
        risk_free_annual
        if risk_free_annual is not None
        else result.config.evaluation.risk_free_annual
    )
    tdpy = (
        trading_days_per_year
        if trading_days_per_year is not None
        else result.config.evaluation.trading_days_per_year
    )
    return compute_metrics(
        result.equity_curve,
        risk_free_annual=rf,
        trading_days_per_year=tdpy,
    )


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _longest_true_run(arr: np.ndarray) -> int:
    """Length of the longest run of True values in a boolean array.

    Used for max-drawdown duration. Pure Python loop is fine — even on a
    100k-bar curve it's microseconds. A vectorized version with
    ``np.diff`` works but is harder to read for marginal speedup.
    """
    longest = 0
    current = 0
    for x in arr:
        if x:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest
