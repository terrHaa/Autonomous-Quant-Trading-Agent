"""sizing.py — volatility targeting and Kelly fraction.

Two related but distinct sizing tools:

- **Vol targeting** scales the portfolio's gross exposure up or down so its
  realized volatility matches a target (e.g., 10% annualized). The whole
  scale calculation is a single ratio: ``target_vol / realized_vol``.
  Bread-and-butter risk control at every systematic shop.

- **Kelly fraction** chooses leverage to maximize expected geometric
  growth, given the strategy's mean and variance. Pure Kelly is
  ``mean / variance``; fractional Kelly (0.25 or 0.5) is the standard
  conservative variant because pure Kelly is wildly aggressive when
  expected return is estimated rather than known.

In practice most desks use vol targeting alone. Kelly is more theoretical
and tends to recommend leverage that fails any sane risk policy.

Important: these functions operate on POST-HOC returns — meaning they use
the full history of the returns series passed in. For LIVE / in-engine
sizing where the strategy must size itself bar-by-bar without seeing the
future, wrap a rolling-window vol estimate in your own strategy class.
This module is for offline portfolio-construction analysis.
"""

from __future__ import annotations

import math

import pandas as pd


# A small floor below which we treat realized vol as zero — same idea as
# the dsr.py guard against pandas's near-zero std on near-constant inputs.
_VOL_FLOOR = 1e-12


def vol_target_scale(
    portfolio_returns: pd.Series,
    *,
    target_vol_annual: float,
    trading_days_per_year: int = 252,
) -> float:
    """Multiplicative scale to bring realized vol to target.

    ``new_vol = scale * old_vol = target_vol_annual``

    Returns 0.0 if the input has essentially zero variance (no signal to
    scale). Caller should treat that as "stay flat".

    Parameters
    ----------
    portfolio_returns
        Daily return series of the candidate portfolio (combine your
        weighted strategy returns into one Series before calling).
    target_vol_annual
        Decimal annualized vol target (e.g., 0.10 = 10%).
    trading_days_per_year
        Annualization factor. 252 for US equities.
    """
    if target_vol_annual <= 0:
        raise ValueError(
            f"target_vol_annual must be positive (got {target_vol_annual})"
        )
    realized_std = float(portfolio_returns.std(ddof=1))
    if realized_std < _VOL_FLOOR:
        # Zero vol → no signal to scale. Stay flat is the safe response.
        return 0.0
    realized_vol_annual = realized_std * math.sqrt(trading_days_per_year)
    return target_vol_annual / realized_vol_annual


def apply_vol_target(
    weights: pd.Series,
    strategy_returns: pd.DataFrame,
    *,
    target_vol_annual: float,
    trading_days_per_year: int = 252,
    max_gross_leverage: float = float("inf"),
) -> pd.Series:
    """Scale a set of weights so the resulting portfolio hits vol target.

    Workflow:
      1. Compute portfolio daily returns: ``(strategy_returns * weights).sum(axis=1)``.
      2. Compute vol-target scale from those returns.
      3. Cap the scale at ``max_gross_leverage / sum(|weights|)``.
      4. Return ``weights * scale``.

    Parameters
    ----------
    weights
        Base weights, indexed by strategy/asset name. Typically from HRP.
        Can sum to anything (1.0 = long-only base; 0 = market-neutral).
    strategy_returns
        DataFrame: rows = time, columns = strategy/asset name. Columns
        must match weights' index.
    target_vol_annual, trading_days_per_year
        As in ``vol_target_scale``.
    max_gross_leverage
        Hard cap on ``sum(|scaled_weights|)``. Default ∞ = uncapped.
        Pass ``config.risk.max_gross_leverage`` from a loaded Config.

    Returns
    -------
    pandas.Series
        Scaled weights. Same index as input.

    Raises
    ------
    ValueError
        If the weight index doesn't match the strategy_returns columns.
    """
    if set(weights.index) != set(strategy_returns.columns):
        raise ValueError(
            f"weights index ({sorted(weights.index)}) must match "
            f"strategy_returns columns ({sorted(strategy_returns.columns)})"
        )

    # Align column order so the dot product is unambiguous.
    aligned_returns = strategy_returns[weights.index]
    portfolio_returns = (aligned_returns * weights).sum(axis=1).dropna()

    raw_scale = vol_target_scale(
        portfolio_returns,
        target_vol_annual=target_vol_annual,
        trading_days_per_year=trading_days_per_year,
    )

    # Apply leverage cap. Compare gross exposure (sum of absolute weights)
    # against the cap; if scale * gross_base > max, clip down.
    gross_base = float(weights.abs().sum())
    if gross_base > 0 and raw_scale * gross_base > max_gross_leverage:
        scale = max_gross_leverage / gross_base
    else:
        scale = raw_scale

    return weights * scale


def kelly_leverage(
    returns: pd.Series,
    *,
    fraction: float = 1.0,
    trading_days_per_year: int = 252,
) -> float:
    """Pure or fractional Kelly optimal leverage.

    The formula:
        leverage = fraction * (mean_return_annual / variance_annual)

    Pure Kelly (fraction=1.0) maximizes long-run geometric growth IF the
    mean and variance are known exactly. They never are — so practitioners
    use half-Kelly (0.5) or quarter-Kelly (0.25). Reading: "I think the
    strategy edge is real, but I want to be ~half as aggressive as the
    naive Kelly recommends."

    Parameters
    ----------
    returns
        Daily return series of the strategy / portfolio.
    fraction
        The fractional Kelly multiplier. 1.0 = pure Kelly. Common
        practical values: 0.25, 0.5.
    trading_days_per_year
        Annualization factor.

    Returns
    -------
    float
        Suggested leverage. Can be negative (strategy has negative mean
        — Kelly says short) or > 1 (lever up). Caller decides whether
        to honor that or cap.
    """
    if fraction <= 0:
        raise ValueError(f"fraction must be positive (got {fraction})")
    mean_annual = float(returns.mean()) * trading_days_per_year
    var_annual = float(returns.var(ddof=1)) * trading_days_per_year
    if var_annual <= _VOL_FLOOR ** 2:
        # No variance → Kelly formula explodes / undefined. Stay flat.
        return 0.0
    return fraction * mean_annual / var_annual
