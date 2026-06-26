"""attribution.py — regress a portfolio's returns on the factor panel.

Answers: of the book's return, how much is ALPHA (the regression
intercept — return unexplained by known factors) versus BETA (loadings
on market/momentum/reversal/low-vol)? A book that's all beta has no
skill; a positive, significant alpha is the thing worth paying for.

OLS with an intercept, Newey-West-agnostic plain standard errors (the
series are daily and roughly serially uncorrelated for a daily-rebalanced
book; if that assumption breaks the t-stats are optimistic — flagged in
the result). Everything is reported both per-day and annualized.

IMPORTANT — statistical honesty: alpha's t-stat is only meaningful with
enough observations. With ~1 month of live history (~20 days) against a
4-factor model the estimate is *directional at best*. The result carries
``n_obs`` so the consumer (and the monthly analyst) can weight it
accordingly, and ``warnings`` flags an underpowered fit explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# Below this many observations the regression is underpowered and we say so.
_MIN_OBS_FOR_INFERENCE = 60


@dataclass(frozen=True)
class AttributionResult:
    """Output of a factor regression of one return series."""

    alpha_daily: float
    alpha_annual: float
    alpha_tstat: float
    betas: dict[str, float]
    beta_tstats: dict[str, float]
    r_squared: float
    resid_vol_annual: float
    n_obs: int
    factor_names: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-block human-readable summary for the monthly email."""
        lines = [
            f"alpha: {self.alpha_annual:+.2%}/yr "
            f"(t={self.alpha_tstat:+.2f}, daily {self.alpha_daily:+.4%})",
            f"R²: {self.r_squared:.2f}   "
            f"residual vol: {self.resid_vol_annual:.1%}/yr   n={self.n_obs}",
            "factor loadings (beta, t):",
        ]
        for f in self.factor_names:
            lines.append(
                f"  {f:8s} {self.betas[f]:+.2f}  (t={self.beta_tstats[f]:+.2f})"
            )
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


def attribute_returns(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> AttributionResult:
    """OLS-regress ``portfolio_returns`` on ``factor_returns`` + intercept.

    Both are indexed by date; they're inner-joined on common dates. The
    intercept is alpha; the slopes are factor betas.

    Raises
    ------
    ValueError
        If fewer than (n_factors + 2) common observations exist — too few
        to fit at all.
    """
    factor_names = tuple(factor_returns.columns)
    joined = pd.concat(
        [portfolio_returns.rename("port"), factor_returns], axis=1, join="inner"
    ).dropna(how="any")
    n = len(joined)
    k = len(factor_names)
    if n < k + 2:
        raise ValueError(
            f"need at least {k + 2} common observations to fit a {k}-factor "
            f"model; got {n}. Portfolio history is too short."
        )

    y = joined["port"].to_numpy()
    X = np.column_stack([np.ones(n), joined[list(factor_names)].to_numpy()])

    # OLS: beta_hat = (X'X)^-1 X'y
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    coef = xtx_inv @ X.T @ y
    resid = y - X @ coef
    dof = max(1, n - X.shape[1])
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    tstats = coef / np.where(se > 0, se, np.nan)

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid**2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    alpha_daily = float(coef[0])
    betas = {f: float(coef[i + 1]) for i, f in enumerate(factor_names)}
    beta_t = {f: float(tstats[i + 1]) for i, f in enumerate(factor_names)}

    warnings: list[str] = []
    if n < _MIN_OBS_FOR_INFERENCE:
        warnings.append(
            f"underpowered: only {n} obs (<{_MIN_OBS_FOR_INFERENCE}); "
            "alpha t-stat is directional, not conclusive."
        )

    return AttributionResult(
        alpha_daily=alpha_daily,
        alpha_annual=alpha_daily * TRADING_DAYS,
        alpha_tstat=float(tstats[0]),
        betas=betas,
        beta_tstats=beta_t,
        r_squared=r2,
        resid_vol_annual=float(np.sqrt(sigma2) * np.sqrt(TRADING_DAYS)),
        n_obs=n,
        factor_names=factor_names,
        warnings=warnings,
    )
