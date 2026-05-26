"""improver.py — monthly grid search over the agent's strategy params.

What it does
------------
Once a month (called by ``monthly_review.py``):

1. Backtest the agent's CURRENT params on the last 2 years of universe bars.
2. Backtest a small grid of CANDIDATE param tuples on the same window.
3. Apply three safety gates to find a candidate to promote:
   - **Sharpe gate**: candidate.sharpe > current.sharpe
   - **Drawdown gate**: candidate.max_drawdown >= current.max_drawdown
     (max_drawdown is negative; "less negative" means smaller drawdown)
   - **DSR gate**: candidate.dsr >= 0.95 against all tested candidates
     (Bailey-López de Prado deflated Sharpe; treats the grid as the
     trial population so the multi-testing inflation is corrected)
4. If multiple pass, pick the highest-Sharpe one.
5. Return an ``ImprovementResult`` describing what happened. The caller
   (``monthly_review.py``) decides whether to write the winning params
   into ``EnsembleState`` based on the result — keeping the apply
   decision out of the search logic makes the search testable in
   isolation.

Known limitation (v1)
---------------------
The backtest engine doesn't simulate the OTO stop-loss the live agent
applies. So candidate Sharpes here are SYSTEMATICALLY OPTIMISTIC vs
what the same params would deliver in live paper trading. The DSR gate
partially compensates by adding conservatism, but a v2 improver should
simulate stops in-backtest. Document the bias in the email body so the
operator interprets the numbers correctly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from quant.agent.params import StrategyParams
from quant.backtest.engine import run_backtest
from quant.config import Config
from quant.evaluation.dsr import (
    deflated_sharpe_ratio,
    estimate_var_sr_from_trials,
)
from quant.evaluation.metrics import metrics_for
from quant.strategies import CrossSectionalMomentum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default search grid. Kept small so monthly runtime stays under a minute.
# top_k × lookback × skip = 2 × 2 × 2 = 8 candidates (incl. the current).
# Expand in v2 once we have evidence the gates aren't too lax / strict.
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K = [5, 10]
_DEFAULT_LOOKBACK = [60, 120]
_DEFAULT_SKIP = [0, 5]


def default_grid() -> list[StrategyParams]:
    """The v1 search grid. Public so tests can introspect it."""
    grid: list[StrategyParams] = []
    for top_k in _DEFAULT_TOP_K:
        for lookback in _DEFAULT_LOOKBACK:
            for skip in _DEFAULT_SKIP:
                if skip >= lookback:
                    continue   # invalid combination
                grid.append(StrategyParams(
                    top_k=top_k, lookback=lookback, skip=skip,
                ))
    return grid


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImprovementCandidate:
    """One backtested grid point."""

    params: StrategyParams
    sharpe: float
    max_drawdown: float       # negative; -0.20 = 20% drawdown
    total_return: float
    n_days: int


@dataclass(frozen=True)
class ImprovementResult:
    """The full report of one improver invocation."""

    current: ImprovementCandidate
    candidates: list[ImprovementCandidate]
    best_passing: ImprovementCandidate | None
    reason: str


# ---------------------------------------------------------------------------
# Single-candidate evaluation
# ---------------------------------------------------------------------------


def evaluate_candidate(
    params: StrategyParams,
    *,
    universe: list[str],
    bars: pd.DataFrame,
    config: Config,
) -> tuple[ImprovementCandidate, pd.Series]:
    """Backtest one candidate. Returns (candidate, daily_returns_series).

    The daily returns are needed downstream for the DSR computation.
    """
    strategy = CrossSectionalMomentum(
        universe,
        top_k=params.top_k,
        lookback=params.lookback,
        skip=params.skip,
    )
    result = run_backtest(config=config, strategy=strategy, bars=bars)
    m = metrics_for(result)
    cand = ImprovementCandidate(
        params=params,
        sharpe=m.sharpe,
        max_drawdown=m.max_drawdown,
        total_return=m.total_return,
        n_days=m.n_days,
    )
    daily_returns = result.equity_curve.pct_change().dropna()
    return cand, daily_returns


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def search_improvements(
    current_params: StrategyParams,
    *,
    universe: list[str],
    bars: pd.DataFrame,
    config: Config,
    grid: Iterable[StrategyParams] | None = None,
    dsr_threshold: float = 0.95,
) -> ImprovementResult:
    """Grid search; return result describing the best gate-passing candidate.

    ``grid`` defaults to :func:`default_grid`. The current params are
    always evaluated regardless of whether they're in the grid (so the
    "improvement vs status quo" comparison is meaningful).
    """
    grid_list = list(grid) if grid is not None else default_grid()

    # Always include current — even if it's not in the grid — so we can
    # compare against it.
    if current_params not in grid_list:
        grid_list = [current_params] + grid_list
    # De-dupe while preserving order.
    seen: set[tuple] = set()
    unique_grid: list[StrategyParams] = []
    for p in grid_list:
        key = (p.top_k, p.lookback, p.skip)
        if key in seen:
            continue
        seen.add(key)
        unique_grid.append(p)

    logger.info("improver: evaluating %d candidates", len(unique_grid))
    candidates: list[ImprovementCandidate] = []
    returns_by_params: dict[tuple, pd.Series] = {}

    for p in unique_grid:
        try:
            cand, returns = evaluate_candidate(
                p, universe=universe, bars=bars, config=config,
            )
        except Exception as e:
            logger.warning("candidate %s failed: %s", p, e)
            continue
        candidates.append(cand)
        returns_by_params[(p.top_k, p.lookback, p.skip)] = returns

    # Find the current candidate's evaluation.
    current_eval = next(
        (c for c in candidates
         if c.params == current_params),
        None,
    )
    if current_eval is None:
        # The current params couldn't be backtested. We can't compare;
        # bail rather than apply something untested.
        return ImprovementResult(
            current=ImprovementCandidate(
                params=current_params, sharpe=0.0, max_drawdown=0.0,
                total_return=0.0, n_days=0,
            ),
            candidates=candidates,
            best_passing=None,
            reason="current params couldn't be backtested; no change",
        )

    # Filter to candidates that pass the Sharpe AND drawdown gates.
    # max_drawdown >= current.max_drawdown means "drawdown is smaller in
    # magnitude" because both are negative.
    better = [
        c for c in candidates
        if c.params != current_params
        and c.sharpe > current_eval.sharpe
        and c.max_drawdown >= current_eval.max_drawdown
    ]
    if not better:
        return ImprovementResult(
            current=current_eval,
            candidates=candidates,
            best_passing=None,
            reason=(
                "no candidate beat current on BOTH Sharpe and max drawdown"
            ),
        )

    # Pick the best Sharpe among those that passed the cheap gates.
    best = max(better, key=lambda c: c.sharpe)

    # Apply the DSR gate. n_trials = total candidates we evaluated;
    # V[SR] = variance of trial Sharpes (the grid IS the trial pop here).
    all_sharpes = [c.sharpe for c in candidates]
    if len(all_sharpes) < 2:
        return ImprovementResult(
            current=current_eval, candidates=candidates,
            best_passing=None,
            reason="not enough trials to estimate V[SR] for DSR",
        )
    var_sr = estimate_var_sr_from_trials(all_sharpes)
    best_returns = returns_by_params[
        (best.params.top_k, best.params.lookback, best.params.skip)
    ]
    dsr = deflated_sharpe_ratio(
        best_returns,
        n_trials=len(candidates),
        var_sr_trials_annual=var_sr,
    )

    if dsr < dsr_threshold:
        return ImprovementResult(
            current=current_eval, candidates=candidates,
            best_passing=None,
            reason=(
                f"best candidate's DSR {dsr:.3f} < {dsr_threshold} threshold "
                f"(n_trials={len(candidates)}, var_sr={var_sr:.4f}). "
                f"Multi-testing-corrected confidence too low to apply."
            ),
        )

    return ImprovementResult(
        current=current_eval, candidates=candidates,
        best_passing=best,
        reason=(
            f"DSR {dsr:.3f} >= {dsr_threshold}; Sharpe up "
            f"{best.sharpe:.2f} vs {current_eval.sharpe:.2f}; "
            f"max DD {best.max_drawdown:+.2%} vs {current_eval.max_drawdown:+.2%}."
        ),
    )
