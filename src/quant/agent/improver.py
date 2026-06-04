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

from quant.agent.params import MrParams, SmaParams, StrategyParams
from quant.backtest.engine import run_backtest
from quant.config import Config
from quant.evaluation.dsr import (
    deflated_sharpe_ratio,
    estimate_var_sr_from_trials,
)
from quant.evaluation.metrics import metrics_for
from quant.strategies import CrossSectionalMomentum, MeanReversion, SmaCrossover

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


# T4.22 — backtest helpers for SMA and MR so the monthly review can
# pass current-baseline performance into the AI analyst. The analyst
# then has visibility into all three strategies' performance, not
# just xsec — and can recommend SMA/MR param changes via the
# structured ``proposed_state_changes`` channel (which now supports
# sma_fast/sma_slow/mr_lookback/mr_threshold_pct in addition to
# trail_pct). Auto-apply remains xsec-only; SMA/MR tuning goes
# through analyst recommendation + operator approval.


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
    return _evaluate_strategy(strategy, params, universe=universe, bars=bars, config=config)


def evaluate_sma_candidate(
    params: SmaParams,
    *,
    universe: list[str],
    bars: pd.DataFrame,
    config: Config,
) -> tuple[ImprovementCandidate, pd.Series]:
    """Backtest one SMA candidate. Same return shape as evaluate_candidate."""
    strategy = SmaCrossover(universe, fast=params.fast, slow=params.slow)
    return _evaluate_strategy(strategy, params, universe=universe, bars=bars, config=config)


def evaluate_mr_candidate(
    params: MrParams,
    *,
    universe: list[str],
    bars: pd.DataFrame,
    config: Config,
) -> tuple[ImprovementCandidate, pd.Series]:
    """Backtest one mean-reversion candidate."""
    strategy = MeanReversion(
        universe,
        lookback=params.lookback,
        threshold_pct=params.threshold_pct,
        vol_normalize=params.vol_normalize,
        vol_multiplier=params.vol_multiplier,
    )
    return _evaluate_strategy(strategy, params, universe=universe, bars=bars, config=config)


def _evaluate_strategy(
    strategy,
    params,
    *,
    universe: list[str],
    bars: pd.DataFrame,
    config: Config,
) -> tuple[ImprovementCandidate, pd.Series]:
    """Shared backtest + metrics-pack-up logic for all three evaluators."""
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
    #
    # T4.21 — turnover constraint (implicit).
    # We do NOT have an explicit annual turnover cap. The Sharpe-improvement
    # gate IS our turnover control: a candidate with materially higher
    # turnover than the current params would have to overcome the implied
    # higher transaction costs (modeled as friction in run_backtest) to beat
    # current Sharpe. If a high-turnover candidate still beats current
    # Sharpe net of those costs AND passes the DSR gate, the turnover is
    # paying for itself. A v2 improver could add an explicit
    # turnover_ratio_max gate; v1 lets the cost model do the work and keeps
    # the search single-objective.
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
