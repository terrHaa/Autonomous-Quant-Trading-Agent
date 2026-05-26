"""ensemble.py — daily multi-strategy allocator for the agent.

What this is for
----------------
Single-strategy agents are fragile: if their one signal stops working,
they lose. This module runs all three strategies the platform has built
(SMA crossover, mean reversion, cross-sectional momentum) every day,
then combines their per-symbol targets via HRP weights across strategies.

The two layers of allocation:

1. **Within each strategy** — each strategy emits its own per-symbol
   weight vector (summing to ≤ 1.0). Cross-sectional momentum picks
   its top-K; SMA picks every uptrending name; mean reversion picks
   oversold names.

2. **Across strategies** — HRP weights ``h_sma + h_mr + h_xsec = 1``
   determine how much of the book each strategy controls. They're
   recomputed weekly from a rolling backtest of each strategy on the
   universe — when one strategy hits a rough patch, HRP shrinks its
   allocation automatically (no human-in-the-loop required).

The final per-symbol target is the inner-product:
    final[sym] = Σ_strategy h_strategy × strategy_target[sym]

The 20%-per-name cap + 5% stop-loss in the executor still apply to
the resulting book.

Why this isn't just diluted beta
--------------------------------
A common worry: combining strategies might just average them into a
broad-equity exposure that earns no alpha. The defense is that HRP
weights aren't equal-weight — they're inversely tied to correlation
structure. When strategies move similarly (during a regime where
everything works the same direction), HRP gives one of them more
weight. When they're uncorrelated, HRP splits more evenly. The Sharpe
of the combination is typically HIGHER than the average of the
components precisely because of this correlation-sensitive allocation.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quant.allocator import hrp_weights
from quant.backtest.engine import run_backtest
from quant.backtest.types import Snapshot, Strategy
from quant.config import Config
from quant.strategies import (
    CrossSectionalMomentum,
    MeanReversion,
    SmaCrossover,
)


logger = logging.getLogger(__name__)


# Where the ensemble state persists. Daily runner reads from here; weekly
# review writes new HRP weights; monthly review writes new params.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_STATE_PATH = _PROJECT_ROOT / "data" / "agent" / "ensemble_state.json"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleState:
    """The agent's tunable knobs — one bag of values, one file on disk.

    Per-strategy params and the HRP weights across strategies live here.
    The HARD operator rules (5% stop, 20% cap, top-100 universe) stay
    as constants in ``daily_runner.py`` — the ensemble never touches them.
    """

    # SMA crossover params.
    sma_fast: int = 50
    sma_slow: int = 200

    # Mean reversion params.
    mr_lookback: int = 5
    mr_threshold_pct: float = 0.02
    mr_allow_short: bool = False

    # Cross-sectional momentum params.
    xsec_top_k: int = 10
    xsec_lookback: int = 60
    xsec_skip: int = 5

    # HRP weights — keyed by the strategies' ``name`` field. Sum to 1.0.
    # Default = equal weight across the three. Refit weekly.
    hrp_weights: dict[str, float] = field(default_factory=lambda: {
        "sma_crossover_50_200": 1 / 3,
        "mean_reversion_5_200bp": 1 / 3,
        "xsec_momo_60_5_10": 1 / 3,
    })
    last_hrp_refit_date: str = ""   # ISO date; empty = never


def load_ensemble_state(path: Path | None = None) -> EnsembleState:
    """Load from JSON; return defaults if no file exists."""
    p = path or DEFAULT_STATE_PATH
    if not p.exists():
        return EnsembleState()
    data = json.loads(p.read_text())
    return EnsembleState(**data)


def save_ensemble_state(state: EnsembleState, path: Path | None = None) -> Path:
    """Persist to JSON. Creates the parent directory if missing."""
    p = path or DEFAULT_STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state), indent=2))
    return p


# ---------------------------------------------------------------------------
# Strategy construction from state
# ---------------------------------------------------------------------------


def build_strategies(state: EnsembleState, universe: list[str]) -> list[Strategy]:
    """Build the three strategy instances from the persisted state.

    Returned in a stable order: SMA, MR, XsecMomentum. Each has the
    standard ``.name`` field; the HRP weights dict keys must match.
    """
    return [
        SmaCrossover(universe, fast=state.sma_fast, slow=state.sma_slow),
        MeanReversion(
            universe,
            lookback=state.mr_lookback,
            threshold_pct=state.mr_threshold_pct,
            allow_short=state.mr_allow_short,
        ),
        CrossSectionalMomentum(
            universe,
            lookback=state.xsec_lookback,
            skip=state.xsec_skip,
            top_k=state.xsec_top_k,
        ),
    ]


# ---------------------------------------------------------------------------
# Daily inner-product: per-strategy targets → final per-symbol targets
# ---------------------------------------------------------------------------


def compute_ensemble_targets(
    strategies: list[Strategy],
    hrp_w: dict[str, float],
    snapshot: Snapshot,
) -> dict[str, float]:
    """Combine strategy outputs into final per-symbol target weights.

    For each strategy s:
      target_contribution = hrp_w[s.name] × s.on_bar(snapshot)

    The final per-symbol weight is the sum of contributions across all
    strategies. Negative contributions (from short signals) net against
    positive ones, which is correct: if one strategy says long AAPL and
    another says short AAPL by the same magnitude, the desk is flat.

    Strategies whose name isn't in ``hrp_w`` get zero weight (defensive
    against stale strategy lists).
    """
    combined: dict[str, float] = defaultdict(float)
    for strat in strategies:
        h = hrp_w.get(strat.name, 0.0)
        if h == 0.0:
            # Skip entirely; running an unused strategy still costs a few
            # ms per bar so don't bother.
            continue
        targets = strat.on_bar(snapshot)
        for sym, w in targets.items():
            # OrderIntent objects flow through unchanged from any strategy
            # that emits them; for the ensemble we only know how to add
            # FLOATS. If a strategy returns an OrderIntent, fall back to
            # its target_weight only. Caller logs in the report.
            if isinstance(w, (int, float)):
                combined[sym] += h * float(w)
            else:
                combined[sym] += h * float(getattr(w, "target_weight", 0.0))

    # Drop dust positions to keep the order book clean. Anything under
    # 0.5% of equity is below the stop-loss noise floor anyway.
    return {s: w for s, w in combined.items() if abs(w) >= 0.005}


# ---------------------------------------------------------------------------
# HRP refit — used by the weekly review
# ---------------------------------------------------------------------------


def refit_hrp_weights(
    state: EnsembleState,
    *,
    universe: list[str],
    bars: pd.DataFrame,
    config: Config,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Backtest each strategy on ``bars``; compute HRP across their returns.

    Returns (new_hrp_weights, diagnostic_info). The caller decides whether
    to save. The diagnostic dict includes per-strategy Sharpe / total
    return so the report can surface what changed and why.

    Robustness: if any strategy produces a degenerate equity curve (all
    flat), we fall back to equal weights so the agent has a working
    allocation rather than zeros.
    """
    strategies = build_strategies(state, universe)
    returns_by_name: dict[str, pd.Series] = {}
    diag: dict[str, Any] = {"per_strategy": {}}

    for strat in strategies:
        result = run_backtest(config=config, strategy=strat, bars=bars)
        rets = result.equity_curve.pct_change().dropna()
        if len(rets) < 30 or float(rets.std(ddof=1)) < 1e-12:
            # Degenerate — strategy never traded or had zero variance.
            # Skip; HRP can't use it.
            diag["per_strategy"][strat.name] = {
                "skipped": True,
                "reason": f"len={len(rets)}, std≈0",
            }
            continue
        returns_by_name[strat.name] = rets
        diag["per_strategy"][strat.name] = {
            "total_return": float(result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1),
            "sharpe": float(rets.mean() / rets.std(ddof=1) * (252 ** 0.5)),
            "n_days": len(rets),
        }

    if len(returns_by_name) < 2:
        # Can't run HRP on < 2 strategies — fall back to equal-weight
        # over the strategies whose name we KNOW (from state defaults).
        logger.warning(
            "refit_hrp_weights: < 2 usable strategies; falling back to equal weights"
        )
        names = [s.name for s in strategies]
        equal = {n: 1.0 / len(names) for n in names}
        diag["fallback"] = "equal_weight (< 2 usable strategies)"
        return equal, diag

    # Align all strategies' returns on the same index, fill missing with
    # 0 (a strategy that wasn't active some day contributed zero return).
    returns_df = pd.DataFrame(returns_by_name).fillna(0.0)

    new_weights = hrp_weights(returns_df).to_dict()
    # Any strategies not in returns_df (because they were degenerate)
    # get zero weight. That's the right behavior — don't allocate to
    # something that couldn't be evaluated.
    for s in strategies:
        new_weights.setdefault(s.name, 0.0)

    diag["hrp_weights_before"] = dict(state.hrp_weights)
    diag["hrp_weights_after"] = dict(new_weights)
    return new_weights, diag
