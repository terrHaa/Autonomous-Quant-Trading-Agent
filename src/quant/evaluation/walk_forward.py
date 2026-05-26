"""walk_forward.py — rolling out-of-sample evaluation.

Why walk-forward exists
-----------------------
DSR (Step 14) controls for in-sample inflation due to selecting the best of
N variants. Walk-forward controls for a different bias: fitting (often
implicitly) to a specific historical window. Even rule-based strategies
like SMA crossover can be "fit" by the researcher choosing the 50/200
parameters *because* they worked in 2015-2024. Walk-forward asks: would
SMA(50,200) have worked in 2018 if you only had data through 2017?

The procedure
-------------
For each fold:
  - Train window: most-recent ``train_years`` years.
  - Test window:  the next ``test_years`` years.
The window slides forward by ``step_years`` per fold.

For each fold we run the engine on bars spanning ``train_start .. test_end``
(so the strategy can see history for SMAs etc.), then extract the test-window
portion of the equity curve. Concatenated test-window curves give the OOS
equity curve — the single number the registry's promotion gate looks at.

For rule-based strategies (no parameter fitting), walk-forward is purely a
*time-period stress test*: does the rule survive in unseen years?
For parameterized strategies (later: HRP, momentum-with-lookback), an
explicit ``fit(train_snapshot)`` step on each fold's train window will
matter more.

Limitations of this implementation
----------------------------------
- **Positions carry from train into test within a fold.** Our engine runs
  continuously through the fold; at test_start the strategy may already
  hold a position built up during training. For path-dependent strategies
  this differs from a "fresh start on test_start" simulation. For SMA
  crossover and similar stateless rules, the difference is negligible.
- **The strategy is reused across folds.** Stateful strategies (those
  with internal state surviving between bars) may leak fold information.
  Document this and prefer stateless strategies; if stateful state is
  needed, the user should construct fresh strategy instances themselves
  (TODO: accept a ``strategy_factory`` callable).
- **No re-fitting yet.** SMA's parameters don't change per fold; we
  evaluate the same rule everywhere. Re-fitting hooks are a future
  enhancement once we have parameterized strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from quant.backtest.engine import BacktestResult, run_backtest
from quant.backtest.types import Strategy
from quant.config import Config
from quant.evaluation.metrics import Metrics, compute_metrics


@dataclass(frozen=True)
class WalkForwardFold:
    """One train/test fold of a walk-forward run.

    Holds the full backtest result for the fold (train + test concatenated),
    plus the dates demarking the test window so callers can extract just
    the OOS slice for analysis.
    """

    fold_idx: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    result: BacktestResult


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregated results from a full walk-forward run.

    ``oos_equity_curve`` is the chain you'd compare against a single-pass
    in-sample run to gauge OOS degradation. ``overall_metrics`` is the
    headline number; ``fold_metrics`` shows the per-fold breakdown for
    spotting "this one bad year" patterns.
    """

    config: Config
    strategy_name: str
    folds: list[WalkForwardFold]
    oos_equity_curve: pd.Series
    fold_metrics: list[Metrics]
    overall_metrics: Metrics
    metadata: dict[str, Any] = field(default_factory=dict)

    def fold_summary(self) -> pd.DataFrame:
        """One-row-per-fold tabular summary, handy for printing/reports."""
        rows = []
        for fold, m in zip(self.folds, self.fold_metrics, strict=True):
            rows.append({
                "fold": fold.fold_idx,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "total_return": m.total_return,
                "sharpe": m.sharpe,
                "max_dd": m.max_drawdown,
                "n_fills": fold.result.metadata.get("n_fills", 0),
            })
        return pd.DataFrame(rows)


def run_walk_forward(
    *,
    config: Config,
    strategy: Strategy,
    bars: pd.DataFrame,
    train_years: int | None = None,
    test_years: int | None = None,
    step_years: int | None = None,
) -> WalkForwardResult:
    """Run a strategy through walk-forward analysis.

    Parameters
    ----------
    config, strategy, bars
        Same as ``run_backtest``.
    train_years, test_years, step_years
        Optional overrides for the walk-forward window sizes. Default to
        ``config.evaluation.walk_forward.*``.

    Returns
    -------
    WalkForwardResult
        Folds, OOS equity curve, per-fold and overall metrics.

    Raises
    ------
    ValueError
        If the bars span is too short for even one fold.
    """
    train_years = train_years or config.evaluation.walk_forward.train_years
    test_years = test_years or config.evaluation.walk_forward.test_years
    step_years = step_years or config.evaluation.walk_forward.step_years
    tdpy = config.evaluation.trading_days_per_year

    # Trading days, sorted ascending. We work in *index* space (i.e., "the
    # N-th trading day") rather than calendar days so fold sizes line up
    # with the trading_days_per_year convention used elsewhere.
    trading_dates = sorted(
        set(d for d in bars.index.get_level_values("timestamp").date.tolist())
    )
    n = len(trading_dates)

    train_n = train_years * tdpy
    test_n = test_years * tdpy
    step_n = step_years * tdpy

    if n < train_n + test_n:
        raise ValueError(
            f"need at least {train_n + test_n} bars for one fold "
            f"(train={train_n}, test={test_n}); got {n}"
        )

    # ---- Build the folds ------------------------------------------------
    folds: list[WalkForwardFold] = []
    i = 0
    while i + train_n + test_n <= n:
        train_start = trading_dates[i]
        train_end = trading_dates[i + train_n - 1]
        test_start = trading_dates[i + train_n]
        test_end = trading_dates[i + train_n + test_n - 1]

        # Slice bars to this fold's window (train + test concatenated).
        # The strategy sees this whole span — that's how SMA gets its
        # 200-day history at the start of the test period.
        ts = bars.index.get_level_values("timestamp")
        fold_mask = (ts.date >= train_start) & (ts.date <= test_end)
        fold_bars = bars[fold_mask]

        # Run the engine on the fold window. The strategy is re-used —
        # caller is responsible for ensuring it's safe to re-run.
        result = run_backtest(config=config, strategy=strategy, bars=fold_bars)

        folds.append(WalkForwardFold(
            fold_idx=len(folds),
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            result=result,
        ))
        i += step_n

    # ---- Chain test-window equity curves --------------------------------
    # Each fold's equity at test_start reflects train-period trading. We
    # don't want that to compound across folds. Instead, scale each fold's
    # test slice so its first point equals the prior fold's last point —
    # equivalent to "the strategy starts each test window with the equity
    # it ended the prior test window with, in returns space".
    oos_chunks: list[pd.Series] = []
    prior_end: float = config.backtest.starting_equity

    for fold in folds:
        eq = fold.result.equity_curve.dropna()
        # All bars on or after test_start belong to the OOS window.
        # eq.index is a list of date objects; boolean masking works.
        test_eq = eq[[d >= fold.test_start for d in eq.index]]
        if test_eq.empty:
            continue
        scale = prior_end / float(test_eq.iloc[0])
        chunk = test_eq * scale
        # Skip the first bar of the chunk to avoid a duplicate boundary
        # value when concatenating consecutive folds.
        if oos_chunks:
            chunk = chunk.iloc[1:]
        if chunk.empty:
            continue
        oos_chunks.append(chunk)
        prior_end = float(chunk.iloc[-1])

    oos_equity_curve = (
        pd.concat(oos_chunks) if oos_chunks
        else pd.Series([config.backtest.starting_equity], dtype=float)
    )

    # ---- Per-fold and overall metrics -----------------------------------
    fold_metrics: list[Metrics] = []
    for fold in folds:
        eq = fold.result.equity_curve.dropna()
        test_eq = eq[[d >= fold.test_start for d in eq.index]]
        if len(test_eq) < 2:
            # Degenerate fold — skip metric computation. Shouldn't happen
            # with realistic test_years, but defend against it.
            continue
        fold_metrics.append(compute_metrics(
            test_eq,
            risk_free_annual=config.evaluation.risk_free_annual,
            trading_days_per_year=tdpy,
        ))

    if len(oos_equity_curve) >= 2:
        overall = compute_metrics(
            oos_equity_curve,
            risk_free_annual=config.evaluation.risk_free_annual,
            trading_days_per_year=tdpy,
        )
    else:
        # No OOS data at all — return zeroed metrics rather than raise.
        # This is an edge case (bars span exactly equals one fold's train),
        # but raising would force every caller to try/except defensively.
        raise ValueError(
            "OOS equity curve has fewer than 2 points; cannot compute metrics. "
            "This usually means no folds completed — increase the bars span."
        )

    return WalkForwardResult(
        config=config,
        strategy_name=strategy.name,
        folds=folds,
        oos_equity_curve=oos_equity_curve,
        fold_metrics=fold_metrics,
        overall_metrics=overall,
        metadata={
            "n_folds": len(folds),
            "train_years": train_years,
            "test_years": test_years,
            "step_years": step_years,
            "oos_start": folds[0].test_start if folds else None,
            "oos_end": folds[-1].test_end if folds else None,
        },
    )
