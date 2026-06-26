"""signal_health.py — per-strategy predictive-power telemetry.

The monthly review used to judge a strategy by eyeballing the equity
curve ("mean-reversion looks bad lately"). That's slow and subjective.
This module measures each strategy's edge directly via its Information
Coefficient (IC): at each rebalance, the cross-sectional rank
correlation between the weights a strategy WANTS and the forward returns
those names actually delivered. A strategy with positive, stable IC has
a real signal; one whose IC has decayed to zero is dead weight in the
ensemble regardless of how its HRP allocation looks.

What it reports per strategy:
  - mean IC and IC information-ratio (mean/std, annualized) — the
    headline "is there signal and how reliable".
  - IC t-stat — is the mean IC distinguishable from zero.
  - hit rate — fraction of rebalances with IC > 0.
  - decay — recent-half IC vs early-half IC (a falling number is a
    fading edge; this is what would have flagged MR objectively).
  - turnover — average name churn between consecutive rebalances
    (high turnover taxes the edge with slippage).
  - regime-conditional IC — IC split by an optional regime label, so a
    strategy that only works in (say) trending markets is visible as
    such instead of being averaged into mush.

No look-ahead for the STRATEGY: it only ever sees a point-in-time
Snapshot (data through ``as_of``). The evaluator legitimately uses the
realized forward returns — that's what measuring predictive power means.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quant.backtest.types import Snapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalHealth:
    """Predictive-power telemetry for one strategy."""

    strategy_name: str
    n_periods: int
    mean_ic: float
    ic_ir: float            # annualized information ratio of the IC series
    ic_tstat: float
    hit_rate: float
    ic_recent: float        # mean IC over the most recent half
    ic_early: float         # mean IC over the earliest half
    avg_turnover: float
    regime_ic: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def decaying(self) -> bool:
        """True if the edge has materially faded (recent IC << early IC)."""
        return self.ic_recent < self.ic_early - 0.02

    def summary(self) -> str:
        lines = [
            f"{self.strategy_name}: mean IC {self.mean_ic:+.3f} "
            f"(IR {self.ic_ir:+.2f}, t={self.ic_tstat:+.2f}, "
            f"hit {self.hit_rate:.0%}, n={self.n_periods})",
            f"  decay: early {self.ic_early:+.3f} → recent "
            f"{self.ic_recent:+.3f}"
            + ("  ⚠ DECAYING" if self.decaying else ""),
            f"  turnover: {self.avg_turnover:.0%}/rebalance",
        ]
        if self.regime_ic:
            parts = "  ".join(f"{k}={v:+.3f}" for k, v in self.regime_ic.items())
            lines.append(f"  regime IC: {parts}")
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, NaN-safe, returns 0.0 on degenerate input."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    ar = pd.Series(a[mask]).rank().to_numpy()
    br = pd.Series(b[mask]).rank().to_numpy()
    if ar.std() == 0 or br.std() == 0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def compute_signal_health(
    strategy,
    full_bars: pd.DataFrame,
    *,
    fwd_horizon: int = 5,
    step: int = 5,
    lookback_days: int = 300,
    regime: dict[date, str] | None = None,
    periods_per_year: int = 252,
) -> SignalHealth:
    """Replay ``strategy`` over history and measure its IC.

    Parameters
    ----------
    strategy
        Any object with ``.name`` and ``.on_bar(Snapshot) -> {sym: weight}``.
    full_bars
        Standard MultiIndex (symbol, timestamp) OHLCV frame.
    fwd_horizon
        Forward-return horizon in trading days over which the signal is
        scored (5 ≈ one week).
    step
        Spacing between evaluation dates in trading days (5 ≈ weekly).
    lookback_days
        How far back (calendar days) to evaluate.
    regime
        Optional {date: label} map; IC is additionally reported per label.
    """
    name = getattr(strategy, "name", strategy.__class__.__name__)

    # Wide close → forward returns the evaluator scores signals against.
    close = full_bars["close"].unstack(level=0).sort_index()
    # python date objects: Snapshot.from_full_bars and the regime map both
    # key on datetime.date, not pandas Timestamp.
    all_dates = [t.date() if hasattr(t, "date") else t for t in close.index]
    if len(all_dates) < fwd_horizon + 2:
        return SignalHealth(name, 0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            warnings=["insufficient history"])

    cutoff = all_dates[-1] - timedelta(days=lookback_days)
    # Evaluation dates: leave fwd_horizon bars of runway at the end.
    eval_idx = [
        i for i in range(0, len(all_dates) - fwd_horizon, step)
        if all_dates[i] >= cutoff
    ]

    ics: list[float] = []
    ic_dates: list[date] = []
    prev_weights: dict[str, float] | None = None
    turnovers: list[float] = []

    for i in eval_idx:
        as_of = all_dates[i]
        try:
            snap = Snapshot.from_full_bars(full_bars, as_of=as_of)
            weights = strategy.on_bar(snap)
        except Exception as e:  # a strategy must never crash the tracker
            logger.debug("signal_health: %s on_bar failed at %s: %s",
                         name, as_of, e)
            continue
        weights = {s: float(w) for s, w in weights.items()
                   if isinstance(w, (int, float))}
        if not weights:
            continue

        fwd = close.iloc[i + fwd_horizon] / close.iloc[i] - 1.0
        syms = [s for s in weights if s in fwd.index]
        if len(syms) < 5:
            continue
        ic = _spearman(
            np.array([weights[s] for s in syms]),
            np.array([fwd[s] for s in syms]),
        )
        if np.isfinite(ic):
            ics.append(ic)
            ic_dates.append(as_of)

        if prev_weights is not None:
            cur = set(weights)
            prev = set(prev_weights)
            union = cur | prev
            turnovers.append(len(cur ^ prev) / len(union) if union else 0.0)
        prev_weights = weights

    if not ics:
        return SignalHealth(name, 0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            warnings=["no IC observations computed"])

    ic_arr = np.array(ics)
    n = len(ic_arr)
    mean_ic = float(ic_arr.mean())
    std_ic = float(ic_arr.std(ddof=1)) if n > 1 else 0.0
    periods_per_yr = periods_per_year / step
    ic_ir = (mean_ic / std_ic * np.sqrt(periods_per_yr)) if std_ic > 0 else 0.0
    ic_tstat = (mean_ic / std_ic * np.sqrt(n)) if std_ic > 0 else 0.0
    hit = float((ic_arr > 0).mean())
    half = n // 2
    ic_early = float(ic_arr[:half].mean()) if half else mean_ic
    ic_recent = float(ic_arr[half:].mean()) if half else mean_ic

    regime_ic: dict[str, float] = {}
    if regime:
        by_label: dict[str, list[float]] = {}
        for d, ic in zip(ic_dates, ics, strict=False):
            label = regime.get(d)
            if label is not None:
                by_label.setdefault(label, []).append(ic)
        regime_ic = {k: float(np.mean(v)) for k, v in by_label.items() if v}

    warnings: list[str] = []
    if n < 20:
        warnings.append(f"only {n} IC periods; estimates are noisy")

    return SignalHealth(
        strategy_name=name,
        n_periods=n,
        mean_ic=mean_ic,
        ic_ir=ic_ir,
        ic_tstat=ic_tstat,
        hit_rate=hit,
        ic_recent=ic_recent,
        ic_early=ic_early,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        regime_ic=regime_ic,
        warnings=warnings,
    )
