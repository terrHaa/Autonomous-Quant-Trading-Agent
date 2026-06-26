"""regime_gate.py — backtest+DSR gate for auto-applying a regime policy.

The operator chose "auto-apply behind gates" for risk/allocation changes.
A regime policy (per-strategy, per-regime sleeve multipliers) is exactly
that kind of change — it re-risks the live book — so it cannot go live on
the analyst's say-so. This module is the gate:

  1. STRUCTURAL validity — known strategies, known regime labels,
     multipliers in [0, MAX_MULT], and not zeroing the whole book.
  2. BACKTEST — replay the ensemble over history with vs without the
     policy (regime-scaled sleeve weights each period) and require the
     policy's Sharpe to beat the static-HRP baseline by a margin.
  3. DSR — the policy's return series must clear a Deflated Sharpe Ratio
     threshold, with the trial count set to the number of multipliers
     the policy touches (the multiple-testing penalty for having
     searched across regimes/sleeves).

If all three pass, ``run_monthly_review`` persists the policy to
``EnsembleState.regime_policy`` and it takes effect on the next daily
run. If any fail, the policy is surfaced in the email as proposed-not-
applied with the failing reason — never silently dropped.

The backtest here models the ALLOCATION decision (relative sleeve
weighting), which is what's being gated: each strategy's own daily book
return, combined under static vs regime-scaled HRP weights. It
deliberately ignores costs/vol-target (those are unchanged by the
policy), so the comparison isolates the allocation effect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quant.backtest.types import Snapshot
from quant.risk.regime import REGIME_LABELS, apply_regime_policy, classify_regime

logger = logging.getLogger(__name__)

MAX_MULT = 1.5                 # cap a sleeve boost (mirrors the candidate-policy cap)
_MIN_SHARPE_IMPROVEMENT = 0.10  # policy must beat static HRP by this much (annualized)
_DSR_THRESHOLD = 0.60          # probabilistic floor (PSR-style, 0..1)
_STEP = 5                      # weekly evaluation cadence for the sim
_LOOKBACK_DAYS = 600


@dataclass(frozen=True)
class RegimePolicyGate:
    """Result of gating a proposed regime policy."""

    passed: bool
    reason: str
    base_sharpe: float = 0.0
    policy_sharpe: float = 0.0
    dsr: float = 0.0
    n_periods: int = 0

    def summary(self) -> str:
        verdict = "PASSED → auto-applied" if self.passed else "REJECTED"
        return (
            f"regime-policy gate: {verdict} — {self.reason} "
            f"(base Sharpe {self.base_sharpe:+.2f}, policy {self.policy_sharpe:+.2f}, "
            f"DSR {self.dsr:.2f}, n={self.n_periods})"
        )


def validate_regime_policy(
    policy: dict[str, dict[str, float]],
    strategy_names: set[str],
) -> tuple[bool, str]:
    """Structural validity check. Returns (ok, reason)."""
    if not policy:
        return False, "empty policy"
    for sname, regimes in policy.items():
        if sname not in strategy_names:
            return False, f"unknown strategy '{sname}'"
        if not isinstance(regimes, dict) or not regimes:
            return False, f"no regime entries for '{sname}'"
        for label, mult in regimes.items():
            if label not in REGIME_LABELS:
                return False, f"unknown regime label '{label}'"
            if not isinstance(mult, (int, float)) or not (0.0 <= mult <= MAX_MULT):
                return False, f"multiplier {mult} for {sname}/{label} out of [0,{MAX_MULT}]"
    return True, "structurally valid"


def _strategy_period_returns(
    strategies: list,
    full_bars: pd.DataFrame,
    *,
    step: int = _STEP,
    lookback_days: int = _LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, dict[date, str]]:
    """Replay each strategy → its own book's period returns + regime/date.

    Returns (R, regime_by_date) where R is index=eval-date,
    columns=strategy name, values=that strategy's realized return over the
    next ``step`` days (weights normalized to sum 1 within the strategy).
    """
    close = full_bars["close"].unstack(level=0).sort_index()
    dates = [t.date() if hasattr(t, "date") else t for t in close.index]
    if len(dates) < step + 2:
        return pd.DataFrame(), {}

    cutoff = dates[-1] - timedelta(days=lookback_days)
    eval_idx = [
        i for i in range(0, len(dates) - step, step) if dates[i] >= cutoff
    ]

    mkt = close.mean(axis=1)
    mkt.index = dates
    rows: dict[date, dict[str, float]] = {}
    regime_by_date: dict[date, str] = {}

    for i in eval_idx:
        as_of = dates[i]
        regime_by_date[as_of] = classify_regime(mkt.iloc[: i + 1])
        fwd = close.iloc[i + step] / close.iloc[i] - 1.0
        try:
            snap = Snapshot.from_full_bars(full_bars, as_of=as_of)
        except Exception:
            continue
        row: dict[str, float] = {}
        for strat in strategies:
            try:
                w = strat.on_bar(snap)
            except Exception:
                continue
            w = {s: float(v) for s, v in w.items()
                 if isinstance(v, (int, float)) and v > 0 and s in fwd.index}
            gross = sum(w.values())
            if gross <= 0:
                continue
            ret = sum((wi / gross) * float(fwd[s]) for s, wi in w.items()
                      if np.isfinite(fwd[s]))
            row[getattr(strat, "name", strat.__class__.__name__)] = ret
        if row:
            rows[as_of] = row

    return pd.DataFrame.from_dict(rows, orient="index"), regime_by_date


def gate_regime_policy(
    policy: dict[str, dict[str, float]],
    strategies: list,
    hrp_weights: dict[str, float],
    full_bars: pd.DataFrame,
    *,
    min_sharpe_improvement: float = _MIN_SHARPE_IMPROVEMENT,
    dsr_threshold: float = _DSR_THRESHOLD,
) -> RegimePolicyGate:
    """Backtest + DSR gate a regime policy. Never raises."""
    names = {getattr(s, "name", s.__class__.__name__) for s in strategies}
    ok, reason = validate_regime_policy(policy, names)
    if not ok:
        return RegimePolicyGate(False, reason)

    try:
        R, regimes = _strategy_period_returns(strategies, full_bars)
    except Exception as e:
        logger.exception("regime_gate: replay failed")
        return RegimePolicyGate(False, f"backtest replay failed: {type(e).__name__}: {e}")
    if R.empty or len(R) < 20:
        return RegimePolicyGate(False, f"insufficient backtest history (n={len(R)})")

    # Static-HRP baseline vs regime-scaled policy, period by period.
    hrp = {k: v for k, v in hrp_weights.items() if k in R.columns}
    base_ret, pol_ret = [], []
    for d, row in R.iterrows():
        avail = {s: hrp.get(s, 0.0) for s in R.columns if pd.notna(row[s])}
        if sum(avail.values()) <= 0:
            continue
        regime = regimes.get(d, "trend_up_calm")
        scaled = apply_regime_policy(avail, regime, policy)
        gb = sum(avail.values())
        gp = sum(scaled.values())
        base_ret.append(sum(avail[s] / gb * row[s] for s in avail))
        pol_ret.append(sum(scaled[s] / gp * row[s] for s in scaled))

    base = pd.Series(base_ret)
    pol = pd.Series(pol_ret)
    ppy = 252 / _STEP

    def _sharpe(x: pd.Series) -> float:
        return float(x.mean() / x.std() * np.sqrt(ppy)) if x.std() > 0 else 0.0

    base_sh, pol_sh = _sharpe(base), _sharpe(pol)

    from quant.evaluation.dsr import deflated_sharpe_ratio
    n_trials = max(1, sum(len(v) for v in policy.values()))
    try:
        dsr = float(deflated_sharpe_ratio(
            pol, n_trials=n_trials, var_sr_trials_annual=0.5,
            trading_days_per_year=int(ppy),
        ))
    except Exception as e:
        logger.warning("regime_gate: DSR failed (%s); treating as 0", e)
        dsr = 0.0

    improvement = pol_sh - base_sh
    if improvement < min_sharpe_improvement:
        return RegimePolicyGate(
            False, f"Sharpe improvement {improvement:+.2f} < {min_sharpe_improvement}",
            base_sh, pol_sh, dsr, len(pol),
        )
    if dsr < dsr_threshold:
        return RegimePolicyGate(
            False, f"DSR {dsr:.2f} < {dsr_threshold} (not significant after trials)",
            base_sh, pol_sh, dsr, len(pol),
        )
    return RegimePolicyGate(
        True, f"Sharpe {base_sh:+.2f}→{pol_sh:+.2f}, DSR {dsr:.2f}",
        base_sh, pol_sh, dsr, len(pol),
    )
