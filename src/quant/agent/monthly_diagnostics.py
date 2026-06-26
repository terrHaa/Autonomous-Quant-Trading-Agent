"""monthly_diagnostics.py — the quant-instrument bundle for the monthly review.

Phase 4 wiring. Runs the three analytical engines built in Phases 1-3 and
packages their output into one structured dict the monthly review both
renders in the email and feeds to the AI analyst. This is what turns the
monthly from narrative guesswork into a numbers-grounded review:

  - Pillar 3 (alpha vs beta): factor attribution of the book.
  - Pillar 1 (signal health):  per-strategy IC, decay, regime split.
  - Pillar 4 (risk/regime):    current regime, correlation de-gross
    signal, and a CANDIDATE regime policy derived from regime-
    conditional IC — the thing the monthly can adopt (behind its
    backtest+shadow+DSR gates) into EnsembleState.regime_policy.

Everything is wrapped so a single engine failure (bad data, short
history) degrades to a noted gap rather than blocking the monthly email.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_FACTOR_HISTORY_START = date(2023, 1, 1)
_SIGNAL_HEALTH_LOOKBACK = 600


def _book_returns(equity_curve: dict[date, float]) -> pd.Series:
    if not equity_curve:
        return pd.Series(dtype=float)
    s = pd.Series(equity_curve).sort_index().pct_change().dropna()
    s.index = pd.DatetimeIndex([pd.Timestamp(d) for d in s.index])
    s.index = [t.date() for t in s.index]
    return s


def _candidate_regime_policy(
    health_by_strategy: dict[str, Any],
    regime_now: str,
    trend_now: str,
) -> dict[str, dict[str, float]]:
    """Derive {strategy: {regime_now: multiplier}} from regime-conditional IC.

    Multiplier = clamp(IC_in_current_regime / |mean_IC|, 0..1.5): a sleeve
    whose edge is absent in this regime gets dialed down; one that thrives
    gets a capped boost. Strategies with non-positive mean IC are zeroed.
    """
    policy: dict[str, dict[str, float]] = {}
    for name, h in health_by_strategy.items():
        if not isinstance(h, dict):
            continue
        mean_ic = h.get("mean_ic")
        if mean_ic is None or mean_ic <= 0:
            policy[name] = {regime_now: 0.0}
            continue
        ic_now = h.get("regime_ic", {}).get(trend_now, mean_ic)
        mult = max(0.0, min(1.5, ic_now / abs(mean_ic)))
        policy[name] = {regime_now: round(mult, 3)}
    return policy


def build_quant_diagnostics(
    *,
    equity_curve: dict[date, float],
    state,
    universe: list[str],
    cache,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Run the factor / signal-health / regime engines; return one dict.

    Never raises — each section is independently guarded and reports
    ``{"error": ...}`` on failure so the monthly review can render what
    succeeded.
    """
    as_of = as_of or date.today()
    out: dict[str, Any] = {"as_of": as_of.isoformat()}

    try:
        bars = cache.get_daily_bars(universe, _FACTOR_HISTORY_START, as_of)
    except Exception as e:
        logger.exception("monthly_diagnostics: bars fetch failed")
        return {**out, "error": f"bars fetch failed: {type(e).__name__}: {e}"}

    # --- Pillar 3: factor attribution (alpha vs beta) ---
    try:
        from quant.factors import attribute_returns, compute_factor_returns
        fr = compute_factor_returns(bars)
        out["factor_premia"] = {
            c: {
                "ann_return": round(float(fr[c].mean() * 252), 4),
                "sharpe": round(float(fr[c].mean() / fr[c].std() * (252**0.5)), 2)
                if fr[c].std() > 0 else 0.0,
            }
            for c in fr.columns
        }
        book = _book_returns(equity_curve)
        try:
            res = attribute_returns(book, fr)
            out["attribution"] = {
                "alpha_annual": round(res.alpha_annual, 4),
                "alpha_tstat": round(res.alpha_tstat, 2),
                "betas": {k: round(v, 2) for k, v in res.betas.items()},
                "r_squared": round(res.r_squared, 2),
                "n_obs": res.n_obs,
                "warnings": res.warnings,
            }
        except ValueError as e:
            out["attribution"] = {"error": str(e)}
    except Exception as e:
        logger.exception("monthly_diagnostics: factor attribution failed")
        out["factor_premia"] = {"error": f"{type(e).__name__}: {e}"}

    # --- regime context (shared by Pillars 1 + 4) ---
    trend_map: dict[date, str] = {}
    regime_now = "trend_up_calm"
    try:
        from quant.risk.regime import (
            average_pairwise_correlation,
            classify_regime,
            correlation_degross_factor,
        )
        wide = bars["close"].unstack(level=0)
        wide.index = [t.date() if hasattr(t, "date") else t for t in wide.index]
        mkt = wide.mean(axis=1)
        regime_now = classify_regime(mkt)
        sma = mkt.rolling(200).mean()
        trend_map = {
            d: ("trend_up" if mkt[d] >= sma[d] else "trend_dn")
            for d in mkt.index if pd.notna(sma[d])
        }
        avg_corr = average_pairwise_correlation(wide.pct_change())
        out["regime"] = {
            "current": regime_now,
            "avg_pairwise_corr": round(avg_corr, 3) if avg_corr == avg_corr else None,
            "correlation_degross_factor": round(correlation_degross_factor(avg_corr), 3),
        }
    except Exception as e:
        logger.exception("monthly_diagnostics: regime classification failed")
        out["regime"] = {"error": f"{type(e).__name__}: {e}"}

    # --- Pillar 1: per-strategy signal health ---
    health_dict: dict[str, Any] = {}
    try:
        from quant.agent.ensemble import build_strategies
        from quant.evaluation.signal_health import compute_signal_health
        for s in build_strategies(state, universe):
            try:
                h = compute_signal_health(
                    s, bars, regime=trend_map or None,
                    lookback_days=_SIGNAL_HEALTH_LOOKBACK,
                )
                health_dict[h.strategy_name] = {
                    "mean_ic": round(h.mean_ic, 4) if h.mean_ic == h.mean_ic else None,
                    "ic_ir": round(h.ic_ir, 2),
                    "ic_tstat": round(h.ic_tstat, 2),
                    "hit_rate": round(h.hit_rate, 2),
                    "ic_early": round(h.ic_early, 4),
                    "ic_recent": round(h.ic_recent, 4),
                    "decaying": h.decaying,
                    "turnover": round(h.avg_turnover, 2),
                    "regime_ic": {k: round(v, 3) for k, v in h.regime_ic.items()},
                    "n_periods": h.n_periods,
                }
            except Exception as e:  # one strategy must not kill the rest
                health_dict[getattr(s, "name", "?")] = {"error": str(e)}
        out["signal_health"] = health_dict
    except Exception as e:
        logger.exception("monthly_diagnostics: signal health failed")
        out["signal_health"] = {"error": f"{type(e).__name__}: {e}"}

    # --- Pillar 4: candidate regime policy (auto-apply target) ---
    trend_now = "trend_up" if regime_now.startswith("trend_up") else "trend_dn"
    out["candidate_regime_policy"] = _candidate_regime_policy(
        health_dict, regime_now, trend_now,
    )
    return out


def render_diagnostics_md(diag: dict[str, Any]) -> str:
    """Markdown block for the monthly email — the comprehensive scorecard."""
    lines = ["## Quant diagnostics (alpha/beta · signal health · regime)", ""]

    att = diag.get("attribution", {})
    if "error" not in att and att:
        warn = "  ⚠ " + "; ".join(att.get("warnings", [])) if att.get("warnings") else ""
        lines.append(
            f"**Alpha vs beta:** alpha {att.get('alpha_annual', 0):+.1%}/yr "
            f"(t={att.get('alpha_tstat', 0):+.2f}), R²={att.get('r_squared', 0):.2f}, "
            f"n={att.get('n_obs', 0)}.{warn}"
        )
        betas = att.get("betas", {})
        if betas:
            lines.append("**Factor loadings:** "
                         + ", ".join(f"{k} {v:+.2f}" for k, v in betas.items()))
        lines.append("")

    sh = diag.get("signal_health", {})
    if isinstance(sh, dict) and "error" not in sh and sh:
        lines.append("**Per-strategy signal health (IC):**")
        lines.append("")
        lines.append("| Strategy | mean IC | decay | turnover | regime IC |")
        lines.append("|---|---|---|---|---|")
        for name, h in sh.items():
            if not isinstance(h, dict) or "error" in h:
                continue
            reg = " ".join(f"{k}={v:+.2f}" for k, v in h.get("regime_ic", {}).items())
            decay = (f"{h.get('ic_early', 0):+.3f}→{h.get('ic_recent', 0):+.3f}"
                     + (" ⚠" if h.get("decaying") else ""))
            lines.append(
                f"| `{name}` | {h.get('mean_ic', 0):+.3f} | {decay} "
                f"| {h.get('turnover', 0):.0%} | {reg} |"
            )
        lines.append("")

    reg = diag.get("regime", {})
    if "error" not in reg and reg:
        lines.append(
            f"**Regime:** {reg.get('current')} · avg pairwise corr "
            f"{reg.get('avg_pairwise_corr')} · de-gross factor "
            f"{reg.get('correlation_degross_factor')}"
        )
    pol = diag.get("candidate_regime_policy", {})
    if pol:
        lines.append("**Candidate regime policy (sleeve multipliers this regime):** "
                     + ", ".join(
                         f"`{k}`×{list(v.values())[0]}" for k, v in pol.items() if v
                     ))
    return "\n".join(lines)
