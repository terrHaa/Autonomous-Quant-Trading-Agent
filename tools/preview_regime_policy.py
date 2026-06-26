"""preview_regime_policy.py — what regime-aware allocation WOULD do.

Run:  .venv/bin/python tools/preview_regime_policy.py

Nothing here touches the live path. It (1) classifies today's market
regime, (2) derives a candidate regime policy from each strategy's
regime-conditional IC (signal-health tracker), and (3) shows how the
current HRP sleeve weights would be re-allocated under that policy.

This is the artifact to review BEFORE wiring regime-aware allocation
into live trading (which needs a backtest + operator sign-off).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quant.agent.ensemble import build_strategies, load_ensemble_state
from quant.data.alpaca_client import AlpacaDataClient
from quant.data.cache import BarsCache
from quant.data.universe import load_active_universe
from quant.evaluation.signal_health import compute_signal_health
from quant.risk.regime import apply_regime_policy, classify_regime


def main() -> None:
    cache = BarsCache(client=AlpacaDataClient(), root=Path("data/bars/daily"))
    uni = load_active_universe(date.today() - timedelta(days=1))
    bars = cache.get_daily_bars(uni, date(2024, 6, 1), date.today() - timedelta(days=1))

    mkt = bars["close"].unstack(level=0).mean(axis=1)
    mkt.index = [t.date() for t in mkt.index]
    regime = classify_regime(mkt)
    print(f"## Current market regime: {regime}\n")

    # Trend regime per date for the signal-health split.
    sma = mkt.rolling(200).mean()
    trend = {d: ("trend_up" if mkt[d] >= sma[d] else "trend_dn")
             for d in mkt.index if pd.notna(sma[d])}
    trend_now = "trend_up" if regime.startswith("trend_up") else "trend_dn"

    state = load_ensemble_state()
    strategies = build_strategies(state, uni)
    hrp = dict(state.hrp_weights)

    # Candidate policy: multiplier = clamp(regime_IC / overall_IC, 0..1.5),
    # so a sleeve dead in this regime gets dialed down, one that thrives
    # gets a (capped) boost.
    policy: dict[str, dict[str, float]] = {}
    print(f"{'strategy':36s} {'IC(now-regime)':>14s} {'mult':>6s}")
    for s in strategies:
        h = compute_signal_health(s, bars, regime=trend, lookback_days=600)
        ic_now = h.regime_ic.get(trend_now, h.mean_ic)
        base = abs(h.mean_ic) if abs(h.mean_ic) > 1e-6 else 1.0
        mult = max(0.0, min(1.5, ic_now / base)) if h.mean_ic > 0 else 0.0
        policy[s.name] = {regime: mult}
        print(f"{s.name:36s} {ic_now:+14.3f} {mult:6.2f}")

    new = apply_regime_policy(hrp, regime, policy)
    print(f"\n{'strategy':36s} {'HRP now':>8s} {'regime-aware':>12s}")
    for name in hrp:
        print(f"{name:36s} {hrp[name]:8.3f} {new[name]:12.3f}")


if __name__ == "__main__":
    main()
