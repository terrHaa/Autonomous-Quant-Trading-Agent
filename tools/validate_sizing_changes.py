"""validate_sizing_changes.py — evidence for the 2026-06 risk-allocation fix.

Run:  .venv/bin/python tools/validate_sizing_changes.py

Replays the NEW risk pipeline (inverse-vol sizing -> sector cap) against
the target weights actually emitted on each of the last N live run-dates,
and reports the before/after on the three things that were broken:

  - IT sector concentration  (the 30% cap was a no-op while the sector
    map only covered 50 of 519 names — see load_sector_map)
  - basket volatility        (signal-proportional sizing bet biggest on
    the most volatile names; corr(weight,vol) was +0.59)
  - implied deployment @12%   (vol-target had to crush gross to ~30%
    because basket vol was ~41%, leaving the book inert in cash)

This is a point-in-time replay on real emitted weights, not a full
order-by-order backtest — it isolates the risk-allocation change, which
is what the fix targets. Read-only against the bars cache.
"""

from __future__ import annotations

import glob
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant.agent.daily_runner import _apply_inverse_vol_sizing, _apply_sector_cap
from quant.data.alpaca_client import AlpacaDataClient
from quant.data.cache import BarsCache
from quant.data.universe import load_sector_map

TRADING_DAYS = 252
VOL_TARGET = 0.12


def _basket_vol(weights: dict[str, float], rdf: pd.DataFrame) -> float:
    ws = pd.Series({s: weights.get(s, 0.0) for s in rdf.columns})
    return float((rdf[ws.index] * ws).sum(axis=1).std() * np.sqrt(TRADING_DAYS))


def _it_pct(weights: dict[str, float], sm: dict[str, str]) -> float:
    gross = sum(v for v in weights.values() if v > 0)
    if gross <= 0:
        return 0.0
    it = sum(
        v for s, v in weights.items()
        if v > 0 and sm.get(s) == "Information Technology"
    )
    return it / gross * 100


def main(n_dates: int = 10) -> None:
    cache = BarsCache(client=AlpacaDataClient(), root=Path("data/bars/daily"))
    sm = load_sector_map()
    print(f"## Risk-allocation replay (last {n_dates} run-dates, "
          f"vol target {VOL_TARGET:.0%})\n")
    print(f"{'date':12s} | {'IT% old':>7s} {'IT% new':>7s} | "
          f"{'vol old':>7s} {'vol new':>7s} | {'deploy old':>10s} {'deploy new':>10s}")
    print("-" * 78)
    for f in sorted(glob.glob("data/agent/runs/*.json"))[-n_dates:]:
        d = json.load(open(f))
        tw = d.get("target_weights", {})
        dt = d.get("date")
        if not tw:
            continue
        end = date.fromisoformat(dt)
        start = end - timedelta(days=120)
        syms = [s for s in tw if tw[s] > 0]
        bars = cache.get_daily_bars(syms, start, end)
        new = _apply_sector_cap(_apply_inverse_vol_sizing(tw, bars), sm)
        rets = {}
        for s in syms:
            try:
                r = bars.loc[s]["close"].pct_change().dropna().tail(60)
                if len(r) >= 10:
                    rets[s] = r
            except (KeyError, ValueError, AttributeError):
                continue
        if len(rets) < 5:
            continue
        rdf = pd.concat(rets, axis=1).dropna(how="any")
        vo, vn = _basket_vol(tw, rdf), _basket_vol(new, rdf)
        print(f"{dt:12s} | {_it_pct(tw, sm):6.0f}% {_it_pct(new, sm):6.0f}% | "
              f"{vo*100:6.0f}% {vn*100:6.0f}% | "
              f"{VOL_TARGET/vo*100:9.0f}% {VOL_TARGET/vn*100:9.0f}%")


if __name__ == "__main__":
    main()
