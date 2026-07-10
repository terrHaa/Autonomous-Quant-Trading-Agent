"""ab_exec_timing.py — A/B: fill at the OPEN vs the CLOSE (evidence only).

Run:  .venv/bin/python tools/ab_exec_timing.py

The 2026-07 finding (tools measured it on 132 real fills): the live system
submits market orders at 9:35 ET, so momentum names that gapped up overnight
are bought at the inflated open — a median +0.44% above the prior close the
signal referenced. The question: would filling at the CLOSE instead (e.g.
market-on-close), where fill price == the signal's reference and liquidity
peaks, actually beat open execution NET — or do we give the saved gap back
by missing the overnight momentum continuation?

This replays the live submitted books both ways and scores the difference:

  arm A (open):  enter each name at day d0's OPEN, hold to d1's OPEN.
  arm B (close): enter at d0's prior CLOSE (== signal price, the frictionless
                 reference), hold close-to-close to d1.

Both pay the same one-way cost so the comparison isolates timing, not cost.
Paired A/B via the harness; both arms logged to the global trial ledger.

This is NOT a proposal to trade intraday — it only changes WHEN a daily
decision is executed. Read-only against cache + run records; writes only to
the ledger. STOP logic / signals unchanged; nothing live is touched.
"""

from __future__ import annotations

import glob
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data.alpaca_client import AlpacaDataClient
from quant.data.cache import BarsCache
from quant.research import TrialLedger, run_ab_test

ONE_WAY_COST = 0.0005   # 5 bps each side; identical for both arms


def _load_runs() -> list[dict]:
    out = []
    for f in sorted(glob.glob("data/agent/runs/*.json")):
        d = json.load(open(f))
        er = d.get("execution_report", {})
        if er.get("target_weights"):
            out.append(d)
    return out


def _wide(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    w = bars[field].unstack(level=0).sort_index()
    w.index = [t.date() if hasattr(t, "date") else t for t in w.index]
    return w


def _simulate(runs: list[dict], bars: pd.DataFrame, *, at: str) -> pd.Series:
    """Period returns entering/exiting at OPEN or CLOSE. ``at`` in {open, close}."""
    px = _wide(bars, at)
    days = list(px.index)
    run_dates = [date.fromisoformat(r["date"]) for r in runs]
    rets: dict[date, float] = {}
    for i, run in enumerate(runs):
        d0 = run_dates[i]
        d1 = run_dates[i + 1] if i + 1 < len(runs) else d0 + timedelta(days=7)
        weights = {s: w for s, w in
                   run["execution_report"]["target_weights"].items() if w > 0}
        if not weights or d0 not in px.index:
            continue
        # Entry day index; exit at the next rebalance's same price point.
        future = [d for d in days if d >= d1]
        if not future:
            continue
        d_exit = future[0]
        gross = sum(weights.values())
        r = 0.0
        for s, w in weights.items():
            if s not in px.columns:
                continue
            p0 = px.at[d0, s] if d0 in px.index else np.nan
            p1 = px.at[d_exit, s]
            if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
                continue
            r += (w / gross) * (p1 / p0 - 1.0 - 2 * ONE_WAY_COST)
        rets[d0] = r
    s = pd.Series(rets).sort_index()
    s.index = pd.DatetimeIndex([pd.Timestamp(d) for d in s.index])
    return s


def main() -> None:
    runs = _load_runs()
    print(f"replaying {len(runs)} live run records")
    syms = sorted({s for r in runs
                   for s in r["execution_report"]["target_weights"]})
    d0 = date.fromisoformat(runs[0]["date"]) - timedelta(days=5)
    d1 = date.fromisoformat(runs[-1]["date"]) + timedelta(days=10)
    cache = BarsCache(client=AlpacaDataClient(), root=Path("data/bars/daily"))
    bars = cache.get_daily_bars(syms, d0, d1)

    open_arm = _simulate(runs, bars, at="open")
    close_arm = _simulate(runs, bars, at="close")

    ledger = TrialLedger()
    # Baseline = current live behaviour (open); variant = close execution.
    res = run_ab_test(
        open_arm, close_arm,
        name="exec_timing_close_vs_open",
        ledger=ledger,
        family="execution/timing",
    )
    print()
    print(res.summary())
    print()
    ann = 252  # per-rebalance ~daily
    for label, arm in [("open (current)", open_arm), ("close (variant)", close_arm)]:
        if len(arm) > 1 and arm.std() > 0:
            sh = arm.mean() / arm.std() * np.sqrt(ann)
            print(f"  {label:16s} mean/day {arm.mean()*100:+.3f}%  Sharpe {sh:+.2f}")
    print()
    print("NB: evidence only — no live change. If close-execution wins, the "
          "implementation is market-on-close orders submitted at the current "
          "run time (fills at 16:00 ET; no schedule change).")


if __name__ == "__main__":
    main()
