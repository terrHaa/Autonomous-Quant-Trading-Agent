"""ab_stop_policy.py — A/B: current tight stops vs wider stops (evidence only).

Run:  .venv/bin/python tools/ab_stop_policy.py

The July 2026 finding: the book submits ~44% gross but only ~16-20%
survives to the next morning — 26 stop fills in two sessions. Diagnosis:
ATR stops capped at STOP_LOSS_PCT=0.05 on names running 50-60% annualized
vol (3-4% daily moves) fire on ordinary noise, paying spread both ways
and un-deploying the book daily.

This experiment replays the ACTUAL submitted books from the live run
records forward on real OHLC bars under two stop policies:

  arm A (current): ATR-normalized stops, capped at 5%  (live policy)
  arm B (variant): 2x the ATR distance, capped at 10%

Mechanics per rebalance date: enter each name at its signal price with
its submitted weight; on each subsequent day until the next rebalance, a
name whose LOW crosses its stop exits at min(stop, open) — gap-downs fill
at the open, not the stop (no fantasy fills). Stop exits pay round-trip
costs (spread+slippage both ways, ~10bps on the stopped notional, since
the name is typically re-bought at the next rebalance). Exited names sit
in cash until the next rebalance. Both arms produce a daily return
series; the paired A/B harness scores the difference and logs both arms
to the global trial ledger.

IMPORTANT: STOP_LOSS_PCT=0.05 is an operator hard rule. This script
changes NOTHING live — it produces the evidence on which the operator
decides whether to relax the cap. Read-only against cache + run records;
writes only to the trial ledger.
"""

from __future__ import annotations

import glob
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant.agent.daily_runner import _compute_atr_normalized_stops
from quant.data.alpaca_client import AlpacaDataClient
from quant.data.cache import BarsCache
from quant.research import TrialLedger, run_ab_test

ROUND_TRIP_COST = 0.0010   # 10 bps: spread+slippage out, then back in next rebalance
WIDE_MULT = 2.0            # variant: 2x the ATR distance
WIDE_CAP = 0.10            # variant cap (vs live 0.05 hard rule)


def _load_runs() -> list[dict]:
    runs = []
    for f in sorted(glob.glob("data/agent/runs/*.json")):
        d = json.load(open(f))
        er = d.get("execution_report", {})
        if er.get("target_weights") and d.get("signal_prices"):
            runs.append(d)
    return runs


def _simulate(
    runs: list[dict],
    bars: pd.DataFrame,
    *,
    stop_mult: float,
    stop_cap: float,
) -> pd.Series:
    """Daily portfolio returns replaying the live books under a stop policy."""
    # Wide frames keyed by python date for O(1) day access.
    def _wide(field: str) -> pd.DataFrame:
        w = bars[field].unstack(level=0).sort_index()
        w.index = [t.date() if hasattr(t, "date") else t for t in w.index]
        return w

    lows, opens, closes = _wide("low"), _wide("open"), _wide("close")
    all_days = list(closes.index)

    daily_returns: dict[date, float] = {}
    run_dates = [date.fromisoformat(r["date"]) for r in runs]

    for i, run in enumerate(runs):
        d0 = run_dates[i]
        d1 = run_dates[i + 1] if i + 1 < len(runs) else d0 + timedelta(days=7)
        weights = {s: w for s, w in run["execution_report"]["target_weights"].items()
                   if w > 0}
        prices = run["signal_prices"]
        if not weights:
            continue

        # Stop distances per the arm's policy, from bars known at d0.
        cutoff = pd.Timestamp(d0)
        ts = bars.index.get_level_values("timestamp")
        hist = bars[ts.tz_localize(None) <= cutoff if ts.tz is not None else ts <= cutoff]
        base = _compute_atr_normalized_stops(symbols=list(weights), bars=hist)
        stops = {
            s: min(dist * stop_mult, stop_cap) for s, dist in base.items()
        }

        entry = {s: prices.get(s) for s in weights}
        stop_px = {
            s: entry[s] * (1 - stops.get(s, stop_cap))
            for s in weights if entry.get(s)
        }
        alive = {s for s in weights if entry.get(s)}
        prev_close = dict(entry)

        # d0 itself is included: entry fills at the 9:35 open of d0, so
        # d0's own low can stop us out same-day (that IS the churn story).
        window = [d for d in all_days if d0 <= d < d1]
        for d in window:
            day_ret = 0.0
            for s in list(alive):
                lo = lows.at[d, s] if s in lows.columns and d in lows.index else None
                op = opens.at[d, s] if s in opens.columns else None
                cl = closes.at[d, s] if s in closes.columns else None
                if lo is None or cl is None or np.isnan(lo) or np.isnan(cl):
                    continue
                w = weights[s]
                if lo <= stop_px[s]:
                    # Stopped: gap-aware fill, pay the round trip, go to cash.
                    fill = min(stop_px[s], op if op and not np.isnan(op) else stop_px[s])
                    day_ret += w * (fill / prev_close[s] - 1.0 - ROUND_TRIP_COST)
                    alive.discard(s)
                else:
                    day_ret += w * (cl / prev_close[s] - 1.0)
                    prev_close[s] = cl
            if d in daily_returns:
                daily_returns[d] += day_ret
            else:
                daily_returns[d] = day_ret

    s = pd.Series(daily_returns).sort_index()
    s.index = pd.DatetimeIndex([pd.Timestamp(d) for d in s.index])
    return s


def main() -> None:
    runs = _load_runs()
    print(f"replaying {len(runs)} live run records")
    syms = sorted({s for r in runs for s in r["execution_report"]["target_weights"]})
    d_first = date.fromisoformat(runs[0]["date"]) - timedelta(days=60)
    d_last = date.fromisoformat(runs[-1]["date"]) + timedelta(days=10)
    cache = BarsCache(client=AlpacaDataClient(), root=Path("data/bars/daily"))
    bars = cache.get_daily_bars(syms, d_first, d_last)

    current = _simulate(runs, bars, stop_mult=1.0, stop_cap=0.05)
    wider = _simulate(runs, bars, stop_mult=WIDE_MULT, stop_cap=WIDE_CAP)

    ledger = TrialLedger()
    res = run_ab_test(
        current, wider,
        name="stop_policy_2xATR_cap10",
        ledger=ledger,
        family="risk/stops",
    )
    print()
    print(res.summary())
    print()
    print(f"(ledger now has {ledger.n_trials()} trials; "
          "STOP_LOSS_PCT unchanged — operator decides on this evidence)")


if __name__ == "__main__":
    main()
