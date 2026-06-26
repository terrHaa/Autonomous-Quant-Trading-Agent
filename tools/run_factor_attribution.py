"""run_factor_attribution.py — decompose the live book into alpha vs beta.

Run:  .venv/bin/python tools/run_factor_attribution.py

Builds the OHLCV factor panel over the cache's available history, then
regresses the live book's daily returns (from the run-record equity
curve) on it. Prints factor premia, the book's loadings, and its alpha.

This is the human-facing version of what Pillar 3 of the monthly review
consumes. With only ~1 month of live history the alpha t-stat is
directional (the engine flags this); the statistically powerful read
comes from attributing a full-window backtest, which the monthly does.
Read-only against the bars cache.
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
from quant.data.universe import load_active_universe
from quant.factors import attribute_returns, compute_factor_returns

TRADING_DAYS = 252


def _book_returns() -> pd.Series:
    eq: dict[pd.Timestamp, float] = {}
    for f in sorted(glob.glob("data/agent/runs/*.json")):
        d = json.load(open(f))
        e = d.get("execution_report", {}).get("account_equity_before")
        if e:
            eq[pd.Timestamp(d["date"])] = float(e)
    s = pd.Series(eq).sort_index().pct_change().dropna()
    s.index = pd.DatetimeIndex([t.date() for t in s.index])
    return s


def main(history_start: date = date(2023, 1, 1)) -> None:
    cache = BarsCache(client=AlpacaDataClient(), root=Path("data/bars/daily"))
    uni = load_active_universe(date.today() - timedelta(days=1))
    bars = cache.get_daily_bars(uni, history_start, date.today() - timedelta(days=1))
    fr = compute_factor_returns(bars)

    print(f"## Factor panel ({len(fr)} days, "
          f"{fr.index.min().date()} → {fr.index.max().date()})\n")
    print(f"{'factor':8s} {'ann.ret':>9s} {'ann.vol':>8s} {'Sharpe':>7s}")
    for c in fr.columns:
        r = fr[c]
        ann = r.mean() * TRADING_DAYS
        vol = r.std() * np.sqrt(TRADING_DAYS)
        print(f"{c:8s} {ann*100:8.1f}% {vol*100:7.1f}% "
              f"{ann/vol if vol > 0 else 0:7.2f}")

    book = _book_returns()
    print(f"\n## Book attribution (book history: {len(book)} days)\n")
    try:
        res = attribute_returns(book, fr)
        print(res.summary())
    except ValueError as e:
        print(f"(cannot attribute yet: {e})")


if __name__ == "__main__":
    main()
