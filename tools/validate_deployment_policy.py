"""validate_deployment_policy.py — evidence for the 2026-06-12 deployment policy.

Run:  .venv/bin/python tools/validate_deployment_policy.py

Two studies, printed as markdown:

1. **SPY regime-filter study (historical)** — as many years of SPY daily
   closes as Alpaca will serve. Compares buy-and-hold vs the live policy
   (100% above the 200d SMA, 50% below, remainder in cash at 0%) on
   CAGR / vol / Sharpe / max drawdown / worst year. The filter's claim
   is drawdown compression at small return cost — verify on OUR data
   feed, not a textbook table.

2. **Drawdown-ladder Monte Carlo** — 10,000 simulated years of a
   12%-vol book at several assumed Sharpes, with and without the
   graduated ladder (-5%→75%, -10%→50%, -12.5%→25%). Reports the
   probability of touching the -15% kill line within a year. This is
   the number that justifies raising vol_target_annual to 0.12: the
   ladder must cut P(kill) below what a 10%-vol unladdered book had.

Read-only against the bars cache (it may extend cached SPY history —
that's what a cache is for). No broker mutations.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant.risk.deployment import DRAWDOWN_LADDER, drawdown_scale

TRADING_DAYS = 252


def _stats(daily_returns: pd.Series, label: str) -> dict:
    eq = (1 + daily_returns).cumprod()
    years = len(daily_returns) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / years) - 1)
    vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS))
    max_dd = float((eq / eq.cummax() - 1).min())
    worst_year = float(
        daily_returns.groupby(daily_returns.index.year)
        .apply(lambda r: (1 + r).prod() - 1)
        .min()
    )
    return {
        "label": label, "cagr": cagr, "vol": vol, "sharpe": sharpe,
        "max_dd": max_dd, "worst_year": worst_year,
    }


def spy_regime_study() -> list[dict]:
    from quant.data.alpaca_client import AlpacaDataClient
    from quant.data.cache import BarsCache

    cache = BarsCache(client=AlpacaDataClient(), root=Path("data/bars/daily"))
    end = date.today() - timedelta(days=1)
    start = date(2016, 1, 1)   # ask for ~10y; feed serves what it has
    bars = cache.get_daily_bars(["SPY"], start, end)
    closes = bars.loc["SPY"]["close"]
    closes.index = pd.to_datetime(closes.index)

    rets = closes.pct_change().dropna()
    sma = closes.rolling(200).mean()
    # Yesterday's regime decides today's exposure — no look-ahead.
    exposure = (closes >= sma).map({True: 1.0, False: 0.5}).shift(1)
    valid = exposure.notna() & sma.shift(1).notna()
    rets, exposure = rets[valid.reindex(rets.index, fill_value=False)], exposure
    filtered = rets * exposure.reindex(rets.index)

    n_years = len(rets) / TRADING_DAYS
    pct_risk_off = float((exposure.reindex(rets.index) < 1.0).mean())
    print(f"\n## Study 1 — SPY regime filter ({rets.index[0].date()} → "
          f"{rets.index[-1].date()}, {n_years:.1f}y, "
          f"{pct_risk_off:.0%} of days risk-off)\n")
    rows = [_stats(rets, "SPY buy & hold"),
            _stats(filtered, "SPY + 200d filter (100/50)")]
    print("| Book | CAGR | Vol | Sharpe | Max DD | Worst yr |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['cagr']:+.1%} | {r['vol']:.1%} "
              f"| {r['sharpe']:.2f} | {r['max_dd']:.1%} | {r['worst_year']:+.1%} |")
    return rows


def ladder_monte_carlo(
    *, vol_annual: float = 0.12, n_sims: int = 10_000, seed: int = 7,
) -> None:
    rng = np.random.default_rng(seed)
    daily_vol = vol_annual / np.sqrt(TRADING_DAYS)
    print(f"\n## Study 2 — drawdown-ladder Monte Carlo "
          f"({n_sims:,} years, {vol_annual:.0%}-vol book, ladder "
          f"{[f'{t:.1%}→×{s}' for t, s in DRAWDOWN_LADDER]})\n")
    print("| Assumed Sharpe | P(touch -15%) no ladder | with ladder | "
          "median ladder max-DD |")
    print("|---|---|---|---|")
    for sharpe in (0.0, 0.5, 1.0, 1.5):
        daily_mu = sharpe * vol_annual / TRADING_DAYS
        z = rng.standard_normal((n_sims, TRADING_DAYS))
        raw = daily_mu + daily_vol * z
        # No ladder: straight cumulative path.
        eq = np.cumprod(1 + raw, axis=1)
        dd = eq / np.maximum.accumulate(eq, axis=1) - 1
        p_raw = float((dd.min(axis=1) <= -0.15).mean())
        # Ladder: day-by-day, yesterday's drawdown sets today's size.
        eq_l = np.ones(n_sims)
        peak = np.ones(n_sims)
        min_dd = np.zeros(n_sims)
        scale = np.ones(n_sims)
        for t in range(TRADING_DAYS):
            eq_l = eq_l * (1 + scale * raw[:, t])
            peak = np.maximum(peak, eq_l)
            cur_dd = eq_l / peak - 1
            min_dd = np.minimum(min_dd, cur_dd)
            scale = np.fromiter(
                (drawdown_scale(d) for d in cur_dd), dtype=float, count=n_sims,
            )
        p_ladder = float((min_dd <= -0.15).mean())
        print(f"| {sharpe:.1f} | {p_raw:.1%} | {p_ladder:.1%} "
              f"| {np.median(min_dd):.1%} |")


if __name__ == "__main__":
    spy_regime_study()
    ladder_monte_carlo()
