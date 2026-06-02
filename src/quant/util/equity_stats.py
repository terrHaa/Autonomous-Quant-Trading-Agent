"""equity_stats.py — pure-Python equity-series statistics.

Shared core math used by both the weekly review and the monthly review
metric builders. Kept dependency-light (math + stdlib datetime) so it
can be imported from anywhere in the agent stack without pulling
numpy/pandas into the daily-trade hot path.

These functions are NOT a replacement for ``quant.evaluation.metrics``,
which operates on full ``BacktestResult`` objects and assumes a pandas
Series with risk-free de-annualization. The agent's review modules just
need a few headline numbers from a daily ``{date: equity}`` map, so we
keep this layer minimal and consistent across weekly + monthly.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any


def daily_returns(equity_curve: dict[date, float]) -> tuple[list[date], list[float]]:
    """From a date→equity map, return (dates_after_first, simple_returns).

    Each return is ``equity[t] / equity[t-1] - 1`` for adjacent days in
    sorted order. Days where the prior equity is non-positive are
    skipped (defensive — broker-state weirdness).
    """
    days = sorted(equity_curve)
    equities = [equity_curve[d] for d in days]
    rets: list[float] = []
    out_dates: list[date] = []
    for i in range(1, len(equities)):
        prev = equities[i - 1]
        if prev > 0:
            rets.append(equities[i] / prev - 1.0)
            out_dates.append(days[i])
    return out_dates, rets


def equity_series_stats(equity_curve: dict[date, float]) -> dict[str, Any]:
    """Compute the headline numbers used by every review email.

    Returns a dict with:
      - n_days, n_daily_returns
      - equity_start, equity_end
      - total_return_pct
      - ann_sharpe (annualized, 252-day, sample std)
      - ann_realized_vol_pct
      - max_drawdown_pct (negative, e.g. -3.2)
      - n_winning_days, n_losing_days, win_rate_pct

    If fewer than 2 equity observations exist, returns ``{insufficient_data: True}``.
    """
    if not equity_curve or len(equity_curve) < 2:
        return {
            "insufficient_data": True,
            "n_days": len(equity_curve),
        }

    days = sorted(equity_curve)
    equities = [equity_curve[d] for d in days]
    n = len(equities)

    _, rets = daily_returns(equity_curve)
    total_return = (equities[-1] / equities[0] - 1.0) if equities[0] > 0 else 0.0

    if len(rets) > 1:
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
        std = math.sqrt(var)
        ann_sharpe = (mean_r / std) * math.sqrt(252) if std > 0 else 0.0
        ann_vol = std * math.sqrt(252)
    else:
        ann_sharpe = 0.0
        ann_vol = 0.0

    # Max drawdown — walking peak, lowest peak-relative trough.
    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            max_dd = min(max_dd, (e - peak) / peak)

    n_pos = sum(1 for r in rets if r > 0)
    n_neg = sum(1 for r in rets if r < 0)
    win_rate = n_pos / len(rets) if rets else 0.0

    return {
        "n_days": n,
        "n_daily_returns": len(rets),
        "equity_start": round(equities[0], 2),
        "equity_end": round(equities[-1], 2),
        "total_return_pct": round(total_return * 100, 4),
        "ann_sharpe": round(ann_sharpe, 3),
        "ann_realized_vol_pct": round(ann_vol * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "n_winning_days": n_pos,
        "n_losing_days": n_neg,
        "win_rate_pct": round(win_rate * 100, 2),
    }


def top_movers(
    daily_runs: list[dict[str, Any]],
    n: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """From a list of daily run payloads, return (top_n_gainers, top_n_losers).

    Each result entry is ``{"symbol": str, "move_pct": float}`` ordered
    most-extreme first. The move is computed from ``signal_prices``
    snapshots at the FIRST and LAST runs in the period — captures the
    period's price action on names the agent actually rebalanced into.

    Empty lists when fewer than 2 runs (can't compute a move).
    """
    if len(daily_runs) < 2:
        return [], []
    sorted_runs = sorted(daily_runs, key=lambda r: r.get("date", ""))
    first_prices = sorted_runs[0].get("signal_prices", {})
    last_prices = sorted_runs[-1].get("signal_prices", {})
    moves: list[tuple[str, float]] = []
    for sym, p0 in first_prices.items():
        if sym in last_prices and p0 > 0:
            moves.append((sym, last_prices[sym] / p0 - 1.0))
    moves.sort(key=lambda x: x[1])
    losers = [{"symbol": s, "move_pct": round(m * 100, 2)} for s, m in moves[:n]]
    gainers = [{"symbol": s, "move_pct": round(m * 100, 2)} for s, m in moves[-n:][::-1]]
    return gainers, losers
