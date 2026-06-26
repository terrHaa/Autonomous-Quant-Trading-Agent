"""library.py — daily long-short factor returns built from OHLCV alone.

Each factor is a dollar-neutral, equal-weighted long-short portfolio
rebalanced daily: long the top decile by the factor signal, short the
bottom decile. The returned series is the daily return of that spread —
i.e. the premium an investor would have earned bearing that factor.

Factors (all computable from price/volume — NO fundamentals needed):
  - MKT    : equal-weighted universe return (market proxy / beta). This
             is the one LONG-ONLY factor (not a spread) — it's the
             ambient market the long-only book is mechanically exposed to.
  - MOM    : cross-sectional 12-1 momentum (Jegadeesh-Titman). Rank by
             return from t-252 to t-21 (skip the most recent month to
             avoid the short-term reversal contamination).
  - STR    : short-term reversal (Lehmann). Rank by the past-5-day
             return; LONG the losers, SHORT the winners.
  - LOWVOL : betting-against-volatility (Frazzini-Pedersen flavour).
             Rank by trailing 60-day realized vol; LONG low, SHORT high.

What is deliberately NOT here: Value (HML), Size (SMB), Quality/
Profitability (RMW), Investment (CMA). Those need a fundamentals feed
(book value, market cap, earnings) that the platform doesn't ingest yet.
Adding them is a data-infrastructure task, called out so the attribution
consumer knows the model is a 4-factor OHLCV model, not full FF5+UMD.

No look-ahead: every signal is formed from data through day ``t-1`` and
applied to day ``t``'s realized returns (the code shifts signals by one
bar before multiplying into forward returns).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Order is the canonical column order of the returned DataFrame.
FACTOR_NAMES: tuple[str, ...] = ("MKT", "MOM", "STR", "LOWVOL")

# Construction knobs. Defaults match the standard academic conventions.
_MOM_LOOKBACK = 252      # ~12 months
_MOM_SKIP = 21           # skip most-recent ~1 month (reversal guard)
_STR_LOOKBACK = 5        # ~1 week
_LOWVOL_LOOKBACK = 60    # ~3 months realized vol
_DECILE = 0.10           # long/short the top/bottom 10% by signal
_MIN_NAMES = 20          # need a minimum cross-section to form deciles


def _wide_close_returns(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (wide_close, daily_returns) as date × symbol frames.

    ``bars`` is the platform's standard MultiIndex (symbol, timestamp)
    OHLCV frame. We pivot ``close`` to a wide matrix and compute simple
    daily returns.
    """
    close = bars["close"].copy()
    # Index levels are (symbol, timestamp); unstack symbols to columns.
    wide = close.unstack(level=0).sort_index()
    wide.index = pd.DatetimeIndex(
        [ts.date() if hasattr(ts, "date") else ts for ts in wide.index]
    )
    returns = wide.pct_change()
    return wide, returns


def _long_short(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    *,
    quantile: float = _DECILE,
) -> pd.Series:
    """Daily return of a long-top / short-bottom decile spread.

    ``signal`` higher = more attractive (goes long). ``signal`` and
    ``fwd_returns`` are aligned date × symbol frames; the caller is
    responsible for having shifted ``signal`` by one bar so there's no
    look-ahead (signal known at t-1, return realized at t).
    """
    out: dict[pd.Timestamp, float] = {}
    sig = signal.to_numpy()
    rets = fwd_returns.to_numpy()
    for i, day in enumerate(signal.index):
        s = sig[i]
        r = rets[i]
        mask = np.isfinite(s) & np.isfinite(r)
        n = int(mask.sum())
        if n < _MIN_NAMES:
            continue
        s_valid = s[mask]
        r_valid = r[mask]
        k = max(1, int(np.floor(n * quantile)))
        order = np.argsort(s_valid)
        short_idx = order[:k]          # lowest signal → short
        long_idx = order[-k:]          # highest signal → long
        out[day] = float(r_valid[long_idx].mean() - r_valid[short_idx].mean())
    return pd.Series(out, name="ls").sort_index()


def compute_factor_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Build the daily factor-return panel from an OHLCV bars frame.

    Returns a DataFrame indexed by date with columns ``FACTOR_NAMES``.
    Rows are dropped where no factor could be computed (insufficient
    cross-section or history). The first ``_MOM_LOOKBACK`` days are
    naturally NaN for MOM and are dropped from the final intersection.
    """
    if bars is None or bars.empty:
        return pd.DataFrame(columns=list(FACTOR_NAMES))

    wide, returns = _wide_close_returns(bars)

    # MKT: equal-weighted cross-sectional mean return (long-only market).
    mkt = returns.mean(axis=1, skipna=True).rename("MKT")

    # MOM: 12-1 momentum. Signal known at t-1 (shift to avoid look-ahead).
    mom_signal = wide.shift(_MOM_SKIP) / wide.shift(_MOM_LOOKBACK) - 1.0
    mom = _long_short(mom_signal.shift(1), returns).rename("MOM")

    # STR: short-term reversal. Long losers → signal = -(past 5d return).
    str_signal = -(wide / wide.shift(_STR_LOOKBACK) - 1.0)
    str_ = _long_short(str_signal.shift(1), returns).rename("STR")

    # LOWVOL: long low realized vol → signal = -(trailing 60d vol).
    lowvol_signal = -returns.rolling(_LOWVOL_LOOKBACK).std()
    lowvol = _long_short(lowvol_signal.shift(1), returns).rename("LOWVOL")

    panel = pd.concat([mkt, mom, str_, lowvol], axis=1)
    panel = panel[list(FACTOR_NAMES)].dropna(how="any")
    logger.info(
        "compute_factor_returns: %d factor-days (%s → %s)",
        len(panel),
        panel.index.min() if len(panel) else "—",
        panel.index.max() if len(panel) else "—",
    )
    return panel
