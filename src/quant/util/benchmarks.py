"""benchmarks.py — S&P 500 and Nasdaq close-to-close returns for the reports.

What this is for
----------------
A portfolio return of +1.3% over a week is meaningless without context.
Was the market up 3%? Down 2%? The reports answer that by surfacing
SPY (proxy for S&P 500) and QQQ (proxy for Nasdaq 100) returns over the
same period, alongside the portfolio's own return.

Why SPY / QQQ instead of the raw indices ^GSPC / ^IXIC
------------------------------------------------------
- Alpaca's data API reliably serves the ETFs (the most-traded names in
  the market) and may or may not serve the raw index symbols.
- ETF returns track the underlying indices within a few bps after
  management fees — close enough for "how did the market do this week".
- Both are universally recognised: any operator reading the email will
  know what SPY +0.5% means without thinking.

What this is NOT
----------------
Not a benchmark for SHARPE attribution (no excess-return calc, no
information-ratio computation). Just close-to-close percent returns
over a date window. The weekly / monthly analyst prompts can do
richer attribution if needed — this helper is the simple "headline
number" for the email reader.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date

from quant.data.cache import BarsProvider

logger = logging.getLogger(__name__)


# The two benchmarks every report surfaces. Order is significant — the
# reports render them in this order (SPY first, QQQ second) for a
# consistent visual.
BENCHMARK_TICKERS: tuple[str, ...] = ("SPY", "QQQ")


def fetch_benchmark_returns(
    cache: BarsProvider,
    start: date,
    end: date,
    *,
    tickers: Iterable[str] = BENCHMARK_TICKERS,
) -> dict[str, float]:
    """Return ``{ticker: pct_return}`` for each benchmark over [start, end].

    Uses close-to-close — the FIRST close in the window vs the LAST close.
    If the window only contains one bar (e.g. caller passed start == end
    and there's only one trading day), the benchmark is omitted from the
    result (you can't compute a return from one point). If the cache /
    underlying provider raises or returns empty for a ticker, that
    ticker is omitted with a warning and the others are returned —
    partial benchmark data is more useful than no benchmark data.

    Parameters
    ----------
    cache
        Any ``BarsProvider`` (typically the same ``BarsCache`` the
        trade routine uses, so SPY/QQQ get cached too and we don't
        re-fetch them every run).
    start, end
        Inclusive date window. Passed straight to ``get_daily_bars``.
    tickers
        Benchmark symbols to fetch. Defaults to ``("SPY", "QQQ")``;
        callers can override (e.g. tests).

    Returns
    -------
    dict[str, float]
        Maps each ticker (that we could fetch and price) to its
        close-to-close decimal return over the window. Tickers we
        couldn't price are simply absent — callers should treat
        absence as "no data" rather than zero.
    """
    tickers = list(tickers)
    out: dict[str, float] = {}
    try:
        bars = cache.get_daily_bars(tickers, start, end)
    except Exception as e:
        # Cache / provider failure should NOT break the report — degrade
        # to "no benchmark data" so the rest of the email still ships.
        logger.warning(
            "fetch_benchmark_returns: cache.get_daily_bars failed "
            "(%s: %s) — returning empty benchmark dict",
            type(e).__name__, e,
        )
        return out

    if bars.empty:
        logger.info(
            "fetch_benchmark_returns: no bars for %s in [%s, %s] — "
            "benchmarks will be omitted from this report",
            tickers, start, end,
        )
        return out

    for sym in tickers:
        try:
            closes = bars.loc[sym]["close"].dropna()
        except KeyError:
            logger.info(
                "fetch_benchmark_returns: no rows for %s in fetched bars "
                "(window=[%s, %s])", sym, start, end,
            )
            continue
        if len(closes) < 2:
            # Single bar can't yield a return. Skip silently — the
            # window genuinely doesn't contain two trading days.
            continue
        first = float(closes.iloc[0])
        last = float(closes.iloc[-1])
        if first <= 0:
            continue
        out[sym] = last / first - 1.0
    return out
