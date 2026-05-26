"""alpaca_client.py — thin wrapper around alpaca-py's historical market-data API.

Why a wrapper at all? Two reasons:

1. **Isolation.** Nothing else in the codebase imports `alpaca` directly. If
   we ever switch data providers (Polygon, IEX direct, Yahoo for free, ...),
   only this file changes — the cache, the backtest engine, and strategies
   keep their existing imports.

2. **Shape control.** We promise downstream code a specific DataFrame shape
   (MultiIndex of (symbol, timestamp), columns = open/high/low/close/volume).
   Provider quirks (extra columns, weird naming, raw vs adjusted) get
   normalized here once, not in every consumer.

A note on credentials: Alpaca's historical data API uses the same key/secret
for both paper and live accounts — *market data is just market data*. The
paper/live distinction matters for the trading API (where orders go), not
for the data API. We default to loading the paper key from `.env` because
that's what's set up; you could pass live keys instead and get identical
results from this module.

A note on the free tier: Alpaca's free historical data is the IEX feed,
covering ~3% of US equity volume. For daily bars on S&P 500 names it's
close enough to consolidated tape to be useful. For accurate intraday
backtests you'd want a paid subscription (SIP feed) — out of scope for now.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

# The exact columns and order we promise to return. Downstream code (cache,
# integrity checks, backtest engine) can rely on this contract — it's part
# of the public interface of this module, not an implementation detail.
BAR_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class AlpacaCredentials:
    """An API key + secret pair.

    Frozen (immutable) so credentials can't be silently mutated in flight.
    If you need different creds, build a new object — don't reassign fields.
    """

    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls, env: str = "paper") -> AlpacaCredentials:
        """Load credentials from environment variables.

        Reads `.env` if present (via python-dotenv), then picks
        ALPACA_PAPER_API_KEY / _SECRET or ALPACA_LIVE_API_KEY / _SECRET
        based on the `env` argument.

        Parameters
        ----------
        env
            Either "paper" (default) or "live".

        Raises
        ------
        RuntimeError
            If the requested env's key or secret is missing. The error
            message tells you exactly which variable to set, because
            "Alpaca rejected my key" is a much less helpful error than
            "you forgot to fill in your .env".
        """
        # `load_dotenv()` is a no-op if `.env` was already loaded earlier in
        # the process, so it's safe to call from every entry point.
        load_dotenv()

        prefix = f"ALPACA_{env.upper()}"
        key = os.environ.get(f"{prefix}_API_KEY")
        secret = os.environ.get(f"{prefix}_API_SECRET")
        if not key or not secret:
            raise RuntimeError(
                f"Missing {prefix}_API_KEY or {prefix}_API_SECRET in environment. "
                f"Did you copy .env.example to .env and fill it in?"
            )
        return cls(api_key=key, api_secret=secret)


class AlpacaDataClient:
    """Historical OHLCV bar fetcher.

    Construct with no arguments to use the paper keys from `.env`:

        >>> client = AlpacaDataClient()
        >>> df = client.get_daily_bars(["AAPL", "MSFT"], date(2024,1,2), date(2024,1,10))

    Or pass explicit credentials:

        >>> creds = AlpacaCredentials.from_env(env="live")
        >>> client = AlpacaDataClient(creds)
    """

    def __init__(self, credentials: AlpacaCredentials | None = None) -> None:
        # Default to whatever's in .env if the caller didn't pass anything.
        # This is a convenience for interactive use; production code should
        # pass credentials explicitly so the source is obvious.
        creds = credentials or AlpacaCredentials.from_env()
        self._client = StockHistoricalDataClient(
            api_key=creds.api_key,
            secret_key=creds.api_secret,
        )

    def get_daily_bars(
        self,
        symbols: Iterable[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for the given symbols and date range.

        Parameters
        ----------
        symbols
            Iterable of ticker strings (e.g. ``["AAPL", "MSFT"]``). Case is
            normalized to upper; duplicates are dropped while preserving order.
        start
            First date to fetch (inclusive).
        end
            Last date to fetch (inclusive).

        Returns
        -------
        pandas.DataFrame
            MultiIndex of (symbol, timestamp), columns = open, high, low,
            close, volume. Empty if no data is available for any symbol in
            the requested window.

            To pull one symbol's slice:  ``df.loc["AAPL"]``
            To get the set of symbols:   ``df.index.get_level_values("symbol").unique()``

        Notes
        -----
        Prices are *split- and dividend-adjusted* (``adjustment="all"``).
        Using raw unadjusted prices in a backtest produces nonsense returns
        on any corporate-action day — you'd see a "30% drop" that was
        actually a 2:1 split. Always research on adjusted prices.
        """
        # Normalize symbols: uppercase, dedupe in order. `dict.fromkeys` is a
        # one-line trick to dedupe while preserving insertion order (since
        # Python 3.7 dicts are insertion-ordered).
        symbols = list(dict.fromkeys(s.upper() for s in symbols))
        if not symbols:
            return _empty_bars_frame()

        # Alpaca wants timezone-aware datetimes. We anchor `start` to UTC
        # midnight and `end` to UTC end-of-day, so the half-open vs closed
        # semantics on Alpaca's side don't cost us a day at either edge.
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime.combine(start, datetime.min.time(), tzinfo=UTC),
            end=datetime.combine(end, datetime.max.time(), tzinfo=UTC),
            # 'all' = adjust for both splits and dividends. The other options
            # are 'raw', 'split', and 'dividend' — we don't use them, but if
            # you ever need raw prices for corporate-action analysis, this is
            # the knob.
            adjustment="all",
        )

        # The SDK returns a BarSet object with a `.df` property that hands
        # back a MultiIndex DataFrame already shaped (symbol, timestamp).
        barset = self._client.get_stock_bars(request)
        df = barset.df

        if df.empty:
            return _empty_bars_frame()

        # Trim to the promised column contract. Alpaca returns extras
        # (trade_count, vwap) that aren't part of our interface — if a
        # downstream consumer wants them, they belong in a separate method.
        # `.copy()` so callers can mutate the result without affecting the
        # SDK's internal state.
        return df[list(BAR_COLUMNS)].copy()


def _empty_bars_frame() -> pd.DataFrame:
    """Return a properly-shaped empty bars DataFrame.

    Having a well-shaped empty frame means downstream code can do things like
    ``df.loc["AAPL"]`` (which returns its own empty frame) without crashing
    on the "no data" path differently than on the "no AAPL data" path.
    """
    return pd.DataFrame(
        columns=list(BAR_COLUMNS),
        index=pd.MultiIndex.from_arrays(
            [[], []],
            names=["symbol", "timestamp"],
        ),
    )
