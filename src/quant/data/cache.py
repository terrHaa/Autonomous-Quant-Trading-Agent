"""cache.py — Parquet-backed cache around a bars provider.

Why this exists
---------------
Every backtest re-reads the same historical OHLCV data, often hundreds of
times across a research session. Each Alpaca round-trip is hundreds of ms
plus you're subject to rate limits. Caching turns a 10-minute fetch into a
sub-second disk read on the second call — and keeps re-runs deterministic
even if Alpaca is down or you're on a plane.

How it works
------------
- One Parquet file per symbol at ``<root>/<SYMBOL>.parquet``.
- Each file holds the *union* of every date range we've ever fetched for
  that symbol.
- A read checks if the requested window is fully covered. If yes, slice
  from disk — zero network. If only the leading or trailing edge is
  missing, fetch just those edges and merge. If nothing is cached yet,
  fetch the requested window and save.

What this is NOT
----------------
- **Not stale-aware.** Alpaca's adjusted prices are computed *relative to
  today* — if AAPL splits 4-for-1 tomorrow, every cached AAPL bar from
  before today becomes off by a factor of 4. We don't track corporate
  actions yet. If you suspect stale data, call ``invalidate_symbol("AAPL")``
  to drop the file and let the next fetch repopulate it.
- **Not concurrency-safe.** Two processes writing the same symbol at the
  same time could corrupt the Parquet. Single-process use only for now.
  (We do write atomically — tmp file + rename — so an *interrupted* write
  won't corrupt anything.)

The cache exposes the same ``get_daily_bars(symbols, start, end)`` shape as
the underlying Alpaca client, so callers can swap them transparently:

    >>> raw    = AlpacaDataClient()
    >>> cached = BarsCache(client=raw, root=Path("data/bars/daily"))
    >>> df     = cached.get_daily_bars(["AAPL"], date(2024,1,1), date(2024,12,31))
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from quant.data.alpaca_client import _empty_bars_frame


class BarsProvider(Protocol):
    """The minimal interface the cache wraps.

    Any object with a ``get_daily_bars(symbols, start, end)`` method that
    returns the standard MultiIndex(symbol, timestamp) OHLCV DataFrame
    satisfies this. ``AlpacaDataClient`` is the production implementation;
    test stubs implement it too. Using a Protocol means we don't have to
    import or subclass anything — Python's duck typing does the work.
    """

    def get_daily_bars(
        self,
        symbols: Iterable[str],
        start: date,
        end: date,
    ) -> pd.DataFrame: ...


class BarsCache:
    """Parquet-backed cache around any ``BarsProvider``."""

    def __init__(self, client: BarsProvider, root: Path | str) -> None:
        """
        Parameters
        ----------
        client
            The underlying bars provider (typically ``AlpacaDataClient``).
        root
            Directory where Parquet files will live, one per symbol.
            Created on first write if it doesn't exist.
        """
        self._client = client
        self._root = Path(root)

    # ------ public API (matches BarsProvider) ----------------------------

    def get_daily_bars(
        self,
        symbols: Iterable[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return bars for ``symbols`` over ``[start, end]``, using disk when possible."""
        # Same normalization as the underlying client — uppercase + dedupe,
        # so a caller passing ["aapl", "AAPL"] gets one symbol fetched once.
        symbols = list(dict.fromkeys(s.upper() for s in symbols))
        if not symbols:
            return _empty_bars_frame()

        pieces: list[pd.DataFrame] = []
        for sym in symbols:
            full = self._ensure_symbol_cached(sym, start, end)
            if full.empty:
                continue
            sliced = _slice_by_date(full, start, end)
            if not sliced.empty:
                pieces.append(sliced)

        if not pieces:
            return _empty_bars_frame()

        # `sort_index()` so the final frame has a stable, deterministic
        # row order regardless of how many cache merges happened.
        return pd.concat(pieces).sort_index()

    def invalidate_symbol(self, symbol: str) -> None:
        """Drop the cached file for one symbol — call this after a known split."""
        # ``missing_ok=True`` means "don't raise if it doesn't exist" —
        # idempotent invalidation is more useful than a 'file not found' error.
        self._path_for(symbol.upper()).unlink(missing_ok=True)

    def clear(self) -> None:
        """Drop every cached file. Mostly useful in tests."""
        if self._root.exists():
            for p in self._root.glob("*.parquet"):
                p.unlink()

    # ------ internals ----------------------------------------------------

    def _path_for(self, symbol: str) -> Path:
        return self._root / f"{symbol}.parquet"

    def _read_cached(self, symbol: str) -> pd.DataFrame | None:
        """Return the cached frame for ``symbol``, or None if nothing on disk."""
        path = self._path_for(symbol)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _write_cached(self, symbol: str, df: pd.DataFrame) -> None:
        """Atomic write: write to ``.tmp`` then rename.

        Prevents a half-written Parquet file from corrupting the cache if
        the process is killed mid-write (rename is atomic on POSIX).
        """
        path = self._path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.replace(path)

    def _ensure_symbol_cached(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Make sure the cache covers ``[start, end]`` for ``symbol``.

        Returns the *full* cached frame for the symbol (not the slice for the
        requested window — callers do the slicing). This way the caller can
        cheaply do multiple slices off a single cache load without re-reading.
        """
        cached = self._read_cached(symbol)

        # Case 1: nothing cached yet → fetch the requested window and save it.
        if cached is None or cached.empty:
            fresh = self._client.get_daily_bars([symbol], start, end)
            if not fresh.empty:
                self._write_cached(symbol, fresh)
            return fresh

        # Case 2: figure out which edges (if any) are missing.
        ts = cached.index.get_level_values("timestamp")
        cached_start = ts.min().date()
        cached_end = ts.max().date()

        to_fetch: list[tuple[date, date]] = []
        if start < cached_start:
            # Need older data — fetch [start, cached_start - 1 day].
            to_fetch.append((start, cached_start - timedelta(days=1)))
        if end > cached_end:
            # Need newer data — fetch [cached_end + 1 day, end].
            to_fetch.append((cached_end + timedelta(days=1), end))

        # Case 3: cache already covers the request → no network at all.
        if not to_fetch:
            return cached

        # Case 4: fetch only the missing edges and merge into the cache.
        pieces = [cached]
        for s, e in to_fetch:
            piece = self._client.get_daily_bars([symbol], s, e)
            if not piece.empty:
                pieces.append(piece)

        # All edge fetches came back empty (e.g., asking for dates before
        # Alpaca's data starts). Don't rewrite the cache for nothing.
        if len(pieces) == 1:
            return cached

        # Merge, sort, and defensively de-duplicate the index. Edges shouldn't
        # overlap by construction (we fetched [start, cached_start - 1] etc.),
        # but a stray duplicate would silently corrupt downstream math.
        merged = pd.concat(pieces).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        self._write_cached(symbol, merged)
        return merged


def _slice_by_date(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Filter a (symbol, timestamp) frame to rows whose date is in [start, end].

    We compare on *date*, not full datetime: Alpaca timestamps daily bars at
    midnight ET (≈ 05:00 UTC), and the user asks in terms of calendar dates.
    A naive ``ts >= start_datetime`` comparison would drop the first row.
    """
    ts = df.index.get_level_values("timestamp")
    mask = (ts.date >= start) & (ts.date <= end)
    return df[mask]
