"""Tests for the Parquet bars cache.

These never touch the network or the real Alpaca client. Instead we drive
the cache with a small stub provider that records every call — this lets
each test assert *exactly* when the underlying client was hit and with what
arguments, which is the whole point of a cache.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from quant.data.alpaca_client import BAR_COLUMNS, _empty_bars_frame
from quant.data.cache import BarsCache


# ----------------------------------------------------------------------------
# Test double: a fake BarsProvider that records its calls and returns
# deterministic, easy-to-eyeball OHLCV rows for the requested window.
# ----------------------------------------------------------------------------


class _FakeClient:
    """Stand-in for AlpacaDataClient. Records every call; returns fake bars.

    Returning *something* (not an empty frame) lets the cache exercise its
    real write path. Recording calls lets the tests assert "the cache hit
    the underlying client exactly this many times for exactly this window."
    """

    def __init__(self) -> None:
        # List of (sorted_symbols, start, end) — sorted so test assertions
        # don't depend on the cache's iteration order.
        self.calls: list[tuple[tuple[str, ...], date, date]] = []

    def get_daily_bars(
        self,
        symbols: Iterable[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        sym_list = list(symbols)
        self.calls.append((tuple(sorted(sym_list)), start, end))

        # `bdate_range` = business-day range, mirrors Alpaca's behavior of
        # not returning Saturday/Sunday bars.
        bdays = pd.bdate_range(start=start, end=end, tz="UTC")
        if len(bdays) == 0 or not sym_list:
            return _empty_bars_frame()

        rows = []
        idx = []
        for sym in sym_list:
            for i, ts in enumerate(bdays):
                rows.append(
                    {
                        "open": 100.0 + i,
                        "high": 101.0 + i,
                        "low": 99.0 + i,
                        "close": 100.5 + i,
                        "volume": 1000 + i,
                    }
                )
                idx.append((sym, ts))

        return pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
            columns=list(BAR_COLUMNS),
        )


# ----------------------------------------------------------------------------
# Tests. Each one targets a single cache behavior, named for what it proves.
# `tmp_path` is a pytest fixture giving a fresh temp directory per test, so
# tests can't leak state into each other or into the real cache.
# ----------------------------------------------------------------------------


def test_first_call_fetches_from_client_and_writes_file(tmp_path: Path) -> None:
    """A cold cache hits the underlying client and persists the result."""
    fake = _FakeClient()
    cache = BarsCache(client=fake, root=tmp_path)

    df = cache.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 12))

    assert len(fake.calls) == 1, "first call should hit the underlying client once"
    assert not df.empty
    assert (tmp_path / "AAPL.parquet").exists(), "cache file should be written"


def test_identical_repeat_call_does_not_hit_client(tmp_path: Path) -> None:
    """The whole point of the cache: same window, second call is free."""
    fake = _FakeClient()
    cache = BarsCache(client=fake, root=tmp_path)

    first = cache.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 12))
    second = cache.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 12))

    assert len(fake.calls) == 1, "second call must come from disk, not the API"
    # Both frames should be identical — frame equality (not just shape).
    pd.testing.assert_frame_equal(first, second)


def test_subset_window_uses_cache_only(tmp_path: Path) -> None:
    """Asking for a slice of an already-cached window does no network."""
    fake = _FakeClient()
    cache = BarsCache(client=fake, root=tmp_path)

    # Prime the cache with a wide window.
    cache.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 31))
    # Now ask for a subset.
    df = cache.get_daily_bars(["AAPL"], date(2024, 1, 8), date(2024, 1, 12))

    assert len(fake.calls) == 1, "no second client call for a cached subset"

    # Returned rows must lie strictly within the requested subset window.
    ts = df.index.get_level_values("timestamp").date
    assert ts.min() >= date(2024, 1, 8)
    assert ts.max() <= date(2024, 1, 12)


def test_extended_later_fetches_only_the_missing_tail(tmp_path: Path) -> None:
    """Extending the right edge fetches only the new dates, not the whole range."""
    fake = _FakeClient()
    cache = BarsCache(client=fake, root=tmp_path)

    cache.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 12))
    cache.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 31))  # extend

    assert len(fake.calls) == 2, "should have fetched the new tail once"
    # The second call's window must be only the new tail.
    _, s, e = fake.calls[1]
    assert s == date(2024, 1, 13), "should start the day after the cached end"
    assert e == date(2024, 1, 31)


def test_extended_earlier_fetches_only_the_missing_head(tmp_path: Path) -> None:
    """Same idea, but extending into older dates."""
    fake = _FakeClient()
    cache = BarsCache(client=fake, root=tmp_path)

    cache.get_daily_bars(["AAPL"], date(2024, 1, 15), date(2024, 1, 31))
    cache.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 31))  # extend back

    assert len(fake.calls) == 2
    _, s, e = fake.calls[1]
    assert s == date(2024, 1, 2)
    assert e == date(2024, 1, 14), "should end the day before the cached start"


def test_cache_survives_process_restart(tmp_path: Path) -> None:
    """The cache is pure-disk state; a fresh BarsCache sees prior writes.

    Important guarantee — your overnight cache survives an interpreter
    restart with no special "load" step required.
    """
    cache_v1 = BarsCache(client=_FakeClient(), root=tmp_path)
    cache_v1.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 12))

    # Simulate a new process: new client, new cache, same root directory.
    fake_v2 = _FakeClient()
    cache_v2 = BarsCache(client=fake_v2, root=tmp_path)
    df = cache_v2.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 12))

    assert len(fake_v2.calls) == 0, "second process should not have hit the API"
    assert not df.empty


def test_invalidate_symbol_forces_refetch(tmp_path: Path) -> None:
    """`invalidate_symbol` drops the file so the next call refetches.

    The intended use case is "I know AAPL just split; my cached adjusted
    prices are now wrong by the split ratio; clear it and let the next
    fetch repopulate it from a fresh adjustment baseline."
    """
    fake = _FakeClient()
    cache = BarsCache(client=fake, root=tmp_path)

    cache.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 12))
    cache.invalidate_symbol("AAPL")
    cache.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 12))

    assert len(fake.calls) == 2, "second call should refetch after invalidation"


def test_multi_symbol_caches_per_symbol(tmp_path: Path) -> None:
    """Adding a new symbol shouldn't refetch ones already cached."""
    fake = _FakeClient()
    cache = BarsCache(client=fake, root=tmp_path)

    cache.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 12))
    cache.get_daily_bars(["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 1, 12))

    # AAPL was cached, MSFT was not → one new call, for MSFT only.
    assert len(fake.calls) == 2
    syms_second_call, _, _ = fake.calls[1]
    assert syms_second_call == ("MSFT",)
