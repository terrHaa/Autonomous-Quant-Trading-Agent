"""Tests for the SPY/QQQ benchmark-return helper used by the reports."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant.data.alpaca_client import BAR_COLUMNS
from quant.util.benchmarks import BENCHMARK_TICKERS, fetch_benchmark_returns


def _bars(closes_by_sym: dict[str, list[float]]) -> pd.DataFrame:
    """Build a MultiIndex(symbol, timestamp) OHLCV frame for the test cache."""
    rows = []
    idx = []
    n = len(next(iter(closes_by_sym.values())))
    bdays = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    for sym, closes in closes_by_sym.items():
        for i, ts in enumerate(bdays):
            c = closes[i]
            rows.append({
                "open": c, "high": c * 1.01, "low": c * 0.99,
                "close": c, "volume": 1_000_000,
            })
            idx.append((sym, ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=list(BAR_COLUMNS),
    )


class _FakeCache:
    """Minimal BarsProvider: returns a pre-built frame; tracks calls."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list = []

    def get_daily_bars(self, symbols, start, end):
        self.calls.append((list(symbols), start, end))
        return self.frame


def test_returns_dict_keyed_by_ticker_when_both_priced() -> None:
    bars = _bars({
        "SPY": [400.0, 402.0, 404.0],  # +1.0% close-to-close
        "QQQ": [350.0, 357.0, 360.0],  # +2.857%
    })
    out = fetch_benchmark_returns(
        _FakeCache(bars), date(2024, 1, 1), date(2024, 1, 10),
    )
    assert set(out.keys()) == {"SPY", "QQQ"}
    assert out["SPY"] == pytest.approx(0.01, rel=1e-3)
    assert out["QQQ"] == pytest.approx(0.02857, rel=1e-3)


def test_skips_ticker_with_only_one_bar() -> None:
    """A 1-bar window can't yield a return; that ticker is omitted."""
    bars = _bars({
        "SPY": [400.0, 405.0],
        # Build a frame that has QQQ but only 1 row by trimming after.
    })
    # Trim QQQ to 1 row by adding it manually with a single index entry.
    extra = pd.DataFrame(
        [{"open": 350.0, "high": 354.0, "low": 347.0, "close": 350.0,
          "volume": 1_000_000}],
        index=pd.MultiIndex.from_tuples(
            [("QQQ", pd.Timestamp("2024-01-02", tz="UTC"))],
            names=["symbol", "timestamp"],
        ),
        columns=list(BAR_COLUMNS),
    )
    bars = pd.concat([bars, extra]).sort_index()
    out = fetch_benchmark_returns(
        _FakeCache(bars), date(2024, 1, 1), date(2024, 1, 10),
    )
    # SPY priced (2 bars), QQQ omitted (1 bar).
    assert "SPY" in out and "QQQ" not in out


def test_empty_bars_returns_empty_dict() -> None:
    """Cache miss → empty dict; caller treats as 'no benchmarks available'."""
    empty = pd.DataFrame(
        columns=list(BAR_COLUMNS),
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )
    out = fetch_benchmark_returns(
        _FakeCache(empty), date(2024, 1, 1), date(2024, 1, 10),
    )
    assert out == {}


def test_cache_exception_is_swallowed_returning_empty() -> None:
    """Provider crash → empty dict + warning; never raise (report must ship)."""
    class _Boom:
        def get_daily_bars(self, *_a, **_kw):
            raise RuntimeError("alpaca down")
    out = fetch_benchmark_returns(_Boom(), date(2024, 1, 1), date(2024, 1, 10))
    assert out == {}


def test_default_tickers_are_spy_and_qqq() -> None:
    """The module-level constant must stay {SPY, QQQ} — the renderers
    have hardcoded labels keyed on these symbols."""
    assert BENCHMARK_TICKERS == ("SPY", "QQQ")


def test_caller_can_override_tickers() -> None:
    """Allow tests / future callers to fetch other benchmarks (e.g., IWM)."""
    bars = _bars({"IWM": [200.0, 198.0]})  # -1%
    out = fetch_benchmark_returns(
        _FakeCache(bars), date(2024, 1, 1), date(2024, 1, 10),
        tickers=["IWM"],
    )
    assert out["IWM"] == pytest.approx(-0.01, rel=1e-3)
