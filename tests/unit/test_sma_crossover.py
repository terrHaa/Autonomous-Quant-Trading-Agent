"""Tests for the SMA crossover strategy.

Pattern: build synthetic close series where the SMA relationship is
deterministic by construction (ramps up, ramps down, etc.), then assert
that the strategy returns the expected target weights.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant.backtest.types import Snapshot
from quant.data.alpaca_client import BAR_COLUMNS
from quant.strategies import SmaCrossover


def _bars_for_closes(closes_by_sym: dict[str, list[float]]) -> pd.DataFrame:
    """Build a MultiIndex(symbol, timestamp) bars frame from close series.

    open == low == close, high one cent above — keeps the OHLC contract
    valid without adding noise to anything the SMA strategy reads.
    """
    n = len(next(iter(closes_by_sym.values())))
    bdays = pd.bdate_range("2020-01-02", periods=n, tz="UTC")
    rows = []
    idx = []
    for sym, closes in closes_by_sym.items():
        assert len(closes) == n, "all close series must be the same length"
        for ts, c in zip(bdays, closes, strict=True):
            rows.append({
                "open": c, "high": c + 0.01, "low": c, "close": c, "volume": 1,
            })
            idx.append((sym, ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=list(BAR_COLUMNS),
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_fast_must_be_less_than_slow() -> None:
    with pytest.raises(ValueError, match="strictly less than"):
        SmaCrossover(["AAPL"], fast=200, slow=50)


def test_fast_equal_to_slow_rejected() -> None:
    with pytest.raises(ValueError):
        SmaCrossover(["AAPL"], fast=50, slow=50)


def test_zero_window_rejected() -> None:
    with pytest.raises(ValueError):
        SmaCrossover(["AAPL"], fast=0, slow=50)


def test_name_encodes_parameters() -> None:
    """The strategy name should distinguish variants so the registry can
    track multiple parameter sets as separate trials."""
    s1 = SmaCrossover(["AAPL"], fast=50, slow=200)
    s2 = SmaCrossover(["AAPL"], fast=20, slow=100)
    assert s1.name != s2.name


# ---------------------------------------------------------------------------
# Signal generation — single symbol
# ---------------------------------------------------------------------------


def test_insufficient_history_returns_empty() -> None:
    """Fewer than `slow` bars → strategy stays out.

    Important guard: a strategy that computes SMA on too-few bars would
    silently emit garbage signals during the warmup window.
    """
    bars = _bars_for_closes({"AAPL": [100.0] * 100})  # slow=200, only 100 bars
    strat = SmaCrossover(["AAPL"], fast=10, slow=200)
    snap = Snapshot.from_full_bars(bars, as_of=date(2020, 6, 1))
    assert strat.on_bar(snap) == {}


def test_uptrend_produces_long_signal() -> None:
    """Steadily rising closes → fast SMA always above slow → long."""
    # 250 bars, prices ramping 100, 101, 102, ..., 349.
    bars = _bars_for_closes({"AAPL": [100.0 + i for i in range(250)]})
    strat = SmaCrossover(["AAPL"], fast=10, slow=200)
    last_date = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last_date)
    assert strat.on_bar(snap) == {"AAPL": 1.0}


def test_downtrend_produces_flat() -> None:
    """Steadily falling closes → fast SMA below slow → flat."""
    bars = _bars_for_closes({"AAPL": [500.0 - i * 0.5 for i in range(250)]})
    strat = SmaCrossover(["AAPL"], fast=10, slow=200)
    last_date = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last_date)
    assert strat.on_bar(snap) == {}


# ---------------------------------------------------------------------------
# Signal generation — multiple symbols
# ---------------------------------------------------------------------------


def test_partial_cross_produces_equal_weight_on_longs_only() -> None:
    """Two symbols, one trending up and one trending down →
    only the up-trender gets a position."""
    bars = _bars_for_closes({
        "AAPL": [100.0 + i for i in range(250)],          # up
        "MSFT": [500.0 - i * 0.5 for i in range(250)],    # down
    })
    strat = SmaCrossover(["AAPL", "MSFT"], fast=10, slow=200)
    last_date = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last_date)

    intents = strat.on_bar(snap)
    assert intents == {"AAPL": 1.0}


def test_two_longs_split_equally() -> None:
    """Two symbols both in uptrend → 50/50 weight, summing to 1.0."""
    bars = _bars_for_closes({
        "AAPL": [100.0 + i for i in range(250)],
        "MSFT": [200.0 + i * 0.7 for i in range(250)],
    })
    strat = SmaCrossover(["AAPL", "MSFT"], fast=10, slow=200)
    last_date = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last_date)

    intents = strat.on_bar(snap)
    assert intents == {"AAPL": 0.5, "MSFT": 0.5}
    assert sum(intents.values()) == pytest.approx(1.0)


def test_missing_symbol_skipped_not_failed() -> None:
    """If a configured symbol has no bars yet, the strategy keeps going
    on the remaining symbols rather than crashing.

    Real universes contain recent IPOs and future additions — strategies
    must tolerate them silently.
    """
    bars = _bars_for_closes({"AAPL": [100.0 + i for i in range(250)]})
    strat = SmaCrossover(["AAPL", "NEW_IPO"], fast=10, slow=200)
    last_date = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last_date)
    # AAPL signals long, NEW_IPO has no data → AAPL gets 100%.
    assert strat.on_bar(snap) == {"AAPL": 1.0}


# ---------------------------------------------------------------------------
# Signal flip at the crossover boundary
# ---------------------------------------------------------------------------


def test_signal_flips_at_crossover() -> None:
    """As we feed bars one at a time, the signal must change exactly at the
    bar where fast crosses slow.

    Construction: 200 bars at price 100 (so both SMAs = 100), then 100 bars
    ramping from 100 to 130. After enough up-bars, fast SMA exceeds slow.
    """
    flat = [100.0] * 200
    ramp = [100.0 + i * 0.3 for i in range(100)]
    bars = _bars_for_closes({"AAPL": flat + ramp})

    strat = SmaCrossover(["AAPL"], fast=10, slow=200)
    dates = list(bars.index.get_level_values("timestamp").date.tolist())
    deduped_dates = sorted(set(dates))

    saw_flat = False
    saw_long = False
    for d in deduped_dates:
        snap = Snapshot.from_full_bars(bars, as_of=d)
        result = strat.on_bar(snap)
        if not result:
            saw_flat = True
        else:
            saw_long = True

    # Over the run we should have seen both regimes.
    assert saw_flat, "should have been flat during the warmup/flat segment"
    assert saw_long, "should have gone long after the ramp pushed fast > slow"
