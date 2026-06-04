"""Tests for the short-term mean reversion strategy.

Pattern: build synthetic close series where the deviation-vs-MA is exactly
computable, then assert on the signal. Same approach as test_sma_crossover.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant.backtest.types import Snapshot
from quant.data.alpaca_client import BAR_COLUMNS
from quant.strategies import MeanReversion


def _bars_for_closes(closes_by_sym: dict[str, list[float]]) -> pd.DataFrame:
    """Build a MultiIndex bars frame from per-symbol close series.

    open == low == close (high one cent above) — keeps OHLC valid without
    adding noise to any signal the strategy reads (it only reads close).
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


def test_lookback_must_be_at_least_two() -> None:
    """A 1-bar window makes MA == current, so the signal never fires."""
    with pytest.raises(ValueError, match="lookback"):
        MeanReversion(["AAPL"], lookback=1)


def test_threshold_must_be_positive() -> None:
    """Zero threshold = trade on any deviation = noise machine. Reject."""
    with pytest.raises(ValueError, match="threshold"):
        MeanReversion(["AAPL"], threshold_pct=0.0)


def test_name_encodes_parameters() -> None:
    """All parameter variations must yield distinct names — needed so the
    registry's trial count for DSR doesn't conflate variants."""
    s_default = MeanReversion(["AAPL"])
    s_other_lookback = MeanReversion(["AAPL"], lookback=10)
    s_other_threshold = MeanReversion(["AAPL"], threshold_pct=0.05)
    s_short = MeanReversion(["AAPL"], allow_short=True)
    s_no_vol = MeanReversion(["AAPL"], vol_normalize=False)
    s_other_vol = MeanReversion(["AAPL"], vol_multiplier=2.0)
    names = {
        s_default.name, s_other_lookback.name, s_other_threshold.name,
        s_short.name, s_no_vol.name, s_other_vol.name,
    }
    assert len(names) == 6


def test_default_is_vol_normalized() -> None:
    """v2 default ON for the live agent. The legacy static-threshold
    behaviour requires opt-out."""
    assert MeanReversion(["AAPL"])._vol_normalize is True
    # Name encodes it so the registry knows variants apart.
    assert "vol20x1.5" in MeanReversion(["AAPL"]).name


def test_static_threshold_when_vol_normalize_off() -> None:
    """Opting out of vol-normalize restores the original deviation < -2% rule."""
    bars = _bars_for_closes({"AAPL": [100.0] * 10 + [95.0]})
    strat = MeanReversion(
        ["AAPL"], lookback=5, threshold_pct=0.02, vol_normalize=False,
    )
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    intents = strat.on_bar(snap)
    assert "AAPL" in intents


# ---------------------------------------------------------------------------
# Signal — single symbol
# ---------------------------------------------------------------------------


def test_insufficient_history_returns_empty() -> None:
    """Fewer than `lookback` closes → strategy stays flat (warmup)."""
    bars = _bars_for_closes({"AAPL": [100.0, 100.0]})   # 2 bars, lookback=5
    strat = MeanReversion(
        ["AAPL"], lookback=5, threshold_pct=0.02, vol_normalize=False,
    )
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    assert strat.on_bar(snap) == {}


def test_oversold_returns_long() -> None:
    """Last close significantly below MA → long signal.

    Construction: 10 closes at 100, then close drops to 95.
    Last-5 MA = (100+100+100+100+95)/5 = 99.0
    Deviation = (95 - 99)/99 = -4.04% < -2% threshold → long.
    """
    bars = _bars_for_closes({"AAPL": [100.0] * 10 + [95.0]})
    strat = MeanReversion(
        ["AAPL"], lookback=5, threshold_pct=0.02, vol_normalize=False,
    )
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    assert strat.on_bar(snap) == {"AAPL": 1.0}


def test_overbought_long_only_stays_flat() -> None:
    """Last close above MA, long-only mode → no position (just don't buy)."""
    bars = _bars_for_closes({"AAPL": [100.0] * 10 + [110.0]})
    strat = MeanReversion(
        ["AAPL"], lookback=5, threshold_pct=0.02, allow_short=False,
    )
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    assert strat.on_bar(snap) == {}


def test_overbought_long_short_returns_short() -> None:
    """Last close above MA, allow_short=True → negative weight (short)."""
    bars = _bars_for_closes({"AAPL": [100.0] * 10 + [110.0]})
    strat = MeanReversion(
        ["AAPL"], lookback=5, threshold_pct=0.02, allow_short=True,
    )
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    assert strat.on_bar(snap) == {"AAPL": -1.0}


def test_within_threshold_returns_flat() -> None:
    """Small deviation < threshold → no signal."""
    # Last-5 MA after closes = [100,100,100,100,100.5] is (100*4+100.5)/5=100.1.
    # Deviation = (100.5 - 100.1)/100.1 ≈ 0.4% < 2% threshold.
    bars = _bars_for_closes({"AAPL": [100.0] * 9 + [100.0, 100.5]})
    strat = MeanReversion(
        ["AAPL"], lookback=5, threshold_pct=0.02, vol_normalize=False,
    )
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    assert strat.on_bar(snap) == {}


# ---------------------------------------------------------------------------
# Signal — multi-symbol weighting
# ---------------------------------------------------------------------------


def test_multiple_longs_equal_deviation_split_equally() -> None:
    """Two equally-oversold symbols → 50/50 weight (conviction is identical)."""
    bars = _bars_for_closes({
        "AAPL": [100.0] * 10 + [95.0],     # ~ -4% deviation
        "MSFT": [200.0] * 10 + [190.0],    # ~ -4% deviation (same %)
    })
    strat = MeanReversion(
        ["AAPL", "MSFT"], lookback=5, threshold_pct=0.02,
        vol_normalize=False,
    )
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    intents = strat.on_bar(snap)
    # Equal deviations → equal conviction → equal weights.
    assert intents["AAPL"] == pytest.approx(0.5)
    assert intents["MSFT"] == pytest.approx(0.5)
    assert sum(intents.values()) == pytest.approx(1.0)


def test_conviction_weighting_deeper_oversold_gets_more_capital() -> None:
    """The whole point of conviction weighting: a deeply-oversold name
    gets MORE capital than a barely-oversold one. Regression guard for
    the equal-weight regression — if anyone ever 'simplifies' back to
    1/N weighting, this test screams."""
    bars = _bars_for_closes({
        "AAPL": [100.0] * 10 + [90.0],     # ~ -10% deviation (deep)
        "MSFT": [200.0] * 10 + [195.0],    # ~ -2.5% deviation (shallow)
    })
    strat = MeanReversion(
        ["AAPL", "MSFT"], lookback=5, threshold_pct=0.02,
        vol_normalize=False,
    )
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    intents = strat.on_bar(snap)
    # AAPL is deeper oversold → gets more capital.
    assert intents["AAPL"] > intents["MSFT"], (
        "deeper-oversold name MUST get more weight (conviction weighting)"
    )
    # Total still sums to 1.0 (no leverage).
    assert sum(intents.values()) == pytest.approx(1.0)
    # Approximate ratio: deviation magnitudes are ~10% vs ~2.5%, so
    # AAPL gets roughly 4× MSFT's weight. Loose tolerance because the
    # deviation uses MA from the last 5 bars, not just the last close.
    assert intents["AAPL"] / intents["MSFT"] == pytest.approx(4.0, rel=0.05)


def test_partial_signal_only_one_side_filled() -> None:
    """One symbol oversold, the other near MA → only the first gets weight."""
    bars = _bars_for_closes({
        "AAPL": [100.0] * 10 + [95.0],          # oversold → long
        "MSFT": [200.0] * 10 + [200.5],         # within threshold → flat
    })
    strat = MeanReversion(["AAPL", "MSFT"], lookback=5, threshold_pct=0.02)
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    assert strat.on_bar(snap) == {"AAPL": 1.0}


def test_long_and_short_sides_split_gross_evenly() -> None:
    """With one long and one short, each side gets half of gross exposure."""
    bars = _bars_for_closes({
        "AAPL": [100.0] * 10 + [95.0],   # oversold → long
        "MSFT": [200.0] * 10 + [210.0],  # overbought → short
    })
    strat = MeanReversion(
        ["AAPL", "MSFT"], lookback=5, threshold_pct=0.02, allow_short=True,
    )
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    intents = strat.on_bar(snap)
    assert intents == {"AAPL": 0.5, "MSFT": -0.5}
    # Net exposure = 0 (dollar-neutral); gross = 1.0.
    assert sum(intents.values()) == pytest.approx(0.0)
    assert sum(abs(v) for v in intents.values()) == pytest.approx(1.0)


def test_missing_symbol_skipped() -> None:
    """Same robustness contract as SmaCrossover: missing bars don't crash."""
    bars = _bars_for_closes({"AAPL": [100.0] * 10 + [95.0]})
    strat = MeanReversion(["AAPL", "NOTYETIPO"], lookback=5, threshold_pct=0.02)
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    assert strat.on_bar(snap) == {"AAPL": 1.0}
