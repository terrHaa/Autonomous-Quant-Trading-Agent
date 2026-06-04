"""Tests for the cross-sectional momentum strategy.

Pattern: build a universe of N synthetic symbols with predictable
trajectories, snapshot at a known date, assert the strategy ranks them
correctly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant.backtest.types import Snapshot
from quant.data.alpaca_client import BAR_COLUMNS
from quant.strategies import CrossSectionalMomentum


def _bars_from_closes(closes: dict[str, list[float]]) -> pd.DataFrame:
    """MultiIndex(symbol, timestamp) bars frame with predictable closes."""
    n = len(next(iter(closes.values())))
    days = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    rows, idx = [], []
    for sym, series in closes.items():
        assert len(series) == n, "all series must be equal length"
        for ts, c in zip(days, series, strict=True):
            rows.append({"open": c, "high": c + 0.01, "low": c, "close": c, "volume": 1})
            idx.append((sym, ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=list(BAR_COLUMNS),
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_rejects_skip_geq_lookback() -> None:
    """skip >= lookback would give a zero/negative signal window."""
    with pytest.raises(ValueError, match="skip"):
        CrossSectionalMomentum(["AAPL", "MSFT"], lookback=5, skip=5, top_k=1)


def test_rejects_top_k_too_large() -> None:
    """Can't pick top-10 from a 5-name universe."""
    with pytest.raises(ValueError, match="top_k"):
        CrossSectionalMomentum(["A", "B", "C"], top_k=5)


def test_rejects_nonsense_parameters() -> None:
    with pytest.raises(ValueError):
        CrossSectionalMomentum(["A", "B"], lookback=1)
    with pytest.raises(ValueError):
        CrossSectionalMomentum(["A", "B"], skip=-1)
    with pytest.raises(ValueError):
        CrossSectionalMomentum(["A", "B"], top_k=0)


# ---------------------------------------------------------------------------
# Ranking correctness
# ---------------------------------------------------------------------------


def test_top_k_one_selects_strongest_momentum_name() -> None:
    """With distinct return trajectories, top-1 must be the highest momer."""
    # 70 bars; ramp UP for WINNER, ramp DOWN for LOSER, FLAT for the rest.
    n = 70
    closes = {
        "WINNER": [100.0 + i * 0.5 for i in range(n)],          # strong uptrend
        "LOSER":  [200.0 - i * 0.5 for i in range(n)],          # strong downtrend
        "FLAT_A": [50.0] * n,
        "FLAT_B": [80.0] * n,
    }
    bars = _bars_from_closes(closes)
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)

    strat = CrossSectionalMomentum(
        ["WINNER", "LOSER", "FLAT_A", "FLAT_B"],
        lookback=60, skip=5, top_k=1,
    )
    intents = strat.on_bar(snap)
    assert intents == {"WINNER": 1.0}


def test_top_k_weighted_by_momentum_strength() -> None:
    """Top-2 names → conviction-weighted by their momentum return.

    BEST has steeper momentum than GOOD, so BEST gets more capital.
    Weights still sum to 1.0 across the top-K.
    """
    n = 70
    closes = {
        "BEST":  [100.0 + i * 0.5 for i in range(n)],
        "GOOD":  [100.0 + i * 0.3 for i in range(n)],
        "BAD":   [100.0 - i * 0.1 for i in range(n)],
        "WORST": [100.0 - i * 0.5 for i in range(n)],
    }
    bars = _bars_from_closes(closes)
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)

    strat = CrossSectionalMomentum(
        ["BEST", "GOOD", "BAD", "WORST"], lookback=60, skip=5, top_k=2,
    )
    intents = strat.on_bar(snap)
    assert set(intents) == {"BEST", "GOOD"}
    assert sum(intents.values()) == pytest.approx(1.0)
    # BEST has steeper momentum → larger weight than GOOD.
    assert intents["BEST"] > intents["GOOD"], (
        "Higher-momentum name MUST get more weight (conviction weighting). "
        "Regression guard against equal-weighting that throws away signal "
        "magnitude — pre-v2 behavior."
    )


def test_negative_momentum_names_dropped_from_top_k() -> None:
    """If the top-K includes a name with NEGATIVE momentum, drop it.

    Better to under-deploy than to bet on a loser just because it was
    'less negative' than other losers."""
    n = 70
    closes = {
        "WINNER": [100.0 + i * 0.5 for i in range(n)],   # +29% over window
        "LOSER1": [100.0 - i * 0.1 for i in range(n)],   # -5% over window
        "LOSER2": [100.0 - i * 0.3 for i in range(n)],   # -17% over window
    }
    bars = _bars_from_closes(closes)
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    strat = CrossSectionalMomentum(
        ["WINNER", "LOSER1", "LOSER2"], lookback=60, skip=5, top_k=3,
    )
    intents = strat.on_bar(snap)
    # Only WINNER survives — losers dropped despite being in the top-K.
    assert list(intents.keys()) == ["WINNER"]
    assert intents["WINNER"] == pytest.approx(1.0)


def test_skip_period_is_excluded_from_signal_window() -> None:
    """A symbol that's been ramping for 55 days then crashed in last 5 days
    should still rank highly — because we EXCLUDE those last 5 days.

    This is the whole point of the skip parameter."""
    # 70 bars. SYM_A: ramps up 60 days, crashes last 5. SYM_B: ramps down
    # for 60 days, rallies last 5. With skip=5, SYM_A should still beat
    # SYM_B because we don't see the recent crash/rally.
    sym_a_ramp = [100.0 + i for i in range(65)]               # 65 ramp days
    sym_a_crash = [sym_a_ramp[-1] * 0.5 for _ in range(5)]    # then halves
    sym_a = sym_a_ramp + sym_a_crash                          # 70 closes

    sym_b_drop = [200.0 - i for i in range(65)]
    sym_b_rally = [sym_b_drop[-1] * 2.0 for _ in range(5)]
    sym_b = sym_b_drop + sym_b_rally

    bars = _bars_from_closes({"SYM_A": sym_a, "SYM_B": sym_b})
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)

    strat = CrossSectionalMomentum(["SYM_A", "SYM_B"], lookback=60, skip=5, top_k=1)
    intents = strat.on_bar(snap)
    # Without skip, SYM_B would win (recent rally). With skip=5, SYM_A
    # wins because the skip discards the crash/rally days.
    assert intents == {"SYM_A": 1.0}


def test_insufficient_history_excluded() -> None:
    """A symbol with fewer bars than lookback is silently dropped.

    Setup: OLD has 70 bars ending at day 69; NEW has 30 bars ALSO ending
    at day 69 (i.e., NEW only started existing 30 days ago — like a
    recent IPO). The as_of is day 69; OLD has enough history, NEW
    doesn't.
    """
    n = 70
    days = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    rows, idx = [], []
    # OLD: full 70-bar history, gentle uptrend.
    for i, ts in enumerate(days):
        c = 100.0 + i * 0.2
        rows.append({"open": c, "high": c + 0.01, "low": c, "close": c, "volume": 1})
        idx.append(("OLD", ts))
    # NEW: only the LAST 30 bars exist.
    for i, ts in enumerate(days[-30:]):
        c = 200.0 + i * 0.5
        rows.append({"open": c, "high": c + 0.01, "low": c, "close": c, "volume": 1})
        idx.append(("NEW", ts))
    bars = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=list(BAR_COLUMNS),
    )

    last = bars.index.get_level_values("timestamp").max().date()
    snap = Snapshot.from_full_bars(bars, as_of=last)
    strat = CrossSectionalMomentum(["OLD", "NEW"], lookback=60, skip=5, top_k=2)
    intents = strat.on_bar(snap)
    # Only OLD has enough history; top_k=2 but we found only 1 ranker
    # → that one gets 100%.
    assert intents == {"OLD": 1.0}


def test_empty_snapshot_returns_empty_intents() -> None:
    """A snapshot with NO bars yet (e.g., engine bar 0) → strategy stays flat."""
    empty = pd.DataFrame(
        columns=list(BAR_COLUMNS),
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )
    snap = Snapshot.from_full_bars(empty, as_of=pd.Timestamp("2024-01-02").date())
    strat = CrossSectionalMomentum(["AAPL", "MSFT"], lookback=60, skip=5, top_k=1)
    assert strat.on_bar(snap) == {}


def test_skip_zero_uses_full_lookback() -> None:
    """skip=0 → use the whole lookback window including the latest bar."""
    n = 70
    closes = {
        "A": [100.0 + i * 0.5 for i in range(n)],
        "B": [100.0] * n,
    }
    bars = _bars_from_closes(closes)
    last = bars.index.get_level_values("timestamp")[-1].date()
    snap = Snapshot.from_full_bars(bars, as_of=last)

    strat = CrossSectionalMomentum(["A", "B"], lookback=60, skip=0, top_k=1)
    intents = strat.on_bar(snap)
    assert intents == {"A": 1.0}


# ---------------------------------------------------------------------------
# Name encoding (for DSR's trial-count distinct-variant tracking)
# ---------------------------------------------------------------------------


def test_name_encodes_parameters() -> None:
    """Different (lookback, skip, top_k) tuples are DIFFERENT trials."""
    s1 = CrossSectionalMomentum(["A", "B", "C"], lookback=60, skip=5, top_k=1)
    s2 = CrossSectionalMomentum(["A", "B", "C"], lookback=120, skip=5, top_k=1)
    s3 = CrossSectionalMomentum(["A", "B", "C"], lookback=60, skip=0, top_k=1)
    s4 = CrossSectionalMomentum(["A", "B", "C"], lookback=60, skip=5, top_k=2)
    names = {s1.name, s2.name, s3.name, s4.name}
    assert len(names) == 4
