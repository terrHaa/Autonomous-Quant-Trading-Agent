"""Tests for quant.risk.deployment — regime filter + drawdown ladder."""

from __future__ import annotations

import pandas as pd
import pytest

from quant.risk.deployment import (
    DRAWDOWN_LADDER,
    REGIME_BELOW_SCALE,
    REGIME_SMA_WINDOW,
    drawdown_scale,
    regime_scale,
)

# ---------------------------------------------------------------------------
# regime_scale
# ---------------------------------------------------------------------------


def _spy_series(n: int, last: float, base: float = 100.0) -> pd.Series:
    """n-1 closes at `base`, final close at `last` — SMA ≈ base."""
    return pd.Series([base] * (n - 1) + [last], dtype=float)


def test_regime_risk_on_above_sma() -> None:
    scale, diag = regime_scale(_spy_series(REGIME_SMA_WINDOW + 50, last=150.0))
    assert scale == 1.0
    assert diag["regime"] == "risk_on"


def test_regime_risk_off_below_sma() -> None:
    scale, diag = regime_scale(_spy_series(REGIME_SMA_WINDOW + 50, last=50.0))
    assert scale == REGIME_BELOW_SCALE
    assert diag["regime"] == "risk_off"
    assert diag["spy_close"] == 50.0


def test_regime_exactly_at_sma_is_risk_on() -> None:
    # Boundary: close == SMA → stay deployed (>= comparison).
    s = pd.Series([100.0] * (REGIME_SMA_WINDOW + 10))
    scale, diag = regime_scale(s)
    assert scale == 1.0
    assert diag["regime"] == "risk_on"


def test_regime_fails_open_on_no_data() -> None:
    for series in (None, pd.Series([], dtype=float)):
        scale, diag = regime_scale(series)
        assert scale == 1.0
        assert diag["regime"] == "unknown"


def test_regime_fails_open_on_short_history() -> None:
    scale, diag = regime_scale(_spy_series(REGIME_SMA_WINDOW - 1, last=50.0))
    assert scale == 1.0
    assert diag["regime"] == "unknown"
    assert "SPY closes" in diag["reason"]


def test_regime_sma_uses_tail_window_only() -> None:
    # Ancient closes far above today must NOT drag the SMA: 300 old
    # closes at 1000, then 200 recent at 100 with last 101 → risk_on.
    s = pd.Series([1000.0] * 300 + [100.0] * 199 + [101.0])
    scale, diag = regime_scale(s)
    assert scale == 1.0
    assert diag["regime"] == "risk_on"


# ---------------------------------------------------------------------------
# drawdown_scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dd", "expected"),
    [
        (0.0, 1.0),          # at peak
        (0.02, 1.0),         # above peak (new high day)
        (-0.049, 1.0),       # just above first rung
        (-0.05, 0.75),       # first rung boundary
        (-0.07, 0.75),
        (-0.10, 0.50),       # second rung boundary
        (-0.11, 0.50),
        (-0.125, 0.25),      # third rung boundary
        (-0.14, 0.25),       # deepest rung holds until the kill switch
        (-0.30, 0.25),       # ladder never returns 0 — the kill is separate
    ],
)
def test_drawdown_ladder_rungs(dd: float, expected: float) -> None:
    assert drawdown_scale(dd) == expected


def test_ladder_constant_is_sane() -> None:
    # Thresholds negative and descending in severity; scales in (0, 1).
    for threshold, scale in DRAWDOWN_LADDER:
        assert threshold < 0
        assert 0.0 < scale < 1.0
    # Deeper drawdown must never INCREASE deployment.
    scales = [drawdown_scale(dd / 100) for dd in range(0, -31, -1)]
    assert scales == sorted(scales, reverse=True)
