"""Tests for the risk overlay.

clip_weights tests: deterministic inputs, exact arithmetic on caps,
audit-trail contents.

KillSwitch tests: behavioral around the trip threshold; sticky-state
discipline; reset doesn't quietly reset the peak.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant.risk import KillSwitch, clip_weights


# ---------------------------------------------------------------------------
# clip_weights
# ---------------------------------------------------------------------------


def test_within_limits_returns_input_unchanged() -> None:
    """If no cap binds, the function is a pass-through."""
    w = pd.Series({"A": 0.04, "B": 0.04, "C": 0.04, "D": 0.04, "E": 0.04})
    # Per-name 5%, gross 1.0 (sum=0.2), net 1.0. All under their caps.
    clipped, audit = clip_weights(
        w, max_position_weight=0.05, max_gross_leverage=1.0, max_net_exposure=1.0,
    )
    pd.testing.assert_series_equal(clipped, w)
    assert audit.per_name_clipped == []
    assert audit.gross_scale == 1.0
    assert audit.net_scale == 1.0


def test_per_name_cap_clips_individual_weight() -> None:
    """A 50% weight in a 5%-cap regime gets clipped to 5%."""
    w = pd.Series({"A": 0.5, "B": 0.03, "C": 0.02})
    clipped, audit = clip_weights(
        w, max_position_weight=0.05, max_gross_leverage=1.0, max_net_exposure=1.0,
    )
    assert clipped["A"] == 0.05
    assert clipped["B"] == 0.03  # untouched
    assert clipped["C"] == 0.02
    assert audit.per_name_clipped == ["A"]


def test_per_name_cap_preserves_sign() -> None:
    """A short position above the cap (in magnitude) is clipped, sign kept."""
    w = pd.Series({"A": -0.30, "B": 0.30})
    clipped, _ = clip_weights(
        w, max_position_weight=0.05, max_gross_leverage=1.0, max_net_exposure=1.0,
    )
    assert clipped["A"] == -0.05
    assert clipped["B"] == 0.05


def test_gross_cap_scales_when_binding() -> None:
    """Sum-of-absolutes above the cap → uniform scale-down."""
    # 4 names at 0.5 each → gross 2.0; cap at 1.5 → scale 0.75
    w = pd.Series({"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5})
    clipped, audit = clip_weights(
        w,
        max_position_weight=1.0,        # per-name cap doesn't bind
        max_gross_leverage=1.5,
        max_net_exposure=10.0,           # net cap doesn't bind
    )
    assert audit.gross_scale == pytest.approx(0.75)
    assert clipped["A"] == pytest.approx(0.375)
    assert audit.final_gross == pytest.approx(1.5)


def test_net_cap_scales_long_book_down() -> None:
    """Net cap binds for a long-only book at gross 1.5 with net cap 1.0."""
    # 3 names at 0.5 each → gross=1.5, net=1.5; net cap 1.0 → scale 0.667
    w = pd.Series({"A": 0.5, "B": 0.5, "C": 0.5})
    clipped, audit = clip_weights(
        w,
        max_position_weight=1.0,
        max_gross_leverage=1.5,         # gross cap doesn't bind
        max_net_exposure=1.0,
    )
    assert audit.net_scale == pytest.approx(1.0 / 1.5)
    assert clipped["A"] == pytest.approx(0.5 * (1.0 / 1.5))
    assert audit.final_net == pytest.approx(1.0)


def test_net_cap_handles_net_short_book() -> None:
    """Symmetric: a net-short book at net=-1.5 with cap=1.0 → scale 2/3."""
    w = pd.Series({"A": -0.6, "B": -0.5, "C": -0.4})  # net = -1.5
    clipped, audit = clip_weights(
        w, max_position_weight=1.0, max_gross_leverage=2.0, max_net_exposure=1.0,
    )
    assert audit.net_scale == pytest.approx(1.0 / 1.5)
    # All weights stay negative.
    assert (clipped < 0).all()
    assert audit.final_net == pytest.approx(-1.0, abs=1e-9)


def test_caps_apply_in_documented_order() -> None:
    """Per-name first, then gross, then net.

    Construction: one big weight (1.5) and four small ones (0.05 each).
      gross_before = 1.5 + 0.2 = 1.7; net_before = 1.7.
    Per-name cap 0.10 → big weight becomes 0.10.
      gross_after_per_name = 0.10 + 0.20 = 0.30; net_after_per_name = 0.30.
    Gross cap 1.0 → doesn't bind.
    Net cap 1.0 → doesn't bind.
    Final: A=0.10, others=0.05.
    """
    w = pd.Series({"A": 1.5, "B": 0.05, "C": 0.05, "D": 0.05, "E": 0.05})
    clipped, audit = clip_weights(
        w,
        max_position_weight=0.10,
        max_gross_leverage=1.0,
        max_net_exposure=1.0,
    )
    assert clipped["A"] == 0.10
    assert audit.gross_scale == 1.0
    assert audit.net_scale == 1.0


def test_negative_or_zero_caps_rejected() -> None:
    """Non-positive caps don't make sense. Refuse to silently do nothing."""
    w = pd.Series({"A": 0.5})
    with pytest.raises(ValueError):
        clip_weights(
            w, max_position_weight=0, max_gross_leverage=1, max_net_exposure=1,
        )
    with pytest.raises(ValueError):
        clip_weights(
            w, max_position_weight=0.1, max_gross_leverage=-1, max_net_exposure=1,
        )


# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------


def test_kill_switch_arms_at_initial_equity() -> None:
    """First check sets the peak; doesn't trip."""
    ks = KillSwitch(max_drawdown=0.15)
    assert ks.check(current_equity=1_000_000) is False
    assert ks.peak == 1_000_000
    assert ks.triggered is False


def test_kill_switch_trips_at_threshold() -> None:
    """Drop just past -15% from peak → tripped."""
    ks = KillSwitch(max_drawdown=0.15)
    ks.check(1_000_000)
    ks.check(1_100_000)        # new peak
    assert ks.check(900_000) is True   # -18.2% from peak
    assert ks.triggered


def test_kill_switch_does_not_trip_above_threshold() -> None:
    """A -10% drop with a -15% threshold should NOT trip."""
    ks = KillSwitch(max_drawdown=0.15)
    ks.check(1_000_000)
    ks.check(1_100_000)
    assert ks.check(990_000) is False  # -10%
    assert not ks.triggered


def test_kill_switch_is_sticky() -> None:
    """Once tripped, stays tripped even if equity recovers."""
    ks = KillSwitch(max_drawdown=0.15)
    ks.check(1_000_000)
    ks.check(800_000)          # -20% → trips
    assert ks.triggered
    # Recovery shouldn't re-arm.
    assert ks.check(1_200_000) is True
    assert ks.triggered


def test_kill_switch_reset_clears_trigger_but_keeps_peak() -> None:
    """After operator review, reset() un-trips — but the peak persists.

    Persisting the peak forces the strategy to claw back to the prior
    high-water mark; giving it a fresh peak would be too generous after
    a blowup.
    """
    ks = KillSwitch(max_drawdown=0.15)
    ks.check(1_000_000)
    ks.check(800_000)          # trip
    prior_peak = ks.peak
    ks.reset()
    assert not ks.triggered
    # Peak unchanged after reset.
    assert ks.peak == prior_peak


def test_kill_switch_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        KillSwitch(max_drawdown=0)
    with pytest.raises(ValueError):
        KillSwitch(max_drawdown=1.0)   # 100% drawdown means total wipeout
    with pytest.raises(ValueError):
        KillSwitch(max_drawdown=-0.15)
