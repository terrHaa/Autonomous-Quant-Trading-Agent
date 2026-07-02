"""Tests for the monthly diagnostics orchestrator's pure helpers (Phase 4)."""
from __future__ import annotations

from quant.agent.monthly_diagnostics import (
    _candidate_regime_policy,
    render_diagnostics_md,
)


def test_candidate_policy_zeroes_dead_sleeve_boosts_live_one() -> None:
    health = {
        "mr": {"mean_ic": 0.027, "regime_ic": {"trend_up": -0.01, "trend_dn": 0.25}},
        "sma": {"mean_ic": 0.019, "regime_ic": {"trend_up": 0.03, "trend_dn": 0.01}},
        "dead": {"mean_ic": -0.01, "regime_ic": {"trend_up": -0.02}},
    }
    pol = _candidate_regime_policy(health, "trend_up_stormy", "trend_up")
    # MR's IC is negative in up-trends → multiplier 0 (switched off).
    assert pol["mr"]["trend_up_stormy"] == 0.0
    # SMA positive in up-trends → positive multiplier.
    assert pol["sma"]["trend_up_stormy"] > 0
    # A sleeve with non-positive overall IC is zeroed outright.
    assert pol["dead"]["trend_up_stormy"] == 0.0


def test_candidate_policy_multiplier_capped() -> None:
    health = {"x": {"mean_ic": 0.01, "regime_ic": {"trend_up": 0.10}}}  # 10x raw
    pol = _candidate_regime_policy(health, "trend_up_calm", "trend_up")
    assert pol["x"]["trend_up_calm"] <= 1.5    # capped


def test_render_handles_missing_sections() -> None:
    # Minimal diag (e.g. attribution failed) must still render without error.
    md = render_diagnostics_md({"regime": {"current": "trend_dn_calm",
                                            "avg_pairwise_corr": 0.4,
                                            "correlation_degross_factor": 0.83}})
    assert "Quant diagnostics" in md
    assert "trend_dn_calm" in md


def test_render_includes_signal_health_table() -> None:
    diag = {
        "signal_health": {
            "mr": {"mean_ic": 0.027, "ic_early": 0.051, "ic_recent": 0.004,
                   "decaying": True, "turnover": 0.94, "regime_ic": {"trend_dn": 0.25}},
        },
        "candidate_regime_policy": {"mr": {"trend_up_stormy": 0.0}},
    }
    md = render_diagnostics_md(diag)
    assert "mr" in md
    assert "⚠" in md            # decay flag rendered


def test_book_returns_has_datetime_index_for_factor_join() -> None:
    """Regression: a datetime.date (object-dtype) index matched nothing
    against the Timestamp-indexed factor panel — Pillar 3 reported
    '0 common observations' on the 2026-07-02 monthly."""
    from datetime import date

    import pandas as pd

    from quant.agent.monthly_diagnostics import _book_returns

    eq = {date(2026, 6, d): 100000.0 + d * 100 for d in range(1, 8)}
    book = _book_returns(eq)
    assert isinstance(book.index, pd.DatetimeIndex)
    # And it actually joins against a Timestamp-indexed frame.
    fr = pd.DataFrame(
        {"MKT": 0.001}, index=pd.DatetimeIndex([pd.Timestamp(d) for d in eq]),
    )
    joined = pd.concat([book.rename("p"), fr], axis=1, join="inner").dropna()
    assert len(joined) == len(book)
