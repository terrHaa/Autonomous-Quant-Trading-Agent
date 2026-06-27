"""Tests for the Pillar-5 reliability + implementation-shortfall module."""
from __future__ import annotations

import json
from datetime import date, timedelta

from quant.agent.reliability import (
    _classify_failure,
    compute_implementation_shortfall,
    compute_reliability_scorecard,
)


def _run(d: str, entries: list[tuple[str, str]], weights: dict[str, float]) -> dict:
    return {
        "date": d,
        "target_weights": weights,
        "execution_report": {
            "submitted_orders": [
                {"symbol": s, "role": "entry", "status": st, "error": ""}
                for s, st in entries
            ],
        },
    }


def test_shortfall_empty() -> None:
    assert compute_implementation_shortfall([]) == {}


def test_shortfall_fidelity_and_leaked_exposure() -> None:
    runs = [
        _run("2026-06-01",
             [("AAA", "submitted"), ("BBB", "failed"), ("CCC", "kept")],
             {"AAA": 0.1, "BBB": 0.05, "CCC": 0.1}),
        _run("2026-06-02",
             [("DDD", "failed"), ("EEE", "submitted")],
             {"DDD": 0.2, "EEE": 0.1}),
    ]
    sf = compute_implementation_shortfall(runs)
    assert sf["entries_intended"] == 5
    assert sf["entries_placed"] == 3      # submitted + kept
    assert sf["entries_failed"] == 2
    assert sf["entry_fidelity_pct"] == 60.0
    # Leaked: day1 BBB=0.05, day2 DDD=0.20 → avg 12.5%.
    assert sf["avg_leaked_exposure_pct"] == 12.5


def test_failure_classification_buckets() -> None:
    assert "insufficient_qty" in _classify_failure(
        'insufficient qty available for order'
    )
    assert "already filled" in _classify_failure('order is already in "filled" state')
    assert "wash" in _classify_failure("potential wash trade detected")
    assert _classify_failure("") == "unspecified"


def test_failure_causes_counted() -> None:
    runs = [{
        "date": "2026-06-01",
        "target_weights": {},
        "execution_report": {"submitted_orders": [
            {"symbol": "A", "role": "entry", "status": "failed",
             "error": "insufficient qty available"},
            {"symbol": "B", "role": "entry", "status": "failed",
             "error": "insufficient qty available"},
        ]},
    }]
    sf = compute_implementation_shortfall(runs)
    cause = next(iter(sf["failure_causes"]))
    assert "insufficient_qty" in cause
    assert sf["failure_causes"][cause] == 2


def test_scorecard_counts_missed_days_against_trading_calendar(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    today = date.today()
    # Three recent weekdays; write run records for two, leave one missing.
    d1, d2, d3 = today - timedelta(days=5), today - timedelta(days=4), today - timedelta(days=3)
    for d in (d1, d3):
        (runs_dir / f"{d.isoformat()}.json").write_text(
            json.dumps({"date": d.isoformat(), "execution_report": {}})
        )
    sc = compute_reliability_scorecard(
        runs_dir=runs_dir,
        logs_dir=tmp_path / "nolyance",
        audits_dir=tmp_path / "noaudit",
        days=10,
        trading_days={d1, d2, d3},
    )
    assert sc["trading_days_expected"] == 3
    assert sc["trading_days_traded"] == 2
    assert d2.isoformat() in sc["missed_trade_days"]
    assert sc["trade_completion_pct"] == round(2 / 3 * 100, 1)
