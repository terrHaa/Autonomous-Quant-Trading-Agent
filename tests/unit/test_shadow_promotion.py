"""Tests for the shadow queue (A2) + promotion criteria (A4)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from quant.research import (
    ShadowQueue,
    TrialLedger,
    evaluate_promotion,
)


def _queue(tmp_path) -> ShadowQueue:
    return ShadowQueue(path=tmp_path / "queue.json")


# --- shadow queue ----------------------------------------------------------

def test_add_and_accrue_returns(tmp_path) -> None:
    q = _queue(tmp_path)
    q.add(candidate_id="c1", name="rsi_14", backtest_sharpe=1.1,
          entered_at=date(2026, 1, 1))
    for i in range(5):
        q.record_return("c1", date(2026, 1, 2) + timedelta(days=i), 0.001)
    c = q.get("c1")
    assert c is not None
    assert c.shadow_days == 5
    assert c.status == "shadow"
    assert len(c.returns_series()) == 5


def test_status_transitions_and_filtering(tmp_path) -> None:
    q = _queue(tmp_path)
    q.add(candidate_id="c1", name="a", backtest_sharpe=1.0)
    q.add(candidate_id="c2", name="b", backtest_sharpe=0.8)
    q.set_status("c1", "promoted")
    assert {c.candidate_id for c in q.candidates(status="shadow")} == {"c2"}
    assert {c.candidate_id for c in q.candidates(status="promoted")} == {"c1"}


def test_queue_survives_reload(tmp_path) -> None:
    p = tmp_path / "queue.json"
    ShadowQueue(path=p).add(candidate_id="c1", name="a", backtest_sharpe=1.0)
    ShadowQueue(path=p).record_return("c1", date(2026, 1, 2), 0.002)
    assert ShadowQueue(path=p).get("c1").shadow_days == 1


# --- promotion criteria ----------------------------------------------------

def _candidate_with_returns(q, cid, sharpe, daily_ret, n_days, start=date(2026, 1, 2)):
    q.add(candidate_id=cid, name=cid, backtest_sharpe=sharpe)
    for i in range(n_days):
        q.record_return(cid, start + timedelta(days=i), daily_ret[i])
    return q.get(cid)


def test_promotion_holds_when_too_few_shadow_days(tmp_path) -> None:
    q = _queue(tmp_path)
    led = TrialLedger(path=tmp_path / "l.jsonl")
    cand = _candidate_with_returns(q, "c1", 1.0, [0.001] * 8, 8)
    dec = evaluate_promotion(cand, ledger=led, min_shadow_days=20)
    assert not dec.promote
    assert not dec.checks["shadow_length"][0]


def test_promotion_holds_when_oos_collapses(tmp_path) -> None:
    q = _queue(tmp_path)
    led = TrialLedger(path=tmp_path / "l.jsonl")
    rng = np.random.default_rng(0)
    # Backtest claimed Sharpe 2.0 but shadow returns are flat-to-negative.
    rets = list(rng.normal(-0.0003, 0.01, 25))
    cand = _candidate_with_returns(q, "c1", 2.0, rets, 25)
    dec = evaluate_promotion(cand, ledger=led, min_shadow_days=20)
    assert not dec.promote
    assert not dec.checks["oos_survival"][0]


def test_promotion_passes_with_strong_uncorrelated_shadow(tmp_path) -> None:
    q = _queue(tmp_path)
    led = TrialLedger(path=tmp_path / "l.jsonl")
    rng = np.random.default_rng(1)
    rets = list(0.0009 + rng.normal(0, 0.006, 30))   # consistent positive edge
    cand = _candidate_with_returns(q, "c1", 1.3, rets, 30)
    # Book returns uncorrelated with the candidate.
    book = pd.Series(
        rng.normal(0, 0.01, 30),
        index=pd.to_datetime([date(2026, 1, 2) + timedelta(days=i) for i in range(30)]),
    )
    dec = evaluate_promotion(cand, ledger=led, book_returns=book,
                             min_shadow_days=20, dsr_floor=0.5)
    assert dec.promote, dec.summary()


def test_promotion_blocks_high_correlation_to_book(tmp_path) -> None:
    q = _queue(tmp_path)
    led = TrialLedger(path=tmp_path / "l.jsonl")
    rng = np.random.default_rng(2)
    base = 0.0009 + rng.normal(0, 0.006, 30)
    cand = _candidate_with_returns(q, "c1", 1.3, list(base), 30)
    idx = pd.to_datetime([date(2026, 1, 2) + timedelta(days=i) for i in range(30)])
    # Book is nearly identical to the candidate → high correlation.
    book = pd.Series(base + rng.normal(0, 0.0005, 30), index=idx)
    dec = evaluate_promotion(cand, ledger=led, book_returns=book,
                             min_shadow_days=20, dsr_floor=0.4, corr_ceiling=0.7)
    assert not dec.promote
    assert not dec.checks["diversification"][0]
