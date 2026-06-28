"""Tests for the backtest A/B harness (research substrate, phase A3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research import TrialLedger, run_ab_test


def _series(arr, start="2024-01-01") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(arr))
    return pd.Series(arr, index=idx)


def test_detects_genuine_improvement() -> None:
    rng = np.random.default_rng(1)
    n = 300
    shared = rng.normal(0, 0.01, n)              # shared market path
    base = _series(shared)
    # Variant adds a small, consistent daily edge on top of the same path.
    variant = _series(shared + 0.0006)
    res = run_ab_test(base, variant, name="add_edge", ledger=None)
    assert res.delta_tstat > 5                    # paired test sees it clearly
    assert res.sharpe_delta > 0
    # Without a ledger, DSR stays 0, so verdict is gated on that — check the
    # stats are right even if verdict is conservative.
    assert res.variant_sharpe > res.baseline_sharpe


def test_noise_is_inconclusive() -> None:
    rng = np.random.default_rng(2)
    n = 300
    base = _series(rng.normal(0, 0.01, n))
    variant = _series(rng.normal(0, 0.01, n))    # independent noise, no real edge
    res = run_ab_test(base, variant, name="noise")
    assert res.verdict == "inconclusive"


def test_worse_variant_flagged() -> None:
    rng = np.random.default_rng(3)
    n = 300
    shared = rng.normal(0, 0.01, n)
    base = _series(shared + 0.0006)
    variant = _series(shared)                     # strictly worse by a constant
    res = run_ab_test(base, variant, name="regress")
    assert res.verdict == "variant worse"
    assert res.delta_tstat < -5


def test_ledger_deflation_gates_adoption(tmp_path) -> None:
    """A variant that's significant in-sample must still clear the global
    DSR floor; a ledger full of prior trials should make adoption harder."""
    rng = np.random.default_rng(4)
    n = 252
    shared = rng.normal(0, 0.01, n)
    base = _series(shared)
    variant = _series(shared + 0.0008)            # real, sizable edge

    led = TrialLedger(path=tmp_path / "l.jsonl")
    res = run_ab_test(base, variant, name="edge", ledger=led, dsr_floor=0.6)
    # Both arms logged.
    assert led.n_trials(kind="structural") == 2
    # Strong real edge → should be adopted with a clean ledger.
    assert res.adopt
    assert res.variant_dsr > 0


def test_ledger_records_both_arms(tmp_path) -> None:
    led = TrialLedger(path=tmp_path / "l.jsonl")
    base = _series(np.full(60, 0.0))
    variant = _series(np.full(60, 0.0003))
    run_ab_test(base, variant, name="exp1", ledger=led)
    names = {t.name for t in led.trials()}
    assert "exp1::baseline" in names
    assert "exp1::variant" in names


def test_too_few_obs_inconclusive() -> None:
    base = _series([0.01, -0.01, 0.0])
    variant = _series([0.02, 0.0, 0.01])
    res = run_ab_test(base, variant, name="tiny")
    assert res.verdict == "inconclusive"
    assert "too few" in res.reason
