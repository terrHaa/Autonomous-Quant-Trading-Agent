"""Project-wide test fixtures.

The autouse fixture below exists because of a live-state corruption
incident (2026-06-11): ``run_daily_trade`` loads and saves
``EnsembleState`` at ``ensemble.DEFAULT_STATE_PATH`` when the caller
doesn't inject state, so any test that exercised it end-to-end silently
overwrote ``data/agent/ensemble_state.json`` — wiping the live
``trail_high`` ratchet map on every full-suite run. Tests isolate
``runs_dir`` via ``tmp_path`` but nothing isolated the state file.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_ensemble_state(tmp_path_factory, monkeypatch):
    """Redirect the ensemble-state default path to a per-test temp file.

    ``load_ensemble_state`` / ``save_ensemble_state`` resolve
    ``DEFAULT_STATE_PATH`` at call time from the module global, so
    monkeypatching the global is sufficient for every code path that
    doesn't pass an explicit ``path=`` (which tests pass ``tmp_path``
    to anyway). The live ``data/agent/ensemble_state.json`` must NEVER
    be read or written by the test suite.
    """
    from quant.agent import ensemble

    state_path = tmp_path_factory.mktemp("ensemble_state") / "ensemble_state.json"
    monkeypatch.setattr(ensemble, "DEFAULT_STATE_PATH", state_path)
    yield
