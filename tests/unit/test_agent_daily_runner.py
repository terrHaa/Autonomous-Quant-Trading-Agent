"""Tests for the agent's daily trade + report routines.

These mock every external dependency (cache, executor, email sender, log
dir) so the tests never touch Alpaca or SMTP. The goal is to verify the
orchestration: the right pieces called in the right order with the right
arguments.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant.agent import daily_runner
from quant.agent.log import save_daily_run
from quant.data.alpaca_client import BAR_COLUMNS
from quant.execution.alpaca_executor import (
    ExecutionReport,
    ProposedOrder,
    SubmittedOrder,
)


@pytest.fixture
def _in_trade_window(monkeypatch):
    """Mark the test as 'inside the trade window' so the 09:00-15:35 ET
    guard doesn't trip on fixed historical test dates.

    NOT autouse — tests that directly call `_outside_trade_window()`
    (the helper unit tests) need the real function. Tests that go
    through `run_daily_trade()` request this fixture explicitly.
    """
    monkeypatch.setattr(daily_runner, "_outside_trade_window", lambda _d: False)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeCache:
    """BarsCache stand-in that returns a hand-crafted bars frame."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars
        self.calls: list = []

    def get_daily_bars(self, symbols, start, end):
        self.calls.append((tuple(symbols), start, end))
        return self._bars


class _FakeExecutor:
    """AlpacaExecutor stand-in that records args and returns a canned report."""

    env = "paper"

    def __init__(self, *, current_positions: dict[str, int] | None = None) -> None:
        self._positions = current_positions or {}
        self.last_call: dict[str, Any] = {}

    def get_positions(self):
        return dict(self._positions)

    def submit_daily_rebalance(self, **kwargs) -> ExecutionReport:
        self.last_call = kwargs
        # Build a fake report capturing the inputs so tests can verify.
        proposed: list[ProposedOrder] = []
        submitted: list[SubmittedOrder] = []
        for sym, weight in kwargs["target_weights"].items():
            price = kwargs["signal_prices"][sym]
            qty = int(weight * 100_000 / price)
            proposed.append(ProposedOrder(
                symbol=sym, side="buy", qty=qty,
                rationale=f"weight {weight}",
            ))
            submitted.append(SubmittedOrder(
                symbol=sym, side="buy", qty=qty,
                status="skipped_dry_run" if kwargs.get("dry_run") else "submitted",
                role="entry",
                alpaca_order_id=f"fake-{sym}",
            ))
            submitted.append(SubmittedOrder(
                symbol=sym, side="sell", qty=qty,
                status="skipped_dry_run" if kwargs.get("dry_run") else "submitted",
                role="stop_loss",
                stop_price=round(price * (1 - kwargs["stop_loss_pct"]), 2),
            ))
        return ExecutionReport(
            env="paper",
            timestamp=datetime.now(UTC),
            account_equity_before=100_000.0,
            positions_before=dict(self._positions),
            target_weights=dict(kwargs["target_weights"]),
            proposed_orders=proposed,
            submitted_orders=submitted,
            dry_run=bool(kwargs.get("dry_run")),
            notes=kwargs.get("notes", ""),
        )


def _make_bars(
    symbols: list[str], n_days: int = 100, *, trend: float = 0.5,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Build a bars frame with n_days of business-day data.

    All symbols ramp UP by `trend` per day starting from base price 100.
    Since they all ramp at the same rate, momentum ranking is determined
    by base price differences (which we make distinct per symbol).

    The latest bar is anchored to ``end_date`` (default: date(2024, 6, 2)
    so it aligns with the historical `today=date(2024, 6, 3)` tests use
    — staying within the run-time freshness window).
    """
    end = end_date or date(2024, 6, 2)
    days = pd.bdate_range(end=pd.Timestamp(end, tz="UTC"), periods=n_days)
    rows, idx = [], []
    for i, sym in enumerate(symbols):
        base = 100.0 + i  # unique base so ranking is deterministic
        for k, ts in enumerate(days):
            c = base + k * trend
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
# run_daily_trade
# ---------------------------------------------------------------------------


def test_run_daily_trade_is_idempotent_when_record_exists(tmp_path: Path, _in_trade_window) -> None:
    """Re-firing on a day whose run JSON already exists must be a no-op.

    This is what makes launchd KeepAlive safe: after a SUCCESSFUL trade,
    subsequent retries (until tomorrow's scheduled fire) detect the
    existing record and exit cleanly without re-trading. Saturday's
    audit will then see one clean run.
    """
    universe = [f"SYM{i}" for i in range(10)]
    bars = _make_bars(universe, n_days=100)
    cache = _FakeCache(bars)
    executor = _FakeExecutor()

    # First call: produces the JSON.
    first = daily_runner.run_daily_trade(
        today=date(2024, 6, 3),
        universe=universe, cache=cache, executor=executor,
        runs_dir=tmp_path,
    )
    assert first is not None and first.exists()
    first_call_snapshot = dict(executor.last_call)
    # Mutate last_call so we can detect a 2nd invocation by it being overwritten.
    executor.last_call = {"_idempotency_marker": True}

    # Second call (simulates launchd KeepAlive retry).
    second = daily_runner.run_daily_trade(
        today=date(2024, 6, 3),
        universe=universe, cache=cache, executor=executor,
        runs_dir=tmp_path,
    )
    # Idempotent skip → no new path, no new orders.
    assert second is None
    assert executor.last_call == {"_idempotency_marker": True}, (
        "executor should NOT have been called again on the second run"
    )
    assert first_call_snapshot, "first call should have hit the executor"


# ---------------------------------------------------------------------------
# T1.5 — ATR-normalized per-symbol stops
# ---------------------------------------------------------------------------


def _make_bars_with_vol(
    symbols: list[str], n_days: int = 30, *, daily_vol: float = 0.01,
) -> pd.DataFrame:
    """Make bars where each symbol's daily returns have the requested std.

    Uses a fixed seed per symbol so tests are deterministic.
    """
    import numpy as np
    days = pd.bdate_range("2024-01-02", periods=n_days, tz="UTC")
    rows, idx = [], []
    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(seed=42 + i)
        rets = rng.normal(0.0, daily_vol, n_days - 1)
        prices = [100.0]
        for r in rets:
            prices.append(prices[-1] * (1 + r))
        for k, ts in enumerate(days):
            c = prices[k]
            rows.append({
                "open": c, "high": c * 1.005, "low": c * 0.995,
                "close": c, "volume": 1_000_000,
            })
            idx.append((sym, ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=list(BAR_COLUMNS),
    )


def test_atr_stops_tighter_for_low_vol_names() -> None:
    """A name with 0.5% daily vol should get a stop near 1.5% (3-sigma),
    far tighter than the 5% flat default."""
    bars = _make_bars_with_vol(["LOWVOL"], n_days=30, daily_vol=0.005)
    stops = daily_runner._compute_atr_normalized_stops(
        symbols=["LOWVOL"], bars=bars,
    )
    assert "LOWVOL" in stops
    # 3-sigma of 0.5% vol = 1.5% → expect stop near that, well below 5%.
    assert stops["LOWVOL"] < 0.025
    assert stops["LOWVOL"] >= daily_runner.ATR_MIN_STOP


def test_atr_stops_capped_at_operator_floor_for_high_vol() -> None:
    """A high-vol name (5% daily) would suggest a 15% stop; clip at 5%."""
    bars = _make_bars_with_vol(["HIGHVOL"], n_days=30, daily_vol=0.05)
    stops = daily_runner._compute_atr_normalized_stops(
        symbols=["HIGHVOL"], bars=bars,
    )
    # Operator's hard rule: max stop is STOP_LOSS_PCT (5%).
    assert stops["HIGHVOL"] == daily_runner.STOP_LOSS_PCT


def test_atr_stops_fallback_for_short_history() -> None:
    """A symbol with insufficient bars falls back to the flat default."""
    bars = _make_bars_with_vol(["SHORT"], n_days=5, daily_vol=0.01)
    stops = daily_runner._compute_atr_normalized_stops(
        symbols=["SHORT"], bars=bars,
    )
    # Default lookback is 20; 5 bars insufficient → falls back to floor.
    assert stops["SHORT"] == daily_runner.STOP_LOSS_PCT


# ---------------------------------------------------------------------------
# T1.3 — Vol-targeting
# ---------------------------------------------------------------------------


def test_vol_target_scales_down_high_vol_book() -> None:
    """A portfolio of high-vol names with realized vol > target should
    have its weights scaled DOWN."""
    import yaml

    from quant.config import DEFAULT_CONFIG_PATH, Config
    config = Config.model_validate(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))

    # Three names, each at 3% daily vol → ~48% annualized.
    # Equal-weight 1/3 each → portfolio vol ≈ 30%+ annualized.
    # Target is 10% → scale should be < 1.
    bars = _make_bars_with_vol(
        ["A", "B", "C"], n_days=80, daily_vol=0.03,
    )
    weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    scaled = daily_runner._apply_vol_target(
        target_weights=weights, bars=bars, config=config, lookback=60,
    )
    # Gross exposure should drop meaningfully.
    pre = sum(weights.values())
    post = sum(scaled.values())
    assert post < pre * 0.8, (
        f"high-vol book should scale down by >20% to hit 10% target; "
        f"got pre={pre:.3f}, post={post:.3f}"
    )


def test_vol_target_returns_original_when_bars_unavailable() -> None:
    """Fail-safe: insufficient/empty bars → weights unchanged."""
    import yaml

    from quant.config import DEFAULT_CONFIG_PATH, Config
    config = Config.model_validate(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))
    empty = pd.DataFrame()
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    scaled = daily_runner._apply_vol_target(
        target_weights=weights, bars=empty, config=config,
    )
    assert scaled == weights


# ---------------------------------------------------------------------------
# Kill switch (existing tests follow)
# ---------------------------------------------------------------------------


def test_run_daily_trade_refuses_stale_bars(
    tmp_path: Path, _in_trade_window,
) -> None:
    """T3.18 — if the cached bars are too old (latest > 5 days from
    today), the runner refuses to trade. Operational safety: signals
    based on week-old data are worse than not trading at all.
    """
    # Build bars whose latest date is way before today.
    universe = [f"SYM{i}" for i in range(5)]
    bars = _make_bars(universe, n_days=30)
    cache = _FakeCache(bars)
    executor = _FakeExecutor()
    # bars' latest date defaults to ~2024-01-30; "today" is far in the
    # future → stale_days > 5 days.
    far_future = date(2030, 1, 1)
    with pytest.raises(RuntimeError, match="stale"):
        daily_runner.run_daily_trade(
            today=far_future,
            universe=universe, cache=cache, executor=executor,
            runs_dir=tmp_path,
        )
    # No record persisted.
    assert not (tmp_path / f"{far_future.isoformat()}.json").exists()


def test_sector_cap_trims_overweight_sector() -> None:
    """If a sector totals > MAX_SECTOR_WEIGHT, names in it are
    proportionally trimmed; the freed weight goes to cash (not
    redistributed to other sectors — preserves the ensemble signal)."""
    # 4 banks at 12% each = 48% Financials. With max=30%, scale = 30/48
    # = 0.625, so each bank → 7.5%.
    weights = {
        "JPM": 0.12, "BAC": 0.12, "GS": 0.12, "MS": 0.12,
        "AAPL": 0.05, "MSFT": 0.05,    # Tech, under cap
    }
    sectors = {
        "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
        "MS": "Financials",
        "AAPL": "Information Technology", "MSFT": "Information Technology",
    }
    out = daily_runner._apply_sector_cap(weights, sectors, max_sector_weight=0.30)

    # Bank weights trimmed proportionally.
    assert sum(out[s] for s in ["JPM", "BAC", "GS", "MS"]) == pytest.approx(0.30)
    for s in ["JPM", "BAC", "GS", "MS"]:
        assert out[s] == pytest.approx(0.075)
    # Tech weights unchanged (under cap).
    assert out["AAPL"] == 0.05
    assert out["MSFT"] == 0.05
    # Freed weight is dropped (not added to tech) — total sum < original.
    assert sum(out.values()) < sum(weights.values())


def test_sector_cap_passes_through_when_under_cap() -> None:
    """No sector exceeds the cap → weights pass through unchanged."""
    weights = {"JPM": 0.10, "AAPL": 0.10, "XOM": 0.10}
    sectors = {"JPM": "Financials", "AAPL": "Information Technology", "XOM": "Energy"}
    out = daily_runner._apply_sector_cap(weights, sectors, max_sector_weight=0.30)
    assert out == weights


def test_sector_cap_handles_unmapped_symbols() -> None:
    """Names not in the sector map pass through unchanged (we can't
    enforce a cap if we don't know the sector)."""
    weights = {"AAPL": 0.05, "WEIRDCO": 0.10}
    sectors = {"AAPL": "Information Technology"}   # WEIRDCO unmapped
    out = daily_runner._apply_sector_cap(weights, sectors, max_sector_weight=0.30)
    assert out["WEIRDCO"] == 0.10   # passed through


def test_sector_cap_loads_real_map_from_csv() -> None:
    """End-to-end: load_sector_map reads the production CSV correctly."""
    from quant.data.universe import load_sector_map
    sector_map = load_sector_map()
    # Spot-check a couple known names.
    assert sector_map.get("AAPL") == "Information Technology"
    assert sector_map.get("JPM") == "Financials"
    # Map should cover the full top-50 universe.
    assert len(sector_map) >= 50


def test_kill_switch_tripped_when_drawdown_exceeds_threshold(tmp_path: Path) -> None:
    """If equity is more than threshold% below peak (computed from run JSONs),
    the kill switch trips. Reads peak across the persisted run history."""
    # Lay down 3 runs with equity series 100k → 110k (peak) → 92k.
    # Current equity 90k → drawdown vs peak = (90k - 110k) / 110k = -18.2%.
    # Threshold 0.15 → tripped.
    from quant.agent.log import save_daily_run
    from quant.execution.alpaca_executor import ExecutionReport
    for d, eq in [
        (date(2024, 6, 3), 100_000.0),
        (date(2024, 6, 4), 110_000.0),
        (date(2024, 6, 5),  92_000.0),
    ]:
        rep = ExecutionReport(
            env="paper", timestamp=datetime.now(UTC),
            account_equity_before=eq, positions_before={},
            target_weights={}, proposed_orders=[], submitted_orders=[],
            dry_run=False, notes="",
        )
        save_daily_run(
            run_date=d, strategy_name="x", strategy_params={},
            target_weights={}, signal_prices={}, execution_report=rep,
            runs_dir=tmp_path,
        )
    tripped, peak, dd = daily_runner._kill_switch_tripped(
        current_equity=90_000.0, runs_dir=tmp_path, threshold=0.15,
    )
    assert tripped is True
    assert peak == 110_000.0
    assert dd < -0.15


def test_kill_switch_does_not_trip_when_drawdown_within_threshold(tmp_path: Path) -> None:
    """A modest drawdown (within threshold) does NOT trip the kill switch."""
    from quant.agent.log import save_daily_run
    from quant.execution.alpaca_executor import ExecutionReport
    for d, eq in [
        (date(2024, 6, 3), 100_000.0),
        (date(2024, 6, 4), 105_000.0),    # peak
    ]:
        rep = ExecutionReport(
            env="paper", timestamp=datetime.now(UTC),
            account_equity_before=eq, positions_before={},
            target_weights={}, proposed_orders=[], submitted_orders=[],
            dry_run=False, notes="",
        )
        save_daily_run(
            run_date=d, strategy_name="x", strategy_params={},
            target_weights={}, signal_prices={}, execution_report=rep,
            runs_dir=tmp_path,
        )
    # Current 100k vs peak 105k = -4.8%, within 15% threshold.
    tripped, peak, dd = daily_runner._kill_switch_tripped(
        current_equity=100_000.0, runs_dir=tmp_path, threshold=0.15,
    )
    assert tripped is False
    assert peak == 105_000.0
    assert dd > -0.15


def test_kill_switch_handles_no_history(tmp_path: Path) -> None:
    """Fresh install (no run JSONs yet) → no peak to compare → don't trip."""
    tripped, peak, dd = daily_runner._kill_switch_tripped(
        current_equity=100_000.0, runs_dir=tmp_path / "nope", threshold=0.15,
    )
    assert tripped is False
    assert dd == 0.0


def test_pipeline_snapshot_exposes_drift_signal() -> None:
    """The snapshot must include both code constants AND config values for
    the same risk knob so the analyst can spot drift. Regression guard
    for the June 2026 incident (20% in code vs 5% in config went unnoticed)."""
    import yaml

    from quant.agent.monthly_review import _build_pipeline_snapshot
    from quant.config import DEFAULT_CONFIG_PATH, Config
    config = Config.model_validate(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))
    snap = _build_pipeline_snapshot(config)
    # Operator hard rules from code are exposed
    code_rules = snap["operator_hard_rules_in_code"]
    assert "MAX_POSITION_WEIGHT" in code_rules
    assert "MAX_DRAWDOWN_KILL" in code_rules
    # Config values for the SAME knobs are exposed alongside
    cfg_vals = snap["config_yaml_values"]
    assert "risk_max_position_weight" in cfg_vals
    assert "risk_max_drawdown_kill" in cfg_vals
    # Wiring status flags
    wiring = snap["wiring_status"]
    assert "drawdown_kill_switch_active_in_daily_trade" in wiring
    assert "vol_targeting_active_in_daily_trade" in wiring
    # Industry-norm reference values
    norms = snap["industry_norms_for_comparison"]
    assert "max_position_weight_institutional" in norms


def test_pipeline_snapshot_position_cap_aligned_with_config() -> None:
    """REGRESSION GUARD for the original drift: the hardcoded operator's
    position cap MUST match configs/default.yaml. The June 2026 incident
    was 20% in code vs 5% in config — silent 4× looser than policy.

    If this test ever fails, either:
      (a) someone changed the code constant without updating yaml, or
      (b) someone changed yaml without updating the code constant.
    Both are bugs of the same class.
    """
    import yaml

    from quant.agent.monthly_review import _build_pipeline_snapshot
    from quant.config import DEFAULT_CONFIG_PATH, Config
    config = Config.model_validate(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))
    snap = _build_pipeline_snapshot(config)
    code_val = snap["operator_hard_rules_in_code"]["MAX_POSITION_WEIGHT"]
    cfg_val = snap["config_yaml_values"]["risk_max_position_weight"]
    assert code_val == cfg_val, (
        f"POSITION CAP DRIFT: code={code_val}, config={cfg_val}. "
        "These must stay in sync — they're the same operator rule."
    )


def test_run_daily_trade_skips_outside_window(
    tmp_path: Path, monkeypatch
) -> None:
    """If wall-clock is outside 09:00-15:35 ET on the trade day, the routine
    exits early without trading. Covers both ends:
      - early off-schedule launchd KeepAlive fire (e.g. plist reload at 23:00 ET)
      - late KeepAlive-loop kill switch (after 15:35 ET).
    """
    monkeypatch.setattr(
        daily_runner, "_outside_trade_window", lambda _d: True,
    )
    universe = [f"SYM{i}" for i in range(5)]
    cache = _FakeCache(_make_bars(universe, n_days=10))
    executor = _FakeExecutor()
    result = daily_runner.run_daily_trade(
        today=date(2024, 6, 3),
        universe=universe, cache=cache, executor=executor,
        runs_dir=tmp_path,
    )
    assert result is None
    # No persistence happened.
    assert not (tmp_path / "2024-06-03.json").exists()
    # No broker calls.
    assert executor.last_call == {}


def test_outside_trade_window_inside_returns_false(monkeypatch) -> None:
    """Helper unit test: 10:00 ET on the trade date is inside the window."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class _FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 6, 3, 10, 0, tzinfo=ZoneInfo("America/New_York"))

    import datetime as _dt
    real_dt = _dt.datetime
    _dt.datetime = _FakeDatetime   # type: ignore[misc]
    try:
        assert daily_runner._outside_trade_window(date(2024, 6, 3)) is False
    finally:
        _dt.datetime = real_dt   # type: ignore[misc]


def test_outside_trade_window_too_early_returns_true(monkeypatch) -> None:
    """05:00 ET on the trade date is BEFORE the 08:00 window open → out."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class _FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 6, 3, 5, 0, tzinfo=ZoneInfo("America/New_York"))

    import datetime as _dt
    real_dt = _dt.datetime
    _dt.datetime = _FakeDatetime   # type: ignore[misc]
    try:
        assert daily_runner._outside_trade_window(date(2024, 6, 3)) is True
    finally:
        _dt.datetime = real_dt   # type: ignore[misc]


def test_outside_trade_window_winter_pre_market_is_inside(monkeypatch) -> None:
    """T-audit fix: 08:35 ET (CST 21:35 in US winter) must be IN the window.

    Regression for the winter-TZ bug — the legacy 09:00 ET floor silently
    disabled trading from Nov-Mar each year when the China-CST launchd
    fire landed at 08:35 ET. The 08:00 ET floor admits pre-market submit.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class _FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            # 08:35 ET on a trading day — the winter-CST fire point.
            return datetime(2024, 12, 3, 8, 35, tzinfo=ZoneInfo("America/New_York"))

    import datetime as _dt
    real_dt = _dt.datetime
    _dt.datetime = _FakeDatetime   # type: ignore[misc]
    try:
        assert daily_runner._outside_trade_window(date(2024, 12, 3)) is False
    finally:
        _dt.datetime = real_dt   # type: ignore[misc]


def test_outside_trade_window_too_late_returns_true(monkeypatch) -> None:
    """16:00 ET on the trade date is AFTER the 15:35 deadline → out."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class _FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 6, 3, 16, 0, tzinfo=ZoneInfo("America/New_York"))

    import datetime as _dt
    real_dt = _dt.datetime
    _dt.datetime = _FakeDatetime   # type: ignore[misc]
    try:
        assert daily_runner._outside_trade_window(date(2024, 6, 3)) is True
    finally:
        _dt.datetime = real_dt   # type: ignore[misc]


def test_run_daily_trade_persists_a_json_record(tmp_path: Path, _in_trade_window) -> None:
    """The runner must write data/agent/runs/<date>.json so the report
    routine can find it later in the day."""
    universe = [f"SYM{i}" for i in range(20)]
    bars = _make_bars(universe, n_days=100)
    cache = _FakeCache(bars)
    executor = _FakeExecutor()

    path = daily_runner.run_daily_trade(
        today=date(2024, 6, 1),
        universe=universe,
        cache=cache,
        executor=executor,
        runs_dir=tmp_path,
        dry_run=True,
    )
    assert path.exists()
    assert path.name == "2024-06-01.json"

    # File contents have the expected top-level keys.
    payload = json.loads(path.read_text())
    for key in ("date", "strategy_name", "target_weights", "signal_prices",
                "execution_report"):
        assert key in payload


def test_run_daily_trade_passes_operator_constants_to_executor(tmp_path: Path, _in_trade_window) -> None:
    """20% cap and 5% stop-loss are HARD-CODED at the agent level —
    verify they flow through to the executor call regardless of
    upstream config defaults."""
    universe = [f"SYM{i}" for i in range(20)]
    bars = _make_bars(universe, n_days=100)
    cache = _FakeCache(bars)
    executor = _FakeExecutor()

    daily_runner.run_daily_trade(
        today=date(2024, 6, 1),
        universe=universe,
        cache=cache,
        executor=executor,
        runs_dir=tmp_path,
        dry_run=True,
    )
    assert executor.last_call["stop_loss_pct"] == daily_runner.STOP_LOSS_PCT == 0.05
    # Operator's per-trade cap: 20%. Concentrated-bet policy. MUST match
    # configs/default.yaml's risk.max_position_weight. The analyst's
    # monthly self-audit flags any drift between the two; this test is
    # the build-time regression guard for the same invariant.
    assert executor.last_call["max_position_weight"] == daily_runner.MAX_POSITION_WEIGHT == 0.20


def test_run_daily_trade_includes_held_names_in_signal_prices(tmp_path: Path, _in_trade_window) -> None:
    """If we currently hold a name not in the target book, the runner
    must still pass its signal price so the executor can size the
    close-out order."""
    universe = [f"SYM{i}" for i in range(20)]
    bars = _make_bars(universe, n_days=100)
    cache = _FakeCache(bars)
    # Hold a name the strategy WON'T target (SYM0 is the lowest-momentum).
    # Actually the strategy picks the TOP 10 by momentum; SYM19 is highest.
    # We hold SYM5 which is below the median, so it won't make the cut.
    executor = _FakeExecutor(current_positions={"SYM5": 100})

    daily_runner.run_daily_trade(
        today=date(2024, 6, 1),
        universe=universe,
        cache=cache,
        executor=executor,
        runs_dir=tmp_path,
        dry_run=True,
    )
    # SYM5 should be in the signal_prices passed to the executor.
    assert "SYM5" in executor.last_call["signal_prices"]


def test_run_daily_trade_raises_on_empty_bars(tmp_path: Path, _in_trade_window) -> None:
    """Empty bars (cache or Alpaca outage) → hard failure, no silent
    'we just won't trade' behavior."""
    empty = pd.DataFrame(
        columns=list(BAR_COLUMNS),
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )
    cache = _FakeCache(empty)
    executor = _FakeExecutor()
    with pytest.raises(RuntimeError, match="no bars"):
        daily_runner.run_daily_trade(
            today=date(2024, 6, 1),
            universe=["AAPL", "MSFT"],
            cache=cache,
            executor=executor,
            runs_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# run_daily_report
# ---------------------------------------------------------------------------


class _RecordingEmailSender:
    """EmailSender stand-in that captures send() calls."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, *, subject, body_text, body_html=None, recipient=None):
        self.sent.append({
            "subject": subject,
            "body_text": body_text,
            "body_html": body_html,
            "recipient": recipient,
        })


def test_run_daily_report_emails_the_persisted_log(tmp_path: Path) -> None:
    """Render-and-email loads the saved JSON and produces a subject + body."""
    # Pre-populate one day's record via the same save function.
    fake_exec_report = ExecutionReport(
        env="paper",
        timestamp=datetime.now(UTC),
        account_equity_before=100_000.0,
        positions_before={},
        target_weights={"AAPL": 0.1, "MSFT": 0.1},
        proposed_orders=[],
        submitted_orders=[
            SubmittedOrder(symbol="AAPL", side="buy", qty=50,
                           status="submitted", role="entry"),
            SubmittedOrder(symbol="AAPL", side="sell", qty=50,
                           status="submitted", role="stop_loss",
                           stop_price=185.0),
        ],
        dry_run=False,
        notes="test",
    )
    save_daily_run(
        run_date=date(2024, 6, 1),
        strategy_name="xsec_momo_60_5_10",
        strategy_params={},
        target_weights={"AAPL": 0.1, "MSFT": 0.1},
        signal_prices={"AAPL": 195.0, "MSFT": 405.0},
        execution_report=fake_exec_report,
        runs_dir=tmp_path,
    )

    sender = _RecordingEmailSender()
    subject = daily_runner.run_daily_report(
        for_date=date(2024, 6, 1),
        runs_dir=tmp_path,
        email_sender=sender,  # type: ignore[arg-type]
    )
    assert len(sender.sent) == 1
    assert subject.startswith("quant agent")
    assert "AAPL" in sender.sent[0]["body_text"]


def test_run_daily_report_raises_when_no_log_exists(tmp_path: Path) -> None:
    """Missing log = the morning routine failed. Surface it loudly."""
    sender = _RecordingEmailSender()
    with pytest.raises(RuntimeError, match="no daily run record"):
        daily_runner.run_daily_report(
            for_date=date(2024, 6, 1),
            runs_dir=tmp_path,
            email_sender=sender,  # type: ignore[arg-type]
        )
