"""daily_runner.py — the agent's once-a-day trade routine and report routine.

The two entry points (both also exposed via console-scripts in
pyproject.toml so launchd can invoke them by name):

- ``run_daily_trade()`` — at ~09:35 ET, compute target weights from
  yesterday's close, submit market entries with atomic stop-losses,
  persist the full record.
- ``run_daily_report(for_date=...)`` — at ~16:05 ET, load the day's
  persisted record, render markdown, email it.

Both are deliberately small: most logic lives in the existing modules
(strategy, executor, log, reports, email_sender). This file is the
orchestration glue.

Failure handling
----------------
If anything raises inside the trade routine, the CLI wrapper catches it,
tries to email the operator with the traceback, then re-raises so
launchd notices the non-zero exit and (per OS log) the operator can
investigate. Email failure on top of trade failure is logged to stderr
and does NOT loop into a new error email — that's how cascades start.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from quant.agent.email_sender import EmailSender
from quant.agent.ensemble import (
    EnsembleState,
    build_strategies,
    compute_ensemble_targets,
    load_ensemble_state,
    save_ensemble_state,
    update_trail_highs,
)
from quant.agent.log import (
    DEFAULT_RUNS_DIR,
    load_daily_run,
    save_daily_run,
)
from quant.agent.reports import render_daily_report
from quant.backtest.types import Snapshot
from quant.config import Config, load_config
from quant.data.alpaca_client import AlpacaDataClient
from quant.data.cache import BarsCache
from quant.data.universe import load_active_universe, load_sector_map
from quant.execution.alpaca_executor import AlpacaExecutor

logger = logging.getLogger(__name__)


# OPERATOR'S HARD RULES — hard-coded here, deliberately NOT in YAML and
# NOT in the tunable StrategyParams file. The auto-improver cannot
# change these. If you want to change them, edit the source.
#
# These MUST stay in sync with configs/default.yaml's `risk:` block;
# the monthly pipeline self-audit (in monthly_review) cross-checks the
# two and surfaces any drift as a critical finding. If you change one,
# change both.
STOP_LOSS_PCT = 0.05            # 5% stop on every entry
MAX_POSITION_WEIGHT = 0.20      # 20% per-trade cap — operator policy.
                                # This is concentrated-bet territory
                                # (institutional norm is 3-5%), but the
                                # operator wants the option for HRP to
                                # put real size on high-conviction names
                                # when all three strategies agree. The
                                # accompanying configs/default.yaml
                                # `risk.max_position_weight` is set to
                                # 0.20 to match — the analyst's monthly
                                # pipeline self-audit will scream if
                                # these ever drift apart again.
# Drawdown kill switch threshold. If equity is more than this far below
# its running peak (across the persisted daily-run history), the daily
# trade routine refuses to open NEW entries. Existing positions retain
# their stops; the system stops adding risk during a meltdown until
# the operator intervenes. Mirrors configs/default.yaml: risk.max_drawdown_kill.
MAX_DRAWDOWN_KILL = 0.15        # -15% halts new entries

# ATR-normalized stop parameters. The per-symbol stop_pct is computed as
#   stop_pct = clip(ATR_MULTIPLIER * realized_vol, ATR_MIN_STOP, STOP_LOSS_PCT)
# where realized_vol = std of last ATR_VOL_LOOKBACK daily returns.
# Effect: low-vol names get tighter stops (e.g. 2-3% for KO) so we don't
# sit through unnecessary drawdowns; high-vol names stay capped at the
# operator's 5% floor (NVDA-like names still risk whipsaw, but the
# alternative — looser stops than 5% — violates operator policy).
ATR_VOL_LOOKBACK = 20            # daily-return std window
ATR_MULTIPLIER = 3.0             # 3-sigma stop on average
ATR_MIN_STOP = 0.005             # 0.5% floor — don't whipsaw on micro-noise
# Slippage threshold for the post-fill stop repair. If actual fill price
# differs from signal price by more than this %, the OTO bracket's stop
# is in the wrong place and we replace it.
STOP_REPAIR_DRIFT_THRESHOLD = 0.01    # 1%
# Seconds to wait for OTO fills before running the stop-repair pass.
# Live: ~30s is enough for market-on-open + 5-min OTO submissions to
# fill. Tests set this to 0.
STOP_REPAIR_WAIT_SECONDS = 30.0

# Sector concentration cap (operator hard rule). No single GICS sector
# may exceed this fraction of equity. 0.30 = 30% per sector — looser
# than the 20-25% institutional norm because the operator's already
# accepted concentrated single-name risk via MAX_POSITION_WEIGHT=0.20,
# so the sector cap mostly guards against IMPLICIT correlation (e.g.,
# 4 banks at 7% each = 28% bank exposure with hidden correlation).
# When exceeded, names within the offending sector are proportionally
# trimmed; the freed weight goes to cash (no rebalancing into other
# sectors — keeping the original ensemble signal intact).
MAX_SECTOR_WEIGHT = 0.30

# Tunable parameters live in StrategyParams (persisted to
# data/agent/strategy_params.json); the auto-improver may swap them
# after passing safety gates. Defaults are the v1 starting point.
LOOKBACK_BUFFER_DAYS = 30       # extra bars to cover non-trading days


# ---------------------------------------------------------------------------
# run_daily_trade — the morning routine
# ---------------------------------------------------------------------------


# Wall-clock guard. If the launchd KeepAlive auto-retry is still kicking
# this many hours after the scheduled fire (09:35 ET), the trade window
# is mostly gone — better to skip than rebalance into the last 30 minutes
# of the session on stale signals. Set to 6h post-open = 15:35 ET (the
# close is 16:00 ET; 25 min before close is too risky).
_TRADE_DEADLINE_HOURS_AFTER_OPEN = 6


def _market_is_closed_today(today: date, executor: Any = None) -> tuple[bool, str]:
    """Query Alpaca's market-calendar API for ``today``.

    Returns ``(is_closed, reason)`` — reason is a human-readable string
    suitable for logs / emails. Failures (network, API outage) treat
    the day as OPEN so we don't accidentally skip a real trading day
    on a transient API issue — the bar-freshness check (T3.18) catches
    "we tried to trade but markets weren't actually open" downstream.

    Detection logic:
      - If today is a weekend or full holiday: no calendar entry →
        closed.
      - If today has a calendar entry: open. Half-day handling
        (Christmas Eve early close at 13:00 ET) is captured by the
        calendar's ``close`` field, which the trade-window guard
        could read in a future change. For now, half-days are
        treated as full days — we trade at 09:35 ET and the early
        close at 13:00 ET still gives us 3.5 hours before our
        15:35 ET deadline guard fires.
    """
    if executor is None:
        # No executor means no broker connection; skip the check.
        # Caller should rely on the bar-freshness check downstream.
        return False, "no executor (skipping market-calendar check)"
    try:
        from alpaca.trading.requests import GetCalendarRequest
        req = GetCalendarRequest(start=today, end=today)
        cal = executor._client.get_calendar(req)
    except Exception as e:
        # Calendar lookup failure — assume open and let other safety
        # layers catch it.
        return False, f"calendar lookup failed ({e}); assuming open"
    if not cal:
        return True, "Alpaca calendar has no entry for today (weekend/holiday)"
    return False, f"market open ({cal[0].open}–{cal[0].close} ET)"


def _outside_trade_window(today_iso_date: date) -> bool:
    """True iff current US/Eastern wall clock is NOT inside the trade window
    for today_iso_date. The window is 08:00–15:35 ET on the trade day.

    This guard covers BOTH ends:

    - Before 08:00 ET: an off-schedule launchd KeepAlive fire (which can
      happen on plist reload — "never ran" satisfies "not SuccessfulExit")
      would otherwise trade at e.g. 23:00 ET the previous evening. The
      orders would just queue overnight at the broker; functionally OK,
      but wastes idempotency budget and clobbers any state the operator
      might want to inspect before market open. Cleaner to no-op and let
      the scheduled fire do its job.

    - After 15:35 ET: KeepAlive kill switch. If the trade has been failing
      for 6+ hours after the scheduled fire, the trade window is too close
      to the close to safely rebalance. Stop retrying; tomorrow's
      scheduled fire will resume normally.

    Why 08:00 ET (not 09:00 ET): launchd fires the plist at 21:35 CST.
    In US summer (EDT, UTC-4) that maps to 09:35 ET — 5 min after open,
    ideal. In US winter (EST, UTC-5) it maps to 08:35 ET — 55 min before
    open. With a 09:00 ET floor the agent would refuse to trade for
    ~5 months/year (Nov–Mar). With 08:00 ET the winter fire is in-window
    and orders queue at Alpaca, filling at the opening cross. Market
    orders + OTO stops behave the same whether submitted at 08:35 ET
    or 09:35 ET because they execute against the open auction.

    Returns False (i.e. "we're in the window, please trade") for now_ET
    on the same calendar date as today_iso_date, between 08:00 and 15:35.
    """
    from datetime import datetime, time
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    now_et = datetime.now(et)
    if now_et.date() != today_iso_date:
        # Date doesn't match today's date in ET → out of window
        # (e.g. retry past CST midnight where today=CST-tomorrow but
        # ET wall is still the original trade day).
        return True
    window_open = time(8, 0)
    window_close = time(9 + _TRADE_DEADLINE_HOURS_AFTER_OPEN, 35)   # 15:35 ET
    return not (window_open <= now_et.time() <= window_close)


def _save_killswitch_record(
    *,
    today: date,
    current_equity: float,
    peak: float,
    drawdown_pct: float,
    runs_dir: Path | None = None,
) -> None:
    """Persist a sentinel run record indicating the kill switch tripped.

    Writing this means the audit + report routines see "the system DID
    fire today, but chose not to trade". Without this sentinel, the audit
    would scream "no run record for today" which is misleading — the run
    DID happen, it just deliberately exited early.
    """
    import json as _json

    from quant.agent.log import DEFAULT_RUNS_DIR, _atomic_write_text
    out_dir = runs_dir or DEFAULT_RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": today.isoformat(),
        "strategy_name": "(kill_switch_tripped)",
        "strategy_params": {
            "kill_switch": {
                "tripped": True,
                "current_equity": current_equity,
                "peak_equity": peak,
                "drawdown_pct": drawdown_pct,
                "threshold_pct": -MAX_DRAWDOWN_KILL,
            },
        },
        "target_weights": {},
        "signal_prices": {},
        "execution_report": {
            "env": "paper",
            "account_equity_before": current_equity,
            "positions_before": {},
            "target_weights": {},
            "proposed_orders": [],
            "submitted_orders": [],
            "dry_run": False,
            "notes": (
                f"DRAWDOWN KILL SWITCH TRIPPED — "
                f"{drawdown_pct:.2%} drawdown vs peak ${peak:,.2f} "
                f"(threshold -{MAX_DRAWDOWN_KILL:.0%}). Refused to open new "
                "entries. Operator review required."
            ),
        },
    }
    out_path = out_dir / f"{today.isoformat()}.json"
    _atomic_write_text(out_path, _json.dumps(payload, default=str, indent=2))


def _kill_switch_tripped(
    current_equity: float,
    *,
    runs_dir: Path | None = None,
    threshold: float = MAX_DRAWDOWN_KILL,
) -> tuple[bool, float, float]:
    """Check the drawdown kill switch against persisted equity history.

    Walks every saved daily-run JSON to find the historical peak equity,
    compares to ``current_equity``, and trips if the drawdown exceeds
    ``threshold``. Returns ``(tripped, peak, drawdown_pct)``.

    Why compute fresh from history each run rather than persisted state:
    the daily-trade routine is stateless across runs (each invocation is
    a clean process). The run JSONs are the source of truth for equity
    history — they're already written by save_daily_run.

    Trip semantics: when tripped, ``run_daily_trade`` REFUSES to open new
    entries (returns None). Existing positions and their GTC stops are
    untouched, so the system stops adding risk but still protects what's
    held. Operator intervention is required to resume — there is no
    auto-reset.
    """
    from quant.agent.log import DEFAULT_RUNS_DIR
    out_dir = runs_dir or DEFAULT_RUNS_DIR
    if not out_dir.exists():
        return False, current_equity, 0.0

    peak = current_equity
    import json as _json
    for p in out_dir.glob("*.json"):
        try:
            payload = _json.loads(p.read_text())
            eq = payload.get("execution_report", {}).get(
                "account_equity_before", 0.0
            )
            if isinstance(eq, (int, float)) and eq > peak:
                peak = float(eq)
        except (OSError, ValueError):
            continue

    if peak <= 0:
        return False, peak, 0.0
    drawdown = (current_equity - peak) / peak
    return drawdown < -threshold, peak, drawdown


def _apply_sector_cap(
    target_weights: dict[str, float],
    sector_map: dict[str, str],
    *,
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
) -> dict[str, float]:
    """Cap per-sector exposure by proportionally trimming names within
    any sector that exceeds the cap.

    Names not in ``sector_map`` are passed through unchanged (we don't
    know what sector they're in, so we can't enforce the cap on them).
    The freed weight from trimming a sector goes to CASH — we don't
    rebalance it into other sectors because that would distort the
    ensemble's original signal mix. The operator's per-name and
    portfolio-level vol-target caps are independent.

    Example: ensemble outputs 6 banks at 7% each = 42% Financials.
    With max_sector_weight=0.30, each is trimmed by 30/42 ≈ 71% →
    each bank ends up at 5%. Total bank exposure = 30%. The remaining
    12% becomes cash.
    """
    if not target_weights or not sector_map:
        return dict(target_weights)

    # Group weights by sector.
    by_sector: dict[str, dict[str, float]] = {}
    untagged: dict[str, float] = {}
    for sym, w in target_weights.items():
        if sym in sector_map:
            by_sector.setdefault(sector_map[sym], {})[sym] = w
        else:
            untagged[sym] = w

    out: dict[str, float] = dict(untagged)
    for sector, names in by_sector.items():
        sector_total = sum(names.values())
        if sector_total <= max_sector_weight:
            # Under the cap — pass through unchanged.
            out.update(names)
        else:
            # Over the cap — proportionally trim each name in this sector.
            scale = max_sector_weight / sector_total
            for sym, w in names.items():
                out[sym] = w * scale
            logger.info(
                "sector cap: %s was %.1f%%, trimmed to %.1f%% (%d names "
                "each scaled by %.3f)",
                sector, sector_total * 100, max_sector_weight * 100,
                len(names), scale,
            )
    return out


def _build_trail_anchors(
    bars,
    target_weights: dict[str, float],
    signal_prices: dict[str, float],
) -> dict[str, float]:
    """Per-symbol price to feed into ``update_trail_highs``.

    Uses the latest bar's HIGH (not CLOSE) for each target symbol —
    that's the actual intraday peak the stock reached. A stock that
    spiked to $250 then closed at $240 gets a trail anchor of $250,
    not $240. The trailing stop is then anchored to that real peak,
    so we lock in more gain on the next bar.

    Fallback to ``signal_prices`` (which is the close) for any symbol
    whose HIGH is missing or non-positive — never worse than the
    pre-change behaviour.
    """
    anchors: dict[str, float] = {}
    for sym in target_weights:
        try:
            high = float(bars.loc[sym]["high"].iloc[-1])
            if high > 0:
                anchors[sym] = high
                continue
        except (KeyError, ValueError, AttributeError, IndexError):
            pass
        # Fallback to close.
        if sym in signal_prices:
            anchors[sym] = signal_prices[sym]
    return anchors


def _compute_atr_normalized_stops(
    *,
    symbols: list[str],
    bars,
    lookback: int = ATR_VOL_LOOKBACK,
    multiplier: float = ATR_MULTIPLIER,
    min_stop: float = ATR_MIN_STOP,
    max_stop: float = STOP_LOSS_PCT,
) -> dict[str, float]:
    """Per-symbol stop distance, scaled by realized daily vol.

    For each symbol, realized_vol = std(returns over `lookback` days).
    Returns ``multiplier * realized_vol`` clipped to [min_stop, max_stop].

    Low-vol names → tighter stops (e.g. KO at ~1% daily vol → 3% stop).
    High-vol names → stops are CAPPED at max_stop (operator's hard rule),
    so NVDA-like names still get whipsawed by the 5% cap — that's a
    policy trade-off, not a code bug. Lifting the cap would require an
    operator policy change.

    Names with insufficient bar history fall back to ``max_stop`` (the
    operator's flat-rate default — safe).
    """
    out: dict[str, float] = {}
    if bars is None or bars.empty:
        return out
    for sym in symbols:
        try:
            closes = bars.loc[sym]["close"]
            if len(closes) < lookback + 1:
                out[sym] = max_stop
                continue
            returns = closes.pct_change().dropna().tail(lookback)
            if len(returns) < 2:
                out[sym] = max_stop
                continue
            realized_vol = float(returns.std(ddof=1))
            target = multiplier * realized_vol
            # Clip to the operator's allowed band.
            out[sym] = max(min_stop, min(max_stop, target))
        except (KeyError, ValueError, AttributeError):
            # Symbol missing from bars or bad data → safe fallback.
            out[sym] = max_stop
    return out


def _apply_vol_target(
    *,
    target_weights: dict[str, float],
    bars,
    config: Config,
    lookback: int = 60,
) -> dict[str, float]:
    """Scale ``target_weights`` so portfolio realized vol ≈ vol_target.

    Uses ``quant.allocator.apply_vol_target`` with a per-symbol returns
    DataFrame computed from the last ``lookback`` days of bars.

    Long-only behavior: vol-targeting can either AMPLIFY (scale > 1,
    levering up — capped by config.risk.max_gross_leverage) or DAMPEN
    (scale < 1, reducing exposure). In practice with a long-only book,
    the protective case (scale < 1 in high-vol regimes) is what matters
    most. We accept the lever-up case too, capped at max_gross_leverage.

    Empty/insufficient-bars input → returns weights unchanged. This is
    the fail-safe path: vol-targeting is an OPTIMIZATION, not a safety
    requirement; if we can't compute it, just use the original weights.
    """
    import pandas as pd  # noqa: PLC0415 — lazy

    from quant.allocator import apply_vol_target

    if not target_weights or bars is None or bars.empty:
        return dict(target_weights)

    syms = [s for s in target_weights if target_weights[s] > 0]
    returns_by_sym: dict[str, pd.Series] = {}
    for sym in syms:
        try:
            closes = bars.loc[sym]["close"]
            r = closes.pct_change().dropna().tail(lookback)
            if len(r) >= 10:   # need some history for a sensible vol
                returns_by_sym[sym] = r
        except (KeyError, ValueError, AttributeError):
            continue

    if len(returns_by_sym) < len(syms) // 2:
        # Less than half the universe has usable history → not enough
        # signal to scale safely. Skip vol-targeting; keep original.
        return dict(target_weights)

    returns_df = pd.concat(returns_by_sym, axis=1).dropna(how="all")
    if returns_df.empty:
        return dict(target_weights)

    weights = pd.Series({s: target_weights[s] for s in returns_df.columns})
    try:
        scaled = apply_vol_target(
            weights=weights,
            strategy_returns=returns_df,
            target_vol_annual=config.risk.vol_target_annual,
            trading_days_per_year=config.evaluation.trading_days_per_year,
            max_gross_leverage=config.risk.max_gross_leverage,
        )
    except Exception as e:
        logger.warning(
            "vol-target apply failed (%s: %s) — using original weights",
            type(e).__name__, e,
        )
        return dict(target_weights)

    # Merge: scaled values for symbols we computed, originals for any
    # we couldn't (defensive).
    out = dict(target_weights)
    for sym, w in scaled.items():
        out[sym] = float(w)
    return out


def run_daily_trade(
    *,
    dry_run: bool = False,
    today: date | None = None,
    config: Config | None = None,
    universe: list[str] | None = None,
    cache: BarsCache | None = None,
    executor: AlpacaExecutor | None = None,
    runs_dir: Path | None = None,
    ensemble_state: EnsembleState | None = None,
) -> Path | None:
    """Execute today's trade cycle. Returns the path of the saved JSON log,
    or None if the routine exited early via idempotency / deadline guard.

    Every dependency can be injected for tests; in production they
    default to the real components.

    Idempotency: returns early without re-trading if today's run JSON
    already exists. This makes the routine SAFE to re-fire (e.g. via
    launchd KeepAlive after a transient failure) — only the first call
    that lands a JSON actually trades.

    Deadline: returns early without trading if the wall clock is past
    15:35 ET on ``today``. Kill switch for the KeepAlive auto-retry
    loop — once the trade window is gone, stop retrying.

    Parameters
    ----------
    dry_run
        If True, the executor reports what WOULD be submitted but
        doesn't actually send orders. Used for first install / debug.
    today
        Override "today's date" (default: system date). Useful for
        re-running a missed day or simulating ahead.
    """
    # T-audit fix H4: compute "today" from the ET wall clock, not the
    # system clock. The agent trades US markets; the trade day must be
    # the ET calendar day. The system clock is China-CST and at CST 21:35
    # equals the same ET date in both summer (09:35 EDT) and winter
    # (08:35 EST). BUT — if the launchd KeepAlive retries past CST
    # midnight (e.g., 00:30 CST Tue after a 21:35 CST Mon failure), the
    # system date jumps to Tue while the ET wall is still Mon 11:30 ET
    # (mid-trade-window). With system-local date.today() we'd compute
    # today=Tue, then _outside_trade_window compares Tue (today) vs Mon
    # (now_et.date()) and bails. ET-anchored today() prevents this:
    # both today and now_et.date() are Mon, retry proceeds correctly.
    if today is None:
        from datetime import datetime as _datetime
        from zoneinfo import ZoneInfo as _ZoneInfo
        today = _datetime.now(tz=_ZoneInfo("America/New_York")).date()
    config = config or load_config()

    # --- 0. Idempotency + deadline guards (BEFORE expensive work) ---
    # Run-record check FIRST so a retry after a successful trade is a
    # cheap no-op. Skipped on dry_run since dry runs don't persist.
    if not dry_run:
        existing = load_daily_run(today, runs_dir=runs_dir)
        if existing is not None:
            logger.info(
                "run_daily_trade: today's run already persisted "
                "(%d orders); idempotent skip.",
                len(existing.get("execution_report", {}).get("submitted_orders", [])),
            )
            return None
    if _outside_trade_window(today):
        logger.warning(
            "run_daily_trade: outside 09:00-15:35 ET trade window for %s; "
            "exiting cleanly (no-op). KeepAlive may have fired us off-schedule, "
            "or the trade window has closed.",
            today,
        )
        return None

    # Point-in-time universe (falls back to top-50 snapshot if the
    # comprehensive membership CSV isn't curated yet — see
    # reference/universe/sp500.csv and tools/curate_sp500_membership.py).
    universe = universe or load_active_universe(today)
    cache = cache or BarsCache(client=AlpacaDataClient(), root=Path("data/bars/daily"))
    executor = executor or AlpacaExecutor()
    state = ensemble_state or load_ensemble_state()

    # T4.20 — Holiday awareness. The trade-window guard above only
    # checks the wall clock; it doesn't know the market calendar. On a
    # full holiday (Christmas Day, Independence Day, etc.) the launchd
    # job still fires at 21:35 CST but the market is closed. Without
    # this check the agent would burn API calls trying to trade.
    if not dry_run:
        is_closed, reason = _market_is_closed_today(today, executor=executor)
        if is_closed:
            logger.info(
                "run_daily_trade: market closed for %s (%s); exiting cleanly.",
                today, reason,
            )
            return None

    logger.info(
        "run_daily_trade: today=%s dry_run=%s universe_size=%d env=%s "
        "hrp_weights=%s",
        today, dry_run, len(universe), executor.env, state.hrp_weights,
    )

    # --- 0b. Drawdown kill switch (operator hard rule) ---
    # If the portfolio is in a meaningful drawdown vs its historical peak,
    # stop opening NEW entries. Existing positions + their GTC stops are
    # left untouched (we still want to protect what's held). The check
    # only fires for non-dry-run; tests + debug invocations bypass it.
    if not dry_run:
        try:
            current_equity = executor.get_equity()
            tripped, peak, dd = _kill_switch_tripped(
                current_equity, runs_dir=runs_dir,
            )
            if tripped:
                logger.error(
                    "DRAWDOWN KILL SWITCH TRIPPED — equity $%.2f is %.1f%% "
                    "below peak $%.2f (threshold = -%.0f%%). Refusing to "
                    "open new entries. Existing positions + GTC stops "
                    "retained. Operator review required before resuming.",
                    current_equity, dd * 100, peak, MAX_DRAWDOWN_KILL * 100,
                )
                # Persist a "kill switch tripped" run record so the audit
                # and reports surface this prominently.
                _save_killswitch_record(
                    today=today, current_equity=current_equity, peak=peak,
                    drawdown_pct=dd, runs_dir=runs_dir,
                )
                return None
        except Exception as e:
            # Failure to READ equity shouldn't block the trade — fall
            # through and let the normal flow handle it (it'll fail
            # later if the broker is genuinely unreachable).
            logger.warning(
                "kill switch check failed (continuing): %s: %s",
                type(e).__name__, e,
            )

    # --- 1. Fetch bars covering the longest signal window ---
    # SMA(50,200) needs the most history; budget 250 trading days + buffer
    # so all three strategies have what they need.
    end = today - timedelta(days=1)   # don't request today's bar (not closed yet)
    start = end - timedelta(days=400)
    bars = cache.get_daily_bars(universe, start, end)
    if bars.empty:
        raise RuntimeError(
            f"no bars fetched for universe of {len(universe)} symbols "
            f"between {start} and {end}. Cache or Alpaca issue?"
        )

    # T-audit fix H9 — Structural integrity check. Validates the
    # standard bars contract: columns/index shape, no nulls, OHLC
    # invariants, no duplicate (symbol, timestamp), no weekend or
    # future timestamps, prices > 0. Without this a single bad bar
    # (negative price from a data outage, duplicate row from a cache
    # merge bug, future timestamp from a clock-skewed fetch) would
    # silently propagate through strategies and produce wrong signals.
    # Cheap: O(rows) with vectorised numpy ops.
    from quant.data.integrity import check_daily_bars
    check_daily_bars(bars)

    # T3.18 — Bar freshness check. The cache could return STALE data
    # (e.g. cache file not refreshed for a week, or Alpaca returned
    # an old window). If the latest bar is more than 5 calendar days
    # old, refuse to trade — the signals would be based on
    # week-old data, which is operationally dangerous.
    latest_bar_ts = bars.index.get_level_values("timestamp").max()
    latest_bar_date = latest_bar_ts.date() if hasattr(latest_bar_ts, "date") else latest_bar_ts
    stale_days = (today - latest_bar_date).days
    if stale_days > 5:
        raise RuntimeError(
            f"Latest bar in cache is {stale_days} calendar days old "
            f"(latest={latest_bar_date}, today={today}). Cache is stale; "
            "refusing to trade on week-old signals. Check the bars cache "
            "at data/bars/daily/ and re-fetch from Alpaca if needed."
        )

    # --- 2a. Graduate any AI strategies whose shadow period has ended -------
    # Shadow strategies live in state.ai_strategy_shadow_until until the
    # ISO date there has passed. On the first daily run after that date,
    # they "graduate": removed from the shadow map, given an equal-split
    # initial HRP weight. The weekly refit takes over from there.
    today_iso = today.isoformat()
    graduated: list[str] = []
    if state.ai_strategy_shadow_until:
        still_shadow: dict[str, str] = {}
        new_hrp = dict(state.hrp_weights)
        for name, until_iso in state.ai_strategy_shadow_until.items():
            if today_iso >= until_iso:
                graduated.append(name)
            else:
                still_shadow[name] = until_iso

        if graduated:
            # Renormalise: existing weights scaled down, each graduate gets equal share.
            n_total = len(new_hrp) + len(graduated)
            equal_w = 1.0 / n_total
            scale = 1.0 - equal_w * len(graduated)
            new_hrp = {k: v * scale for k, v in new_hrp.items()}
            for name in graduated:
                new_hrp[name] = equal_w

            state = replace(
                state,
                hrp_weights=new_hrp,
                ai_strategy_shadow_until=still_shadow,
            )
            save_ensemble_state(state)
            logger.info(
                "graduated %d AI strategies from shadow → active: %s",
                len(graduated), graduated,
            )

    # Names still in shadow on today's run.
    shadow_today: set[str] = set(state.ai_strategy_shadow_until.keys())

    # --- 2b. Build snapshot, run ALL strategies, combine via HRP weights ----
    ts = bars.index.get_level_values("timestamp")
    as_of = ts.max().date()
    snapshot = Snapshot.from_full_bars(bars, as_of=as_of)
    strategies = build_strategies(state, universe)
    shadow_targets: dict[str, dict[str, float]] = {}
    target_weights = compute_ensemble_targets(
        strategies, state.hrp_weights, snapshot,
        shadow_strategies=shadow_today,
        record_shadow_targets=shadow_targets,
    )
    logger.info(
        "ensemble emitted %d target positions (as_of=%s); %d strategies in shadow",
        len(target_weights), as_of, len(shadow_today),
    )

    # --- 2c. Sector concentration cap (operator hard rule) ---
    # Trim any GICS sector exceeding MAX_SECTOR_WEIGHT (30%). Defends
    # against the hidden-correlation case: 6 banks at 7% each = 42%
    # Financials with one underlying risk factor. Trim is proportional
    # within the offending sector; freed weight goes to cash (we don't
    # rebalance into other sectors — preserves the original ensemble
    # signal mix). Names not in the sector map pass through unchanged.
    sector_map = load_sector_map()
    target_weights = _apply_sector_cap(target_weights, sector_map)

    # --- 3. Signal prices = the last available close per target name. ---
    signal_prices: dict[str, float] = {}
    for sym in target_weights:
        try:
            sym_closes = bars.loc[sym]["close"]
            signal_prices[sym] = float(sym_closes.iloc[-1])
        except KeyError:
            # Strategy targeted a name with no bar data — shouldn't
            # happen since the strategy only ranks names it can see,
            # but defensive.
            logger.warning("no signal price for %s; skipping", sym)
    # ALSO include any currently-held names not in targets so the
    # executor can compute their close-out notionals.
    held = executor.get_positions()
    for sym in held:
        if sym in signal_prices:
            continue
        try:
            sym_closes = bars.loc[sym]["close"]
            signal_prices[sym] = float(sym_closes.iloc[-1])
        except KeyError:
            # Held name not in the universe (manual prior trade, etc.).
            # The executor will skip closing it; warn so the operator
            # notices.
            logger.warning(
                "held name %s has no bars in current universe; agent "
                "won't auto-close it. Liquidate manually if desired.",
                sym,
            )

    # --- 3b. Update the trailing-stop ratchet -----------------------------
    # Compute new trail_high for every name we're about to hold. Bumped
    # to today's HIGH bar (not close) if higher than the prior tracked
    # high; fresh entries seeded with today's high; names being flatted
    # are dropped. Using HIGH not CLOSE means a stock that spiked
    # intraday to $250 but closed at $240 STILL ratchets the trail to
    # $250 — capturing the actual peak the stock hit, not just where
    # the day finished. The trailing stop is anchored to that peak,
    # so we lock in more gain when winners spike intraday.
    trail_anchors = _build_trail_anchors(bars, target_weights, signal_prices)
    new_trail = update_trail_highs(
        prev_trail=dict(state.trail_high),
        new_targets=set(target_weights.keys()),
        signal_prices=trail_anchors,
    )

    # --- 3c. ATR-normalized per-symbol stop distances (T1.5) ----------
    # Replace the flat 5% stop with vol-adjusted stops: tighter on
    # low-vol names, capped at STOP_LOSS_PCT on high-vol names.
    stop_pcts = _compute_atr_normalized_stops(
        symbols=list(target_weights.keys()),
        bars=bars,
    )
    logger.info(
        "ATR-normalized stops: %d symbols; mean=%.2f%%, range=[%.2f%%, %.2f%%]",
        len(stop_pcts),
        100 * (sum(stop_pcts.values()) / len(stop_pcts)) if stop_pcts else 0,
        100 * min(stop_pcts.values()) if stop_pcts else 0,
        100 * max(stop_pcts.values()) if stop_pcts else 0,
    )

    # --- 3d. Vol-target the portfolio (T1.3) --------------------------
    # Scale gross exposure so realized portfolio vol ≈ config target.
    # The protective case is when realized vol is HIGH: we scale DOWN,
    # reducing exposure during stormy regimes. Long-only bookkeeping
    # is handled inside _apply_vol_target.
    scaled_weights = _apply_vol_target(
        target_weights=target_weights,
        bars=bars,
        config=config,
    )
    pre_gross = sum(target_weights.values())
    post_gross = sum(scaled_weights.values())
    if abs(post_gross - pre_gross) > 1e-6:
        logger.info(
            "vol-targeting: gross exposure %.4f → %.4f (scale %.3f) "
            "to hit %.0f%% target vol",
            pre_gross, post_gross,
            (post_gross / pre_gross) if pre_gross > 0 else 0.0,
            config.risk.vol_target_annual * 100,
        )

    # --- 4. Submit via the executor's agent flow ---
    report = executor.submit_daily_rebalance(
        target_weights=scaled_weights,
        signal_prices=signal_prices,
        stop_loss_pct=STOP_LOSS_PCT,
        max_position_weight=MAX_POSITION_WEIGHT,
        dry_run=dry_run,
        notes=f"daily trade {today.isoformat()}"
              + (" (dry-run)" if dry_run else ""),
        trail_highs=new_trail,
        trail_pct=state.trail_pct,
        stop_pcts=stop_pcts,
        repair_stops_after_fill=not dry_run,
        fill_wait_seconds=STOP_REPAIR_WAIT_SECONDS,
    )

    # Persist the new trail map back to disk (skipping on dry-run so test
    # / debug invocations don't pollute live state).
    if not dry_run:
        state = replace(state, trail_high=new_trail)
        save_ensemble_state(state)

    # --- 5. Persist the full record. ---
    # Ensemble means there's no single "strategy" — record the names of
    # all three plus their HRP weights, so the daily report and reviews
    # can show what was active and how.
    path = save_daily_run(
        run_date=today,
        strategy_name=f"ensemble({len(strategies)})",
        strategy_params={
            "ensemble_state": {
                "sma_fast": state.sma_fast,
                "sma_slow": state.sma_slow,
                "mr_lookback": state.mr_lookback,
                "mr_threshold_pct": state.mr_threshold_pct,
                "mr_allow_short": state.mr_allow_short,
                "mr_vol_normalize": state.mr_vol_normalize,
                "mr_vol_window": state.mr_vol_window,
                "mr_vol_multiplier": state.mr_vol_multiplier,
                "xsec_top_k": state.xsec_top_k,
                "xsec_lookback": state.xsec_lookback,
                "xsec_skip": state.xsec_skip,
                "hrp_weights": state.hrp_weights,
                "last_hrp_refit_date": state.last_hrp_refit_date,
                "ai_strategy_names": list(state.ai_strategy_names),
                "ai_strategy_shadow_until": dict(state.ai_strategy_shadow_until),
                "ai_strategies_graduated_today": graduated,
                "shadow_targets_today": shadow_targets,
                "trail_high": dict(new_trail),
                "trail_pct": state.trail_pct,
            },
            "stop_loss_pct": STOP_LOSS_PCT,
            "max_position_weight": MAX_POSITION_WEIGHT,
        },
        target_weights=target_weights,
        signal_prices=signal_prices,
        execution_report=report,
        runs_dir=runs_dir,
    )
    logger.info("saved daily run to %s", path)
    return path


# ---------------------------------------------------------------------------
# run_daily_report — the close-of-day email
# ---------------------------------------------------------------------------


def run_daily_report(
    *,
    for_date: date | None = None,
    runs_dir: Path | None = None,
    email_sender: EmailSender | None = None,
) -> str:
    """Render today's report and email it. Returns the subject sent.

    Loads the persisted JSON from earlier in the day, renders markdown,
    emails to the configured REPORT_TO recipient. Raises if no run was
    persisted for ``for_date``.
    """
    for_date = for_date or date.today()
    payload = load_daily_run(for_date, runs_dir=runs_dir)
    if payload is None:
        raise RuntimeError(
            f"no daily run record for {for_date.isoformat()} at "
            f"{runs_dir or DEFAULT_RUNS_DIR}. Did the morning trade routine run?"
        )

    # Reconstitute just enough of ExecutionReport for the renderer.
    # The renderer reads only a subset — order rows, equity, env, etc.
    # We could fully deserialize but a SimpleNamespace is easier.
    er = payload["execution_report"]
    report_view = _ExecutionReportView(
        env=er.get("env", "paper"),
        timestamp=er.get("timestamp", ""),
        account_equity_before=float(er.get("account_equity_before", 0.0)),
        positions_before=er.get("positions_before", {}),
        target_weights=er.get("target_weights", {}),
        proposed_orders=[],   # not rendered in daily report
        submitted_orders=[
            _SubmittedOrderView(**o) for o in er.get("submitted_orders", [])
        ],
        dry_run=bool(er.get("dry_run", False)),
        notes=er.get("notes", ""),
    )

    subject, body = render_daily_report(
        run_date=for_date,
        strategy_name=payload.get("strategy_name", "(unknown)"),
        target_weights=payload.get("target_weights", {}),
        execution_report=report_view,
    )

    sender = email_sender or EmailSender()
    sender.send(subject=subject, body_text=body, body_html=_markdown_to_html(body))
    logger.info("daily report emailed: %s", subject)
    return subject


# ---------------------------------------------------------------------------
# View shims — let the renderer treat the deserialized dict like an
# ExecutionReport without re-running the dataclass machinery.
# ---------------------------------------------------------------------------


class _SubmittedOrderView:
    """Minimal duck-type matching SubmittedOrder for the renderer's needs."""

    def __init__(self, **kw: Any) -> None:
        self.symbol = kw.get("symbol", "")
        self.side = kw.get("side", "buy")
        self.qty = int(kw.get("qty", 0))
        self.status = kw.get("status", "")
        self.alpaca_order_id = kw.get("alpaca_order_id")
        self.role = kw.get("role", "entry")
        self.stop_price = kw.get("stop_price")
        self.error = kw.get("error")


class _ExecutionReportView:
    """Minimal duck-type matching ExecutionReport for the renderer."""

    def __init__(
        self, *, env, timestamp, account_equity_before, positions_before,
        target_weights, proposed_orders, submitted_orders, dry_run, notes,
    ) -> None:
        self.env = env
        self.timestamp = timestamp
        self.account_equity_before = account_equity_before
        self.positions_before = positions_before
        self.target_weights = target_weights
        self.proposed_orders = proposed_orders
        self.submitted_orders = submitted_orders
        self.dry_run = dry_run
        self.notes = notes


# ---------------------------------------------------------------------------
# Markdown -> HTML (trivial — wrap in <pre> for now; v2 can use markdown lib)
# ---------------------------------------------------------------------------


def _markdown_to_html(md: str) -> str:
    """Wrap markdown in <pre> for HTML clients that don't render md natively.

    For v2 we'd use the ``markdown`` package; for now, monospaced preformatted
    text is readable in every mail client without adding a dependency.
    """
    # Minimal HTML-escape to avoid raw markdown breaking the HTML.
    escaped = (
        md.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        '<html><body style="font-family: -apple-system, Segoe UI, sans-serif">'
        f"<pre style='font-size: 13px; line-height: 1.4'>{escaped}</pre>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# CLI entry points (registered as console_scripts in pyproject.toml)
# ---------------------------------------------------------------------------


def _email_failure(routine: str, exc: Exception) -> None:
    """Best-effort failure notification. Never raises."""
    try:
        sender = EmailSender()
        sender.send(
            subject=f"quant agent FAILURE — {routine}",
            body_text=(
                f"The {routine} routine raised an exception.\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                f"Traceback:\n{traceback.format_exc()}\n\n"
                f"Check the agent's logs for context. The launchd job exit "
                f"code will also be non-zero."
            ),
        )
    except Exception as send_err:
        # Never let an email-on-error failure mask the real error.
        # launchd will surface the real one via the non-zero exit.
        print(
            f"[agent] failed to send failure email: {send_err}",
            file=sys.stderr,
        )


def cli_run_trade() -> None:
    """Console-script entry point: ``uv run quant-daily-trade``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Run the agent's daily trade.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="don't submit orders to Alpaca; report what would happen",
    )
    args = parser.parse_args()
    try:
        path = run_daily_trade(dry_run=args.dry_run)
        if path is None:
            # Idempotent skip OR past-deadline skip. Both are clean exits
            # so launchd's KeepAlive doesn't keep re-firing.
            print("[agent] daily trade skipped (idempotent or past deadline)")
        else:
            print(f"[agent] daily trade complete; log: {path}")
    except Exception as e:
        _email_failure("daily trade", e)
        raise


def cli_run_report() -> None:
    """Console-script entry point: ``uv run quant-daily-report``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Send the agent's daily report.")
    parser.add_argument(
        "--for-date", default=None,
        help="ISO date YYYY-MM-DD; defaults to today",
    )
    args = parser.parse_args()
    for_date = date.fromisoformat(args.for_date) if args.for_date else None
    try:
        subject = run_daily_report(for_date=for_date)
        print(f"[agent] daily report sent: {subject}")
    except Exception as e:
        _email_failure("daily report", e)
        raise
