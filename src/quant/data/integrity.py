"""integrity.py — assertion helpers for the standard daily-bars DataFrame.

What this is for
----------------
Junk data is the #1 way to quietly ruin a backtest. A duplicate date
double-counts a day's return; a stray negative volume blows up a turnover
metric; a "high < close" row makes a stop-loss strategy look impossible.
None of these failures are LOUD — the backtest still finishes, just with
wrong numbers. So we run a battery of cheap structural checks at every
boundary where bad data would cause downstream damage:

- before persisting to cache (TODO — not wired in yet, see "Trade-off" below)
- before feeding bars into the backtest engine
- in CI on a sample fetch, so an Alpaca-side regression breaks tests
  before it breaks any actual research

Design
------
One public function: ``check_daily_bars(df)``. It runs every check, gathers
ALL issues into a single ``BarsIntegrityError`` (not just the first), and
raises with a multi-line message that names every problem. The "report
everything in one go" choice matters: when you're debugging dirty data,
fixing one issue only to discover three more is much more painful than
seeing all four at once.

Trade-off: we deliberately do NOT call ``check_daily_bars`` from inside
the cache. That keeps the cache pure (writes whatever it's handed) and
the checker pure (looks at frames, no opinion on storage). The cost is
that bad data CAN be cached — if you ever suspect it, run the check on
the cached frame and use ``cache.invalidate_symbol`` to refetch.
"""

from __future__ import annotations

import pandas as pd

from quant.data.alpaca_client import BAR_COLUMNS


class BarsIntegrityError(ValueError):
    """Raised when a bars DataFrame violates the integrity contract.

    The full list of issues is on ``self.issues`` — useful when the caller
    wants to log structured details rather than just the formatted message.
    """

    def __init__(self, issues: list[str]) -> None:
        self.issues = list(issues)
        message = (
            f"Bars failed integrity check ({len(self.issues)} issue"
            f"{'s' if len(self.issues) != 1 else ''}):\n"
            + "\n".join(f"  - {i}" for i in self.issues)
        )
        super().__init__(message)


def check_daily_bars(df: pd.DataFrame) -> None:
    """Validate a daily-bars DataFrame against the standard contract.

    The contract:
      - Columns: open, high, low, close, volume (exactly, in that order).
      - Index:   MultiIndex with names ``("symbol", "timestamp")``.
      - No nulls anywhere in the data.
      - No duplicate ``(symbol, timestamp)`` rows.
      - Prices strictly positive (zero or negative means broken/halted data).
      - Volumes non-negative.
      - OHLC invariants per row:
          high >= low,  high >= open,  high >= close,
          low  <= open, low  <= close.
      - Timestamps timezone-aware (UTC).
      - No Saturday/Sunday timestamps (no weekend trading in US equities).
      - No timestamps in the future (sanity vs system clock).

    An EMPTY frame with the right shape passes — emptiness is a valid
    "no data" response from the cache, not corruption.

    Raises
    ------
    BarsIntegrityError
        If ANY check fails. The exception's message lists every issue
        found; ``error.issues`` is the underlying list of strings.
    """
    issues: list[str] = []

    # ---- Structural checks (columns + index shape) ----
    # We check these first because every subsequent check assumes them.
    if tuple(df.columns) != BAR_COLUMNS:
        issues.append(
            f"columns are {tuple(df.columns)}, expected {BAR_COLUMNS}"
        )
    if list(df.index.names) != ["symbol", "timestamp"]:
        issues.append(
            f"index names are {list(df.index.names)}, expected "
            f"['symbol', 'timestamp']"
        )

    # An empty frame is a valid response — once we've confirmed the shape,
    # there's nothing more to check on the rows.
    if df.empty:
        if issues:
            raise BarsIntegrityError(issues)
        return

    # ---- Null check ----
    # `any(axis=None)` returns a single bool over the whole frame. We then
    # ask which columns have nulls so the error message can name them.
    if df.isna().any(axis=None):
        cols_with_nulls = df.columns[df.isna().any()].tolist()
        issues.append(f"null values in columns: {cols_with_nulls}")

    # ---- Duplicate (symbol, timestamp) rows ----
    dup_mask = df.index.duplicated()
    if dup_mask.any():
        # Show the first 5 dupes so the error stays readable even on
        # pathological inputs with thousands of duplicates.
        dupes = list(df.index[dup_mask][:5])
        issues.append(
            f"{int(dup_mask.sum())} duplicate (symbol, timestamp) rows; "
            f"first: {dupes}"
        )

    # ---- Price positivity ----
    # Strict > 0. Zero prices in OHLCV mean "no data" or "halted" and
    # would silently produce -100% or +inf returns in math downstream.
    for col in ("open", "high", "low", "close"):
        # Skip column if it's missing (already reported in structural check).
        if col not in df.columns:
            continue
        bad = df[df[col] <= 0]
        if len(bad):
            issues.append(
                f"{col} <= 0 in {len(bad)} row(s); first: {bad.index[0]}"
            )

    # ---- Volume non-negativity ----
    if "volume" in df.columns:
        bad_vol = df[df["volume"] < 0]
        if len(bad_vol):
            issues.append(
                f"negative volume in {len(bad_vol)} row(s); first: {bad_vol.index[0]}"
            )

    # ---- OHLC invariants ----
    # Done as five separate checks because each violation usually has a
    # different upstream cause (bad split adjustment, exchange print, etc.)
    # and seeing them separately speeds debugging.
    ohlc_present = all(c in df.columns for c in ("open", "high", "low", "close"))
    if ohlc_present:
        for desc, mask in (
            ("high < low",   df["high"] < df["low"]),
            ("high < open",  df["high"] < df["open"]),
            ("high < close", df["high"] < df["close"]),
            ("low > open",   df["low"]  > df["open"]),
            ("low > close",  df["low"]  > df["close"]),
        ):
            if mask.any():
                first_bad = df.index[mask][0]
                issues.append(
                    f"OHLC invariant violated ({desc}) in "
                    f"{int(mask.sum())} row(s); first: {first_bad}"
                )

    # ---- Timestamp checks (timezone, weekends, future) ----
    # Only meaningful if the index has the expected names — otherwise the
    # level may not be timestamps at all.
    if "timestamp" in (df.index.names or []):
        ts = df.index.get_level_values("timestamp")

        # Timezone-aware. A naive datetime would silently compare wrong
        # against a tz-aware "now" or against tz-aware ranges from the SDK.
        if ts.tz is None:
            issues.append("timestamps are timezone-naive; expected tz-aware (UTC)")

        # No weekends. Alpaca daily bars are Monday-Friday only; a weekend
        # row means something weird is going on with how data was stitched.
        weekend_mask = ts.weekday >= 5  # 0=Mon..4=Fri, 5=Sat, 6=Sun
        if weekend_mask.any():
            issues.append(
                f"weekend timestamps in {int(weekend_mask.sum())} row(s); "
                f"first: {ts[weekend_mask][0]}"
            )

        # No future timestamps. If we have data dated tomorrow, either the
        # system clock is wrong or we're reading from a corrupted source.
        # Comparison only valid if ts is tz-aware; otherwise skip to avoid
        # piling on a redundant error.
        if ts.tz is not None:
            now = pd.Timestamp.now(tz="UTC")
            future_mask = ts > now
            if future_mask.any():
                issues.append(
                    f"future timestamps in {int(future_mask.sum())} row(s); "
                    f"first: {ts[future_mask][0]}"
                )

    if issues:
        raise BarsIntegrityError(issues)
