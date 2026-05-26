"""Tests for the bars integrity checker.

Pattern: each test starts from ``clean_bars()`` — a known-good DataFrame —
then mutates exactly the field under test. That way every failing test
points at exactly one bug, and adding a new check only requires adding
one new test (not editing the baseline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.data.alpaca_client import BAR_COLUMNS
from quant.data.integrity import BarsIntegrityError, check_daily_bars


def _clean_bars() -> pd.DataFrame:
    """A known-good 2-symbol, 5-business-day bars frame.

    Constructed manually so we can be sure it satisfies every check —
    no dependence on Alpaca's behavior or our own cache.
    """
    bdays = pd.bdate_range("2024-01-02", "2024-01-08", tz="UTC")
    rows = []
    idx = []
    for sym in ("AAPL", "MSFT"):
        for i, ts in enumerate(bdays):
            rows.append(
                {
                    "open": 100.0 + i,
                    "high": 102.0 + i,
                    "low": 99.0 + i,
                    "close": 101.0 + i,
                    "volume": 1_000_000 + 10_000 * i,
                }
            )
            idx.append((sym, ts))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]),
        columns=list(BAR_COLUMNS),
    )


# ----------------------------------------------------------------------------
# Happy paths
# ----------------------------------------------------------------------------


def test_clean_bars_pass() -> None:
    """The known-good frame must pass with no issues raised."""
    check_daily_bars(_clean_bars())  # should not raise


def test_empty_well_shaped_frame_passes() -> None:
    """An empty frame with the right index/columns is a valid 'no data' response."""
    empty = pd.DataFrame(
        columns=list(BAR_COLUMNS),
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )
    check_daily_bars(empty)  # should not raise


# ----------------------------------------------------------------------------
# Structural failures
# ----------------------------------------------------------------------------


def test_wrong_columns_caught() -> None:
    """Renaming a column means downstream code reading by name will get the wrong field."""
    bad = _clean_bars().rename(columns={"close": "settle"})
    with pytest.raises(BarsIntegrityError, match="columns are"):
        check_daily_bars(bad)


def test_missing_index_names_caught() -> None:
    """The MultiIndex must be named so downstream code can ask for level by name."""
    bad = _clean_bars()
    bad.index.names = [None, None]
    with pytest.raises(BarsIntegrityError, match=r"index names"):
        check_daily_bars(bad)


# ----------------------------------------------------------------------------
# Row-level failures
# ----------------------------------------------------------------------------


def test_nulls_caught() -> None:
    bad = _clean_bars()
    bad.iloc[0, bad.columns.get_loc("close")] = np.nan
    with pytest.raises(BarsIntegrityError, match="null values"):
        check_daily_bars(bad)


def test_duplicate_index_caught() -> None:
    """Two rows with the same (symbol, timestamp) would double-count a day."""
    clean = _clean_bars()
    # Duplicate the first row.
    bad = pd.concat([clean, clean.iloc[[0]]])
    with pytest.raises(BarsIntegrityError, match="duplicate"):
        check_daily_bars(bad)


def test_zero_price_caught() -> None:
    """Zero close would produce a -100% return on the prior day's diff."""
    bad = _clean_bars()
    bad.iloc[0, bad.columns.get_loc("close")] = 0.0
    with pytest.raises(BarsIntegrityError, match="close <= 0"):
        check_daily_bars(bad)


def test_negative_price_caught() -> None:
    """Crude oil went negative once; equities never should."""
    bad = _clean_bars()
    bad.iloc[0, bad.columns.get_loc("low")] = -1.0
    with pytest.raises(BarsIntegrityError, match="low <= 0"):
        check_daily_bars(bad)


def test_negative_volume_caught() -> None:
    bad = _clean_bars()
    bad.iloc[0, bad.columns.get_loc("volume")] = -100
    with pytest.raises(BarsIntegrityError, match="negative volume"):
        check_daily_bars(bad)


def test_high_below_low_caught() -> None:
    """The classic OHLC corruption — usually a feed bug."""
    bad = _clean_bars()
    bad.iloc[0, bad.columns.get_loc("high")] = 50.0   # below the low of 99
    with pytest.raises(BarsIntegrityError, match="high < low"):
        check_daily_bars(bad)


def test_close_outside_high_low_caught() -> None:
    """close must lie within [low, high]."""
    bad = _clean_bars()
    bad.iloc[0, bad.columns.get_loc("close")] = 200.0  # above the high
    with pytest.raises(BarsIntegrityError, match="high < close"):
        check_daily_bars(bad)


def test_naive_timestamps_caught() -> None:
    """Tz-naive timestamps silently mis-compare against tz-aware ranges."""
    bad = _clean_bars()
    bad.index = pd.MultiIndex.from_tuples(
        [(s, ts.tz_localize(None)) for s, ts in bad.index],
        names=["symbol", "timestamp"],
    )
    with pytest.raises(BarsIntegrityError, match="timezone-naive"):
        check_daily_bars(bad)


def test_weekend_timestamp_caught() -> None:
    """No Saturday/Sunday bars for US equities."""
    clean = _clean_bars()
    saturday = pd.Timestamp("2024-01-06", tz="UTC")  # a Saturday
    saturday_row = clean.iloc[[0]].copy()
    saturday_row.index = pd.MultiIndex.from_tuples(
        [("AAPL", saturday)], names=["symbol", "timestamp"]
    )
    bad = pd.concat([clean, saturday_row])
    with pytest.raises(BarsIntegrityError, match="weekend"):
        check_daily_bars(bad)


def test_future_timestamp_caught() -> None:
    """A timestamp in the future means something is wrong with the source or our clock."""
    clean = _clean_bars()
    future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=365)
    future_row = clean.iloc[[0]].copy()
    future_row.index = pd.MultiIndex.from_tuples(
        [("AAPL", future)], names=["symbol", "timestamp"]
    )
    bad = pd.concat([clean, future_row])
    with pytest.raises(BarsIntegrityError, match="future timestamps"):
        check_daily_bars(bad)


# ----------------------------------------------------------------------------
# All-issues-reported-together check
# ----------------------------------------------------------------------------


def test_multiple_issues_reported_together() -> None:
    """One bad frame with several issues should surface them all at once.

    This is the design promise — chasing issues one-at-a-time across
    debug-cycle reruns is the worst kind of data-debugging slog.
    """
    bad = _clean_bars()
    bad.iloc[0, bad.columns.get_loc("close")] = np.nan      # null
    bad.iloc[1, bad.columns.get_loc("volume")] = -1          # neg volume
    bad.iloc[2, bad.columns.get_loc("high")] = 0             # zero price + OHLC violation

    with pytest.raises(BarsIntegrityError) as exc_info:
        check_daily_bars(bad)

    # Multiple distinct issues should be in the exception's structured list.
    assert len(exc_info.value.issues) >= 3
