"""Integration tests for the Alpaca data client.

These hit the real Alpaca API using whatever keys are in `.env`. They:
  - confirm our keys still work (so a key rotation breaks tests, not strategies)
  - confirm Alpaca hasn't changed the response shape under us
  - confirm our normalization (column selection, MultiIndex) is correct

Marked `integration` so you can skip them in offline / no-network runs:
  uv run pytest -m "not integration"

The fixture below auto-skips if no credentials are configured, so this file
never causes a test failure on a clean checkout — it just doesn't run.
"""

from __future__ import annotations

from datetime import date

import pytest

from quant.data.alpaca_client import (
    BAR_COLUMNS,
    AlpacaCredentials,
    AlpacaDataClient,
)
from quant.data.integrity import check_daily_bars


@pytest.fixture
def client() -> AlpacaDataClient:
    """Build a real client, or skip if .env isn't set up."""
    try:
        creds = AlpacaCredentials.from_env(env="paper")
    except RuntimeError as e:
        pytest.skip(f"No Alpaca credentials available: {e}")
    return AlpacaDataClient(creds)


@pytest.mark.integration
def test_fetch_two_symbols_returns_promised_shape(client: AlpacaDataClient) -> None:
    """A two-symbol, ~7-trading-day fetch should return a sensible frame."""
    # Picked a week that's fully in the past (so we're not racing the 15-min
    # free-tier delay) and contains only standard trading days.
    df = client.get_daily_bars(
        ["AAPL", "MSFT"],
        start=date(2024, 1, 2),   # Tue (Mon was New Year's Day, market closed)
        end=date(2024, 1, 10),    # Wed — covers 7 trading days for both symbols
    )

    # Roughly 7 trading days × 2 symbols = 14 rows. Use a loose lower bound
    # so a single missing bar (e.g. a feed gap) doesn't fail the test.
    assert len(df) >= 12, f"Expected ~14 rows, got {len(df)}"

    # The MultiIndex contract.
    assert df.index.names == ["symbol", "timestamp"]
    symbols = set(df.index.get_level_values("symbol").unique())
    assert symbols == {"AAPL", "MSFT"}

    # The column contract.
    assert tuple(df.columns) == BAR_COLUMNS


@pytest.mark.integration
def test_returned_bars_satisfy_ohlc_invariants(client: AlpacaDataClient) -> None:
    """OHLC bars must be internally consistent.

    This is the most common form of "junk data" you'll see from feeds:
    bars where high < low, or close outside [low, high]. Catch it once
    here so no strategy ever has to handle it.
    """
    df = client.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 31))
    assert not df.empty

    # High must be the period high; low must be the period low.
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()

    # No negative volume.
    assert (df["volume"] >= 0).all()


@pytest.mark.integration
def test_real_alpaca_bars_pass_integrity_checks(client: AlpacaDataClient) -> None:
    """A real Alpaca fetch must satisfy our full integrity contract.

    This is the cross-product check: if Alpaca starts returning subtly
    broken data (zero volume on illiquid IEX names, weekend prints from
    a feed glitch, etc.), we fail in CI here rather than discovering it
    weeks later via a weird backtest result.
    """
    df = client.get_daily_bars(["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 1, 31))
    check_daily_bars(df)  # raises BarsIntegrityError if anything is off


@pytest.mark.integration
def test_weekend_only_range_returns_empty_frame(client: AlpacaDataClient) -> None:
    """A date range with no trading days returns an empty frame, not an error.

    This is the real-world "no data in window" case — comes up when a
    universe contains a ticker whose listing dates don't overlap the
    backtest window (recent IPOs, delistings, etc.). Asking for a Saturday
    + Sunday is a clean, deterministic way to provoke the same code path.

    NOTE: we deliberately do NOT test malformed symbols (e.g. with
    underscores or numbers). Alpaca rejects those with HTTP 400, which is
    correct — a malformed ticker is a programmer bug, not a "no data"
    situation, and silently swallowing it would mask real errors upstream.
    """
    df = client.get_daily_bars(
        ["AAPL"],
        start=date(2024, 1, 6),   # Saturday
        end=date(2024, 1, 7),     # Sunday
    )

    assert df.empty
    # Even for an empty response, the column contract must hold so
    # downstream code doesn't have to special-case the shape.
    assert tuple(df.columns) == BAR_COLUMNS
