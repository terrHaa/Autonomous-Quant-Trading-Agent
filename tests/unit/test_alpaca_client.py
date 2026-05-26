"""Unit tests for the Alpaca data client.

These tests do NOT hit the network. They verify the bits we own — credentials
handling and the shape of our outputs — without depending on Alpaca being up
or on our keys being correct. Real-API verification lives in
`tests/integration/test_alpaca_data.py`.
"""

from __future__ import annotations

import pytest

from quant.data.alpaca_client import (
    BAR_COLUMNS,
    AlpacaCredentials,
    _empty_bars_frame,
)


def test_credentials_from_env_reads_paper_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env(env='paper') should pick up ALPACA_PAPER_API_KEY/SECRET."""
    # `monkeypatch.setenv` only affects this test — gets torn down automatically.
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "fake_paper_key")
    monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "fake_paper_secret")

    creds = AlpacaCredentials.from_env(env="paper")
    assert creds.api_key == "fake_paper_key"
    assert creds.api_secret == "fake_paper_secret"


def test_credentials_from_env_reads_live_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env(env='live') should pick up ALPACA_LIVE_API_KEY/SECRET.

    Independent test so we can't accidentally read paper keys when asked for
    live ones — that mix-up has burned more than one trading firm.
    """
    monkeypatch.setenv("ALPACA_LIVE_API_KEY", "fake_live_key")
    monkeypatch.setenv("ALPACA_LIVE_API_SECRET", "fake_live_secret")

    creds = AlpacaCredentials.from_env(env="live")
    assert creds.api_key == "fake_live_key"
    assert creds.api_secret == "fake_live_secret"


def test_credentials_from_env_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key should raise a clear error, not return None silently.

    If we returned None, the failure would happen far downstream when
    Alpaca rejects the empty auth header — and the error message wouldn't
    tell you to check your .env.
    """
    # Strip both possible sources of paper keys so the test is independent
    # of the developer's `.env`.
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="Missing ALPACA_PAPER_API_KEY"):
        # `load_dotenv()` inside `from_env` would re-populate from .env;
        # patch it out so the test really is checking the missing case.
        monkeypatch.setattr("quant.data.alpaca_client.load_dotenv", lambda: None)
        AlpacaCredentials.from_env(env="paper")


def test_credentials_are_frozen() -> None:
    """Credentials shouldn't be mutable after construction.

    A live key swap mid-run is a great way to send orders to the wrong
    account — `frozen=True` on the dataclass makes that an error.
    """
    creds = AlpacaCredentials(api_key="k", api_secret="s")
    # `dataclasses.FrozenInstanceError` is the precise type; it inherits
    # from AttributeError. Be specific so the test doesn't pass on an
    # unrelated exception.
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        creds.api_key = "different"  # type: ignore[misc]


def test_empty_bars_frame_has_correct_shape() -> None:
    """The empty frame should match our promised contract.

    Downstream code branches on `df.empty`, not on whether the columns are
    present — so an empty frame with the wrong columns would be a sneaky bug.
    """
    df = _empty_bars_frame()

    assert df.empty
    assert tuple(df.columns) == BAR_COLUMNS
    assert df.index.names == ["symbol", "timestamp"]
