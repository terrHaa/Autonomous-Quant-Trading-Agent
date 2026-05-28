"""Tests for the retry helper: backoff, exhaustion, non-transient errors."""

from __future__ import annotations

import pytest

from quant.util.retry import retry_on_transient


class _Transient(Exception):
    """A made-up exception class for testing the 'should retry' path."""


class _Permanent(Exception):
    """A made-up exception class for testing the 'should NOT retry' path."""


def test_returns_immediately_on_first_success() -> None:
    """No retries; the value is returned and fn is called exactly once."""
    calls = [0]

    def fn() -> str:
        calls[0] += 1
        return "ok"

    result = retry_on_transient(
        fn,
        transient=(_Transient,),
        description="test",
        backoffs=(),   # zero retries — still fine if fn succeeds first try
    )
    assert result == "ok"
    assert calls[0] == 1


def test_retries_on_transient_then_succeeds(monkeypatch) -> None:
    """Fail twice, succeed third — total 3 attempts, return last value."""
    # Stub sleep so the test runs instantly.
    monkeypatch.setattr("quant.util.retry.time.sleep", lambda _: None)

    calls = [0]

    def fn() -> str:
        calls[0] += 1
        if calls[0] < 3:
            raise _Transient(f"flake #{calls[0]}")
        return "finally"

    result = retry_on_transient(
        fn,
        transient=(_Transient,),
        description="test",
        backoffs=(0.0, 0.0),
    )
    assert result == "finally"
    assert calls[0] == 3


def test_raises_after_all_retries_exhausted(monkeypatch) -> None:
    """If every attempt fails with a transient, the LAST exception is raised."""
    monkeypatch.setattr("quant.util.retry.time.sleep", lambda _: None)

    calls = [0]

    def fn() -> None:
        calls[0] += 1
        raise _Transient(f"flake #{calls[0]}")

    with pytest.raises(_Transient, match="flake #3"):
        retry_on_transient(
            fn,
            transient=(_Transient,),
            description="test",
            backoffs=(0.0, 0.0),
        )
    assert calls[0] == 3  # initial + 2 retries


def test_permanent_error_not_retried() -> None:
    """An exception NOT in `transient` is raised immediately, no retries."""
    calls = [0]

    def fn() -> None:
        calls[0] += 1
        raise _Permanent("auth failed")

    with pytest.raises(_Permanent, match="auth failed"):
        retry_on_transient(
            fn,
            transient=(_Transient,),
            description="test",
            backoffs=(1.0, 1.0),   # would sleep ages, but we never reach the sleep
        )
    assert calls[0] == 1   # only one attempt


def test_backoff_durations_used_in_order(monkeypatch) -> None:
    """The sleeps between retries follow the `backoffs` tuple."""
    sleeps: list[float] = []
    monkeypatch.setattr("quant.util.retry.time.sleep", lambda s: sleeps.append(s))

    calls = [0]

    def fn() -> None:
        calls[0] += 1
        raise _Transient("flake")

    with pytest.raises(_Transient):
        retry_on_transient(
            fn,
            transient=(_Transient,),
            description="test",
            backoffs=(0.5, 1.5, 4.5),
        )
    # 4 attempts, 3 retries → 3 sleeps in order.
    assert sleeps == [0.5, 1.5, 4.5]
    assert calls[0] == 4


def test_default_backoffs_apply_three_retries(monkeypatch) -> None:
    """When backoffs not specified, defaults give 4 attempts total (1+3+9s)."""
    monkeypatch.setattr("quant.util.retry.time.sleep", lambda _: None)
    calls = [0]

    def fn() -> str:
        calls[0] += 1
        raise _Transient("flake")

    with pytest.raises(_Transient):
        retry_on_transient(
            fn, transient=(_Transient,), description="test",
        )
    assert calls[0] == 4   # initial + 3 defaults
