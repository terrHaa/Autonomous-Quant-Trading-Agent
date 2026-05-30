"""Tiny retry helper for transient network errors.

The agent talks to two services that occasionally fail mid-handshake
when routed through a VPN/proxy: Alpaca's market-data API (TLS handshake
flakes, intermittent DNS) and Gmail's SMTP (RST mid-session). Neither
failure is persistent — a fresh connection a second later almost always
succeeds. Without retry, every flake costs an entire day's trade or
report.

Design choices
--------------
- Caller passes the exception classes that are "transient". The helper
  doesn't try to enumerate the network error universe itself, because
  the right list depends on the call site (e.g. SMTPAuthenticationError
  should NOT be retried even though it's an SMTP error).
- Exponential backoff with a small base: 1s, 3s, 9s. Three attempts
  total. This covers ~99% of the transient failures we've seen, while
  keeping the worst-case latency under ~15 seconds — which matters for
  the morning trade routine where market open won't wait.
- No external dependency. ``tenacity`` would be overkill for one helper.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default backoff schedule in seconds. The sum (1 + 3 + 9 = 13s of waits
# plus call time) is our worst-case added latency.
_DEFAULT_BACKOFFS: tuple[float, ...] = (1.0, 3.0, 9.0)


def retry_on_transient(
    fn: Callable[[], T],
    *,
    transient: tuple[type[BaseException], ...],
    description: str,
    backoffs: tuple[float, ...] = _DEFAULT_BACKOFFS,
) -> T:
    """Call ``fn()`` and retry on ``transient`` exceptions.

    Parameters
    ----------
    fn
        Zero-argument callable. Use a lambda/closure to bind real args.
    transient
        Exception classes that should trigger a retry. ANY exception
        whose class is NOT in this tuple is re-raised immediately —
        permanent errors (auth failures, 4xx responses) must not be
        retried, or we waste time and may exceed rate limits.
    description
        Short human-readable label for log messages (e.g. "Alpaca bars
        fetch" or "SMTP send"). Helps when the agent log shows multiple
        retry events from different call sites.
    backoffs
        Seconds to wait BEFORE each retry. ``len(backoffs)`` retries
        are attempted, so total attempts = len(backoffs) + 1.

    Returns
    -------
    The return value of ``fn()`` on the first successful attempt.

    Raises
    ------
    The last ``transient`` exception if every attempt fails. Non-transient
    exceptions are raised on the attempt where they occur.
    """
    last_exc: BaseException | None = None
    total_attempts = len(backoffs) + 1

    for attempt_idx in range(total_attempts):
        try:
            return fn()
        except transient as e:
            last_exc = e
            # If this was the last attempt, fall through and re-raise.
            if attempt_idx == total_attempts - 1:
                break
            wait = backoffs[attempt_idx]
            # Per-retry messages are logged at INFO (stdout via launchd's
            # .out file), NOT WARNING (which would land in .err and trip
            # the audit's error_logs check on every flaky-but-successful
            # run). The genuine WARNING comes below if ALL retries fail.
            logger.info(
                "%s: transient error on attempt %d/%d (%s: %s); "
                "retrying in %.1fs",
                description, attempt_idx + 1, total_attempts,
                type(e).__name__, e, wait,
            )
            time.sleep(wait)

    # All attempts exhausted — this is the real failure. WARN so the .err
    # log captures it for the audit to flag.
    assert last_exc is not None  # for type checker
    logger.warning(
        "%s: all %d retries exhausted; re-raising %s",
        description, total_attempts, type(last_exc).__name__,
    )
    raise last_exc
