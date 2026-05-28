from __future__ import annotations

import os
import time
from typing import Callable, TypeVar

T = TypeVar("T")

# Seconds between successive online requests on free tiers (~1 req/s limit).
_DEFAULT_MIN_INTERVAL = 1.5

# Substrings that identify an HTTP 429 / rate-limit error across the Mistral and
# OpenAI-compatible SDKs (their exception types differ, but the text does not).
_RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limited", "too many requests")


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception looks like a provider rate-limit error."""
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 2,
    base_delay: float = 3.0,
) -> T:
    """Call ``fn``, retrying only on rate-limit errors with exponential backoff.

    Free LLM tiers (used here because they are reachable from Cuba) impose tight
    rate limits, so a batch of sequential scoring calls regularly hits HTTP 429.
    Any other exception is re-raised immediately so genuine failures still
    surface (and can trigger the local fallback).

    Args:
        fn: Zero-argument callable performing the API request.
        max_retries: Extra attempts after the first (delays: 3s, 6s by default).
        base_delay: Base backoff in seconds; attempt n waits base_delay * 2**n.

    Returns:
        Whatever ``fn`` returns on success.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries or not is_rate_limit_error(exc):
                raise
            time.sleep(base_delay * (2**attempt))
    raise RuntimeError("unreachable")  # pragma: no cover


class MinIntervalThrottle:
    """Enforce a minimum wall-clock interval between successive calls.

    Free LLM tiers cap sustained throughput (roughly one request per second),
    so bursting a document's segments triggers 429s. Holding one throttle per
    provider instance spaces every real API call far enough apart to stay under
    the limit and get genuine scores instead of degraded fallbacks. An interval
    of 0 disables it (useful for paid tiers and tests).
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.monotonic()


def throttle_from_env() -> MinIntervalThrottle:
    """Build a throttle from LLM_MIN_REQUEST_INTERVAL_SECONDS (default 1.5s)."""
    return MinIntervalThrottle(
        float(
            os.environ.get(
                "LLM_MIN_REQUEST_INTERVAL_SECONDS", str(_DEFAULT_MIN_INTERVAL)
            )
        )
    )
