# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Pacing for content-generating requests.

GitHub allows no more than 80 content-generating requests a minute and
500 an hour, far below the general request rate:
https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api#about-secondary-rate-limits

Reviews, comments, merges, branch updates and GraphQL mutations all
count.  A bulk run that approves and merges a few hundred pull requests
crosses the hourly figure in minutes, and GitHub then rejects writes,
often as a ``422`` saying the content was "submitted too quickly".
This limiter keeps those requests inside a budget with headroom for the
user's other activity, and backs off further when GitHub signals a
throttle.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable

#: Sustained budget, under GitHub's 80 a minute.
DEFAULT_CONTENT_PER_MINUTE = 60.0
#: Rolling budget, under GitHub's 500 an hour.
DEFAULT_CONTENT_PER_HOUR = 450

CONTENT_PER_MINUTE_ENV = "DEPENDAMERGE_CONTENT_PER_MINUTE"
CONTENT_PER_HOUR_ENV = "DEPENDAMERGE_CONTENT_PER_HOUR"

_MINUTE = 60.0
_HOUR = 3600.0

_CONTENT_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _env_number(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {raw!r}")
    return value


def content_budget_from_env() -> tuple[float, int]:
    """The ``(per_minute, per_hour)`` budget, honouring overrides.

    Raises:
        ValueError: If an override is not a positive number.
    """
    per_minute = _env_number(CONTENT_PER_MINUTE_ENV, DEFAULT_CONTENT_PER_MINUTE)
    per_hour = _env_number(CONTENT_PER_HOUR_ENV, DEFAULT_CONTENT_PER_HOUR)
    return per_minute, max(1, int(per_hour))


def is_content_request(
    method: str, url: str, graphql_url: str, json_body: object
) -> bool:
    """Whether a request generates content in GitHub's accounting.

    REST writes always do.  GraphQL travels as a ``POST`` either way, so
    only a mutation counts; a query is a read.
    """
    if method.upper() not in _CONTENT_METHODS:
        return False
    if url != graphql_url:
        return True
    query = json_body.get("query") if isinstance(json_body, dict) else None
    return isinstance(query, str) and query.lstrip().lower().startswith("mutation")


class ContentLimiter:
    """A per-minute and rolling hourly budget for content requests.

    Admission is serialised, so waiting requests go through in arrival
    order.  A throttle signal halves the per-minute rate and pauses
    admission; after a quiet period the rate climbs back additively,
    the same shape the client's general throttling follows.
    """

    #: The per-minute rate never drops below this.
    MIN_PER_MINUTE = 5.0
    #: GitHub advises waiting at least a minute after a secondary limit.
    MIN_PAUSE_SECONDS = 60.0
    #: Quiet time required before each recovery step.
    RECOVERY_SECONDS = 120.0

    def __init__(
        self,
        per_minute: float = DEFAULT_CONTENT_PER_MINUTE,
        per_hour: int = DEFAULT_CONTENT_PER_HOUR,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if per_minute <= 0 or per_hour <= 0:
            raise ValueError("content budgets must be greater than zero")
        self.base_per_minute = float(per_minute)
        self.per_minute = float(per_minute)
        self.per_hour = int(per_hour)
        self._clock = clock
        self._sleep = sleep
        self._admitted: deque[float] = deque()
        self._paused_until = 0.0
        self._last_change: float | None = None
        self._hourly_wait_until: float | None = None
        self._lock = asyncio.Lock()

    def _window_wait(self, now: float, cap: int, window: float) -> float:
        """Seconds until one more request fits ``cap`` per ``window``."""
        count = 0
        for stamp in reversed(self._admitted):
            if stamp <= now - window:
                break
            count += 1
        if count < cap:
            return 0.0
        oldest_counted = self._admitted[len(self._admitted) - cap]
        return max(0.0, oldest_counted + window - now)

    def _recover(self, now: float) -> None:
        if self.per_minute >= self.base_per_minute or self._last_change is None:
            return
        if now - self._last_change < self.RECOVERY_SECONDS:
            return
        step = max(1.0, self.base_per_minute / 6)
        self.per_minute = min(self.base_per_minute, self.per_minute + step)
        self._last_change = now

    def _wait_for(self, now: float) -> tuple[float, bool]:
        """The wait before admitting a request, and whether it is hourly."""
        while self._admitted and self._admitted[0] <= now - _HOUR:
            self._admitted.popleft()
        pause = max(0.0, self._paused_until - now)
        minute = self._window_wait(now, max(1, int(self.per_minute)), _MINUTE)
        hour = self._window_wait(now, self.per_hour, _HOUR)
        wait = max(pause, minute, hour)
        return wait, wait > 0 and hour >= wait

    async def acquire(self) -> None:
        """Wait until the budget admits one more content request."""
        async with self._lock:
            while True:
                now = self._clock()
                self._recover(now)
                wait, hourly = self._wait_for(now)
                if wait <= 0:
                    self._hourly_wait_until = None
                    self._admitted.append(now)
                    return
                self._hourly_wait_until = now + wait if hourly else None
                sleep = self._sleep or asyncio.sleep
                await sleep(wait)

    def throttled(self, retry_after: float | None = None) -> None:
        """Back off after GitHub signalled a content-creation throttle.

        Halves the per-minute rate and pauses admission for
        ``retry_after`` when given, otherwise for at least a minute.
        """
        now = self._clock()
        self.per_minute = max(self.MIN_PER_MINUTE, self.per_minute / 2)
        pause = retry_after if retry_after is not None else self.MIN_PAUSE_SECONDS
        self._paused_until = max(self._paused_until, now + max(0.0, pause))
        self._last_change = now

    def hourly_wait_remaining(self) -> float:
        """Seconds left while the hourly budget is what holds requests back."""
        if self._hourly_wait_until is None:
            return 0.0
        return max(0.0, self._hourly_wait_until - self._clock())
