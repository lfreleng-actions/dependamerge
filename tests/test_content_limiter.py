# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Pacing content-generating requests under GitHub's limits.

GitHub allows 80 content-generating requests a minute and 500 an hour.
A bulk run that approved and merged 264 pull requests in 12 minutes
crossed the hourly figure and had an approval rejected.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from dependamerge.github_async import GitHubAsync, SecondaryRateLimitError
from dependamerge.github_async._content_limiter import (
    CONTENT_PER_HOUR_ENV,
    CONTENT_PER_MINUTE_ENV,
    ContentLimiter,
    is_content_request,
)
from dependamerge.github_async._responses import _handle_secondary_rate_limit
from tests.conftest import make_merge_manager


class FakeClock:
    """A clock that only moves when the limiter sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _limiter(clock: FakeClock, **kwargs) -> ContentLimiter:
    return ContentLimiter(clock=clock, sleep=clock.sleep, **kwargs)


async def _admit(limiter: ContentLimiter, clock: FakeClock, n: int) -> list[float]:
    times = []
    for _ in range(n):
        await limiter.acquire()
        times.append(clock.now)
    return times


def _max_in_window(times: list[float], window: float) -> int:
    return max(sum(1 for t in times if start <= t < start + window) for start in times)


class TestBudgets:
    @pytest.mark.asyncio
    async def test_per_minute_budget_is_respected(self) -> None:
        clock = FakeClock()
        limiter = _limiter(clock, per_minute=3, per_hour=100)

        times = await _admit(limiter, clock, 10)

        assert _max_in_window(times, 60.0) == 3
        assert times[3] - times[0] == pytest.approx(60.0)

    @pytest.mark.asyncio
    async def test_hourly_budget_is_respected(self) -> None:
        clock = FakeClock()
        limiter = _limiter(clock, per_minute=100, per_hour=5)
        seen: list[float] = []

        async def sleep(seconds: float) -> None:
            seen.append(limiter.hourly_wait_remaining())
            await clock.sleep(seconds)

        limiter._sleep = sleep
        times = await _admit(limiter, clock, 7)

        assert _max_in_window(times, 3600.0) == 5
        assert times[5] - times[0] == pytest.approx(3600.0)
        # While the hourly budget held requests back, it said so.
        assert seen and seen[0] > 3000
        assert limiter.hourly_wait_remaining() == 0.0

    @pytest.mark.asyncio
    async def test_a_minute_wait_is_not_reported_as_hourly(self) -> None:
        clock = FakeClock()
        limiter = _limiter(clock, per_minute=1, per_hour=100)
        seen: list[float] = []

        async def sleep(seconds: float) -> None:
            seen.append(limiter.hourly_wait_remaining())
            await clock.sleep(seconds)

        limiter._sleep = sleep
        await _admit(limiter, clock, 2)

        assert seen == [0.0]

    def test_defaults_sit_under_githubs_limits(self) -> None:
        limiter = ContentLimiter()
        assert limiter.per_minute < 80
        assert limiter.per_hour < 500


class TestThrottleSignal:
    @pytest.mark.asyncio
    async def test_signal_pauses_and_lowers_the_rate(self) -> None:
        clock = FakeClock()
        limiter = _limiter(clock, per_minute=60, per_hour=1000)

        limiter.throttled()
        await limiter.acquire()

        assert limiter.per_minute == 30
        assert clock.sleeps == [pytest.approx(60.0)]

    @pytest.mark.asyncio
    async def test_signal_honours_retry_after(self) -> None:
        clock = FakeClock()
        limiter = _limiter(clock)

        limiter.throttled(120.0)
        await limiter.acquire()

        assert clock.sleeps == [pytest.approx(120.0)]

    @pytest.mark.asyncio
    async def test_rate_recovers_after_a_quiet_period(self) -> None:
        clock = FakeClock()
        limiter = _limiter(clock, per_minute=60, per_hour=10_000)
        limiter.throttled()
        limiter.throttled()
        assert limiter.per_minute == 15

        for _ in range(20):
            clock.now += ContentLimiter.RECOVERY_SECONDS
            await limiter.acquire()

        assert limiter.per_minute == 60

    @pytest.mark.asyncio
    async def test_no_recovery_while_signals_continue(self) -> None:
        clock = FakeClock()
        limiter = _limiter(clock, per_minute=60, per_hour=10_000)
        limiter.throttled()
        clock.now += ContentLimiter.RECOVERY_SECONDS / 2
        limiter.throttled()
        clock.now += ContentLimiter.RECOVERY_SECONDS / 2
        await limiter.acquire()

        assert limiter.per_minute == 15


class TestWhatCounts:
    GQL = "https://api.github.com/graphql"

    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    def test_rest_writes_count(self, method: str) -> None:
        url = "https://api.github.com/repos/o/r/pulls/1/reviews"
        assert is_content_request(method, url, self.GQL, {"event": "APPROVE"})

    def test_reads_do_not(self) -> None:
        url = "https://api.github.com/repos/o/r/pulls/1"
        assert not is_content_request("GET", url, self.GQL, None)

    def test_graphql_mutations_count_but_queries_do_not(self) -> None:
        mutation = {"query": "\n  mutation { enablePullRequestAutoMerge }"}
        query = {"query": "query { viewer { login } }"}
        assert is_content_request("POST", self.GQL, self.GQL, mutation)
        assert not is_content_request("POST", self.GQL, self.GQL, query)


class TestClientWiring:
    @staticmethod
    def _client() -> tuple[GitHubAsync, AsyncMock]:
        c = GitHubAsync(token="t")
        gate = AsyncMock()
        c.content_limiter.acquire = gate  # type: ignore[method-assign]
        c._client.request = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda method, url, **kw: httpx.Response(
                200, json={"data": {}}, request=httpx.Request(method, url)
            )
        )
        return c, gate

    @pytest.mark.asyncio
    async def test_reads_are_not_gated(self) -> None:
        c, gate = self._client()

        await c.get("/repos/o/r/pulls/1")
        await c.graphql("query { viewer { login } }")

        gate.assert_not_awaited()
        await c.aclose()

    @pytest.mark.asyncio
    async def test_writes_and_mutations_are_gated(self) -> None:
        c, gate = self._client()

        await c.post("/repos/o/r/issues/1/comments", json={"body": "hi"})
        await c.graphql("mutation { enablePullRequestAutoMerge }")

        assert gate.await_count == 2
        await c.aclose()

    def test_budget_is_configurable_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CONTENT_PER_MINUTE_ENV, "20")
        monkeypatch.setenv(CONTENT_PER_HOUR_ENV, "200")

        c = GitHubAsync(token="t")

        assert c.content_limiter.per_minute == 20
        assert c.content_limiter.per_hour == 200

    def test_an_invalid_budget_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CONTENT_PER_HOUR_ENV, "lots")
        with pytest.raises(ValueError, match=CONTENT_PER_HOUR_ENV):
            GitHubAsync(token="t")

    @pytest.mark.asyncio
    async def test_secondary_limit_signals_the_limiter(self, monkeypatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        c = GitHubAsync(token="t")
        c.content_limiter = MagicMock()
        response = httpx.Response(
            403,
            headers={"Retry-After": "30"},
            text="You have exceeded a secondary rate limit",
            request=httpx.Request("POST", "https://api.github.com/x"),
        )

        with pytest.raises(SecondaryRateLimitError):
            await _handle_secondary_rate_limit(c, response)

        c.content_limiter.throttled.assert_called_once_with(30.0)
        await c.aclose()

    @pytest.mark.asyncio
    async def test_throttled_approval_signals_the_limiter(self, monkeypatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        c = GitHubAsync(token="t")
        c.content_limiter = MagicMock()
        request = httpx.Request("POST", "https://api.github.com/repos/o/r/pulls/1")
        response = httpx.Response(
            422,
            json={"message": "Validation Failed", "errors": ["submitted too quickly"]},
            request=request,
        )
        throttled = httpx.HTTPStatusError("422", request=request, response=response)
        c.post = AsyncMock(side_effect=[throttled, {}])  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await c.approve_pull_request("o", "r", 1, "lgtm")

        c.content_limiter.throttled.assert_called_once_with(None)
        await c.aclose()


class TestProgressMessage:
    def test_names_the_content_limit_while_it_holds_the_run(self) -> None:
        clock = FakeClock()
        limiter = _limiter(clock, per_hour=1)
        limiter._hourly_wait_until = clock.now + 90
        mgr, client = make_merge_manager()
        client.content_limiter = limiter

        assert mgr._content_limit_message() == (
            "⏳ Waiting on GitHub's content-creation limit [90s]"
        )

    def test_silent_otherwise(self) -> None:
        mgr, client = make_merge_manager()
        client.content_limiter = ContentLimiter()

        assert mgr._content_limit_message() == ""
