# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Retrying a request GitHub failed to answer (issue #688).

The transport retries a 500 on a read only, because repeating a write
that may have landed is unsafe.  The merge and the approval, which can
check their own effect first, back off 2s, 6s and 18s instead.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
import tenacity

from dependamerge.github_async import GitHubAsync, GitHubServerError
from dependamerge.merge_manager import AsyncMergeManager
from dependamerge.merge_manager._head_pin import approve_at_head
from dependamerge.slot_lease import (
    holding_dispatch_lock,
    holding_slot,
    parked,
    parked_off_dispatch,
)
from tests.conftest import make_merge_manager, make_pr_info


def _status_error(status: int, method: str = "PUT") -> httpx.HTTPStatusError:
    request = httpx.Request(method, "https://api.github.com/repos/o/r/pulls/1/merge")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(
        f"Server error '{status} Oops' for url '{request.url}'",
        request=request,
        response=response,
    )


def _wrapped(status: int) -> Exception:
    """A merge failure as ``_validate_merge_result`` raises it."""
    inner = _status_error(status)
    try:
        raise Exception(f"Failed to merge PR o/r#1. Error: {inner}") from inner
    except Exception as error:
        return error


def _clock_spent_by_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loop clock that only the patched ``asyncio.sleep`` moves on."""
    loop = asyncio.get_running_loop()
    now = [loop.time()]

    async def sleep(delay: float) -> None:
        now[0] += delay

    monkeypatch.setattr(loop, "time", lambda: now[0])
    monkeypatch.setattr(asyncio, "sleep", sleep)


def _slow_read(
    monkeypatch: pytest.MonkeyPatch, seconds: float, result: Any
) -> AsyncMock:
    """A read that spends *seconds* of a loop clock only it moves on."""
    loop = asyncio.get_running_loop()
    now = [loop.time()]
    monkeypatch.setattr(loop, "time", lambda: now[0])
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    async def read(*_args: Any, **_kwargs: Any) -> Any:
        now[0] += seconds
        return result

    return AsyncMock(side_effect=read)


class TestTheTransport:
    @staticmethod
    def _client(statuses: list[int], seen: list[str]) -> GitHubAsync:
        replies = iter(statuses)

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.method)
            return httpx.Response(next(replies), json={})

        api = GitHubAsync(token="t")
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return api

    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        controllers = (
            GitHubAsync._request.retry,  # type: ignore[attr-defined]
            GitHubAsync._request_json.retry,  # type: ignore[attr-defined]
        )
        for controller in controllers:
            monkeypatch.setattr(
                cast(tenacity.AsyncRetrying, controller), "wait", tenacity.wait_none()
            )
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    @pytest.mark.asyncio
    async def test_a_read_outlasts_two_500s(self) -> None:
        seen: list[str] = []
        api = self._client([500, 500, 200], seen)
        try:
            assert await api.get("/repos/o/r/pulls/1") == {}
        finally:
            await api._client.aclose()
        assert seen == ["GET", "GET", "GET"]

    @pytest.mark.asyncio
    async def test_a_write_is_not_repeated_on_a_500(self) -> None:
        seen: list[str] = []
        api = self._client([500, 200], seen)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await api.put("/repos/o/r/pulls/1/merge", json={})
        finally:
            await api._client.aclose()
        assert seen == ["PUT"]


class TestTheMerge:
    @staticmethod
    def _manager(*outcomes: Any, no_wait: bool = False) -> AsyncMergeManager:
        mgr, _client = make_merge_manager(merge_method="merge", max_retries=2)
        mgr._no_wait = no_wait
        mgr._run_deadline = None
        mgr._dispatch_merge = AsyncMock(side_effect=list(outcomes))  # type: ignore[method-assign]
        mgr._recheck_pr_before_retry = AsyncMock(return_value=None)  # type: ignore[method-assign]
        return mgr

    @staticmethod
    def _sleeps(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        return sleep

    @staticmethod
    def _waits(sleep: AsyncMock) -> list[float]:
        return [call.args[0] for call in sleep.await_args_list]

    @pytest.mark.asyncio
    async def test_three_500s_then_a_merge(self, monkeypatch) -> None:
        sleep = self._sleeps(monkeypatch)
        mgr = self._manager(_wrapped(500), _wrapped(500), _wrapped(500), True)

        assert await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        assert self._waits(sleep) == [2.0, 6.0, 18.0]

    @pytest.mark.asyncio
    async def test_four_attempts_and_no_more(self, monkeypatch) -> None:
        self._sleeps(monkeypatch)
        mgr = self._manager(*[_wrapped(502)] * 5)

        assert not await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        assert cast(AsyncMock, mgr._dispatch_merge).await_count == 4

    @pytest.mark.asyncio
    async def test_a_merge_that_landed_is_not_repeated(self, monkeypatch) -> None:
        self._sleeps(monkeypatch)
        mgr = self._manager(_wrapped(500), True)
        mgr._recheck_pr_before_retry = AsyncMock(return_value=True)  # type: ignore[method-assign]

        assert await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        assert cast(AsyncMock, mgr._dispatch_merge).await_count == 1

    @pytest.mark.asyncio
    async def test_a_timeout_counts_as_a_server_error(self, monkeypatch) -> None:
        sleep = self._sleeps(monkeypatch)
        mgr = self._manager(httpx.ReadTimeout("slow"), True)

        assert await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        assert self._waits(sleep) == [2.0]

    @pytest.mark.asyncio
    async def test_max_wait_zero_keeps_the_short_wait(self, monkeypatch) -> None:
        sleep = self._sleeps(monkeypatch)
        mgr = self._manager(*[_wrapped(500)] * 5, no_wait=True)

        assert not await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        assert self._waits(sleep) == [1.0, 1.0]

    @pytest.mark.asyncio
    async def test_max_wait_zero_counts_every_attempt(self, monkeypatch) -> None:
        self._sleeps(monkeypatch)
        mgr = self._manager(Exception("boom"), *[_wrapped(500)] * 4, no_wait=True)

        assert not await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        assert cast(AsyncMock, mgr._dispatch_merge).await_count == 3

    @pytest.mark.asyncio
    async def test_the_wait_lets_go_of_the_dispatch_lock(self, monkeypatch) -> None:
        lock = asyncio.Lock()
        held_during_wait: list[bool] = []

        async def sleep(_delay: float) -> None:
            held_during_wait.append(lock.locked())

        monkeypatch.setattr(asyncio, "sleep", sleep)
        mgr = self._manager(_wrapped(500), True)

        async with holding_dispatch_lock(lock):
            assert await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
            assert lock.locked()
        assert held_during_wait == [False]

    @pytest.mark.asyncio
    async def test_the_run_deadline_shortens_the_wait(self, monkeypatch) -> None:
        sleep = self._sleeps(monkeypatch)
        mgr = self._manager(_wrapped(500), True)
        mgr._run_deadline = asyncio.get_running_loop().time() + 0.5

        assert await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        (wait,) = self._waits(sleep)
        assert 0 < wait <= 0.5

    @pytest.mark.asyncio
    async def test_a_passed_deadline_stops_the_retries(self, monkeypatch) -> None:
        sleep = self._sleeps(monkeypatch)
        mgr = self._manager(_wrapped(500), True)
        mgr._run_deadline = asyncio.get_running_loop().time() - 1

        assert not await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_deadline_spent_waiting_stops_the_retries(
        self, monkeypatch
    ) -> None:
        _clock_spent_by_sleep(monkeypatch)
        mgr = self._manager(_wrapped(500), True)
        mgr._run_deadline = asyncio.get_running_loop().time() + 0.5

        assert not await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        assert cast(AsyncMock, mgr._dispatch_merge).await_count == 1

    @pytest.mark.asyncio
    async def test_a_deadline_spent_reading_stops_the_retries(
        self, monkeypatch
    ) -> None:
        mgr = self._manager(_wrapped(500), True)
        mgr._recheck_pr_before_retry = _slow_read(monkeypatch, 5, None)  # type: ignore[method-assign]
        mgr._run_deadline = asyncio.get_running_loop().time() + 3

        assert not await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        assert cast(AsyncMock, mgr._dispatch_merge).await_count == 1
        assert mgr._merge_server_error("o/r#1", policy_refusal=False)

    @pytest.mark.parametrize("status", [403, 405, 409, 422])
    @pytest.mark.asyncio
    async def test_a_client_error_is_not_backed_off(self, status, monkeypatch) -> None:
        sleep = self._sleeps(monkeypatch)
        mgr = self._manager(*[_wrapped(status)] * 5)
        mgr._decide_after_405 = AsyncMock()  # type: ignore[method-assign]

        await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")

        assert not {2.0, 6.0, 18.0} & set(self._waits(sleep))


class TestTheRequiredWorkflowsMerge:
    """The merge after a pending-workflow 405 backs off as well."""

    @staticmethod
    def _manager(*outcomes: Any) -> AsyncMergeManager:
        mgr, _client = make_merge_manager(merge_method="merge")
        mgr._no_wait = False
        mgr._run_deadline = None
        mgr._wait_for_auto_merge = AsyncMock(return_value=(False, False))  # type: ignore[method-assign]
        mgr._dispatch_after_required_workflows = AsyncMock(  # type: ignore[method-assign]
            side_effect=list(outcomes)
        )
        mgr._recheck_pr_before_retry = AsyncMock(return_value=None)  # type: ignore[method-assign]
        return mgr

    @staticmethod
    async def _retry(mgr: AsyncMergeManager, deadline: float) -> bool:
        return await mgr._retry_until_workflows_finish(
            make_pr_info(), "o", "r", "merge", deadline, None
        )

    @pytest.mark.asyncio
    async def test_three_500s_then_a_merge(self, monkeypatch) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        mgr = self._manager(*[_wrapped(500)] * 3, True)

        deadline = asyncio.get_running_loop().time() + 600
        assert await self._retry(mgr, deadline)
        assert [call.args[0] for call in sleep.await_args_list] == [2.0, 6.0, 18.0]

    @pytest.mark.asyncio
    async def test_four_500s_leave_the_server_error(self, monkeypatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        mgr = self._manager(*[_wrapped(500)] * 5)

        deadline = asyncio.get_running_loop().time() + 600
        assert not await self._retry(mgr, deadline)
        dispatch = cast(AsyncMock, mgr._dispatch_after_required_workflows)
        assert dispatch.await_count == 4
        assert mgr._merge_server_error("o/r#1", policy_refusal=False)

    @pytest.mark.asyncio
    async def test_a_merge_that_landed_is_not_repeated(self, monkeypatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        mgr = self._manager(_wrapped(500), True)
        mgr._recheck_pr_before_retry = AsyncMock(return_value=True)  # type: ignore[method-assign]

        deadline = asyncio.get_running_loop().time() + 600
        assert await self._retry(mgr, deadline)
        dispatch = cast(AsyncMock, mgr._dispatch_after_required_workflows)
        assert dispatch.await_count == 1

    @pytest.mark.asyncio
    async def test_its_deadline_shortens_the_wait(self, monkeypatch) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        mgr = self._manager(_wrapped(500), True)

        deadline = asyncio.get_running_loop().time() + 0.5
        assert await self._retry(mgr, deadline)
        (wait,) = [call.args[0] for call in sleep.await_args_list]
        assert 0 < wait <= 0.5

    @pytest.mark.asyncio
    async def test_a_deadline_spent_reading_stops_the_retries(
        self, monkeypatch
    ) -> None:
        mgr = self._manager(_wrapped(500), True)
        mgr._recheck_pr_before_retry = _slow_read(monkeypatch, 5, None)  # type: ignore[method-assign]

        deadline = asyncio.get_running_loop().time() + 3
        assert not await self._retry(mgr, deadline)
        dispatch = cast(AsyncMock, mgr._dispatch_after_required_workflows)
        assert dispatch.await_count == 1
        assert mgr._merge_server_error("o/r#1", policy_refusal=False)

    @pytest.mark.asyncio
    async def test_a_deadline_spent_under_the_lock_sends_no_retry(
        self, monkeypatch
    ) -> None:
        mgr, client = make_merge_manager(merge_method="merge")
        mgr._no_wait = False
        mgr._run_deadline = None
        mgr._repo_scoped = False
        mgr._wait_for_auto_merge = AsyncMock(return_value=(False, False))  # type: ignore[method-assign]
        mgr._recheck_pr_before_retry = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._fetch_pr_state = _slow_read(monkeypatch, 8, {"draft": False})  # type: ignore[method-assign]
        client.merge_pull_request = AsyncMock(side_effect=[_wrapped(500), True])

        deadline = asyncio.get_running_loop().time() + 12
        assert not await self._retry(mgr, deadline)
        assert client.merge_pull_request.await_count == 1
        assert mgr._merge_server_error("o/r#1", policy_refusal=False)


class TestTheReplacementMerge:
    """A recreated PR's merge backs off as the main merge does."""

    @staticmethod
    def _manager(*outcomes: Any) -> AsyncMergeManager:
        mgr, _client = make_merge_manager(merge_method="merge", max_retries=2)
        mgr._no_wait = False
        mgr._run_deadline = None
        mgr._dispatch_recreated_merge_once = AsyncMock(  # type: ignore[method-assign]
            side_effect=list(outcomes)
        )
        mgr._recheck_pr_before_retry = AsyncMock(return_value=None)  # type: ignore[method-assign]
        return mgr

    @pytest.mark.asyncio
    async def test_three_500s_then_a_merge(self, monkeypatch) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        mgr = self._manager(*[_wrapped(500)] * 3, (True, False))

        out = await mgr._dispatch_recreated_merge("o", "r", make_pr_info(number=2))

        assert out == (True, False, False)
        assert [call.args[0] for call in sleep.await_args_list] == [2.0, 6.0, 18.0]

    @pytest.mark.asyncio
    async def test_four_500s_raise_the_server_error(self, monkeypatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        mgr = self._manager(*[_wrapped(500)] * 5)

        with pytest.raises(GitHubServerError, match=r"\(500\) while merging"):
            await mgr._dispatch_recreated_merge("o", "r", make_pr_info(number=2))
        assert cast(AsyncMock, mgr._dispatch_recreated_merge_once).await_count == 4

    @pytest.mark.asyncio
    async def test_a_merge_that_landed_is_not_repeated(self, monkeypatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        mgr = self._manager(_wrapped(500), (True, False))
        mgr._recheck_pr_before_retry = AsyncMock(return_value=True)  # type: ignore[method-assign]

        out = await mgr._dispatch_recreated_merge("o", "r", make_pr_info(number=2))

        assert out == (True, False, False)
        assert cast(AsyncMock, mgr._dispatch_recreated_merge_once).await_count == 1

    @pytest.mark.asyncio
    async def test_a_deadline_spent_reading_stops_the_retries(
        self, monkeypatch
    ) -> None:
        mgr = self._manager(_wrapped(500), (True, False))
        mgr._recheck_pr_before_retry = _slow_read(monkeypatch, 5, None)  # type: ignore[method-assign]
        mgr._run_deadline = asyncio.get_running_loop().time() + 3

        with pytest.raises(GitHubServerError, match=r"\(500\) while merging"):
            await mgr._dispatch_recreated_merge("o", "r", make_pr_info(number=2))
        assert cast(AsyncMock, mgr._dispatch_recreated_merge_once).await_count == 1

    @pytest.mark.asyncio
    async def test_a_deadline_spent_under_the_lock_sends_no_retry(
        self, monkeypatch
    ) -> None:
        mgr, client = make_merge_manager(merge_method="merge", max_retries=2)
        mgr._no_wait = False
        mgr._recheck_pr_before_retry = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._fetch_pr_state = _slow_read(monkeypatch, 8, {"draft": False})  # type: ignore[method-assign]
        mgr._run_deadline = asyncio.get_running_loop().time() + 12
        client.merge_pull_request = AsyncMock(side_effect=[_wrapped(500), True])

        with pytest.raises(GitHubServerError, match=r"\(500\) while merging"):
            await mgr._dispatch_recreated_merge("o", "r", make_pr_info(number=2))
        assert client.merge_pull_request.await_count == 1


class TestTheApproval:
    @pytest.mark.asyncio
    async def test_three_500s_then_an_approval(self, monkeypatch) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        api = GitHubAsync(token="t")
        errors = [_status_error(500, "POST")] * 3
        api.post = AsyncMock(side_effect=[*errors, {}])  # type: ignore[method-assign]
        api._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await api.approve_pull_request("o", "r", 1, "lgtm", commit_id="abc123")

        assert api.post.await_count == 4
        assert [call.args[0] for call in sleep.await_args_list] == [2.0, 6.0, 18.0]
        await api.aclose()

    @pytest.mark.asyncio
    async def test_an_approval_that_landed_is_not_posted_again(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        api = GitHubAsync(token="t")
        api.post = AsyncMock(side_effect=_status_error(502, "POST"))  # type: ignore[method-assign]
        api._has_own_approval = AsyncMock(side_effect=[False, True])  # type: ignore[method-assign]

        await api.approve_pull_request("o", "r", 1, "lgtm", commit_id="abc123")

        assert api.post.await_count == 2
        await api.aclose()

    @staticmethod
    def _failing_api(monkeypatch: pytest.MonkeyPatch) -> tuple[GitHubAsync, AsyncMock]:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        api = GitHubAsync(token="t")
        api.post = AsyncMock(side_effect=[_status_error(500, "POST"), {}])  # type: ignore[method-assign]
        api._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]
        return api, sleep

    @pytest.mark.asyncio
    async def test_the_run_deadline_shortens_the_wait(self, monkeypatch) -> None:
        api, sleep = self._failing_api(monkeypatch)
        deadline = asyncio.get_running_loop().time() + 0.5

        await api.approve_pull_request(
            "o", "r", 1, "lgtm", commit_id="abc123", deadline=deadline
        )

        (wait,) = [call.args[0] for call in sleep.await_args_list]
        assert 0 < wait <= 0.5
        await api.aclose()

    @pytest.mark.asyncio
    async def test_a_passed_deadline_stops_the_retries(self, monkeypatch) -> None:
        api, sleep = self._failing_api(monkeypatch)
        deadline = asyncio.get_running_loop().time() - 1

        with pytest.raises(httpx.HTTPStatusError):
            await api.approve_pull_request(
                "o", "r", 1, "lgtm", commit_id="abc123", deadline=deadline
            )

        assert cast(AsyncMock, api.post).await_count == 1
        sleep.assert_not_awaited()
        await api.aclose()

    @pytest.mark.asyncio
    async def test_a_deadline_spent_waiting_stops_the_retries(
        self, monkeypatch
    ) -> None:
        api, _sleep = self._failing_api(monkeypatch)
        _clock_spent_by_sleep(monkeypatch)
        deadline = asyncio.get_running_loop().time() + 0.5

        with pytest.raises(httpx.HTTPStatusError):
            await api.approve_pull_request(
                "o", "r", 1, "lgtm", commit_id="abc123", deadline=deadline
            )

        assert cast(AsyncMock, api.post).await_count == 1
        cast(AsyncMock, api._has_own_approval).assert_awaited()
        await api.aclose()

    @pytest.mark.asyncio
    async def test_a_deadline_spent_in_the_lookup_stops_the_retries(
        self, monkeypatch
    ) -> None:
        api, _sleep = self._failing_api(monkeypatch)
        api._has_own_approval = _slow_read(monkeypatch, 5, False)  # type: ignore[method-assign]
        deadline = asyncio.get_running_loop().time() + 3

        with pytest.raises(httpx.HTTPStatusError):
            await api.approve_pull_request(
                "o", "r", 1, "lgtm", commit_id="abc123", deadline=deadline
            )

        assert cast(AsyncMock, api.post).await_count == 1
        assert cast(AsyncMock, api._has_own_approval).await_count == 1
        await api.aclose()

    @pytest.mark.asyncio
    async def test_a_late_lookup_still_finds_the_approval(self, monkeypatch) -> None:
        api, _sleep = self._failing_api(monkeypatch)
        api._has_own_approval = _slow_read(monkeypatch, 5, True)  # type: ignore[method-assign]
        deadline = asyncio.get_running_loop().time() + 3

        await api.approve_pull_request(
            "o", "r", 1, "lgtm", commit_id="abc123", deadline=deadline
        )

        assert cast(AsyncMock, api.post).await_count == 1
        await api.aclose()

    @pytest.mark.asyncio
    async def test_max_wait_zero_keeps_the_short_waits(self, monkeypatch) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        api = GitHubAsync(token="t")
        api.post = AsyncMock(side_effect=_status_error(500, "POST"))  # type: ignore[method-assign]
        api._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        with pytest.raises(httpx.HTTPStatusError):
            await api.approve_pull_request(
                "o", "r", 1, "lgtm", commit_id="abc123", short_waits=True
            )

        assert api.post.await_count == 3
        assert [call.args[0] for call in sleep.await_args_list] == [2.0, 4.0]
        await api.aclose()

    @pytest.mark.parametrize("no_wait", [True, False])
    @pytest.mark.asyncio
    async def test_the_manager_passes_max_wait_zero_on(self, no_wait: bool) -> None:
        mgr, client = make_merge_manager()
        mgr._no_wait = no_wait
        approve = AsyncMock()
        client.approve_pull_request = approve

        await approve_at_head(mgr, "o", "r", make_pr_info(), "abc123")

        approve.assert_awaited_once()
        assert approve.call_args.kwargs["short_waits"] is no_wait

    @pytest.mark.asyncio
    async def test_the_wait_lets_go_of_the_dispatch_lock(self, monkeypatch) -> None:
        api, _sleep = self._failing_api(monkeypatch)
        lock = asyncio.Lock()
        held_during_wait: list[bool] = []

        async def sleep(_delay: float) -> None:
            held_during_wait.append(lock.locked())

        monkeypatch.setattr(asyncio, "sleep", sleep)

        async with holding_dispatch_lock(lock):
            await api.approve_pull_request("o", "r", 1, "lgtm", commit_id="abc123")
            assert lock.locked()
        assert held_during_wait == [False]
        await api.aclose()


class TestTheDispatchLock:
    """A wait under a repository's dispatch lock must not strand a sibling."""

    @staticmethod
    async def _siblings(park: Any) -> None:
        slots = asyncio.Semaphore(1)
        lock = asyncio.Lock()
        sibling_merged = asyncio.Event()

        async def sibling() -> None:
            async with holding_slot(slots), holding_dispatch_lock(lock):
                sibling_merged.set()

        tasks: list[asyncio.Task[None]] = []

        async def waiter() -> None:
            async with holding_slot(slots), holding_dispatch_lock(lock):
                tasks.append(asyncio.create_task(sibling()))
                async with park():
                    await asyncio.sleep(0.05)
                assert lock.locked()
            await tasks[0]

        try:
            await asyncio.wait_for(waiter(), timeout=1)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert sibling_merged.is_set()

    @pytest.mark.asyncio
    async def test_parking_off_the_lock_lets_a_sibling_through(self) -> None:
        await self._siblings(parked_off_dispatch)

    @pytest.mark.asyncio
    async def test_parking_under_the_lock_deadlocks(self) -> None:
        with pytest.raises(asyncio.TimeoutError):
            await self._siblings(parked)

    @pytest.mark.asyncio
    async def test_a_cancelled_park_does_not_wait_for_the_lock(self) -> None:
        """Cancelled while taking its slot back, a task unwinds at once."""
        slots = asyncio.Semaphore(1)
        lock = asyncio.Lock()
        parked_now = asyncio.Event()
        sibling_has_lock = asyncio.Event()
        release_sibling = asyncio.Event()

        async def waiter() -> None:
            async with holding_slot(slots), holding_dispatch_lock(lock):
                async with parked_off_dispatch():
                    parked_now.set()
                    await sibling_has_lock.wait()

        async def sibling() -> None:
            await parked_now.wait()
            async with holding_slot(slots), holding_dispatch_lock(lock):
                sibling_has_lock.set()
                await release_sibling.wait()

        sib = asyncio.create_task(sibling())
        task = asyncio.create_task(waiter())
        await sibling_has_lock.wait()
        # The waiter is now queued for the slot the sibling holds.
        await asyncio.sleep(0)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert lock.locked()
        finally:
            release_sibling.set()
            await sib
        assert not lock.locked()
        assert not slots.locked()

    @pytest.mark.asyncio
    async def test_a_park_cancelled_in_its_wait_takes_nothing_back(self) -> None:
        """Cancelled mid-wait, a task does not wait for a sibling's slot."""
        slots = asyncio.Semaphore(1)
        lock = asyncio.Lock()
        sibling_has_slot = asyncio.Event()
        release_sibling = asyncio.Event()

        async def sibling() -> None:
            async with holding_slot(slots):
                sibling_has_slot.set()
                await release_sibling.wait()

        sib: list[asyncio.Task[None]] = []

        async def waiter() -> None:
            async with holding_slot(slots), holding_dispatch_lock(lock):
                async with parked_off_dispatch():
                    sib.append(asyncio.create_task(sibling()))
                    await asyncio.Event().wait()

        task = asyncio.create_task(waiter())
        try:
            await sibling_has_slot.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert not lock.locked()
        finally:
            release_sibling.set()
            await asyncio.gather(*sib)
        assert not slots.locked()

    @pytest.mark.asyncio
    async def test_a_park_with_no_lock_cancelled_takes_nothing_back(self) -> None:
        """An approval's wait before dispatch holds no lock, and unwinds too."""
        slots = asyncio.Semaphore(1)
        sibling_has_slot = asyncio.Event()
        release_sibling = asyncio.Event()

        async def sibling() -> None:
            async with holding_slot(slots):
                sibling_has_slot.set()
                await release_sibling.wait()

        sib: list[asyncio.Task[None]] = []

        async def waiter() -> None:
            async with holding_slot(slots), parked_off_dispatch():
                sib.append(asyncio.create_task(sibling()))
                await asyncio.Event().wait()

        task = asyncio.create_task(waiter())
        try:
            await sibling_has_slot.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
        finally:
            release_sibling.set()
            await asyncio.gather(*sib)
        assert not slots.locked()
