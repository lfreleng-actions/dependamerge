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

from dependamerge.github_async import GitHubAsync
from dependamerge.merge_manager import AsyncMergeManager
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

    @pytest.mark.xfail(strict=True, reason="server errors not yet retried")
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

    @pytest.mark.xfail(strict=True, reason="server errors not yet retried")
    @pytest.mark.asyncio
    async def test_three_500s_then_a_merge(self, monkeypatch) -> None:
        sleep = self._sleeps(monkeypatch)
        mgr = self._manager(_wrapped(500), _wrapped(500), _wrapped(500), True)

        assert await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        assert self._waits(sleep) == [2.0, 6.0, 18.0]

    @pytest.mark.xfail(strict=True, reason="server errors not yet retried")
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

    @pytest.mark.xfail(strict=True, reason="server errors not yet retried")
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

    @pytest.mark.xfail(strict=True, reason="server errors not yet retried")
    @pytest.mark.asyncio
    async def test_the_run_deadline_shortens_the_wait(self, monkeypatch) -> None:
        sleep = self._sleeps(monkeypatch)
        mgr = self._manager(_wrapped(500), True)
        mgr._run_deadline = asyncio.get_running_loop().time() + 0.5

        assert await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        (wait,) = self._waits(sleep)
        assert 0 < wait <= 0.5

    @pytest.mark.xfail(strict=True, reason="server errors not yet retried")
    @pytest.mark.asyncio
    async def test_a_passed_deadline_stops_the_retries(self, monkeypatch) -> None:
        sleep = self._sleeps(monkeypatch)
        mgr = self._manager(_wrapped(500), True)
        mgr._run_deadline = asyncio.get_running_loop().time() - 1

        assert not await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")
        sleep.assert_not_awaited()

    @pytest.mark.parametrize("status", [403, 405, 409, 422])
    @pytest.mark.asyncio
    async def test_a_client_error_is_not_backed_off(self, status, monkeypatch) -> None:
        sleep = self._sleeps(monkeypatch)
        mgr = self._manager(*[_wrapped(status)] * 5)
        mgr._decide_after_405 = AsyncMock()  # type: ignore[method-assign]

        await mgr._merge_pr_with_retry(make_pr_info(), "o", "r")

        assert not {2.0, 6.0, 18.0} & set(self._waits(sleep))


class TestTheApproval:
    @pytest.mark.xfail(strict=True, reason="server errors not yet retried")
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

    @pytest.mark.xfail(strict=True, reason="server errors not yet retried")
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
