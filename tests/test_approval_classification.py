# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Classifying approval failures by HTTP status and GitHub's own message.

A bulk run reported one throttled approval as a token permission
problem: every approval ``422`` became a ``PermissionError``, because
the classifier matched httpx's status line rather than reading the
response.  The status and body decide here instead.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

import dependamerge.github_async as ga
from dependamerge.github_async import GitHubAsync
from tests.conftest import make_merge_manager


def _status_error(
    status: int,
    body: Any = None,
    *,
    path: str = "/repos/o/r/pulls/1/reviews",
    method: str = "POST",
    headers: dict[str, str] | None = None,
) -> httpx.HTTPStatusError:
    """A real ``HTTPStatusError``, with httpx's own message and URL."""
    request = httpx.Request(method, f"https://api.github.com{path}")
    response = httpx.Response(status, json=body, headers=headers, request=request)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise AssertionError("raise_for_status did not raise")


THROTTLED = {
    "message": "Validation Failed",
    "errors": ["was submitted too quickly"],
}
SELF_APPROVAL = {
    "message": "Unprocessable Entity",
    "errors": ["Can not approve your own pull request"],
}
OTHER = {
    "message": "Validation Failed",
    "errors": [{"resource": "PullRequestReview", "message": "Head sha moved"}],
}


class TestErrorDetail:
    def test_reads_message_and_errors(self) -> None:
        err = _status_error(
            422,
            {
                "message": "Validation Failed",
                "errors": ["first", {"message": "second"}, {"code": "custom"}],
            },
        )
        assert ga.github_error_detail(err) == "Validation Failed; first; second"

    def test_without_response_is_empty(self) -> None:
        assert ga.github_error_detail(RuntimeError("boom")) == ""


class TestApprovalClassification:
    @pytest.mark.parametrize(
        ("body", "permission"),
        [
            (SELF_APPROVAL, True),
            (
                {
                    "message": "Validation Failed",
                    "errors": [
                        {
                            "message": "Review cannot be requested from "
                            "pull request author."
                        }
                    ],
                },
                True,
            ),
            (THROTTLED, False),
            ({"message": "You have exceeded a secondary rate limit"}, False),
            (OTHER, False),
            (None, False),
        ],
    )
    def test_only_a_refused_approval_is_a_permission_error(
        self, body: Any, permission: bool
    ) -> None:
        c = GitHubAsync(token="t")
        perm = c._parse_permission_error(_status_error(422, body), "approve", "o", "r")
        assert (perm is not None) is permission
        if perm is not None:
            assert getattr(perm, "status_code", None) == 422

    @pytest.mark.parametrize("number", [401, 403])
    @pytest.mark.parametrize("operation", ["approve", "merge", "close"])
    def test_url_digits_do_not_set_the_status(
        self, number: int, operation: str
    ) -> None:
        c = GitHubAsync(token="t")
        err = _status_error(422, OTHER, path=f"/repos/o/r/pulls/{number}/reviews")
        assert str(number) in str(err)
        assert c._parse_permission_error(err, operation, "o", "r") is None

    def test_a_repository_named_with_403_is_not_forbidden(self) -> None:
        c = GitHubAsync(token="t")
        err = _status_error(500, None, path="/repos/o/r403/pulls/7/merge")
        assert c._parse_permission_error(err, "merge", "o", "r403") is None

    def test_a_genuine_403_still_is(self) -> None:
        c = GitHubAsync(token="t")
        err = _status_error(403, {"message": "Resource not accessible"})
        perm = c._parse_permission_error(err, "approve", "o", "r")
        assert perm is not None
        assert getattr(perm, "status_code", None) == 403


class TestApprovePullRequest:
    @pytest.mark.asyncio
    async def test_other_422_carries_githubs_message(self) -> None:
        c = GitHubAsync(token="t")
        c.post = AsyncMock(side_effect=_status_error(422, OTHER))  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        with pytest.raises(ga.ApprovalError) as info:
            await c.approve_pull_request("o", "r", 1, "lgtm")

        assert "Head sha moved" in str(info.value)
        assert not isinstance(info.value, ga.PermissionError)
        c.post.assert_awaited_once()
        await c.aclose()

    @pytest.mark.asyncio
    async def test_throttled_then_success_approves_once(self, monkeypatch) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        c = GitHubAsync(token="t")
        c.post = AsyncMock(side_effect=[_status_error(422, THROTTLED), {}])  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await c.approve_pull_request("o", "r", 1, "lgtm")

        assert c.post.await_count == 2
        c._has_own_approval.assert_awaited_once()
        # GitHub advises at least a minute for a secondary limit.
        assert sleep.await_args_list[-1].args[0] >= 60
        await c.aclose()

    @pytest.mark.asyncio
    async def test_throttled_honours_retry_after(self, monkeypatch) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        c = GitHubAsync(token="t")
        throttled = _status_error(422, THROTTLED, headers={"Retry-After": "90"})
        c.post = AsyncMock(side_effect=[throttled, {}])  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await c.approve_pull_request("o", "r", 1, "lgtm")

        assert sleep.await_args_list[-1].args[0] == 90
        await c.aclose()

    @pytest.mark.asyncio
    async def test_throttled_but_landed_is_success(self, monkeypatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        c = GitHubAsync(token="t")
        c.post = AsyncMock(side_effect=_status_error(422, THROTTLED))  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=True)  # type: ignore[method-assign]

        await c.approve_pull_request("o", "r", 1, "lgtm")

        c.post.assert_awaited_once()
        await c.aclose()

    @pytest.mark.asyncio
    async def test_throttled_until_exhausted_reports_githubs_message(
        self, monkeypatch
    ) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        c = GitHubAsync(token="t")
        c.post = AsyncMock(side_effect=_status_error(422, THROTTLED))  # type: ignore[method-assign]
        c._has_own_approval = AsyncMock(return_value=False)  # type: ignore[method-assign]

        with pytest.raises(ga.ApprovalError) as info:
            await c.approve_pull_request("o", "r", 1, "lgtm")

        assert "submitted too quickly" in str(info.value)
        delays = [call.args[0] for call in sleep.await_args_list]
        assert delays == sorted(delays)
        assert delays[0] >= 60
        await c.aclose()


class TestManagerReportsGitHubsMessage:
    @pytest.mark.asyncio
    async def test_refused_approval_is_not_read_as_already_approved(self) -> None:
        mgr, client = make_merge_manager()
        mgr._should_skip_approval = AsyncMock(return_value=(False, None))  # type: ignore[method-assign]
        client.approve_pull_request = AsyncMock(
            side_effect=ga.ApprovalError("o/r#1", 422, "Head sha moved")
        )

        with pytest.raises(Exception, match="Head sha moved"):
            await mgr._approve_pr("o", "r", 1)
