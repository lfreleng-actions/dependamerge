# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for which Copilot review threads ``--dismiss-copilot`` resolves."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dependamerge.copilot_handler import CopilotCommentHandler
from dependamerge.github_graphql import (
    GET_PR_REVIEW_THREADS,
    GET_REVIEW_THREAD_COMMENTS,
    RESOLVE_REVIEW_THREAD,
)

XFAIL = pytest.mark.xfail(strict=True, reason="Copilot thread judgement not fixed yet")

COPILOT = "copilot-pull-request-reviewer"
HUMAN = "octocat"


def _comment(login: str | None, body: str) -> dict[str, Any]:
    author = None if login is None else {"login": login}
    return {"id": f"c-{body[:8]}", "author": author, "body": body, "createdAt": ""}


def _thread(
    thread_id: str,
    comments: list[dict[str, Any]],
    *,
    outdated: bool = False,
    resolved: bool = False,
    has_next: bool = False,
    end_cursor: str | None = None,
) -> dict[str, Any]:
    return {
        "id": thread_id,
        "isResolved": resolved,
        "isOutdated": outdated,
        "path": "file.py",
        "line": 1,
        "comments": {
            "pageInfo": {"hasNextPage": has_next, "endCursor": end_cursor},
            "nodes": comments,
        },
    }


def _client(
    threads: list[dict[str, Any]],
    comment_pages: dict[tuple[str, str], Any] | None = None,
) -> MagicMock:
    """Return a client whose ``graphql`` answers the thread queries."""
    pages = comment_pages or {}

    async def graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if query == GET_PR_REVIEW_THREADS:
            return {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": threads,
                        }
                    }
                }
            }
        if query == GET_REVIEW_THREAD_COMMENTS:
            page = pages[(variables["threadId"], variables["cursor"])]
            if isinstance(page, Exception):
                raise page
            return {"node": {"comments": page}}
        if query == RESOLVE_REVIEW_THREAD:
            return {
                "resolveReviewThread": {
                    "thread": {"id": variables["threadId"], "isResolved": True}
                }
            }
        raise AssertionError(f"unexpected query: {query}")

    client = MagicMock()
    client.graphql = AsyncMock(side_effect=graphql)
    return client


def _resolved_ids(client: MagicMock) -> list[str]:
    return [
        call.args[1]["threadId"]
        for call in client.graphql.await_args_list
        if call.args[0] == RESOLVE_REVIEW_THREAD
    ]


async def _run(
    threads: list[dict[str, Any]],
    comment_pages: dict[tuple[str, str], Any] | None = None,
    *,
    preview: bool = False,
) -> tuple[tuple[int, int], MagicMock]:
    client = _client(threads, comment_pages)
    handler = CopilotCommentHandler(client, preview_mode=preview)
    result = await handler.resolve_copilot_threads_for_commented_review(
        "org", "repo", 1, "R1"
    )
    return result, client


class TestCopilotThreadIdentity:
    """A thread is Copilot's only when Copilot wrote its first comment."""

    @XFAIL
    async def test_human_thread_mentioning_copilot_is_not_resolved(self) -> None:
        threads = [
            _thread(
                "T1",
                [
                    _comment(HUMAN, "Please rename this"),
                    _comment(HUMAN, "GitHub Copilot suggested the same"),
                ],
            ),
            _thread("T2", [_comment(HUMAN, "🤖 looks generated")]),
        ]
        handler = CopilotCommentHandler(MagicMock())
        assert not handler.is_copilot_thread(threads[0])
        assert not handler.is_copilot_thread(threads[1])

        result, client = await _run(threads)

        assert result == (0, 2)
        assert _resolved_ids(client) == []

    @XFAIL
    def test_human_thread_with_copilot_reply_is_not_copilots(self) -> None:
        thread = _thread("T1", [_comment(HUMAN, "Fix this"), _comment(COPILOT, "Done")])
        assert not CopilotCommentHandler(MagicMock()).is_copilot_thread(thread)

    @XFAIL
    def test_thread_without_readable_first_author_is_not_copilots(self) -> None:
        handler = CopilotCommentHandler(MagicMock())
        assert not handler.is_copilot_thread(
            _thread("T1", [_comment(None, "x"), _comment(COPILOT, "y")])
        )
        assert not handler.is_copilot_thread(_thread("T2", []))

    @pytest.mark.parametrize(
        "login",
        [
            "copilot-pull-request-reviewer",
            "copilot-pull-request-reviewer[bot]",
            "Copilot",
            "github-copilot[bot]",
            "copilot-swe-agent",
        ],
    )
    def test_thread_opened_by_copilot_is_copilots(self, login: str) -> None:
        thread = _thread("T1", [_comment(login, "Nit")])
        assert CopilotCommentHandler(MagicMock()).is_copilot_thread(thread)


class TestCopilotThreadSafety:
    """Any unsafe keyword in a thread keeps it open."""

    @XFAIL
    def test_unsafe_keyword_beats_safe_phrase_in_one_comment(self) -> None:
        thread = _thread(
            "T1",
            [
                _comment(
                    COPILOT,
                    "Consider using a parameterised query, this is a security bug",
                )
            ],
        )
        assert not CopilotCommentHandler(MagicMock()).is_safe_copilot_thread_to_resolve(
            thread
        )

    @XFAIL
    def test_unsafe_keyword_in_later_comment(self) -> None:
        thread = _thread(
            "T1",
            [
                _comment(COPILOT, "Style: rename this"),
                _comment(HUMAN, "This hides a real bug"),
            ],
        )
        assert not CopilotCommentHandler(MagicMock()).is_safe_copilot_thread_to_resolve(
            thread
        )

    @XFAIL
    def test_outdated_thread_with_unsafe_keyword_is_not_safe(self) -> None:
        handler = CopilotCommentHandler(MagicMock())
        unsafe = _thread("T1", [_comment(COPILOT, "Vulnerability here")], outdated=True)
        safe = _thread("T2", [_comment(COPILOT, "Rename this")], outdated=True)
        assert not handler.is_safe_copilot_thread_to_resolve(unsafe)
        assert handler.is_safe_copilot_thread_to_resolve(safe)

    @XFAIL
    def test_unread_comment_pages_are_not_safe(self) -> None:
        thread = _thread(
            "T1", [_comment(COPILOT, "Nit")], has_next=True, end_cursor="c1"
        )
        assert not CopilotCommentHandler(MagicMock()).is_safe_copilot_thread_to_resolve(
            thread
        )


class TestCopilotThreadResolution:
    """End-to-end thread selection through the commented-review path."""

    async def test_copilot_thread_without_keyword_is_resolved(self) -> None:
        threads = [_thread("T1", [_comment(COPILOT, "Rename this variable")])]

        result, client = await _run(threads)

        assert result == (1, 1)
        assert _resolved_ids(client) == ["T1"]

    async def test_resolved_thread_is_not_resolved_again(self) -> None:
        threads = [_thread("T1", [_comment(COPILOT, "Nit")], resolved=True)]

        result, client = await _run(threads)

        assert result == (0, 1)
        assert _resolved_ids(client) == []

    @XFAIL
    async def test_every_comment_page_is_read(self) -> None:
        first = [_comment(COPILOT, "Style: nit")] + [
            _comment(HUMAN, f"reply {i}") for i in range(10)
        ]
        later = {
            "pageInfo": {"hasNextPage": True, "endCursor": "c2"},
            "nodes": [_comment(HUMAN, f"more {i}") for i in range(5)],
        }
        last = {
            "pageInfo": {"hasNextPage": False, "endCursor": "c3"},
            "nodes": [_comment(HUMAN, "This is still broken")],
        }
        threads = [_thread("T1", first, has_next=True, end_cursor="c1")]

        result, client = await _run(threads, {("T1", "c1"): later, ("T1", "c2"): last})

        assert result == (0, 1)
        assert _resolved_ids(client) == []
        assert len(threads[0]["comments"]["nodes"]) == 17

    async def test_further_pages_read_before_resolving(self) -> None:
        last = {
            "pageInfo": {"hasNextPage": False, "endCursor": "c2"},
            "nodes": [_comment(HUMAN, "Thanks")],
        }
        threads = [
            _thread("T1", [_comment(COPILOT, "Nit")], has_next=True, end_cursor="c1")
        ]

        result, client = await _run(threads, {("T1", "c1"): last})

        assert result == (1, 1)
        assert _resolved_ids(client) == ["T1"]

    @XFAIL
    async def test_failed_comment_page_leaves_thread_unresolved(self) -> None:
        threads = [
            _thread("T1", [_comment(COPILOT, "Nit")], has_next=True, end_cursor="c1")
        ]

        result, client = await _run(threads, {("T1", "c1"): RuntimeError("boom")})

        assert result == (0, 1)
        assert _resolved_ids(client) == []

    async def test_preview_mode_resolves_nothing(self) -> None:
        threads = [_thread("T1", [_comment(COPILOT, "Rename this variable")])]

        result, client = await _run(threads, preview=True)

        assert result == (1, 1)
        assert _resolved_ids(client) == []

    @XFAIL
    @pytest.mark.parametrize(
        "malformed",
        [
            [None],
            [{}],
            [{"author": {"login": HUMAN}, "body": None}],
            [{"author": {"login": HUMAN}, "body": 42}],
        ],
    )
    async def test_malformed_final_comment_page_leaves_thread_unresolved(
        self, malformed: list[Any]
    ) -> None:
        last = {
            "pageInfo": {"hasNextPage": False, "endCursor": "c2"},
            "nodes": malformed,
        }
        threads = [
            _thread("T1", [_comment(COPILOT, "Nit")], has_next=True, end_cursor="c1")
        ]

        result, client = await _run(threads, {("T1", "c1"): last})

        assert result == (0, 1)
        assert _resolved_ids(client) == []
