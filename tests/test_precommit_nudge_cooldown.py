# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""The ``pre-commit.ci run`` nudge is posted at most once per cooldown.

pre-commit.ci answers a nudge with a fresh ``queued`` status.  A guard
that only counts comments newer than the status reads that answer as a
new stall, so every later run posted another comment.  The cooldown is
keyed on the newest exact trigger comment instead, whoever posted it.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dependamerge.github_async._reviews import IssueComments
from dependamerge.merge_manager import AsyncMergeManager
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

NOW = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
CONTEXT = "pre-commit.ci - pr"


@pytest.fixture(autouse=True)
def _clock() -> Iterator[None]:
    with (
        patch("dependamerge.merge_manager._precommit_ci._utcnow", return_value=NOW),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        yield


def _iso(ago: timedelta) -> str:
    return (NOW - ago).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pr() -> PullRequestInfo:
    return PullRequestInfo(
        number=7,
        title="Chore: Bump foo from 1 to 2",
        body="bump",
        author="dependabot[bot]",
        head_sha="cafe" * 10,
        base_branch="main",
        head_branch="dependabot/pip/foo-2",
        state="open",
        mergeable=False,
        mergeable_state="blocked",
        behind_by=0,
        files_changed=[],
        repository_full_name="o/r",
        html_url="https://github.com/o/r/pull/7",
    )


def _manager(
    status: dict[str, Any] | None, comments: Any
) -> tuple[AsyncMergeManager, AsyncMock]:
    mgr, client = make_merge_manager(preview_mode=False)
    client.get_required_status_checks = AsyncMock(return_value=[{"context": CONTEXT}])
    statuses = [status] if status is not None else []
    answered = {"statuses": [{"context": CONTEXT, "state": "success"}]}
    client.get.side_effect = [{"statuses": statuses}, answered, answered]
    if isinstance(comments, BaseException):
        client.list_issue_comments = AsyncMock(side_effect=comments)
    else:
        client.list_issue_comments = AsyncMock(return_value=comments)
    client.post_issue_comment = AsyncMock()
    return mgr, client


def _queued(ago: timedelta) -> dict[str, Any]:
    return {"context": CONTEXT, "state": "pending", "updated_at": _iso(ago)}


def _nudge(ago: timedelta, body: str = "pre-commit.ci run") -> dict[str, Any]:
    return {"body": body, "created_at": _iso(ago), "user": {"login": "someone"}}


class TestTheCooldown:
    @pytest.mark.asyncio
    async def test_a_requeued_status_after_a_recent_nudge_posts_nothing(self) -> None:
        """The case the old guard got wrong: the answer is not a new stall."""
        mgr, client = _manager(
            _queued(timedelta(minutes=10)),
            IssueComments([_nudge(timedelta(minutes=20))], complete=True),
        )

        assert await mgr._trigger_stale_precommit_ci(_pr()) is False

        client.post_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_nudge_two_hours_old_posts(self) -> None:
        mgr, client = _manager(
            _queued(timedelta(minutes=10)),
            IssueComments([_nudge(timedelta(hours=2))], complete=True),
        )

        await mgr._trigger_stale_precommit_ci(_pr())

        client.post_issue_comment.assert_called_once_with(
            "o", "r", 7, "pre-commit.ci run"
        )

    @pytest.mark.asyncio
    async def test_the_newest_of_several_comments_decides(self) -> None:
        mgr, client = _manager(
            _queued(timedelta(minutes=10)),
            IssueComments(
                [
                    _nudge(timedelta(hours=5)),
                    _nudge(timedelta(minutes=30)),
                    _nudge(timedelta(hours=3)),
                ],
                complete=True,
            ),
        )

        assert await mgr._trigger_stale_precommit_ci(_pr()) is False
        client.post_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_comment_quoting_the_phrase_does_not_count(self) -> None:
        mgr, client = _manager(
            _queued(timedelta(minutes=10)),
            IssueComments(
                [_nudge(timedelta(minutes=5), body="try `pre-commit.ci run` here")],
                complete=True,
            ),
        )

        await mgr._trigger_stale_precommit_ci(_pr())

        client.post_issue_comment.assert_called_once()


class TestUnreadableComments:
    @pytest.mark.asyncio
    async def test_a_failed_read_posts(self) -> None:
        mgr, client = _manager(
            _queued(timedelta(minutes=10)), RuntimeError("comments unavailable")
        )

        await mgr._trigger_stale_precommit_ci(_pr())

        client.post_issue_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_an_incomplete_read_without_a_recent_nudge_posts(self) -> None:
        mgr, client = _manager(
            _queued(timedelta(minutes=10)),
            IssueComments([_nudge(timedelta(hours=2))], complete=False),
        )

        await mgr._trigger_stale_precommit_ci(_pr())

        client.post_issue_comment.assert_called_once()


class TestTheNudgeRecord:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("reading", ["fails", "misses-it"])
    async def test_a_second_repair_honours_the_nudge_the_first_posted(
        self, reading: str
    ) -> None:
        """The post is remembered when the next comment reading misses it."""
        stuck = {"statuses": [_queued(timedelta(minutes=10))]}
        answered = {"statuses": [{"context": CONTEXT, "state": "success"}]}
        mgr, client = _manager(None, IssueComments([], complete=True))
        client.get.side_effect = [stuck, answered, stuck, answered, answered]
        assert await mgr._trigger_stale_precommit_ci(_pr()) is True
        client.post_issue_comment.assert_called_once()
        if reading == "fails":
            client.list_issue_comments = AsyncMock(side_effect=RuntimeError("boom"))
        else:
            client.list_issue_comments = AsyncMock(
                return_value=IssueComments([], complete=False)
            )

        assert await mgr._trigger_stale_precommit_ci(_pr()) is False

        client.post_issue_comment.assert_called_once()


class TestWhatIsNudged:
    @pytest.mark.asyncio
    async def test_a_failure_is_never_nudged(self) -> None:
        mgr, client = _manager(
            {"context": CONTEXT, "state": "failure", "updated_at": _iso(timedelta(1))},
            IssueComments([], complete=True),
        )

        assert await mgr._trigger_stale_precommit_ci(_pr()) is False
        client.post_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_error_outside_the_cooldown_is_nudged(self) -> None:
        mgr, client = _manager(
            {"context": CONTEXT, "state": "error", "updated_at": _iso(timedelta(0))},
            IssueComments([_nudge(timedelta(hours=2))], complete=True),
        )

        await mgr._trigger_stale_precommit_ci(_pr())

        client.post_issue_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_an_error_inside_the_cooldown_is_not(self) -> None:
        mgr, client = _manager(
            {"context": CONTEXT, "state": "error", "updated_at": _iso(timedelta(0))},
            IssueComments([_nudge(timedelta(minutes=20))], complete=True),
        )

        assert await mgr._trigger_stale_precommit_ci(_pr()) is False
        client.post_issue_comment.assert_not_called()
