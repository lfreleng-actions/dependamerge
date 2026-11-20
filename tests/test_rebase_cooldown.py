# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Pacing of ``@dependabot rebase``: in flight, cooling down, or due.

The newest ``@dependabot rebase`` comment decides.  Younger than
``REBASE_IN_FLIGHT_SECONDS`` on an unmoved head, a rebase is under way
and callers wait as before.  Otherwise younger than the cooldown,
nothing is posted, nothing is counted, and no caller waits, falls back
to another rebase path, records a refusal or arms auto-merge on its
strength.  Otherwise the macro is posted.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dependamerge.github_async import GitHubAsync, IssueComments
from dependamerge.merge_manager import AsyncMergeManager, MergeResult, MergeStatus
from dependamerge.merge_manager._single_pr_context import _MergeFlow
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

NOW = datetime(2026, 11, 20, 14, 35, tzinfo=UTC)
NOTE = "rebase last requested 3h 0m ago, next request in 5h 0m"


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _pr(state: str = "behind") -> PullRequestInfo:
    return PullRequestInfo(
        number=125,
        title="Chore: Bump foo from 1 to 2",
        body="bump",
        author="dependabot[bot]",
        head_sha="beef" * 10,
        base_branch="main",
        head_branch="dependabot/pip/foo-2",
        state="open",
        mergeable=False,
        mergeable_state=state,
        behind_by=1,
        files_changed=[],
        repository_full_name="o/r",
        html_url="https://github.com/o/r/pull/125",
    )


def _manager(
    request_age: timedelta | None,
    head_age: timedelta,
    **overrides: Any,
) -> tuple[AsyncMergeManager, AsyncMock]:
    """A manager whose PR was last asked to rebase *request_age* ago."""
    mgr, client = make_merge_manager(**overrides)
    comments = [{"body": "LGTM", "created_at": _iso(NOW - timedelta(days=3))}]
    if request_age is not None:
        comments.append(
            {"body": "@dependabot rebase", "created_at": _iso(NOW - request_age)}
        )
    client.list_issue_comments = AsyncMock(return_value=IssueComments(comments))
    client.get = AsyncMock(
        return_value={"commit": {"committer": {"date": _iso(NOW - head_age)}}}
    )
    client.post_issue_comment = AsyncMock()
    mgr._record_rebase = MagicMock()  # type: ignore[method-assign]
    mgr._record_retrigger = MagicMock()  # type: ignore[method-assign]
    mgr._refuse_unsigned_rebase = AsyncMock(return_value=True)  # type: ignore[method-assign]
    mgr._enable_auto_merge_with_approval = AsyncMock(return_value=True)  # type: ignore[method-assign]
    return mgr, client


def _cooling(**overrides: Any) -> tuple[AsyncMergeManager, AsyncMock]:
    """Asked 3 hours ago, and dependabot rebased 1 minute later."""
    return _manager(
        timedelta(hours=3), timedelta(hours=3) - timedelta(minutes=1), **overrides
    )


def _assert_nothing_done(mgr: AsyncMergeManager, client: AsyncMock) -> None:
    client.post_issue_comment.assert_not_called()
    client.update_branch.assert_not_called()
    mgr._record_rebase.assert_not_called()  # type: ignore[attr-defined]
    mgr._record_retrigger.assert_not_called()  # type: ignore[attr-defined]
    mgr._refuse_unsigned_rebase.assert_not_called()  # type: ignore[attr-defined]
    mgr._enable_auto_merge_with_approval.assert_not_called()  # type: ignore[attr-defined]
    assert not mgr._rebase_requested
    assert not mgr._rebased_prs
    assert not mgr._local_rebase_refused


@pytest.fixture(autouse=True)
def _clock() -> Iterator[None]:
    with patch("dependamerge.merge_manager._rebase_request._utcnow", return_value=NOW):
        yield


class TestDecision:
    """``_request_dependabot_rebase`` on its own."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cooldown", [8.0, 0.0])
    async def test_in_flight_posts_nothing(self, cooldown: float) -> None:
        mgr, client = _manager(
            timedelta(minutes=2), timedelta(hours=1), rebase_cooldown=cooldown
        )

        assert await mgr._request_dependabot_rebase(_pr(), "o", "r") is True

        client.post_issue_comment.assert_not_called()
        mgr._record_rebase.assert_not_called()  # type: ignore[attr-defined]
        assert not mgr._rebase_suppressed

    @pytest.mark.asyncio
    async def test_cooling_down_after_rebase_is_suppressed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mgr, client = _cooling()

        with caplog.at_level(logging.INFO):
            assert await mgr._request_dependabot_rebase(_pr(), "o", "r") is False

        _assert_nothing_done(mgr, client)
        assert mgr._rebase_suppressed == {"o/r#125": NOTE}
        assert f"Rebase cooling down: https://github.com/o/r/pull/125 [{NOTE}]" in (
            caplog.text
        )

    @pytest.mark.asyncio
    async def test_cooling_down_after_unanswered_request(self) -> None:
        """Head not moved, but past the in-flight window: not in flight."""
        mgr, client = _manager(timedelta(hours=3), timedelta(hours=5))

        assert await mgr._request_dependabot_rebase(_pr(), "o", "r") is False

        client.post_issue_comment.assert_not_called()
        assert "o/r#125" in mgr._rebase_suppressed

    @pytest.mark.asyncio
    async def test_suppression_announced_once_per_run(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mgr, _client = _cooling()

        with caplog.at_level(logging.INFO):
            await mgr._request_dependabot_rebase(_pr(), "o", "r")
            await mgr._request_dependabot_rebase(_pr(), "o", "r")

        assert caplog.text.count("Rebase cooling down") == 1

    @pytest.mark.asyncio
    async def test_due_after_cooldown_posts(self) -> None:
        mgr, client = _manager(timedelta(hours=9), timedelta(hours=9))

        assert await mgr._request_dependabot_rebase(_pr(), "o", "r") is True

        client.post_issue_comment.assert_awaited_once_with(
            "o", "r", 125, "@dependabot rebase"
        )
        mgr._record_rebase.assert_called_once()  # type: ignore[attr-defined]
        mgr._record_retrigger.assert_called_once()  # type: ignore[attr-defined]
        assert not mgr._rebase_suppressed

    @pytest.mark.asyncio
    async def test_cooldown_zero_posts_once_head_moved(self) -> None:
        mgr, client = _cooling(rebase_cooldown=0)

        assert await mgr._request_dependabot_rebase(_pr(), "o", "r") is True

        client.post_issue_comment.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("cooldown", "posted"), [(8.0, False), (0.0, True)])
    async def test_recent_request_on_moved_head_is_not_in_flight(
        self, cooldown: float, posted: bool
    ) -> None:
        mgr, client = _manager(
            timedelta(minutes=2), timedelta(minutes=1), rebase_cooldown=cooldown
        )

        assert await mgr._request_dependabot_rebase(_pr(), "o", "r") is posted

        assert client.post_issue_comment.await_count == int(posted)
        assert ("o/r#125" in mgr._rebase_suppressed) is not posted

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("cooldown", "posted"), [(8.0, False), (0.0, True)])
    async def test_live_head_decides_not_scanned_head(
        self, cooldown: float, posted: bool
    ) -> None:
        """Dependabot pushed after the scan: the live head has moved."""
        mgr, client = _manager(
            timedelta(minutes=2), timedelta(hours=1), rebase_cooldown=cooldown
        )
        live = "cafe" * 10
        dates = {
            _pr().head_sha: _iso(NOW - timedelta(hours=1)),
            live: _iso(NOW - timedelta(minutes=1)),
        }

        async def get(path: str, **_: Any) -> dict[str, Any]:
            if path == "/repos/o/r/pulls/125":
                return {"head": {"sha": live}}
            sha = path.rsplit("/", 1)[-1]
            return {"commit": {"committer": {"date": dates[sha]}}}

        client.get = AsyncMock(side_effect=get)

        assert await mgr._request_dependabot_rebase(_pr(), "o", "r") is posted

        assert client.post_issue_comment.await_count == int(posted)
        assert ("o/r#125" in mgr._rebase_suppressed) is not posted
        client.get.assert_any_await(f"/repos/o/r/commits/{live}")

    @pytest.mark.asyncio
    async def test_newest_request_on_later_page_decides(self) -> None:
        gh = GitHubAsync(token="fake-token")
        old = {
            "body": "@dependabot rebase",
            "created_at": _iso(NOW - timedelta(days=2)),
        }
        recent = {
            "body": "@dependabot rebase",
            "created_at": _iso(NOW - timedelta(hours=3)),
        }

        async def pages(path: str, **_: Any) -> AsyncIterator[Any]:
            yield [old] + [{"body": "noise", "created_at": _iso(NOW)}] * 99
            yield [recent]

        gh.get_paginated = pages  # type: ignore[method-assign]
        gh.get = AsyncMock(  # type: ignore[method-assign]
            return_value={"commit": {"committer": {"date": _iso(NOW)}}}
        )
        gh.post_issue_comment = AsyncMock()  # type: ignore[method-assign]
        mgr, _client = make_merge_manager()
        mgr._github_client = gh

        assert await mgr._request_dependabot_rebase(_pr(), "o", "r") is False

        gh.post_issue_comment.assert_not_called()
        assert "o/r#125" in mgr._rebase_suppressed

    @pytest.mark.asyncio
    async def test_incomplete_reading_posts(self) -> None:
        mgr, client = _cooling()
        client.list_issue_comments = AsyncMock(
            return_value=IssueComments(
                [{"body": "@dependabot rebase", "created_at": _iso(NOW)}],
                complete=False,
            )
        )

        assert await mgr._request_dependabot_rebase(_pr(), "o", "r") is True

        client.post_issue_comment.assert_awaited_once()


class TestCallersWhileCoolingDown:
    """Each of the five callers, with the request held back."""

    @pytest.mark.asyncio
    async def test_conflict_fails_fast_with_note(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mgr, client = _cooling()
        mgr._wait_for_auto_merge = AsyncMock()  # type: ignore[method-assign]
        pr = _pr("dirty")

        with caplog.at_level(logging.INFO):
            result = await mgr._handle_merge_conflict(
                pr, "o", "r", MergeResult(pr_info=pr, status=MergeStatus.PENDING)
            )

        assert result.status == MergeStatus.FAILED
        assert result.merge_refused is True
        assert result.error == f"merge conflicts; {NOTE}"
        mgr._wait_for_auto_merge.assert_not_called()
        assert "Requesting dependabot rebase" not in caplog.text
        _assert_nothing_done(mgr, client)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("armed", [True, False])
    async def test_conflict_no_wait_does_not_claim_a_rebase(
        self, armed: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        mgr, client = _cooling(max_wait=0)
        mgr._approve_pr = AsyncMock()  # type: ignore[method-assign]
        mgr._enable_auto_merge_for_pr = AsyncMock(return_value=armed)  # type: ignore[method-assign]
        pr = _pr("dirty")

        with caplog.at_level(logging.INFO):
            result = await mgr._handle_merge_conflict(
                pr, "o", "r", MergeResult(pr_info=pr, status=MergeStatus.PENDING)
            )

        expected = MergeStatus.AUTO_MERGE_PENDING if armed else MergeStatus.BLOCKED
        assert result.status == expected
        assert NOTE in (result.error or "")
        assert "rebase requested" not in (result.error or "")
        assert "Requesting dependabot rebase" not in caplog.text
        client.post_issue_comment.assert_not_called()
        mgr._record_rebase.assert_not_called()  # type: ignore[attr-defined]
        assert not mgr._rebase_requested

    @pytest.mark.asyncio
    async def test_step5_skips_fallback_and_wait(self) -> None:
        mgr, client = _cooling(merge_timeout=60)
        pr = _pr("behind")
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="o",
            repo_name="r",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
            needs_rebase=True,
        )
        local = AsyncMock()
        rest = AsyncMock()

        with (
            patch(
                "dependamerge.rebase.dispatch.should_use_local_rebase",
                AsyncMock(return_value=(True, "signatures required")),
            ),
            patch("dependamerge.rebase.dispatch.in_github_actions", return_value=True),
            patch("dependamerge.rebase.dispatch._run_local_path", local),
            patch("dependamerge.rebase.dispatch._run_rest_path", rest),
        ):
            terminal = await mgr._run_step5_rebase(flow)

        assert terminal is None
        assert "o/r#125" in mgr._rebase_suppressed
        local.assert_not_called()
        rest.assert_not_called()
        assert mgr._should_wait_for_checks(flow) is False
        _assert_nothing_done(mgr, client)
        # Without the suppression the same flow would wait for the rebase.
        mgr._rebase_suppressed.clear()
        assert mgr._should_wait_for_checks(flow) is True

    @pytest.mark.asyncio
    async def test_rejected_merge_reports_with_note(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mgr, client = _cooling(fix_out_of_date=True)
        pr = _pr("behind")

        with caplog.at_level(logging.INFO):
            retry = await mgr._handle_merge_failure(pr, "o", "r")

        assert retry is False
        assert "o/r#125" in mgr._rebase_suppressed
        assert "is behind -" not in caplog.text
        assert "Rebase cooling down" in caplog.text
        _assert_nothing_done(mgr, client)

        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        await mgr._report_merge_failure(
            pr, "o", "r", result, "Merge rejected: branch is out of date"
        )
        assert NOTE in (result.error or "")

    @pytest.mark.asyncio
    async def test_confirmed_live_blocker_keeps_note(self) -> None:
        mgr, client = _cooling()
        pr = _pr("behind")
        await mgr._request_dependabot_rebase(pr, "o", "r")
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        await mgr._report_merge_failure(
            pr, "o", "r", result, "Merge rejected: branch is out of date", refused=True
        )
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable_state": "blocked",
                "head": {"sha": pr.head_sha},
                "base": {"ref": "main"},
            }
        )
        mgr._live_blocking_conditions = AsyncMock(  # type: ignore[method-assign]
            return_value=(["required check: tests"], [], True)
        )

        confirmed = await mgr._confirm_failure(pr, result)

        assert confirmed.status == MergeStatus.FAILED
        assert confirmed.error == f"blocked by required check: tests; {NOTE}"

    @pytest.mark.asyncio
    async def test_stale_head_settles_without_refusal(self) -> None:
        mgr, client = _cooling()

        assert await mgr._recover_stale_head(_pr("blocked"), "o", "r") is True

        assert "o/r#125" in mgr._rebase_suppressed
        _assert_nothing_done(mgr, client)


class TestCallersWhileInFlight:
    """An in-flight request keeps today's behaviour: arm and wait."""

    @pytest.mark.asyncio
    async def test_stale_head_arms_auto_merge(self) -> None:
        mgr, client = _manager(timedelta(minutes=2), timedelta(hours=1))

        assert await mgr._recover_stale_head(_pr("blocked"), "o", "r") is True

        client.post_issue_comment.assert_not_called()
        assert "o/r#125" in mgr._rebase_requested
        mgr._enable_auto_merge_with_approval.assert_awaited_once()  # type: ignore[attr-defined]
        mgr._refuse_unsigned_rebase.assert_not_called()  # type: ignore[attr-defined]


class TestDependabotRebasingNotice:
    """Dependabot's own "rebasing" notice outranks an older macro's age."""

    REBASING = "bump\n\nDependabot is rebasing this PR"

    @pytest.mark.asyncio
    async def test_notice_is_under_way_not_suppressed(self) -> None:
        mgr, client = _cooling()
        pr = _pr()
        pr.body = self.REBASING

        assert await mgr._request_dependabot_rebase(pr, "o", "r") is True

        client.post_issue_comment.assert_not_called()
        mgr._record_rebase.assert_not_called()  # type: ignore[attr-defined]
        assert not mgr._rebase_suppressed

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rebasing_at_step5", [True, False])
    async def test_step5_then_recovery_reports_pending(
        self, rebasing_at_step5: bool
    ) -> None:
        """Step 5 leaves no suppression behind once dependabot is rebasing."""
        mgr, _client = _cooling(merge_timeout=60)
        pr = _pr("behind")
        if rebasing_at_step5:
            pr.body = self.REBASING
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="o",
            repo_name="r",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
            needs_rebase=True,
        )

        with (
            patch(
                "dependamerge.rebase.dispatch.should_use_local_rebase",
                AsyncMock(return_value=(True, "signatures required")),
            ),
            patch("dependamerge.rebase.dispatch.in_github_actions", return_value=True),
            patch("dependamerge.rebase.dispatch._run_local_path", AsyncMock()),
            patch("dependamerge.rebase.dispatch._run_rest_path", AsyncMock()),
        ):
            await mgr._run_step5_rebase(flow)
        assert ("o/r#125" in mgr._rebase_suppressed) is not rebasing_at_step5

        pr.body = self.REBASING
        assert await mgr._recover_stale_head(pr, "o", "r") is True
        mgr._auto_merge_enabled.add("o/r#125")

        assert not mgr._rebase_suppressed
        result = mgr._behind_auto_merge_result(flow)
        assert result is not None
        assert result.status == MergeStatus.AUTO_MERGE_PENDING
