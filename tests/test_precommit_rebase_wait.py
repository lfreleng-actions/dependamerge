# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Step 5.5 waits for an asynchronous rebase this run requested.

The block reason is read before Step 5.  When that reason is a failing
check on the old head, it will not clear on its own, so the wait used
to be skipped and the merge was refused moments after the rebase was
asked for.  The rebase is what clears it, so the wait now bridges it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dependamerge.merge_manager import AsyncMergeManager, MergeResult, MergeStatus
from dependamerge.merge_manager._head_pin import merge_head
from dependamerge.merge_manager._single_pr_context import _MergeFlow
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

OLD_HEAD = "0123456789abcdef0123456789abcdef01234567"
NEW_HEAD = "fedcba9876543210fedcba9876543210fedcba98"
FAILING = "required status check: pre-commit.ci - pr failed"


def _pr() -> PullRequestInfo:
    return PullRequestInfo(
        number=11,
        title="Chore: Bump foo from 1 to 2",
        body="bump",
        author="dependabot[bot]",
        head_sha=OLD_HEAD,
        base_branch="main",
        head_branch="dependabot/pip/foo-2",
        state="open",
        mergeable=False,
        mergeable_state="blocked",
        behind_by=3,
        files_changed=[],
        repository_full_name="o/r",
        html_url="https://github.com/o/r/pull/11",
    )


def _flow(pr: PullRequestInfo) -> _MergeFlow:
    """Blocked on a failing check, behind, a rebase requested, not armed."""
    return _MergeFlow(
        pr_info=pr,
        repo_owner="o",
        repo_name="r",
        result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        blocked_reason=FAILING,
        blocked_analysis_ok=True,
        needs_rebase=True,
    )


def _manager(**overrides: object) -> tuple[AsyncMergeManager, AsyncMock]:
    mgr, client = make_merge_manager(preview_mode=False, **overrides)
    mgr._no_wait = False
    return mgr, client


class TestTheWaitDecision:
    @pytest.mark.xfail(strict=True, reason="Step 5.5 skips a requested rebase")
    def test_a_requested_rebase_is_waited_for(self) -> None:
        mgr, _ = _manager()

        assert mgr._should_wait_for_checks(_flow(_pr())) is True

    def test_without_a_rebase_the_failing_check_still_skips_the_wait(self) -> None:
        mgr, _ = _manager()
        flow = _flow(_pr())
        flow.needs_rebase = False

        assert mgr._should_wait_for_checks(flow) is False

    def test_a_rebase_the_cooldown_held_back_is_not_waited_for(self) -> None:
        mgr, _ = _manager()
        flow = _flow(_pr())
        mgr._rebase_suppressed[flow.pr_key] = "rebase requested 10m ago"

        assert mgr._should_wait_for_checks(flow) is False

    def test_max_wait_zero_does_not_wait(self) -> None:
        mgr, _ = _manager()
        mgr._no_wait = True

        assert mgr._should_wait_for_checks(_flow(_pr())) is False

    def test_a_rebase_step5_already_waited_for_is_not_waited_again(self) -> None:
        mgr, _ = _manager()
        flow = _flow(_pr())
        flow.already_rebased = True

        assert mgr._should_wait_for_checks(flow) is False


class TestTheMergeAfterTheWait:
    @pytest.mark.xfail(strict=True, reason="Step 5.5 skips a requested rebase")
    @pytest.mark.asyncio
    async def test_the_merge_names_the_rebased_head(self) -> None:
        mgr, client = _manager(merge_timeout=60)
        pr = _pr()
        flow = _flow(pr)
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "head": {"sha": NEW_HEAD},
            }
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            assert mgr._should_wait_for_checks(flow) is True
            assert await mgr._await_auto_merge(flow) is None

        assert pr.mergeable_state == "clean"
        assert await merge_head(mgr, pr, "o", "r") == NEW_HEAD
