# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for classifying the merge of a recreated pull request.

``_dispatch_recreated_merge`` returned a bare ``bool``, which reported
an uninitialised client, a missing token scope and a 502 with the same
``False`` GitHub uses to decline a merge.  Its caller therefore could
not mark the failure as a refusal: doing so would have identified a
permission error as a mergeability verdict and then hidden it when a
later reading found the replacement clean.

So a replacement that GitHub genuinely refused, and that has since gone
clean, reported ``BLOCKED`` --- "needs a human" --- where the rest of
the pipeline would say ``UNSETTLED`` --- "re-run to merge it".

The ``BLOCKED``-not-``FAILED`` choice for a replacement that genuinely
will not merge is preserved, and is asserted here rather than left to
the reader: ``_merge_recreated_pr`` returns early when the confirmation
corrected the outcome, so a refusal that cleared never reaches the
``BLOCKED`` line, and one that did not still does.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dependamerge.github_async import PermissionError as GitHubPermissionError
from dependamerge.merge_manager import AsyncMergeManager, MergeStatus
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

OWNER = "lfreleng-actions"
REPO = "test-python-project"
FULL = f"{OWNER}/{REPO}"


def _pr(number: int = 107) -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="Chore: Bump step-security/harden-runner",
        body="bump",
        author="dependabot[bot]",
        head_sha="feedface" * 5,
        base_branch="main",
        head_branch="dependabot/github_actions/harden-runner",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=None,
        files_changed=[],
        repository_full_name=FULL,
        html_url=f"https://github.com/{FULL}/pull/{number}",
        reviews=[],
        review_comments=[],
    )


def _mgr() -> tuple[AsyncMergeManager, AsyncMock]:
    mgr, client = make_merge_manager()
    return mgr, client


class TestTheDispatchSaysWhichKindOfFailure:
    """The distinction the bare boolean could not carry."""

    @pytest.mark.asyncio
    async def test_an_api_answer_is_a_refusal(self) -> None:
        """Nothing raised, so GitHub judged the replacement itself."""
        mgr, client = _mgr()
        client.merge_pull_request = AsyncMock(return_value=False)

        merged, refused = await mgr._dispatch_recreated_merge(OWNER, REPO, _pr())

        assert merged is False
        assert refused is True

    @pytest.mark.asyncio
    async def test_a_successful_merge_refuses_nothing(self) -> None:
        """A merge that landed has no refusal to report."""
        mgr, client = _mgr()
        client.merge_pull_request = AsyncMock(return_value=True)

        merged, refused = await mgr._dispatch_recreated_merge(OWNER, REPO, _pr())

        assert merged is True
        assert refused is False

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_not_a_refusal(self) -> None:
        """A 502 says nothing about whether the replacement would merge."""
        mgr, client = _mgr()
        client.merge_pull_request = AsyncMock(
            side_effect=RuntimeError(
                "Server error '502 Bad Gateway' for url 'https://api.github.com/x'"
            )
        )

        merged, refused = await mgr._dispatch_recreated_merge(OWNER, REPO, _pr())

        assert merged is False
        assert refused is False

    @pytest.mark.asyncio
    async def test_a_permission_error_is_not_a_refusal(self) -> None:
        """The case that kept the whole path unmarked.

        Withdrawing this would identify a missing token scope as a
        mergeability verdict and then hide it behind a clean reading,
        which is worse than reporting it as needing a human.
        """
        mgr, client = _mgr()
        client.merge_pull_request = AsyncMock(
            side_effect=GitHubPermissionError(
                operation="merge_workflow",
                message="Missing 'workflow' scope",
            )
        )

        merged, refused = await mgr._dispatch_recreated_merge(OWNER, REPO, _pr())

        assert merged is False
        assert refused is False

    @pytest.mark.asyncio
    async def test_an_uninitialised_client_is_not_a_refusal(self) -> None:
        """It carries no status at all, so nothing says the PR was judged."""
        mgr, _ = _mgr()
        mgr._github_client = None

        merged, refused = await mgr._dispatch_recreated_merge(OWNER, REPO, _pr())

        assert merged is False
        assert refused is False

    @pytest.mark.asyncio
    async def test_a_ruleset_rejection_is_a_refusal(self) -> None:
        """405 carrying GitHub's own words is a reading of the PR."""
        mgr, client = _mgr()
        client.merge_pull_request = AsyncMock(
            side_effect=RuntimeError(
                "Client error '405 Method Not Allowed' for url "
                "'https://api.github.com/x' - GitHub: Repository rule "
                "violations found"
            )
        )

        merged, refused = await mgr._dispatch_recreated_merge(OWNER, REPO, _pr())

        assert merged is False
        assert refused is True


class TestTheReplacementOutcomeFollowsTheClassification:
    """End to end, through ``_merge_recreated_pr``.

    The early return after ``_confirm_failure`` already distinguishes a
    corrected outcome from a standing one, so marking the refusal is all
    that was needed --- the ``BLOCKED`` line is not reached when the
    confirmation withdrew the failure, and is reached when it did not.
    """

    @staticmethod
    def _flow(mgr: AsyncMergeManager):  # type: ignore[no-untyped-def]
        from dependamerge.merge_manager._single_pr_context import _MergeFlow
        from dependamerge.merge_manager._types import MergeResult

        original = _pr(number=106)
        return _MergeFlow(
            pr_info=original,
            result=MergeResult(pr_info=original, status=MergeStatus.PENDING),
            repo_owner=OWNER,
            repo_name=REPO,
        )

    @pytest.mark.asyncio
    async def test_a_refused_replacement_that_cleared_is_re_runnable(self) -> None:
        """The outcome #496 exists to reach."""
        mgr, client = _mgr()
        replacement = _pr()
        mgr._approve_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
        mgr._dispatch_recreated_merge = AsyncMock(return_value=(False, True))  # type: ignore[method-assign]
        # By confirmation time the replacement reads mergeable.
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )

        flow = self._flow(mgr)
        await mgr._merge_recreated_pr(flow, replacement)

        assert flow.result.status is MergeStatus.UNSETTLED

    @pytest.mark.asyncio
    async def test_a_run_side_failure_on_a_clean_replacement_stays_blocked(
        self,
    ) -> None:
        """The control, and the reason the flag is classified not assumed.

        Identical payload to the test above --- the replacement reads
        ``clean`` either way.  Only the kind of failure differs, and a
        permission error must keep its message rather than be rewritten
        into advice to re-run.
        """
        mgr, client = _mgr()
        replacement = _pr()
        mgr._approve_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
        mgr._dispatch_recreated_merge = AsyncMock(return_value=(False, False))  # type: ignore[method-assign]
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )

        flow = self._flow(mgr)
        await mgr._merge_recreated_pr(flow, replacement)

        assert flow.result.status is MergeStatus.BLOCKED
        assert "merge still failed" in (flow.result.error or "")
