# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for which failures a later reading may withdraw.

``MergeResult.merge_refused`` marks a failure as GitHub declining the
merge *on the pull request's state*.  Only such a failure may be
withdrawn when the confirmation step finds the PR mergeable, because a
run-side failure — an exception, a rebase that did not complete, a
missing token scope — says nothing about mergeability and would be
buried rather than corrected.

The conflict paths reach a state verdict and did not opt in, so
``lfreleng-actions/hw-bom-javascript#317`` was reported as::

    rebase cleared the conflict but the PR could not be merged
    (auto-merge unavailable)

and merged untouched on the next run over the same organisation.  It was
re-runnable when the run called it failed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dependamerge.merge_manager import AsyncMergeManager, MergeResult, MergeStatus
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/hw-bom-javascript"


def _pr(state: str = "dirty") -> PullRequestInfo:
    return PullRequestInfo(
        number=317,
        title="Chore: Bump @opentelemetry/sdk-node",
        body="bump",
        author="dependabot[bot]",
        head_sha="c0ffee11" * 5,
        base_branch="main",
        head_branch="dependabot/npm/x",
        state="open",
        mergeable=False,
        mergeable_state=state,
        behind_by=None,
        files_changed=[],
        repository_full_name=REPO,
        html_url=f"https://github.com/{REPO}/pull/317",
        reviews=[],
        review_comments=[],
    )


class TestAStateVerdictMayBeWithdrawn:
    """Both conflict paths judge the pull request, not the run."""

    def test_an_unresolved_conflict_is_a_refusal(self) -> None:
        mgr, _ = make_merge_manager()
        pr = _pr()
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)

        out = mgr._report_unresolved_conflict(pr, result)

        assert out.status is MergeStatus.FAILED
        assert out.merge_refused is True

    @pytest.mark.asyncio
    async def test_a_rebased_pr_that_would_not_merge_is_a_refusal(self) -> None:
        mgr, _ = make_merge_manager()
        pr = _pr(state="blocked")
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)

        out = await mgr._merge_rebased_pr(
            pr, "lfreleng-actions", "hw-bom-javascript", result
        )

        assert out.status is MergeStatus.FAILED
        assert out.merge_refused is True


class TestARunSideFailureMayNot:
    """The control, and the reason the flag is per path rather than global."""

    @pytest.mark.asyncio
    async def test_an_approval_exception_is_not_a_refusal(self) -> None:
        """Its message carries the exception; withdrawing would bury it."""
        mgr, _ = make_merge_manager()
        pr = _pr(state="clean")
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)

        out = mgr._report_rebase_approval_failure(
            pr,
            "lfreleng-actions",
            "hw-bom-javascript",
            result,
            RuntimeError("403 Forbidden"),
        )

        assert out.status is MergeStatus.FAILED
        assert out.merge_refused is False
        assert "403 Forbidden" in (out.error or "")


class TestAnAttemptedMergeIsClassifiedByWhatStoppedIt:
    """``_merge_pr_with_retry`` returns False for both kinds of failure.

    It exhausts its retries on a 502 and stops on a 422 the same way it
    returns False when GitHub answers ``merged: false``, so the ``clean``
    branch cannot read a verdict off the return value alone.  It asks
    ``_failure_summary_from_exception`` about the stored exception
    instead --- the same classifier ``_get_failure_summary`` uses, so the
    two cannot disagree about the same exception.
    """

    @staticmethod
    async def _attempt(mgr: AsyncMergeManager, pr: PullRequestInfo) -> MergeResult:
        mgr._merge_pr_with_retry = AsyncMock(return_value=False)  # type: ignore[method-assign]
        return await mgr._merge_rebased_pr(
            pr,
            "lfreleng-actions",
            "hw-bom-javascript",
            MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )

    @pytest.mark.asyncio
    async def test_no_exception_means_the_api_answered(self) -> None:
        """Nothing raised, so the False is GitHub's ``merged: false``."""
        mgr, _ = make_merge_manager()

        out = await self._attempt(mgr, _pr(state="clean"))

        assert out.status is MergeStatus.FAILED
        assert out.merge_refused is True

    @pytest.mark.asyncio
    async def test_a_transport_failure_keeps_its_message(self) -> None:
        """A 502 says nothing about mergeability, so it is not withdrawable."""
        mgr, _ = make_merge_manager()
        pr = _pr(state="clean")
        mgr._last_merge_exception[f"{REPO}#317"] = RuntimeError(
            "Server error '502 Bad Gateway' for url 'https://api.github.com/x'"
        )

        out = await self._attempt(mgr, pr)

        assert out.status is MergeStatus.FAILED
        assert out.merge_refused is False

    @pytest.mark.asyncio
    async def test_a_405_carrying_githubs_own_words_is_a_verdict(self) -> None:
        """A ruleset rejection arrives as a 405 with a body, and does expire."""
        mgr, _ = make_merge_manager()
        pr = _pr(state="clean")
        mgr._last_merge_exception[f"{REPO}#317"] = RuntimeError(
            "Client error '405 Method Not Allowed' for url "
            "'https://api.github.com/x' - GitHub: Required status check "
            '"Testing \U0001f9ea" is expected.'
        )

        out = await self._attempt(mgr, pr)

        assert out.status is MergeStatus.FAILED
        assert out.merge_refused is True

    @pytest.mark.asyncio
    async def test_a_bodyless_405_on_a_clean_pr_is_transient(self) -> None:
        """The case a bare status allowlist would get wrong.

        ``_failure_summary_from_exception`` singles this out: a 405 with
        no explanation, on a PR that reads mergeable, is GitHub having
        trouble rather than GitHub judging the PR.  This helper runs on
        exactly that state, so classifying by status alone would withdraw
        the one 405 the codebase says is not a verdict.
        """
        mgr, _ = make_merge_manager()
        pr = _pr(state="clean")
        mgr._last_merge_exception[f"{REPO}#317"] = RuntimeError(
            "Client error '405 Method Not Allowed' for url 'https://api.github.com/x'"
        )

        out = await self._attempt(mgr, pr)

        assert out.status is MergeStatus.FAILED
        assert out.merge_refused is False


class TestTheClearedConflictIsReportedAsReRunnable:
    """End to end: the shape #317 was in when it was called failed."""

    @pytest.mark.asyncio
    async def test_a_conflict_that_cleared_becomes_unsettled(self) -> None:
        mgr, client = make_merge_manager()
        pr = _pr()
        result = mgr._report_unresolved_conflict(
            pr, MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        )
        # By report time the rebase has landed and the PR is mergeable.
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "clean",
            }
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status is MergeStatus.UNSETTLED
        assert out.warning is not None
        assert "merge conflicts" in out.warning

    @pytest.mark.asyncio
    async def test_a_conflict_that_persists_is_still_a_failure(self) -> None:
        """The control: only a cleared conflict is withdrawn."""
        mgr, client = make_merge_manager()
        pr = _pr()
        result = mgr._report_unresolved_conflict(
            pr, MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        )
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": False,
                "mergeable_state": "dirty",
            }
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status is MergeStatus.FAILED
        assert out.error == "merge conflicts"


class TestAConflictThatClearedIntoABlockNamesTheBlocker:
    """The path this change newly reaches, and #497 newly words.

    ``_settle_stale_failure`` returns at its first line for a failure
    that is not a refusal, so before this change a conflict never
    reached ``_name_the_live_blocker`` at all.  Marking the conflict
    paths opens that route: a PR whose conflict rebased away into a
    checks-pending state now gets the live reading rather than a stale
    ``merge conflicts``.

    Asserted here because the two changes were written independently ---
    this is the seam between them, and a seam nobody drives is where a
    regression goes unnoticed.
    """

    @pytest.mark.asyncio
    async def test_the_live_blocker_replaces_the_stale_conflict(self) -> None:
        mgr, client = make_merge_manager()
        pr = _pr()
        result = mgr._report_unresolved_conflict(
            pr, MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        )
        # The rebase landed, so the conflict is gone --- but a required
        # status check is failing, which is what holds it now.  This is
        # the pre-commit.ci shape the owner-wide run kept reporting.
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "blocked",
            }
        )
        client.get_required_status_checks_reliable = AsyncMock(
            return_value=([{"context": "pre-commit.ci - pr"}], True)
        )
        client.get_failing_status_contexts = AsyncMock(
            return_value=["pre-commit.ci - pr"]
        )
        client.get_check_runs_for_ref = AsyncMock(return_value=[])
        # Mocked explicitly rather than left to the bare AsyncMock: its
        # default return iterates as empty, so an unmocked probe would
        # pass for a reason unrelated to what this asserts.
        client.get_failing_workflow_run_names_for_sha = AsyncMock(return_value=[])

        out = await mgr._confirm_failure(pr, result)

        assert out.status is MergeStatus.FAILED
        assert "pre-commit.ci - pr" in (out.error or "")
        assert "merge conflicts" not in (out.error or "")
        # The rejection survives as the note, never as the cause.
        assert "merge conflicts" in (out.warning or "")
