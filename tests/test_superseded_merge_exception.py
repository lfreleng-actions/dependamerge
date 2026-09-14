# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for an exception outliving the attempt that raised it.

``_handle_merge_exception`` stores whatever it caught and nothing clears
it, so a retry sequence of ``502`` then ``merged: false`` leaves the 502
in ``_last_merge_exception`` even though the attempt that decided the
outcome was a clean refusal by GitHub.

The field serves two different questions, which is why this is fixed by
recording the last event rather than by clearing:

* *Why did this merge fail?* --- ``_get_failure_summary`` and
  ``_rebased_merge_was_refused`` want the **terminal** attempt.  A
  superseded exception makes them name a transport error as the cause of
  a refusal, and withhold a withdrawal the refusal has earned.
* *Did GitHub ever say this?* --- the approval and pending-workflow
  recoveries want the **history**.  Clearing the exception would disable
  them, which is the regression this arrangement exists to avoid.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dependamerge.merge_manager import AsyncMergeManager, MergeResult, MergeStatus
from dependamerge.merge_manager._retry import _RetryDecision
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

OWNER = "lfreleng-actions"
REPO = "python-workflows"
FULL = f"{OWNER}/{REPO}"
KEY = f"{FULL}#84"

_BAD_GATEWAY = "Server error '502 Bad Gateway' for url 'https://api.github.com/x'"


def _pr(state: str = "blocked") -> PullRequestInfo:
    return PullRequestInfo(
        number=84,
        title="Chore: pre-commit autoupdate",
        body="bump",
        author="pre-commit-ci[bot]",
        head_sha="d00dfeed" * 5,
        base_branch="main",
        head_branch="pre-commit-ci-update-config",
        state="open",
        mergeable=True,
        mergeable_state=state,
        behind_by=None,
        files_changed=[],
        repository_full_name=FULL,
        html_url=f"https://github.com/{FULL}/pull/84",
        reviews=[],
        review_comments=[],
    )


class TestTheLoopRecordsWhichCameLast:
    """The signal itself, driven through the real retry loop.

    Asserted against ``_merge_pr_with_retry`` rather than set by hand,
    so the tests below rest on what the loop actually records.
    """

    @staticmethod
    def _manager() -> tuple[AsyncMergeManager, AsyncMock]:
        mgr, client = make_merge_manager()
        mgr._recheck_pr_before_retry = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._retry_after_false_merge = AsyncMock(  # type: ignore[method-assign]
            return_value=_RetryDecision.STOP
        )
        return mgr, client

    @pytest.mark.asyncio
    async def test_an_answer_after_an_exception_is_recorded(self) -> None:
        """The sequence in the report: 502, then GitHub answers."""
        mgr, _ = self._manager()
        pr = _pr()
        mgr._dispatch_merge = AsyncMock(  # type: ignore[method-assign]
            side_effect=[RuntimeError(_BAD_GATEWAY), False]
        )

        merged = await mgr._merge_pr_with_retry(pr, OWNER, REPO)

        assert merged is False
        assert mgr._last_merge_was_answered[KEY] is True
        # The exception is kept, not cleared --- the recoveries read it.
        assert str(mgr._last_merge_exception[KEY]) == _BAD_GATEWAY

    @pytest.mark.asyncio
    async def test_an_exception_after_an_answer_is_recorded(self) -> None:
        """The reverse order, so the flag tracks rather than latches."""
        mgr, _ = self._manager()
        pr = _pr()
        mgr._retry_after_false_merge = AsyncMock(  # type: ignore[method-assign]
            return_value=_RetryDecision.RETRY
        )
        mgr._dispatch_merge = AsyncMock(  # type: ignore[method-assign]
            side_effect=[False, RuntimeError(_BAD_GATEWAY)]
        )

        await mgr._merge_pr_with_retry(pr, OWNER, REPO)

        assert mgr._last_merge_was_answered[KEY] is False

    @pytest.mark.asyncio
    async def test_an_exception_alone_records_no_answer(self) -> None:
        """The control: nothing answered, so nothing supersedes."""
        mgr, _ = self._manager()
        pr = _pr()
        mgr._dispatch_merge = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError(_BAD_GATEWAY)
        )

        await mgr._merge_pr_with_retry(pr, OWNER, REPO)

        assert mgr._last_merge_was_answered.get(KEY) is False


class TestTheDiagnosisReadsTheTerminalAttempt:
    """``_get_failure_summary`` is the reader with most to lose."""

    @pytest.mark.asyncio
    async def test_a_superseded_exception_does_not_name_the_cause(self) -> None:
        """The 502 did not cause this failure, so it must not explain it."""
        mgr, _ = make_merge_manager()
        pr = _pr(state="dirty")
        mgr._last_merge_exception[KEY] = RuntimeError(_BAD_GATEWAY)
        mgr._last_merge_was_answered[KEY] = True

        reason, refused = await mgr._get_failure_summary(pr)

        assert reason == "merge conflicts"
        assert "502" not in reason
        # GitHub answered on the PR's state, so the failure may be
        # withdrawn when a later reading finds it mergeable.
        assert refused is True

    @pytest.mark.asyncio
    async def test_an_unsuperseded_exception_still_explains(self) -> None:
        """The control: with no answer after it, the 502 is the cause."""
        mgr, _ = make_merge_manager()
        pr = _pr(state="dirty")
        mgr._last_merge_exception[KEY] = RuntimeError(_BAD_GATEWAY)

        reason, refused = await mgr._get_failure_summary(pr)

        assert "502" in reason
        assert refused is False


class TestTheConflictPathReadsItToo:
    """``_rebased_merge_was_refused``, added for the conflict paths."""

    @pytest.mark.asyncio
    async def test_a_superseded_502_no_longer_withholds_the_withdrawal(self) -> None:
        mgr, _ = make_merge_manager()
        pr = _pr(state="clean")
        mgr._merge_pr_with_retry = AsyncMock(return_value=False)  # type: ignore[method-assign]
        mgr._last_merge_exception[KEY] = RuntimeError(_BAD_GATEWAY)
        mgr._last_merge_was_answered[KEY] = True

        out = await mgr._merge_rebased_pr(
            pr, OWNER, REPO, MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        )

        assert out.status is MergeStatus.FAILED
        assert out.merge_refused is True

    @pytest.mark.asyncio
    async def test_an_unsuperseded_502_still_stands(self) -> None:
        """The control, and the behaviour #503 introduced."""
        mgr, _ = make_merge_manager()
        pr = _pr(state="clean")
        mgr._merge_pr_with_retry = AsyncMock(return_value=False)  # type: ignore[method-assign]
        mgr._last_merge_exception[KEY] = RuntimeError(_BAD_GATEWAY)

        out = await mgr._merge_rebased_pr(
            pr, OWNER, REPO, MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        )

        assert out.merge_refused is False


class TestTheDirectMergeInRecoveryKeepsTheSignal:
    """``_wait_for_required_workflows_and_retry`` is a second writer.

    It deliberately dispatches a *single* merge call rather than going
    through ``_merge_pr_with_retry``, whose retry ladder would multiply
    the attempts --- so it maintains ``_last_merge_exception`` itself,
    and has to maintain the ordering signal alongside it.

    Both directions matter, and the second is the damaging one: a stale
    ``True`` makes the diagnosis discard the *freshest* rejection it
    has, which is worse than the staleness this change exists to fix.
    """

    @staticmethod
    def _manager() -> tuple[AsyncMergeManager, AsyncMock]:
        mgr, client = make_merge_manager()
        # One cycle, then the shared deadline ends the loop.
        mgr._merge_timeout = 0.0
        mgr._repo_scoped = False
        client.get = AsyncMock(return_value={})
        mgr._wait_for_auto_merge = AsyncMock(return_value=(False, False))  # type: ignore[method-assign]
        mgr._stop_for_undispatched_workflows = AsyncMock(return_value=False)  # type: ignore[method-assign]
        return mgr, client

    @pytest.mark.asyncio
    async def test_a_direct_answer_supersedes_an_earlier_exception(self) -> None:
        mgr, client = self._manager()
        pr = _pr()
        mgr._last_merge_exception[KEY] = RuntimeError(_BAD_GATEWAY)
        mgr._last_merge_was_answered[KEY] = False
        client.merge_pull_request = AsyncMock(return_value=False)

        merged = await mgr._wait_for_required_workflows_and_retry(pr, OWNER, REPO)

        assert merged is False
        assert mgr._last_merge_was_answered[KEY] is True

    @pytest.mark.asyncio
    async def test_a_direct_exception_supersedes_an_earlier_answer(self) -> None:
        """The freshest rejection must not be discarded as superseded."""
        mgr, client = self._manager()
        pr = _pr()
        mgr._last_merge_was_answered[KEY] = True
        rejection = RuntimeError(
            "Client error '405 Method Not Allowed' for url "
            "'https://api.github.com/x' - GitHub: Repository rule violations found"
        )
        client.merge_pull_request = AsyncMock(side_effect=rejection)

        await mgr._wait_for_required_workflows_and_retry(pr, OWNER, REPO)

        assert mgr._last_merge_was_answered[KEY] is False
        # And the diagnosis now reads it rather than stepping over it.
        reason, _refused = await mgr._get_failure_summary(pr)
        assert "Repository rule violations found" in reason


class TestTheRecoveriesKeepTheirHistory:
    """The regression a clearing fix would have caused.

    ``_approve_and_retry_if_review_required`` asks whether GitHub named a
    missing approval *at any point* this run, and acts on it.  The answer
    is evidence about the repository's rules, not about one attempt, so
    an answer arriving afterwards does not retract it --- and clearing
    the exception would have stranded exactly the pull requests that
    recovery exists to save.
    """

    @pytest.mark.asyncio
    async def test_a_superseded_approval_rejection_still_recovers(self) -> None:
        mgr, _ = make_merge_manager()
        pr = _pr()
        mgr._last_merge_exception[KEY] = RuntimeError(
            "Repository rule violations found "
            "Waiting on required approvals from reviewers"
        )
        # An answer arrived after it, which supersedes the *diagnosis*
        # but must not retract the approval finding.
        mgr._last_merge_was_answered[KEY] = True
        mgr._ensure_pr_approved = AsyncMock(return_value=True)  # type: ignore[method-assign]
        mgr._retry_merge_under_dispatch_lock = AsyncMock(return_value=True)  # type: ignore[method-assign]

        recovered = await mgr._approve_and_retry_if_review_required(pr, OWNER, REPO)

        assert recovered is True
        mgr._ensure_pr_approved.assert_awaited_once()
