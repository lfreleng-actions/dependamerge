# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for reactive behind-PR recovery after a rejected merge.

Step 5 only rebases proactively when branch protection demands
up-to-date heads.  When that probe misses (or the base moved after
it ran) and GitHub rejects the merge with the PR ``behind``, the
recovery is reactive:

- ``_handle_merge_failure``: dependabot PRs get the ``@dependabot
  rebase`` macro (signed rebase, no immediate retry) with auto-merge
  armed; other authors keep the REST ``update-branch`` + retry path.
- ``_merge_single_pr``'s not-merged classification: a behind PR left
  with auto-merge armed reports ``AUTO_MERGE_PENDING`` instead of a
  spurious failure.
"""

from unittest.mock import AsyncMock, patch

import pytest

from dependamerge.github2gerrit_detector import GitHub2GerritDetectionResult
from dependamerge.merge_manager import MergeStatus
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

_BEHIND_PR = PullRequestInfo(
    number=88,
    node_id="PR_kwDOTestNode88",
    title="Chore: Bump zizmor from 1.4.0 to 1.5.0",
    body="Dependabot PR",
    author="dependabot[bot]",
    head_sha="abc123",
    base_branch="main",
    head_branch="dependabot/pip/zizmor-1.5.0",
    state="open",
    mergeable=True,
    mergeable_state="behind",
    behind_by=1,
    files_changed=[],
    repository_full_name="owner/repo",
    html_url="https://github.com/owner/repo/pull/88",
    reviews=[],
    review_comments=[],
)

#: GitHub's actual REST answer for a stale head, as the merge layer
#: wraps it: no "behind" anywhere in the message.
_OUT_OF_DATE_405 = (
    "Failed to merge PR #88 in owner/repo. Error: Client error '405 Method "
    "Not Allowed'. GitHub: Head branch is out of date. Review and try the "
    "merge again. (PR state: open, mergeable: True, mergeable_state: "
    "unknown)"
)


class TestHandleMergeFailureBehind:
    """Reactive branch-refresh strategy after a rejected merge."""

    @pytest.mark.asyncio
    async def test_dependabot_behind_uses_macro_not_update_branch(self) -> None:
        """Dependabot PR → macro + auto-merge armed, no REST update."""
        mgr, client = make_merge_manager(fix_out_of_date=True)
        client.get = AsyncMock(return_value=[])  # no existing comments
        client.post_issue_comment = AsyncMock()
        client.update_branch = AsyncMock()
        arm = AsyncMock(return_value=True)

        pr = _BEHIND_PR.model_copy()
        with patch.object(mgr, "_enable_auto_merge_with_approval", arm):
            should_retry = await mgr._handle_merge_failure(pr, "owner", "repo")

        # No immediate retry: dependabot rebases asynchronously.
        assert should_retry is False
        client.post_issue_comment.assert_awaited_once_with(
            "owner", "repo", 88, "@dependabot rebase"
        )
        client.update_branch.assert_not_awaited()
        arm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dependabot_self_rebase_not_duplicated(self) -> None:
        """A dependabot self-rebase in progress → no duplicate macro."""
        mgr, client = make_merge_manager(fix_out_of_date=True)
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock()
        client.update_branch = AsyncMock()
        arm = AsyncMock(return_value=True)

        pr = _BEHIND_PR.model_copy(update={"body": "Dependabot is rebasing this PR"})
        with patch.object(mgr, "_enable_auto_merge_with_approval", arm):
            should_retry = await mgr._handle_merge_failure(pr, "owner", "repo")

        assert should_retry is False
        client.post_issue_comment.assert_not_awaited()
        client.update_branch.assert_not_awaited()
        arm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_dependabot_behind_uses_update_branch(self) -> None:
        """Non-dependabot bots keep the REST update-branch + retry path."""
        mgr, client = make_merge_manager(fix_out_of_date=True)
        client.update_branch = AsyncMock()

        pr = _BEHIND_PR.model_copy(update={"author": "pre-commit-ci[bot]"})
        should_retry = await mgr._handle_merge_failure(pr, "owner", "repo")

        assert should_retry is True
        client.update_branch.assert_awaited_once_with("owner", "repo", 88)

    @pytest.mark.asyncio
    async def test_signed_rebase_is_refused_under_actions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under Actions, a PR needing the signed path is not REST-rebased."""
        from dependamerge import rebase

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(fix_out_of_date=True, rebase_local=True)
        client.update_branch = AsyncMock()

        # pre-commit-ci always needs the local path (no recreate macro).
        pr = _BEHIND_PR.model_copy(update={"author": "pre-commit-ci[bot]"})
        should_retry = await mgr._handle_merge_failure(pr, "owner", "repo")

        assert should_retry is False
        client.update_branch.assert_not_awaited()
        # Tool policy, not GitHub's verdict: never withdrawable.
        reason, refused = await mgr._get_failure_summary(pr)
        assert reason == rebase.LOCAL_REBASE_UNAVAILABLE
        assert refused is False

    @pytest.mark.asyncio
    async def test_the_refusal_survives_reporting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stuck check cannot replace the manual-rebase guidance."""
        from dependamerge import rebase
        from dependamerge.merge_manager import MergeResult, MergeStatus

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.update_branch = AsyncMock()
        pr = _BEHIND_PR.model_copy(update={"author": "pre-commit-ci[bot]"})
        await mgr._handle_merge_failure(pr, "owner", "repo")
        reason, refused = await mgr._get_failure_summary(pr)

        stuck = AsyncMock(return_value=(True, "pre-commit.ci - pr"))
        with patch.object(mgr, "_detect_stuck_required_check", stuck):
            result = await mgr._report_merge_failure(
                pr,
                "owner",
                "repo",
                MergeResult(pr_info=pr, status=MergeStatus.PENDING),
                reason,
                refused,
            )

        stuck.assert_not_awaited()
        assert result.status == MergeStatus.FAILED
        assert result.error == rebase.LOCAL_REBASE_UNAVAILABLE
        assert result.merge_refused is False

    @pytest.mark.asyncio
    async def test_a_failed_dependabot_macro_records_the_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No macro landed and no local path under Actions: say so."""
        from dependamerge import rebase

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(fix_out_of_date=True, rebase_local=True)
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock(side_effect=RuntimeError("403"))
        client.update_branch = AsyncMock()
        client.requires_commit_signatures = AsyncMock(return_value=True)
        client.check_pr_commit_signatures = AsyncMock(return_value=(True, []))

        pr = _BEHIND_PR.model_copy()  # dependabot
        should_retry = await mgr._handle_merge_failure(pr, "owner", "repo")

        assert should_retry is False
        client.update_branch.assert_not_awaited()
        reason, _ = await mgr._get_failure_summary(pr)
        assert reason == rebase.LOCAL_REBASE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_unsigned_base_still_rebases_under_actions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No signature at stake: REST update-branch is safe in Actions too."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(fix_out_of_date=True, rebase_local=True)
        client.update_branch = AsyncMock()
        client.requires_commit_signatures = AsyncMock(return_value=False)

        pr = _BEHIND_PR.model_copy(update={"author": "renovate[bot]"})
        should_retry = await mgr._handle_merge_failure(pr, "owner", "repo")

        assert should_retry is True
        client.update_branch.assert_awaited_once_with("owner", "repo", 88)

    @pytest.mark.asyncio
    async def test_no_fix_disables_recovery(self) -> None:
        """--no-fix → no macro, no update-branch, no retry."""
        mgr, client = make_merge_manager(fix_out_of_date=False)
        client.post_issue_comment = AsyncMock()
        client.update_branch = AsyncMock()

        should_retry = await mgr._handle_merge_failure(
            _BEHIND_PR.model_copy(), "owner", "repo"
        )

        assert should_retry is False
        client.post_issue_comment.assert_not_awaited()
        client.update_branch.assert_not_awaited()


class TestBehindAutoMergePendingClassification:
    """Behind + auto-merge armed after a failed merge → AUTO_MERGE_PENDING."""

    @pytest.mark.asyncio
    async def test_failed_merge_with_armed_auto_merge_reports_pending(
        self,
    ) -> None:
        mgr, client = make_merge_manager(
            preview_mode=False, fix_out_of_date=True, merge_timeout=0.1
        )
        pr = _BEHIND_PR.model_copy()

        # Behind + non-strict → Step 5 skips the rebase and Step 5.5
        # skips the wait; the direct merge attempt fails, and the
        # reactive recovery (exercised separately above) armed
        # auto-merge for this PR.
        client.requires_strict_status_checks = AsyncMock(return_value=False)
        client.get = AsyncMock(
            return_value={
                "mergeable": True,
                "mergeable_state": "behind",
                "state": "open",
                "merged": False,
            }
        )
        client.get_required_status_checks = AsyncMock(return_value=[])

        # The reactive recovery arms auto-merge *during* the failed
        # merge attempt (inside ``_merge_pr_with_retry`` →
        # ``_handle_merge_failure``), so mimic that here: the key is
        # NOT armed when the Step 6 skip gate runs, only afterwards.
        async def _fail_and_arm(
            pr_info: PullRequestInfo, owner: str, repo: str
        ) -> bool:
            mgr._auto_merge_enabled.add("owner/repo#88")
            return False

        no_g2g = GitHub2GerritDetectionResult()
        with (
            patch.object(
                mgr,
                "_detect_github2gerrit",
                new_callable=AsyncMock,
                return_value=no_g2g,
            ),
            patch.object(
                mgr,
                "_get_merge_method_for_repo",
                new_callable=AsyncMock,
                return_value="merge",
            ),
            patch.object(
                mgr,
                "_trigger_stale_precommit_ci",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_check_merge_requirements",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                mgr,
                "_approve_pr",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                mgr,
                "_approve_and_retry_if_review_required",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_merge_pr_with_retry",
                side_effect=_fail_and_arm,
            ),
        ):
            result = await mgr._merge_single_pr(pr)

        assert result.status == MergeStatus.AUTO_MERGE_PENDING
        assert result.error is not None
        assert "behind" in result.error


class TestRefusalBeatsPendingAutoMerge:
    """Auto-merge armed earlier cannot finish a rebase that was refused."""

    @pytest.mark.asyncio
    async def test_refused_pr_is_reported_failed_not_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dependamerge import rebase
        from dependamerge.merge_manager import MergeResult
        from dependamerge.merge_manager._single_pr_context import _MergeFlow
        from dependamerge.merge_manager._types import RecreateResult

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.update_branch = AsyncMock()
        pr = _BEHIND_PR.model_copy(update={"author": "pre-commit-ci[bot]"})
        # Auto-merge was armed before the reactive recovery refused.
        mgr._auto_merge_enabled.add("owner/repo#88")
        await mgr._handle_merge_failure(pr, "owner", "repo")

        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="owner",
            repo_name="repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )
        with (
            patch.object(mgr, "_external_closure_result", AsyncMock(return_value=None)),
            patch.object(
                mgr,
                "_maybe_recreate_dependabot_pr",
                AsyncMock(return_value=RecreateResult.none()),
            ),
            patch.object(
                mgr, "_detect_stuck_required_check", AsyncMock(return_value=None)
            ),
            patch.object(
                mgr, "_confirm_failure", AsyncMock(side_effect=lambda p, r: r)
            ),
        ):
            early = await mgr._handle_failed_merge(flow)

        # No early (pending) result: the failure path recorded the refusal.
        assert early is None
        assert flow.result.status == MergeStatus.FAILED
        assert flow.result.error == rebase.LOCAL_REBASE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_a_refused_dependabot_pr_is_not_recreated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed rebase comment plus a stuck check must not recreate."""
        from dependamerge import rebase
        from dependamerge.merge_manager import MergeResult
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock(side_effect=RuntimeError("403"))
        client.update_branch = AsyncMock()
        client.requires_commit_signatures = AsyncMock(return_value=True)
        client.check_pr_commit_signatures = AsyncMock(return_value=(True, []))
        pr = _BEHIND_PR.model_copy()  # dependabot
        await mgr._handle_merge_failure(pr, "owner", "repo")

        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="owner",
            repo_name="repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )
        recreate = AsyncMock()
        with (
            patch.object(mgr, "_external_closure_result", AsyncMock(return_value=None)),
            patch.object(mgr, "_maybe_recreate_dependabot_pr", recreate),
            patch.object(
                mgr,
                "_detect_stuck_required_check",
                AsyncMock(return_value=(True, "pre-commit.ci - pr")),
            ),
        ):
            await mgr._handle_failed_merge(flow)

        recreate.assert_not_awaited()
        assert flow.result.status == MergeStatus.FAILED
        assert flow.result.error == rebase.LOCAL_REBASE_UNAVAILABLE


class TestBehind405UnderActions:
    """A 405 'behind' rejection takes the same refusal path under Actions."""

    @pytest.mark.asyncio
    async def test_a_behind_405_records_the_refusal_and_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dependamerge import rebase

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.update_branch = AsyncMock()
        client.merge_pull_request = AsyncMock(
            side_effect=RuntimeError(
                "Client error '405 Method Not Allowed': GitHub: Head branch "
                "is behind the base branch"
            )
        )
        # Stale cached state: the 405 alone says the PR is behind.
        pr = _BEHIND_PR.model_copy(
            update={"author": "pre-commit-ci[bot]", "mergeable_state": "clean"}
        )

        with patch("dependamerge.merge_manager._retry.asyncio.sleep", AsyncMock()):
            merged = await mgr._merge_pr_with_retry(pr, "owner", "repo")

        assert merged is False
        client.update_branch.assert_not_awaited()
        # One attempt: a refused PR is not retried against the backoff.
        assert client.merge_pull_request.await_count == 1
        reason, refused = await mgr._get_failure_summary(pr)
        assert reason == rebase.LOCAL_REBASE_UNAVAILABLE
        assert refused is False

    @pytest.mark.asyncio
    async def test_outside_actions_a_behind_405_still_backs_off(self) -> None:
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.update_branch = AsyncMock()
        client.merge_pull_request = AsyncMock(
            side_effect=RuntimeError(
                "Client error '405 Method Not Allowed': GitHub: Head branch "
                "is behind the base branch"
            )
        )
        pr = _BEHIND_PR.model_copy(update={"author": "pre-commit-ci[bot]"})

        with patch("dependamerge.merge_manager._retry.asyncio.sleep", AsyncMock()):
            await mgr._merge_pr_with_retry(pr, "owner", "repo")

        assert client.merge_pull_request.await_count > 1
        assert "owner/repo#88" not in mgr._local_rebase_refused

    @pytest.mark.asyncio
    async def test_an_accepted_dependabot_rebase_is_pending_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful macro after a stale-'clean' 405 is auto-merge pending."""
        from dependamerge.merge_manager import MergeResult
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock()
        client.update_branch = AsyncMock()
        client.merge_pull_request = AsyncMock(
            side_effect=RuntimeError(
                "Client error '405 Method Not Allowed': GitHub: Head branch "
                "is behind the base branch"
            )
        )

        async def arm(pr, owner, repo):
            mgr._auto_merge_enabled.add(f"{owner}/{repo}#{pr.number}")
            return True

        pr = _BEHIND_PR.model_copy(update={"mergeable_state": "clean"})  # dependabot
        with (
            patch("dependamerge.merge_manager._retry.asyncio.sleep", AsyncMock()),
            patch.object(mgr, "_enable_auto_merge_with_approval", arm),
        ):
            merged = await mgr._merge_pr_with_retry(pr, "owner", "repo")

        assert merged is False
        # Accepted asynchronously: one attempt, no immediate retry.
        assert client.merge_pull_request.await_count == 1
        assert ("owner", "repo", 88, "@dependabot rebase") in [
            c.args for c in client.post_issue_comment.await_args_list
        ]
        # The shared snapshot now says behind, so the run reports pending.
        assert pr.mergeable_state == "behind"
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="owner",
            repo_name="repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )
        result = mgr._behind_auto_merge_result(flow)
        assert result is not None
        assert result.status == MergeStatus.AUTO_MERGE_PENDING

    @pytest.mark.asyncio
    async def test_without_a_signature_requirement_a_behind_405_backs_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No signature at stake: no REST update here, and the back-off retries."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.requires_commit_signatures = AsyncMock(return_value=False)
        client.update_branch = AsyncMock()
        client.merge_pull_request = AsyncMock(
            side_effect=RuntimeError(
                "Client error '405 Method Not Allowed': GitHub: Head branch "
                "is behind the base branch"
            )
        )
        pr = _BEHIND_PR.model_copy(update={"author": "renovate[bot]"})

        with patch("dependamerge.merge_manager._retry.asyncio.sleep", AsyncMock()):
            await mgr._merge_pr_with_retry(pr, "owner", "repo")

        client.update_branch.assert_not_awaited()
        assert client.merge_pull_request.await_count > 1
        assert "owner/repo#88" not in mgr._local_rebase_refused

    @pytest.mark.asyncio
    async def test_a_failed_dependabot_macro_backs_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No rebase accepted and none refused: retry, do not stop."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock(side_effect=RuntimeError("502"))
        client.requires_commit_signatures = AsyncMock(return_value=False)
        client.update_branch = AsyncMock()
        client.merge_pull_request = AsyncMock(
            side_effect=RuntimeError(
                "Client error '405 Method Not Allowed': GitHub: Head branch "
                "is behind the base branch"
            )
        )
        arm = AsyncMock(return_value=True)
        pr = _BEHIND_PR.model_copy(update={"mergeable_state": "clean"})  # dependabot

        with (
            patch("dependamerge.merge_manager._retry.asyncio.sleep", AsyncMock()),
            patch.object(mgr, "_enable_auto_merge_with_approval", arm),
        ):
            merged = await mgr._merge_pr_with_retry(pr, "owner", "repo")

        assert merged is False
        client.post_issue_comment.assert_awaited()
        arm.assert_not_awaited()
        client.update_branch.assert_not_awaited()
        assert client.merge_pull_request.await_count > 1
        assert "owner/repo#88" not in mgr._local_rebase_refused

    @pytest.mark.asyncio
    async def test_githubs_out_of_date_wording_is_recognised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GitHub's REST answer never says "behind"; the recovery still runs."""
        from dependamerge import rebase

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.update_branch = AsyncMock()
        client.merge_pull_request = AsyncMock(
            side_effect=RuntimeError(
                "Failed to merge PR #88 in owner/repo. Error: Client error "
                "'405 Method Not Allowed'. GitHub: Head branch is out of "
                "date. Review and try the merge again. (PR state: open, "
                "mergeable: True, mergeable_state: unknown)"
            )
        )
        pr = _BEHIND_PR.model_copy(
            update={"author": "pre-commit-ci[bot]", "mergeable_state": "clean"}
        )

        with patch("dependamerge.merge_manager._retry.asyncio.sleep", AsyncMock()):
            merged = await mgr._merge_pr_with_retry(pr, "owner", "repo")

        assert merged is False
        client.update_branch.assert_not_awaited()
        assert client.merge_pull_request.await_count == 1
        reason, _refused = await mgr._get_failure_summary(pr)
        assert reason == rebase.LOCAL_REBASE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_without_auto_merge_an_accepted_macro_keeps_retrying(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A running rebase with no auto-merge armed is not given up on."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock()
        client.update_branch = AsyncMock()
        client.merge_pull_request = AsyncMock(
            side_effect=[RuntimeError(_OUT_OF_DATE_405), True]
        )
        arm = AsyncMock(return_value=False)
        pr = _BEHIND_PR.model_copy(update={"mergeable_state": "clean"})  # dependabot

        with (
            patch("dependamerge.merge_manager._retry.asyncio.sleep", AsyncMock()),
            patch.object(mgr, "_enable_auto_merge_with_approval", arm),
        ):
            merged = await mgr._merge_pr_with_retry(pr, "owner", "repo")

        # GitHub's wording never says "behind", yet the back-off retries.
        assert merged is True
        assert client.merge_pull_request.await_count == 2
        arm.assert_awaited_once()
        assert ("owner", "repo", 88, "@dependabot rebase") in [
            c.args for c in client.post_issue_comment.await_args_list
        ]

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                "Failed to merge PR #88 in owner/behind-repo. Error: Client "
                "error '405 Method Not Allowed' for url 'https://api.github"
                ".com/repos/owner/behind-repo/pulls/88/merge'. GitHub: At "
                "least 1 approving review is required by reviewers with "
                "write access. (PR state: open, mergeable: True, "
                "mergeable_state: blocked) [blocked by branch protection / "
                "required checks]",
                id="repository-name",
            ),
            pytest.param(
                "Failed to merge PR #88 in owner/behind-repo. Error: Client "
                "error '405 Method Not Allowed'. GitHub: Required workflows "
                "'ci' are not satisfied. (PR state: open, mergeable: True, "
                "mergeable_state: behind) [PR branch is behind base branch]",
                id="state-suffix",
            ),
            pytest.param(
                "Failed to merge PR #88 in owner/behind-repo. Error: Client "
                "error '405 Method Not Allowed'. GitHub: Repository rule "
                "violations found Required workflows 'Behind Check' are not "
                "satisfied (PR state: open, mergeable: True, "
                "mergeable_state: blocked)",
                id="workflow-name",
            ),
            pytest.param(
                "Failed to merge PR #88 in owner/behind-repo. Error: Client "
                "error '405 Method Not Allowed'. GitHub: Required status "
                'check "out of date guard" is expected. (PR state: open, '
                "mergeable: True, mergeable_state: blocked)",
                id="check-name",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_behind_outside_githubs_detail_starts_no_recovery(
        self, monkeypatch: pytest.MonkeyPatch, error: str
    ) -> None:
        """Only GitHub's own reason may start a rebase or a refusal."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock()
        client.update_branch = AsyncMock()
        client.requires_commit_signatures = AsyncMock(return_value=True)
        client.merge_pull_request = AsyncMock(side_effect=RuntimeError(error))
        refuse = AsyncMock(return_value=True)
        pr = _BEHIND_PR.model_copy(
            update={
                "mergeable_state": "clean",
                "repository_full_name": "owner/behind-repo",
            }
        )  # dependabot

        with (
            patch("dependamerge.merge_manager._retry.asyncio.sleep", AsyncMock()),
            patch.object(mgr, "_refuse_unsigned_rebase", refuse),
        ):
            await mgr._merge_pr_with_retry(pr, "owner", "behind-repo")

        client.post_issue_comment.assert_not_awaited()
        client.update_branch.assert_not_awaited()
        refuse.assert_not_awaited()
        assert pr.mergeable_state != "behind"
        assert "owner/behind-repo#88" not in mgr._local_rebase_refused


class TestRepoScopedRefusal:
    """A dirty refresh cannot turn a recorded refusal into a conflict."""

    @pytest.mark.asyncio
    async def test_refusal_wins_over_a_dirty_refresh(self) -> None:
        from dependamerge.merge_manager import MergeResult
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        mgr, _client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, repo_scoped=True
        )
        pr = _BEHIND_PR.model_copy(update={"author": "pre-commit-ci[bot]"})
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="owner",
            repo_name="repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )

        async def dispatch(flow):
            # The merge failed and the recovery recorded a refusal.
            mgr._local_rebase_refused.add(flow.pr_key)
            return False, False

        async def refresh(pr_info, owner, repo):
            pr_info.mergeable_state = "dirty"

        conflict = AsyncMock()
        with (
            patch.object(mgr, "_approve_if_review_mandated", AsyncMock()),
            patch.object(mgr, "_merge_under_dispatch_lock", dispatch),
            patch.object(mgr, "_refresh_pr_mergeability", refresh),
            patch.object(mgr, "_handle_merge_conflict", conflict),
        ):
            merged, early = await mgr._attempt_direct_merge(flow)

        assert (merged, early) == (False, None)
        conflict.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_refusal_ends_the_merge_step(self) -> None:
        """A retained workflow 405 and a dirty snapshot cannot reopen it."""
        from dependamerge import rebase
        from dependamerge.merge_manager import MergeResult
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        mgr, _client = make_merge_manager(
            fix_out_of_date=True,
            rebase_local=True,
            repo_scoped=True,
            preview_mode=False,
        )
        pr = _BEHIND_PR.model_copy(update={"author": "pre-commit-ci[bot]"})
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="owner",
            repo_name="repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )
        workflow_405 = (
            "Failed to merge PR #88 in owner/repo. Error: Client error '405 "
            "Method Not Allowed'. GitHub: Required workflows 'ci' are not "
            "satisfied. (PR state: open, mergeable: True, mergeable_state: "
            "blocked)"
        )
        assert mgr._merge_error_indicates_pending_workflows(workflow_405)

        async def dispatch(flow):
            # An earlier attempt raised the workflow 405; a later one was
            # answered with false, and its recovery recorded a refusal.
            mgr._last_merge_exception[flow.pr_key] = RuntimeError(workflow_405)
            mgr._local_rebase_refused.add(flow.pr_key)
            flow.pr_info.mergeable_state = "dirty"
            return False, False

        wait = AsyncMock(return_value=True)
        approve = AsyncMock(return_value=True)
        conflict = AsyncMock()
        with (
            patch.object(mgr, "_auto_merge_will_handle", AsyncMock(return_value=False)),
            patch.object(mgr, "_approve_if_review_mandated", AsyncMock()),
            patch.object(mgr, "_merge_under_dispatch_lock", dispatch),
            patch.object(mgr, "_wait_for_required_workflows_and_retry", wait),
            patch.object(mgr, "_approve_and_retry_if_review_required", approve),
            patch.object(mgr, "_handle_merge_conflict", conflict),
            patch.object(mgr, "_external_closure_result", AsyncMock(return_value=None)),
            patch.object(
                mgr, "_confirm_failure", AsyncMock(side_effect=lambda p, r: r)
            ),
        ):
            early = await mgr._perform_merge(flow)

        assert early is None
        wait.assert_not_awaited()
        approve.assert_not_awaited()
        conflict.assert_not_awaited()
        assert flow.result.status == MergeStatus.FAILED
        assert flow.result.error == rebase.LOCAL_REBASE_UNAVAILABLE
        assert flow.result.merge_refused is False

    @pytest.mark.asyncio
    async def test_a_refusal_from_approval_recovery_ends_the_merge_step(
        self,
    ) -> None:
        """Recovery's own retry can record the refusal; nothing may reopen it."""
        from dependamerge import rebase
        from dependamerge.merge_manager import MergeResult
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        mgr, _client = make_merge_manager(
            fix_out_of_date=True,
            rebase_local=True,
            repo_scoped=True,
            preview_mode=False,
        )
        pr = _BEHIND_PR.model_copy(update={"author": "pre-commit-ci[bot]"})
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="owner",
            repo_name="repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )
        workflow_405 = (
            "Failed to merge PR #88 in owner/repo. Error: Client error '405 "
            "Method Not Allowed'. GitHub: Required workflows 'ci' are not "
            "satisfied. (PR state: open, mergeable: True, mergeable_state: "
            "blocked)"
        )

        async def dispatch(flow):
            # The direct attempt met running workflows; nothing refused yet.
            mgr._last_merge_exception[flow.pr_key] = RuntimeError(workflow_405)
            return False, False

        async def approve(pr_info, owner, repo):
            # Approval recovery retried the merge: answered false on a
            # behind PR, its recovery recorded the refusal, and the
            # workflow 405 stayed behind as the last exception.
            mgr._local_rebase_refused.add(flow.pr_key)
            pr_info.mergeable_state = "dirty"
            return False

        wait = AsyncMock(return_value=False)
        conflict = AsyncMock()
        with (
            patch.object(mgr, "_auto_merge_will_handle", AsyncMock(return_value=False)),
            patch.object(mgr, "_approve_if_review_mandated", AsyncMock()),
            patch.object(mgr, "_merge_under_dispatch_lock", dispatch),
            patch.object(mgr, "_approve_and_retry_if_review_required", approve),
            patch.object(mgr, "_wait_for_required_workflows_and_retry", wait),
            patch.object(mgr, "_handle_merge_conflict", conflict),
            patch.object(mgr, "_external_closure_result", AsyncMock(return_value=None)),
            patch.object(
                mgr, "_confirm_failure", AsyncMock(side_effect=lambda p, r: r)
            ),
        ):
            early = await mgr._perform_merge(flow)

        assert early is None
        wait.assert_not_awaited()
        conflict.assert_not_awaited()
        assert flow.result.status == MergeStatus.FAILED
        assert flow.result.error == rebase.LOCAL_REBASE_UNAVAILABLE
        assert flow.result.merge_refused is False

    @pytest.mark.asyncio
    async def test_a_stale_head_after_the_workflow_wait_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Workflows finish, the base moves: the retry's 405 gets recovery."""
        from dependamerge import rebase
        from dependamerge.merge_manager import MergeResult
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.get = AsyncMock(return_value={})
        client.update_branch = AsyncMock()
        client.merge_pull_request = AsyncMock(
            side_effect=RuntimeError(_OUT_OF_DATE_405)
        )
        pr = _BEHIND_PR.model_copy(
            update={"author": "pre-commit-ci[bot]", "mergeable_state": "blocked"}
        )
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="owner",
            repo_name="repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )
        workflow_405 = (
            "Failed to merge PR #88 in owner/repo. Error: Client error '405 "
            "Method Not Allowed'. GitHub: Required workflows 'ci' are not "
            "satisfied. (PR state: open, mergeable: True, mergeable_state: "
            "blocked)"
        )

        async def dispatch(flow):
            # The first merge met still-running required workflows.
            mgr._last_merge_exception[flow.pr_key] = RuntimeError(workflow_405)
            return False, False

        with (
            patch.object(mgr, "_auto_merge_will_handle", AsyncMock(return_value=False)),
            patch.object(mgr, "_approve_if_review_mandated", AsyncMock()),
            patch.object(mgr, "_merge_under_dispatch_lock", dispatch),
            patch.object(
                mgr,
                "_approve_and_retry_if_review_required",
                AsyncMock(return_value=False),
            ),
            patch.object(
                mgr, "_wait_for_auto_merge", AsyncMock(return_value=(False, False))
            ),
            patch.object(mgr, "_external_closure_result", AsyncMock(return_value=None)),
            patch.object(
                mgr, "_confirm_failure", AsyncMock(side_effect=lambda p, r: r)
            ),
        ):
            early = await mgr._perform_merge(flow)

        assert early is None
        client.merge_pull_request.assert_awaited_once()
        client.update_branch.assert_not_awaited()
        assert flow.result.status == MergeStatus.FAILED
        assert flow.result.error == rebase.LOCAL_REBASE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_after_the_workflow_wait_an_unarmed_macro_still_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Macro accepted, auto-merge unavailable: the merge is retried."""
        from dependamerge.merge_manager import MergeResult
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True, rebase_local=True, preview_mode=False
        )
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock()
        client.update_branch = AsyncMock()
        workflow_405 = (
            "Failed to merge PR #88 in owner/repo. Error: Client error '405 "
            "Method Not Allowed'. GitHub: Required workflows 'ci' are not "
            "satisfied. (PR state: open, mergeable: True, mergeable_state: "
            "blocked)"
        )
        # The first merge meets running workflows, the retry after them a
        # base that moved, and the bounded retry after the macro merges.
        client.merge_pull_request = AsyncMock(
            side_effect=[
                RuntimeError(workflow_405),
                RuntimeError(_OUT_OF_DATE_405),
                True,
            ]
        )
        arm = AsyncMock(return_value=False)
        pr = _BEHIND_PR.model_copy(update={"mergeable_state": "blocked"})  # dependabot
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="owner",
            repo_name="repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )

        with (
            patch("dependamerge.merge_manager._retry.asyncio.sleep", AsyncMock()),
            patch.object(mgr, "_auto_merge_will_handle", AsyncMock(return_value=False)),
            patch.object(mgr, "_approve_if_review_mandated", AsyncMock()),
            patch.object(
                mgr,
                "_approve_and_retry_if_review_required",
                AsyncMock(return_value=False),
            ),
            patch.object(
                mgr,
                "_stop_for_undispatched_workflows",
                AsyncMock(return_value=False),
            ),
            patch.object(
                mgr, "_wait_for_auto_merge", AsyncMock(return_value=(False, False))
            ),
            patch.object(mgr, "_enable_auto_merge_with_approval", arm),
        ):
            early = await mgr._perform_merge(flow)

        assert early is None
        assert flow.result.status == MergeStatus.MERGED
        assert client.merge_pull_request.await_count == 3
        arm.assert_awaited_once()
        client.update_branch.assert_not_awaited()
        assert ("owner", "repo", 88, "@dependabot rebase") in [
            c.args for c in client.post_issue_comment.await_args_list
        ]

    @pytest.mark.asyncio
    async def test_a_landed_rebase_stays_pending_after_a_blocked_refresh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rebase landed and checks run: pending, not failed or recreated."""
        from dependamerge.merge_manager import MergeResult
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        mgr, client = make_merge_manager(
            fix_out_of_date=True,
            rebase_local=True,
            repo_scoped=True,
            preview_mode=False,
        )
        client.get = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock()
        client.update_branch = AsyncMock()
        client.merge_pull_request = AsyncMock(
            side_effect=RuntimeError(_OUT_OF_DATE_405)
        )
        pr = _BEHIND_PR.model_copy(update={"mergeable_state": "clean"})  # dependabot
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="owner",
            repo_name="repo",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )

        async def arm(pr_info, owner, repo):
            mgr._auto_merge_enabled.add(f"{owner}/{repo}#{pr_info.number}")
            return True

        async def refresh(pr_info, owner, repo):
            # The rebase has landed; required checks are still running.
            pr_info.mergeable_state = "blocked"

        recreate = AsyncMock()
        with (
            patch("dependamerge.merge_manager._retry.asyncio.sleep", AsyncMock()),
            patch.object(mgr, "_auto_merge_will_handle", AsyncMock(return_value=False)),
            patch.object(mgr, "_approve_if_review_mandated", AsyncMock()),
            patch.object(mgr, "_is_pr_dirty_now", AsyncMock(return_value=False)),
            patch.object(mgr, "_refresh_pr_mergeability", refresh),
            patch.object(
                mgr,
                "_approve_and_retry_if_review_required",
                AsyncMock(return_value=False),
            ),
            patch.object(mgr, "_enable_auto_merge_with_approval", arm),
            patch.object(mgr, "_external_closure_result", AsyncMock(return_value=None)),
            patch.object(mgr, "_maybe_recreate_dependabot_pr", recreate),
        ):
            result = await mgr._perform_merge(flow)

        assert pr.mergeable_state == "blocked"
        assert result is not None
        assert result.status == MergeStatus.AUTO_MERGE_PENDING
        recreate.assert_not_awaited()
        client.merge_pull_request.assert_awaited_once()


class TestStaleHeadWording:
    """Only GitHub's own sentence about the head counts as stale."""

    @pytest.mark.parametrize(
        ("detail", "stale"),
        [
            ("Head branch is out of date. Review and try the merge again.", True),
            ("Head branch is not up to date with the base branch", True),
            ("Repository rule violations found Head branch is out of date", True),
            ("Required workflows 'Behind Check' are not satisfied", False),
            ('Required status check "out of date guard" is expected.', False),
            ("Required workflows 'head branch is behind' are not satisfied", False),
            ("Required workflows 'CI: head branch is behind' are not satisfied", False),
            (
                'Required status check "Guard. Head branch is out of date" is expected.',
                False,
            ),
            (
                "Repository rule violations found Required workflows 'violations "
                "found Head branch is out of date' are not satisfied",
                False,
            ),
            ("At least 1 approving review is required", False),
        ],
    )
    def test_the_sentence_decides(self, detail: str, stale: bool) -> None:
        from dependamerge.merge_manager._stale_head import _says_head_is_stale

        message = (
            "Failed to merge PR #88 in owner/out-of-date. Error: Client error "
            f"'405 Method Not Allowed'. GitHub: {detail} (PR state: open, "
            "mergeable: True, mergeable_state: behind) [PR branch is behind "
            "base branch]"
        )
        assert _says_head_is_stale(message) is stale
