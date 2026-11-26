# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Reporting a pull request GitHub failed to answer about (issue #688).

A server error says nothing about the pull request, so the reason names
the status and the operation, not the URL or "merge failed"; and a PR
that reads mergeable afterwards is unsettled rather than failed.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from dependamerge.github_async import GitHubServerError, server_failure
from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.merge_manager._single_pr_context import _MergeFlow
from tests.conftest import make_merge_manager, make_pr_info
from tests.test_commit_pinning import OLD, _live, _manager


def _status_error(status: int, body: str | None = None) -> httpx.HTTPStatusError:
    url = "https://api.github.com/repos/o/r/pulls/1/merge"
    request = httpx.Request("PUT", url)
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError(
        f"Server error '{status} Oops' for url '{url}'",
        request=request,
        response=response,
    )


def _wrapped(inner: Exception) -> Exception:
    """A merge failure as ``_validate_merge_result`` raises it."""
    try:
        raise Exception(f"Failed to merge PR o/r#1. Error: {inner}") from inner
    except Exception as error:
        return error


class TestTheReason:
    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    @pytest.mark.parametrize("body", [None, '{"message": "Server Error"}'])
    @pytest.mark.asyncio
    async def test_a_status_is_named(self, status: int, body: str | None) -> None:
        mgr, _client = make_merge_manager()
        pr = make_pr_info()
        mgr._last_merge_exception["o/r#1"] = _wrapped(_status_error(status, body))

        reason, refused = await mgr._get_failure_summary(pr)

        assert reason == f"GitHub returned a server error ({status}) while merging"
        assert refused is False

    @pytest.mark.asyncio
    async def test_no_response_is_named(self) -> None:
        mgr, _client = make_merge_manager()
        mgr._last_merge_exception["o/r#1"] = _wrapped(httpx.ReadTimeout("slow"))

        reason, _refused = await mgr._get_failure_summary(make_pr_info())

        assert reason == "GitHub returned a server error (no response) while merging"

    @pytest.mark.parametrize("status", [403, 405, 409, 422])
    def test_a_client_error_is_not_a_server_error(self, status: int) -> None:
        assert server_failure(_wrapped(_status_error(status))) is None

    def test_the_failed_merge_carries_the_marker(self) -> None:
        mgr, _client = make_merge_manager()
        mgr._last_merge_exception["o/r#1"] = _wrapped(_status_error(502))

        assert mgr._merge_server_error("o/r#1", policy_refusal=False) == (
            "GitHub returned a server error (502) while merging"
        )
        assert mgr._merge_server_error("o/r#1", policy_refusal=True) is None


class TestTheStuckCheck:
    @pytest.mark.asyncio
    async def test_a_server_error_is_not_replaced_by_a_stuck_check(self) -> None:
        mgr, _client = make_merge_manager()
        pr = make_pr_info(author="someone")
        mgr._last_merge_exception["o/r#1"] = _wrapped(_status_error(500))
        detect = AsyncMock(return_value=(True, "ci / build"))
        mgr._detect_stuck_required_check = detect  # type: ignore[method-assign]
        mgr._enable_auto_merge_with_approval = AsyncMock()  # type: ignore[method-assign]
        reason = "GitHub returned a server error (500) while merging"
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)

        out = await mgr._report_merge_failure(pr, "o", "r", result, reason)

        detect.assert_not_awaited()
        assert out.status is MergeStatus.FAILED
        assert out.error == reason
        assert out.server_error == reason
        assert out.merge_refused is False

    @pytest.mark.asyncio
    async def test_a_dependabot_pr_is_not_recreated(self) -> None:
        mgr, _client = make_merge_manager()
        pr = make_pr_info(author="dependabot[bot]")
        mgr._last_merge_exception["o/r#1"] = _wrapped(_status_error(500))
        mgr._external_closure_result = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._behind_auto_merge_result = lambda flow: None  # type: ignore[method-assign]
        detect = AsyncMock(return_value=(True, "ci / build", 3600.0))
        mgr._detect_stuck_required_check = detect  # type: ignore[method-assign]
        mgr._trigger_dependabot_recreate = AsyncMock()  # type: ignore[method-assign]
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        flow = _MergeFlow(pr_info=pr, repo_owner="o", repo_name="r", result=result)

        await mgr._handle_failed_merge(flow)

        detect.assert_not_awaited()
        mgr._trigger_dependabot_recreate.assert_not_awaited()
        assert result.status is MergeStatus.FAILED
        assert result.server_error == (
            "GitHub returned a server error (500) while merging"
        )


class TestTheSettledOutcome:
    REASON = "GitHub returned a server error (500) while merging"

    def _failed(self) -> MergeResult:
        return MergeResult(
            pr_info=make_pr_info(),
            status=MergeStatus.FAILED,
            error=self.REASON,
            server_error=self.REASON,
        )

    @pytest.mark.asyncio
    async def test_clean_and_mergeable_is_unsettled(self) -> None:
        mgr, _client = make_merge_manager()
        result = self._failed()

        out = await mgr._settle_stale_failure(result.pr_info, result, _live(OLD))

        assert out.status is MergeStatus.UNSETTLED
        assert out.error == (
            f"{self.REASON}; now clean and expected to merge on a re-run"
        )

    @pytest.mark.asyncio
    async def test_blocked_stays_failed_with_its_reason(self) -> None:
        mgr, _client = make_merge_manager()
        result = self._failed()
        refreshed: dict[str, Any] = _live(OLD, mergeable_state="blocked")

        out = await mgr._settle_stale_failure(result.pr_info, result, refreshed)

        assert out.status is MergeStatus.FAILED
        assert out.error == self.REASON

    @pytest.mark.asyncio
    async def test_merged_is_merged(self) -> None:
        mgr, client = make_merge_manager()
        result = self._failed()
        client.get = AsyncMock(
            return_value=_live(OLD, state="closed", merged=True, merged_at="now")
        )

        out = await mgr._confirm_failure(result.pr_info, result)

        assert out.status is MergeStatus.MERGED
        assert out.error is None


class TestAMergeThatLanded:
    """A merge answered with a server error that landed all the same."""

    def _flow(self, mgr: Any) -> _MergeFlow:
        pr = make_pr_info(author="someone")
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="o",
            repo_name="r",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )
        mgr._last_merge_exception[flow.pr_key] = _wrapped(_status_error(500))
        return flow

    @pytest.mark.asyncio
    async def test_it_is_confirmed_merged(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        merged = _live(OLD, state="closed", merged=True, merged_at="now")
        client.get = AsyncMock(return_value=merged)
        flow = self._flow(mgr)

        assert await mgr._handle_failed_merge(flow) is None
        out = await mgr._confirm_failure(flow.pr_info, flow.result)

        assert out.status is MergeStatus.MERGED
        assert out.error is None

    @pytest.mark.asyncio
    async def test_a_moved_head_is_not_this_runs_merge(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        merged = _live("b" * 40, state="closed", merged=True, merged_at="now")
        client.get = AsyncMock(return_value=merged)
        flow = self._flow(mgr)

        assert await mgr._handle_failed_merge(flow) is None
        out = await mgr._confirm_failure(flow.pr_info, flow.result)

        assert out.status is MergeStatus.SKIPPED
        assert out.error is not None
        assert "already merged externally" not in out.error


class TestTheApproval:
    @pytest.mark.asyncio
    async def test_a_server_error_fails_the_pr_while_approving(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(OLD))
        client.approve_pull_request = AsyncMock(side_effect=_status_error(503))
        mgr._confirm_failure = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda _pr, result: result
        )

        async def sequence(flow: Any) -> MergeResult:
            await mgr._approve_pr("o", "r", flow.pr_info)
            result: MergeResult = flow.result
            return result

        mgr._run_merge_sequence = sequence  # type: ignore[method-assign]
        result = await mgr._merge_single_pr(make_pr_info(head_sha=OLD))

        assert result.status is MergeStatus.FAILED
        assert result.error == "GitHub returned a server error (503) while approving"
        assert result.server_error == result.error

    @pytest.mark.asyncio
    async def test_auto_merge_is_not_armed_past_a_server_error(self) -> None:
        """The checks-wait path arms auto-merge only after approving."""
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(OLD))
        failure = server_failure(_status_error(500))
        assert failure is not None
        mgr._ensure_pr_approved = AsyncMock(  # type: ignore[method-assign]
            side_effect=GitHubServerError(failure, "approving")
        )
        arm = AsyncMock(return_value=True)
        mgr._enable_auto_merge_for_pr = arm  # type: ignore[method-assign]
        mgr._confirm_failure = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda _pr, result: result
        )

        async def sequence(flow: Any) -> MergeResult:
            await mgr._arm_auto_merge_before_wait(flow)
            result: MergeResult = flow.result
            return result

        mgr._run_merge_sequence = sequence  # type: ignore[method-assign]
        result = await mgr._merge_single_pr(make_pr_info(head_sha=OLD))

        arm.assert_not_awaited()
        assert result.status is MergeStatus.FAILED
        assert result.error == "GitHub returned a server error (500) while approving"
        assert result.server_error == result.error

    @pytest.mark.asyncio
    async def test_the_approval_raises_the_typed_error(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(OLD))
        client.approve_pull_request = AsyncMock(side_effect=httpx.ConnectError("x"))

        with pytest.raises(GitHubServerError, match=r"\(no response\) while approving"):
            await mgr._approve_pr("o", "r", make_pr_info(head_sha=OLD))


class TestTheRecoveryPaths:
    """The rebase and recreate paths keep the server error as the cause."""

    REASON = "GitHub returned a server error (500) while merging"

    @pytest.mark.asyncio
    async def test_a_rebased_merge_reports_the_server_error(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(OLD))
        pr = make_pr_info(head_sha=OLD)
        mgr._merge_pr_with_retry = AsyncMock(return_value=False)  # type: ignore[method-assign]
        mgr._last_merge_exception["o/r#1"] = _wrapped(_status_error(500))

        out = await mgr._merge_rebased_pr(
            pr, "o", "r", MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        )

        assert out.status is MergeStatus.FAILED
        assert out.error == self.REASON
        assert out.server_error == self.REASON
        settled = await mgr._settle_stale_failure(pr, out, _live(OLD))
        assert settled.status is MergeStatus.UNSETTLED

    def test_a_rebase_approval_reports_the_server_error(self) -> None:
        mgr, _client = make_merge_manager()
        pr = make_pr_info()
        failure = server_failure(_status_error(503))
        assert failure is not None

        out = mgr._report_rebase_approval_failure(
            pr,
            "o",
            "r",
            MergeResult(pr_info=pr, status=MergeStatus.PENDING),
            GitHubServerError(failure, "approving"),
        )

        assert out.status is MergeStatus.FAILED
        assert out.error == "GitHub returned a server error (503) while approving"
        assert out.server_error == out.error

    @pytest.mark.asyncio
    async def test_a_replacement_merge_is_left_for_a_re_run(self, monkeypatch) -> None:
        """Not ``BLOCKED``: GitHub failing to answer says nothing about it."""
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(OLD))
        client.merge_pull_request = AsyncMock(side_effect=_status_error(500))
        mgr._approve_pr = AsyncMock()  # type: ignore[method-assign]
        replacement = make_pr_info(head_sha=OLD, number=2)

        async def sequence(flow: Any) -> MergeResult:
            await mgr._merge_recreated_pr(flow, replacement)
            result: MergeResult = flow.result
            return result

        mgr._run_merge_sequence = sequence  # type: ignore[method-assign]
        result = await mgr._merge_single_pr(make_pr_info(head_sha=OLD))

        assert result.pr_info is replacement
        assert result.status is MergeStatus.UNSETTLED
        assert result.error == (
            f"{self.REASON}; now clean and expected to merge on a re-run"
        )

    @pytest.mark.asyncio
    async def test_a_replacement_still_blocked_stays_failed(self, monkeypatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(OLD))
        client.merge_pull_request = AsyncMock(side_effect=_status_error(500))
        mgr._approve_pr = AsyncMock()  # type: ignore[method-assign]
        replacement = make_pr_info(head_sha=OLD, number=2)

        async def sequence(flow: Any) -> MergeResult:
            await mgr._merge_recreated_pr(flow, replacement)
            result: MergeResult = flow.result
            return result

        mgr._run_merge_sequence = sequence  # type: ignore[method-assign]
        client.get = AsyncMock(return_value=_live(OLD, mergeable_state="blocked"))
        result = await mgr._merge_single_pr(make_pr_info(head_sha=OLD))

        assert result.status is MergeStatus.FAILED
        assert result.error == self.REASON

    @pytest.mark.parametrize("dirtied_by", ["refresh", "recovery"])
    @pytest.mark.asyncio
    async def test_a_dirty_reading_does_not_replace_it(self, dirtied_by: str) -> None:
        """Repo-scoped: the retries end on a 500, then the PR reads ``dirty``."""
        mgr, _client = make_merge_manager(repo_scoped=True, preview_mode=False)
        pr = make_pr_info(author="someone")
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="o",
            repo_name="r",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )

        async def dispatch(flow: _MergeFlow) -> tuple[bool, bool]:
            mgr._last_merge_exception[flow.pr_key] = _wrapped(_status_error(500))
            return False, False

        async def dirty(pr_info: Any, *_args: Any) -> bool:
            pr_info.mergeable_state = "dirty"
            return False

        refresh = AsyncMock(side_effect=dirty if dirtied_by == "refresh" else None)
        recovery = AsyncMock(
            side_effect=dirty if dirtied_by == "recovery" else None, return_value=False
        )
        conflict = AsyncMock()
        mgr._approve_if_review_mandated = AsyncMock()  # type: ignore[method-assign]
        mgr._merge_under_dispatch_lock = dispatch  # type: ignore[method-assign]
        mgr._refresh_pr_mergeability = refresh  # type: ignore[method-assign]
        mgr._approve_and_retry_if_review_required = recovery  # type: ignore[method-assign]
        mgr._handle_merge_conflict = conflict  # type: ignore[method-assign]
        mgr._external_closure_result = AsyncMock(return_value=None)  # type: ignore[method-assign]

        assert await mgr._perform_merge(flow) is None

        conflict.assert_not_awaited()
        assert flow.result.status is MergeStatus.FAILED
        assert flow.result.error == self.REASON
        assert flow.result.server_error == self.REASON

    @pytest.mark.asyncio
    async def test_a_missing_approval_starts_no_new_writes(self) -> None:
        """A blocked PR whose retries ended on a 500 is not approved again."""
        mgr, client = make_merge_manager(preview_mode=False)
        pr = make_pr_info(author="someone", mergeable_state="blocked")
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="o",
            repo_name="r",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )

        async def dispatch(flow: _MergeFlow) -> tuple[bool, bool]:
            mgr._last_merge_exception[flow.pr_key] = _wrapped(_status_error(500))
            return False, False

        client.analyze_block_reason = AsyncMock(return_value="Requires approval")
        approve = AsyncMock(return_value=True)
        retry = AsyncMock(return_value=True)
        mgr._approve_if_review_mandated = AsyncMock()  # type: ignore[method-assign]
        mgr._merge_under_dispatch_lock = dispatch  # type: ignore[method-assign]
        mgr._ensure_pr_approved = approve  # type: ignore[method-assign]
        mgr._retry_merge_under_dispatch_lock = retry  # type: ignore[method-assign]
        mgr._external_closure_result = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._detect_stuck_required_check = AsyncMock(return_value=None)  # type: ignore[method-assign]

        assert await mgr._perform_merge(flow) is None

        approve.assert_not_awaited()
        retry.assert_not_awaited()
        assert flow.result.status is MergeStatus.FAILED
        assert flow.result.error == self.REASON
        assert flow.result.server_error == self.REASON

    @pytest.mark.asyncio
    async def test_an_armed_behind_pr_keeps_the_server_error(self) -> None:
        """``--force=all``: auto-merge armed on a behind PR does not hide it."""
        mgr, _client = make_merge_manager(force_level="all")
        pr = make_pr_info(author="someone", mergeable_state="behind")
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="o",
            repo_name="r",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )
        mgr._auto_merge_enabled.add(flow.pr_key)
        mgr._rebase_requested.add(flow.pr_key)
        mgr._last_merge_exception[flow.pr_key] = _wrapped(_status_error(500))
        mgr._external_closure_result = AsyncMock(return_value=None)  # type: ignore[method-assign]

        assert await mgr._handle_failed_merge(flow) is None

        assert flow.result.status is MergeStatus.FAILED
        assert flow.result.error == self.REASON
        assert flow.result.server_error == self.REASON


def _approval_error(status: int = 500) -> GitHubServerError:
    failure = server_failure(_status_error(status))
    assert failure is not None
    return GitHubServerError(failure, "approving")


class TestTheApprovalReachesTheHandler:
    """An approval that spent its retries is never swallowed on the way up."""

    REASON = "GitHub returned a server error (500) while approving"

    @pytest.mark.asyncio
    async def test_a_replacement_is_not_armed_and_is_named(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(OLD, mergeable_state="blocked"))
        mgr._ensure_pr_approved = AsyncMock(  # type: ignore[method-assign]
            side_effect=_approval_error()
        )
        arm = AsyncMock(return_value=True)
        mgr._enable_auto_merge_for_pr = arm  # type: ignore[method-assign]
        replacement = make_pr_info(head_sha=OLD, number=2, node_id="PR_2")

        async def sequence(flow: Any) -> MergeResult:
            await mgr._auto_merge_recreated_pr("o", "r", replacement)
            result: MergeResult = flow.result
            return result

        mgr._run_merge_sequence = sequence  # type: ignore[method-assign]
        result = await mgr._merge_single_pr(make_pr_info(head_sha=OLD))

        arm.assert_not_awaited()
        assert result.pr_info is replacement
        assert result.status is MergeStatus.FAILED
        assert result.error == self.REASON
        assert result.server_error == self.REASON

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["rest", "macro", "local"])
    async def test_the_step_five_rebase_passes_it_on(
        self, path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import MagicMock

        from dependamerge.rebase import local_rebase, paths
        from dependamerge.rebase.context import RebaseContext

        monkeypatch.setattr(
            local_rebase, "local_rebase_pr", AsyncMock(return_value=True)
        )
        ctx = RebaseContext(
            github_client=AsyncMock(),
            token="t",
            host="github.com",
            rebase_local=False,
            preview_mode=False,
            merge_recheck_interval=0.0,
            merge_poll_max_attempts=1,
            log=MagicMock(),
            console=MagicMock(),
            rebased_prs=set(),
            enable_auto_merge=AsyncMock(side_effect=_approval_error()),
            request_dependabot_rebase=AsyncMock(return_value=True),
        )
        pr = make_pr_info()
        kw: dict[str, Any] = {"ctx": ctx, "pr_info": pr, "owner": "o", "repo": "r"}
        run = {
            "rest": lambda: paths._run_rest_path(**kw),
            "macro": lambda: paths._run_dependabot_macro_path(
                **kw, local_reason="test"
            ),
            "local": lambda: paths._run_local_path(**kw, local_reason="test"),
        }[path]

        with pytest.raises(GitHubServerError, match="while approving"):
            await run()
        assert not ctx.rebased_prs

    @pytest.mark.asyncio
    async def test_the_no_wait_conflict_path_does_not_arm(self) -> None:
        mgr, _client = make_merge_manager()
        mgr._approve_pr = AsyncMock(side_effect=_approval_error())  # type: ignore[method-assign]
        mgr._request_dependabot_rebase = AsyncMock(return_value=True)  # type: ignore[method-assign]
        arm = AsyncMock(return_value=True)
        mgr._enable_auto_merge_for_pr = arm  # type: ignore[method-assign]
        pr = make_pr_info(mergeable_state="dirty")

        with pytest.raises(GitHubServerError, match="while approving"):
            await mgr._handle_conflict_no_wait(
                pr, "o", "r", MergeResult(pr_info=pr, status=MergeStatus.PENDING), False
            )
        arm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_merge_retry_loop_does_not_retry_it(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(OLD, mergeable_state="behind"))
        client.merge_pull_request = AsyncMock(return_value=False)
        mgr._handle_merge_failure = AsyncMock(  # type: ignore[method-assign]
            side_effect=_approval_error()
        )
        pr = make_pr_info(head_sha=OLD, mergeable_state="behind")

        with pytest.raises(GitHubServerError, match="while approving"):
            await mgr._merge_pr_with_retry(pr, "o", "r")
        assert client.merge_pull_request.await_count == 1
