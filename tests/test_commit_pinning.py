# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Every merge, approval and auto-merge names the commit it means.

Two rules (issue #678):

* A pull request nobody confirmed merges at the head the run last
  evaluated; when GitHub refuses because the head moved, the snapshot
  is refreshed and the bounded retry tries again at the new head.
* A pull request a person confirmed is approved and merged only at the
  commit they confirmed, never has auto-merge armed, and is skipped
  when its head moves or must be moved.

"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dependamerge.github_async import ApprovalError, GitHubAsync
from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager, make_pr_info

OLD = "a" * 40
NEW = "b" * 40

HEAD_MOVED_409 = (
    "Client error '409 Conflict' for url "
    "'https://api.github.com/repos/o/r/pulls/1/merge'\n"
    "GitHub: Head branch was modified. Review and try the merge again."
)


def _live(head: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": "open",
        "merged": False,
        "draft": False,
        "mergeable": True,
        "mergeable_state": "clean",
        "head": {"sha": head},
    }
    payload.update(extra)
    return payload


def _last(mock: Any) -> Any:
    call = mock.await_args
    assert call is not None
    return call


def _manager(*, confirmed: str | None = None, label: str | None = None, **kw: Any):
    from dependamerge.merge_manager import ConfirmedHead, head_key

    heads = (
        {head_key("O/R", 1): ConfirmedHead(confirmed, label)}
        if confirmed is not None
        else None
    )
    mgr, client = make_merge_manager(
        merge_method="merge", max_retries=2, confirmed_heads=heads, **kw
    )
    mgr._pr_merge_methods["o/r"] = "merge"
    mgr._should_skip_approval = AsyncMock(return_value=(False, None))  # type: ignore[method-assign]
    return mgr, client


class TestEveryWriteNamesACommit:
    def test_no_client_method_writes_without_a_commit(self) -> None:
        for method, arg in (
            (GitHubAsync.merge_pull_request, "sha"),
            (GitHubAsync.approve_pull_request, "commit_id"),
            (GitHubAsync.enable_auto_merge, "expected_head_oid"),
        ):
            param = inspect.signature(method).parameters[arg]
            assert param.kind is inspect.Parameter.KEYWORD_ONLY
            assert param.default is inspect.Parameter.empty

    def test_the_sync_wrappers_are_gone(self) -> None:
        from dependamerge.github_client import GitHubClient

        assert not hasattr(GitHubClient, "merge_pull_request")
        assert not hasattr(GitHubClient, "approve_pull_request")

    @pytest.mark.asyncio
    async def test_payloads_carry_the_commit(self) -> None:
        api = GitHubAsync(token="t")
        api.put = AsyncMock(return_value={"merged": True})  # type: ignore[method-assign]
        api.post = AsyncMock(return_value={})  # type: ignore[method-assign]
        api.graphql = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "enablePullRequestAutoMerge": {
                    "pullRequest": {"autoMergeRequest": {"enabledAt": "now"}}
                }
            }
        )
        try:
            await api.merge_pull_request("o", "r", 1, "squash", sha=OLD)
            await api.approve_pull_request("o", "r", 1, "ok", commit_id=OLD)
            await api.enable_auto_merge("PR_node", "merge", expected_head_oid=OLD)
        finally:
            await api.aclose()
        assert _last(api.put).kwargs["json"] == {
            "merge_method": "squash",
            "sha": OLD,
        }
        assert _last(api.post).kwargs["json"]["commit_id"] == OLD
        query, variables = _last(api.graphql).args
        assert variables["expectedHeadOid"] == OLD
        assert "expectedHeadOid: $expectedHeadOid" in query

    @pytest.mark.asyncio
    async def test_an_empty_commit_is_refused(self) -> None:
        api = GitHubAsync(token="t")
        api.put = AsyncMock()  # type: ignore[method-assign]
        api.post = AsyncMock()  # type: ignore[method-assign]
        api.graphql = AsyncMock()  # type: ignore[method-assign]
        try:
            with pytest.raises(ValueError):
                await api.merge_pull_request("o", "r", 1, sha="")
            with pytest.raises(ValueError):
                await api.approve_pull_request("o", "r", 1, "ok", commit_id="")
            with pytest.raises(ValueError):
                await api.enable_auto_merge("PR_node", expected_head_oid="")
        finally:
            await api.aclose()
        api.put.assert_not_awaited()
        api.post.assert_not_awaited()
        api.graphql.assert_not_awaited()


class TestUnconfirmed:
    @pytest.mark.asyncio
    async def test_a_409_retries_at_the_refreshed_head(self) -> None:
        mgr, client = _manager()
        pr = make_pr_info(head_sha=OLD)
        client.merge_pull_request = AsyncMock(
            side_effect=[Exception(HEAD_MOVED_409), True]
        )
        client.get = AsyncMock(return_value=_live(NEW))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            assert await mgr._merge_pr_with_retry(pr, "o", "r") is True

        shas = [c.kwargs["sha"] for c in client.merge_pull_request.await_args_list]
        assert shas == [OLD, NEW]
        assert pr.head_sha == NEW

    @pytest.mark.asyncio
    async def test_a_head_that_keeps_moving_is_a_refusal(self) -> None:
        from dependamerge.merge_manager._head_pin import HEAD_CHANGED_DURING_MERGE

        mgr, client = _manager()
        pr = make_pr_info(head_sha=OLD)
        error = Exception(HEAD_MOVED_409)
        client.merge_pull_request = AsyncMock(side_effect=error)
        client.get = AsyncMock(return_value=_live(NEW))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            assert await mgr._merge_pr_with_retry(pr, "o", "r") is False

        calls = client.merge_pull_request.await_args_list
        assert len(calls) == mgr.max_retries + 1
        assert all(c.kwargs["sha"] for c in calls)
        assert mgr._failure_summary_from_exception("o/r#1", error, pr) == (
            HEAD_CHANGED_DURING_MERGE,
            True,
        )

    @pytest.mark.asyncio
    async def test_a_refresh_after_a_rebase_moves_the_merge_head(self) -> None:
        mgr, client = _manager()
        pr = make_pr_info(head_sha=OLD, mergeable_state="behind")
        client.get = AsyncMock(return_value=_live(NEW))
        client.merge_pull_request = AsyncMock(return_value=True)

        await mgr._refresh_pr_mergeable("o", "r", pr, "o/r#1")
        assert await mgr._dispatch_merge(pr, "o", "r") is True

        assert pr.mergeable_state == "clean"
        assert _last(client.merge_pull_request).kwargs["sha"] == NEW

    @pytest.mark.asyncio
    async def test_a_dirty_check_records_the_head_it_read(self) -> None:
        mgr, client = _manager()
        pr = make_pr_info(head_sha=OLD)
        client.get = AsyncMock(return_value=_live(NEW))

        assert await mgr._is_pr_dirty_now(pr, "o", "r") is False
        assert pr.head_sha == NEW

    @pytest.mark.asyncio
    async def test_approval_names_the_live_head(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(NEW))

        assert await mgr._approve_pr("o", "r", make_pr_info(head_sha=OLD)) is True

        assert _last(client.approve_pull_request).kwargs["commit_id"] == NEW

    @pytest.mark.asyncio
    async def test_a_refused_commit_id_retries_at_the_new_head(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(side_effect=[_live(OLD), _live(NEW)])
        client.approve_pull_request = AsyncMock(
            side_effect=[ApprovalError("o/r#1", 422, "Head sha moved"), None]
        )

        assert await mgr._approve_pr("o", "r", make_pr_info(head_sha=OLD)) is True

        ids = [c.kwargs["commit_id"] for c in client.approve_pull_request.mock_calls]
        assert ids == [OLD, NEW]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("flag", "signal"),
        [(True, "DraftPullRequestError"), (None, "DraftStateUnknownError")],
    )
    async def test_a_refused_commit_id_is_retried_only_on_a_non_draft(
        self, flag: bool | None, signal: str
    ) -> None:
        from dependamerge import merge_signals

        moved = _live(NEW, draft=flag)
        if flag is None:
            del moved["draft"]
        mgr, client = _manager()
        client.get = AsyncMock(side_effect=[_live(OLD), moved])
        client.approve_pull_request = AsyncMock(
            side_effect=ApprovalError("o/r#1", 422, "Head sha moved")
        )

        with pytest.raises(getattr(merge_signals, signal)):
            await mgr._approve_pr("o", "r", make_pr_info(head_sha=OLD))

        client.approve_pull_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_merge_is_armed_at_the_snapshot_head(self) -> None:
        mgr, client = _manager()
        client.enable_auto_merge = AsyncMock(return_value=True)

        pr = make_pr_info(head_sha=OLD, node_id="PR_node")
        assert await mgr._enable_auto_merge_for_pr(pr, "o", "r") is True

        client.enable_auto_merge.assert_awaited_once_with(
            "PR_node", "merge", expected_head_oid=OLD
        )

    @pytest.mark.asyncio
    async def test_a_refused_arming_retries_at_the_live_head(self) -> None:
        mgr, client = _manager()
        client.enable_auto_merge = AsyncMock(side_effect=[False, True])
        client.get = AsyncMock(return_value=_live(NEW, auto_merge=None))

        pr = make_pr_info(head_sha=OLD, node_id="PR_node")
        assert await mgr._enable_auto_merge_for_pr(pr, "o", "r") is True

        heads = [
            c.kwargs["expected_head_oid"]
            for c in client.enable_auto_merge.await_args_list
        ]
        assert heads == [OLD, NEW]

    @pytest.mark.asyncio
    async def test_the_required_workflows_retry_names_the_head(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(OLD))
        client.merge_pull_request = AsyncMock(return_value=True)

        pr = make_pr_info(head_sha=OLD)
        merged = await mgr._dispatch_after_required_workflows(
            pr, "o", "r", "o/r#1", "merge"
        )

        assert merged is True
        assert _last(client.merge_pull_request).kwargs["sha"] == OLD

    @pytest.mark.asyncio
    async def test_a_required_workflows_head_move_waits_again(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(side_effect=[_live(OLD), _live(NEW)])
        client.merge_pull_request = AsyncMock(side_effect=Exception(HEAD_MOVED_409))

        pr = make_pr_info(head_sha=OLD)
        merged = await mgr._dispatch_after_required_workflows(
            pr, "o", "r", "o/r#1", "merge"
        )

        assert merged is False
        assert pr.head_sha == NEW
        assert mgr._last_merge_exception_head["o/r#1"] == OLD

    @pytest.mark.asyncio
    async def test_a_recreated_pr_merges_at_its_head(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(NEW))
        client.merge_pull_request = AsyncMock(return_value=True)

        replacement = make_pr_info(number=2, head_sha=NEW)
        outcome = await mgr._dispatch_recreated_merge("o", "r", replacement)

        assert outcome == (True, False, False)
        assert _last(client.merge_pull_request).kwargs["sha"] == NEW

    @pytest.mark.asyncio
    async def test_a_recreated_pr_retries_at_a_moved_head(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(NEW))
        client.merge_pull_request = AsyncMock(
            side_effect=[Exception(HEAD_MOVED_409), True]
        )

        replacement = make_pr_info(number=2, head_sha=OLD)
        outcome = await mgr._dispatch_recreated_merge("o", "r", replacement)

        assert outcome == (True, False, False)
        shas = [c.kwargs["sha"] for c in client.merge_pull_request.mock_calls]
        assert shas == [OLD, NEW]

    @pytest.mark.asyncio
    async def test_a_recreated_pr_whose_head_keeps_moving(self) -> None:
        from dependamerge.merge_manager._head_pin import HEAD_CHANGED_DURING_MERGE
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        heads = iter(f"{n:040x}" for n in range(1, 20))
        mgr, client = _manager()
        client.get = AsyncMock(side_effect=lambda *_a, **_k: _live(next(heads)))
        client.merge_pull_request = AsyncMock(side_effect=Exception(HEAD_MOVED_409))
        replacement = make_pr_info(number=2, head_sha=OLD)

        outcome = await mgr._dispatch_recreated_merge("o", "r", replacement)

        assert outcome == (False, True, True)
        assert client.merge_pull_request.await_count == mgr.max_retries + 1

        mgr._approve_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
        mgr._dispatch_recreated_merge = AsyncMock(  # type: ignore[method-assign]
            return_value=outcome
        )
        mgr._confirm_failure = AsyncMock()  # type: ignore[method-assign]
        original = make_pr_info()
        flow = _MergeFlow(
            pr_info=original,
            result=MergeResult(pr_info=original, status=MergeStatus.PENDING),
            repo_owner="o",
            repo_name="r",
        )
        await mgr._merge_recreated_pr(flow, replacement)

        assert flow.result.status is MergeStatus.FAILED
        assert flow.result.error == HEAD_CHANGED_DURING_MERGE
        assert flow.result.merge_refused is True


class TestHeadMovedVerdict:
    @pytest.mark.parametrize(
        ("message", "moved"),
        [
            (HEAD_MOVED_409, True),
            (
                "Failed to merge PR #1 in o/r. Error: boom. GitHub: Head branch "
                "was modified. Review and try the merge again.",
                True,
            ),
            (
                "Failed to merge PR #1 in o/r. Error: Client error '405 Method Not "
                "Allowed' for url 'https://x/merge'. GitHub: Required workflows "
                "'Head SHA Moved' are not satisfied (PR state: open)",
                False,
            ),
            (
                "Failed to merge PR #1 in o/r. Error: Client error '405 Method Not "
                "Allowed' for url 'https://x/merge'. GitHub: Required status check "
                "'Head branch was modified' is expected. (PR state: open)",
                False,
            ),
            (
                "Failed to merge PR #1 in o/r. Error: Client error '405 Method Not "
                "Allowed' for url 'https://x/merge'. GitHub: Required status check "
                "'409 Conflict' for url is expected. (PR state: open)",
                False,
            ),
        ],
    )
    def test_only_the_status_or_githubs_verdict_counts(
        self, message: str, moved: bool
    ) -> None:
        from dependamerge.merge_manager._head_pin import says_head_moved

        assert says_head_moved(message) is moved

    @pytest.mark.parametrize(
        ("status", "detail", "moved"),
        [
            (422, "Unprocessable Entity; Head sha moved", True),
            (422, "commit_id is not part of the pull request", True),
            (422, "Required check 'head sha moved' failed", False),
            (403, "Head sha moved", False),
        ],
    )
    def test_an_approval_refusal_names_the_head_only_as_an_entry(
        self, status: int, detail: str, moved: bool
    ) -> None:
        from dependamerge.merge_manager._head_pin import approval_says_head_moved

        refusal = ApprovalError("o/r#1", status, detail)
        assert approval_says_head_moved(refusal) is moved


class TestConfirmed:
    @pytest.mark.asyncio
    async def test_unmoved_is_approved_and_merged_at_the_confirmed_commit(
        self,
    ) -> None:
        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(return_value=_live(OLD))
        client.merge_pull_request = AsyncMock(return_value=True)
        pr = make_pr_info(head_sha=OLD)

        assert await mgr._approve_pr("o", "r", pr) is True
        assert await mgr._merge_pr_with_retry(pr, "o", "r") is True

        assert _last(client.approve_pull_request).kwargs["commit_id"] == OLD
        assert _last(client.merge_pull_request).kwargs["sha"] == OLD

    @pytest.mark.asyncio
    async def test_a_moved_head_is_not_approved(self) -> None:
        from dependamerge.merge_signals import ConfirmedHeadSignal

        mgr, client = _manager(confirmed=OLD, label="major version bump")
        client.get = AsyncMock(return_value=_live(NEW))

        with pytest.raises(ConfirmedHeadSignal) as info:
            await mgr._approve_pr("o", "r", make_pr_info(head_sha=OLD))

        assert info.value.reason == (
            "major version bump; head changed since it was confirmed "
            f"(was {OLD[:7]}, now {NEW[:7]}); re-run to confirm the new commit"
        )
        client.approve_pull_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_409_on_a_moved_head_is_not_retried(self) -> None:
        from dependamerge.merge_signals import ConfirmedHeadSignal

        mgr, client = _manager(confirmed=OLD)
        client.merge_pull_request = AsyncMock(side_effect=Exception(HEAD_MOVED_409))
        client.get = AsyncMock(return_value=_live(NEW))

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            await mgr._merge_pr_with_retry(make_pr_info(head_sha=OLD), "o", "r")

        client.merge_pull_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_moved_snapshot_is_not_merged(self) -> None:
        from dependamerge.merge_signals import ConfirmedHeadSignal

        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(return_value=_live(NEW))

        pr = make_pr_info(head_sha=NEW)
        with pytest.raises(ConfirmedHeadSignal):
            await mgr._dispatch_after_required_workflows(pr, "o", "r", "o/r#1", "merge")
        client.merge_pull_request.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("read", [Exception("boom"), _live(OLD), {"draft": False}])
    async def test_an_inconclusive_read_after_a_409_is_a_skip(self, read: Any) -> None:
        from dependamerge.merge_signals import ConfirmedHeadSignal

        reads = AsyncMock(side_effect=read) if isinstance(read, Exception) else None
        mgr, client = _manager(confirmed=OLD)
        client.merge_pull_request = AsyncMock(side_effect=Exception(HEAD_MOVED_409))
        mgr._fetch_pr_state = reads or AsyncMock(return_value=read)  # type: ignore[method-assign]
        client.get = AsyncMock(return_value=_live(OLD))
        unknown = f"(was {OLD[:7]}, now unknown)"

        with pytest.raises(ConfirmedHeadSignal) as info:
            await mgr._merge_pr_with_retry(make_pr_info(head_sha=OLD), "o", "r")
        assert unknown in info.value.reason
        with pytest.raises(ConfirmedHeadSignal) as info:
            await mgr._dispatch_after_required_workflows(
                make_pr_info(head_sha=OLD), "o", "r", "o/r#1", "merge"
            )
        assert unknown in info.value.reason
        assert client.merge_pull_request.await_count == 2

    @pytest.mark.asyncio
    async def test_a_refused_commit_id_with_an_unreadable_head_is_a_skip(
        self,
    ) -> None:
        from dependamerge.merge_signals import ConfirmedHeadSignal

        mgr, client = _manager(confirmed=OLD)
        mgr._fetch_pr_state = AsyncMock(  # type: ignore[method-assign]
            side_effect=[_live(OLD), Exception("boom")]
        )
        client.approve_pull_request = AsyncMock(
            side_effect=ApprovalError("o/r#1", 422, "Head sha moved")
        )

        with pytest.raises(ConfirmedHeadSignal, match="now unknown"):
            await mgr._approve_pr("o", "r", make_pr_info(head_sha=OLD))
        client.approve_pull_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_merge_is_never_armed(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        pr = make_pr_info(head_sha=OLD, node_id="PR_node")

        assert await mgr._enable_auto_merge_for_pr(pr, "o", "r") is False
        assert await mgr._enable_auto_merge_with_approval(pr, "o", "r") is False

        client.enable_auto_merge.assert_not_awaited()
        client.approve_pull_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_armed_auto_merge_is_a_skip_before_any_write(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(
            return_value=_live(OLD, auto_merge={"merge_method": "merge"})
        )
        mgr._gate_github2gerrit = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._gate_pr_still_open = AsyncMock(return_value=None)  # type: ignore[method-assign]
        mgr._gate_mergeable_state = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await mgr._merge_single_pr(make_pr_info(head_sha=OLD))

        assert result.status == MergeStatus.SKIPPED
        assert result.error is not None
        assert result.error.startswith("auto-merge is enabled")
        mgr._gate_mergeable_state.assert_not_awaited()
        client.approve_pull_request.assert_not_awaited()
        client.merge_pull_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_pre_write_check_passes_an_unmoved_head(self) -> None:
        from dependamerge.merge_manager._head_pin import check_confirmed_pr

        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(return_value=_live(OLD, auto_merge=None))

        await check_confirmed_pr(mgr, make_pr_info(head_sha=OLD), "o", "r")

    @pytest.mark.asyncio
    async def test_an_unknown_confirmed_head_writes_nothing(self) -> None:
        from dependamerge.merge_manager._head_pin import check_confirmed_pr
        from dependamerge.merge_signals import ConfirmedHeadSignal

        mgr, client = _manager(confirmed="")
        client.get = AsyncMock(return_value=_live(NEW))
        pr = make_pr_info(head_sha="")

        with pytest.raises(ConfirmedHeadSignal, match="head unknown"):
            await check_confirmed_pr(mgr, pr, "o", "r")
        with pytest.raises(ConfirmedHeadSignal, match="head unknown"):
            await mgr._merge_pr_with_retry(pr, "o", "r")
        with pytest.raises(ConfirmedHeadSignal, match="head unknown"):
            await mgr._approve_pr("o", "r", pr)
        client.approve_pull_request.assert_not_awaited()
        client.merge_pull_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_rebase_request_stops_after_it_is_made(self) -> None:
        from dependamerge.merge_signals import ConfirmedHeadSignal

        mgr, _client = _manager(confirmed=OLD)
        ask = AsyncMock(return_value=True)
        mgr._ask_dependabot_to_rebase = ask  # type: ignore[method-assign]

        with pytest.raises(ConfirmedHeadSignal) as info:
            await mgr._request_dependabot_rebase(make_pr_info(), "o", "r")

        ask.assert_awaited_once()
        assert info.value.reason == (
            "rebase requested; re-run to confirm the rebased commit"
        )

    @pytest.mark.asyncio
    async def test_an_update_branch_is_counted_then_stops(self) -> None:
        from dependamerge.merge_signals import ConfirmedHeadSignal

        mgr, client = _manager(confirmed=OLD)
        mgr.progress_tracker = MagicMock()
        client.update_branch = AsyncMock()

        flow = MagicMock()
        flow.pr_info = make_pr_info(head_sha=OLD, mergeable_state="behind")
        flow.repo_owner, flow.repo_name = "o", "r"
        with (
            patch(
                "dependamerge.rebase.dispatch.should_use_local_rebase",
                new=AsyncMock(return_value=(False, None)),
            ),
            pytest.raises(ConfirmedHeadSignal, match="rebase requested"),
        ):
            await mgr._run_step5_rebase(flow)

        client.update_branch.assert_awaited_once_with("o", "r", 1)
        mgr.progress_tracker.record_rebase.assert_called_once()
        client.enable_auto_merge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_recreate_writes_nothing_to_the_replacement(self) -> None:
        from dependamerge.merge_manager._types import RecreateCause
        from dependamerge.merge_signals import ConfirmedHeadSignal

        mgr, _client = _manager(confirmed=OLD)
        mgr._recreate_already_requested = AsyncMock(return_value=False)  # type: ignore[method-assign]
        mgr._post_recreate_comment = AsyncMock(return_value=True)  # type: ignore[method-assign]
        wait = AsyncMock()
        mgr._await_dependabot_recreate = wait  # type: ignore[method-assign]
        pr = make_pr_info(head_sha=OLD)

        with pytest.raises(ConfirmedHeadSignal, match="recreate requested"):
            await mgr._trigger_dependabot_recreate(pr, RecreateCause.STUCK_CHECK)

        mgr._post_recreate_comment.assert_awaited_once()
        wait.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_stop_is_a_skip_that_keeps_the_exit_code(self) -> None:
        from dependamerge.cli._merge_report import _exit_if_any_failed

        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(return_value=_live(NEW))
        pr = make_pr_info(head_sha=OLD)

        async def sequence(flow: Any) -> MergeResult:
            await mgr._approve_pr("o", "r", flow.pr_info)
            result: MergeResult = flow.result
            return result

        mgr._run_merge_sequence = sequence  # type: ignore[method-assign]
        result = await mgr._merge_single_pr(pr)

        assert result.status == MergeStatus.SKIPPED
        assert result.error is not None
        assert result.error.startswith("head changed since it was confirmed")
        client.approve_pull_request.assert_not_awaited()
        client.merge_pull_request.assert_not_awaited()
        _exit_if_any_failed([result])

    @pytest.mark.asyncio
    async def test_other_pull_requests_are_unaffected(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(return_value=_live(NEW))
        client.merge_pull_request = AsyncMock(return_value=True)

        other = make_pr_info(number=2, head_sha=NEW)
        assert await mgr._merge_pr_with_retry(other, "o", "r") is True
        assert _last(client.merge_pull_request).kwargs["sha"] == NEW


def _ctx(**kw: Any) -> Any:
    from dependamerge.cli._context import _MergeContext

    fields: dict[str, Any] = {"dry_run": False}
    fields.update(kw)
    ctx = MagicMock(spec=_MergeContext)
    ctx.confirmed_heads = {}
    for name, value in fields.items():
        setattr(ctx, name, value)
    return ctx


class TestRecording:
    def test_human_override_records_each_head(self) -> None:
        from dependamerge.cli._human_gate import _record_confirmed_human_heads
        from dependamerge.merge_manager import ConfirmedHead

        ctx = _ctx()
        prs: list[PullRequestInfo] = [
            make_pr_info(repository_full_name="Acme/Widget", number=4, head_sha=OLD),
            make_pr_info(repository_full_name="acme/gadget", number=5, head_sha=NEW),
        ]
        _record_confirmed_human_heads(ctx, prs)

        assert ctx.confirmed_heads == {
            ("acme/widget", 4): ConfirmedHead(OLD),
            ("acme/gadget", 5): ConfirmedHead(NEW),
        }

    def test_a_dry_run_records_nothing(self) -> None:
        from dependamerge.cli._human_gate import _record_confirmed_human_heads

        ctx = _ctx(dry_run=True)
        _record_confirmed_human_heads(ctx, [make_pr_info()])
        assert ctx.confirmed_heads == {}

    def test_a_pr_without_a_head_is_still_recorded(self) -> None:
        from dependamerge.cli import _major_gate
        from dependamerge.cli._human_gate import _record_confirmed_human_heads
        from dependamerge.merge_manager import ConfirmedHead

        ctx = _ctx()
        _record_confirmed_human_heads(ctx, [make_pr_info(number=4, head_sha="")])
        with patch.object(_major_gate, "_gate_reason", return_value="major bump"):
            _major_gate._record_confirmed_heads(
                ctx, [(make_pr_info(number=5, head_sha=""), MagicMock())]
            )

        assert ctx.confirmed_heads == {
            ("o/r", 4): ConfirmedHead(""),
            ("o/r", 5): ConfirmedHead("", "major bump"),
        }

    def test_a_major_bump_keeps_its_gate_reason(self) -> None:
        from dependamerge.cli import _major_gate
        from dependamerge.merge_manager import ConfirmedHead

        ctx = _ctx()
        pr = make_pr_info(head_sha=OLD)
        with patch.object(_major_gate, "_gate_reason", return_value="major bump"):
            _major_gate._record_confirmed_heads(ctx, [(pr, MagicMock())])

        assert ctx.confirmed_heads == {("o/r", 1): ConfirmedHead(OLD, "major bump")}

    def test_a_scope_override_records_the_human_prs(self) -> None:
        from dependamerge.cli._human_gate import (
            _confirm_human_prs_in_scope,
            _human_set_override,
        )

        prs = [make_pr_info(author="someone", head_sha=OLD)]
        ctx = _ctx(override=_human_set_override("o/r", prs))
        _confirm_human_prs_in_scope(ctx, "o/r", prs)

        assert set(ctx.confirmed_heads) == {("o/r", 1)}
        assert ctx.confirmed_heads[("o/r", 1)].sha == OLD
