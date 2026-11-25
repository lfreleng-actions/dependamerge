# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""A confirmed head holds through waits, rebases already under way, an
approval whose head keeps moving, and every reading that would accept a
merge as this run's (issue #678)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.github_async import ApprovalError, GitHubAsync
from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.merge_manager._head_pin import (
    HEAD_CHANGED_DURING_MERGE,
    ApprovalHeadMovedError,
    says_head_moved,
)
from dependamerge.merge_signals import ConfirmedHeadSignal
from tests.conftest import make_pr_info
from tests.test_commit_pinning import HEAD_MOVED_409, NEW, OLD, _live, _manager

MOVED = ApprovalError("o/r#1", 422, "Head sha moved")


def _polling(mgr: Any, *payloads: dict[str, Any]) -> None:
    mgr._merge_recheck_interval = 0
    mgr._fetch_pr_state = AsyncMock(side_effect=list(payloads))


class TestWaitsCheckTheConfirmedHead:
    @pytest.mark.asyncio
    async def test_a_merge_at_another_head_is_not_reported_as_ours(self) -> None:
        mgr, _client = _manager(confirmed=OLD)
        _polling(mgr, _live(NEW, state="closed", merged=True))

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            await mgr._wait_for_auto_merge(
                make_pr_info(head_sha=OLD, mergeable_state="blocked"),
                "o",
                "r",
                continue_states=("blocked",),
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("confirmed", "head"), [(OLD, OLD), (None, NEW)])
    async def test_a_merge_at_the_pinned_or_any_head_is_accepted(
        self, confirmed: str | None, head: str
    ) -> None:
        mgr, _client = _manager(confirmed=confirmed)
        _polling(mgr, _live(head, state="closed", merged=True))

        closed, merged = await mgr._wait_for_auto_merge(
            make_pr_info(head_sha=OLD, mergeable_state="blocked"),
            "o",
            "r",
            continue_states=("blocked",),
        )

        assert (closed, merged) == (True, True)

    def test_the_late_refresh_checks_it_too(self) -> None:
        mgr, _client = _manager(confirmed=OLD)
        flow: Any = AsyncMock()
        flow.pr_info = make_pr_info(head_sha=OLD)
        flow.result = MergeResult(pr_info=flow.pr_info, status=MergeStatus.PENDING)

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            mgr._apply_late_refresh(flow, _live(NEW, state="closed", merged=True))
        assert flow.result.status == MergeStatus.PENDING

    @pytest.mark.asyncio
    async def test_a_headless_closed_poll_is_not_accepted(self) -> None:
        mgr, _client = _manager(confirmed=OLD)
        headless = {"state": "closed", "merged": True}
        _polling(mgr, headless)

        with pytest.raises(ConfirmedHeadSignal, match="commit merged is unknown"):
            await mgr._wait_for_auto_merge(
                make_pr_info(head_sha=OLD, mergeable_state="blocked"),
                "o",
                "r",
                continue_states=("blocked",),
            )

    def test_nor_a_headless_late_refresh(self) -> None:
        mgr, _client = _manager(confirmed=OLD)
        flow: Any = AsyncMock()
        flow.pr_info = make_pr_info(head_sha=OLD)
        flow.result = MergeResult(pr_info=flow.pr_info, status=MergeStatus.PENDING)

        with pytest.raises(ConfirmedHeadSignal, match="commit merged is unknown"):
            mgr._apply_late_refresh(flow, {"state": "closed", "merged": True})


class TestMergedReadingsCheckTheConfirmedHead:
    @pytest.mark.asyncio
    async def test_the_retry_recheck(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(return_value=_live(NEW, state="closed", merged=True))

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            await mgr._recheck_pr_before_retry("o", "r", make_pr_info(head_sha=OLD), 1)

    @pytest.mark.asyncio
    async def test_the_in_progress_poll(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0
        )
        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(return_value=_live(NEW, state="closed", merged=True))

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            await mgr._await_in_progress_merge(
                "o", "r", make_pr_info(head_sha=OLD), "o/r#1"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("confirmed", "head", "status"),
        [
            (OLD, NEW, MergeStatus.SKIPPED),
            (OLD, OLD, MergeStatus.MERGED),
            (None, NEW, MergeStatus.MERGED),
        ],
    )
    async def test_failure_reconciliation(
        self, confirmed: str | None, head: str, status: MergeStatus
    ) -> None:
        mgr, client = _manager(confirmed=confirmed)
        client.get = AsyncMock(return_value=_live(head, state="closed", merged=True))
        pr = make_pr_info(head_sha=OLD)
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error="x")

        out = await mgr._confirm_failure(pr, result)

        assert out.status == status
        if status is MergeStatus.SKIPPED:
            assert out.error is not None
            assert out.error.startswith("head changed since")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("live", "status", "state"),
        [
            (_live(NEW, mergeable_state="blocked"), MergeStatus.SKIPPED, "open"),
            (_live(NEW, state="closed"), MergeStatus.CLOSED, "closed"),
            (_live(OLD, mergeable_state="blocked"), MergeStatus.FAILED, "open"),
        ],
    )
    async def test_failure_reconciliation_of_an_open_or_closed_pr(
        self, live: dict[str, Any], status: MergeStatus, state: str
    ) -> None:
        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(return_value=live)
        mgr._name_the_live_blocker = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda _pr, result, _state, _data: result
        )
        pr = make_pr_info(head_sha=OLD)
        result = MergeResult(
            pr_info=pr, status=MergeStatus.FAILED, error="x", merge_refused=True
        )

        out = await mgr._confirm_failure(pr, result)

        assert out.status == status
        assert pr.state == state

    @pytest.mark.asyncio
    async def test_the_pre_dispatch_dirty_check(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(return_value=_live(NEW, mergeable_state="dirty"))
        pr = make_pr_info(head_sha=OLD, mergeable_state="clean")

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            await mgr._is_pr_dirty_now(pr, "o", "r")
        assert (pr.head_sha, pr.mergeable_state) == (OLD, "clean")

    @pytest.mark.asyncio
    async def test_an_unconfirmed_pr_still_routes_to_conflict_recovery(
        self,
    ) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(return_value=_live(NEW, mergeable_state="dirty"))
        pr = make_pr_info(head_sha=OLD)

        assert await mgr._is_pr_dirty_now(pr, "o", "r") is True
        assert pr.head_sha == NEW

    @pytest.mark.asyncio
    async def test_the_post_failure_mergeability_refresh(self) -> None:
        mgr, _ = _manager(confirmed=OLD)
        _polling(mgr, _live(NEW, mergeable_state="dirty"))
        pr = make_pr_info(head_sha=OLD, mergeable_state="clean")

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            await mgr._refresh_pr_mergeability(pr, "o", "r")
        assert (pr.head_sha, pr.mergeable_state) == (OLD, "clean")

    @pytest.mark.asyncio
    async def test_an_unconfirmed_refresh_records_the_new_head(self) -> None:
        mgr, _ = _manager()
        _polling(mgr, _live(NEW, mergeable_state="dirty"))
        pr = make_pr_info(head_sha=OLD)

        await mgr._refresh_pr_mergeability(pr, "o", "r")
        assert (pr.head_sha, pr.mergeable_state) == (NEW, "dirty")


class TestRefreshesCheckTheConfirmedHead:
    """A refresh never records a moved confirmed head as the PR's state."""

    @staticmethod
    def _repair_flow(mgr: Any, live: dict[str, Any]) -> Any:
        mgr._trigger_stale_precommit_ci = AsyncMock(return_value=True)
        mgr._align_semantic_title = AsyncMock()
        mgr._github_client.get = AsyncMock(return_value=live)
        flow: Any = AsyncMock()
        flow.repo_owner, flow.repo_name = "o", "r"
        flow.pr_info = make_pr_info(head_sha=OLD, mergeable_state="blocked")
        return flow

    @pytest.mark.asyncio
    async def test_the_post_approval_refresh(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        mgr._post_approval_delay = 0
        client.get = AsyncMock(return_value=_live(NEW, mergeable_state="dirty"))
        pr = make_pr_info(head_sha=OLD, mergeable_state="blocked")

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            await mgr._blocked_pr_became_clean("o", "r", pr, "o/r#1")
        assert (pr.head_sha, pr.mergeable_state) == (OLD, "blocked")

    @pytest.mark.asyncio
    async def test_the_transient_405_refresh(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        client.get = AsyncMock(return_value=_live(NEW, mergeable_state="dirty"))
        pr = make_pr_info(head_sha=OLD, mergeable_state="clean")

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            await mgr._refresh_pr_mergeable("o", "r", pr, "o/r#1")
        assert (pr.head_sha, pr.mergeable_state) == (OLD, "clean")

    @pytest.mark.asyncio
    async def test_the_stale_precommit_ci_repair(self) -> None:
        mgr, _ = _manager(confirmed=OLD)
        flow = self._repair_flow(mgr, _live(NEW, mergeable_state="behind"))

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            await mgr._repair_blocked_pr(flow)
        assert (flow.pr_info.head_sha, flow.pr_info.mergeable_state) == (
            OLD,
            "blocked",
        )

    @pytest.mark.asyncio
    async def test_the_required_workflows_refresh(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        mgr._wait_for_auto_merge = AsyncMock(return_value=(False, False))
        client.get = AsyncMock(return_value=_live(NEW, mergeable_state="blocked"))
        client.merge_pull_request = AsyncMock(return_value=True)
        pr = make_pr_info(head_sha=OLD, mergeable_state="clean")

        with pytest.raises(ConfirmedHeadSignal, match="head changed since"):
            await mgr._wait_for_required_workflows_and_retry(pr, "o", "r")
        assert (pr.head_sha, pr.mergeable_state) == (OLD, "clean")
        mgr._wait_for_auto_merge.assert_not_called()
        client.merge_pull_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unconfirmed_repair_records_the_new_head(self) -> None:
        mgr, _ = _manager()
        flow = self._repair_flow(mgr, _live(NEW, mergeable_state="behind"))

        await mgr._repair_blocked_pr(flow)
        assert (flow.pr_info.head_sha, flow.pr_info.mergeable_state) == (
            NEW,
            "behind",
        )


class TestMergeRecoveryNamesTheCommit:
    @staticmethod
    def _client(after: dict[str, Any]) -> GitHubAsync:
        api = GitHubAsync(token="t")
        error = Exception(HEAD_MOVED_409)
        api.put = AsyncMock(side_effect=error)  # type: ignore[method-assign]
        api.get = AsyncMock(return_value=after)  # type: ignore[method-assign]
        return api

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "after",
        [
            _live(NEW, state="closed", merged=True),
            {"state": "closed", "merged": True},
        ],
    )
    async def test_a_merge_at_another_or_unknown_head_stays_refused(
        self, after: dict[str, Any]
    ) -> None:
        api = self._client(after)

        with pytest.raises(Exception, match="merged at") as info:
            await api.merge_pull_request("o", "r", 1, "merge", sha=OLD)
        assert says_head_moved(str(info.value))

    @pytest.mark.asyncio
    async def test_a_merge_at_the_named_head_succeeds(self) -> None:
        api = self._client(_live(OLD, state="closed", merged=True))

        assert await api.merge_pull_request("o", "r", 1, "merge", sha=OLD)


class TestAlreadyRebasing:
    @pytest.mark.asyncio
    async def test_no_wait_stops_before_approving(self) -> None:
        mgr, client = _manager(confirmed=OLD)
        pr = make_pr_info(head_sha=OLD)

        with pytest.raises(ConfirmedHeadSignal, match="rebase requested"):
            await mgr._handle_conflict_no_wait(
                pr, "o", "r", MergeResult(pr_info=pr, status=MergeStatus.PENDING), True
            )

        client.approve_pull_request.assert_not_awaited()
        client.enable_auto_merge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_waiting_path_stops_before_waiting(self) -> None:
        mgr, _client = _manager(confirmed=OLD)

        with pytest.raises(ConfirmedHeadSignal, match="rebase requested"):
            await mgr._ensure_rebase_requested(
                make_pr_info(head_sha=OLD), "o", "r", True
            )

    @pytest.mark.asyncio
    async def test_an_unconfirmed_pr_waits_as_before(self) -> None:
        mgr, _client = _manager()

        assert await mgr._ensure_rebase_requested(make_pr_info(), "o", "r", True)


class TestApprovalHeadKeepsMoving:
    @pytest.mark.asyncio
    async def test_a_second_refusal_is_a_head_change(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(side_effect=[_live(OLD), _live(NEW)])
        client.approve_pull_request = AsyncMock(side_effect=[MOVED, MOVED])

        with pytest.raises(ApprovalHeadMovedError):
            await mgr._approve_pr("o", "r", make_pr_info(head_sha=OLD))
        assert client.approve_pull_request.await_count == 2

    @pytest.mark.asyncio
    async def test_another_refusal_on_the_retry_is_kept(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(side_effect=[_live(OLD), _live(NEW)])
        other = ApprovalError("o/r#1", 422, "Reviews may only be requested")
        client.approve_pull_request = AsyncMock(side_effect=[MOVED, other])

        with pytest.raises(ApprovalError) as info:
            await mgr._approve_pr("o", "r", make_pr_info(head_sha=OLD))
        assert info.value is other

    @pytest.mark.asyncio
    async def test_it_is_reported_as_a_refusal(self) -> None:
        mgr, client = _manager()
        client.get = AsyncMock(side_effect=[_live(OLD), _live(NEW)])
        client.approve_pull_request = AsyncMock(side_effect=[MOVED, MOVED])
        mgr._confirm_failure = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda _pr, result: result
        )

        async def sequence(flow: Any) -> MergeResult:
            await mgr._approve_pr("o", "r", flow.pr_info)
            result: MergeResult = flow.result
            return result

        mgr._run_merge_sequence = sequence  # type: ignore[method-assign]
        result = await mgr._merge_single_pr(make_pr_info(head_sha=OLD))

        assert result.status == MergeStatus.FAILED
        assert result.error == HEAD_CHANGED_DURING_MERGE
        assert result.merge_refused is True
        client.merge_pull_request.assert_not_awaited()

    def test_after_a_conflict_rebase_too(self) -> None:
        mgr, _client = _manager()
        pr = make_pr_info(head_sha=OLD)
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)

        mgr._report_rebase_approval_failure(
            pr, "o", "r", result, ApprovalHeadMovedError("o/r#1", 422, "Head sha moved")
        )

        assert result.error == HEAD_CHANGED_DURING_MERGE
        assert result.merge_refused is True
