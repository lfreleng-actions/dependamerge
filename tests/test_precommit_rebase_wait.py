# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Step 5.5 waits for an asynchronous rebase this run requested.

The block reason is read before Step 5.  When that reason is a failing
check on the old head, it will not clear on its own, so the wait used
to be skipped and the merge was refused moments after the rebase was
asked for.  The rebase is what clears it, so the wait now bridges it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dependamerge import rebase as rebase_module
from dependamerge.merge_manager import AsyncMergeManager, MergeResult, MergeStatus
from dependamerge.merge_manager._head_pin import merge_head
from dependamerge.merge_manager._single_pr_context import _MergeFlow
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

OLD_HEAD = "0123456789abcdef0123456789abcdef01234567"
NEW_HEAD = "fedcba9876543210fedcba9876543210fedcba98"
LIVE_HEAD = "5555555555555555555555555555555555555555"
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
        rebase_requested=True,
    )


def _manager(**overrides: object) -> tuple[AsyncMergeManager, AsyncMock]:
    mgr, client = make_merge_manager(preview_mode=False, **overrides)
    mgr._no_wait = False
    return mgr, client


class TestTheWaitDecision:
    def test_a_requested_rebase_is_waited_for(self) -> None:
        mgr, _ = _manager()

        assert mgr._should_wait_for_checks(_flow(_pr())) is True

    def test_without_a_rebase_the_failing_check_still_skips_the_wait(self) -> None:
        mgr, _ = _manager()
        flow = _flow(_pr())
        flow.needs_rebase = False
        flow.rebase_requested = False

        assert mgr._should_wait_for_checks(flow) is False

    def test_a_rebase_that_never_went_out_is_not_waited_for(self) -> None:
        mgr, _ = _manager()
        flow = _flow(_pr())
        flow.rebase_requested = False

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

    @pytest.mark.parametrize("state", ["unknown", None, "", "clean"])
    def test_a_requested_rebase_is_waited_for_in_any_state(
        self, state: str | None
    ) -> None:
        mgr, _ = _manager()
        pr = _pr()
        pr.head_sha, pr.mergeable, pr.mergeable_state = NEW_HEAD, None, state
        flow = _flow(pr)
        flow.rebase_from_head = OLD_HEAD

        assert mgr._should_wait_for_checks(flow) is True

    @pytest.mark.parametrize(
        "guard",
        ["preview", "force_all", "already_rebased", "suppressed", "no_wait"],
    )
    def test_an_unread_state_keeps_the_other_guards(self, guard: str) -> None:
        mgr, _ = _manager()
        pr = _pr()
        pr.mergeable, pr.mergeable_state = None, "unknown"
        flow = _flow(pr)
        if guard == "preview":
            mgr.preview_mode = True
        elif guard == "force_all":
            mgr.force_level = "all"
        elif guard == "already_rebased":
            flow.already_rebased = True
        elif guard == "suppressed":
            mgr._rebase_suppressed[flow.pr_key] = "rebase requested 10m ago"
        else:
            mgr._no_wait = True

        assert mgr._should_wait_for_checks(flow) is False


class TestTheMergeAfterTheWait:
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

    @pytest.mark.asyncio
    async def test_the_old_head_going_clean_does_not_end_the_wait(self) -> None:
        mgr, client = _manager(merge_timeout=60)
        pr = _pr()
        flow = _flow(pr)
        flow.rebase_from_head = OLD_HEAD
        client.get = AsyncMock(
            side_effect=[
                _payload(OLD_HEAD, "clean", mergeable=True),
                _payload(OLD_HEAD, "unstable", mergeable=True),
                _payload(OLD_HEAD, "dirty", mergeable=False),
                _payload(NEW_HEAD, "clean", mergeable=True),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            assert await mgr._await_auto_merge(flow) is None

        assert client.get.await_count == 4
        assert await merge_head(mgr, pr, "o", "r") == NEW_HEAD

    @pytest.mark.asyncio
    async def test_the_new_head_unread_does_not_inherit_the_old_clean(self) -> None:
        mgr, client = _manager(merge_timeout=60)
        pr = _pr()
        flow = _flow(pr)
        flow.rebase_from_head = OLD_HEAD
        unread = {**_payload(NEW_HEAD, "unknown", mergeable=True), "mergeable": None}
        client.get = AsyncMock(
            side_effect=[
                _payload(OLD_HEAD, "clean", mergeable=True),
                unread,
                unread,
                _payload(NEW_HEAD, "blocked", mergeable=False),
                _payload(NEW_HEAD, "clean", mergeable=True),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            assert await mgr._await_auto_merge(flow) is None

        assert client.get.await_count == 5
        assert (pr.head_sha, pr.mergeable_state) == (NEW_HEAD, "clean")

    @pytest.mark.asyncio
    async def test_a_new_head_still_unread_at_the_deadline_is_not_clean(
        self,
    ) -> None:
        mgr, client = _manager(merge_timeout=60)
        _expire_quickly(mgr)
        pr = _pr()
        flow = _flow(pr)
        flow.rebase_from_head = OLD_HEAD
        unread = {**_payload(NEW_HEAD, "unknown", mergeable=True), "mergeable": None}
        client.get = AsyncMock(
            side_effect=[_payload(OLD_HEAD, "clean", mergeable=True)] + [unread] * 1000
        )

        assert await mgr._await_auto_merge(flow) is None

        assert (pr.head_sha, pr.mergeable, pr.mergeable_state) == (
            NEW_HEAD,
            None,
            "unknown",
        )

    @pytest.mark.asyncio
    async def test_a_rebase_that_never_lands_is_not_merged_at_the_deadline(
        self,
    ) -> None:
        mgr, client = _manager(merge_timeout=60)
        _expire_quickly(mgr)
        pr = _pr()
        flow = _flow(pr)
        client.get = AsyncMock(return_value=_payload(OLD_HEAD, "clean", mergeable=True))
        merge = AsyncMock(return_value=True)

        with (
            patch.object(mgr, "_merge_pr_with_retry", merge),
            patch.object(mgr, "_enable_auto_merge_with_approval", AsyncMock()),
        ):
            result = await _run_sequence(mgr, flow)

        assert client.get.await_count > 1
        merge.assert_not_awaited()
        assert result.status is MergeStatus.FAILED
        assert result.error is not None and OLD_HEAD[:7] in result.error
        assert pr.head_sha == OLD_HEAD

    @pytest.mark.asyncio
    async def test_armed_auto_merge_reports_pending_when_it_never_lands(
        self,
    ) -> None:
        mgr, client = _manager(merge_timeout=60)
        _expire_quickly(mgr)
        pr = _pr()
        flow = _flow(pr)
        flow.rebase_from_head = OLD_HEAD
        mgr._auto_merge_enabled.add(flow.pr_key)
        client.get = AsyncMock(return_value=_payload(OLD_HEAD, "clean", mergeable=True))

        result = await mgr._await_auto_merge(flow)

        assert result is not None
        assert result.status is MergeStatus.AUTO_MERGE_PENDING

    @pytest.mark.asyncio
    async def test_a_pr_merged_before_the_rebase_lands_ends_the_wait(self) -> None:
        mgr, client = _manager(merge_timeout=60)
        pr = _pr()
        flow = _flow(pr)
        flow.rebase_from_head = OLD_HEAD
        merged = {**_payload(OLD_HEAD, "clean", mergeable=True)}
        merged.update(state="closed", merged=True)
        client.get = AsyncMock(return_value=merged)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            assert await mgr._wait_through_rebase(flow) == (True, True)

        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_a_step5_reading_of_the_unread_new_head_still_waits(
        self,
    ) -> None:
        mgr, client = _manager(merge_timeout=60)
        pr = _pr()
        flow = _flow(pr)
        unread = {**_payload(NEW_HEAD, "unknown", mergeable=True), "mergeable": None}
        readings = [
            unread,
            _payload(NEW_HEAD, "blocked", mergeable=False),
            _payload(NEW_HEAD, "clean", mergeable=True),
        ]
        pr_reads = 0

        async def get(url: str, **_: object) -> object:
            nonlocal pr_reads
            if not url.endswith("/pulls/11"):
                return []
            pr_reads += 1
            return readings[min(pr_reads, len(readings)) - 1]

        client.get = AsyncMock(side_effect=get)
        seen: list[tuple[str | None, str | None]] = []

        async def merge(pr_info: PullRequestInfo, *_: object) -> bool:
            seen.append((pr_info.head_sha, pr_info.mergeable_state))
            return True

        async def step5(flow: _MergeFlow) -> None:
            # The approval after the macro reads the rebased head before
            # GitHub has computed its mergeability.
            flow.rebase_requested = True
            flow.rebase_from_head = OLD_HEAD
            flow.pr_info.head_sha = NEW_HEAD
            flow.pr_info.mergeable = None
            flow.pr_info.mergeable_state = "unknown"

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch.object(mgr, "_merge_pr_with_retry", AsyncMock(side_effect=merge)),
            patch.object(mgr, "_enable_auto_merge_with_approval", AsyncMock()),
        ):
            await _run_sequence(mgr, flow, step5=step5)

        assert pr_reads >= len(readings)
        assert seen == [(NEW_HEAD, "clean")]


def _expire_quickly(mgr: AsyncMergeManager) -> None:
    mgr._merge_recheck_interval = 0.01
    mgr._merge_timeout = 0.05


async def _run_sequence(
    mgr: AsyncMergeManager,
    flow: _MergeFlow,
    *,
    step5: Callable[[_MergeFlow], Awaitable[None]] | None = None,
) -> MergeResult:
    """Steps 5 to 6, the gates before them passing and Step 5 asking."""

    async def asking(flow: _MergeFlow) -> None:
        flow.rebase_requested = True
        flow.rebase_from_head = flow.pr_info.head_sha

    flow.rebase_requested = False
    passing = AsyncMock(return_value=None)
    with (
        patch.object(mgr, "_gate_github2gerrit", passing),
        patch.object(mgr, "_gate_pr_still_open", passing),
        patch("dependamerge.merge_manager._single_pr.check_confirmed_pr", passing),
        patch.object(mgr, "_gate_mergeable_state", passing),
        patch.object(mgr, "_repair_blocked_pr", passing),
        patch.object(mgr, "_gate_merge_requirements", passing),
        patch.object(mgr, "_gate_copilot_reviews", passing),
        patch.object(mgr, "_analyze_blocked_state", passing),
        patch.object(mgr, "_pr_needs_rebase", AsyncMock(return_value=True)),
        patch.object(mgr, "_run_step5_rebase", step5 or asking),
    ):
        return await mgr._run_merge_sequence(flow)


def _payload(head: str, state: str, *, mergeable: bool) -> dict[str, object]:
    return {
        "state": "open",
        "merged": False,
        "draft": False,
        "mergeable": mergeable,
        "mergeable_state": state,
        "head": {"sha": head},
    }


def _rebase_ctx(**overrides: object) -> rebase_module.RebaseContext:
    fields: dict[str, object] = {
        "github_client": AsyncMock(),
        "token": "t",
        "host": "github.com",
        "rebase_local": True,
        "preview_mode": False,
        "merge_recheck_interval": 0.0,
        "merge_poll_max_attempts": 1,
        "log": logging.getLogger("test"),
        "console": MagicMock(),
        "rebased_prs": set(),
        "enable_auto_merge": AsyncMock(return_value=False),
        **overrides,
    }
    return rebase_module.RebaseContext(**fields)  # type: ignore[arg-type]


async def _step5(ctx: rebase_module.RebaseContext) -> rebase_module.Step5Outcome:
    with (
        patch(
            "dependamerge.rebase.dispatch.should_use_local_rebase",
            AsyncMock(return_value=(True, "signatures required")),
        ),
        patch("dependamerge.rebase.dispatch.in_github_actions", return_value=False),
    ):
        return await rebase_module.perform_step5_rebase(
            ctx=ctx, pr_info=_pr(), owner="o", repo="r"
        )


class TestStep5ReportsWhetherARebaseIsComing:
    @pytest.mark.parametrize("pushed", [True, False])
    @pytest.mark.asyncio
    async def test_a_local_rebase_counts_only_when_it_pushed(
        self, pushed: bool
    ) -> None:
        with patch.object(
            rebase_module.local_rebase,
            "local_rebase_pr",
            new=AsyncMock(return_value=pushed),
        ):
            outcome = await _step5(_rebase_ctx())

        assert not outcome.failed
        assert outcome.rebase_requested is pushed

    @pytest.mark.asyncio
    async def test_a_posted_macro_counts(self) -> None:
        ctx = _rebase_ctx(request_dependabot_rebase=AsyncMock(return_value=True))

        assert (await _step5(ctx)).rebase_requested is True

    @pytest.mark.asyncio
    async def test_a_macro_the_cooldown_held_back_does_not_count(self) -> None:
        ctx = _rebase_ctx(
            request_dependabot_rebase=AsyncMock(return_value=False),
            suppressed_rebases={"o/r#11": "rebase requested 10m ago"},
        )

        assert (await _step5(ctx)).rebase_requested is False

    @pytest.mark.parametrize("requested", [True, False])
    @pytest.mark.asyncio
    async def test_the_flow_waits_only_for_a_rebase_that_went_out(
        self, requested: bool
    ) -> None:
        mgr, _ = _manager()
        flow = _flow(_pr())
        flow.rebase_requested = False
        outcome = rebase_module.Step5Outcome(rebase_requested=requested)

        async def rebase_moving_the_snapshot(**kwargs: object) -> object:
            kwargs["pr_info"].head_sha = NEW_HEAD  # type: ignore[attr-defined]
            return outcome

        with patch.object(
            rebase_module, "perform_step5_rebase", rebase_moving_the_snapshot
        ):
            assert await mgr._run_step5_rebase(flow) is None

        assert mgr._should_wait_for_checks(flow) is requested
        assert flow.rebase_from_head == (OLD_HEAD if requested else None)

    @pytest.mark.asyncio
    async def test_the_baseline_is_the_live_head_not_the_snapshot(self) -> None:
        mgr, client = _manager(merge_timeout=60)
        _expire_quickly(mgr)
        pr = _pr()
        flow = _flow(pr)
        flow.rebase_requested = False
        client.get = AsyncMock(
            return_value=_payload(LIVE_HEAD, "clean", mergeable=True)
        )

        async def approval_reading_the_live_head(**kwargs: object) -> object:
            kwargs["pr_info"].head_sha = LIVE_HEAD  # type: ignore[attr-defined]
            return rebase_module.Step5Outcome(rebase_requested=True)

        with patch.object(
            rebase_module, "perform_step5_rebase", approval_reading_the_live_head
        ):
            assert await mgr._run_step5_rebase(flow) is None

        assert flow.rebase_from_head == LIVE_HEAD
        result = await mgr._await_auto_merge(flow)

        assert result is not None
        assert result.status is MergeStatus.FAILED
        assert result.error is not None and LIVE_HEAD[:7] in result.error
