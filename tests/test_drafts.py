# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Draft pull requests are recognised, set aside, and never acted on.

Regression cover for lfreleng-actions/dependamerge#566.  GitHub retired
the ``DRAFT`` merge state: both APIs report a draft as ``blocked`` (or
``dirty``) like any other PR, and only GraphQL ``isDraft`` and REST
``draft`` say it is one.  Every draft check compared the merge state, so
none could fire and drafts entered the merge pipeline.  The fixtures
here use the states GitHub actually reports for drafts.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dependamerge.cli._close import _validate_close_authorization
from dependamerge.cli._drafts import DRAFT_REASON, _set_aside_drafts
from dependamerge.cli._merge_inputs import _fetch_and_validate_source_pr
from dependamerge.error_codes import ExitCode
from dependamerge.github_client.queries import _build_pull_request_info
from dependamerge.github_service import GitHubService
from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.merge_signals import DraftPullRequestError, DraftStateUnknownError
from dependamerge.models import PullRequestInfo
from dependamerge.pr_poller import _to_rest_shape
from tests.conftest import make_merge_manager


def _pr(number: int, *, is_draft: bool, state: str = "blocked") -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="t",
        body=None,
        author="dependabot[bot]",
        head_sha="abc",
        base_branch="main",
        head_branch="topic",
        state="open",
        mergeable=True,
        mergeable_state=state,
        is_draft=is_draft,
        behind_by=0,
        files_changed=[],
        repository_full_name="acme/widget",
        html_url=f"https://github.com/acme/widget/pull/{number}",
    )


class TestRecognition:
    """Drafts are recognised by their flag, whatever their merge state."""

    @pytest.mark.parametrize("merge_state", ["BLOCKED", "DIRTY"])
    @pytest.mark.parametrize("is_draft", [True, False])
    def test_graphql_node(self, merge_state: str, is_draft: bool) -> None:
        node: dict[str, Any] = {
            "number": 7,
            "isDraft": is_draft,
            "mergeStateStatus": merge_state,
            "author": {"__typename": "Bot", "login": "dependabot"},
        }
        pr = GitHubService(token="t").to_pull_request_info("acme/widget", node)
        assert pr.is_draft is is_draft

    @pytest.mark.parametrize("is_draft", [True, False])
    def test_rest_payload(self, is_draft: bool) -> None:
        payload = {"number": 7, "draft": is_draft, "mergeable_state": "blocked"}
        pr = _build_pull_request_info(payload, "acme", "widget", 7, [], [])
        assert pr.is_draft is is_draft

    def test_batched_poller_reports_the_flag(self) -> None:
        shape = _to_rest_shape(
            {"number": 7, "state": "OPEN", "isDraft": True, "mergeStateStatus": "DIRTY"}
        )
        assert shape["draft"] is True
        assert shape["mergeable_state"] == "dirty"


class TestSetAside:
    def test_drafts_are_listed_recorded_and_removed(self, capsys) -> None:
        ctx = MagicMock()
        ctx.draft_results = []
        kept = _set_aside_drafts(
            ctx, [_pr(1, is_draft=False), _pr(2, is_draft=True, state="dirty")]
        )
        assert [pr.number for pr in kept] == [1]
        assert "Draft PRs skipped: 1" in capsys.readouterr().out
        [recorded] = ctx.draft_results
        assert recorded.status is MergeStatus.SKIPPED
        assert recorded.error == DRAFT_REASON
        assert recorded.pr_info.number == 2

    def test_no_drafts_prints_nothing(self, capsys) -> None:
        ctx = MagicMock()
        ctx.draft_results = []
        prs = [_pr(1, is_draft=False)]
        assert _set_aside_drafts(ctx, prs) == prs
        assert capsys.readouterr().out == ""


class TestSourcePullRequest:
    def test_merge_stops_at_a_draft_source(self) -> None:
        ctx = MagicMock()
        ctx.progress_tracker = None
        ctx.pr_number = 7
        ctx.github_client.get_pull_request_info.return_value = _pr(7, is_draft=True)
        with pytest.raises(SystemExit) as raised:
            _fetch_and_validate_source_pr(ctx)
        assert raised.value.code == ExitCode.PR_STATE_ERROR

    def test_close_stops_at_a_draft_source(self) -> None:
        ctx = MagicMock()
        ctx.pr_number = 7
        ctx.source_pr = _pr(7, is_draft=True)
        with pytest.raises(SystemExit) as raised:
            _validate_close_authorization(
                ctx, None, include_human_prs=False, dry_run=False
            )
        assert raised.value.code == ExitCode.PR_STATE_ERROR
        ctx.github_client.is_automation_author.assert_not_called()


class TestMergeManager:
    @pytest.mark.asyncio
    async def test_draft_is_skipped_on_entry_without_api_calls(self) -> None:
        mgr, client = make_merge_manager()
        result = await mgr._merge_single_pr_impl(_pr(1, is_draft=True))
        assert result.status is MergeStatus.SKIPPED
        assert result.error == DRAFT_REASON
        assert not client.mock_calls

    def test_draft_is_not_worth_attempting(self) -> None:
        mgr, _client = make_merge_manager()
        assert not mgr._is_pr_mergeable(_pr(1, is_draft=True))
        assert mgr._is_pr_mergeable(_pr(1, is_draft=False))

    @pytest.mark.asyncio
    async def test_a_pr_turned_draft_is_never_approved(self) -> None:
        mgr, _client = make_merge_manager()
        mgr._approve_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
        with pytest.raises(DraftPullRequestError):
            await mgr._ensure_pr_approved(_pr(1, is_draft=True), "acme", "widget")
        mgr._approve_pr.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_conversion_while_queued_stops_approval(self) -> None:
        # The snapshot says ready; the live read at the one place every
        # approval passes says draft (Copilot review on #567).
        mgr, client = make_merge_manager()
        mgr._fetch_pr_state = AsyncMock(  # type: ignore[method-assign]
            return_value={"state": "open", "draft": True, "mergeable_state": "blocked"}
        )
        with pytest.raises(DraftPullRequestError):
            await mgr._approve_pr("acme", "widget", 1)
        client.approve_pull_request.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_handle_conflict_no_wait", ("acme", "widget", None, True)),
            ("_complete_after_rebase", ("acme", "widget", None, 0.0)),
        ],
    )
    async def test_recovery_approvals_pass_the_conversion_on(
        self, method: str, args: tuple[Any, ...]
    ) -> None:
        # Recovery paths catch approval failures broadly; a conversion to
        # draft must still reach the skip handler, not be logged away.
        mgr, _client = make_merge_manager()
        mgr._approve_pr = AsyncMock(  # type: ignore[method-assign]
            side_effect=DraftPullRequestError("acme/widget#1")
        )
        mgr._request_dependabot_rebase = AsyncMock(return_value=True)  # type: ignore[method-assign]
        owner, repo, _, last = args
        result = MergeResult(pr_info=_pr(1, is_draft=False), status=MergeStatus.PENDING)
        with pytest.raises(DraftPullRequestError):
            await getattr(mgr, method)(
                _pr(1, is_draft=False), owner, repo, result, last
            )

    @pytest.mark.asyncio
    async def test_auto_merge_approval_passes_the_conversion_on(self) -> None:
        mgr, _client = make_merge_manager()
        mgr._ensure_pr_approved = AsyncMock(  # type: ignore[method-assign]
            side_effect=DraftPullRequestError("acme/widget#1")
        )
        mgr._enable_auto_merge_for_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
        with pytest.raises(DraftPullRequestError):
            await mgr._enable_auto_merge_with_approval(
                _pr(1, is_draft=False), "acme", "widget"
            )
        mgr._enable_auto_merge_for_pr.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_conversion_ends_the_pr_as_skipped(self) -> None:
        mgr, _client = make_merge_manager()
        pr = _pr(1, is_draft=False)
        mgr._get_merge_method_for_repo = AsyncMock(return_value="merge")  # type: ignore[method-assign]
        mgr._run_merge_sequence = AsyncMock(  # type: ignore[method-assign]
            side_effect=DraftPullRequestError("acme/widget#1")
        )
        result = await mgr._merge_single_pr_impl(pr)
        assert result.status is MergeStatus.SKIPPED
        assert result.error == DRAFT_REASON

    @pytest.mark.asyncio
    async def test_the_pre_dispatch_read_records_a_conversion(self) -> None:
        mgr, client = make_merge_manager()
        client.get = AsyncMock(
            return_value={"state": "open", "draft": True, "mergeable_state": "blocked"}
        )
        pr = _pr(1, is_draft=False)
        assert await mgr._is_pr_dirty_now(pr, "acme", "widget") is False
        assert pr.is_draft is True

    @pytest.mark.asyncio
    async def test_refresh_picks_up_a_conversion_to_draft(self) -> None:
        mgr, _client = make_merge_manager()
        pr = _pr(1, is_draft=False)
        mgr._fetch_pr_state = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "mergeable": True,
                "mergeable_state": "blocked",
                "draft": True,
            }
        )
        with pytest.raises(DraftPullRequestError):
            await mgr._refresh_pr_mergeable("acme", "widget", pr, "acme/widget#1")
        assert pr.is_draft is True

    @pytest.mark.asyncio
    async def test_a_transient_405_retry_skips_a_new_draft(self, monkeypatch) -> None:
        # A clean PR answered with a 405, then found a draft by the
        # retry's refresh, must not get a second merge dispatch (Copilot
        # review on #567); through the real retry loop.
        import dependamerge.merge_manager._retry as retry_module

        monkeypatch.setattr(retry_module.asyncio, "sleep", AsyncMock())
        mgr, client = make_merge_manager()
        refused = RuntimeError("405 Method Not Allowed: Pull Request is not mergeable")
        client.merge_pull_request = AsyncMock(side_effect=refused)
        mgr._fetch_pr_state = AsyncMock(  # type: ignore[method-assign]
            return_value={"mergeable": True, "mergeable_state": "clean", "draft": True}
        )
        pr = _pr(1, is_draft=False, state="clean")
        with pytest.raises(DraftPullRequestError):
            await mgr._merge_pr_with_retry(pr, "acme", "widget")
        assert client.merge_pull_request.await_count == 1

        mgr._get_merge_method_for_repo = AsyncMock(return_value="merge")  # type: ignore[method-assign]
        mgr._run_merge_sequence = AsyncMock(  # type: ignore[method-assign]
            side_effect=DraftPullRequestError("acme/widget#1")
        )
        result = await mgr._merge_single_pr_impl(
            pr.model_copy(update={"is_draft": False})
        )
        assert result.status is MergeStatus.SKIPPED


class TestDraftStateUnknown:
    """An approval waits for an explicit "not a draft" (Copilot on #567).

    Approving a draft satisfies required reviews once it is marked
    ready, and refusing the later merge cannot undo that, so a failed or
    incomplete live read stops the approval instead.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reading",
        [
            AsyncMock(side_effect=RuntimeError("read failed")),
            AsyncMock(return_value={"state": "open"}),
            AsyncMock(return_value=None),
        ],
    )
    async def test_an_inconclusive_read_writes_no_approval(self, reading) -> None:
        mgr, client = make_merge_manager()
        mgr._fetch_pr_state = reading  # type: ignore[method-assign]
        with pytest.raises(DraftStateUnknownError):
            await mgr._approve_pr("acme", "widget", 1)
        client.approve_pull_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_explicit_ready_read_approves(self) -> None:
        mgr, client = make_merge_manager()
        mgr._fetch_pr_state = AsyncMock(return_value={"draft": False})  # type: ignore[method-assign]
        mgr._should_skip_approval = AsyncMock(return_value=(False, None))  # type: ignore[method-assign]
        assert await mgr._approve_pr("acme", "widget", 1) is True
        client.approve_pull_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_the_pr_fails_without_an_approval(self) -> None:
        mgr, _client = make_merge_manager()
        mgr._get_merge_method_for_repo = AsyncMock(return_value="merge")  # type: ignore[method-assign]
        mgr._run_merge_sequence = AsyncMock(  # type: ignore[method-assign]
            side_effect=DraftStateUnknownError("acme/widget#1")
        )
        result = await mgr._merge_single_pr_impl(_pr(1, is_draft=False))
        assert result.status is MergeStatus.FAILED
        assert result.error is not None
        assert "not a draft" in result.error


class TestStepFiveRebase:
    """The draft signal survives the Step 5 rebase handlers (Copilot on #567).

    Each path catches the auto-merge callback's failures to keep the
    rebase going; a signal must still reach the single-PR handler, or a
    draft would be reported as a failed rebase (exit 7) and not skipped.
    """

    def _ctx(self, signal: Exception | None = None, reading: Any = None):
        from dependamerge.rebase.context import RebaseContext

        client = AsyncMock()
        client.update_branch = AsyncMock(return_value=None)
        client.get = AsyncMock(return_value=reading)
        return RebaseContext(
            github_client=client,
            token="t",
            host="github.com",
            rebase_local=False,
            preview_mode=False,
            merge_recheck_interval=0.0,
            merge_poll_max_attempts=1,
            log=MagicMock(),
            console=MagicMock(),
            rebased_prs=set(),
            enable_auto_merge=AsyncMock(side_effect=signal, return_value=True),
            request_dependabot_rebase=AsyncMock(return_value=True),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "signal",
        [
            DraftPullRequestError("acme/widget#1"),
            DraftStateUnknownError("acme/widget#1"),
        ],
    )
    async def test_the_rest_path_passes_the_signal_on(self, signal) -> None:
        from dependamerge.rebase.paths import _run_rest_path

        with pytest.raises(type(signal)):
            await _run_rest_path(
                ctx=self._ctx(signal),
                pr_info=_pr(1, is_draft=False),
                owner="acme",
                repo="widget",
            )

    @pytest.mark.asyncio
    async def test_the_dependabot_macro_path_passes_the_signal_on(self) -> None:
        from dependamerge.rebase.paths import _run_dependabot_macro_path

        signal = DraftPullRequestError("acme/widget#1")
        with pytest.raises(DraftPullRequestError):
            await _run_dependabot_macro_path(
                ctx=self._ctx(signal),
                pr_info=_pr(1, is_draft=False),
                owner="acme",
                repo="widget",
                local_reason="test",
            )

    @pytest.mark.asyncio
    async def test_the_rest_path_poll_stops_at_a_new_draft(self) -> None:
        # Marked draft while the REST rebase settles: the poll stops before
        # the PR is recorded as rebased, after which Step 6 would defer it
        # to auto-merge as pending without reading it again.
        from dependamerge.rebase.paths import _run_rest_path

        ctx = self._ctx(
            reading={"state": "open", "draft": True, "mergeable_state": "blocked"}
        )
        pr = _pr(1, is_draft=False)
        with pytest.raises(DraftPullRequestError):
            await _run_rest_path(ctx=ctx, pr_info=pr, owner="acme", repo="widget")
        assert pr.is_draft is True
        assert not ctx.rebased_prs


class TestRecoveryAndPolling:
    """Stop signals survive merge recovery, and polling keeps unknowns (#567)."""

    @pytest.mark.asyncio
    async def test_a_dirty_draft_is_skipped_by_the_recovery_dispatch(self) -> None:
        # A live PR can be both dirty and a draft; the recovery dispatch
        # must skip it rather than route it to conflict recovery.
        mgr, client = make_merge_manager(repo_scoped=True)
        client.get = AsyncMock(
            return_value={"state": "open", "draft": True, "mergeable_state": "dirty"}
        )
        mgr._merge_pr_with_retry = AsyncMock(return_value=True)  # type: ignore[method-assign]
        with pytest.raises(DraftPullRequestError):
            await mgr._retry_merge_under_dispatch_lock(
                _pr(1, is_draft=False), "acme", "widget"
            )
        mgr._merge_pr_with_retry.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "signal",
        [
            DraftPullRequestError("acme/widget#1"),
            DraftStateUnknownError("acme/widget#1"),
        ],
    )
    async def test_the_merge_retry_loop_passes_signals_on(self, signal) -> None:
        # A false merge's recovery reaches the auto-merge approval, whose
        # signal the retry loop must not treat as a retryable error.
        mgr, client = make_merge_manager()
        client.merge_pull_request = AsyncMock(return_value=False)
        mgr._retry_after_false_merge = AsyncMock(side_effect=signal)  # type: ignore[method-assign]
        with pytest.raises(type(signal)):
            await mgr._merge_pr_with_retry(_pr(1, is_draft=False), "acme", "widget")

    @pytest.mark.asyncio
    async def test_the_rebased_merge_reads_the_draft_flag_in_its_lock(self) -> None:
        # Conflict recovery's own dispatch, once the rebase cleared the
        # conflict: a PR marked draft since the wait's last poll is not
        # sent a merge, as at every other dispatch (Copilot review on #567).
        mgr, _client = make_merge_manager()
        mgr._fetch_pr_state = AsyncMock(  # type: ignore[method-assign]
            return_value={"state": "open", "draft": True}
        )
        mgr._merge_pr_with_retry = AsyncMock(return_value=True)  # type: ignore[method-assign]
        pr = _pr(1, is_draft=False, state="clean")
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        with pytest.raises(DraftPullRequestError):
            await mgr._merge_rebased_pr(pr, "acme", "widget", result)
        mgr._merge_pr_with_retry.assert_not_called()
        assert pr.is_draft is True

    @pytest.mark.asyncio
    async def test_an_incomplete_poll_leaves_the_draft_state_unknown(self) -> None:
        # Through the real batched poller: a node with no isDraft must not
        # read as "not a draft", or the approval would go ahead.
        from dependamerge.pr_poller import PullRequestStatePoller

        api = AsyncMock()
        node = {
            "number": 1,
            "state": "OPEN",
            "merged": False,
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "BLOCKED",
        }
        api.graphql = AsyncMock(return_value={"p0": {"pullRequest": node}})
        mgr, client = make_merge_manager()
        mgr._pr_poller = PullRequestStatePoller(api, window=0.0)
        with pytest.raises(DraftStateUnknownError):
            await mgr._approve_pr("acme", "widget", 1)
        client.approve_pull_request.assert_not_called()

    def test_a_null_is_draft_is_left_out_of_the_shape(self) -> None:
        shape = _to_rest_shape({"number": 1, "state": "OPEN", "isDraft": None})
        assert "draft" not in shape


class TestPostFailureRefresh:
    @pytest.mark.asyncio
    async def test_a_dirty_draft_never_enters_conflict_recovery(self) -> None:
        # A repo-scoped merge returns false and the post-failure refresh
        # finds a dirty draft: it is skipped, not routed to recovery
        # (Copilot review on #567); through ``_attempt_direct_merge``.
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        mgr, _client = make_merge_manager(repo_scoped=True)
        pr = _pr(1, is_draft=False)
        flow = _MergeFlow(
            pr_info=pr,
            repo_owner="acme",
            repo_name="widget",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )
        mgr._approve_if_review_mandated = AsyncMock()  # type: ignore[method-assign]
        mgr._merge_under_dispatch_lock = AsyncMock(return_value=(False, False))  # type: ignore[method-assign]
        mgr._fetch_pr_state = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "state": "open",
                "draft": True,
                "mergeable": False,
                "mergeable_state": "dirty",
            }
        )
        mgr._handle_merge_conflict = AsyncMock()  # type: ignore[method-assign]
        with pytest.raises(DraftPullRequestError):
            await mgr._attempt_direct_merge(flow)
        mgr._handle_merge_conflict.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_required_workflows_retry_skips_a_draft(self) -> None:
        # The required-workflows recovery dispatches its own merge; its
        # pre-dispatch read must skip a draft, and its broad handler must
        # not swallow the signal.
        mgr, client = make_merge_manager(repo_scoped=True)
        mgr._wait_for_auto_merge = AsyncMock(return_value=(False, False))  # type: ignore[method-assign]
        client.get = AsyncMock(
            return_value={"state": "open", "draft": True, "mergeable_state": "blocked"}
        )
        client.merge_pull_request = AsyncMock(return_value=True)
        with pytest.raises(DraftPullRequestError):
            await mgr._wait_for_required_workflows_and_retry(
                _pr(1, is_draft=False), "acme", "widget"
            )
        client.merge_pull_request.assert_not_called()


class TestWaitsEndAtADraft:
    """A PR marked draft while a wait reads it ends that wait (#567).

    No check makes a draft mergeable, so waiting on would time out and
    Step 6 could defer the PR to auto-merge as pending without reading
    it again (Copilot review on #567).
    """

    @staticmethod
    def _flow(pr: PullRequestInfo) -> Any:
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        return _MergeFlow(
            pr_info=pr,
            repo_owner="acme",
            repo_name="widget",
            result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
        )

    @pytest.mark.asyncio
    async def test_a_conversion_during_the_auto_merge_wait_ends_it(self) -> None:
        mgr, _client = make_merge_manager()
        # Polls at once, and gives up quickly should the draft go unseen.
        mgr._merge_recheck_interval = 0.0
        mgr._merge_timeout = 0.2
        mgr._enable_auto_merge_with_approval = AsyncMock(return_value=True)  # type: ignore[method-assign]
        mgr._fetch_pr_state = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "state": "open",
                "draft": True,
                "mergeable": False,
                "mergeable_state": "blocked",
            }
        )
        pr = _pr(1, is_draft=False)
        with pytest.raises(DraftPullRequestError):
            await mgr._wait_for_checks(self._flow(pr))
        assert pr.is_draft is True
        assert mgr._fetch_pr_state.await_count == 1
        assert not mgr._waiting_prs

    @pytest.mark.asyncio
    async def test_the_refresh_after_a_late_precommit_rerun_stops_too(self) -> None:
        mgr, client = make_merge_manager()
        mgr._trigger_stale_precommit_ci = AsyncMock(return_value=True)  # type: ignore[method-assign]
        client.get = AsyncMock(
            return_value={"state": "open", "draft": True, "mergeable_state": "blocked"}
        )
        pr = _pr(1, is_draft=False)
        with pytest.raises(DraftPullRequestError):
            await mgr._recheck_late_precommit(self._flow(pr))
        assert pr.is_draft is True


class TestRecreatedReplacement:
    """A stop signal on dependabot's replacement is that PR's outcome.

    Dependabot closes the original when it recreates it, so a failure
    recorded against the original was confirmed as closed, and the run
    could exit 0 with the replacement neither approved nor armed
    (Copilot review on #567).  Driven through ``_merge_single_pr``, which
    confirms failures.
    """

    @staticmethod
    async def _arm(mgr: Any, flow: Any) -> None:
        """The wait path: approve the replacement and arm auto-merge."""
        await mgr._approve_and_arm_replacement("acme", "widget", _pr(2, is_draft=False))

    @staticmethod
    async def _merge(mgr: Any, flow: Any) -> None:
        """The ready path: approve the replacement and merge it."""
        await mgr._merge_recreated_pr(flow, _pr(2, is_draft=False))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["_arm", "_merge"])
    @pytest.mark.parametrize(
        ("signal", "status"),
        [
            (DraftStateUnknownError, MergeStatus.FAILED),
            (DraftPullRequestError, MergeStatus.SKIPPED),
        ],
    )
    async def test_the_outcome_names_the_replacement(
        self, path: str, signal: type[Exception], status: MergeStatus
    ) -> None:
        from dependamerge.merge_manager._single_pr_context import _MergeFlow

        mgr, client = make_merge_manager()

        async def read(endpoint: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
            # The original, #1, is closed by the recreate; #2 is open.
            if endpoint.endswith("/pulls/1"):
                return {"state": "closed", "merged": False}
            return {"state": "open", "merged": False, "mergeable_state": "blocked"}

        client.get = AsyncMock(side_effect=read)
        raised = signal("acme/widget#2")
        mgr._ensure_pr_approved = AsyncMock(side_effect=raised)  # type: ignore[method-assign]
        mgr._approve_pr = AsyncMock(side_effect=raised)  # type: ignore[method-assign]
        mgr._enable_auto_merge_for_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
        mgr._get_merge_method_for_repo = AsyncMock(return_value="merge")  # type: ignore[method-assign]
        step = getattr(self, path)

        async def sequence(flow: _MergeFlow) -> MergeResult:
            await step(mgr, flow)
            return flow.result

        mgr._run_merge_sequence = sequence  # type: ignore[method-assign]
        result = await mgr._merge_single_pr(_pr(1, is_draft=False))
        assert result.status is status
        assert result.pr_info.number == 2
        mgr._enable_auto_merge_for_pr.assert_not_called()
        client.merge_pull_request.assert_not_called()
