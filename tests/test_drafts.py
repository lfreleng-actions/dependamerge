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
from dependamerge.merge_manager import MergeStatus
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
        with pytest.raises(RuntimeError, match="draft PR"):
            await mgr._ensure_pr_approved(_pr(1, is_draft=True), "acme", "widget")
        mgr._approve_pr.assert_not_called()

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
        await mgr._refresh_pr_mergeable("acme", "widget", pr, "acme/widget#1")
        assert pr.is_draft is True
