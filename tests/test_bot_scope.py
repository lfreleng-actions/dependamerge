# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""How a run applies its bot policy.

Covers the command-line side (building and checking the policy, the
source PR gate, dropping and reporting out-of-scope PRs) and the merge
manager's own checks, which hold even for a PR that reaches it by
another route.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dependamerge.bot_policy import BotPolicy, run_policy_source
from dependamerge.cli._bot_scope import (
    _build_bot_policy,
    _check_listed_bots,
    _remove_out_of_scope_bots,
)
from dependamerge.cli._merge_authors import _validate_automation_author
from dependamerge.error_codes import ExitCode
from dependamerge.github_service._similar import _SimilarPullRequestsMixin
from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

CODE_MONKEY = "lf-releng-code-monkey-bot[bot]"
LOOKUP = "dependamerge.cli._bot_scope._lookup_apps"


def _pr(author: str, number: int = 1, repo: str = "acme/widget") -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="t",
        body=None,
        author=author,
        head_sha="abc",
        base_branch="main",
        head_branch="topic",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name=repo,
        html_url=f"https://github.com/{repo}/pull/{number}",
    )


class TestBuildPolicy:
    def test_flags_and_user_file_combine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "dependamerge").mkdir()
        (tmp_path / "dependamerge" / "config.json").write_text(
            json.dumps({"include_bots": [CODE_MONKEY]}), encoding="utf-8"
        )
        policy = _build_bot_policy(None, ["renovate"])
        assert policy.admits(CODE_MONKEY, "acme/widget")
        assert not policy.admits("renovate[bot]", "acme/widget")

    def test_invalid_flags_stop_with_a_configuration_error(self) -> None:
        with pytest.raises(SystemExit) as raised:
            _build_bot_policy(["not@app"], None)
        assert raised.value.code == ExitCode.CONFIGURATION_ERROR

    def test_invalid_user_file_stops_with_a_configuration_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "dependamerge").mkdir()
        (tmp_path / "dependamerge" / "config.json").write_text(
            json.dumps({"exlude_bots": ["renovate"]}), encoding="utf-8"
        )
        with pytest.raises(SystemExit) as raised:
            _build_bot_policy(None, None)
        assert raised.value.code == ExitCode.CONFIGURATION_ERROR


class TestCheckListedBots:
    def test_nothing_listed_looks_nothing_up(self) -> None:
        with patch(LOOKUP) as lookup:
            _check_listed_bots(BotPolicy(), "t", "github.com")
        lookup.assert_not_called()

    def test_unknown_exclusion_stops_the_run(self) -> None:
        # A misspelt exclusion leaves in scope the App it meant to stop.
        policy = BotPolicy(run=run_policy_source(None, ["renovat"]))
        with patch(LOOKUP, return_value={"renovat[bot]": False}):
            with pytest.raises(SystemExit) as raised:
                _check_listed_bots(policy, "t", "github.com")
        assert raised.value.code == ExitCode.CONFIGURATION_ERROR

    def test_unknown_inclusion_only_warns(self, capsys) -> None:
        policy = BotPolicy(run=run_policy_source(["no-such-app"], None))
        with patch(LOOKUP, return_value={"no-such-app[bot]": False}):
            _check_listed_bots(policy, "t", "github.com")
        assert "No GitHub App is named no-such-app" in capsys.readouterr().out

    def test_failed_lookup_is_not_held_against_a_name(self) -> None:
        policy = BotPolicy(run=run_policy_source(None, ["renovate"]))
        with patch(LOOKUP, return_value={"renovate[bot]": None}):
            _check_listed_bots(policy, "t", "github.com")


class TestRemoveOutOfScopeBots:
    def test_out_of_scope_apps_are_dropped_and_reported(self, capsys) -> None:
        prs = [
            _pr("dependabot[bot]", 1),
            _pr(CODE_MONKEY, 2),
            _pr(CODE_MONKEY, 3),
            _pr("octocat", 4),
        ]
        kept = _remove_out_of_scope_bots(BotPolicy(), prs)
        assert [pr.number for pr in kept] == [1, 4]
        out = capsys.readouterr().out
        assert "Bot PRs out of scope: 2" in out
        assert f"{CODE_MONKEY} ×2 (not included by any policy)" in out

    def test_included_apps_are_kept_silently(self, capsys) -> None:
        policy = BotPolicy(run=run_policy_source([CODE_MONKEY], None))
        kept = _remove_out_of_scope_bots(policy, [_pr(CODE_MONKEY)])
        assert len(kept) == 1
        assert "out of scope" not in capsys.readouterr().out


class TestSourceGate:
    def _ctx(self, author: str, policy: BotPolicy) -> MagicMock:
        ctx = MagicMock()
        ctx.github_client.is_automation_author.side_effect = lambda a: a.endswith(
            "[bot]"
        )
        ctx.source_pr = _pr(author)
        ctx.bot_policy = policy
        return ctx

    def test_out_of_scope_app_stops_without_an_override(self, capsys) -> None:
        ctx = self._ctx(CODE_MONKEY, BotPolicy())
        with pytest.raises(SystemExit) as raised:
            _validate_automation_author(ctx)
        assert raised.value.code == ExitCode.VALIDATION_ERROR
        out = capsys.readouterr().out
        assert "--include-bots lf-releng-code-monkey-bot" in out
        assert "--override" not in out
        ctx.github_client.get_pull_request_commits.assert_not_called()

    def test_included_app_passes(self) -> None:
        policy = BotPolicy(run=run_policy_source([CODE_MONKEY], None))
        _validate_automation_author(self._ctx(CODE_MONKEY, policy))

    def test_excluded_builtin_stops(self) -> None:
        policy = BotPolicy(run=run_policy_source(None, ["dependabot"]))
        with pytest.raises(SystemExit):
            _validate_automation_author(self._ctx("dependabot[bot]", policy))


class TestSimilarCandidates:
    source = _pr("dependabot[bot]", 1)

    def test_out_of_scope_apps_are_not_candidates(self) -> None:
        assert not _SimilarPullRequestsMixin._is_similar_pr_candidate(
            self.source, _pr(CODE_MONKEY, 2), only_automation=True
        )

    def test_included_apps_are_candidates(self) -> None:
        policy = BotPolicy(run=run_policy_source([CODE_MONKEY], None))
        assert _SimilarPullRequestsMixin._is_similar_pr_candidate(
            self.source, _pr(CODE_MONKEY, 2), only_automation=True, bot_policy=policy
        )


class TestMergeManagerChecks:
    """The manager refuses out-of-scope Apps whatever brought the PR."""

    @pytest.mark.asyncio
    async def test_out_of_scope_app_is_skipped_without_api_calls(self) -> None:
        mgr, client = make_merge_manager()
        result = await mgr._merge_single_pr_impl(_pr(CODE_MONKEY))
        assert result.status is MergeStatus.SKIPPED
        assert result.error is not None
        assert "not in this run's scope" in result.error
        assert not client.mock_calls

    @pytest.mark.asyncio
    async def test_included_app_reaches_the_merge_sequence(self) -> None:
        policy = BotPolicy(run=run_policy_source([CODE_MONKEY], None))
        mgr, _client = make_merge_manager(bot_policy=policy)
        pr = _pr(CODE_MONKEY)
        merged = MergeResult(pr_info=pr, status=MergeStatus.MERGED)
        mgr._get_merge_method_for_repo = AsyncMock(return_value="merge")  # type: ignore[method-assign]
        mgr._run_merge_sequence = AsyncMock(return_value=merged)  # type: ignore[method-assign]
        result = await mgr._merge_single_pr_impl(pr)
        assert result.status is MergeStatus.MERGED

    @pytest.mark.asyncio
    async def test_people_are_left_to_the_caller(self) -> None:
        mgr, _client = make_merge_manager()
        pr = _pr("octocat")
        merged = MergeResult(pr_info=pr, status=MergeStatus.MERGED)
        mgr._get_merge_method_for_repo = AsyncMock(return_value="merge")  # type: ignore[method-assign]
        mgr._run_merge_sequence = AsyncMock(return_value=merged)  # type: ignore[method-assign]
        result = await mgr._merge_single_pr_impl(pr)
        assert result.status is MergeStatus.MERGED

    @pytest.mark.asyncio
    async def test_approval_of_an_out_of_scope_app_is_refused(self) -> None:
        mgr, _client = make_merge_manager()
        mgr._approve_pr = AsyncMock(return_value=True)  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="refusing to approve"):
            await mgr._ensure_pr_approved(_pr(CODE_MONKEY), "acme", "widget")
        mgr._approve_pr.assert_not_called()
