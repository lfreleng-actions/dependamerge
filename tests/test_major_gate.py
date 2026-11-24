# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Merging a major version bump is a deliberate act.

Regression cover for lfreleng-actions/dependamerge#671.  An unattended
run holds gated bumps back as skipped; an interactive run asks for a
value bound to exactly the gated pull requests it lists; ``--allow-major``
turns the gate off.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import ExitStack
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.cli._context import _MergeContext
from dependamerge.cli._human_gate import _human_set_override
from dependamerge.cli._major_gate import (
    ALLOW_MAJOR_HINT,
    _gate_reason,
    _gated_bump,
    _major_set_value,
)
from dependamerge.cli._merge_dispatch import _run_single_pr_merge
from dependamerge.cli._merge_scan import _handle_preview_confirmation
from dependamerge.error_codes import ExitCode
from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.models import ComparisonResult, PullRequestInfo

_PERMS_OK = {
    "approve": {"has_permission": True},
    "merge": {"has_permission": True},
    "branch_protection": {"has_permission": True},
}

_MAJOR = "Bumps [foo](https://x) from 1.3.3 to 2.0.0.\n"
_MINOR = "Bumps [bar](https://x) from 1.3.3 to 1.4.0.\n"
_ZERO_MINOR = "Bumps [baz](https://x) from 0.7.0 to 0.8.0.\n"
_DIGEST = "Bumps [qux](https://x) from 1a2b3c4d to 5e6f7a8b.\n"
_GROUPED = (
    "Bumps the pip group with 2 updates.\n"
    "Updates `foo` from 1.3.3 to 2.0.0\n"
    "Updates `bar` from 1.0.0 to 1.1.0\n"
)
_FOO_REASON = "major version bump: foo 1.3.3 → 2.0.0"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _pr(
    number: int,
    body: str = _MINOR,
    *,
    repo: str = "acme/widget",
    sha: str = "a" * 40,
    author: str = "dependabot[bot]",
) -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="Bump a dependency",
        body=body,
        author=author,
        head_sha=sha,
        base_branch="main",
        head_branch=f"dependabot/pip/dep-{number}",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name=repo,
        html_url=f"https://github.com/{repo}/pull/{number}",
    )


def _ctx(**overrides: Any) -> _MergeContext:
    ctx = _MergeContext(
        pr_url="https://github.com/acme/widget/pull/1",
        no_confirm=False,
        similarity_threshold=0.8,
        merge_method="merge",
        token="t",
        override=None,
        no_fix=False,
        merge_timeout=300.0,
        show_progress=False,
        debug_matching=False,
        dismiss_copilot=False,
        force="none",
        verbose=False,
        no_netrc=False,
        netrc_file=None,
        netrc_optional=True,
        github2gerrit_mode="ignore",
    )
    for name, value in overrides.items():
        setattr(ctx, name, value)
    return ctx


class _FakeMerge:
    """A ``_run_parallel_merge`` stand-in recording each pass."""

    def __init__(self, failing: tuple[int, ...] = ()) -> None:
        self.failing = failing
        self.calls: list[tuple[bool, list[int]]] = []

    def __call__(self, ctx, prs, preview, **kwargs) -> list[MergeResult]:
        self.calls.append((preview, [pr.number for pr, _ in prs]))
        return [
            MergeResult(pr, MergeStatus.FAILED, error="boom")
            if pr.number in self.failing
            else MergeResult(pr, MergeStatus.MERGED)
            for pr, _ in prs
        ]

    @property
    def merged(self) -> list[int]:
        """The PRs the real (non-preview) pass was given."""
        return [n for preview, numbers in self.calls if not preview for n in numbers]


def _published(mock_publish: Mock) -> list[MergeResult]:
    mock_publish.assert_called_once()
    results: list[MergeResult] = mock_publish.call_args.args[1]
    return results


def _skipped(results: list[MergeResult]) -> dict[int, str | None]:
    return {
        r.pr_info.number: r.error for r in results if r.status == MergeStatus.SKIPPED
    }


def _repo_hash(count: int) -> str:
    combined = f"repo-merge:acme/widget:{count}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


class TestWhatIsGated:
    def test_a_major_bump_is_gated(self) -> None:
        assert _gated_bump(_ctx(), _pr(1, _MAJOR)) is not None

    def test_allow_major_gates_nothing(self) -> None:
        assert _gated_bump(_ctx(allow_major=True), _pr(1, _MAJOR)) is None

    def test_a_zero_minor_bump_needs_the_option(self) -> None:
        assert _gated_bump(_ctx(), _pr(1, _ZERO_MINOR)) is None
        bump = _gated_bump(_ctx(gate_zero_minor=True), _pr(1, _ZERO_MINOR))
        assert bump is not None
        assert _gate_reason(_ctx(gate_zero_minor=True), bump) == (
            "0.x minor version bump: baz 0.7.0 → 0.8.0"
        )

    @pytest.mark.parametrize("gate_zero_minor", [False, True])
    def test_unknown_is_never_gated(self, gate_zero_minor: bool) -> None:
        ctx = _ctx(gate_zero_minor=gate_zero_minor)
        assert _gated_bump(ctx, _pr(1, _DIGEST)) is None

    def test_a_grouped_update_names_its_major_dependency(self) -> None:
        bump = _gated_bump(_ctx(), _pr(1, _GROUPED))
        assert bump is not None
        assert _gate_reason(_ctx(), bump) == _FOO_REASON


class TestConfirmationValue:
    def test_it_changes_with_a_head_commit(self) -> None:
        assert _major_set_value("acme", [_pr(1, sha="a" * 40)]) != (
            _major_set_value("acme", [_pr(1, sha="b" * 40)])
        )

    def test_it_changes_when_another_pr_joins(self) -> None:
        assert _major_set_value("acme", [_pr(1)]) != (
            _major_set_value("acme", [_pr(1), _pr(2)])
        )

    def test_it_ignores_order(self) -> None:
        assert _major_set_value("acme", [_pr(1), _pr(2)]) == (
            _major_set_value("acme", [_pr(2), _pr(1)])
        )

    def test_it_cannot_stand_in_for_the_human_override(self) -> None:
        assert _major_set_value("acme", [_pr(1)]) != (
            _human_set_override("acme", [_pr(1)])
        )


@patch("dependamerge.cli._merge_report._publish_results")
@patch("dependamerge.cli._run_parallel_merge")
@patch("dependamerge.cli.GitHubClient")
@patch("dependamerge.cli.asyncio.run")
class TestCommandLine:
    runner = CliRunner()

    def _invoke(self, target: str, *extra: str, input: str | None = None):
        result = self.runner.invoke(
            app,
            ["merge", target, "--token", "t", "--no-progress", *extra],
            input=input,
        )
        return result, _strip_ansi(result.stdout)

    @staticmethod
    def _setup(
        mock_run,
        mock_client,
        fetched: list[PullRequestInfo],
        owner: bool,
        *,
        dry_run: bool = False,
    ):
        client = mock_client.return_value = Mock(token="t")
        # The real pass re-reads each confirmed major bump's head.
        live = {pr.number: pr for pr in fetched}
        client.get_pull_request_info.side_effect = lambda _o, _r, n: live[n]
        # An owner run fetches before its permission check; a repository
        # run checks first; a dry run skips the check.
        if dry_run:
            canned: list[object] = [fetched]
        elif owner:
            canned = [(fetched, []), _PERMS_OK]
        else:
            canned = [_PERMS_OK, fetched]

        def _run(coro):
            coro.close()
            return canned.pop(0)

        mock_run.side_effect = _run

    @pytest.mark.parametrize("owner", [False, True], ids=["repository", "owner"])
    def test_unattended_holds_a_major_back(
        self, mock_run, mock_client, mock_merge, mock_publish, owner
    ) -> None:
        self._setup(mock_run, mock_client, [_pr(1, _MAJOR), _pr(2)], owner)
        mock_merge.side_effect = fake = _FakeMerge()

        target = "https://github.com/acme" if owner else "acme/widget"
        result, out = self._invoke(target, "--no-confirm")

        assert result.exit_code == 0, out
        assert fake.merged == [2]
        assert ALLOW_MAJOR_HINT in out
        assert f"acme/widget#1: {_FOO_REASON}" in out
        assert _skipped(_published(mock_publish)) == {1: _FOO_REASON}

    def test_allow_major_merges_and_reports_it(
        self, mock_run, mock_client, mock_merge, mock_publish
    ) -> None:
        self._setup(mock_run, mock_client, [_pr(1, _MAJOR), _pr(2)], False)
        mock_merge.side_effect = fake = _FakeMerge()

        result, out = self._invoke("acme/widget", "--no-confirm", "--allow-major")

        assert result.exit_code == 0, out
        assert fake.merged == [1, 2]
        assert "Major version bumps merged: 1" in out
        assert ALLOW_MAJOR_HINT not in out

    @pytest.mark.parametrize("gate", [True, False])
    def test_gate_zero_minor_holds_a_zero_minor_back(
        self, mock_run, mock_client, mock_merge, mock_publish, gate
    ) -> None:
        self._setup(mock_run, mock_client, [_pr(1, _ZERO_MINOR), _pr(2)], False)
        mock_merge.side_effect = fake = _FakeMerge()
        extra = ["--gate-zero-minor"] if gate else []

        result, out = self._invoke("acme/widget", "--no-confirm", *extra)

        assert result.exit_code == 0, out
        assert fake.merged == ([2] if gate else [1, 2])

    @pytest.mark.parametrize(
        ("entered", "merged"),
        [
            pytest.param("value", [1, 2], id="the-value"),
            pytest.param("", [2], id="enter"),
            pytest.param("nope", [2], id="a-wrong-value"),
        ],
    )
    def test_interactive_value_decides(
        self, mock_run, mock_client, mock_merge, mock_publish, entered, merged
    ) -> None:
        major = _pr(1, _MAJOR)
        self._setup(mock_run, mock_client, [major, _pr(2)], False)
        mock_merge.side_effect = fake = _FakeMerge()
        value = _major_set_value("acme/widget", [major])
        typed = value if entered == "value" else entered

        # The continue value derives from the count left to merge.
        result, out = self._invoke(
            "acme/widget", input=f"{typed}\n{_repo_hash(len(merged))}\n"
        )

        assert result.exit_code == 0, out
        assert f"To merge them too enter: {value}" in out
        assert fake.merged == merged
        expected = {} if entered == "value" else {1: _FOO_REASON}
        assert _skipped(_published(mock_publish)) == expected

    def test_a_head_moved_after_the_value_leaves_it_out(
        self, mock_run, mock_client, mock_merge, mock_publish
    ) -> None:
        major = _pr(1, _MAJOR)
        self._setup(mock_run, mock_client, [major, _pr(2)], False)
        mock_merge.side_effect = fake = _FakeMerge()
        value = _major_set_value("acme/widget", [major])
        pushed = _pr(1, _MAJOR, sha="b" * 40)
        mock_client.return_value.get_pull_request_info.side_effect = None
        mock_client.return_value.get_pull_request_info.return_value = pushed

        result, out = self._invoke("acme/widget", input=f"{value}\n{_repo_hash(2)}\n")

        assert result.exit_code == 0, out
        assert "their head moved: 1" in out
        assert fake.merged == [2]
        moved = f"{_FOO_REASON}; head changed since it was confirmed"
        assert _skipped(_published(mock_publish)) == {1: moved}

    def test_eof_leaves_them_out(
        self, mock_run, mock_client, mock_merge, mock_publish
    ) -> None:
        self._setup(mock_run, mock_client, [_pr(1, _MAJOR), _pr(2)], False)
        mock_merge.side_effect = fake = _FakeMerge()

        result, out = self._invoke("acme/widget", input="")

        assert result.exit_code == 0, out
        assert "No value entered: major version bumps left out" in out
        assert "Mergeable without them: 1/2 PRs" in out
        assert fake.merged == []
        assert _skipped(_published(mock_publish)) == {1: _FOO_REASON}

    def test_an_unmergeable_major_prompts_for_nothing(
        self, mock_run, mock_client, mock_merge, mock_publish
    ) -> None:
        self._setup(mock_run, mock_client, [_pr(1, _MAJOR), _pr(2)], False)
        mock_merge.side_effect = fake = _FakeMerge(failing=(1,))

        result, out = self._invoke("acme/widget", input=f"{_repo_hash(1)}\n")

        assert result.exit_code == 0, out
        assert "To merge them too enter" not in out
        assert fake.merged == [2]

    def test_a_dry_run_prompts_for_nothing_and_merges_nothing(
        self, mock_run, mock_client, mock_merge, mock_publish
    ) -> None:
        prs = [_pr(1, _MAJOR), _pr(2)]
        self._setup(mock_run, mock_client, prs, False, dry_run=True)
        mock_merge.side_effect = fake = _FakeMerge()

        result, out = self._invoke("acme/widget", "--dry-run")

        assert result.exit_code == 0, out
        assert "Dry run: a real run asks for a value" in out
        assert "To merge them too enter" not in out
        assert fake.calls == [(True, [1, 2])]

    def test_the_human_and_major_gates_each_need_their_own_value(
        self, mock_run, mock_client, mock_merge, mock_publish
    ) -> None:
        human = _pr(1, _MAJOR, author="alice")
        self._setup(mock_run, mock_client, [human, _pr(2)], False)
        mock_merge.side_effect = fake = _FakeMerge()
        override = _human_set_override("acme/widget", [human])

        # The --override value does not confirm the major bump.
        result, out = self._invoke(
            "acme/widget",
            "--include-human-prs",
            "--override",
            override,
            input=f"{override}\n{_repo_hash(1)}\n",
        )

        assert result.exit_code == 0, out
        assert "The value does not match" in out
        assert fake.merged == [2]


class TestSinglePullRequest:
    @staticmethod
    def _run(
        ctx: _MergeContext,
        source: PullRequestInfo,
        similar,
        *,
        validate_author: bool = False,
    ) -> _FakeMerge:
        ctx.source_pr = source

        def _scan(c: _MergeContext) -> None:
            c.all_similar_prs = [(pr, cast(ComparisonResult, None)) for pr in similar]

        fake = _FakeMerge()
        prefix = "dependamerge.cli._merge_dispatch."
        with ExitStack() as stack:
            for name in (
                "_init_github_merge",
                "_fetch_and_validate_source_pr",
                "_load_source_policy_files",
                "_maybe_check_merge_permissions",
            ):
                stack.enter_context(patch(prefix + name))
            if not validate_author:
                stack.enter_context(patch(prefix + "_validate_automation_author"))
            stack.enter_context(
                patch(prefix + "_scan_and_find_similar", side_effect=_scan)
            )
            stack.enter_context(
                patch("dependamerge.cli._run_parallel_merge", side_effect=fake)
            )
            _run_single_pr_merge(ctx)
        return fake

    @staticmethod
    def _human_ctx() -> _MergeContext:
        client = Mock()
        client.is_automation_author.return_value = False
        client.get_pull_request_commits.return_value = ["Bump foo"]
        return _ctx(
            no_confirm=True,
            include_human_prs=True,
            github_client=client,
            owner="acme",
            repo_name="widget",
            pr_number=1,
        )

    @patch("dependamerge.cli._merge_report._publish_results")
    def test_a_held_back_human_source_needs_no_override(self, mock_publish) -> None:
        ctx = self._human_ctx()
        source = _pr(1, _MAJOR, author="alice")
        similar = [_pr(2, _MAJOR, repo="acme/other", author="alice")]

        fake = self._run(ctx, source, similar, validate_author=True)

        assert fake.calls == []
        assert _skipped(_published(mock_publish)) == {
            1: _FOO_REASON,
            2: _FOO_REASON,
        }

    @patch("dependamerge.cli._merge_report._publish_results")
    def test_a_human_pr_left_to_merge_still_needs_the_override(
        self, mock_publish
    ) -> None:
        ctx = self._human_ctx()
        source = _pr(1, _MAJOR, author="alice")
        similar = [_pr(2, repo="acme/other", author="alice")]

        with pytest.raises(SystemExit) as exited:
            self._run(ctx, source, similar, validate_author=True)

        assert exited.value.code == ExitCode.OVERRIDE_REQUIRED
        mock_publish.assert_not_called()

    @patch("dependamerge.cli._merge_report._publish_results")
    def test_unattended_holds_a_major_back(self, mock_publish, capsys) -> None:
        ctx = _ctx(no_confirm=True)
        fake = self._run(ctx, _pr(1), [_pr(2, _MAJOR, repo="acme/other")])

        assert fake.merged == [1]
        assert ALLOW_MAJOR_HINT in capsys.readouterr().out
        assert _skipped(_published(mock_publish)) == {2: _FOO_REASON}

    @patch("dependamerge.cli._merge_report._publish_results")
    def test_unattended_holds_a_major_source_back(self, mock_publish) -> None:
        ctx = _ctx(no_confirm=True)
        source = _pr(1, _MAJOR)
        fake = self._run(ctx, source, [_pr(2, _MAJOR, repo="acme/other")])

        assert fake.calls == []
        assert _skipped(_published(mock_publish)) == {
            1: _FOO_REASON,
            2: _FOO_REASON,
        }

    @patch("dependamerge.cli._merge_report._publish_results")
    def test_a_major_source_warns_without_a_second_value(
        self, mock_publish, capsys
    ) -> None:
        client = Mock()
        client.get_pull_request_commits.return_value = ["Bump foo"]
        ctx = _ctx(github_client=client, owner="acme", repo_name="widget")
        ctx.source_pr = source = _pr(1, _MAJOR)
        similar = _pr(2, _MAJOR, repo="acme/other")
        results = [MergeResult(similar, MergeStatus.MERGED)]
        results.append(MergeResult(source, MergeStatus.MERGED))

        with patch("typer.prompt") as prompt:
            _handle_preview_confirmation(
                ctx, results, [(similar, None), (source, None)], 2, 2
            )

        prompt.assert_not_called()
        out = _strip_ansi(capsys.readouterr().out)
        warning = f"The source PR is a {_FOO_REASON}"
        assert warning in out
        assert out.index(warning) < out.index("To proceed with merging enter")
        assert _skipped(_published(mock_publish)) == {}

    @patch("dependamerge.cli._merge_report._publish_results")
    def test_a_major_similar_pr_needs_the_value(self, mock_publish) -> None:
        client = Mock()
        client.get_pull_request_commits.return_value = ["Bump bar"]
        ctx = _ctx(github_client=client, owner="acme", repo_name="widget")
        ctx.source_pr = source = _pr(1)
        similar = _pr(2, _MAJOR, repo="acme/other")
        results = [
            MergeResult(similar, MergeStatus.MERGED),
            MergeResult(source, MergeStatus.MERGED),
        ]

        with patch("typer.prompt", return_value="") as prompt:
            _handle_preview_confirmation(
                ctx, results, [(similar, None), (source, None)], 2, 2
            )

        prompt.assert_called_once()
        assert _skipped(_published(mock_publish)) == {2: _FOO_REASON}
