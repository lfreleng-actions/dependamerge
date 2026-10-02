# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""A real merge run exits non-zero when it leaves work for a human."""

import re
from unittest.mock import Mock, patch

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.cli._gerrit_merge import _run_gerrit_submission
from dependamerge.cli._merge_report import _exit_if_any_failed
from dependamerge.error_codes import ExitCode
from dependamerge.gerrit.models import GerritSubmitResult
from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.models import PullRequestInfo

_PERMS_OK = {
    "approve": {"has_permission": True},
    "merge": {"has_permission": True},
    "branch_protection": {"has_permission": True},
}


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_pr(number: int, repo: str = "acme/widget") -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="Bump foo from 1.0 to 2.0",
        body="Automated dependency update",
        author="dependabot[bot]",
        head_sha="abc123",
        base_branch="main",
        head_branch="dependabot/pip/foo-2.0",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name=repo,
        html_url=f"https://github.com/{repo}/pull/{number}",
    )


def _result(status: MergeStatus, number: int = 1) -> MergeResult:
    error = "boom" if status in (MergeStatus.FAILED, MergeStatus.BLOCKED) else None
    return MergeResult(pr_info=_make_pr(number), status=status, error=error)


def _closing_run(results: list[object]):
    """An ``asyncio.run`` stand-in returning ``results`` in order."""
    remaining = list(results)

    def _run(coro):
        coro.close()
        return remaining.pop(0)

    return _run


class TestExitIfAnyFailed:
    @pytest.mark.parametrize("status", [MergeStatus.FAILED, MergeStatus.BLOCKED])
    def test_outcomes_needing_a_human_exit_with_merge_error(self, status):
        results = [_result(MergeStatus.MERGED), _result(status, 2)]
        with pytest.raises(typer.Exit) as excinfo:
            _exit_if_any_failed(results)
        assert excinfo.value.exit_code == ExitCode.MERGE_ERROR

    @pytest.mark.parametrize(
        "status",
        [
            MergeStatus.MERGED,
            MergeStatus.AUTO_MERGE_PENDING,
            MergeStatus.UNSETTLED,
            MergeStatus.SKIPPED,
            MergeStatus.CLOSED,
        ],
    )
    def test_other_outcomes_do_not_exit(self, status):
        _exit_if_any_failed([_result(status)])

    def test_an_empty_run_does_not_exit(self):
        _exit_if_any_failed([])


class TestOwnerWideExitCode:
    runner = CliRunner()

    def _invoke(self, *extra: str):
        return self.runner.invoke(
            app,
            ["merge", "https://github.com/acme", "--token", "test_token", *extra],
        )

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_no_confirm_run_with_a_failure_exits_7(
        self, mock_asyncio_run, mock_client_class
    ):
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run(
            [
                ([_make_pr(1), _make_pr(2)], []),
                _PERMS_OK,
                [_result(MergeStatus.MERGED), _result(MergeStatus.FAILED, 2)],
            ]
        )

        result = self._invoke("--no-confirm", "--no-progress")

        assert result.exit_code == ExitCode.MERGE_ERROR, result.stdout
        out = _strip_ansi(result.stdout)
        # The summary is still printed in full before exiting.
        assert "Final Results: 1 merged, 1 failed" in out
        assert "1 PR failed or blocked; exit code 7" in out

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_no_confirm_run_that_merges_everything_exits_0(
        self, mock_asyncio_run, mock_client_class
    ):
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run(
            [([_make_pr(1)], []), _PERMS_OK, [_result(MergeStatus.MERGED)]]
        )

        result = self._invoke("--no-confirm", "--no-progress")

        assert result.exit_code == 0, result.stdout

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_dry_run_predicting_failures_exits_0(
        self, mock_asyncio_run, mock_client_class
    ):
        # A preview predicts; it has not failed anything.
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run(
            [([_make_pr(1)], []), [_result(MergeStatus.FAILED)]]
        )

        result = self._invoke("--dry-run", "--no-progress")

        assert result.exit_code == 0, result.stdout


class TestRepoExitCode:
    runner = CliRunner()

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_no_confirm_run_with_a_blocked_pr_exits_7(
        self, mock_asyncio_run, mock_client_class
    ):
        mock_client = Mock(token="test_token")
        mock_client.is_automation_author.return_value = True
        mock_client_class.return_value = mock_client
        mock_asyncio_run.side_effect = _closing_run(
            [_PERMS_OK, [_make_pr(1)], [_result(MergeStatus.BLOCKED)]]
        )

        result = self.runner.invoke(
            app,
            [
                "merge",
                "https://github.com/acme/widget",
                "--token",
                "test_token",
                "--no-confirm",
                "--no-progress",
            ],
        )

        assert result.exit_code == ExitCode.MERGE_ERROR, result.stdout


class TestGerritExitCode:
    def _submit(self, results: list[GerritSubmitResult]) -> None:
        manager = Mock()
        manager.submit_changes_parallel.return_value = results
        parsed_url = Mock(host="gerrit.example.org", base_path=None)
        with patch("dependamerge.cli.create_submit_manager", return_value=manager):
            _run_gerrit_submission(
                parsed_url,
                Mock(username="user", password="secret"),
                [],
                False,
                Console(file=None),
            )

    def test_an_unsubmitted_change_exits_7(self):
        results = [
            GerritSubmitResult(change_number=1, project="p", success=True),
            GerritSubmitResult(
                change_number=2, project="p", success=False, error="conflict"
            ),
        ]
        with pytest.raises(typer.Exit) as excinfo:
            self._submit(results)
        assert excinfo.value.exit_code == ExitCode.MERGE_ERROR

    def test_every_change_submitted_does_not_exit(self):
        self._submit([GerritSubmitResult(change_number=1, project="p", success=True)])
