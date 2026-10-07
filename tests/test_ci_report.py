# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the CI results document and its Markdown rendering."""

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from dependamerge.ci_report import (
    OUTPUT_KEY,
    SCHEMA_VERSION,
    build_document,
    in_github_actions,
    load_document,
    render_markdown,
    write_results_file,
)
from dependamerge.ci_report.__main__ import main
from dependamerge.cli import app
from dependamerge.cli._merge_report import _format_failure_reason
from dependamerge.error_codes import ExitCode
from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.models import PullRequestInfo

_PERMS_OK = {
    "approve": {"has_permission": True},
    "merge": {"has_permission": True},
    "branch_protection": {"has_permission": True},
}


def _make_pr(
    number: int, repo: str = "acme/widget", title: str = "Bump foo"
) -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title=title,
        body=None,
        author="dependabot[bot]",
        head_sha="abc123",
        base_branch="main",
        head_branch="dependabot/pip/foo",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name=repo,
        html_url=f"https://github.com/{repo}/pull/{number}",
    )


def _document(results, **overrides):
    options = {
        "explain": _format_failure_reason,
        "target": "https://github.com/acme",
        "scope": "owner",
        "preview": False,
        "dry_run": False,
    }
    options.update(overrides)
    return build_document(results, **options)


def _closing_run(results: list[object]):
    remaining = list(results)

    def _run(coro):
        coro.close()
        return remaining.pop(0)

    return _run


def _shown(markdown: str) -> str:
    """The text a reader sees: Markdown backslash escapes removed."""
    return re.sub(r"\\(.)", r"\1", markdown)


class TestDetection:
    def test_true_only_when_the_runner_says_so(self):
        assert in_github_actions({"GITHUB_ACTIONS": "true"})
        assert not in_github_actions({"GITHUB_ACTIONS": "false"})
        assert not in_github_actions({})


class TestBuildDocument:
    def test_counts_cover_every_outcome(self):
        document = _document(
            [
                MergeResult(_make_pr(1), MergeStatus.MERGED),
                MergeResult(_make_pr(2), MergeStatus.FAILED, error="boom"),
            ]
        )
        assert document["schema_version"] == SCHEMA_VERSION
        assert document["counts"]["merged"] == 1
        assert document["counts"]["failed"] == 1
        # Outcomes that did not occur are still present, as zero.
        assert document["counts"]["blocked"] == 0

    def test_entries_carry_the_explained_reason(self):
        reason = "Repository rule violations found\n\nRequired workflow 'Lint' failed."
        document = _document(
            [MergeResult(_make_pr(7), MergeStatus.FAILED, error=reason)],
            selection="Excluding: beta",
            scan_errors=["Error scanning repository acme/broken: boom"],
        )
        entry = document["results"][0]
        assert entry["repository"] == "acme/widget"
        assert entry["number"] == 7
        assert entry["url"] == "https://github.com/acme/widget/pull/7"
        assert entry["status"] == "failed"
        assert entry["reason"]
        assert all(not detail.startswith("\u2022") for detail in entry["details"])
        assert document["selection"] == "Excluding: beta"
        assert document["scan_errors"] == [
            "Error scanning repository acme/broken: boom"
        ]

    def test_a_success_has_no_reason(self):
        document = _document([MergeResult(_make_pr(1), MergeStatus.MERGED)])
        entry = document["results"][0]
        assert entry["reason"] is None
        assert entry["details"] == []

    @pytest.mark.parametrize("error", [None, ""])
    def test_an_unexplained_failure_says_so_as_the_console_does(self, error):
        document = _document(
            [MergeResult(_make_pr(1), MergeStatus.FAILED, error=error)]
        )
        assert document["results"][0]["reason"] == "no reason reported"


class TestWriteResultsFile:
    def test_private_file_in_runner_temp_named_in_the_output(self, tmp_path):
        output = tmp_path / "github_output"
        env = {"RUNNER_TEMP": str(tmp_path), "GITHUB_OUTPUT": str(output)}

        path = write_results_file({"schema_version": SCHEMA_VERSION}, env)

        assert path.parent == tmp_path
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert json.loads(path.read_text()) == {"schema_version": SCHEMA_VERSION}
        assert output.read_text() == f"{OUTPUT_KEY}={path}\n"

    def test_without_an_output_file_the_results_are_still_written(self, tmp_path):
        path = write_results_file({"a": 1}, {"RUNNER_TEMP": str(tmp_path)})
        assert path.exists()


class TestLoadDocument:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text(json.dumps(_document([])))
        assert load_document(path)["scope"] == "owner"

    @pytest.mark.parametrize("version", [999, True, 1.0, "1", None])
    def test_unknown_schema_is_refused(self, tmp_path, version):
        path = tmp_path / "results.json"
        path.write_text(json.dumps({"schema_version": version}))
        with pytest.raises(ValueError, match="schema_version"):
            load_document(path)

    def test_non_object_is_refused(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text("[]")
        with pytest.raises(ValueError, match="not a results document"):
            load_document(path)

    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            ({"tool": []}, "'tool' must be a dict"),
            ({"counts": []}, "'counts' must be a dict"),
            ({"results": {}}, "'results' must be a list"),
            ({"scan_errors": "x"}, "'scan_errors' must be a list"),
            ({"results": [1]}, "every entry in 'results'"),
            ({"results": [{"status": "failed", "details": 1}]}, "'details'"),
            ({"counts": {"merged": "1"}}, "every count"),
            ({"counts": {"merged": True}}, "every count"),
        ],
    )
    def test_a_malformed_shape_is_a_bad_file(self, tmp_path, override, expected):
        path = tmp_path / "results.json"
        document = _document([])
        document.update(override)
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match=re.escape(expected)):
            load_document(path)


class TestRenderMarkdown:
    def test_failures_come_before_merges(self):
        markdown = render_markdown(
            _document(
                [
                    MergeResult(_make_pr(1), MergeStatus.MERGED),
                    MergeResult(_make_pr(2), MergeStatus.FAILED, error="boom"),
                ]
            )
        )
        assert markdown.index("### \u274c Failed") < markdown.index("### \u2705 Merged")
        assert "| \u274c Failed | 1 |" in markdown
        assert "[#2](https://github.com/acme/widget/pull/2)" in markdown
        assert "boom" in markdown

    def test_preview_labels_say_what_would_happen(self):
        markdown = render_markdown(
            _document(
                [
                    MergeResult(_make_pr(1), MergeStatus.MERGED),
                    MergeResult(_make_pr(2), MergeStatus.FAILED, error="x"),
                ],
                preview=True,
                dry_run=True,
            )
        )
        assert "**Mode:** dry run" in markdown
        assert "### \u2705 Mergeable" in markdown
        assert "### \u274c Would fail" in markdown

    def test_cells_cannot_break_the_table(self):
        hostile = "a | b <script>\nnew row"
        markdown = render_markdown(
            _document([MergeResult(_make_pr(1, title=hostile), MergeStatus.MERGED)])
        )
        row = next(line for line in markdown.splitlines() if "[#1]" in line)
        assert "a \\| b &lt;script&gt; new row" in row
        # Four columns' worth of unescaped separators, no more.
        assert len(re.findall(r"(?<!\\)\|", row)) == 4

    @pytest.mark.parametrize(
        "title",
        [
            "[View logs](https://attacker.example)",
            "![status](https://attacker.example/x.png)",
            "see https://attacker.example or www.attacker.example",
            "**bold** _em_ `code` <b>x</b>",
        ],
    )
    def test_titles_stay_literal_text(self, title):
        markdown = render_markdown(
            _document([MergeResult(_make_pr(1, title=title), MergeStatus.MERGED)])
        )
        row = next(line for line in markdown.splitlines() if "[#1]" in line)
        title_cell = row.split(" | ")[2]
        # No live link, image, emphasis or autolink: every mark is escaped.
        assert not re.search(r"(?<!\\)[\[\]()*_`!:.]", title_cell)
        assert "<" not in title_cell

    def test_only_plain_http_urls_become_links(self):
        pr = _make_pr(1)
        hostile = pr.model_copy(update={"html_url": "javascript:alert(1)"})
        markdown = render_markdown(
            _document([MergeResult(hostile, MergeStatus.MERGED)])
        )
        assert "javascript" not in markdown
        assert "| #1 |" in markdown

    def test_nothing_to_do_says_so(self):
        markdown = render_markdown(_document([], selection="Limited to: alpha"))
        assert "No pull requests to process." in markdown
        assert "**Scope:** Limited to: alpha" in _shown(markdown)
        assert "| Outcome |" not in markdown

    def test_unscanned_repositories_are_listed(self):
        markdown = render_markdown(
            _document([], scan_errors=["Error scanning repository acme/x: boom"])
        )
        assert "Repositories not scanned" in markdown
        assert "- Error scanning repository acme/x: boom" in _shown(markdown)


class TestModuleEntryPoint:
    def test_renders_a_results_file(self, tmp_path, capsys):
        path = tmp_path / "results.json"
        path.write_text(json.dumps(_document([])))
        assert main(["markdown", str(path)]) == 0
        assert "Dependamerge results" in capsys.readouterr().out

    def test_bad_usage_exits_2(self, capsys):
        assert main(["html"]) == 2
        assert "usage" in capsys.readouterr().err

    def test_outputs_lists_every_count_and_the_total(self, tmp_path, capsys):
        path = tmp_path / "results.json"
        document = _document(
            [
                MergeResult(_make_pr(1), MergeStatus.MERGED),
                MergeResult(_make_pr(2), MergeStatus.BLOCKED, error="x"),
            ]
        )
        path.write_text(json.dumps(document))
        assert main(["outputs", str(path)]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "merged=1" in lines
        assert "blocked=1" in lines
        assert "failed=0" in lines
        assert lines[-1] == "total=2"
        assert all(re.fullmatch(r"[a-z_]+=\d+", line) for line in lines)

    def test_a_missing_file_exits_1(self, tmp_path, capsys):
        assert main(["markdown", str(tmp_path / "absent.json")]) == 1
        assert "dependamerge.ci_report" in capsys.readouterr().err

    def test_a_malformed_file_exits_1_without_a_traceback(self, tmp_path, capsys):
        path = tmp_path / "results.json"
        path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "tool": []}))
        assert main(["markdown", str(path)]) == 1
        assert "'tool' must be a dict" in capsys.readouterr().err


class TestCliPublishesResults:
    runner = CliRunner()

    @pytest.fixture
    def actions_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        output = tmp_path / "github_output"
        output.touch()
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        return output

    @staticmethod
    def _published(output: Path) -> dict[str, Any]:
        key, _, value = output.read_text().strip().partition("=")
        assert key == OUTPUT_KEY
        return load_document(Path(value))

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_failed_run_publishes_before_exiting(
        self, mock_asyncio_run, mock_client_class, actions_env
    ):
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run(
            [
                ([_make_pr(1), _make_pr(2)], ["Error scanning repository acme/x: e"]),
                _PERMS_OK,
                [
                    MergeResult(_make_pr(1), MergeStatus.MERGED),
                    MergeResult(_make_pr(2), MergeStatus.FAILED, error="boom"),
                ],
            ]
        )

        result = self.runner.invoke(
            app,
            ["merge", "acme", "--token", "t", "--no-confirm", "--no-progress"],
        )

        assert result.exit_code == ExitCode.MERGE_ERROR, result.stdout
        document = self._published(actions_env)
        assert document["scope"] == "owner"
        assert document["preview"] is False
        assert document["counts"]["merged"] == 1
        assert document["counts"]["failed"] == 1
        assert document["scan_errors"] == ["Error scanning repository acme/x: e"]

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_nothing_to_merge_still_publishes(
        self, mock_asyncio_run, mock_client_class, actions_env
    ):
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run([([], [])])

        result = self.runner.invoke(
            app, ["merge", "acme", "--token", "t", "--dry-run", "--no-progress"]
        )

        assert result.exit_code == 0, result.stdout
        document = self._published(actions_env)
        assert document["results"] == []
        assert document["preview"] is True
        assert document["dry_run"] is True

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_outside_actions_nothing_is_written(
        self, mock_asyncio_run, mock_client_class, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run([([], [])])

        result = self.runner.invoke(
            app, ["merge", "acme", "--token", "t", "--dry-run", "--no-progress"]
        )

        assert result.exit_code == 0, result.stdout
        assert list(tmp_path.iterdir()) == []

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_a_credential_in_the_target_is_redacted(
        self, mock_asyncio_run, mock_client_class, actions_env
    ):
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run([([], [])])

        result = self.runner.invoke(
            app,
            [
                "merge",
                "https://github.com/acme?token=hunter2#frag",
                "--token",
                "t",
                "--dry-run",
                "--no-progress",
            ],
        )

        assert result.exit_code == 0, result.stdout
        document = self._published(actions_env)
        assert "hunter2" not in json.dumps(document)
        assert "frag" not in document["target"]
        assert document["target"].startswith("https://github.com/acme")

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_an_unconfirmed_preview_still_publishes(
        self, mock_asyncio_run, mock_client_class, actions_env
    ):
        # Without --no-confirm an unattended step reads EOF at the prompt;
        # the preview is then the run's result and must be published.
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run(
            [
                ([_make_pr(1)], []),
                _PERMS_OK,
                [MergeResult(_make_pr(1), MergeStatus.MERGED)],
            ]
        )

        result = self.runner.invoke(
            app, ["merge", "acme", "--token", "t", "--no-progress"], input=""
        )

        assert result.exit_code == 0, result.stdout
        document = self._published(actions_env)
        assert document["preview"] is True
        assert document["counts"]["merged"] == 1

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_a_confirmed_run_keeps_what_the_preview_rejected(
        self, mock_asyncio_run, mock_client_class, actions_env
    ):
        # The confirmed pass re-runs only PR 1, which the preview judged
        # mergeable; PR 2, blocked in the preview, must stay on record.
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run(
            [
                ([_make_pr(1), _make_pr(2)], []),
                _PERMS_OK,
                [
                    MergeResult(_make_pr(1), MergeStatus.MERGED),
                    MergeResult(_make_pr(2), MergeStatus.BLOCKED, error="conflict"),
                ],
                [MergeResult(_make_pr(1), MergeStatus.MERGED)],
            ]
        )
        token = hashlib.sha256(b"org-merge:acme:1").hexdigest()[:16]

        result = self.runner.invoke(
            app,
            ["merge", "acme", "--token", "t", "--no-progress"],
            input=f"{token}\n",
        )

        # Everything attempted merged, so the exit code is unchanged by
        # recording the PR the preview rejected.
        assert result.exit_code == 0, result.stdout
        document = self._published(actions_env)
        assert document["preview"] is False
        assert document["counts"]["merged"] == 1
        assert document["counts"]["blocked"] == 1
        assert {e["number"] for e in document["results"]} == {1, 2}

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_a_confirmed_run_exits_7_when_an_attempted_pr_fails(
        self, mock_asyncio_run, mock_client_class, actions_env
    ):
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run(
            [
                ([_make_pr(1)], []),
                _PERMS_OK,
                [MergeResult(_make_pr(1), MergeStatus.MERGED)],
                [MergeResult(_make_pr(1), MergeStatus.FAILED, error="boom")],
            ]
        )
        token = hashlib.sha256(b"org-merge:acme:1").hexdigest()[:16]

        result = self.runner.invoke(
            app,
            ["merge", "acme", "--token", "t", "--no-progress"],
            input=f"{token}\n",
        )

        assert result.exit_code == ExitCode.MERGE_ERROR, result.stdout
        assert self._published(actions_env)["counts"]["failed"] == 1

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_declined_human_prs_still_publish_owner_wide(
        self, mock_asyncio_run, mock_client_class, actions_env
    ):
        human = _make_pr(1).model_copy(update={"author": "alice"})
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run([([human], [])])

        result = self.runner.invoke(
            app,
            ["merge", "acme", "--token", "t", "--include-human-prs", "--no-progress"],
            input="\n",
        )

        assert result.exit_code == 0, result.stdout
        document = self._published(actions_env)
        assert document["results"] == []
        assert document["preview"] is True

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_declined_human_prs_still_publish_for_a_repository(
        self, mock_asyncio_run, mock_client_class, actions_env
    ):
        human = _make_pr(1).model_copy(update={"author": "alice"})
        mock_client = Mock(token="test_token")
        mock_client.is_automation_author.return_value = False
        mock_client_class.return_value = mock_client
        mock_asyncio_run.side_effect = _closing_run([_PERMS_OK, [human]])

        result = self.runner.invoke(
            app,
            [
                "merge",
                "acme/widget",
                "--token",
                "t",
                "--include-human-prs",
                "--no-progress",
            ],
            input="\n",
        )

        assert result.exit_code == 0, result.stdout
        document = self._published(actions_env)
        assert document["scope"] == "repository"
        assert document["results"] == []

    def test_a_single_pr_preview_publishes_before_any_prompt(self, actions_env):
        # The single-PR route's unconfirmed exits, including the test-mode
        # return this suite itself takes, must publish the preview too.
        from dependamerge.cli._context import _MergeContext
        from dependamerge.cli._merge_scan import _handle_preview_confirmation

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
        ctx.scope = "pull_request"
        ctx.github_client = Mock()
        ctx.github_client.get_pull_request_commits.return_value = ["Bump foo"]
        ctx.source_pr = _make_pr(1)
        results = [MergeResult(_make_pr(1), MergeStatus.MERGED)]

        _handle_preview_confirmation(ctx, results, [(_make_pr(1), None)], 1, 1)

        document = self._published(actions_env)
        assert document["scope"] == "pull_request"
        assert document["preview"] is True
        assert document["counts"]["merged"] == 1
