# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for limiting owner-wide merges with --include-repos/--exclude-repos."""

import re
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.github_service import GitHubService
from dependamerge.models import PullRequestInfo
from dependamerge.repo_selection import parse_repo_selection


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_pr(number: int, repo: str) -> PullRequestInfo:
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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseRepoSelection:
    def test_neither_option_selects_everything(self):
        assert parse_repo_selection("acme", None, None) is None

    def test_options_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            parse_repo_selection("acme", ["alpha"], ["beta"])

    @pytest.mark.parametrize("values", [[""], [","], [" , "], []])
    def test_naming_nothing_is_refused(self, values):
        # An explicit narrowing must never widen to every repository.
        with pytest.raises(ValueError, match="names no repository"):
            parse_repo_selection("acme", values, None)

    def test_commas_whitespace_and_repeats_all_separate(self):
        selection = parse_repo_selection(
            "acme", ["alpha,beta", "gamma\ndelta", " epsilon "], None
        )
        assert selection is not None
        assert list(selection.names.values()) == [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
        ]

    def test_owner_qualified_names_for_this_owner_are_accepted(self):
        selection = parse_repo_selection("acme", ["ACME/alpha"], None)
        assert selection is not None
        assert list(selection.names) == ["alpha"]

    def test_names_for_another_owner_are_refused(self):
        with pytest.raises(ValueError, match="belongs to 'other'"):
            parse_repo_selection("acme", ["other/alpha"], None)

    def test_too_many_path_segments_are_refused(self):
        with pytest.raises(ValueError, match="not 'name' or 'owner/name'"):
            parse_repo_selection("acme", None, ["acme/alpha/beta"])

    def test_malformed_names_are_refused(self):
        with pytest.raises(ValueError, match="--exclude-repos"):
            parse_repo_selection("acme", None, ["has:colon"])

    def test_duplicates_collapse_case_insensitively(self):
        selection = parse_repo_selection("acme", ["Alpha", "alpha", "ALPHA"], None)
        assert selection is not None
        assert selection.names == {"alpha": "Alpha"}


# ---------------------------------------------------------------------------
# Selection behaviour
# ---------------------------------------------------------------------------


class TestRepoSelection:
    def test_include_admits_only_named_repositories(self):
        selection = parse_repo_selection("acme", ["alpha"], None)
        assert selection is not None
        assert selection.admits("acme/Alpha")
        assert not selection.admits("acme/beta")

    def test_exclude_admits_everything_else(self):
        selection = parse_repo_selection("acme", None, ["alpha"])
        assert selection is not None
        assert not selection.admits("acme/alpha")
        assert selection.admits("acme/beta")

    def test_unmatched_reports_names_never_enumerated(self):
        selection = parse_repo_selection("acme", ["alpha", "Typo"], None)
        assert selection is not None
        selection.admits("acme/alpha")
        selection.admits("acme/beta")
        assert selection.unmatched() == ["Typo"]

    def test_describe_names_the_scope(self):
        include = parse_repo_selection("acme", ["alpha", "beta"], None)
        exclude = parse_repo_selection("acme", None, ["gamma"])
        assert include is not None and exclude is not None
        assert include.describe() == "Limited to: alpha, beta"
        assert exclude.describe() == "Excluding: gamma"


# ---------------------------------------------------------------------------
# Service: fetch_owner_open_prs honours the selection
# ---------------------------------------------------------------------------


class TestFetchOwnerOpenPrsSelection:
    @pytest.mark.asyncio
    async def test_unselected_repositories_are_never_scanned(self):
        progress = Mock()
        svc = GitHubService(token="test_token", progress_tracker=progress)
        scanned: list[str] = []

        async def fake_iter(owner):
            for name in ("acme/alpha", "acme/beta", "acme/gamma"):
                yield {"nameWithOwner": name}

        async def fake_collect(owner, repo, *, only_automation):
            scanned.append(repo)
            return [_make_pr(1, f"{owner}/{repo}")]

        svc._iter_owner_repositories = fake_iter  # type: ignore[assignment]
        svc._collect_repo_open_prs = fake_collect  # type: ignore[assignment]

        selection = parse_repo_selection("acme", None, ["beta"])
        prs, errors = await svc.fetch_owner_open_prs("acme", selection=selection)
        await svc.close()

        assert sorted(scanned) == ["alpha", "gamma"]
        assert sorted(pr.repository_full_name for pr in prs) == [
            "acme/alpha",
            "acme/gamma",
        ]
        assert errors == []
        # The progress total is the admitted count, not the owner's.
        progress.update_total_repositories.assert_called_with(2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PERMS_OK = {
    "approve": {"has_permission": True},
    "merge": {"has_permission": True},
    "branch_protection": {"has_permission": True},
}


def _closing_run(results: list[object]):
    """An ``asyncio.run`` stand-in returning ``results`` in order."""
    remaining = list(results)

    def _run(coro):
        coro.close()
        return remaining.pop(0)

    return _run


class TestMergeRepoSelectionCli:
    runner = CliRunner()

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_selection_reaches_the_scan_and_is_announced(
        self, mock_asyncio_run, mock_client_class
    ):
        mock_client_class.return_value = Mock(token="test_token")
        mock_asyncio_run.side_effect = _closing_run([_PERMS_OK, []])
        seen = {}

        def fake_fetch(parsed_org, ctx, *, only_automation):
            seen["selection"] = ctx.repo_selection
            ctx.repo_selection.admits("acme/alpha")
            return [_make_pr(1, "acme/alpha")], []

        with patch("dependamerge.cli._org_merge._fetch_owner_prs", fake_fetch):
            result = self.runner.invoke(
                app,
                [
                    "merge",
                    "https://github.com/acme",
                    "--token",
                    "test_token",
                    "--include-repos",
                    "alpha",
                ],
            )

        assert result.exit_code == 0, result.stdout
        assert list(seen["selection"].names) == ["alpha"]
        assert "Limited to: alpha" in _strip_ansi(result.stdout)

    @patch("dependamerge.cli.GitHubClient")
    @patch("dependamerge.cli.asyncio.run")
    def test_unmatched_name_stops_before_anything_merges(
        self, mock_asyncio_run, mock_client_class
    ):
        mock_client_class.return_value = Mock(token="test_token")

        def fake_fetch(parsed_org, ctx, *, only_automation):
            ctx.repo_selection.admits("acme/alpha")
            return [_make_pr(1, "acme/alpha")], []

        with patch("dependamerge.cli._org_merge._fetch_owner_prs", fake_fetch):
            result = self.runner.invoke(
                app,
                [
                    "merge",
                    "https://github.com/acme",
                    "--token",
                    "test_token",
                    "--exclude-repos",
                    "no-such-repo",
                ],
            )

        assert result.exit_code == 2, result.stdout
        assert "no-such-repo" in _strip_ansi(result.stdout)
        # Neither the permission check nor a merge pass ever ran.
        mock_asyncio_run.assert_not_called()

    def test_refused_for_a_repository_target(self):
        result = self.runner.invoke(
            app,
            [
                "merge",
                "https://github.com/acme/widget",
                "--token",
                "test_token",
                "--include-repos",
                "widget",
            ],
        )
        assert result.exit_code == 2, result.stdout
        assert "owner-wide" in _strip_ansi(result.stdout)

    def test_both_options_are_refused(self):
        result = self.runner.invoke(
            app,
            [
                "merge",
                "https://github.com/acme",
                "--token",
                "test_token",
                "--include-repos",
                "alpha",
                "--exclude-repos",
                "beta",
            ],
        )
        assert result.exit_code == 2, result.stdout
        assert "mutually exclusive" in _strip_ansi(result.stdout)
