# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Every path that reads a pull request's author classifies it by identity.

Whether an author is a bot follows from the account type the API
reports, never from words in a login, title or body.  Three classifiers
used to disagree: one matched any ``[bot]`` login *and* bare tool names
whatever the account type, one matched any login containing "bot", and
one matched "bot" or "automated" anywhere in the title or body.  These
tests pin each data path to the single rule.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.github_client.queries import _build_pull_request_info
from dependamerge.github_service import GitHubService
from dependamerge.github_service._similar import _SimilarPullRequestsMixin
from dependamerge.models import PullRequestInfo
from dependamerge.pr_comparator import PRComparator


def _pr(author: str, *, number: int = 2, body: str = "") -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="Chore: Bump foo from 1.0 to 1.1",
        body=body or "Bumps foo from 1.0 to 1.1",
        author=author,
        head_sha="abc",
        base_branch="main",
        head_branch="topic",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name=f"acme/repo{number}",
        html_url=f"https://github.com/acme/repo{number}/pull/{number}",
    )


def _graphql_node(login: str, typename: str) -> dict[str, Any]:
    return {
        "number": 7,
        "title": "t",
        "url": "https://github.com/acme/widget/pull/7",
        "author": {"__typename": typename, "login": login},
    }


class TestGraphQLBoundary:
    """``to_pull_request_info`` canonicalises by ``__typename``."""

    @pytest.mark.parametrize(
        ("login", "expected"),
        [
            ("dependabot", "dependabot[bot]"),
            ("lf-releng-code-monkey-bot", "lf-releng-code-monkey-bot[bot]"),
            ("copilot-swe-agent", "copilot-swe-agent[bot]"),
        ],
    )
    def test_app_node_gains_suffix(self, login: str, expected: str) -> None:
        svc = GitHubService(token="test_token")
        pr = svc.to_pull_request_info("acme/widget", _graphql_node(login, "Bot"))
        assert pr.author == expected

    @pytest.mark.parametrize("login", ["renovate", "copilot-swe-agent", "talbot"])
    def test_user_node_keeps_bare_login(self, login: str) -> None:
        svc = GitHubService(token="test_token")
        pr = svc.to_pull_request_info("acme/widget", _graphql_node(login, "User"))
        assert pr.author == login


class TestRestBoundary:
    """``_build_pull_request_info`` decides by ``user.type``."""

    def _build(self, user: dict[str, Any]) -> PullRequestInfo:
        payload = {"number": 3, "title": "t", "user": user, "state": "open"}
        return _build_pull_request_info(payload, "acme", "widget", 3, [], [])

    def test_copilot_agent_matches_its_graphql_form(self) -> None:
        pr = self._build({"login": "Copilot", "type": "Bot", "id": 198982749})
        assert pr.author == "copilot-swe-agent[bot]"

    def test_user_account_named_like_a_tool_stays_a_person(self) -> None:
        pr = self._build({"login": "renovate", "type": "User", "id": 1})
        assert pr.author == "renovate"


class TestSimilarPullRequestCandidates:
    """The similar-PR candidate filter uses the identity rule."""

    source = _pr("dependabot[bot]", number=1)

    @pytest.mark.parametrize("login", ["talbot", "abbott", "botanist", "renovate"])
    def test_people_are_not_automation_candidates(self, login: str) -> None:
        assert not _SimilarPullRequestsMixin._is_similar_pr_candidate(
            self.source, _pr(login), only_automation=True
        )

    def test_apps_are_automation_candidates(self) -> None:
        assert _SimilarPullRequestsMixin._is_similar_pr_candidate(
            self.source, _pr("renovate[bot]"), only_automation=True
        )


class TestComparatorAutomationCheck:
    """Words in a pull request say nothing about who wrote it."""

    def test_person_describing_automation_is_not_automation(self) -> None:
        pr = _pr("octocat", body="Automated bot-style auto-update of pins")
        assert not PRComparator()._is_automation_pr(pr)

    def test_app_is_automation(self) -> None:
        assert PRComparator()._is_automation_pr(_pr("dependabot[bot]"))


class TestStatusStatistics:
    """Status counts classify raw GraphQL and REST authors by type."""

    @pytest.mark.asyncio
    async def test_open_and_merged_counts_use_account_type(self) -> None:
        svc = GitHubService(token="test_token")
        open_nodes = [
            {"author": {"__typename": "Bot", "login": "dependabot"}},
            {"author": {"__typename": "User", "login": "renovate"}},
        ]
        merged = [
            {"user": {"login": "pre-commit-ci[bot]", "type": "Bot", "id": 66853113}},
            {"user": {"login": "github-actions", "type": "User", "id": 2}},
        ]
        svc._fetch_repo_prs_first_page = AsyncMock(  # type: ignore[method-assign]
            return_value=(open_nodes, {"hasNextPage": False})
        )
        svc._get_merged_prs_since = AsyncMock(  # type: ignore[method-assign]
            return_value=merged
        )
        try:
            stats = await svc._gather_pr_statistics("acme", "widget", "2026/01/01")
        finally:
            await svc.close()

        assert stats["open_prs_automation"] == 1
        assert stats["open_prs_human"] == 1
        assert stats["merged_prs_automation"] == 1
        assert stats["merged_prs_human"] == 1
