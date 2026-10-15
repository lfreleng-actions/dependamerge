# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Tests for collecting bot policy files during a run.

Reading them must cost no request of its own: an owner's file arrives
with the profile query that tells an organization from a user, and a
repository's with the first page of its open pull requests.  GitHub is
answered over HTTP, through the real client, so the response shapes it
returns for missing repositories are exercised end to end.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from dependamerge.bot_policy_files import PolicyFiles
from dependamerge.github_graphql import (
    ORG_REPOS_ONLY,
    OWNER_PROFILE,
    REPO_OPEN_PRS_PAGE,
    USER_REPOS_ONLY,
)
from dependamerge.github_service import GitHubService
from dependamerge.models import ComparisonResult, PullRequestInfo

CODE_MONKEY = "lf-releng-code-monkey-bot"

Answer = Callable[[str, dict[str, Any]], dict[str, Any]]


def _blob(text: str) -> dict[str, Any]:
    return {
        "text": text,
        "byteSize": len(text.encode("utf-8")),
        "isTruncated": False,
        "isBinary": False,
    }


def _file(**lists: Any) -> dict[str, Any]:
    return _blob(json.dumps(lists))


def _missing(alias: str) -> dict[str, Any]:
    """The error GitHub adds when a repository alias names nothing."""
    return {
        "type": "NOT_FOUND",
        "path": ["repositoryOwner", alias],
        "message": "Could not resolve to a Repository with the name 'acme/x'.",
    }


def _profile(typename: str | None, *errors: Any, **repos: Any) -> dict[str, Any]:
    owner = None if typename is None else {"__typename": typename, **repos}
    body: dict[str, Any] = {"data": {"repositoryOwner": owner}}
    if errors:
        body["errors"] = list(errors)
    return body


def _listing(root: str, *names: str) -> dict[str, Any]:
    nodes = [{"nameWithOwner": n, "isArchived": False, "isFork": False} for n in names]
    page = {"hasNextPage": False, "endCursor": None}
    connection = {"totalCount": len(nodes), "pageInfo": page, "nodes": nodes}
    return {"data": {root: {"repositories": connection}}}


def _service(
    answer: Answer,
) -> tuple[GitHubService, list[tuple[str, dict[str, Any]]]]:
    """A service whose GraphQL requests ``answer`` handles, and what it sent."""
    sent: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append((payload["query"], payload["variables"]))
        return httpx.Response(200, json=answer(payload["query"], payload["variables"]))

    svc = GitHubService(token="t")
    svc._api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return svc, sent


async def _list(svc: GitHubService, owner: str = "acme") -> list[str]:
    return [r["nameWithOwner"] async for r in svc._iter_owner_repositories(owner)]


class TestOwnerProfile:
    @pytest.mark.asyncio
    async def test_an_organization_needs_one_profile_and_one_listing_request(
        self,
    ) -> None:
        policy = _file(include_bots=[CODE_MONKEY])

        def answer(query: str, variables: dict[str, Any]) -> dict[str, Any]:
            if query == OWNER_PROFILE:
                return _profile("Organization", dotgithub={"policy": policy})
            return _listing("organization", "acme/a", "acme/b")

        svc, sent = _service(answer)
        try:
            assert await _list(svc) == ["acme/a", "acme/b"]
        finally:
            await svc.close()
        # The first page of repositories is fetched once, to list them.
        assert [query for query, _ in sent] == [OWNER_PROFILE, ORG_REPOS_ONLY]
        assert sent[0][1] == {"owner": "acme", "repo": "", "withRepo": False}
        assert svc.policy_files == PolicyFiles(owners={"acme": policy})

    @pytest.mark.asyncio
    async def test_a_user_without_a_dotgithub_repository_is_listed_as_a_user(
        self,
    ) -> None:
        def answer(query: str, variables: dict[str, Any]) -> dict[str, Any]:
            if query == OWNER_PROFILE:
                # GitHub reports the missing repository as an error too.
                return _profile("User", _missing("dotgithub"), dotgithub=None)
            return _listing("user", "auser/a")

        svc, sent = _service(answer)
        try:
            assert await _list(svc, "AUser") == ["auser/a"]
        finally:
            await svc.close()
        assert [query for query, _ in sent] == [OWNER_PROFILE, USER_REPOS_ONLY]
        assert svc.policy_files == PolicyFiles(owners={"auser": None})

    @pytest.mark.asyncio
    async def test_an_unknown_owner_is_tried_as_a_user_and_records_nothing(
        self,
    ) -> None:
        svc, sent = _service(lambda query, variables: _profile(None))
        try:
            assert await svc._resolve_owner_root("ghost") == ("user", USER_REPOS_ONLY)
        finally:
            await svc.close()
        assert len(sent) == 1
        assert svc.policy_files == PolicyFiles()

    @pytest.mark.asyncio
    async def test_files_fetched_first_settle_the_owner_for_listing(self) -> None:
        widget = _file(exclude_bots="renovate")

        def answer(query: str, variables: dict[str, Any]) -> dict[str, Any]:
            if query == OWNER_PROFILE:
                return _profile(
                    "Organization",
                    dotgithub={"policy": {}},
                    target={"policy": widget},
                )
            return _listing("organization", "acme/widget")

        svc, sent = _service(answer)
        try:
            files = await svc.fetch_policy_files("Acme", "Widget")
            assert await _list(svc, "Acme") == ["acme/widget"]
        finally:
            await svc.close()
        assert sent[0][1] == {"owner": "Acme", "repo": "Widget", "withRepo": True}
        assert [query for query, _ in sent] == [OWNER_PROFILE, ORG_REPOS_ONLY]
        # A directory at the path is kept as GitHub returned it: not a file.
        # The variables were read too, and this fake API holds none.
        assert files == PolicyFiles(
            owners={"acme": {}},
            repositories={"acme/widget": widget},
            owner_variables={"acme": None},
            repository_variables={"acme/widget": None},
        )

    @pytest.mark.asyncio
    async def test_a_missing_target_repository_records_no_file_for_it(self) -> None:
        def answer(query: str, variables: dict[str, Any]) -> dict[str, Any]:
            return _profile(
                "Organization",
                _missing("dotgithub"),
                _missing("target"),
                dotgithub=None,
                target=None,
            )

        svc, _ = _service(answer)
        try:
            files = await svc.fetch_policy_files("acme", "nope")
        finally:
            await svc.close()
        # The owner's variable is read too (absent here); the missing
        # target repository has neither a file nor a variable read.
        assert files == PolicyFiles(
            owners={"acme": None}, owner_variables={"acme": None}
        )


def _pr_page(
    name: str, numbers: list[int], *, cursor: str | None = None, **policy: Any
) -> dict[str, Any]:
    """One page of open pull requests; ``policy`` only when asked for."""
    nodes = [
        {
            "number": n,
            "title": f"Chore: Bump x to {n}",
            "url": f"https://github.com/acme/{name}/pull/{n}",
            # Odd numbers are the code monkey's, even ones Dependabot's.
            "author": {
                "__typename": "Bot",
                "login": CODE_MONKEY if n % 2 else "dependabot",
            },
        }
        for n in numbers
    ]
    page_info = {"hasNextPage": cursor is not None, "endCursor": cursor}
    repository = {
        "nameWithOwner": f"acme/{name}",
        **policy,
        "pullRequests": {"pageInfo": page_info, "nodes": nodes},
    }
    return {"data": {"repository": repository}}


class TestFirstPullRequestPage:
    @pytest.mark.asyncio
    async def test_only_the_first_page_asks_for_the_repository_file(self) -> None:
        widget = _file(include_bots="*")

        def answer(query: str, variables: dict[str, Any]) -> dict[str, Any]:
            assert query == REPO_OPEN_PRS_PAGE
            if variables["prsCursor"] is None:
                return _pr_page("widget", [1], policy=widget, cursor="c1")
            return _pr_page("widget", [2])

        svc, sent = _service(answer)
        try:
            nodes, page_info = await svc._fetch_repo_prs_first_page("Acme", "Widget")
            rest = [
                node
                async for node in svc._iter_repo_open_prs_pages(
                    "Acme", "Widget", page_info["endCursor"]
                )
            ]
        finally:
            await svc.close()
        assert [node["number"] for node in nodes + rest] == [1, 2]
        assert [variables["withPolicy"] for _, variables in sent] == [True, False]
        files = svc.policy_files
        assert files == PolicyFiles(repositories={"acme/widget": widget})
        # A snapshot: what the run reads later does not change it.
        with pytest.raises(TypeError):
            files.repositories["acme/other"] = None  # type: ignore[index]


class _AllSimilar:
    def compare_pull_requests(self, source, target, only_automation):
        return ComparisonResult(is_similar=True, confidence_score=1.0, reasons=["x"])


class TestSimilarPullRequests:
    @pytest.mark.asyncio
    async def test_each_repository_is_judged_by_its_own_file(self) -> None:
        widget = _file(include_bots=[CODE_MONKEY])
        pages = {
            "widget": _pr_page("widget", [3, 4], policy=widget),
            "other": _pr_page("other", [5, 6], policy=None),
            # A directory where the file should be: invalid.
            "broken": _pr_page("broken", [7, 8], policy={}),
        }

        def answer(query: str, variables: dict[str, Any]) -> dict[str, Any]:
            if query == OWNER_PROFILE:
                return _profile("Organization", dotgithub={"policy": None})
            if query == ORG_REPOS_ONLY:
                return _listing("organization", *(f"acme/{n}" for n in pages))
            return pages[variables["name"]]

        source = PullRequestInfo(
            number=1,
            title="Chore: Bump x to 1",
            body=None,
            author=f"{CODE_MONKEY}[bot]",
            head_sha="abc",
            base_branch="main",
            head_branch="update",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[],
            repository_full_name="acme/widget",
            html_url="https://github.com/acme/widget/pull/1",
        )
        svc, _ = _service(answer)
        try:
            found = await svc.find_similar_prs(
                "acme", source, _AllSimilar(), only_automation=True
            )
        finally:
            await svc.close()
        # The code monkey is a candidate only where the file includes it;
        # nothing in the repository whose file is invalid is.
        assert sorted(
            (pr.repository_full_name, pr.number, pr.author) for pr, _ in found
        ) == [
            ("acme/other", 6, "dependabot[bot]"),
            ("acme/widget", 3, f"{CODE_MONKEY}[bot]"),
            ("acme/widget", 4, "dependabot[bot]"),
        ]
        # The files the search read are the run's too.
        assert svc.policy_files.repositories == {
            "acme/widget": widget,
            "acme/other": None,
            "acme/broken": {},
        }
