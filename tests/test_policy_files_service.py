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
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from dependamerge.bot_policy_files import PolicyFiles
from dependamerge.cli._policy_files import _warn_unreadable_variables
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
    answer: Answer, variables: dict[str, str] | None = None
) -> tuple[GitHubService, list[tuple[str, dict[str, Any]]]]:
    """A service whose GraphQL requests ``answer`` handles, and what it sent.

    ``variables`` holds repositories' ``DEPENDAMERGE_BOTS`` values; a
    repository it does not name has none.
    """
    sent: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "/actions/variables/" in request.url.path:
            scope = request.url.path.split("/actions/")[0]
            value = (variables or {}).get(scope.removeprefix("/repos/"))
            if value is None:
                # GitHub's answer for an unset variable.
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json={"name": "x", "value": value})
        payload = json.loads(request.content)
        sent.append((payload["query"], payload["variables"]))
        return httpx.Response(200, json=answer(payload["query"], payload["variables"]))

    svc = GitHubService(token="t", read_policies=True)
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
        assert sent[0][1] == {
            "owner": "acme",
            "repo": "",
            "withRepo": False,
            "withPolicy": True,
        }
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
        assert sent[0][1] == {
            "owner": "Acme",
            "repo": "Widget",
            "withRepo": True,
            "withPolicy": True,
        }
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


def _author(number: int, people: tuple[int, ...]) -> dict[str, str]:
    if number in people:
        return {"__typename": "User", "login": "octocat"}
    login = CODE_MONKEY if number % 2 else "dependabot"
    return {"__typename": "Bot", "login": login}


def _pr_page(
    name: str,
    numbers: list[int],
    *,
    cursor: str | None = None,
    people: tuple[int, ...] = (),
    **policy: Any,
) -> dict[str, Any]:
    """One page of open pull requests; ``policy`` only when asked for.

    Those in ``people`` are octocat's, other odd numbers the code
    monkey's, and even ones Dependabot's.
    """
    nodes = [
        {
            "number": n,
            "title": f"Chore: Bump x to {n}",
            "url": f"https://github.com/acme/{name}/pull/{n}",
            "author": _author(n, people),
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

    @pytest.mark.asyncio
    async def test_a_reporting_service_reads_no_policy(self) -> None:
        # status and blocked share these queries but never act on PRs, so
        # a token that cannot read the files must not break their scans
        # (Copilot review on #567).
        def answer(query: str, variables: dict[str, Any]) -> dict[str, Any]:
            if query == OWNER_PROFILE:
                return _profile("Organization")
            if query == REPO_OPEN_PRS_PAGE:
                return _pr_page("widget", [1])
            return _listing("organization", "acme/widget")

        svc, sent = _service(answer)
        svc._read_policies = False
        try:
            await _list(svc, "acme")
            await svc._fetch_repo_prs_first_page("acme", "widget")
        finally:
            await svc.close()
        flags = [v["withPolicy"] for _, v in sent if "withPolicy" in v]
        assert flags == [False, False]
        assert svc.policy_files == PolicyFiles()


class _AllSimilar:
    def compare_pull_requests(self, source, target, only_automation):
        return ComparisonResult(is_similar=True, confidence_score=1.0, reasons=["x"])


class _NoneSimilar:
    def compare_pull_requests(self, source, target, only_automation):
        return ComparisonResult(is_similar=False, confidence_score=0.0, reasons=[])


def _source(author: str, repository: str) -> PullRequestInfo:
    return PullRequestInfo(
        number=1,
        title="Chore: Bump x to 1",
        body=None,
        author=author,
        head_sha="abc",
        base_branch="main",
        head_branch="update",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name=repository,
        html_url=f"https://github.com/{repository}/pull/1",
    )


def _organisation(pages: dict[str, dict[str, Any]]) -> Answer:
    """Answers for an organisation holding ``pages``' repositories."""

    def answer(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if query == OWNER_PROFILE:
            return _profile("Organization", dotgithub={"policy": None})
        if query == ORG_REPOS_ONLY:
            return _listing("organization", *(f"acme/{n}" for n in pages))
        return pages[variables["name"]]

    return answer


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
        source = _source(f"{CODE_MONKEY}[bot]", "acme/widget")
        svc, _ = _service(_organisation(pages))
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

    @pytest.mark.asyncio
    async def test_a_persons_search_reads_variables_where_they_matter(self) -> None:
        # The policy judges none of a person's PRs, but an invalid one
        # keeps the search out of its repository; that needs the variable
        # where an App's PR is open beside the person's own (Copilot review
        # on #567).
        pages = {
            # The file excludes the code monkey; the variable includes it.
            "widget": _pr_page(
                "widget", [2, 10], people=(10,), policy=_file(exclude_bots=CODE_MONKEY)
            ),
            # Nothing of octocat's to act on.
            "quiet": _pr_page("quiet", [4], policy=None),
            # No App's PR, so no policy decides anything, as in every run.
            "people": _pr_page("people", [20], people=(20,), policy=None),
        }
        unparsable = "{not json"
        svc, _ = _service(
            _organisation(pages),
            variables={
                "acme/widget": json.dumps({"include_bots": [CODE_MONKEY]}),
                "acme/quiet": unparsable,
                "acme/people": unparsable,
            },
        )
        try:
            found = await svc.find_similar_prs(
                "acme",
                _source("octocat", "acme/home"),
                _AllSimilar(),
                only_automation=False,
            )
        finally:
            await svc.close()
        assert [(pr.repository_full_name, pr.number) for pr, _ in found] == [
            ("acme/people", 20)
        ]
        assert list(svc.policy_files.repository_variables) == ["acme/widget"]

    @pytest.mark.asyncio
    async def test_every_app_pr_seen_is_kept_matching_or_not(self) -> None:
        # Whether an unreadable variable is worth a warning depends on
        # the App PRs the search saw, not only those it kept: an App no
        # policy admits, or whose PR does not match, is still one a
        # variable might have decided (Copilot review on #585).
        pages = {
            # The code monkey: no file admits it, so not a candidate.
            "other": _pr_page("other", [3, 4], policy=None),
            # Admitted here, but nothing matches.
            "widget": _pr_page(
                "widget", [5, 6], people=(6,), policy=_file(include_bots=[CODE_MONKEY])
            ),
        }
        svc, _ = _service(_organisation(pages))
        try:
            found = await svc.find_similar_prs(
                "acme",
                _source("dependabot[bot]", "acme/home"),
                _NoneSimilar(),
                only_automation=True,
            )
        finally:
            await svc.close()
        assert found == []
        # People's PRs are not kept: no policy judges them.
        assert sorted(
            (pr.repository_full_name, pr.number, pr.author) for pr in svc.seen_app_prs
        ) == [
            ("acme/other", 3, f"{CODE_MONKEY}[bot]"),
            ("acme/other", 4, "dependabot[bot]"),
            ("acme/widget", 5, f"{CODE_MONKEY}[bot]"),
        ]

    @pytest.mark.asyncio
    async def test_an_invalid_repositorys_app_prs_are_still_seen(self, capsys) -> None:
        # A repository whose policy is invalid has nothing in scope, but
        # its App PRs still decide whether an unreadable variable is worth
        # a warning; a directory where its file should be covers neither
        # level (Copilot review on #585).
        pages = {"broken": _pr_page("broken", [3, 4, 10], people=(10,), policy={})}
        svc, _ = _service(_organisation(pages))
        try:
            found = await svc.find_similar_prs(
                "acme",
                _source("dependabot[bot]", "acme/home"),
                _AllSimilar(),
                only_automation=True,
            )
        finally:
            await svc.close()
        assert found == []
        assert sorted((pr.number, pr.author) for pr in svc.seen_app_prs) == [
            (3, f"{CODE_MONKEY}[bot]"),
            (4, "dependabot[bot]"),
        ]
        files = replace(
            svc.policy_files, variables_refused=frozenset({"orgs", "repos"})
        )
        _warn_unreadable_variables(
            SimpleNamespace(policy_files=files), svc.seen_app_prs
        )
        out = capsys.readouterr().out
        assert "cannot read the organisation DEPENDAMERGE_BOTS variable" in out
        assert "cannot read repository DEPENDAMERGE_BOTS variables" in out
