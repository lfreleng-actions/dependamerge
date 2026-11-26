# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Tests for skipping repositories the listing counts no open PR in.

The owner listing carries each repository's open pull request count, so
the scans need not ask an empty repository for its first page.  The
GraphQL transport is replaced by a fake that records every request and
answers for an owner of 150 repositories, 12 of them with open PRs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from dependamerge.github_async import GraphQLError
from dependamerge.github_graphql import (
    ORG_REPOS_ONLY,
    OWNER_PROFILE,
    REPO_OPEN_PRS_PAGE,
    USER_REPOS_ONLY,
)
from dependamerge.github_service import GitHubService
from dependamerge.models import ComparisonResult, PullRequestInfo
from dependamerge.repo_selection import RepoSelection
from tests.conftest import with_selected_lists

REPOSITORIES = 150
ACTIVE = frozenset(range(0, 144, 12))
POOL = 4
POLICY_TEXT = json.dumps({"include_bots": ["renovate"]})
POLICY = {
    "text": POLICY_TEXT,
    "byteSize": len(POLICY_TEXT),
    "isTruncated": False,
    "isBinary": False,
}


def _name(index: int) -> str:
    return f"repo-{index:03d}"


class _Owner:
    """A fake GraphQL endpoint for one owner, recording what it is asked.

    ``counted`` false lists repositories without their open PR count,
    which is what every scan read before the count was listed.  ``count``
    replaces the listed count of chosen repositories, ``delay`` how long
    a repository's page takes, and ``fail`` names repositories whose
    page raises.  ``stall`` holds the second listing page until set, and
    ``hold`` the first repository's page.
    ``variables`` answers a repository's ``DEPENDAMERGE_BOTS`` read with
    an HTTP status or a value, 404 for any other; ``variable_reads``
    records the repositories read.
    """

    def __init__(
        self,
        *,
        root: str = "organization",
        counted: bool = True,
        count: dict[int, Any] | None = None,
        delay: Any = None,
        fail: frozenset[int] = frozenset(),
        stall: asyncio.Event | None = None,
        hold: asyncio.Event | None = None,
        variables: dict[int, int | str] | None = None,
    ) -> None:
        self.root = root
        self.counted = counted
        self.count = count or {}
        self.delay = delay
        self.fail = fail
        self.stall = stall
        self.hold = hold
        self.variables = variables or {}
        self.variable_reads: list[str] = []
        self.listings = 0
        self.pages: list[str] = []
        self.selections: set[frozenset[str]] = set()
        self.in_flight = 0
        self.most_in_flight = 0

    def _node(self, index: int) -> dict[str, Any]:
        node: dict[str, Any] = {
            "nameWithOwner": f"acme/{_name(index)}",
            "isArchived": False,
            "isFork": False,
        }
        if index in self.count:
            node["pullRequests"] = self.count[index]
        elif self.counted:
            node["pullRequests"] = {"totalCount": 2 if index in ACTIVE else 0}
        return node

    def _listing(self, cursor: str | None) -> dict[str, Any]:
        self.listings += 1
        start = 0 if cursor is None else 100
        end = min(start + 100, REPOSITORIES)
        more = end < REPOSITORIES
        return {
            self.root: {
                "repositories": {
                    "totalCount": REPOSITORIES,
                    "pageInfo": {"hasNextPage": more, "endCursor": "p2"},
                    "nodes": [self._node(i) for i in range(start, end)],
                }
            }
        }

    def _page(self, name: str, with_policy: bool) -> dict[str, Any]:
        index = int(name.removeprefix("repo-"))
        if index in self.fail:
            raise RuntimeError(f"{name} cannot be read")
        nodes = []
        if index in ACTIVE:
            nodes = [
                {
                    "number": 1,
                    "title": "Bump foo from 1.0 to 1.1",
                    "url": f"https://github.com/acme/{name}/pull/1",
                    "author": {"__typename": "Bot", "login": "dependabot"},
                    "mergeable": "CONFLICTING",
                    "mergeStateStatus": "DIRTY",
                },
                {
                    "number": 2,
                    "title": "Fix: Correct a typo",
                    "url": f"https://github.com/acme/{name}/pull/2",
                    "author": {"__typename": "User", "login": "octocat"},
                    "mergeable": "MERGEABLE",
                    "mergeStateStatus": "CLEAN",
                },
            ]
        repository: dict[str, Any] = {
            "pullRequests": {
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": nodes,
            }
        }
        if with_policy:
            repository["policy"] = POLICY if index == 12 else None
        return {"repository": repository}

    async def graphql(
        self, query: str, variables: dict[str, Any] | None = None, **_: Any
    ) -> dict[str, Any]:
        variables = variables or {}
        if query == OWNER_PROFILE:
            typename = "Organization" if self.root == "organization" else "User"
            return {"repositoryOwner": {"__typename": typename, "dotgithub": None}}
        if query in (ORG_REPOS_ONLY, USER_REPOS_ONLY):
            expected = (
                ORG_REPOS_ONLY if self.root == "organization" else USER_REPOS_ONLY
            )
            assert query == expected
            if self.stall is not None and variables.get("reposCursor"):
                await self.stall.wait()
            return self._listing(variables.get("reposCursor"))
        assert query == REPO_OPEN_PRS_PAGE
        name = variables["name"]
        self.pages.append(name)
        self.selections.add(
            frozenset(k for k in _LIST_SWITCHES if variables.get(k) is True)
        )
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            if self.hold is not None and name == _name(0):
                await self.hold.wait()
            await asyncio.sleep(self.delay(name) if self.delay else 0.001)
            return with_selected_lists(
                self._page(name, bool(variables.get("withPolicy"))), variables
            )
        finally:
            self.in_flight -= 1

    async def get(self, path: str, params: Any = None) -> Any:
        name = path.split("/")[3]
        self.variable_reads.append(name)
        await asyncio.sleep(0.001)
        answer = self.variables.get(int(name.removeprefix("repo-")), 404)
        if isinstance(answer, int):
            request = httpx.Request("GET", f"https://api.github.com{path}")
            raise httpx.HTTPStatusError(
                "error",
                request=request,
                response=httpx.Response(answer, request=request),
            )
        return {"name": "DEPENDAMERGE_BOTS", "value": answer}


class _Progress:
    """Counts what a scan reports to its progress display."""

    def __init__(self) -> None:
        self.total = 0
        self.completed = 0
        self.errors = 0

    def update_total_repositories(self, total: int) -> None:
        self.total = total

    def complete_repository(self, unmergeable_count: int = 0) -> None:
        self.completed += 1

    def add_error(self) -> None:
        self.errors += 1

    def start_repository(self, repo_name: str) -> None:
        pass

    def update_operation(self, operation: str) -> None:
        pass

    def analyze_pr(self, pr_number: int, repo_name: str = "") -> None:
        pass

    def found_similar_pr(self) -> None:
        pass


class _AllSimilar:
    def compare_pull_requests(self, source, target, only_automation):
        return ComparisonResult(
            is_similar=True, confidence_score=1.0, reasons=[target.repository_full_name]
        )


SOURCE = PullRequestInfo(
    number=7,
    title="Bump foo from 1.0 to 1.1",
    body=None,
    author="dependabot[bot]",
    head_sha="abc",
    base_branch="main",
    head_branch="dependabot/foo",
    state="open",
    mergeable=True,
    mergeable_state="clean",
    behind_by=0,
    files_changed=[],
    repository_full_name="acme/home",
    html_url="https://github.com/acme/home/pull/7",
)

ACTIVE_NAMES = [_name(i) for i in sorted(ACTIVE)]


def _service(owner: _Owner, progress: Any = None) -> GitHubService:
    svc = GitHubService(
        token="t", read_policies=True, max_repo_tasks=POOL, progress_tracker=progress
    )
    svc._api.graphql = owner.graphql  # type: ignore[method-assign]
    svc._api.get = owner.get  # type: ignore[method-assign]
    return svc


async def _similar(owner: _Owner, progress: Any = None) -> tuple[Any, GitHubService]:
    svc = _service(owner, progress)
    try:
        found = await svc.find_similar_prs(
            "acme", SOURCE, _AllSimilar(), only_automation=True
        )
    finally:
        await svc.close()
    return found, svc


async def _owner_wide(
    owner: _Owner, progress: Any = None, selection: RepoSelection | None = None
) -> tuple[Any, GitHubService]:
    svc = _service(owner, progress)
    try:
        found = await svc.fetch_owner_open_prs(
            "acme", only_automation=False, selection=selection
        )
    finally:
        await svc.close()
    return found, svc


async def _blocked(owner: _Owner, progress: Any = None) -> Any:
    svc = _service(owner, progress)
    svc._read_policies = False
    try:
        return await svc.scan_organization("acme")
    finally:
        await svc.close()


async def _status(owner: _Owner, mocker) -> Any:
    svc = _service(owner)
    svc._read_policies = False
    mocker.patch.object(svc, "_get_latest_tag", return_value=(None, None))
    mocker.patch.object(svc, "_get_latest_release", return_value=(None, None))
    try:
        return await svc.gather_organization_status("acme")
    finally:
        await svc.close()


def _found(found: Any) -> list[tuple[str, int, list[str]]]:
    return [(pr.repository_full_name, pr.number, c.reasons) for pr, c in found]


_LIST_SWITCHES = ("withFiles", "withReviews", "withChecks", "withComments")


class TestListSelection:
    """Each scan asks only for the nested lists it reads."""

    @pytest.mark.asyncio
    async def test_merge_scans_select_files_and_reviews(self) -> None:
        for scan in (_similar, _owner_wide):
            owner = _Owner(counted=True)
            await scan(owner)
            assert owner.selections == {frozenset({"withFiles", "withReviews"})}

    @pytest.mark.asyncio
    async def test_blocked_selects_checks_and_comments(self) -> None:
        owner = _Owner(counted=True)
        await _blocked(owner)
        assert owner.selections == {frozenset({"withChecks", "withComments"})}

    @pytest.mark.asyncio
    async def test_status_selects_files_only(self, mocker) -> None:
        owner = _Owner(counted=True)
        await _status(owner, mocker)
        assert owner.selections == {frozenset({"withFiles"})}


class TestRequests:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("root", ["organization", "user"])
    async def test_similar_scan_reads_only_repositories_with_open_prs(
        self, root: str
    ) -> None:
        owner = _Owner(root=root)
        found, _ = await _similar(owner)
        assert owner.listings == 2
        assert sorted(owner.pages) == ACTIVE_NAMES
        assert len(found) == len(ACTIVE)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("root", ["organization", "user"])
    async def test_owner_wide_scan_reads_only_repositories_with_open_prs(
        self, root: str
    ) -> None:
        owner = _Owner(root=root)
        (prs, errors), _ = await _owner_wide(owner)
        assert owner.listings == 2
        assert sorted(owner.pages) == ACTIVE_NAMES
        assert len(prs) == 2 * len(ACTIVE)
        assert errors == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("root", ["organization", "user"])
    async def test_blocked_scan_reads_only_repositories_with_open_prs(
        self, root: str
    ) -> None:
        owner = _Owner(root=root)
        result = await _blocked(owner)
        assert owner.listings == 2
        assert sorted(owner.pages) == ACTIVE_NAMES
        assert result.total_repositories == REPOSITORIES
        assert result.scanned_repositories == REPOSITORIES
        assert len(result.unmergeable_prs) == len(ACTIVE)

    @pytest.mark.asyncio
    async def test_status_reads_open_prs_only_where_there_are_some(
        self, mocker
    ) -> None:
        owner = _Owner()
        result = await _status(owner, mocker)
        assert sorted(owner.pages) == ACTIVE_NAMES
        assert len(result.repository_statuses) == REPOSITORIES

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "count",
        [None, {}, {"totalCount": None}, {"totalCount": "0"}, {"totalCount": False}],
    )
    async def test_a_node_without_a_readable_count_is_read(self, count: Any) -> None:
        owner = _Owner(count={1: count})
        await _similar(owner)
        assert _name(1) in owner.pages


class TestSimilarScanConcurrency:
    @pytest.mark.asyncio
    async def test_repositories_are_read_concurrently_within_the_pool(self) -> None:
        owner = _Owner(counted=False, delay=lambda _: 0.005)
        await _similar(owner)
        assert 1 < owner.most_in_flight <= POOL

    @pytest.mark.asyncio
    async def test_results_come_back_in_listing_order(self) -> None:
        # Earlier repositories answer last.
        owner = _Owner(delay=lambda name: (150 - int(name[-3:])) * 0.0005)
        found, svc = await _similar(owner)
        listed = [f"acme/{n}" for n in ACTIVE_NAMES]
        assert [pr.repository_full_name for pr, _ in found] == listed
        assert [pr.repository_full_name for pr in svc.seen_app_prs] == listed

    @pytest.mark.asyncio
    async def test_a_stalled_repository_bounds_the_fetches_awaiting_judgment(
        self,
    ) -> None:
        # While the first repository is held, the others are fetched only
        # as far as the window allows, not to the end of the owner.
        hold = asyncio.Event()
        owner = _Owner(counted=False, hold=hold)
        scan = asyncio.create_task(_similar(owner))
        await asyncio.sleep(0.2)
        assert not scan.done()
        assert len(owner.pages) <= 2 * POOL
        hold.set()
        found, _ = await asyncio.wait_for(scan, timeout=5)
        assert len(owner.pages) == REPOSITORIES
        assert len(found) == len(ACTIVE)

    @pytest.mark.asyncio
    async def test_an_unreadable_repository_fails_the_similar_scan(self) -> None:
        with pytest.raises(RuntimeError, match="repo-012 cannot be read"):
            await _similar(_Owner(fail=frozenset({12})))

    @pytest.mark.asyncio
    async def test_a_failure_ends_the_scan_before_the_listing_does(self) -> None:
        # The second listing page never arrives: the failure must not
        # wait for it.
        owner = _Owner(fail=frozenset({12}), stall=asyncio.Event())
        with pytest.raises(RuntimeError, match="repo-012 cannot be read"):
            await asyncio.wait_for(_similar(owner), timeout=5)

    @pytest.mark.asyncio
    async def test_a_later_refusal_does_not_skip_an_earlier_variable(self) -> None:
        # repo-024 answers last, after repo-036 is refused its variable;
        # a serial search reads repo-024's first, and enforces it.
        owner = _Owner(
            delay=lambda name: 0.05 if name == _name(24) else 0.001,
            variables={24: json.dumps({"exclude_bots": ["dependabot"]}), 36: 403},
        )
        found, svc = await _similar(owner)
        names = [pr.repository_full_name for pr, _ in found]
        assert f"acme/{_name(24)}" not in names
        assert f"acme/{_name(36)}" in names
        assert owner.variable_reads == [_name(i) for i in (0, 12, 24, 36)]
        assert svc.policy_files.variables_refused == frozenset({"repos"})

    @pytest.mark.asyncio
    async def test_the_owner_wide_scan_records_it_and_carries_on(self) -> None:
        (prs, errors), _ = await _owner_wide(_Owner(fail=frozenset({12})))
        assert errors == [
            "Error scanning repository acme/repo-012: repo-012 cannot be read"
        ]
        assert len(prs) == 2 * (len(ACTIVE) - 1)


class TestUnchanged:
    @pytest.mark.asyncio
    async def test_results_match_a_scan_that_reads_every_repository(
        self, mocker
    ) -> None:
        def strip(result: Any) -> dict[str, Any]:
            dumped: dict[str, Any] = result.model_dump(exclude={"scan_timestamp"})
            return dumped

        runs: dict[bool, dict[str, Any]] = {}
        for counted in (True, False):
            similar, similar_svc = await _similar(_Owner(counted=counted))
            owner_wide, owner_svc = await _owner_wide(_Owner(counted=counted))
            runs[counted] = {
                "similar": _found(similar),
                "seen": [pr.model_dump() for pr in similar_svc.seen_app_prs],
                "owner_wide": (
                    [pr.model_dump() for pr in owner_wide[0]],
                    owner_wide[1],
                ),
                "blocked": strip(await _blocked(_Owner(counted=counted))),
                "status": strip(await _status(_Owner(counted=counted), mocker)),
                "files": [
                    dict(similar_svc.policy_files.repositories),
                    dict(owner_svc.policy_files.repositories),
                ],
            }
        skipped, read_all = runs[True], runs[False]
        assert skipped["files"][0]["acme/repo-012"] == POLICY
        # A repository with no open PR has no PR for its file to judge;
        # reading every repository records only that it has no file.
        for files in read_all["files"]:
            assert {key for key, blob in files.items() if blob is not None} == {
                "acme/repo-012"
            }
        active = {f"acme/{n}" for n in ACTIVE_NAMES}
        read_all["files"] = [
            {key: blob for key, blob in files.items() if key in active}
            for files in read_all["files"]
        ]
        # The owner-wide scan collects in completion order, as before.
        for run in (skipped, read_all):
            run["owner_wide"][0].sort(key=lambda pr: pr["html_url"])
        assert skipped == read_all


class TestSelectionAndProgress:
    @pytest.mark.asyncio
    async def test_an_included_repository_with_no_open_pr_is_matched(self) -> None:
        selection = RepoSelection(
            owner="acme",
            exclude=False,
            names={_name(1): _name(1), _name(12): _name(12)},
        )
        (prs, _), _ = await _owner_wide(_Owner(), selection=selection)
        assert selection.unmatched() == []
        assert {pr.repository_full_name for pr in prs} == {f"acme/{_name(12)}"}

    @pytest.mark.asyncio
    async def test_every_repository_is_counted_as_processed(self) -> None:
        trackers = [_Progress() for _ in range(3)]
        await _similar(_Owner(), trackers[0])
        await _owner_wide(_Owner(), trackers[1])
        await _blocked(_Owner(), trackers[2])
        for tracker in trackers:
            assert tracker.total == REPOSITORIES
            assert tracker.completed == REPOSITORIES
            assert tracker.errors == 0


def _count_error(root: str, index: int) -> dict[str, Any]:
    return {
        "type": "FORBIDDEN",
        "path": [root, "repositories", "nodes", index, "pullRequests"],
        "message": "Resource not accessible by integration",
    }


class _Transport:
    """Answers GraphQL over HTTP for three repositories.

    While the listing asks for open PR counts, or on every listing with
    ``always``, it answers with ``errors`` and the data GitHub would send
    with them.
    """

    def __init__(
        self, root: str, errors: list[dict[str, Any]], *, always: bool = False
    ) -> None:
        self.root = root
        self.errors = errors
        self.always = always
        self.listings: list[bool] = []
        self.pages: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST":
            return httpx.Response(404, json={"message": "Not Found"})
        sent = json.loads(request.content)
        query, variables = sent["query"], sent["variables"]
        if query == OWNER_PROFILE:
            typename = "Organization" if self.root == "organization" else "User"
            data = {"repositoryOwner": {"__typename": typename, "dotgithub": None}}
            return httpx.Response(200, json={"data": data})
        if query == REPO_OPEN_PRS_PAGE:
            self.pages.append(variables["name"])
            prs = {"pageInfo": {"hasNextPage": False}, "nodes": []}
            return httpx.Response(
                200,
                json=with_selected_lists(
                    {"data": {"repository": {"pullRequests": prs}}}, variables
                ),
            )
        counted = "pullRequests(states: OPEN)" in query and variables.get(
            "withCounts", True
        )
        self.listings.append(bool(counted))
        nodes: list[Any] = [
            {"nameWithOwner": f"acme/{_name(i)}", "isArchived": False} for i in range(3)
        ]
        body: dict[str, Any] = {}
        if counted:
            for node in nodes:
                node["pullRequests"] = {"totalCount": 1}
        if self.errors and (counted or self.always):
            nodes[1] = None
            body["errors"] = self.errors
        repos = {"totalCount": 3, "pageInfo": {"hasNextPage": False}, "nodes": nodes}
        body["data"] = {self.root: {"repositories": repos}}
        return httpx.Response(200, json=body)


async def _owner_wide_over_http(transport: _Transport) -> Any:
    svc = GitHubService(token="t", read_policies=False, max_repo_tasks=POOL)
    svc._api._client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    try:
        return await svc.fetch_owner_open_prs("acme", only_automation=False)
    finally:
        await svc.close()


class TestUncountedListing:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("root", ["organization", "user"])
    async def test_a_page_github_cannot_count_is_read_without_counts(
        self, root: str
    ) -> None:
        transport = _Transport(root, [_count_error(root, 1)])
        prs, errors = await _owner_wide_over_http(transport)
        assert sorted(transport.pages) == [_name(i) for i in range(3)]
        assert (prs, errors) == ([], [])
        assert transport.listings[-1] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "errors",
        [
            [{"type": "FORBIDDEN", "path": ["organization", "repositories"]}],
            [_count_error("organization", 1), {"type": "FORBIDDEN"}],
            [{**_count_error("organization", 1), "type": "RATE_LIMITED"}],
        ],
    )
    async def test_any_other_listing_error_still_fails(
        self, errors: list[dict[str, Any]]
    ) -> None:
        transport = _Transport("organization", errors, always=True)
        with pytest.raises(GraphQLError):
            await _owner_wide_over_http(transport)
        assert len(transport.listings) == 1
        assert transport.pages == []
