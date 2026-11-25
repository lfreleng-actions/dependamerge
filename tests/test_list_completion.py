# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""A short files or comments list is completed before anything uses it.

The open pull request page holds the first ``SCAN_FILES_PAGE_SIZE``
files and ``SCAN_COMMENTS_PAGE_SIZE`` comments of each pull request.
When a list says it is longer, the pull request's whole list is read
over REST first: for a similarity candidate, for ``status``'s action
and workflow tests, and for ``blocked``'s Copilot comment count.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.bot_policy import BotPolicy
from dependamerge.github_async._reviews import IssueComments
from dependamerge.github_graphql.pull_requests import (
    SCAN_COMMENTS_PAGE_SIZE,
    SCAN_FILES_PAGE_SIZE,
)
from dependamerge.github_service import GitHubService
from dependamerge.github_service._list_completion import complete_pr_files
from dependamerge.github_service._similar import _Fetched
from dependamerge.models import ComparisonResult, PullRequestInfo

_REPO = "acme/widget"


def _paths(count: int, last: str | None = None) -> list[str]:
    paths = [f"src/f{i}.py" for i in range(count)]
    if last is not None:
        paths[-1] = last
    return paths


def _files(paths: list[str]) -> dict[str, Any]:
    nodes = [
        {"path": path, "additions": 1, "deletions": 0}
        for path in paths[:SCAN_FILES_PAGE_SIZE]
    ]
    return {"totalCount": len(paths), "nodes": nodes}


def _node(number: int, author: str, paths: list[str]) -> dict[str, Any]:
    return {
        "number": number,
        "title": "Bump foo from 1.0 to 1.1",
        "url": f"https://github.com/{_REPO}/pull/{number}",
        "author": {"__typename": "Bot", "login": author},
        "baseRefName": "main",
        "headRefName": f"dependabot/foo-{number}",
        "headRefOid": f"sha{number}",
        "files": _files(paths),
        "reviews": {"totalCount": 0, "nodes": []},
    }


class _Files:
    """Serve ``/pulls/{n}/files`` from ``paths``, a page at a time.

    ``fail_after`` makes the page after that many pages raise.
    ``nameless`` lists paths served as an entry without a filename.
    ``requests`` records each pull request number read.
    """

    def __init__(
        self,
        paths: dict[int, list[str]],
        fail_after: int | None = None,
        nameless: frozenset[str] = frozenset(),
    ) -> None:
        self.paths = paths
        self.fail_after = fail_after
        self.nameless = nameless
        self.requests: list[int] = []

    async def __call__(
        self, path: str, per_page: int = 100, **_: Any
    ) -> AsyncIterator[list[dict[str, Any]]]:
        number = int(path.rstrip("/").split("/")[-2])
        self.requests.append(number)
        files = self.paths[number]
        for page, start in enumerate(range(0, len(files), per_page)):
            if self.fail_after is not None and page >= self.fail_after:
                raise RuntimeError("502 Bad Gateway")
            yield [
                {"additions": 1, "deletions": 0}
                | ({} if name in self.nameless else {"filename": name})
                for name in files[start : start + per_page]
            ]


class _Recorder:
    """A comparator that records the files each candidate is scored on."""

    def __init__(self) -> None:
        self.scored: dict[int, list[str]] = {}

    def compare_pull_requests(
        self, source: PullRequestInfo, target: PullRequestInfo, only_automation: bool
    ) -> ComparisonResult:
        self.scored[target.number] = [f.filename for f in target.files_changed]
        return ComparisonResult(is_similar=True, confidence_score=1.0, reasons=[])


def _source() -> PullRequestInfo:
    return PullRequestInfo(
        number=1,
        title="Bump foo from 1.0 to 1.1",
        body=None,
        author="dependabot[bot]",
        head_sha="sha1",
        base_branch="main",
        head_branch="dependabot/foo",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name="acme/home",
        html_url="https://github.com/acme/home/pull/1",
    )


def _service(files: _Files) -> GitHubService:
    svc = GitHubService(token="t")
    svc._api.get_paginated = files  # type: ignore[assignment,method-assign]
    svc._read_repository_variable = AsyncMock()  # type: ignore[method-assign]
    return svc


class TestSimilarScan:
    @pytest.mark.asyncio
    async def test_a_long_candidate_is_scored_on_every_file(self) -> None:
        paths = {7: _paths(66), 8: _paths(66)}
        files = _Files(paths)
        svc = _service(files)
        prs = [
            svc.to_pull_request_info(_REPO, _node(7, "dependabot", paths[7])),
            # A person's PR: the candidate filter rejects it.
            svc.to_pull_request_info(_REPO, _node(8, "octocat", paths[8])),
        ]
        assert [pr.files_complete for pr in prs] == [False, False]
        comparator = _Recorder()
        found = await svc._find_similar_prs_in_repo(
            _Fetched(_REPO, prs, prs[:1]),
            _source(),
            comparator,
            only_automation=True,
            bot_policy=BotPolicy(),
        )
        assert comparator.scored == {7: paths[7]}
        assert files.requests == [7]
        ((pr, _),) = found
        assert pr.files_complete is True

    @pytest.mark.asyncio
    async def test_a_short_source_is_completed_before_the_search(self) -> None:
        paths = _paths(66)
        files = _Files({1: paths})
        svc = _service(files)
        source = _source().model_copy(
            update={
                "files_changed": svc.to_pull_request_info(
                    _REPO, _node(1, "dependabot", paths)
                ).files_changed,
                "files_complete": False,
            }
        )
        seen: list[list[str]] = []

        async def judge(_repo: Any, source_pr: PullRequestInfo, *_: Any, **__: Any):
            seen.append([f.filename for f in source_pr.files_changed])
            return []

        async def listing(_org: str) -> AsyncIterator[dict[str, Any]]:
            yield {"nameWithOwner": _REPO}

        svc._iter_org_repositories_with_open_prs = listing  # type: ignore[method-assign,assignment]
        svc._fetch_listed_repo = AsyncMock(  # type: ignore[method-assign]
            return_value=_Fetched(_REPO, [], [])
        )
        svc._find_similar_prs_in_repo = judge  # type: ignore[method-assign,assignment]
        await svc.find_similar_prs("acme", source, _Recorder(), only_automation=True)
        assert seen == [paths]
        assert files.requests == [1]

    @pytest.mark.asyncio
    async def test_a_whole_page_costs_no_request(self) -> None:
        files = _Files({})
        svc = _service(files)
        pr = svc.to_pull_request_info(_REPO, _node(7, "dependabot", _paths(3)))
        assert await complete_pr_files(svc._api, pr) is pr
        assert files.requests == []


class TestFilesReadFailure:
    @pytest.mark.asyncio
    async def test_a_failed_read_warns_and_keeps_what_was_read(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        paths = _paths(150)
        files = _Files({7: paths}, fail_after=1)
        svc = _service(files)
        pr = svc.to_pull_request_info(_REPO, _node(7, "dependabot", paths))
        with caplog.at_level(logging.WARNING):
            completed = await complete_pr_files(svc._api, pr)
        assert [f.filename for f in completed.files_changed] == paths[:100]
        assert completed.files_complete is False
        warnings = [r.getMessage() for r in caplog.records]
        assert any(f"{_REPO}#7" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_a_capped_read_short_of_the_total_is_incomplete(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The files endpoint stops at 3,000 files and ends as if whole.
        paths = _paths(3001)
        files = _Files({7: paths[:3000]})
        svc = _service(files)
        pr = svc.to_pull_request_info(_REPO, _node(7, "dependabot", paths))
        with caplog.at_level(logging.WARNING):
            completed = await complete_pr_files(svc._api, pr)
        assert [f.filename for f in completed.files_changed] == paths[:3000]
        assert completed.files_complete is False
        warnings = [r.getMessage() for r in caplog.records]
        assert any("3000 of 3001" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_a_whole_read_below_a_stale_total_is_complete(self) -> None:
        # A force-push between the scan and the read removed the workflow,
        # so the page's total of 61 is one more than the 60 files read.
        paths = _paths(60)
        files = _Files({7: paths})
        svc = _service(files)
        node = _node(7, "dependabot", [*paths, ".github/workflows/ci.yaml"])
        pr = svc.to_pull_request_info(_REPO, node)
        assert pr.files_total == 61
        completed = await complete_pr_files(svc._api, pr)
        assert [f.filename for f in completed.files_changed] == paths
        assert completed.files_complete is True

    @pytest.mark.asyncio
    async def test_a_repeated_filename_is_incomplete(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A force-push between pages shifted the second page back by one:
        # f99 is read twice and f100 never.
        paths = _paths(150)
        served = [*paths[:100], *paths[99:100], *paths[101:]]
        files = _Files({7: served})
        svc = _service(files)
        pr = svc.to_pull_request_info(_REPO, _node(7, "dependabot", paths))
        with caplog.at_level(logging.WARNING):
            completed = await complete_pr_files(svc._api, pr)
        names = [f.filename for f in completed.files_changed]
        assert names == [*paths[:100], *paths[101:]]
        assert completed.files_complete is False
        warnings = [r.getMessage() for r in caplog.records]
        assert any(f"{_REPO}#7 changed" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_a_read_matching_the_total_is_complete(self) -> None:
        paths = _paths(66)
        svc = _service(_Files({7: paths}))
        pr = svc.to_pull_request_info(_REPO, _node(7, "dependabot", paths))
        assert pr.files_total == 66
        completed = await complete_pr_files(svc._api, pr)
        assert completed.files_complete is True


class TestStatus:
    @pytest.mark.asyncio
    async def test_an_entry_without_a_filename_keeps_the_page(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The read serves the workflow the page holds without its filename:
        # a malformed read, so the page's paths still count.
        workflow = ".github/workflows/ci.yaml"
        paths = [workflow, *_paths(60)]
        files = _Files({7: paths}, nameless=frozenset({workflow}))
        svc = _service(files)
        node = _node(7, "dependabot", paths)
        del node["reviews"]
        svc._fetch_repo_prs_first_page = AsyncMock(  # type: ignore[method-assign]
            return_value=([node], {"hasNextPage": False})
        )
        with caplog.at_level(logging.WARNING):
            stats = await svc._gather_pr_statistics("acme", "widget", None)
        assert stats["workflow_prs_automation"] == 1
        warnings = [r.getMessage() for r in caplog.records]
        assert any(f"{_REPO}#7 is incomplete" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_a_workflow_file_past_the_page_counts(self) -> None:
        paths = _paths(61, last=".github/workflows/ci.yaml")
        files = _Files({7: paths})
        svc = _service(files)
        node = _node(7, "dependabot", paths)
        del node["reviews"]
        svc._fetch_repo_prs_first_page = AsyncMock(  # type: ignore[method-assign]
            return_value=([node], {"hasNextPage": False})
        )
        stats = await svc._gather_pr_statistics("acme", "widget", None)
        assert stats["workflow_prs_automation"] == 1
        assert files.requests == [7]

    @pytest.mark.asyncio
    async def test_a_whole_read_replaces_a_stale_page(self) -> None:
        # A force-push between the scan and the read removed the workflow.
        paths = _paths(61)
        files = _Files({7: paths})
        svc = _service(files)
        node = _node(7, "dependabot", [".github/workflows/ci.yaml", *paths[1:]])
        del node["reviews"]
        svc._fetch_repo_prs_first_page = AsyncMock(  # type: ignore[method-assign]
            return_value=([node], {"hasNextPage": False})
        )
        stats = await svc._gather_pr_statistics("acme", "widget", None)
        assert stats["workflow_prs_automation"] == 0
        assert files.requests == [7]

    @pytest.mark.asyncio
    async def test_a_shrunk_total_does_not_restore_a_removed_file(self) -> None:
        # The page listed 61 files, the workflow among its first 50; a
        # force-push then removed it, and the whole read holds 60.
        paths = _paths(60)
        files = _Files({7: paths})
        svc = _service(files)
        node = _node(7, "dependabot", [".github/workflows/ci.yaml", *paths])
        del node["reviews"]
        svc._fetch_repo_prs_first_page = AsyncMock(  # type: ignore[method-assign]
            return_value=([node], {"hasNextPage": False})
        )
        stats = await svc._gather_pr_statistics("acme", "widget", None)
        assert stats["workflow_prs_automation"] == 0
        assert files.requests == [7]


def _comment(index: int, login: str = "octocat") -> dict[str, Any]:
    return {
        "id": index,
        "user": {"login": login},
        "body": f"comment {index}",
        "created_at": "2026-01-01T00:00:00Z",
    }


def _blocked_node(comments: list[dict[str, Any]]) -> dict[str, Any]:
    page = [
        {
            "author": {"login": c["user"]["login"]},
            "body": c["body"],
            "createdAt": c["created_at"],
        }
        for c in comments[:SCAN_COMMENTS_PAGE_SIZE]
    ]
    return {
        "number": 7,
        "title": "Bump foo",
        "url": f"https://github.com/{_REPO}/pull/7",
        "author": {"__typename": "Bot", "login": "dependabot"},
        "mergeable": "CONFLICTING",
        "mergeStateStatus": "DIRTY",
        "commits": {"nodes": []},
        "comments": {"totalCount": len(comments), "nodes": page},
    }


class TestBlockedComments:
    @pytest.mark.asyncio
    async def test_a_copilot_comment_past_the_page_is_counted(self) -> None:
        comments = [_comment(i) for i in range(12)]
        comments[11] = _comment(11, "Copilot")
        svc = GitHubService(token="t")
        svc._api.list_issue_comments = AsyncMock(  # type: ignore[method-assign]
            return_value=IssueComments(comments, True)
        )
        result = await svc._analyze_pr_node(_REPO, _blocked_node(comments))
        assert result is not None
        assert result.copilot_comments_count == 1
        assert result.copilot_comments[0].id == 11
        svc._api.list_issue_comments.assert_awaited_once_with("acme", "widget", 7)

    @pytest.mark.asyncio
    async def test_a_suffixed_rest_copilot_login_is_counted(self) -> None:
        comments = [_comment(i, "Copilot") for i in range(12)]
        rest = [_comment(i, "copilot[bot]") for i in range(12)]
        svc = GitHubService(token="t")
        svc._api.list_issue_comments = AsyncMock(  # type: ignore[method-assign]
            return_value=IssueComments(rest, True)
        )
        result = await svc._analyze_pr_node(_REPO, _blocked_node(comments))
        assert result is not None
        assert result.copilot_comments_count == 12

    @pytest.mark.asyncio
    async def test_a_short_failed_read_keeps_the_page(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        comments = [_comment(i, "Copilot") for i in range(12)]
        svc = GitHubService(token="t")
        svc._api.list_issue_comments = AsyncMock(  # type: ignore[method-assign]
            return_value=IssueComments(comments[:3], False)
        )
        with caplog.at_level(logging.WARNING):
            result = await svc._analyze_pr_node(_REPO, _blocked_node(comments))
        assert result is not None
        assert result.copilot_comments_count == SCAN_COMMENTS_PAGE_SIZE
        assert any(f"{_REPO}#7" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_longer_failed_read_keeps_the_newer_page(self) -> None:
        # REST reads oldest first; the page holds the newest comments.
        older = [_comment(i) for i in range(SCAN_COMMENTS_PAGE_SIZE + 2)]
        older[0] = _comment(0, "Copilot")
        newest = _comment(99, "Copilot")
        page = [
            {
                "databaseId": c["id"],
                "author": {"login": c["user"]["login"]},
                "body": c["body"],
                "createdAt": c["created_at"],
            }
            for c in (newest, older[0])
        ]
        node = _blocked_node([])
        node["comments"] = {"totalCount": 20, "nodes": page}
        svc = GitHubService(token="t")
        svc._api.list_issue_comments = AsyncMock(  # type: ignore[method-assign]
            return_value=IssueComments(older, False)
        )
        result = await svc._analyze_pr_node(_REPO, node)
        assert result is not None
        assert sorted(c.id for c in result.copilot_comments) == [0, 99]

    @pytest.mark.asyncio
    async def test_a_repeated_rest_id_is_counted_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        comments = [_comment(i, "Copilot") for i in range(12)]
        svc = GitHubService(token="t")
        svc._api.list_issue_comments = AsyncMock(  # type: ignore[method-assign]
            return_value=IssueComments([*comments, comments[11]], True)
        )
        with caplog.at_level(logging.WARNING):
            result = await svc._analyze_pr_node(_REPO, _blocked_node(comments))
        assert result is not None
        assert sorted(c.id for c in result.copilot_comments) == list(range(12))
        assert any(f"{_REPO}#7" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "malformed",
        [{}, {"id": 1, "user": "Copilot"}],
        ids=["empty", "non-object-user"],
    )
    async def test_a_malformed_rest_comment_keeps_the_page(
        self, caplog: pytest.LogCaptureFixture, malformed: dict[str, Any]
    ) -> None:
        newest = _comment(99, "Copilot")
        page = [
            {
                "databaseId": newest["id"],
                "author": {"login": "Copilot"},
                "body": newest["body"],
                "createdAt": newest["created_at"],
            }
        ]
        node = _blocked_node([])
        node["comments"] = {"totalCount": 20, "nodes": page}
        svc = GitHubService(token="t")
        svc._api.list_issue_comments = AsyncMock(  # type: ignore[method-assign]
            return_value=IssueComments([_comment(0), malformed], True)
        )
        with caplog.at_level(logging.WARNING):
            result = await svc._analyze_pr_node(_REPO, node)
        assert result is not None
        assert [c.id for c in result.copilot_comments] == [99]
        assert any(f"{_REPO}#7" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_whole_page_costs_no_request(self) -> None:
        comments = [_comment(0, "Copilot")]
        svc = GitHubService(token="t")
        svc._api.list_issue_comments = AsyncMock()  # type: ignore[method-assign]
        result = await svc._analyze_pr_node(_REPO, _blocked_node(comments))
        assert result is not None
        assert result.copilot_comments_count == 1
        # This page comment carries no databaseId.
        assert result.copilot_comments[0].id == 0
        svc._api.list_issue_comments.assert_not_awaited()
