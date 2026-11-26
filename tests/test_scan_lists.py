# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""The open pull request page selects, sizes and reports its nested lists.

Each scan selects only the nested lists it reads.  An extractor asked
for a list the scan did not select raises rather than read an empty
list, and each list says whether the page holds all of it.  Nested
sizes stay fixed under rate limiting, so a result never depends on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import dependamerge
from dependamerge.github_graphql.pull_requests import (
    REPO_OPEN_PRS_PAGE,
    SCAN_CHECKS_PAGE_SIZE,
    SCAN_COMMENTS_PAGE_SIZE,
    SCAN_FILES_PAGE_SIZE,
    SCAN_REVIEWS_PAGE_SIZE,
)
from dependamerge.github_service import GitHubService
from dependamerge.github_service._pr_extract import (
    ListNotSelectedError,
    scan_comments,
    scan_files,
)
from dependamerge.github_service._scan_lists import (
    BLOCKED_SCAN_LISTS,
    MERGE_SCAN_LISTS,
    STATUS_SCAN_LISTS,
)


def _files(count: int, total: Any) -> dict[str, Any]:
    nodes = [
        {"path": f"f{i}", "additions": 1, "deletions": 0, "changeType": "MODIFIED"}
        for i in range(count)
    ]
    return {"totalCount": total, "nodes": nodes}


def _node(**lists: Any) -> dict[str, Any]:
    return {
        "number": 1,
        "title": "Bump foo",
        "author": {"login": "dependabot"},
        "headRefOid": "abc",
        **lists,
    }


@pytest.fixture
def svc() -> GitHubService:
    return GitHubService(token="t")


class TestQuery:
    def test_nested_lists_are_switched_and_counted(self) -> None:
        for switch in ("withFiles", "withReviews", "withComments", "withChecks"):
            assert f"${switch}: Boolean!" in REPO_OPEN_PRS_PAGE
            assert f"@include(if: ${switch})" in REPO_OPEN_PRS_PAGE
        for count in ("checkRunCount", "statusContextCount"):
            assert count in REPO_OPEN_PRS_PAGE
            assert f"{count}sByState" in REPO_OPEN_PRS_PAGE

    def test_nested_sizes_are_written_into_the_query(self) -> None:
        for key, size in (
            ("files", SCAN_FILES_PAGE_SIZE),
            ("comments", SCAN_COMMENTS_PAGE_SIZE),
            ("reviews", SCAN_REVIEWS_PAGE_SIZE),
            ("contexts", SCAN_CHECKS_PAGE_SIZE),
        ):
            assert f"{key}(first: {size}" in REPO_OPEN_PRS_PAGE
        assert SCAN_CHECKS_PAGE_SIZE == 100

    def test_the_unused_owner_wide_query_is_gone(self) -> None:
        source = Path(dependamerge.__file__).parent
        users = [
            str(path)
            for path in source.rglob("*.py")
            if "ORG_REPOS_WITH_OPEN_PRS" in path.read_text(encoding="utf-8")
        ]
        assert users == []


class TestPageVariables:
    @pytest.mark.parametrize(
        "lists", [MERGE_SCAN_LISTS, STATUS_SCAN_LISTS, BLOCKED_SCAN_LISTS]
    )
    def test_rate_limiting_halves_only_the_pull_request_page(
        self, svc: GitHubService, lists: Any
    ) -> None:
        normal = svc._open_prs_page_variables(
            "acme", "widget", None, with_policy=False, lists=lists
        )
        svc._rate_limited = True
        limited = svc._open_prs_page_variables(
            "acme", "widget", None, with_policy=False, lists=lists
        )
        assert limited["prsPageSize"] == max(10, normal["prsPageSize"] // 2)
        assert limited["prsPageSize"] < normal["prsPageSize"]
        assert {k: v for k, v in limited.items() if k != "prsPageSize"} == {
            k: v for k, v in normal.items() if k != "prsPageSize"
        }
        assert {k: v for k, v in normal.items() if k.startswith("with")} == {
            "withPolicy": False,
            **lists.variables(),
        }


class TestUnselectedLists:
    def test_each_extractor_raises_for_an_unselected_list(
        self, svc: GitHubService
    ) -> None:
        pr = _node()
        for extract in (
            svc._extract_file_changes,
            svc._extract_reviews,
            svc._extract_copilot_comments,
            GitHubService._extract_failing_checks,
        ):
            with pytest.raises(ListNotSelectedError):
                extract(pr)

    def test_a_selected_empty_list_reads_as_empty(self, svc: GitHubService) -> None:
        pr = _node(files=_files(0, 0), commits={"nodes": []})
        assert svc._extract_file_changes(pr) == []
        assert GitHubService._extract_failing_checks(pr) == []


class TestCompleteness:
    @pytest.mark.parametrize(
        ("count", "total", "complete"),
        [
            (3, 3, True),
            # A short page whose total says more is not the whole list.
            (3, 4, False),
            # A page short of the size is the whole list.
            (3, None, True),
            (SCAN_FILES_PAGE_SIZE, SCAN_FILES_PAGE_SIZE, True),
            (SCAN_FILES_PAGE_SIZE, 66, False),
            # A full page with no total may hold only part of the list.
            (SCAN_FILES_PAGE_SIZE, None, False),
        ],
    )
    def test_files(self, count: int, total: Any, complete: bool) -> None:
        assert scan_files(_node(files=_files(count, total)))[1] is complete

    def test_a_null_connection_is_not_known_whole(self) -> None:
        assert scan_files(_node(files=None)) == ([], False)

    def test_comments(self) -> None:
        nodes = [{"author": {"login": "x"}, "body": ""}] * SCAN_COMMENTS_PAGE_SIZE
        whole = {"totalCount": SCAN_COMMENTS_PAGE_SIZE, "nodes": nodes}
        part = {"totalCount": SCAN_COMMENTS_PAGE_SIZE + 1, "nodes": nodes}
        assert scan_comments(_node(comments=whole))[1] is True
        assert scan_comments(_node(comments=part))[1] is False

    def test_pull_request_info_carries_files_complete(self, svc: GitHubService) -> None:
        base = {
            "reviews": {"totalCount": 0, "nodes": []},
            "baseRefName": "main",
            "headRefName": "dependabot/foo",
            "url": "https://github.com/acme/widget/pull/1",
        }
        whole = svc.to_pull_request_info(
            "acme/widget", _node(files=_files(2, 2), **base)
        )
        part = svc.to_pull_request_info(
            "acme/widget", _node(files=_files(SCAN_FILES_PAGE_SIZE, 66), **base)
        )
        assert whole.files_complete is True
        assert part.files_complete is False
        assert len(part.files_changed) == SCAN_FILES_PAGE_SIZE
