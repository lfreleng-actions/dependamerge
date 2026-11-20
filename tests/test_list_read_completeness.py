# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""List reads that used to stop short now finish, or say they did not.

Each read here once stopped before the end of its list without a word
(or with one only at debug level).  None drives a merge decision alone,
so a short read must never fail the run; it must either read on or warn.

The client's ``_fetch_json`` is the only thing mocked, so the real
``get_paginated`` runs: page numbers, the ``Link`` header and the page
ceiling behave as they do against GitHub.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from urllib.parse import parse_qsl

import pytest

from dependamerge.github_async import GitHubAsync
from dependamerge.github_client.queries import _fetch_pr_files
from dependamerge.github_service import GitHubService
from tests.conftest import make_merge_manager

# A route answers ``(path, page)`` with ``(body, more_pages)`` or raises.
Route = Callable[[str, int], tuple[Any, bool]]


def _wire(gh: GitHubAsync, route: Route) -> list[tuple[str, int]]:
    """Serve every request *gh* makes from *route*; return what was asked.

    A query string in the path is folded into the parameters, so a
    request for ``...?per_page=100`` and one passing ``params`` are seen
    alike.
    """
    seen: list[tuple[str, int]] = []

    async def fetch_json(
        _method: str, url: str, *, params: dict[str, Any] | None = None, **_kw: Any
    ) -> tuple[Any, Any]:
        path, _, query = url.removeprefix(gh.api_url).partition("?")
        merged = dict(parse_qsl(query))
        merged.update(params or {})
        page = int(merged.get("page", 1))
        seen.append((path, page))
        body, more = route(path, page)
        headers = {"Link": '<https://next>; rel="next"'} if more else {}
        return SimpleNamespace(headers=headers), body

    gh._fetch_json = AsyncMock(side_effect=fetch_json)  # type: ignore[method-assign]
    return seen


def _closed_prs(
    count: int, *, updated: str, merged: str | None
) -> list[dict[str, Any]]:
    return [
        {"number": n, "updated_at": updated, "merged_at": merged, "user": {}}
        for n in range(count)
    ]


class TestMergedPullRequestsSince:
    """Closed pull requests come newest-updated first."""

    @pytest.mark.asyncio
    async def test_stops_at_a_page_reaching_past_the_since_date(self) -> None:
        page1 = _closed_prs(99, updated="2026-03-01T00:00:00Z", merged="2026-02-01")
        page1 += _closed_prs(1, updated="2025-12-01T00:00:00Z", merged=None)

        def route(path: str, _page: int) -> tuple[Any, bool]:
            assert path == "/repos/acme/widget/pulls"
            return page1, True

        async with GitHubAsync(token="t") as gh:
            seen = _wire(gh, route)
            svc = GitHubService(token="t", client=gh)
            merged = await svc._get_merged_prs_since("acme", "widget", "2026/01/01")

        assert len(merged) == 99
        assert seen == [("/repos/acme/widget/pulls", 1)]

    @pytest.mark.asyncio
    async def test_reads_on_while_a_page_is_newer_than_the_since_date(self) -> None:
        pages = [
            _closed_prs(100, updated="2026-03-01T00:00:00Z", merged="2026-02-01"),
            _closed_prs(50, updated="2025-12-01T00:00:00Z", merged="2025-11-01"),
        ]

        def route(_path: str, page: int) -> tuple[Any, bool]:
            return pages[page - 1], page < len(pages)

        async with GitHubAsync(token="t") as gh:
            seen = _wire(gh, route)
            svc = GitHubService(token="t", client=gh)
            merged = await svc._get_merged_prs_since("acme", "widget", "2026/01/01")

        assert len(merged) == 100
        assert [page for _, page in seen] == [1, 2]

    @pytest.mark.asyncio
    async def test_warns_when_the_ceiling_cuts_the_read_short(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        full = _closed_prs(100, updated="2026-03-01T00:00:00Z", merged="2026-02-01")

        async with GitHubAsync(token="t") as gh:
            seen = _wire(gh, lambda _path, _page: (full, True))
            svc = GitHubService(token="t", client=gh)
            with caplog.at_level(logging.WARNING):
                merged = await svc._get_merged_prs_since("acme", "widget", "2026/01/01")

        assert len(merged) == 1000
        assert len(seen) == 10
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("acme/widget" in r.getMessage() for r in warnings)


class TestPullRequestFiles:
    @pytest.mark.asyncio
    async def test_a_failed_second_page_warns_and_keeps_the_first(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = [{"filename": f"f{n}.py", "additions": 1} for n in range(100)]

        def route(_path: str, page: int) -> tuple[Any, bool]:
            if page == 1:
                return first, True
            raise RuntimeError("502 Bad Gateway")

        async with GitHubAsync(token="t") as gh:
            _wire(gh, route)
            with caplog.at_level(logging.DEBUG):
                files = await _fetch_pr_files(gh, "acme", "widget", 7)

        assert [f.filename for f in files] == [f["filename"] for f in first]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("acme/widget#7" in r.getMessage() for r in warnings)


class TestLatestRelease:
    @pytest.mark.asyncio
    async def test_a_full_release_after_a_page_of_prereleases(self) -> None:
        prereleases = [
            {"tag_name": f"v2.0.0-rc{n}", "prerelease": True, "draft": False}
            for n in range(30)
        ]
        release = {
            "tag_name": "v1.0.0",
            "prerelease": False,
            "draft": False,
            "published_at": "2026-02-03T10:00:00Z",
        }

        def route(path: str, page: int) -> tuple[Any, bool]:
            assert path == "/repos/acme/widget/releases"
            return (prereleases, True) if page == 1 else ([release], False)

        async with GitHubAsync(token="t") as gh:
            _wire(gh, route)
            svc = GitHubService(token="t", client=gh)
            latest = await svc._get_latest_release("acme", "widget")

        assert latest == ("v1.0.0", "2026/02/03")

    @pytest.mark.asyncio
    async def test_warns_when_the_ceiling_cuts_the_search_short(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        prereleases = [
            {"tag_name": f"v2.0.0-rc{n}", "prerelease": True, "draft": False}
            for n in range(100)
        ]

        async with GitHubAsync(token="t") as gh:
            seen = _wire(gh, lambda _path, _page: (prereleases, True))
            svc = GitHubService(token="t", client=gh)
            with caplog.at_level(logging.WARNING):
                latest = await svc._get_latest_release("acme", "widget")

        assert latest == (None, None)
        assert len(seen) == 3
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("acme/widget" in r.getMessage() for r in warnings)

    @pytest.mark.asyncio
    async def test_no_warning_when_the_list_ends_without_a_release(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        drafts = [{"tag_name": "v3.0.0", "prerelease": False, "draft": True}]

        async with GitHubAsync(token="t") as gh:
            _wire(gh, lambda _path, _page: (drafts, False))
            svc = GitHubService(token="t", client=gh)
            with caplog.at_level(logging.WARNING):
                latest = await svc._get_latest_release("acme", "widget")

        assert latest == (None, None)
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def _ruleset_route(*, second_page_fails: bool) -> Route:
    detail = {
        "conditions": {},
        "rules": [
            {
                "type": "required_status_checks",
                "parameters": {"required_status_checks": [{"context": "DCO"}]},
            }
        ],
    }

    def route(path: str, page: int) -> tuple[Any, bool]:
        if path == "/repos/acme/widget/rulesets":
            if page == 1:
                return [{"id": 1}], True
            if second_page_fails:
                raise RuntimeError("502 Bad Gateway")
            return [{"id": 2}], False
        if path == "/repos/acme/widget/rulesets/1":
            return {"conditions": {}, "rules": []}, False
        if path == "/repos/acme/widget/rulesets/2":
            return detail, False
        raise AssertionError(f"unexpected request: {path}")

    return route


class TestRulesetRequiredChecks:
    @pytest.mark.asyncio
    async def test_a_check_from_a_second_page_ruleset_is_found(self) -> None:
        async with GitHubAsync(token="t") as gh:
            _wire(gh, _ruleset_route(second_page_fails=False))
            checks, reliable = await gh._fetch_ruleset_required_checks(
                "acme", "widget", "main", "main"
            )

        assert [c["context"] for c in checks] == ["DCO"]
        assert reliable is True

    @pytest.mark.asyncio
    async def test_a_failed_second_page_is_unreliable(self) -> None:
        async with GitHubAsync(token="t") as gh:
            _wire(gh, _ruleset_route(second_page_fails=True))
            _, reliable = await gh._fetch_ruleset_required_checks(
                "acme", "widget", "main", "main"
            )

        assert reliable is False


def _branch_rules_route(*, second_page: list[dict[str, Any]] | Exception) -> Route:
    first = [{"type": "deletion"} for _ in range(30)]

    def route(path: str, page: int) -> tuple[Any, bool]:
        assert path == "/repos/acme/widget/rules/branches/main"
        if page == 1:
            return first, True
        if isinstance(second_page, Exception):
            raise second_page
        return second_page, False

    return route


class TestEffectiveBranchRules:
    @pytest.mark.xfail(strict=True, reason="only the first page is read")
    @pytest.mark.asyncio
    async def test_a_pull_request_rule_on_the_second_page_counts(self) -> None:
        rule = {
            "type": "pull_request",
            "parameters": {"required_approving_review_count": 1},
        }
        mgr, _ = make_merge_manager()
        async with GitHubAsync(token="t") as gh:
            _wire(gh, _branch_rules_route(second_page=[rule]))
            mgr._github_client = gh
            requires = await mgr._effective_branch_requires_approval(
                "acme", "widget", "main"
            )

        assert requires is True

    @pytest.mark.asyncio
    async def test_an_error_still_answers_false(self) -> None:
        mgr, _ = make_merge_manager()
        async with GitHubAsync(token="t") as gh:
            _wire(gh, _branch_rules_route(second_page=RuntimeError("502")))
            mgr._github_client = gh
            requires = await mgr._effective_branch_requires_approval(
                "acme", "widget", "main"
            )

        assert requires is False
