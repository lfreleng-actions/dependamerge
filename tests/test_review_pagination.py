# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Decisions read every review before resting on one being absent.

The scan selects the first 20 reviews of each pull request, and the
unpaginated REST reads answered 30.  A change request, an approval or
a Copilot review past that page read as absent: the PR merged over
the change request, a duplicate approval was posted, or "requires
approval" was reported on an approved PR.  Absence is evidence only
when the reading is whole; a failed read must not stand for it.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from dependamerge.copilot_handler import CopilotCommentHandler
from dependamerge.github_async import GitHubAsync, ReviewReading, _review_listings
from dependamerge.github_client import GitHubClient
from dependamerge.github_graphql.pull_requests import SCAN_REVIEWS_PAGE_SIZE
from dependamerge.github_service import GitHubService
from dependamerge.github_service._pr_extract import scan_reviews_complete
from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.merge_manager._single_pr_context import _MergeFlow
from dependamerge.merge_signals import ReviewsUnreadableError
from dependamerge.models import PullRequestInfo, ReviewInfo
from tests.conftest import make_merge_manager

_OWNER, _REPO, _NUMBER = "owner", "repo", 7


def _review(index: int, state: str = "COMMENTED", login: str = "reviewer"):
    return {
        "id": index,
        "node_id": f"PRR_{index}",
        "state": state,
        "user": {"login": login},
        "submitted_at": "2026-01-01T00:00:00Z",
        "body": "",
    }


def _reviews(count: int, *, last_state: str, last_login: str = "maintainer"):
    entries = [_review(i) for i in range(1, count)]
    entries.append(_review(count, last_state, last_login))
    return entries


class PagedListings:
    """Serve review and review-comment listings in ``per_page`` pages.

    ``fail_on_page`` makes that page's request raise.  ``requests``
    records each ``(suffix, page)`` read.  Any other path is answered
    from ``other`` as one page.
    """

    def __init__(
        self,
        reviews: list[Any] | None = None,
        comments: list[Any] | None = None,
        *,
        fail_on_page: int | None = None,
        other: dict[str, Any] | None = None,
    ) -> None:
        self.listings = {"/reviews": reviews or [], "/comments": comments or []}
        self.fail_on_page = fail_on_page
        self.other = other or {}
        self.requests: list[tuple[str, int]] = []

    def install(self, api: Any) -> None:
        api.get = self.get
        api.get_paginated = self.get_paginated

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        for suffix, value in self.other.items():
            if path.endswith(suffix):
                return value
        return {}

    async def get_paginated(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        per_page: int = 100,
        max_pages: int | None = None,
        yield_malformed: bool = False,
    ):
        suffix = next((s for s in self.listings if path.endswith(s)), None)
        if suffix is None:
            page = await self.get(path)
            if page:
                yield page
            return
        entries = self.listings[suffix]
        page = 1
        while True:
            self.requests.append((suffix, page))
            if page == self.fail_on_page:
                raise RuntimeError(f"502 Bad Gateway on {suffix} page {page}")
            yield entries[(page - 1) * per_page : page * per_page]
            if page * per_page >= len(entries):
                return
            if max_pages is not None and page >= max_pages:
                return
            page += 1


def _pr_info(reviews: list[ReviewInfo], *, complete: bool) -> PullRequestInfo:
    return PullRequestInfo(
        number=_NUMBER,
        title="Bump thing",
        body=None,
        author="dependabot[bot]",
        head_sha="abc123",
        base_branch="main",
        head_branch="dependabot/thing",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name=f"{_OWNER}/{_REPO}",
        html_url=f"https://github.com/{_OWNER}/{_REPO}/pull/{_NUMBER}",
        reviews=reviews,
        reviews_complete=complete,
    )


def _scan_page() -> list[ReviewInfo]:
    """The first page of reviews the scan holds: nothing blocks."""
    return [
        ReviewInfo(
            id=f"PRR_{i}",
            user="reviewer",
            state="COMMENTED",
            submitted_at="2026-01-01T00:00:00Z",
        )
        for i in range(1, SCAN_REVIEWS_PAGE_SIZE + 1)
    ]


def _flow(pr: PullRequestInfo) -> _MergeFlow:
    return _MergeFlow(
        pr_info=pr,
        repo_owner=_OWNER,
        repo_name=_REPO,
        result=MergeResult(pr_info=pr, status=MergeStatus.PENDING),
    )


class TestReader:
    """The listing reader accounts for what it read."""

    @pytest.mark.asyncio
    async def test_reads_every_page(self) -> None:
        fake = PagedListings(_reviews(205, last_state="APPROVED"))
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            reading = await api.read_pull_request_reviews(_OWNER, _REPO, _NUMBER)
        assert reading.complete is True
        assert len(reading.entries) == 205
        assert reading.entries[-1]["state"] == "APPROVED"
        assert fake.requests == [("/reviews", 1), ("/reviews", 2), ("/reviews", 3)]

    @pytest.mark.asyncio
    async def test_failed_page_keeps_what_was_read(self) -> None:
        fake = PagedListings(_reviews(150, last_state="APPROVED"), fail_on_page=2)
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            reading = await api.read_pull_request_reviews(_OWNER, _REPO, _NUMBER)
        assert reading.complete is False
        assert len(reading.entries) == 100

    @pytest.mark.asyncio
    async def test_repeated_id_replaces_and_marks_incomplete(self) -> None:
        entries = [_review(i) for i in range(1, 101)]
        entries.append(_review(100, "APPROVED"))
        fake = PagedListings(entries)
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            reading = await api.read_pull_request_reviews(_OWNER, _REPO, _NUMBER)
        assert reading.complete is False
        assert len(reading.entries) == 100
        assert reading.entries[-1]["state"] == "APPROVED"

    @pytest.mark.asyncio
    async def test_malformed_page_marks_incomplete(self) -> None:
        async def _malformed(path: str, **_kwargs: Any):
            yield {"message": "not a list"}

        async with GitHubAsync(token="t") as api:
            api.get_paginated = _malformed  # type: ignore[method-assign]
            reading = await api.read_pull_request_reviews(_OWNER, _REPO, _NUMBER)
        assert reading == ReviewReading([], False)

    @pytest.mark.asyncio
    async def test_page_ceiling_on_a_full_page_marks_incomplete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_review_listings, "REVIEW_LISTING_MAX_PAGES", 2)
        fake = PagedListings(_reviews(250, last_state="APPROVED"))
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            reading = await api.read_pull_request_reviews(_OWNER, _REPO, _NUMBER)
        assert reading.complete is False
        assert len(reading.entries) == 200

    @pytest.mark.asyncio
    async def test_review_comments_past_the_default_page_are_read(self) -> None:
        comments = [
            {"id": i, "user": {"login": "someone"}, "body": f"c{i}"}
            for i in range(1, 131)
        ]
        fake = PagedListings(comments=comments)
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            reading = await api.read_pull_request_review_comments(
                _OWNER, _REPO, _NUMBER
            )
        assert reading.complete is True
        assert len(reading.entries) == 130


class TestScanCompleteness:
    """The scan says whether its first page holds every review."""

    @pytest.mark.parametrize(
        ("count", "total", "expected"),
        [
            (5, None, True),
            (SCAN_REVIEWS_PAGE_SIZE, SCAN_REVIEWS_PAGE_SIZE, True),
            (SCAN_REVIEWS_PAGE_SIZE, SCAN_REVIEWS_PAGE_SIZE + 5, False),
            (SCAN_REVIEWS_PAGE_SIZE, None, False),
        ],
    )
    def test_scan_reviews_complete(
        self, count: int, total: int | None, expected: bool
    ) -> None:
        connection: dict[str, Any] = {"nodes": [{"id": i} for i in range(count)]}
        if total is not None:
            connection["totalCount"] = total
        assert scan_reviews_complete({"reviews": connection}) is expected

    def test_pull_request_info_carries_the_flag(self) -> None:
        node = {
            "number": _NUMBER,
            "author": {"__typename": "Bot", "login": "dependabot"},
            "reviews": {
                "totalCount": 25,
                "nodes": [
                    {"id": f"PRR_{i}", "state": "COMMENTED", "author": {"login": "r"}}
                    for i in range(SCAN_REVIEWS_PAGE_SIZE)
                ],
            },
        }
        pr = GitHubService(token="t").to_pull_request_info(f"{_OWNER}/{_REPO}", node)
        assert pr.reviews_complete is False


class TestChangeRequestPastTheFirstPage:
    """A change request after the 20th review blocks the merge."""

    @pytest.mark.asyncio
    async def test_scan_path_reads_the_rest_and_skips(self) -> None:
        mgr, _ = make_merge_manager()
        fake = PagedListings(_reviews(25, last_state="CHANGES_REQUESTED"))
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            mgr._github_client = api
            pr = _pr_info(_scan_page(), complete=False)
            result = await mgr._gate_mergeable_state(_flow(pr))
        assert result is not None
        assert result.status == MergeStatus.SKIPPED
        assert "requesting changes" in (result.error or "")
        assert pr.reviews_complete is True
        assert pr.reviews[-1].id == "PRR_25"

    @pytest.mark.asyncio
    async def test_unreadable_reviews_skip_rather_than_merge(self) -> None:
        mgr, _ = make_merge_manager()
        fake = PagedListings(_reviews(150, last_state="CHANGES_REQUESTED"))
        fake.fail_on_page = 2
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            mgr._github_client = api
            pr = _pr_info(_scan_page(), complete=False)
            result = await mgr._gate_mergeable_state(_flow(pr))
        assert result is not None
        assert result.status == MergeStatus.SKIPPED
        assert "could not all be read" in (result.error or "")
        assert pr.reviews_complete is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fail_on_page", "reason"),
        [(None, "requesting changes"), (2, "could not all be read")],
    )
    async def test_conflicted_pr_is_screened_before_recovery(
        self, fail_on_page: int | None, reason: str
    ) -> None:
        mgr, _ = make_merge_manager()
        mgr._handle_merge_conflict = AsyncMock()  # type: ignore[method-assign]
        fake = PagedListings(
            _reviews(150, last_state="CHANGES_REQUESTED"), fail_on_page=fail_on_page
        )
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            mgr._github_client = api
            pr = _pr_info(_scan_page(), complete=False)
            pr.mergeable_state = "dirty"
            result = await mgr._gate_mergeable_state(_flow(pr))
        assert result is not None
        assert result.status == MergeStatus.SKIPPED
        assert reason in (result.error or "")
        mgr._handle_merge_conflict.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_review_that_does_not_convert_skips(self) -> None:
        mgr, _ = make_merge_manager()
        entries = [_review(i) for i in range(1, 25)]
        entries.append({"id": 25})
        fake = PagedListings(entries)
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            mgr._github_client = api
            pr = _pr_info(_scan_page(), complete=False)
            result = await mgr._gate_mergeable_state(_flow(pr))
        assert result is not None
        assert result.status == MergeStatus.SKIPPED
        assert "could not all be read" in (result.error or "")
        assert pr.reviews_complete is False

    @pytest.mark.asyncio
    async def test_a_whole_scan_costs_no_request(self) -> None:
        mgr, client = make_merge_manager()
        pr = _pr_info(_scan_page()[:5], complete=True)
        result = await mgr._gate_mergeable_state(_flow(pr))
        assert result is None
        client.read_pull_request_reviews.assert_not_awaited()

    def test_url_path_reads_every_review(self) -> None:
        fake = PagedListings(
            _reviews(25, last_state="CHANGES_REQUESTED"),
            other={
                f"/pulls/{_NUMBER}": {
                    "number": _NUMBER,
                    "title": "Bump thing",
                    "body": None,
                    "user": {"login": "dependabot[bot]"},
                    "head": {"sha": "abc123", "ref": "dependabot/thing"},
                    "base": {"ref": "main"},
                    "state": "open",
                    "mergeable": True,
                    "mergeable_state": "clean",
                    "html_url": "https://github.com/owner/repo/pull/7",
                }
            },
        )

        original_aenter = GitHubAsync.__aenter__

        async def _aenter(self: GitHubAsync) -> GitHubAsync:
            api = await original_aenter(self)
            fake.install(api)
            return api

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(GitHubAsync, "__aenter__", _aenter)
            pr = GitHubClient(token="t").get_pull_request_info(_OWNER, _REPO, _NUMBER)

        assert len(pr.reviews) == 25
        assert pr.reviews[-1].state == "CHANGES_REQUESTED"
        assert pr.reviews_complete is True


class TestApprovalPastTheFirstPage:
    """An approval after the 30th review is seen."""

    @pytest.mark.asyncio
    async def test_own_approval_on_page_two_skips_approving(self) -> None:
        mgr, _ = make_merge_manager()
        fake = PagedListings(
            _reviews(150, last_state="APPROVED", last_login="dependamerge-bot"),
            other={f"/pulls/{_NUMBER}": {"mergeable_state": "blocked"}},
        )
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            api.get_authenticated_user_login = AsyncMock(  # type: ignore[method-assign]
                return_value="dependamerge-bot"
            )
            api.approve_pull_request = AsyncMock()  # type: ignore[method-assign]
            mgr._github_client = api
            skip, approvers = await mgr._should_skip_approval(_OWNER, _REPO, _NUMBER)
        assert (skip, approvers) == (True, "dependamerge-bot")
        api.approve_pull_request.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fail_on_page", [1, 2])
    @pytest.mark.parametrize("approver", ["dependamerge-bot", "other-maintainer"])
    async def test_a_partial_reading_stops_the_approval(
        self, fail_on_page: int, approver: str
    ) -> None:
        mgr, _ = make_merge_manager()
        mgr._fetch_pr_state = AsyncMock(return_value={"draft": False})  # type: ignore[method-assign]
        reviews = [_review(1, "APPROVED", approver)]
        reviews += [_review(i) for i in range(2, 151)]
        fake = PagedListings(
            reviews,
            fail_on_page=fail_on_page,
            other={f"/pulls/{_NUMBER}": {"mergeable_state": "clean"}},
        )
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            api.get_authenticated_user_login = AsyncMock(  # type: ignore[method-assign]
                return_value="dependamerge-bot"
            )
            api.post = AsyncMock()  # type: ignore[method-assign]
            mgr._github_client = api
            with pytest.raises(ReviewsUnreadableError):
                await mgr._approve_pr(_OWNER, _REPO, _NUMBER)
        api.post.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("own_on_page_one", "stops"), [(False, True), (True, False)]
    )
    async def test_a_partial_re_read_after_a_failed_post_never_reposts(
        self, monkeypatch: pytest.MonkeyPatch, own_on_page_one: bool, stops: bool
    ) -> None:
        """A 500 may have created the review among those not read."""
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        login = "dependamerge-bot" if own_on_page_one else "other-maintainer"
        reviews = [_review(1, "APPROVED", login)]
        reviews += [_review(i) for i in range(2, 151)]
        fake = PagedListings(reviews, fail_on_page=2)
        request = httpx.Request("POST", "https://api.github.com/x")
        error = httpx.HTTPStatusError(
            "boom", request=request, response=httpx.Response(500, request=request)
        )
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            api.get_authenticated_user_login = AsyncMock(  # type: ignore[method-assign]
                return_value="dependamerge-bot"
            )
            api.post = AsyncMock(side_effect=[error, {}])  # type: ignore[method-assign]
            if stops:
                with pytest.raises(ReviewsUnreadableError):
                    await api.approve_pull_request(_OWNER, _REPO, _NUMBER, "ok")
            else:
                await api.approve_pull_request(_OWNER, _REPO, _NUMBER, "ok")
        api.post.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "lookup",
        [AsyncMock(return_value=None), AsyncMock(side_effect=RuntimeError("x"))],
        ids=["login-none", "login-raises"],
    )
    async def test_an_unknown_identity_after_a_failed_post_never_reposts(
        self, monkeypatch: pytest.MonkeyPatch, lookup: AsyncMock
    ) -> None:
        """Without our login, an approval the 500 created goes unseen."""
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        fake = PagedListings([_review(1, "APPROVED", "dependamerge-bot")])
        request = httpx.Request("POST", "https://api.github.com/x")
        error = httpx.HTTPStatusError(
            "boom", request=request, response=httpx.Response(500, request=request)
        )
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            api.get_authenticated_user_login = lookup  # type: ignore[method-assign]
            api.post = AsyncMock(side_effect=[error, {}])  # type: ignore[method-assign]
            with pytest.raises(ReviewsUnreadableError):
                await api.approve_pull_request(_OWNER, _REPO, _NUMBER, "ok")
        api.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unreadable_reviews_skip_the_pr_unapproved(self) -> None:
        mgr, client = make_merge_manager()
        mgr._get_merge_method_for_repo = AsyncMock(return_value="merge")  # type: ignore[method-assign]
        mgr._run_merge_sequence = AsyncMock(  # type: ignore[method-assign]
            side_effect=ReviewsUnreadableError(f"{_OWNER}/{_REPO}#{_NUMBER}")
        )
        result = await mgr._merge_single_pr_impl(_pr_info([], complete=True))
        assert result.status is MergeStatus.SKIPPED
        assert result.error == "reviews could not all be read; not approved"
        client.approve_pull_request.assert_not_called()
        client.merge_pull_request.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fail_on_page", "expected_complete"), [(None, True), (2, False)]
    )
    async def test_requires_approval_needs_a_whole_reading(
        self, fail_on_page: int | None, expected_complete: bool
    ) -> None:
        reviews = [_review(i) for i in range(1, 151)]
        if fail_on_page is None:
            reviews[-1] = _review(150, "APPROVED")
        fake = PagedListings(
            reviews,
            fail_on_page=fail_on_page,
            other={
                f"/pulls/{_NUMBER}": {"base": {"ref": "main"}},
                "/check-runs": {"check_runs": []},
                "/status": {"statuses": []},
            },
        )
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            api.get_required_status_checks = AsyncMock(return_value=[])  # type: ignore[method-assign]
            api._detect_branch_protection_kind = AsyncMock(  # type: ignore[method-assign]
                return_value="ruleset"
            )
            reason = await api.analyze_block_reason(
                _OWNER, _REPO, _NUMBER, "abc123", base_branch="main"
            )
            reading = await api.read_pull_request_reviews(_OWNER, _REPO, _NUMBER)
        assert reading.complete is expected_complete
        assert "requires approval" not in reason

    @staticmethod
    async def _block_reason(fake: PagedListings) -> str:
        fake.other = {
            f"/pulls/{_NUMBER}": {"base": {"ref": "main"}},
            "/check-runs": {"check_runs": []},
            "/status": {"statuses": []},
        }
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            api.get_required_status_checks = AsyncMock(return_value=[])  # type: ignore[method-assign]
            api._detect_branch_protection_kind = AsyncMock(  # type: ignore[method-assign]
                return_value="ruleset"
            )
            return await api.analyze_block_reason(
                _OWNER, _REPO, _NUMBER, "abc123", base_branch="main"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fail_on_page", [None, 2])
    async def test_requires_approval_needs_every_review_comment(
        self, fail_on_page: int | None
    ) -> None:
        comments = [
            {"id": i, "user": {"login": "someone"}, "body": f"c{i}"}
            for i in range(1, 151)
        ]
        fake = PagedListings(
            _reviews(5, last_state="COMMENTED"),
            comments,
            fail_on_page=fail_on_page,
        )
        reason = await self._block_reason(fake)
        if fail_on_page is None:
            assert reason == "Blocked by branch protection (requires approval)"
        else:
            assert "requires approval" not in reason

    @pytest.mark.asyncio
    async def test_a_review_without_a_state_leaves_the_reading_incomplete(
        self,
    ) -> None:
        reviews: list[Any] = [_review(i) for i in range(1, 25)]
        reviews.append({"id": 25})
        fake = PagedListings(reviews)
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            reading = await api.read_pull_request_reviews(_OWNER, _REPO, _NUMBER)
        assert reading.complete is False
        assert len(reading.entries) == 25
        assert "requires approval" not in await self._block_reason(fake)


class TestCopilotReviewPastTheFirstPage:
    """A Copilot review after the 20th is offered for dismissal."""

    @pytest.mark.asyncio
    async def test_copilot_review_reaches_the_handler(self) -> None:
        mgr, _ = make_merge_manager(dismiss_copilot=True)
        handler = MagicMock()
        handler.dismiss_copilot_comments_for_pr = AsyncMock()
        mgr._copilot_handler = handler
        fake = PagedListings(
            _reviews(
                25,
                last_state="COMMENTED",
                last_login="copilot-pull-request-reviewer[bot]",
            )
        )
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            mgr._github_client = api
            pr = _pr_info(_scan_page(), complete=False)
            result = await mgr._gate_copilot_reviews(_flow(pr))
        assert result is None
        offered = handler.dismiss_copilot_comments_for_pr.await_args.args[0]
        assert offered.reviews[-1].user == "copilot-pull-request-reviewer[bot]"

    @pytest.mark.asyncio
    async def test_unreadable_reviews_fail_copilot_processing(self) -> None:
        mgr, _ = make_merge_manager(dismiss_copilot=True)
        handler = MagicMock()
        handler.dismiss_copilot_comments_for_pr = AsyncMock()
        mgr._copilot_handler = handler
        fake = PagedListings(_reviews(150, last_state="COMMENTED"), fail_on_page=2)
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            mgr._github_client = api
            pr = _pr_info(_scan_page(), complete=False)
            result = await mgr._gate_copilot_reviews(_flow(pr))
        assert result is not None
        assert result.status == MergeStatus.FAILED
        handler.dismiss_copilot_comments_for_pr.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unreadable_review_comments_fail_copilot_processing(
        self,
    ) -> None:
        mgr, _ = make_merge_manager(dismiss_copilot=True)
        comments = [
            {"id": i, "user": {"login": "someone"}, "body": f"c{i}"}
            for i in range(1, 151)
        ]
        fake = PagedListings(comments=comments, fail_on_page=2)
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            mgr._github_client = api
            handler = CopilotCommentHandler(api)
            handler.resolve_copilot_review = AsyncMock()  # type: ignore[method-assign]
            mgr._copilot_handler = handler
            copilot = ReviewInfo(
                id="PRR_copilot",
                user="copilot-pull-request-reviewer[bot]",
                state="CHANGES_REQUESTED",
                submitted_at="2026-01-01T00:00:00Z",
            )
            pr = _pr_info([*_scan_page()[:5], copilot], complete=True)
            result = await mgr._gate_copilot_reviews(_flow(pr))
        assert result is not None
        assert result.status == MergeStatus.FAILED
        assert fake.requests == [("/comments", 1), ("/comments", 2)]
        handler.resolve_copilot_review.assert_not_awaited()
