# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Block-reason analysis reads every check run and status context.

Both listings default to 30 entries a page.  A required check beyond
the first page used to read as never having run, and the false
"missing required status" reason led to needless rebase requests.
Absence is evidence only when the reading is whole; a failed or
truncated read must not invent a missing check.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.github_async import GitHubAsync
from dependamerge.models import PullRequestInfo
from tests.conftest import FakeCommitListings, make_merge_manager

_REQUIRED = ["DCO", "pre-commit.ci - pr"]


def _run(name: str, status: str = "completed", conclusion: str | None = "success"):
    return {"name": name, "status": status, "conclusion": conclusion}


def _fillers(count: int) -> list[dict[str, Any]]:
    return [_run(f"job-{i}") for i in range(count)]


def _precommit_status() -> list[dict[str, Any]]:
    return [{"context": "pre-commit.ci - pr", "state": "success"}]


async def _reason(fake: FakeCommitListings, required: list[str] | None = None) -> str:
    fake.other = {
        "/reviews": [{"state": "APPROVED", "user": {"login": "reviewer"}}],
        "/comments": [],
    }
    async with GitHubAsync(token="t") as api:
        fake.install(api)
        api.get_required_status_checks = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"context": c} for c in (required or [])]
        )
        api._detect_branch_protection_kind = AsyncMock(  # type: ignore[method-assign]
            return_value="ruleset"
        )
        return await api.analyze_block_reason(
            "owner", "repo", 126, "5d2b5433", base_branch="main"
        )


_ALL_CLEAR = "Blocked by repository ruleset (no specific failing condition detected)"


class TestEveryPageIsRead:
    @pytest.mark.asyncio
    async def test_required_run_beyond_first_30_is_not_missing(self) -> None:
        # GitHub lists runs newest first, so DCO, which reports first,
        # sits last: on page two at the default page size.
        fake = FakeCommitListings(_fillers(48) + [_run("DCO")], _precommit_status())

        reason = await _reason(fake, _REQUIRED)

        assert reason == _ALL_CLEAR
        mgr, client = make_merge_manager(fix_out_of_date=True)
        client.get_behind_by = AsyncMock(return_value=3)
        pr = PullRequestInfo(
            number=126,
            title="Chore: Bump foo",
            body="",
            author="dependabot[bot]",
            head_sha="5d2b5433",
            base_branch="main",
            head_branch="dependabot/foo",
            state="open",
            mergeable=True,
            mergeable_state="blocked",
            behind_by=0,
            files_changed=[],
            repository_full_name="owner/repo",
            html_url="https://github.com/owner/repo/pull/126",
        )
        assert not await mgr._blocked_pr_needs_rebase(pr, "owner", "repo", reason)

    @pytest.mark.asyncio
    async def test_required_status_beyond_first_30_is_not_missing(self) -> None:
        statuses = [
            {"context": f"ctx-{i}", "state": "success"} for i in range(40)
        ] + _precommit_status()
        fake = FakeCommitListings([_run("DCO")], statuses)

        assert await _reason(fake, _REQUIRED) == _ALL_CLEAR

    @pytest.mark.asyncio
    async def test_failing_run_on_second_page_is_reported(self) -> None:
        runs = _fillers(120) + [_run("Zizmor Scan", conclusion="failure")]
        fake = FakeCommitListings(runs + _fillers(10), _precommit_status())

        assert await _reason(fake) == "Blocked by failing check: Zizmor Scan"

    @pytest.mark.asyncio
    async def test_pending_required_run_on_second_page_is_pending(self) -> None:
        runs = _fillers(120) + [_run("DCO", status="in_progress", conclusion=None)]
        fake = FakeCommitListings(runs, _precommit_status())

        assert await _reason(fake, _REQUIRED) == (
            "Blocked by pending required check: DCO"
        )

    @pytest.mark.asyncio
    async def test_a_short_listing_costs_one_request_each(self) -> None:
        fake = FakeCommitListings(_fillers(29) + [_run("DCO")], _precommit_status())

        assert await _reason(fake, _REQUIRED) == _ALL_CLEAR
        assert sorted(fake.requests) == [("check-runs", 1), ("status", 1)]


class TestAbsenceNeedsACompleteReading:
    @pytest.mark.asyncio
    async def test_failed_second_page_reports_nothing_missing(self) -> None:
        fake = FakeCommitListings(
            _fillers(120) + [_run("DCO")],
            _precommit_status(),
            fail_on={("check-runs", 2)},
        )

        assert await _reason(fake, _REQUIRED) == _ALL_CLEAR

    @pytest.mark.asyncio
    async def test_failing_run_from_partial_reading_is_reported(self) -> None:
        runs = [_run("Lint", conclusion="failure")] + _fillers(119) + [_run("DCO")]
        fake = FakeCommitListings(
            runs, _precommit_status(), fail_on={("check-runs", 2)}
        )

        assert await _reason(fake, _REQUIRED) == "Blocked by failing check: Lint"

    @pytest.mark.asyncio
    async def test_failed_first_request_reports_nothing_missing(self) -> None:
        fake = FakeCommitListings(
            [_run("DCO")], _precommit_status(), fail_on={("check-runs", 1)}
        )

        assert await _reason(fake, _REQUIRED) == _ALL_CLEAR

    @pytest.mark.asyncio
    async def test_absent_from_complete_reading_is_still_missing(self) -> None:
        fake = FakeCommitListings(_fillers(130), _precommit_status())

        assert await _reason(fake, _REQUIRED) == (
            "Blocked by missing required status: DCO"
        )


def _serve(api: GitHubAsync, pages: list[Any]) -> None:
    """Answer every ``get_paginated`` call on *api* with *pages*."""

    async def _pages(
        path, *, params=None, per_page=100, max_pages=None, yield_malformed=False
    ):
        for page in pages:
            yield page

    api.get_paginated = _pages  # type: ignore[method-assign]


class TestMalformedListings:
    @pytest.mark.asyncio
    async def test_unusable_run_name_does_not_abort_analysis(self) -> None:
        runs = [{"name": 1, "status": "completed"}, _run("Lint", conclusion="failure")]
        fake = FakeCommitListings(runs, _precommit_status())

        assert await _reason(fake, _REQUIRED) == "Blocked by failing check: Lint"

    @pytest.mark.asyncio
    async def test_unusable_status_context_reports_nothing_missing(self) -> None:
        statuses = [{"context": ["DCO"], "state": "success"}] + _precommit_status()
        fake = FakeCommitListings([], statuses)

        assert await _reason(fake, _REQUIRED) == _ALL_CLEAR

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "pages",
        [
            pytest.param([], id="no-page"),
            pytest.param([{"message": "Not Found"}], id="listing-key-missing"),
            pytest.param([{"check_runs": None}], id="null"),
            pytest.param([{"check_runs": ["DCO"]}], id="non-dict-entry"),
        ],
    )
    async def test_malformed_reading_is_incomplete(self, pages: list[Any]) -> None:
        async with GitHubAsync(token="t") as api:
            _serve(api, pages)
            reading = await api.read_check_runs_for_ref("o", "r", "sha")

        assert reading.entries == []
        assert not reading.complete

    @pytest.mark.asyncio
    async def test_repeated_entry_id_keeps_later_observation(self) -> None:
        # A run posted between pages shifts "b" onto page two: three
        # rows for three advertised runs, one of them never returned.
        # "b" finished between the two reads; the later state counts.
        pages = [
            {
                "total_count": 2,
                "check_runs": [
                    {"id": 1, "name": "a", "status": "completed"},
                    {"id": 2, "name": "b", "status": "in_progress"},
                ],
            },
            {
                "total_count": 3,
                "check_runs": [
                    {
                        "id": 2,
                        "name": "b",
                        "status": "completed",
                        "conclusion": "failure",
                    }
                ],
            },
        ]
        async with GitHubAsync(token="t") as api:
            _serve(api, pages)
            reading = await api.read_check_runs_for_ref("o", "r", "sha")

        assert [(run["name"], run["status"]) for run in reading.entries] == [
            ("a", "completed"),
            ("b", "completed"),
        ]
        assert not reading.complete

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param({"id": 3, "status": "completed"}, id="name-missing"),
            pytest.param({"id": 3, "name": None, "status": "completed"}, id="null"),
            pytest.param({"id": 3, "name": "  ", "status": "completed"}, id="blank"),
            pytest.param({"id": 3, "name": "DCO"}, id="status-missing"),
            pytest.param({"id": 3, "name": "DCO", "status": None}, id="null-status"),
        ],
    )
    async def test_unidentifiable_run_is_incomplete(
        self, entry: dict[str, Any]
    ) -> None:
        page = {"total_count": 1, "check_runs": [entry]}
        async with GitHubAsync(token="t") as api:
            _serve(api, [page])
            reading = await api.read_check_runs_for_ref("o", "r", "sha")

        assert reading == ([], False)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param({"id": 3, "context": "", "state": "success"}, id="blank"),
            pytest.param({"id": 3, "context": "DCO"}, id="state-missing"),
        ],
    )
    async def test_unidentifiable_status_is_incomplete(
        self, entry: dict[str, Any]
    ) -> None:
        page = {"total_count": 1, "statuses": [entry]}
        async with GitHubAsync(token="t") as api:
            _serve(api, [page])
            reading = await api.read_commit_statuses("o", "r", "sha")

        assert reading == ([], False)

    @pytest.mark.asyncio
    async def test_entry_without_id_across_pages_is_incomplete(self) -> None:
        # Without ids a row repeated across a shifted page boundary
        # cannot be told apart from a new one, so it could stand in for
        # the run that went unseen.
        run = {"name": "a", "status": "completed", "conclusion": "success"}
        pages = [
            {"total_count": 2, "check_runs": [run]},
            {"total_count": 2, "check_runs": [run]},
        ]
        async with GitHubAsync(token="t") as api:
            _serve(api, pages)
            reading = await api.read_check_runs_for_ref("o", "r", "sha")

        assert not reading.complete

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "count",
        [
            pytest.param(None, id="null"),
            pytest.param("1", id="string"),
            pytest.param(-1, id="negative"),
            pytest.param(True, id="bool"),
        ],
    )
    async def test_unusable_total_count_is_incomplete(self, count: Any) -> None:
        run = {"id": 1, "name": "a", "status": "completed", "conclusion": "success"}
        async with GitHubAsync(token="t") as api:
            _serve(api, [{"total_count": count, "check_runs": [run]}])
            reading = await api.read_check_runs_for_ref("o", "r", "sha")

        assert reading == ([run], False)

    @pytest.mark.asyncio
    async def test_page_without_total_count_is_incomplete(self) -> None:
        # The transport ends silently on an empty page, so a later page
        # that never arrived leaves only the count to notice it.
        status = {"id": 1, "context": "a", "state": "success"}
        async with GitHubAsync(token="t") as api:
            _serve(api, [{"statuses": [status]}])
            reading = await api.read_commit_statuses("o", "r", "sha")

        assert reading == ([status], False)

    @pytest.mark.asyncio
    async def test_strict_readers_raise_on_incomplete_reading(self) -> None:
        from dependamerge.github_async._commit_listings import IncompleteReadingError

        run = {"id": 1, "name": "a", "status": "completed"}
        status = {"id": 1, "context": "a", "state": "success"}
        async with GitHubAsync(token="t") as api:
            _serve(api, [{"total_count": 201, "check_runs": [run]}])
            with pytest.raises(IncompleteReadingError):
                await api.get_check_runs_for_ref("o", "r", "sha")
            _serve(api, [{"total_count": 2, "statuses": [status]}])
            with pytest.raises(IncompleteReadingError):
                await api.get_commit_statuses("o", "r", "sha")
