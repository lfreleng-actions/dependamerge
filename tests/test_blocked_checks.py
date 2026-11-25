# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""``blocked`` decides failing checks from the rollup's counts.

The page holds the first ``SCAN_CHECKS_PAGE_SIZE`` checks; the rollup's
per-state counts cover them all.  No failing count means no failing
check and no further request.  A failing count with a short page reads
the commit's check runs and statuses in full, and a read that fails
keeps the pull request in the report with a note.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.cli._app import console
from dependamerge.cli._reports import _display_blocked_results
from dependamerge.github_async._commit_listings import CommitReading
from dependamerge.github_graphql.pull_requests import SCAN_CHECKS_PAGE_SIZE
from dependamerge.github_service import GitHubService
from dependamerge.models import OrganizationScanResult

_OID = "c0ffee"


def _run(index: int, conclusion: str = "SUCCESS", at: str = "01") -> dict[str, Any]:
    return {
        "__typename": "CheckRun",
        "name": f"check-{index}",
        "status": "COMPLETED",
        "conclusion": conclusion,
        "completedAt": f"2026-01-01T00:00:{at}Z",
    }


def _rest(run: dict[str, Any]) -> dict[str, Any]:
    """The REST spelling of a GraphQL check run."""
    return {
        "id": hash((run["name"], run["completedAt"])),
        "name": run["name"],
        "status": run["status"].lower(),
        "conclusion": run["conclusion"].lower(),
        "completed_at": run["completedAt"],
    }


def _node(runs: list[dict[str, Any]], *, counts: bool = True) -> dict[str, Any]:
    contexts: dict[str, Any] = {"nodes": runs[:SCAN_CHECKS_PAGE_SIZE]}
    if counts:
        by_state: dict[str, int] = {}
        for run in runs:
            by_state[run["conclusion"]] = by_state.get(run["conclusion"], 0) + 1
        contexts |= {
            "checkRunCount": len(runs),
            "checkRunCountsByState": [
                {"state": state, "count": count} for state, count in by_state.items()
            ],
            "statusContextCount": 0,
            "statusContextCountsByState": [],
        }
    commit = {"oid": _OID, "statusCheckRollup": {"contexts": contexts}}
    return {
        "number": 9,
        "title": "Bump foo",
        "url": "https://github.com/acme/widget/pull/9",
        "author": {"__typename": "Bot", "login": "dependabot"},
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "BLOCKED",
        "headRefOid": "head",
        "commits": {"nodes": [{"commit": commit}]},
        "comments": {"totalCount": 0, "nodes": []},
    }


def _service(
    runs: list[dict[str, Any]] | None = None, *, complete: bool = True
) -> GitHubService:
    svc = GitHubService(token="t")
    entries = [_rest(run) for run in runs or []]
    svc._api.read_check_runs_for_ref = AsyncMock(  # type: ignore[method-assign]
        return_value=CommitReading(entries, complete)
    )
    svc._api.read_commit_statuses = AsyncMock(  # type: ignore[method-assign]
        return_value=CommitReading([], True)
    )
    return svc


def _reads(svc: GitHubService) -> int:
    runs: AsyncMock = svc._api.read_check_runs_for_ref  # type: ignore[assignment]
    statuses: AsyncMock = svc._api.read_commit_statuses  # type: ignore[assignment]
    return runs.await_count + statuses.await_count


@pytest.mark.asyncio
async def test_no_failing_count_reads_nothing_further() -> None:
    runs = [_run(i) for i in range(150)]
    svc = _service(runs)
    assert await svc._analyze_pr_node("acme/widget", _node(runs)) is None
    assert _reads(svc) == 0


@pytest.mark.asyncio
async def test_a_failing_check_past_the_page_is_found_by_a_full_read() -> None:
    runs = [_run(i) for i in range(150)]
    runs[120] = _run(120, "FAILURE")
    svc = _service(runs)
    result = await svc._analyze_pr_node("acme/widget", _node(runs))
    assert result is not None
    (reason,) = result.reasons
    assert reason.details == "Failing checks: check-120"
    svc._api.read_check_runs_for_ref.assert_awaited_once_with(  # type: ignore[attr-defined]
        "acme", "widget", _OID
    )


@pytest.mark.asyncio
async def test_a_superseded_cancelled_run_past_the_page_is_not_reported() -> None:
    runs = [_run(i) for i in range(SCAN_CHECKS_PAGE_SIZE)]
    runs += [_run(100, "CANCELLED", at="01"), _run(100, "SUCCESS", at="09")]
    svc = _service(runs)
    assert await svc._analyze_pr_node("acme/widget", _node(runs)) is None
    assert _reads(svc) == 2


@pytest.mark.asyncio
async def test_a_failed_full_read_keeps_the_pull_request_with_a_note() -> None:
    runs = [_run(i) for i in range(150)]
    runs[120] = _run(120, "FAILURE")
    svc = _service([], complete=False)
    result = await svc._analyze_pr_node("acme/widget", _node(runs))
    assert result is not None
    (reason,) = result.reasons
    assert reason.type == "failing_checks"
    assert reason.details is not None
    assert "could not be read in full" in reason.details
    assert reason.description == (
        "Required status checks are failing (the check list could not be read in full)"
    )


@pytest.mark.asyncio
async def test_the_table_shows_that_the_check_list_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = [_run(i) for i in range(150)]
    runs[120] = _run(120, "FAILURE")
    svc = _service([], complete=False)
    pr = await svc._analyze_pr_node("acme/widget", _node(runs))
    assert pr is not None
    scan = OrganizationScanResult(
        organization="acme",
        total_repositories=1,
        scanned_repositories=1,
        total_prs=1,
        unmergeable_prs=[pr],
        scan_timestamp="2026-01-01T00:00:00Z",
    )
    monkeypatch.setenv("COLUMNS", "400")
    with console.capture() as captured:
        _display_blocked_results(scan, "table")
    assert "the check list could not be read in full" in captured.get()


@pytest.mark.asyncio
async def test_a_partial_read_reports_what_it_found() -> None:
    runs = [_run(i) for i in range(150)]
    runs[5] = _run(5, "FAILURE")
    runs[120] = _run(120, "TIMED_OUT")
    svc = _service(runs[110:130], complete=False)
    result = await svc._analyze_pr_node("acme/widget", _node(runs))
    assert result is not None
    assert result.reasons[0].details == (
        "Failing checks: check-120, check-5 (the check list could not be read in full)"
    )


@pytest.mark.asyncio
async def test_a_failing_check_within_the_page_costs_no_request() -> None:
    runs = [_run(i) for i in range(30)]
    runs[7] = _run(7, "STARTUP_FAILURE")
    svc = _service(runs)
    result = await svc._analyze_pr_node("acme/widget", _node(runs))
    assert result is not None
    assert result.reasons[0].details == "Failing checks: check-7"
    assert _reads(svc) == 0


@pytest.mark.asyncio
async def test_a_short_page_without_counts_is_judged_as_before() -> None:
    runs = [_run(i) for i in range(12)]
    runs[3] = _run(3, "FAILURE")
    svc = _service(runs)
    result = await svc._analyze_pr_node("acme/widget", _node(runs, counts=False))
    assert result is not None
    assert result.reasons[0].details == "Failing checks: check-3"
    assert _reads(svc) == 0


@pytest.mark.asyncio
async def test_a_failing_status_past_the_page_is_found() -> None:
    runs = [_run(i) for i in range(SCAN_CHECKS_PAGE_SIZE)]
    node = _node(runs)
    contexts = node["commits"]["nodes"][0]["commit"]["statusCheckRollup"]["contexts"]
    contexts["statusContextCount"] = 1
    contexts["statusContextCountsByState"] = [{"state": "ERROR", "count": 1}]
    svc = _service(runs)
    svc._api.read_commit_statuses = AsyncMock(  # type: ignore[method-assign]
        return_value=CommitReading([{"context": "ci/legacy", "state": "error"}], True)
    )
    result = await svc._analyze_pr_node("acme/widget", node)
    assert result is not None
    assert result.reasons[0].details == "Failing checks: ci/legacy"
