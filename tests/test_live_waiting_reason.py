# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""The end-of-run reason for a refused, ``blocked`` PR names its live head.

After a rebase the failure recorded against the old head no longer holds
the PR; a required check that has not finished on the new head does.
The reason says so, and when nothing can be read it at least says which
head it describes.  The status is never changed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dependamerge.github_async._commit_listings import CommitReading
from dependamerge.github_async._reviews import IssueComments
from dependamerge.merge_manager import AsyncMergeManager, MergeResult, MergeStatus
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

NOW = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
OLD_HEAD = "0123456789abcdef0123456789abcdef01234567"
NEW_HEAD = "fedcba9876543210fedcba9876543210fedcba98"
PRECOMMIT = "pre-commit.ci - pr"
STALE = "required status check: pre-commit.ci - pr failed"

pytestmark = pytest.mark.xfail(
    strict=True, reason="the end-of-run reason does not read unfinished checks yet"
)


@pytest.fixture(autouse=True)
def _clock() -> Iterator[None]:
    with patch("dependamerge.merge_manager._live_waiting._utcnow", return_value=NOW):
        yield


def _iso(ago: timedelta) -> str:
    return (NOW - ago).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pr() -> PullRequestInfo:
    return PullRequestInfo(
        number=9,
        title="Chore: Bump foo from 1 to 2",
        body="bump",
        author="dependabot[bot]",
        head_sha=OLD_HEAD,
        base_branch="main",
        head_branch="dependabot/pip/foo-2",
        state="open",
        mergeable=False,
        mergeable_state="blocked",
        behind_by=0,
        files_changed=[],
        repository_full_name="o/r",
        html_url="https://github.com/o/r/pull/9",
    )


def _manager(
    required: list[str | dict[str, Any]],
    *,
    statuses: list[dict[str, Any]] | None = None,
    runs: list[dict[str, Any]] | None = None,
    failing_contexts: list[str] | None = None,
    complete: bool = True,
    runs_complete: bool | None = None,
    required_ok: bool = True,
    comments: list[dict[str, Any]] | None = None,
) -> tuple[AsyncMergeManager, AsyncMock]:
    mgr, client = make_merge_manager()
    checks = [
        entry if isinstance(entry, dict) else {"context": entry} for entry in required
    ]
    client.get_required_status_checks_reliable = AsyncMock(
        return_value=(checks, required_ok)
    )
    client.get_failing_status_contexts = AsyncMock(
        return_value=list(failing_contexts or [])
    )
    client.get_check_runs_for_ref = AsyncMock(return_value=[])
    client.get_failing_workflow_run_names_for_sha = AsyncMock(return_value=[])
    client.read_commit_statuses = AsyncMock(
        return_value=CommitReading(list(statuses or []), complete)
    )
    client.read_check_runs_for_ref = AsyncMock(
        return_value=CommitReading(
            list(runs or []), complete if runs_complete is None else runs_complete
        )
    )
    client.list_issue_comments = AsyncMock(
        return_value=IssueComments(list(comments or []), complete=True)
    )
    return mgr, client


async def _reason(mgr: AsyncMergeManager, pr: PullRequestInfo) -> MergeResult:
    result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error=STALE)
    refreshed = {"head": {"sha": NEW_HEAD}, "base": {"ref": "main"}}
    return await mgr._name_the_live_blocker(pr, result, "blocked", refreshed)


class TestAnUnfinishedRequiredCheckIsTheReason:
    @pytest.mark.asyncio
    async def test_a_pending_status_on_the_live_head(self) -> None:
        mgr, client = _manager(
            ["DCO"],
            statuses=[
                {
                    "context": "DCO",
                    "state": "pending",
                    "updated_at": _iso(timedelta(0, 42)),
                }
            ],
        )

        out = await _reason(mgr, _pr())

        assert out.error == "waiting on required status check: DCO (pending 42s)"
        assert out.status is MergeStatus.FAILED
        assert STALE in (out.warning or "")
        client.read_commit_statuses.assert_awaited_once_with("o", "r", NEW_HEAD)

    @pytest.mark.asyncio
    async def test_a_check_run_in_progress(self) -> None:
        mgr, _ = _manager(
            ["build"],
            runs=[
                {
                    "name": "build",
                    "status": "in_progress",
                    "started_at": _iso(timedelta(minutes=3)),
                }
            ],
        )

        out = await _reason(mgr, _pr())

        assert out.error == "waiting on required status check: build (pending 3m)"

    @pytest.mark.asyncio
    async def test_a_context_not_reported_yet(self) -> None:
        mgr, _ = _manager(["DCO"])

        out = await _reason(mgr, _pr())

        assert out.error == "waiting on required status check: DCO (not reported yet)"

    @pytest.mark.asyncio
    async def test_a_required_failure_beside_a_pending_one_is_the_reason(
        self,
    ) -> None:
        mgr, _ = _manager(
            ["DCO", "lint"],
            statuses=[
                {
                    "context": "DCO",
                    "state": "pending",
                    "updated_at": _iso(timedelta(0)),
                },
                {"context": "lint", "state": "failure"},
            ],
            failing_contexts=["lint"],
        )

        out = await _reason(mgr, _pr())

        assert out.error == "blocked by required status check: lint"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("state", "probe_fails"),
        [("failure", True), ("error", False)],
        ids=["failed-probe", "failed-since-probe"],
    )
    async def test_a_required_status_failure_the_probe_missed_is_the_reason(
        self, state: str, probe_fails: bool
    ) -> None:
        mgr, client = _manager(
            ["DCO", "lint"],
            statuses=[
                {
                    "context": "DCO",
                    "state": "pending",
                    "updated_at": _iso(timedelta(0)),
                },
                {"context": "lint", "state": state},
            ],
        )
        if probe_fails:
            client.get_failing_status_contexts = AsyncMock(
                side_effect=RuntimeError("boom")
            )

        out = await _reason(mgr, _pr())

        assert out.error == "blocked by required status check: lint"

    @pytest.mark.asyncio
    async def test_an_incomplete_reading_claims_nothing_unreported(self) -> None:
        mgr, _ = _manager(["DCO"], complete=False)

        out = await _reason(mgr, _pr())

        assert "not reported" not in (out.error or "")
        assert out.error == (
            f"{STALE} (read at {OLD_HEAD[:7]}; the head is now {NEW_HEAD[:7]})"
        )


class TestThePrecommitNote:
    @pytest.mark.asyncio
    async def test_a_pending_run_without_a_nudge_says_when_one_comes(self) -> None:
        mgr, _ = _manager(
            [PRECOMMIT],
            statuses=[
                {
                    "context": PRECOMMIT,
                    "state": "pending",
                    "updated_at": _iso(timedelta(seconds=42)),
                }
            ],
        )

        out = await _reason(mgr, _pr())

        assert out.error == (
            "waiting on required status check: pre-commit.ci - pr (pending 42s)"
            "; a re-run nudges it once it has been pending 5m"
        )

    @pytest.mark.asyncio
    async def test_a_recent_nudge_is_quoted_with_its_age(self) -> None:
        mgr, _ = _manager(
            [PRECOMMIT],
            statuses=[
                {
                    "context": PRECOMMIT,
                    "state": "pending",
                    "updated_at": _iso(timedelta(minutes=2)),
                }
            ],
            comments=[
                {"body": "pre-commit.ci run", "created_at": _iso(timedelta(hours=3))},
                {
                    "body": "pre-commit.ci run",
                    "created_at": _iso(timedelta(minutes=20)),
                },
            ],
        )

        out = await _reason(mgr, _pr())

        assert out.error == (
            "waiting on required status check: pre-commit.ci - pr (pending 2m)"
            '; "pre-commit.ci run" posted 20m ago'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reading", ["fails", "misses-it"])
    async def test_a_nudge_this_run_posted_is_quoted_when_rereading_misses_it(
        self, reading: str
    ) -> None:
        mgr, client = _manager(
            [PRECOMMIT],
            statuses=[
                {
                    "context": PRECOMMIT,
                    "state": "pending",
                    "updated_at": _iso(timedelta(minutes=2)),
                }
            ],
        )
        pr = _pr()
        with patch(
            "dependamerge.merge_manager._precommit_ci._utcnow",
            return_value=NOW - timedelta(minutes=20),
        ):
            assert await mgr._post_precommit_trigger("o", "r", pr)
        if reading == "fails":
            client.list_issue_comments = AsyncMock(side_effect=RuntimeError("boom"))
        else:
            client.list_issue_comments = AsyncMock(
                return_value=IssueComments([], complete=False)
            )

        out = await _reason(mgr, pr)

        assert out.error == (
            "waiting on required status check: pre-commit.ci - pr (pending 2m)"
            '; "pre-commit.ci run" posted 20m ago'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reading", ["fails", "misses-it"])
    async def test_a_nudge_the_cooldown_saw_is_quoted_when_rereading_misses_it(
        self, reading: str
    ) -> None:
        """A manual nudge that held the post back is still the reason's note."""
        mgr, client = _manager(
            [PRECOMMIT],
            statuses=[
                {
                    "context": PRECOMMIT,
                    "state": "pending",
                    "updated_at": _iso(timedelta(minutes=2)),
                }
            ],
            comments=[
                {
                    "body": "pre-commit.ci run",
                    "created_at": _iso(timedelta(minutes=20)),
                }
            ],
        )
        client.post_issue_comment = AsyncMock()
        pr = _pr()
        with patch(
            "dependamerge.merge_manager._precommit_ci._utcnow", return_value=NOW
        ):
            assert await mgr._precommit_nudge_cooling_down("o", "r", pr) is True
        client.post_issue_comment.assert_not_called()
        if reading == "fails":
            client.list_issue_comments = AsyncMock(side_effect=RuntimeError("boom"))
        else:
            client.list_issue_comments = AsyncMock(
                return_value=IssueComments([], complete=False)
            )

        out = await _reason(mgr, pr)

        assert out.error == (
            "waiting on required status check: pre-commit.ci - pr (pending 2m)"
            '; "pre-commit.ci run" posted 20m ago'
        )


class TestNothingEstablished:
    @pytest.mark.asyncio
    async def test_a_moved_head_is_named_beside_the_old_reason(self) -> None:
        mgr, _ = _manager([], required_ok=False, complete=False)
        pr = _pr()

        out = await _reason(mgr, pr)
        # The end-of-run pass reads a second time; the mark is not doubled.
        out = await mgr._name_the_live_blocker(
            pr, out, "blocked", {"head": {"sha": NEW_HEAD}}
        )

        assert out.error == (
            f"{STALE} (read at {OLD_HEAD[:7]}; the head is now {NEW_HEAD[:7]})"
        )
        assert out.status is MergeStatus.FAILED

    @pytest.mark.asyncio
    async def test_an_unmoved_head_leaves_the_reason_alone(self) -> None:
        mgr, _ = _manager([], required_ok=False, complete=False)
        pr = _pr()
        result = MergeResult(pr_info=pr, status=MergeStatus.FAILED, error=STALE)

        out = await mgr._name_the_live_blocker(
            pr, result, "blocked", {"head": {"sha": OLD_HEAD}}
        )

        assert out.error == STALE


def _run(
    name: str, app: int, status: str, started: timedelta, **extra: Any
) -> dict[str, Any]:
    return {
        "name": name,
        "app": {"id": app},
        "status": status,
        "started_at": _iso(started),
        **extra,
    }


class TestEveryRequiredReportIsRead:
    @pytest.mark.asyncio
    async def test_a_run_from_another_app_does_not_hide_the_bound_one(self) -> None:
        mgr, _ = _manager(
            [{"context": "build", "app_id": 11}],
            runs=[
                _run("build", 11, "in_progress", timedelta(minutes=3)),
                _run(
                    "build",
                    22,
                    "completed",
                    timedelta(minutes=1),
                    conclusion="success",
                ),
            ],
        )

        out = await _reason(mgr, _pr())

        assert out.error == "waiting on required status check: build (pending 3m)"

    @pytest.mark.asyncio
    async def test_a_failed_required_run_beside_a_pending_one_is_the_reason(
        self,
    ) -> None:
        failed = _run(
            "build", 11, "completed", timedelta(minutes=5), conclusion="failure"
        )
        mgr, client = _manager(
            [{"context": "build", "app_id": 11}, "DCO", "lint"],
            statuses=[
                {"context": "DCO", "state": "pending", "updated_at": _iso(timedelta(0))}
            ],
            runs=[
                failed,
                _run("lint", 11, "completed", timedelta(1), conclusion="success"),
            ],
        )
        client.get_check_runs_for_ref = AsyncMock(return_value=[failed])

        out = await _reason(mgr, _pr())

        assert out.error == "blocked by required status check: build"

    @pytest.mark.asyncio
    async def test_a_failed_run_from_another_app_is_not_the_required_one(
        self,
    ) -> None:
        mgr, _ = _manager(
            [{"context": "build", "app_id": 11}],
            runs=[
                _run("build", 11, "in_progress", timedelta(minutes=3)),
                _run(
                    "build",
                    22,
                    "completed",
                    timedelta(minutes=1),
                    conclusion="failure",
                ),
            ],
        )

        out = await _reason(mgr, _pr())

        assert out.error == "waiting on required status check: build (pending 3m)"

    @pytest.mark.asyncio
    async def test_a_bound_app_with_no_run_is_not_reported_yet(self) -> None:
        mgr, _ = _manager(
            [{"context": "build", "app_id": 11}],
            runs=[
                _run(
                    "build",
                    22,
                    "completed",
                    timedelta(minutes=1),
                    conclusion="success",
                )
            ],
        )

        out = await _reason(mgr, _pr())

        assert out.error == (
            "waiting on required status check: build (not reported yet)"
        )

    @pytest.mark.asyncio
    async def test_a_finished_status_does_not_hide_an_unfinished_run(self) -> None:
        mgr, _ = _manager(
            ["build"],
            statuses=[{"context": "build", "state": "success"}],
            runs=[_run("build", 11, "in_progress", timedelta(minutes=2))],
        )

        out = await _reason(mgr, _pr())

        assert out.error == "waiting on required status check: build (pending 2m)"

    @pytest.mark.asyncio
    async def test_a_success_in_another_suite_does_not_hide_a_pending_run(
        self,
    ) -> None:
        mgr, _ = _manager(
            ["build"],
            runs=[
                _run(
                    "build",
                    11,
                    "in_progress",
                    timedelta(minutes=3),
                    id=1,
                    check_suite={"id": 100},
                ),
                _run(
                    "build",
                    11,
                    "completed",
                    timedelta(minutes=1),
                    conclusion="success",
                    completed_at=_iso(timedelta(0, 30)),
                    id=2,
                    check_suite={"id": 200},
                ),
            ],
        )

        out = await _reason(mgr, _pr())

        assert out.error == "waiting on required status check: build (pending 3m)"

    @pytest.mark.asyncio
    async def test_a_superseded_run_finishing_late_does_not_hide_its_rerun(
        self,
    ) -> None:
        mgr, _ = _manager(
            ["build"],
            runs=[
                _run(
                    "build",
                    11,
                    "completed",
                    timedelta(minutes=5),
                    conclusion="success",
                    completed_at=_iso(timedelta(0, 30)),
                    id=1,
                    check_suite={"id": 100},
                ),
                _run(
                    "build",
                    11,
                    "in_progress",
                    timedelta(minutes=2),
                    id=2,
                    check_suite={"id": 100},
                ),
            ],
        )

        out = await _reason(mgr, _pr())

        assert out.error == "waiting on required status check: build (pending 2m)"

    @pytest.mark.asyncio
    async def test_a_finished_status_is_not_called_unreported(self) -> None:
        mgr, _ = _manager(["DCO"], statuses=[{"context": "DCO", "state": "success"}])

        out = await _reason(mgr, _pr())

        assert out.error == (
            f"{STALE} (read at {OLD_HEAD[:7]}; the head is now {NEW_HEAD[:7]})"
        )

    @pytest.mark.asyncio
    async def test_a_pending_run_from_a_partial_listing_is_not_believed(
        self,
    ) -> None:
        mgr, _ = _manager(
            ["build"],
            runs=[_run("build", 11, "in_progress", timedelta(minutes=9))],
            runs_complete=False,
        )

        out = await _reason(mgr, _pr())

        assert out.error == (
            f"{STALE} (read at {OLD_HEAD[:7]}; the head is now {NEW_HEAD[:7]})"
        )

    @pytest.mark.parametrize("state", ["pending", "failure"])
    @pytest.mark.asyncio
    async def test_a_partial_listing_does_not_let_a_status_stand_for_a_bound_app(
        self, state: str
    ) -> None:
        mgr, _ = _manager(
            [{"context": "build", "app_id": 11}],
            statuses=[
                {"context": "build", "state": state, "updated_at": _iso(timedelta(0))}
            ],
            runs=[
                _run("build", 22, "completed", timedelta(1), conclusion="success"),
            ],
            runs_complete=False,
        )

        out = await _reason(mgr, _pr())

        assert out.error == (
            f"{STALE} (read at {OLD_HEAD[:7]}; the head is now {NEW_HEAD[:7]})"
        )

    @pytest.mark.parametrize("runs_complete", [True, False], ids=["passed", "partial"])
    @pytest.mark.asyncio
    async def test_a_failed_status_is_not_promoted_for_a_bound_app(
        self, runs_complete: bool
    ) -> None:
        mgr, _ = _manager(
            [{"context": "build", "app_id": 11}, "DCO"],
            statuses=[
                {"context": "build", "state": "failure"},
                {
                    "context": "DCO",
                    "state": "pending",
                    "updated_at": _iso(timedelta(0)),
                },
            ],
            runs=[_run("build", 11, "completed", timedelta(1), conclusion="success")],
            failing_contexts=["build"],
            runs_complete=runs_complete,
        )

        out = await _reason(mgr, _pr())

        assert out.error == (
            "waiting on required status check: DCO (pending 0s); also failing: build"
        )

    @pytest.mark.asyncio
    async def test_a_failed_status_stands_for_a_bound_app_with_no_run(self) -> None:
        mgr, _ = _manager(
            [{"context": "build", "app_id": 11}, "DCO"],
            statuses=[
                {"context": "build", "state": "failure"},
                {
                    "context": "DCO",
                    "state": "pending",
                    "updated_at": _iso(timedelta(0)),
                },
            ],
            failing_contexts=["build"],
        )

        out = await _reason(mgr, _pr())

        assert out.error == "blocked by required status check: build"

    @pytest.mark.asyncio
    async def test_a_quoted_workflow_stays_the_blocker_beside_a_bound_status(
        self,
    ) -> None:
        mgr, client = _manager(
            [{"context": "build", "app_id": 11}, "DCO"],
            statuses=[
                {"context": "build", "state": "failure"},
                {
                    "context": "DCO",
                    "state": "pending",
                    "updated_at": _iso(timedelta(0)),
                },
            ],
            runs=[_run("build", 11, "completed", timedelta(1), conclusion="success")],
            failing_contexts=["build"],
        )
        client.get_failing_workflow_run_names_for_sha = AsyncMock(return_value=["CI"])
        pr = _pr()
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            error="Repository rule violations found Required workflows 'CI' failed",
        )
        refreshed = {"head": {"sha": NEW_HEAD}, "base": {"ref": "main"}}

        out = await mgr._name_the_live_blocker(pr, result, "blocked", refreshed)

        assert out.error == "blocked by required workflow: CI; also failing: build"

    @pytest.mark.asyncio
    async def test_a_second_confirmation_keeps_the_quoted_workflow(self) -> None:
        mgr, client = _manager(
            ["DCO"],
            statuses=[
                {
                    "context": "DCO",
                    "state": "pending",
                    "updated_at": _iso(timedelta(0)),
                }
            ],
        )
        client.get_failing_workflow_run_names_for_sha = AsyncMock(return_value=["CI"])
        pr = _pr()
        result = MergeResult(
            pr_info=pr,
            status=MergeStatus.FAILED,
            error="Repository rule violations found Required workflows 'CI' failed",
        )
        refreshed = {"head": {"sha": NEW_HEAD}, "base": {"ref": "main"}}

        out = await mgr._name_the_live_blocker(pr, result, "blocked", refreshed)
        # The end-of-run pass confirms the failure again, unchanged checks.
        out = await mgr._name_the_live_blocker(pr, out, "blocked", refreshed)

        assert out.error == "blocked by required workflow: CI"
        assert "Required workflows 'CI'" in (out.warning or "")


class TestTheMarkNamesTheReasonsHead:
    @pytest.mark.asyncio
    async def test_a_live_reason_is_marked_with_the_head_it_was_read_at(
        self,
    ) -> None:
        mgr, client = _manager(["DCO"])
        pr = _pr()
        out = await _reason(mgr, pr)
        live = out.error
        client.get_required_status_checks_reliable.return_value = ([], False)
        newer = "1111111222222233333334444444555555566666"

        out = await mgr._name_the_live_blocker(
            pr, out, "blocked", {"head": {"sha": newer}}
        )

        assert out.error == (
            f"{live} (read at {NEW_HEAD[:7]}; the head is now {newer[:7]})"
        )

    @pytest.mark.asyncio
    async def test_a_later_head_replaces_the_mark(self) -> None:
        mgr, _ = _manager([], required_ok=False, complete=False)
        pr = _pr()
        out = await _reason(mgr, pr)
        pr.head_sha = NEW_HEAD
        newer = "1111111222222233333334444444555555566666"

        out = await mgr._name_the_live_blocker(
            pr, out, "blocked", {"head": {"sha": newer}}
        )

        assert out.error == (
            f"{STALE} (read at {OLD_HEAD[:7]}; the head is now {newer[:7]})"
        )
