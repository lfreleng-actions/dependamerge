# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Unit tests for ``AsyncMergeManager._detect_stuck_required_check``.

Covers:
- DCO check stuck longer than the threshold on a sufficiently old
  PR -> detected (returns ``(True, name, age)``).
- Sub-threshold stuck DCO check -> not detected.
- Stuck pending check whose name is not DCO-shaped and not required
  -> not detected.
- A stuck *required* non-DCO check (e.g. ``build``) -> detected.
- A stuck *required* pre-commit.ci check -> not detected (it has its
  own recovery via ``_trigger_stale_precommit_ci``).
- DCO check stuck but PR was just touched (updated_at younger than
  threshold) -> not detected (age floor protects against false
  positives on freshly-pushed PRs).
- DCO check completed -> not detected.
- Stuck status-context (older API) DCO entry -> detected.
- API failures degrade to ``(False, None, 0.0)``.
- ``_merge_single_pr`` triggers ``_trigger_dependabot_recreate`` when
  ``_detect_stuck_required_check`` returns True for a dependabot PR.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dependamerge.merge_manager import (
    STUCK_CHECK_THRESHOLD_SECONDS,
    MergeResult,
    MergeStatus,
)
from dependamerge.models import PullRequestInfo


def _iso(dt: datetime) -> str:
    """Return ``dt`` in the RFC 3339 ``...Z`` form GitHub emits."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_pr_info(**overrides: Any) -> PullRequestInfo:
    """Helper to build a PullRequestInfo with sensible defaults."""
    defaults: dict[str, Any] = {
        "number": 342,
        "title": "Chore: Bump foo from 1 to 2",
        "body": "Dependabot PR",
        "author": "dependabot[bot]",
        "head_sha": "abc123",
        "base_branch": "main",
        "head_branch": "dependabot/foo",
        "state": "open",
        "mergeable": True,
        "mergeable_state": "blocked",
        "behind_by": 0,
        "files_changed": [],
        "repository_full_name": "lfreleng-actions/lftools-uv",
        "html_url": "https://github.com/lfreleng-actions/lftools-uv/pull/342",
    }
    defaults.update(overrides)
    return PullRequestInfo(**defaults)


def _make_manager(**overrides: Any):
    """Build an AsyncMergeManager with a mocked GitHub client.

    Returns ``(manager, client)`` — see ``tests/conftest.py`` for the
    typed-mock-client pattern.
    """
    from tests.conftest import make_merge_manager

    defaults: dict[str, Any] = {"preview_mode": False}
    defaults.update(overrides)
    mgr, client = make_merge_manager(**defaults)

    # The detector reads both listings through the client's paginated
    # readers; serve them from whatever ``client.get`` routes, looked up
    # at call time so each test can still assign ``client.get``.
    async def _listing(path: str, key: str) -> list[dict[str, Any]]:
        data = await client.get(path)
        return list(data.get(key) or []) if isinstance(data, dict) else []

    async def _runs(owner: str, repo: str, ref: str) -> list[dict[str, Any]]:
        return await _listing(
            f"/repos/{owner}/{repo}/commits/{ref}/check-runs", "check_runs"
        )

    async def _statuses(owner: str, repo: str, ref: str) -> list[dict[str, Any]]:
        return await _listing(f"/repos/{owner}/{repo}/commits/{ref}/status", "statuses")

    client.get_check_runs_for_ref = AsyncMock(side_effect=_runs)
    client.get_commit_statuses = AsyncMock(side_effect=_statuses)
    return mgr, client


def _build_get_responder(
    pr_response: dict[str, Any] | None = None,
    check_runs_response: dict[str, Any] | None = None,
    status_response: dict[str, Any] | None = None,
):
    """Return an ``AsyncMock``-compatible side effect dispatcher.

    Routes ``GET`` requests to the right canned response based on the
    URL path — this mirrors how ``_detect_stuck_required_check`` actually calls
    the client.
    """

    async def _get(path: str, *args: Any, **kwargs: Any) -> Any:
        if path.endswith("/check-runs"):
            return check_runs_response
        if path.endswith("/status"):
            return status_response
        # Anything else is the PR detail fetch.
        return pr_response

    return _get


# ---------------------------------------------------------------------------
# _detect_stuck_required_check — positive and negative cases
# ---------------------------------------------------------------------------


class TestDetectStuckDcoCheckRuns:
    """Detection via the modern ``check-runs`` API."""

    @pytest.mark.asyncio
    async def test_detects_stuck_dco_check_run(self) -> None:
        """A queued/in_progress DCO check older than the threshold
        on an old-enough PR is detected."""
        mgr, client = _make_manager()
        now = datetime.now(UTC)
        old = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        pr = _make_pr_info()

        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    "created_at": _iso(old),
                    "updated_at": _iso(old),
                },
                check_runs_response={
                    "check_runs": [
                        {
                            "name": "DCO",
                            "status": "in_progress",
                            "conclusion": None,
                            "started_at": _iso(old),
                        }
                    ],
                },
                status_response={"statuses": []},
            )
        )

        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is True
        assert name == "DCO"
        assert age >= STUCK_CHECK_THRESHOLD_SECONDS

    @pytest.mark.asyncio
    async def test_sub_threshold_age_is_not_stuck(self) -> None:
        """A DCO check pending for less than the threshold is not stuck."""
        mgr, client = _make_manager()
        now = datetime.now(UTC)
        # PR is old enough; the check itself is too young.
        pr_age = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        check_started = now - timedelta(seconds=10)
        pr = _make_pr_info()

        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    "created_at": _iso(pr_age),
                    "updated_at": _iso(check_started),
                },
                check_runs_response={
                    "check_runs": [
                        {
                            "name": "DCO",
                            "status": "in_progress",
                            "started_at": _iso(check_started),
                        }
                    ],
                },
                status_response={"statuses": []},
            )
        )

        # The PR-level idle floor (updated_at) catches this case
        # before we even look at check timing.
        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is False
        assert name is None
        assert age == 0.0

    @pytest.mark.asyncio
    async def test_completed_dco_is_not_stuck(self) -> None:
        """A completed DCO check (success or failure) is not stuck."""
        mgr, client = _make_manager()
        now = datetime.now(UTC)
        old = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        pr = _make_pr_info()

        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    "created_at": _iso(old),
                    "updated_at": _iso(old),
                },
                check_runs_response={
                    "check_runs": [
                        {
                            "name": "DCO",
                            "status": "completed",
                            "conclusion": "success",
                            "started_at": _iso(old),
                        }
                    ],
                },
                status_response={"statuses": []},
            )
        )

        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is False
        assert name is None
        # Non-stuck outcomes always report a zero age (no candidate check).
        assert age == 0.0

    @pytest.mark.asyncio
    async def test_stuck_non_dco_non_required_check_is_ignored(self) -> None:
        """A stuck check that is neither DCO-shaped nor required on
        the base branch is not detected."""
        mgr, client = _make_manager()
        now = datetime.now(UTC)
        old = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        pr = _make_pr_info()

        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    "created_at": _iso(old),
                    "updated_at": _iso(old),
                },
                check_runs_response={
                    "check_runs": [
                        {
                            "name": "build",
                            "status": "in_progress",
                            "started_at": _iso(old),
                        }
                    ],
                },
                status_response={"statuses": []},
            )
        )

        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is False
        assert name is None
        # Non-stuck outcomes always report a zero age (no candidate check).
        assert age == 0.0

    @pytest.mark.asyncio
    async def test_detects_stuck_required_non_dco_check(self) -> None:
        """A stuck *required* non-DCO check (e.g. ``build``) is detected.

        Generalises the original DCO-only behaviour: any required
        verification check that stalls indefinitely should drive the
        dependabot recreate recovery.
        """
        mgr, client = _make_manager()
        now = datetime.now(UTC)
        old = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        pr = _make_pr_info()

        # ``build`` is required on the base branch.
        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "build"}]
        )
        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    "created_at": _iso(old),
                    "updated_at": _iso(old),
                },
                check_runs_response={
                    "check_runs": [
                        {
                            "name": "build",
                            "status": "in_progress",
                            "started_at": _iso(old),
                        }
                    ],
                },
                status_response={"statuses": []},
            )
        )

        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is True
        assert name == "build"
        assert age >= STUCK_CHECK_THRESHOLD_SECONDS

    @pytest.mark.asyncio
    async def test_stuck_required_precommit_ci_is_ignored(self) -> None:
        """A stuck *required* pre-commit.ci check is not detected here.

        pre-commit.ci has its own recovery path
        (``_trigger_stale_precommit_ci`` posts ``pre-commit.ci run``);
        dependabot's ``recreate`` macro does not retrigger it, so it
        must be excluded from the recreate-driving detector even when
        it is a required, stuck check.
        """
        mgr, client = _make_manager()
        now = datetime.now(UTC)
        old = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    "created_at": _iso(old),
                    "updated_at": _iso(old),
                },
                check_runs_response={
                    "check_runs": [
                        {
                            "name": "pre-commit.ci - pr",
                            "status": "in_progress",
                            "started_at": _iso(old),
                        }
                    ],
                },
                status_response={"statuses": []},
            )
        )

        is_stuck, name, _age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is False
        assert name is None

    @pytest.mark.asyncio
    async def test_pr_age_floor_blocks_detection(self) -> None:
        """A young PR (created < threshold ago) is never flagged."""
        mgr, client = _make_manager()
        now = datetime.now(UTC)
        young = now - timedelta(seconds=10)
        pr = _make_pr_info()

        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    "created_at": _iso(young),
                    "updated_at": _iso(young),
                },
                check_runs_response={
                    "check_runs": [
                        {
                            "name": "DCO",
                            "status": "in_progress",
                            "started_at": _iso(young),
                        }
                    ],
                },
                status_response={"statuses": []},
            )
        )

        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is False
        assert name is None
        # Non-stuck outcomes always report a zero age (floor blocks detection).
        assert age == 0.0

    @pytest.mark.asyncio
    async def test_naive_pr_timestamp_does_not_crash(self) -> None:
        """A tz-naive PR timestamp fails closed instead of crashing.

        ``parse_ts`` would otherwise return a naive datetime that
        raises ``TypeError`` when subtracted from the tz-aware ``now``;
        it must degrade to ``None`` so the detector returns
        ``(False, None, 0.0)`` rather than aborting the merge run.
        """
        mgr, client = _make_manager()
        now = datetime.now(UTC)
        old = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        pr = _make_pr_info()

        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    # No trailing "Z"/offset -> naive datetimes.
                    "created_at": "2026-06-08T16:00:00",
                    "updated_at": "2026-06-08T16:00:00",
                },
                check_runs_response={
                    "check_runs": [
                        {
                            "name": "DCO",
                            "status": "in_progress",
                            "started_at": _iso(old),
                        }
                    ],
                },
                status_response={"statuses": []},
            )
        )

        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is False
        assert name is None
        assert age == 0.0

    @pytest.mark.asyncio
    async def test_dco_slash_dco_name_variant_matches(self) -> None:
        """The ``dco/dco`` status-context naming is matched."""
        mgr, client = _make_manager()
        now = datetime.now(UTC)
        old = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        pr = _make_pr_info()

        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    "created_at": _iso(old),
                    "updated_at": _iso(old),
                },
                check_runs_response={
                    "check_runs": [
                        {
                            "name": "dco/dco",
                            "status": "queued",
                            "started_at": _iso(old),
                        }
                    ],
                },
                status_response={"statuses": []},
            )
        )

        is_stuck, name, _age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is True
        assert name == "dco/dco"


class TestDetectStuckDcoStatusContexts:
    """Detection via the older ``commits/{sha}/status`` API."""

    @pytest.mark.asyncio
    async def test_detects_stuck_dco_status_context(self) -> None:
        mgr, client = _make_manager()
        now = datetime.now(UTC)
        old = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        pr = _make_pr_info()

        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    "created_at": _iso(old),
                    "updated_at": _iso(old),
                },
                check_runs_response={"check_runs": []},
                status_response={
                    "statuses": [
                        {
                            "context": "DCO",
                            "state": "pending",
                            "updated_at": _iso(old),
                        }
                    ],
                },
            )
        )

        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is True
        assert name == "DCO"
        assert age >= STUCK_CHECK_THRESHOLD_SECONDS


class TestDetectStuckBeyondTheFirstPage:
    """A stuck required check is found wherever it sits in the listing."""

    @staticmethod
    async def _detect(
        check_runs: list[dict[str, Any]],
        statuses: list[dict[str, Any]],
        required: list[dict[str, Any]] | None = None,
        protection: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None, float]:
        """Run the detector; *protection* routes it through the real collector.

        With *protection*, the required checks come from that classic
        branch protection response and a ruleset requiring *required*.
        """
        from dependamerge.github_async import GitHubAsync
        from tests.conftest import FakeCommitListings

        mgr, _client = _make_manager()
        old = datetime.now(UTC) - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        other: dict[str, Any] = {
            "/pulls/342": {"created_at": _iso(old), "updated_at": _iso(old)}
        }
        if protection is not None:
            other |= {
                "/repos/lfreleng-actions/lftools-uv": {"default_branch": "main"},
                "/rulesets?per_page=100": [{"id": 7}] if required else [],
                "/rulesets/7": {
                    "id": 7,
                    "conditions": {},
                    "rules": [
                        {
                            "type": "required_status_checks",
                            "parameters": {"required_status_checks": required},
                        }
                    ],
                },
                "/protection/required_status_checks": protection,
            }
        fake = FakeCommitListings(check_runs, statuses, other=other)
        async with GitHubAsync(token="t") as api:
            fake.install(api)
            if protection is None:
                api.get_required_status_checks = AsyncMock(  # type: ignore[method-assign]
                    return_value=required or [{"context": "build"}]
                )
            mgr._github_client = api
            result: tuple[bool, str | None, float]
            result = await mgr._detect_stuck_required_check(_make_pr_info())
            return result

    @staticmethod
    def _fillers(count: int) -> list[dict[str, Any]]:
        return [
            {"name": f"job-{i}", "status": "completed", "conclusion": "success"}
            for i in range(count)
        ]

    @pytest.mark.asyncio
    async def test_stuck_required_run_on_second_page(self) -> None:
        old = datetime.now(UTC) - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        runs = self._fillers(120) + [
            {"name": "build", "status": "in_progress", "started_at": _iso(old)}
        ]

        is_stuck, name, _age = await self._detect(runs, [])

        assert (is_stuck, name) == (True, "build")

    @pytest.mark.asyncio
    async def test_stuck_required_status_on_second_page(self) -> None:
        old = datetime.now(UTC) - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        statuses = [{"context": f"ctx-{i}", "state": "success"} for i in range(120)] + [
            {"context": "DCO", "state": "pending", "updated_at": _iso(old)}
        ]

        is_stuck, name, _age = await self._detect([], statuses)

        assert (is_stuck, name) == (True, "DCO")

    @staticmethod
    def _one_suite(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Put every run in one check suite of one App, so order decides."""
        return [{"app": {"id": 1}, **run, "check_suite": {"id": 9}} for run in runs]

    @staticmethod
    def _rerun_pair(
        pending_suite: int | None, later_suite: int | None
    ) -> list[dict[str, Any]]:
        """A stale pending run and a later success of the same name."""
        now = datetime.now(UTC)
        stale = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 300)
        finished = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 100)
        later: dict[str, Any] = {
            "id": 1002,
            "name": "build",
            "status": "completed",
            "conclusion": "success",
            "started_at": _iso(finished),
            "completed_at": _iso(finished),
        }
        pending: dict[str, Any] = {
            "id": 1001,
            "name": "build",
            "status": "in_progress",
            "started_at": _iso(stale),
        }
        if later_suite is not None:
            later["check_suite"] = {"id": later_suite}
        if pending_suite is not None:
            pending["check_suite"] = {"id": pending_suite}
        return [later, *TestDetectStuckBeyondTheFirstPage._fillers(120), pending]

    @pytest.mark.asyncio
    async def test_superseded_pending_run_is_not_stuck(self) -> None:
        runs = self._rerun_pair(pending_suite=9, later_suite=9)

        assert await self._detect(runs, []) == (False, None, 0.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("pending_suite", "later_suite"),
        [(9, 10), (None, 9), (None, None)],
        ids=["other-suite", "pending-suite-unknown", "suites-unknown"],
    )
    async def test_run_from_another_or_unknown_suite_does_not_supersede(
        self, pending_suite: int | None, later_suite: int | None
    ) -> None:
        # A later workflow_dispatch run of the same job, in its own check
        # suite, cannot satisfy the pull request's requirement.
        runs = self._rerun_pair(pending_suite, later_suite)

        is_stuck, name, _age = await self._detect(runs, [])

        assert (is_stuck, name) == (True, "build")

    @pytest.mark.asyncio
    async def test_newer_run_that_started_first_is_still_stuck(self) -> None:
        # An older run left queued can start after a newer one; when start
        # time and id disagree, neither run supersedes the other.
        now = datetime.now(UTC)
        stale = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 300)
        finished = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 100)
        runs = [
            {
                "id": 1001,
                "name": "build",
                "status": "completed",
                "conclusion": "success",
                "started_at": _iso(finished),
                "completed_at": _iso(finished),
            },
            *self._fillers(120),
            {
                "id": 1002,
                "name": "build",
                "status": "in_progress",
                "started_at": _iso(stale),
            },
        ]

        is_stuck, name, _age = await self._detect(self._one_suite(runs), [])

        assert (is_stuck, name) == (True, "build")

    @pytest.mark.asyncio
    async def test_newer_pending_rerun_is_still_stuck(self) -> None:
        now = datetime.now(UTC)
        finished = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 300)
        rerun = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 100)
        runs = [
            {"name": "build", "status": "in_progress", "started_at": _iso(rerun)},
            *self._fillers(120),
            {
                "name": "build",
                "status": "completed",
                "conclusion": "success",
                "started_at": _iso(finished),
                "completed_at": _iso(finished),
            },
        ]

        is_stuck, name, _age = await self._detect(self._one_suite(runs), [])

        assert (is_stuck, name) == (True, "build")

    @pytest.mark.asyncio
    async def test_newer_run_from_another_app_does_not_supersede(self) -> None:
        now = datetime.now(UTC)
        stale = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 300)
        later = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 100)
        runs = [
            {
                "name": "build",
                "app": {"id": 1},
                "status": "in_progress",
                "started_at": _iso(stale),
            },
            *self._fillers(120),
            {
                "name": "build",
                "app": {"id": 2},
                "status": "completed",
                "conclusion": "success",
                "started_at": _iso(later),
                "completed_at": _iso(later),
            },
        ]

        is_stuck, name, _age = await self._detect(self._one_suite(runs), [])

        assert (is_stuck, name) == (True, "build")

    @pytest.mark.asyncio
    async def test_overlapping_older_run_does_not_hide_rerun(self) -> None:
        # The older run finishes after the rerun starts; ordering by
        # completion would pick the older success.
        now = datetime.now(UTC)
        first = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 300)
        rerun = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 200)
        finished = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 100)
        runs = [
            {
                "name": "build",
                "status": "completed",
                "conclusion": "success",
                "started_at": _iso(first),
                "completed_at": _iso(finished),
            },
            *self._fillers(120),
            {"name": "build", "status": "in_progress", "started_at": _iso(rerun)},
        ]

        is_stuck, name, _age = await self._detect(self._one_suite(runs), [])

        assert (is_stuck, name) == (True, "build")

    @pytest.mark.asyncio
    async def test_unordered_pending_run_is_still_judged(self) -> None:
        runs = [
            {
                "id": "a",
                "name": "build",
                "status": "completed",
                "conclusion": "success",
            },
            {"id": "b", "name": "build", "status": "queued"},
        ]

        is_stuck, name, _age = await self._detect(self._one_suite(runs), [])

        assert (is_stuck, name) == (True, "build")

    @pytest.mark.asyncio
    async def test_mixed_start_times_do_not_order_cyclically(self) -> None:
        # Comparing ids when one start time is missing made each run
        # supersede another, so all three were set aside.
        now = datetime.now(UTC)
        runs = [
            {
                "id": 1,
                "name": "build",
                "status": "completed",
                "conclusion": "success",
                "started_at": _iso(now - timedelta(days=1)),
            },
            {"id": 2, "name": "build", "status": "queued"},
            {
                "id": 3,
                "name": "build",
                "status": "completed",
                "conclusion": "success",
                "started_at": _iso(now - timedelta(days=2)),
            },
        ]

        is_stuck, name, _age = await self._detect(self._one_suite(runs), [])

        assert (is_stuck, name) == (True, "build")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "required",
        [
            [{"context": "build", "integration_id": 1}],
            [{"context": "build", "app_id": 1}],
        ],
    )
    async def test_pending_run_from_unrequired_app_is_not_stuck(
        self, required: list[dict[str, Any]]
    ) -> None:
        now = datetime.now(UTC)
        stale = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 300)
        later = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 100)
        runs = [
            {
                "name": "build",
                "app": {"id": 1},
                "status": "completed",
                "conclusion": "success",
                "started_at": _iso(later),
            },
            *self._fillers(120),
            {
                "name": "build",
                "app": {"id": 2},
                "status": "in_progress",
                "started_at": _iso(stale),
            },
        ]

        assert await self._detect(runs, [], required) == (False, None, 0.0)

    @pytest.mark.asyncio
    async def test_pending_run_from_required_app_is_stuck(self) -> None:
        now = datetime.now(UTC)
        stale = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 300)
        later = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 100)
        runs = [
            {
                "name": "build",
                "app": {"id": 1},
                "status": "in_progress",
                "started_at": _iso(stale),
            },
            *self._fillers(120),
            {
                "name": "build",
                "app": {"id": 2},
                "status": "completed",
                "conclusion": "success",
                "started_at": _iso(later),
            },
        ]

        is_stuck, name, _age = await self._detect(
            runs, [], [{"context": "build", "integration_id": 1}]
        )

        assert (is_stuck, name) == (True, "build")

    @staticmethod
    def _cross_app_runs(stuck_app: int, passed_app: int) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        stale = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 300)
        later = now - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 100)
        return [
            {
                "name": "build",
                "app": {"id": passed_app},
                "status": "completed",
                "conclusion": "success",
                "started_at": _iso(later),
            },
            {
                "name": "build",
                "app": {"id": stuck_app},
                "status": "in_progress",
                "started_at": _iso(stale),
            },
        ]

    @pytest.mark.asyncio
    async def test_classic_protection_app_binding_reaches_detector(self) -> None:
        protection = {
            "contexts": ["build"],
            "checks": [{"context": "build", "app_id": 1}],
        }

        result = await self._detect(
            self._cross_app_runs(stuck_app=2, passed_app=1), [], [], protection
        )

        assert result == (False, None, 0.0)

    @pytest.mark.asyncio
    async def test_conflicting_app_bindings_judge_every_app(self) -> None:
        protection = {
            "contexts": ["build"],
            "checks": [{"context": "build", "app_id": 2}],
        }

        is_stuck, name, _age = await self._detect(
            self._cross_app_runs(stuck_app=2, passed_app=1),
            [],
            [{"context": "build", "integration_id": 1}],
            protection,
        )

        assert (is_stuck, name) == (True, "build")

    @pytest.mark.asyncio
    async def test_conflicting_app_bindings_exclude_other_apps(self) -> None:
        # A ruleset binding App 1 and protection binding App 2 require
        # neither App 3's run nor any other App's.
        protection = {
            "contexts": ["build"],
            "checks": [{"context": "build", "app_id": 2}],
        }

        result = await self._detect(
            self._cross_app_runs(stuck_app=3, passed_app=1),
            [],
            [{"context": "build", "integration_id": 1}],
            protection,
        )

        assert result == (False, None, 0.0)

    @staticmethod
    def _pending_build_status() -> list[dict[str, Any]]:
        old = datetime.now(UTC) - timedelta(seconds=STUCK_CHECK_THRESHOLD_SECONDS + 30)
        return [{"context": "build", "state": "pending", "updated_at": _iso(old)}]

    @pytest.mark.asyncio
    async def test_status_for_app_bound_check_is_judged(self) -> None:
        # The required App posted no check run, so it reports through
        # the status, which can be the only stuck signal.
        is_stuck, name, _age = await self._detect(
            [],
            self._pending_build_status(),
            [{"context": "build", "integration_id": 1}],
        )

        assert (is_stuck, name) == (True, "build")

    @pytest.mark.asyncio
    async def test_status_beside_required_app_run_is_not_judged(self) -> None:
        # The required App reports through its run; the pending status of
        # the same name comes from another source.
        runs = [
            {
                "name": "build",
                "app": {"id": 1},
                "status": "completed",
                "conclusion": "success",
            }
        ]

        result = await self._detect(
            runs,
            self._pending_build_status(),
            [{"context": "build", "integration_id": 1}],
        )

        assert result == (False, None, 0.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("run_apps", "expected"),
        [((1,), (True, "build")), ((1, 2), (False, None))],
        ids=["one-bound-app-ran", "every-bound-app-ran"],
    )
    async def test_status_beside_runs_of_several_bound_apps(
        self, run_apps: tuple[int, ...], expected: tuple[bool, str | None]
    ) -> None:
        # A ruleset binds build to App 1 and protection to App 2.  App 1's
        # run does not satisfy App 2, which may report through the status.
        protection = {
            "contexts": ["build"],
            "checks": [{"context": "build", "app_id": 2}],
        }
        runs = [
            {
                "name": "build",
                "app": {"id": app},
                "status": "completed",
                "conclusion": "success",
            }
            for app in run_apps
        ]

        is_stuck, name, _age = await self._detect(
            runs,
            self._pending_build_status(),
            [{"context": "build", "integration_id": 1}],
            protection,
        )

        assert (is_stuck, name) == expected


class TestDetectStuckDcoRobustness:
    """Defensive behaviour when the GitHub API misbehaves."""

    @pytest.mark.asyncio
    async def test_pr_fetch_failure_returns_safe_default(self) -> None:
        mgr, client = _make_manager()
        client.get = AsyncMock(side_effect=RuntimeError("boom"))
        pr = _make_pr_info()

        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is False
        assert name is None
        assert age == 0.0

    @pytest.mark.asyncio
    async def test_unparsable_timestamps_return_safe_default(self) -> None:
        mgr, client = _make_manager()
        pr = _make_pr_info()

        client.get = AsyncMock(
            side_effect=_build_get_responder(
                pr_response={
                    "created_at": "not-a-real-date",
                    "updated_at": None,
                },
                check_runs_response={"check_runs": []},
                status_response={"statuses": []},
            )
        )

        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is False
        assert name is None
        assert age == 0.0

    @pytest.mark.asyncio
    async def test_no_github_client_returns_safe_default(self) -> None:
        mgr, _client = _make_manager()
        # Simulate the manager being used outside the async context.
        mgr._github_client = None
        pr = _make_pr_info()

        is_stuck, name, age = await mgr._detect_stuck_required_check(pr)
        assert is_stuck is False
        assert name is None
        assert age == 0.0


def _printed(console_mock) -> str:
    """Join the positional text of every ``console.print`` call."""
    return " ".join(
        str(call.args[0]) for call in console_mock.print.call_args_list if call.args
    )


class TestReportMergeFailure:
    """Non-dependabot stuck-check reporting in ``_report_merge_failure``."""

    @pytest.mark.asyncio
    async def test_stuck_check_reports_and_arms_auto_merge(self) -> None:
        """A stuck check sets the error reason and arms auto-merge (non-dirty)."""
        mgr, _client = _make_manager()
        pr = _make_pr_info(author="someuser", mergeable_state="blocked")
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        with (
            patch.object(
                mgr,
                "_detect_stuck_required_check",
                new_callable=AsyncMock,
                return_value=(True, "DCO", 120.0),
            ),
            patch.object(
                mgr,
                "_enable_auto_merge_with_approval",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_enable,
            patch.object(mgr, "_console") as mock_console,
        ):
            out = await mgr._report_merge_failure(pr, "o", "r", result, "blocked")

        assert out.status == MergeStatus.FAILED
        assert out.error == "stuck check: DCO"
        mock_enable.assert_awaited_once()
        # Real merge runs keep the console clean: per-PR status goes
        # to the log only, and the stuck-check reason reaches the
        # user via ``result.error`` in the end-of-run summary.
        printed = _printed(mock_console)
        assert "⚠️ Stuck check" not in printed
        assert "❌ Failed" not in printed

    @pytest.mark.asyncio
    async def test_stuck_check_when_dirty_does_not_arm_auto_merge(self) -> None:
        """A dirty PR cannot take auto-merge, so it is not armed."""
        mgr, _client = _make_manager()
        pr = _make_pr_info(author="someuser", mergeable_state="dirty")
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        with (
            patch.object(
                mgr,
                "_detect_stuck_required_check",
                new_callable=AsyncMock,
                return_value=(True, "DCO", 120.0),
            ),
            patch.object(
                mgr, "_enable_auto_merge_for_pr", new_callable=AsyncMock
            ) as mock_enable,
            patch.object(mgr, "_console"),
        ):
            out = await mgr._report_merge_failure(
                pr, "o", "r", result, "merge conflicts"
            )

        assert out.status == MergeStatus.FAILED
        mock_enable.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_stuck_check_emits_generic_failure(self) -> None:
        mgr, _client = _make_manager()
        pr = _make_pr_info(author="someuser")
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        with (
            patch.object(
                mgr,
                "_detect_stuck_required_check",
                new_callable=AsyncMock,
                return_value=(False, None, 0.0),
            ),
            patch.object(mgr, "_console") as mock_console,
        ):
            out = await mgr._report_merge_failure(
                pr, "o", "r", result, "branch protection rules prevent merge"
            )

        assert out.status == MergeStatus.FAILED
        # The result error now surfaces the real reason so the
        # end-of-run summary is informative (was a generic message).
        assert out.error == "branch protection rules prevent merge"
        # Real merge runs no longer print per-PR failure lines; the
        # reason is reported once in the end-of-run summary via
        # ``result.error``.
        printed = _printed(mock_console)
        assert "❌ Failed" not in printed
        assert "branch protection" not in printed

    @pytest.mark.asyncio
    async def test_dependabot_skips_stuck_detection(self) -> None:
        """For dependabot the recreate path handles stuck checks, not here."""
        mgr, _client = _make_manager()
        pr = _make_pr_info(author="dependabot[bot]")
        result = MergeResult(pr_info=pr, status=MergeStatus.PENDING)
        with (
            patch.object(
                mgr, "_detect_stuck_required_check", new_callable=AsyncMock
            ) as mock_detect,
            patch.object(mgr, "_console"),
        ):
            out = await mgr._report_merge_failure(pr, "o", "r", result, "blocked")

        assert out.status == MergeStatus.FAILED
        mock_detect.assert_not_called()
