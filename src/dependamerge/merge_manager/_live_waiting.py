# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Naming the required checks a pull request is still waiting on, read live.

:mod:`_live_blockers` names what is *failing*.  A ``blocked`` pull
request can equally be held by a required check that has not finished:
pre-commit.ci left ``queued`` on a head dependabot has just pushed, or a
DCO check that has not reported on it yet.  Neither fails, so a reading
of failures finds nothing and the reason recorded against an older head
would stand, describing a commit nobody is trying to merge.

A required check run that failed is still a failure, and it comes first.
The failing reading proves a status context no App binds, or a quoted
workflow, required, but names a check run, or a status an App binds,
only as failing, so this reading, which knows each requirement's App,
is what proves such a run or status required.  A
required status that failed comes first too: the failing reading may
not have answered, or the status may have failed since it was taken.

Absence is only evidence from a complete reading.  A required context
missing from a status or check-run listing that stopped early may sit
in the part not read, so it is never called "not reported" then.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ..check_names import summarise_check_names
from ..check_runs import FAILING_CONCLUSIONS
from ..github_async._commit_listings import CommitReading
from ..models import PullRequestInfo
from ._live_blockers import _LiveBlockerMixin
from ._precommit_nudge_log import _PrecommitNudgeLogMixin
from ._precommit_status import (
    PRECOMMIT_CONTEXT,
    PRECOMMIT_TRIGGER,
    newest_precommit_trigger,
    parse_timestamp,
)
from ._rebase_request import _format_duration
from ._run_supersession import current_runs, run_source
from ._types import MergeResult

_PENDING_STATUS_STATES = frozenset({"pending"})
_FAILING_STATUS_STATES = frozenset({"failure", "error"})
_HEAD_MARK = re.compile(r" \(read at [^;()]*; the head is now [^;()]*\)$")


def _utcnow() -> datetime:
    """The current time; a module function so tests can substitute it."""
    return datetime.now(UTC)


def _format_age(seconds: float) -> str:
    """Render an age as ``42s`` under a minute, else as ``5m`` or ``1h 3m``."""
    if seconds < 60:
        return f"{max(0, int(seconds))}s"
    return _format_duration(seconds)


def _pending_since(entry: Mapping[str, Any], *keys: str) -> datetime | None:
    """The first of *keys* on *entry* that parses as a timestamp."""
    for key in keys:
        parsed = parse_timestamp(entry.get(key))
        if parsed is not None:
            return parsed
    return None


class _LiveWaitingMixin(_LiveBlockerMixin, _PrecommitNudgeLogMixin):
    """Deriving the unfinished required checks from live state."""

    async def _live_waiting_reason(
        self,
        pr_info: PullRequestInfo,
        *,
        head_sha: str,
        base_branch: str,
        also: list[str] | None = None,
    ) -> str | None:
        """Word the required checks that have not finished on *head_sha*.

        A required check run or status that failed outranks them all: it
        is named as the blocker, as a failing required context is, and
        nothing waiting is mentioned.  Checks in *also*, failing but not shown to
        be required, follow the reason as ``also failing``.

        Returns ``None`` when no failed or unfinished required check can
        be shown: the required contexts could not be read reliably, or
        each one has finished or might have reported in a part of a
        listing that was not read.
        """
        if self._github_client is None:
            return None
        try:
            owner, repo = pr_info.repository_full_name.split("/", 1)
        except ValueError:
            return None
        required, required_ok = await self._required_context_apps(
            owner, repo, base_branch, pr_info
        )
        if not required_ok or not required:
            return None
        statuses = await self._commit_reading(
            self._github_client.read_commit_statuses, owner, repo, head_sha, pr_info
        )
        runs = await self._commit_reading(
            self._github_client.read_check_runs_for_ref, owner, repo, head_sha, pr_info
        )
        others = list(also or [])
        failed = [
            name
            for name in sorted(required)
            if self._failed_check_run(name, required[name], runs)
            or self._failed_status(name, required[name], statuses, runs)
        ]
        if failed:
            others = [name for name in others if name.strip() not in failed]
            reason = "blocked by " + "; ".join(
                f"required status check: {name}" for name in failed
            )
            if others:
                reason += f"; also failing: {summarise_check_names(others)}"
            return reason
        now = _utcnow()
        entries: list[str] = []
        for name in sorted(required):
            waiting = self._unfinished_check(name, required[name], statuses, runs, now)
            if waiting is None:
                continue
            entry = f"required status check: {name} ({waiting})"
            if name == PRECOMMIT_CONTEXT:
                entry += await self._precommit_rerun_note(
                    owner, repo, pr_info, waiting, now
                )
            entries.append(entry)
        if not entries:
            return None
        if others:
            entries.append(f"also failing: {summarise_check_names(others)}")
        return "waiting on " + "; ".join(entries)

    @staticmethod
    def _mark_reason_head(
        pr_info: PullRequestInfo, result: MergeResult, head_sha: str
    ) -> None:
        """Mark a reason read at a head the PR has left with both heads.

        The head a reason was read at is the one a live reading recorded
        beside it, else the snapshot's; it is pinned on first marking, so
        a snapshot moved later cannot relabel it.  A previous mark is
        replaced rather than added to, and dropped when the PR is back
        on the head the reason describes.
        """
        if not result.error:
            return
        read_at = result.reason_head or pr_info.head_sha
        if not (read_at and head_sha):
            return
        result.reason_head = read_at
        reason = _HEAD_MARK.sub("", result.error)
        if read_at != head_sha:
            reason = f"{reason} (read at {read_at[:7]}; the head is now {head_sha[:7]})"
        result.error = reason

    async def _commit_reading(
        self,
        reader: Any,
        owner: str,
        repo: str,
        head_sha: str,
        pr_info: PullRequestInfo,
    ) -> CommitReading:
        """A tolerant commit listing, read as incomplete when it fails."""
        try:
            reading = await reader(owner, repo, head_sha)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.debug(
                "Could not read the checks on %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return CommitReading([], False)
        if not isinstance(reading, CommitReading):
            return CommitReading([], False)
        return reading

    @staticmethod
    def _eligible_runs(
        name: str, apps: frozenset[int] | None, runs: CommitReading
    ) -> list[list[dict[str, Any]]]:
        """The current runs of *name*, one list per App that may satisfy it.

        Every current run counts, as in :mod:`_stuck_checks`: one that no
        later run of its App, check suite and name supersedes.  When
        *apps* binds the check, each bound App's runs are a list of their
        own, empty when that App has posted none; otherwise one list
        holds every run of the name.
        """
        named = [
            run
            for run in current_runs(runs.entries)
            if (run.get("name") or "").strip() == name
        ]
        if apps is None:
            return [named]
        return [
            [run for run in named if run_source(run)[0] == app] for app in sorted(apps)
        ]

    @classmethod
    def _failed_check_run(
        cls, name: str, apps: frozenset[int] | None, runs: CommitReading
    ) -> bool:
        """Whether a current run of required check *name* has failed.

        Only runs from an App the requirement accepts count, and only
        from a complete listing: a partial one can hold a failed run and
        miss the re-run that replaced it.
        """
        if not runs.complete:
            return False
        return any(
            (run.get("status") or "").strip().lower() == "completed"
            and (run.get("conclusion") or "").strip().lower() in FAILING_CONCLUSIONS
            for source in cls._eligible_runs(name, apps, runs)
            for run in source
        )

    @staticmethod
    def _required_status(
        name: str,
        apps: frozenset[int] | None,
        statuses: CommitReading,
        runs: CommitReading,
        sources: list[list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        """The status of *name* when it can stand for the requirement.

        The combined status keeps one entry per context, its latest.  It
        counts when *apps* binds nothing, or while a bound App has posted
        no run in *sources*.  Only a complete run listing shows that: a
        partial one can leave a bound App's run in the part not read.
        """
        if apps is not None and (not runs.complete or all(sources)):
            return None
        return next(
            (
                entry
                for entry in statuses.entries
                if (entry.get("context") or "").strip() == name
            ),
            None,
        )

    @classmethod
    def _failed_status(
        cls,
        name: str,
        apps: frozenset[int] | None,
        statuses: CommitReading,
        runs: CommitReading,
    ) -> bool:
        """Whether the status that stands for required check *name* failed.

        A partial status listing still proves it: each entry is its
        context's latest state.  For an App-bound check the run listing
        must be complete (:meth:`_required_status`).
        """
        status = cls._required_status(
            name, apps, statuses, runs, cls._eligible_runs(name, apps, runs)
        )
        if status is None:
            return False
        return (status.get("state") or "").strip().lower() in _FAILING_STATUS_STATES

    @classmethod
    def _unfinished_check(
        cls,
        name: str,
        apps: frozenset[int] | None,
        statuses: CommitReading,
        runs: CommitReading,
        now: datetime,
    ) -> str | None:
        """How required check *name* stands if unfinished, else ``None``.

        A status and a check run of the same name must both pass, so
        both are read.  The runs are :meth:`_eligible_runs`.  A run from
        another suite, such as a manually dispatched workflow, cannot
        satisfy the pull request's own run, so its success hides nothing.
        When *apps* binds the check, the status counts while a complete
        listing shows a bound App has posted no run.  A pending run is
        believed only from a complete listing: a partial one can hold an
        older pending run and miss the newer run that finished.
        """
        sources = cls._eligible_runs(name, apps, runs)
        pending: list[datetime | None] = []
        unreported = False
        for source in sources:
            if not source:
                unreported = True
            elif runs.complete:
                pending.extend(
                    _pending_since(run, "started_at", "created_at")
                    for run in source
                    if (run.get("status") or "").strip().lower() != "completed"
                )
        status = cls._required_status(name, apps, statuses, runs, sources)
        if status is not None:
            unreported = False
            state = (status.get("state") or "").strip().lower()
            if state in _PENDING_STATUS_STATES:
                pending.append(_pending_since(status, "updated_at", "created_at"))
        if pending:
            known = [since for since in pending if since is not None]
            return _pending_text(min(known) if known else None, now)
        if unreported and statuses.complete and runs.complete:
            return "not reported yet"
        return None

    async def _precommit_rerun_note(
        self,
        owner: str,
        repo: str,
        pr_info: PullRequestInfo,
        waiting: str,
        now: datetime,
    ) -> str:
        """What a re-run will do about a pre-commit.ci run that has not finished.

        A ``pre-commit.ci run`` comment inside the nudge cooldown is
        quoted with its age, since a re-run will not post another until
        the cooldown ends.  Otherwise a re-run nudges a pending run once
        it has been pending past the stuck threshold.  A nudge this
        manager posted, or saw when the cooldown read the comments,
        counts even when this reading fails or stops before it; other
        unreadable comments read as no recent nudge.
        """
        # Resolved through the package at call time, so a test rebinding
        # the constant on the package is observed.
        from dependamerge import merge_manager as _mm

        posted = self._logged_precommit_nudge(pr_info)
        newest: datetime | None = None
        if self._github_client is not None:
            try:
                reading = await self._github_client.list_issue_comments(
                    owner, repo, pr_info.number
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log.debug(
                    "Could not list comments for %s#%s: %s",
                    pr_info.repository_full_name,
                    pr_info.number,
                    exc,
                )
            else:
                newest, _ = newest_precommit_trigger(reading.comments)
        if posted is not None:
            newest = posted if newest is None else max(newest, posted)
        if newest is not None:
            age = (now - newest).total_seconds()
            if age < _mm.PRECOMMIT_CI_NUDGE_COOLDOWN_SECONDS:
                return f'; "{PRECOMMIT_TRIGGER}" posted {_format_age(age)} ago'
        if waiting.startswith("pending"):
            threshold = _format_age(_mm.PRECOMMIT_CI_STUCK_PENDING_SECONDS)
            return f"; a re-run nudges it once it has been pending {threshold}"
        return ""


def _pending_text(since: datetime | None, now: datetime) -> str:
    """``pending 42s``, or plain ``pending`` when the start is unknown."""
    if since is None:
        return "pending"
    return f"pending {_format_age((now - since).total_seconds())}"
