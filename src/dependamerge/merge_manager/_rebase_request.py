# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Pacing of ``@dependabot rebase`` requests.

A pull request whose body carries dependabot's "rebasing" notice has a
rebase under way, whatever its comments say.  Otherwise the newest
``@dependabot rebase`` comment on it decides what a new request does:

1. **In flight.**  Younger than ``REBASE_IN_FLIGHT_SECONDS`` and the PR's
   live head has not moved since: a rebase is under way, so nothing is posted and
   callers wait for it.
2. **Cooling down.**  Otherwise younger than the cooldown: nothing is
   posted and no rebase is under way.  The suppression is recorded in
   ``_rebase_suppressed`` so callers neither wait nor fall back.
3. **Due.**  Otherwise the macro is posted and counted.

When the newest request cannot be established (the comments could not
all be read) the macro is posted: a duplicate request is harmless and is
better than skipping recovery.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._constants import DEFAULT_REBASE_COOLDOWN_HOURS
from ._head_pin import stop_after_head_move

_REBASE_MACRO = "@dependabot rebase"


def _utcnow() -> datetime:
    """The current time; a module function so tests can substitute it."""
    return datetime.now(UTC)


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a GitHub ISO 8601 timestamp, or return None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _newest_rebase_request(comments: Any) -> datetime | None:
    """The ``created_at`` of the newest comment asking dependabot to rebase."""
    newest: datetime | None = None
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        body = comment.get("body")
        if not (isinstance(body, str) and _REBASE_MACRO in body):
            continue
        created = _parse_timestamp(comment.get("created_at"))
        if created is not None and (newest is None or created > newest):
            newest = created
    return newest


def _format_duration(seconds: float, *, round_up: bool = False) -> str:
    """Render a duration as ``3h 10m`` or ``45m``."""
    raw = max(0.0, seconds) / 60
    minutes = math.ceil(raw) if round_up else math.floor(raw)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


class _RebaseRequestMixin(_MergeManagerBase):
    """Requesting a dependabot rebase, at most once per cooldown."""

    _rebase_cooldown_seconds: float = DEFAULT_REBASE_COOLDOWN_HOURS * 3600

    async def _request_dependabot_rebase(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Post ``@dependabot rebase`` unless one is in flight or cooling down.

        Dependabot responds by rebasing the PR branch onto the latest
        base, regenerating any lockfiles and re-signing the commit ---
        the reliable way to clear a ``uv.lock`` / dependency conflict
        that a plain ``git rebase`` cannot resolve.

        Returns:
            True when a rebase is under way: dependabot's notice in the
            body says it is rebasing, one was just requested, or an
            earlier request is still in flight.  False when no rebase
            is under way: the macro could not be posted, or the cooldown
            held it back.  The latter is recorded in
            ``_rebase_suppressed``, which callers consult so that they
            neither wait nor fall back to another rebase path.

        Raises:
            ConfirmedHeadSignal: When a rebase is under way for a PR
                confirmed at its current commit; the request is made and
                counted first.
        """
        under_way = await self._ask_dependabot_to_rebase(pr_info, owner, repo)
        if under_way:
            stop_after_head_move(self, pr_info, "rebase")
        return under_way

    async def _ask_dependabot_to_rebase(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """The request itself; see :meth:`_request_dependabot_rebase`."""
        pr_key = f"{owner}/{repo}#{pr_info.number}"
        previous_note = self._rebase_suppressed.pop(pr_key, None)
        if self._dependabot_is_rebasing(pr_info.body):
            # Dependabot's own notice outranks an older macro's age: a
            # rebase is under way, so nothing is posted or held back.
            self.log.info(
                "Dependabot already rebasing %s#%s; not requesting another.",
                pr_info.repository_full_name,
                pr_info.number,
            )
            return True
        if self._github_client is None:
            return False

        # Resolved through the package at call time, so a test rebinding
        # the constant on the package is observed.
        from dependamerge import merge_manager as _mm

        last = await self._last_rebase_request(pr_info, owner, repo)
        if last is not None:
            age = (_utcnow() - last).total_seconds()
            if age < _mm.REBASE_IN_FLIGHT_SECONDS and not await self._head_moved_since(
                pr_info, owner, repo, last
            ):
                self.log.info(
                    "@dependabot rebase on %s#%s requested %s ago and the "
                    "head has not moved since; treating rebase as in flight.",
                    pr_info.repository_full_name,
                    pr_info.number,
                    _format_duration(age),
                )
                return True
            if age < self._rebase_cooldown_seconds:
                self._suppress_rebase(
                    pr_info, pr_key, age, announce=previous_note is None
                )
                return False

        try:
            await self._github_client.post_issue_comment(
                owner, repo, pr_info.number, _REBASE_MACRO
            )
        except Exception as exc:
            self.log.warning(
                "Failed to post @dependabot rebase on %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return False
        # One macro comment and one rebase request: both totals move.
        # The in-flight and cooldown paths above deliberately count
        # nothing, since neither posted anything.
        self._record_retrigger()
        self._record_rebase()
        return True

    async def _last_rebase_request(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> datetime | None:
        """When ``@dependabot rebase`` was last requested, if that is known.

        None both when no request exists and when the comments could not
        all be read: the newest request may sit in the unread tail, and
        posting is the safe answer to either.
        """
        if self._github_client is None:
            return None
        try:
            reading = await self._github_client.list_issue_comments(
                owner, repo, pr_info.number
            )
        except Exception as exc:
            self.log.debug(
                "Could not list comments for %s#%s before rebase request: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return None
        if not reading.complete:
            self.log.debug(
                "Could not read every comment on %s#%s before rebase request; "
                "requesting anyway.",
                pr_info.repository_full_name,
                pr_info.number,
            )
            return None
        return _newest_rebase_request(reading.comments)

    async def _head_moved_since(
        self, pr_info: PullRequestInfo, owner: str, repo: str, since: datetime
    ) -> bool:
        """Whether the PR's head commit was committed after *since*.

        The head is the PR's live one: the scanned SHA predates any push
        dependabot made since, and reading it would keep a request in
        flight that has already been answered.  The scanned SHA stands in
        when the live head cannot be read.

        Unknown reads as "not moved", so an unreadable head keeps a young
        request in flight, as the guard always treated it.
        """
        if self._github_client is None:
            return False
        head_sha = await self._live_head_sha(pr_info, owner, repo)
        if not head_sha:
            return False
        try:
            data = await self._github_client.get(
                f"/repos/{owner}/{repo}/commits/{head_sha}"
            )
        except Exception as exc:
            self.log.debug(
                "Could not read head commit of %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return False
        commit = data.get("commit") if isinstance(data, dict) else None
        committer = commit.get("committer") if isinstance(commit, dict) else None
        if not isinstance(committer, dict):
            return False
        committed = _parse_timestamp(committer.get("date"))
        return committed is not None and committed > since

    async def _live_head_sha(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> str | None:
        """The PR's current head SHA, or the scanned one if unreadable."""
        try:
            data = await self._fetch_pr_state(owner, repo, pr_info.number)
        except Exception as exc:
            self.log.debug(
                "Could not read live head of %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            data = None
        head = data.get("head") if isinstance(data, dict) else None
        sha = head.get("sha") if isinstance(head, dict) else None
        return sha if isinstance(sha, str) and sha else pr_info.head_sha or None

    def _suppress_rebase(
        self, pr_info: PullRequestInfo, pr_key: str, age: float, *, announce: bool
    ) -> None:
        """Record a request the cooldown held back, and say so once."""
        remaining = self._rebase_cooldown_seconds - age
        note = (
            f"rebase last requested {_format_duration(age)} ago, "
            f"next request in {_format_duration(remaining, round_up=True)}"
        )
        self._rebase_suppressed[pr_key] = note
        if announce:
            self._pr_status(
                f"⏸️ Rebase cooling down: {pr_info.html_url} [{note}]",
                level="info",
            )
