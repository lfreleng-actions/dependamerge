# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Re-triggering a stalled pre-commit.ci run.

Reading the status is :mod:`_precommit_status`; deciding whether that
reading means the run has stalled, and nudging it when it has, is here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..models import PullRequestInfo
from ._precommit_status import (
    PRECOMMIT_CONTEXT,
    PRECOMMIT_TRIGGER,
    find_precommit_status,
    newest_precommit_trigger,
    parse_timestamp,
    pending_age,
)
from ._precommit_wait import _PrecommitWaitMixin


def _utcnow() -> datetime:
    """The current time; a module function so tests can substitute it."""
    return datetime.now(UTC)


class _PrecommitCiMixin(_PrecommitWaitMixin):
    """Nudging pre-commit.ci when its check has gone stale."""

    async def _trigger_stale_precommit_ci(
        self, pr_info: PullRequestInfo, *, treat_missing_as_stuck: bool = True
    ) -> bool:
        """Detect and retrigger a stuck pre-commit.ci run by posting a comment.

        pre-commit.ci uses the commit status API and sometimes gets
        stuck — either never reporting a status at all, or leaving the
        ``pre-commit.ci - pr`` context in ``pending`` indefinitely.
        Either way the PR stays blocked when that context is a required
        status check.  Posting ``pre-commit.ci run`` triggers a fresh
        run.

        A run is treated as stuck when the status is missing entirely,
        when pre-commit.ci reported ``error`` (it could not complete a
        run, as distinct from the hooks failing), or when it has been
        ``pending`` for longer than
        :data:`PRECOMMIT_CI_STUCK_PENDING_SECONDS` (a slow-but-normal
        run within that window is left alone).

        Args:
            pr_info: Pull request information
            treat_missing_as_stuck: Whether a status pre-commit.ci has
                not reported at all counts as stuck.  True only when
                GitHub has settled the PR's mergeable state: an absent
                status is indistinguishable from one that has not
                propagated yet, so on a PR GitHub is still working out
                it means "too early to tell" rather than "stalled".

        Returns:
            True if a retrigger comment was posted and the status check
            subsequently completed, False otherwise.
        """
        if not self._github_client:
            return False

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        # 1. Check whether pre-commit.ci is a required status check
        if not await self._precommit_ci_required(repo_owner, repo_name, pr_info):
            return False

        # 2. Inspect the existing pre-commit.ci status.  Retrigger when
        #    it is missing entirely, when pre-commit.ci reported an
        #    ``error``, or when it has been ``pending`` past the stuck
        #    threshold; leave a reported ``success`` / ``failure``, and
        #    a pending run still within its normal window.
        now = _utcnow()
        fetched, precommit_status = await self._precommit_status(
            repo_owner, repo_name, pr_info
        )
        if not fetched:
            return False
        reported_at = parse_timestamp(
            (precommit_status or {}).get("updated_at")
            or (precommit_status or {}).get("created_at")
        )
        if precommit_status is None:
            if not treat_missing_as_stuck:
                self.log.debug(
                    "pre-commit.ci has not reported on %s#%s and GitHub has "
                    "not settled the PR yet; too early to call it stuck.",
                    pr_info.repository_full_name,
                    pr_info.number,
                )
                return False
        elif not self._precommit_run_is_stuck(precommit_status, now, pr_info):
            return False

        # 3. The run is stale (missing, errored, or stuck pending) ---
        # but a nudge inside the cooldown, whoever posted it, means
        # pre-commit.ci has been asked recently.  Timed by the comment
        # rather than by the status: pre-commit.ci answers a nudge with
        # a fresh ``queued`` status, which a status-relative rule reads
        # as a new stall, so every later run would post again.
        if await self._precommit_nudge_cooling_down(repo_owner, repo_name, pr_info):
            return False

        self._pr_status(
            f"🔄 Re-triggering pre-commit.ci: {pr_info.html_url}",
            level="info",
        )

        if not await self._post_precommit_trigger(repo_owner, repo_name, pr_info):
            return False

        # 4. Poll for the status to appear (up to ~5 minutes).  The
        # reading that prompted the nudge is passed along so the poll
        # cannot mistake it for the answer: an ``error`` sits on the
        # commit until pre-commit.ci replaces it, and the first poll
        # would otherwise return that same error and end the wait on the
        # very failure being retried.
        return await self._await_precommit_ci(
            repo_owner, repo_name, pr_info, since=reported_at
        )

    async def _precommit_ci_required(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> bool:
        """Report whether pre-commit.ci blocks merges on the base branch.

        A failure to read the required checks counts as "not required":
        without evidence that the context blocks the merge there is
        nothing worth retriggering.
        """
        if not self._github_client:
            return False
        try:
            required_checks = await self._github_client.get_required_status_checks(
                repo_owner, repo_name, pr_info.base_branch or "main"
            )
            required_contexts = [
                c.get("context", "") for c in required_checks if isinstance(c, dict)
            ]
            if PRECOMMIT_CONTEXT not in required_contexts:
                return False
        except Exception:
            return False
        return True

    async def _precommit_status(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> tuple[bool, dict[str, Any] | None]:
        """Read the pre-commit.ci status on the PR's head commit.

        Returns a ``(fetched, status)`` pair.  ``fetched`` is False only
        when the status was **absent from an incomplete reading**, which
        suppresses the retrigger: nothing was established either way.  A
        True with a status is that status, whether or not the reading
        finished; a True with ``None`` is confirmed absence from a
        complete one, meaning pre-commit.ci has reported nothing yet.

        Finding the status settles the question however the reading
        ended, so a partial one is consulted before it is judged.  Only
        **absence** needs the reading to be complete --- a context
        missing from pages we chose to stop fetching is unknown rather
        than unreported, and acting on it posts a nudge for a hook that
        may have reported perfectly well.
        """
        complete, statuses = await self._combined_statuses(
            repo_owner, repo_name, pr_info
        )
        status = find_precommit_status({"statuses": statuses})
        if status is not None:
            return True, status
        return complete, None

    def _precommit_run_is_stuck(
        self,
        precommit_status: dict[str, Any],
        now: datetime,
        pr_info: PullRequestInfo,
    ) -> bool:
        """Judge a reported pre-commit.ci status as stuck or healthy."""
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        state = precommit_status.get("state")
        if state == "error":
            # The two terminal states mean different things, the way the
            # commit status API intends: ``failure`` is the hooks
            # reporting a genuine problem with the change, ``error`` is
            # pre-commit.ci failing to complete a run at all --- an
            # upload that 5xx'd, a container that died.  The first
            # reports the same result however often it is re-run; the
            # second routinely clears on the next attempt, and until it
            # does the PR stays blocked on a verdict nobody reached.
            #
            # Age is not consulted.  A terminal state does not become
            # more stuck with time, and there is nothing left to await.
            self.log.info(
                "pre-commit.ci on %s#%s reported an infrastructure error; "
                "treating as stuck.",
                pr_info.repository_full_name,
                pr_info.number,
            )
            return True
        if state != "pending":
            # ``success`` or ``failure`` is an answer, not a stall.
            # Re-running a genuine hook failure would post a comment and
            # then wait five minutes to be told the same thing.
            return False
        # Pending: only stuck once it has been pending longer than
        # the threshold.
        age = pending_age(precommit_status, now)
        if age is None or age < _mm.PRECOMMIT_CI_STUCK_PENDING_SECONDS:
            # Still within the normal window (or no timestamp to
            # judge by) --- leave the run to finish.
            return False
        self.log.info(
            "pre-commit.ci on %s#%s pending for %.0fs; treating as stuck.",
            pr_info.repository_full_name,
            pr_info.number,
            age,
        )
        return True

    async def _precommit_nudge_cooling_down(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> bool:
        """Report whether ``pre-commit.ci run`` was posted inside the cooldown.

        The newest trigger comment decides, whoever posted it, so the
        comment is posted at most once per
        :data:`PRECOMMIT_CI_NUDGE_COOLDOWN_SECONDS` per pull request.

        An undated trigger comment counts as inside the cooldown, which
        keeps an unparsable payload on the side of not commenting twice.
        A reading that is incomplete, or that failed, and holds no nudge
        inside the cooldown does not hold the nudge back: the newest
        comment may sit in the unread tail, and a duplicate nudge is
        better than leaving a stalled run unrepaired.
        """
        if not self._github_client:
            return False
        # Resolved through the package at call time, so a test rebinding
        # the constant on the package is observed.
        from dependamerge import merge_manager as _mm

        try:
            reading = await self._github_client.list_issue_comments(
                repo_owner, repo_name, pr_info.number
            )
        except Exception as exc:
            self.log.debug(
                "Could not list comments for %s#%s before pre-commit.ci run: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return False
        newest, undated = newest_precommit_trigger(reading.comments)
        recent = (
            newest is not None
            and (_utcnow() - newest).total_seconds()
            < _mm.PRECOMMIT_CI_NUDGE_COOLDOWN_SECONDS
        )
        if not (recent or undated):
            return False
        self.log.info(
            "pre-commit.ci run already posted on %s#%s inside the cooldown; "
            "not posting another.",
            pr_info.repository_full_name,
            pr_info.number,
        )
        return True

    async def _post_precommit_trigger(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> bool:
        """Post ``pre-commit.ci run``, reporting whether it was accepted."""
        if not self._github_client:
            return False
        try:
            await self._github_client.post_issue_comment(
                repo_owner, repo_name, pr_info.number, PRECOMMIT_TRIGGER
            )
            self._record_retrigger()
        except Exception as e:
            self.log.warning(
                f"Failed to post pre-commit.ci trigger comment on "
                f"{pr_info.repository_full_name}#{pr_info.number}: {e}"
            )
            return False
        return True
