# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Waiting for a rebase this run requested to land.

Step 5 asks dependabot to rebase and moves on; dependabot force-pushes a
minute or so later.  Until it does, every reading of the pull request
describes the head being replaced.  That head can read ``clean``, or
mergeable ``unstable``, before the rebase lands, and the check wait
would end on it, so Step 6 would merge the head dependabot is about to
replace, racing the force-push.  This wait runs first and ends only
when GitHub has read the new head, the pull request closes, or the
deadline it shares with the check wait passes.  A rebase that has not
landed by then ends the pull request's processing: nothing it read
describes the head that will be merged.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..models import PullRequestInfo
from ._single_pr_context import _MergeFlow
from ._types import MergeResult, MergeStatus
from ._wait import _AutoMergeWaitMixin

# What GitHub answers for ``mergeable_state`` while it recomputes.
_UNREAD_STATES = (None, "", "unknown")


def rebase_settled(pr_info: PullRequestInfo, from_head: str) -> bool:
    """Whether the PR has left *from_head* and GitHub has read the new one."""
    return pr_info.head_sha != from_head and pr_info.mergeable_state not in (
        _UNREAD_STATES
    )


def _forget_old_head(pr_info: PullRequestInfo, refreshed: dict[str, Any]) -> None:
    """Drop the old head's mergeability that a new-head reading left in place.

    ``_apply_wait_refresh`` keeps the previous values when GitHub has not
    computed them yet.  Across a head change those values describe the
    head being replaced, and a ``clean`` carried onto the new head would
    end the check wait before the new head's checks have run.
    """
    if refreshed.get("mergeable") is None:
        pr_info.mergeable = None
    if refreshed.get("mergeable_state") in _UNREAD_STATES:
        pr_info.mergeable_state = "unknown"


class _RebaseLandingMixin(_AutoMergeWaitMixin):
    """Holding the check wait until a requested rebase moves the head."""

    async def _await_rebase_landing(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        *,
        from_head: str,
        deadline: float,
    ) -> tuple[bool, bool]:
        """Poll until the rebase settles, the PR closes, or *deadline* passes.

        Settled means :func:`rebase_settled`: a head other than
        *from_head*, with a mergeable state GitHub has computed for it.
        Returns ``(closed, merged)`` as :meth:`_wait_for_auto_merge`
        does, and folds each reading into *pr_info* the same way, so a
        draft or a confirmed PR's moved head raises here too.  Under
        ``--max-wait 0`` nothing is waited for.  The wait runs outside
        the dispatch lock, so it parks this worker's slot as the check
        wait does.
        """
        # Resolved through the package at call time, so a test rebinding
        # ``parked`` on the package is observed.
        from dependamerge import merge_manager as _mm

        if self._github_client is None or self._no_wait:
            return False, False
        pr_key = f"{owner}/{repo}#{pr_info.number}"
        loop = asyncio.get_running_loop()
        async with self._waiting_lock:
            self._waiting_prs[pr_key] = deadline
        try:
            async with _mm.parked():
                while not rebase_settled(pr_info, from_head) and loop.time() < deadline:
                    remaining = max(0.0, deadline - loop.time())
                    await asyncio.sleep(min(self._merge_recheck_interval, remaining))
                    try:
                        refreshed = await self._fetch_pr_state(
                            owner, repo, pr_info.number
                        )
                    except Exception as exc:
                        self.log.debug(
                            "Failed to refresh PR state while waiting for the "
                            "rebase of %s: %s",
                            pr_key,
                            exc,
                        )
                        continue
                    if not isinstance(refreshed, dict):
                        continue
                    read_from = pr_info.head_sha
                    if self._apply_wait_refresh(pr_info, refreshed):
                        return True, bool(refreshed.get("merged", False))
                    if pr_info.head_sha != read_from:
                        _forget_old_head(pr_info, refreshed)
        finally:
            async with self._waiting_lock:
                self._waiting_prs.pop(pr_key, None)
        return False, False

    def _rebase_not_landed(self, flow: _MergeFlow) -> MergeResult:
        """End a PR whose requested rebase did not land within the wait.

        Its head is still the one Step 5 asked dependabot to replace, so
        a direct merge would land the stale commit.  Auto-merge, when
        armed, is left to finish the job once the rebase arrives.
        """
        pr_info = flow.pr_info
        result = flow.result
        head = (pr_info.head_sha or "unknown")[:7]
        if flow.pr_key in self._auto_merge_enabled:
            result.status = MergeStatus.AUTO_MERGE_PENDING
            result.error = f"auto-merge pending: requested rebase of {head} not landed"
            self._pr_status(
                f"⏳ Waiting: {pr_info.html_url} [requested rebase not landed]",
                level="debug",
            )
            return result
        result.status = MergeStatus.FAILED
        result.error = (
            f"requested rebase did not land before the wait ended "
            f"(head still {head}); re-run once it has"
        )
        self._pr_status(
            f"❌ Failed: {pr_info.html_url} [requested rebase not landed]",
            level="error",
        )
        return result
