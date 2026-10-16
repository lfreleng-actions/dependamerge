# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The poll loop that waits for a pending merge to complete.

It applies the adaptive head start from ``_wait_head_start``:
repositories whose checks are consistently slow have the first stretch
of the wait skipped, which saves API calls without materially delaying
the outcome.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..merge_signals import DraftPullRequestError
from ..models import PullRequestInfo
from ._wait_head_start import _WaitHeadStartMixin


class _AutoMergeWaitMixin(_WaitHeadStartMixin):
    """Waiting for auto-merge, with an adaptive head start."""

    async def _wait_for_auto_merge(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        *,
        continue_states: tuple[str, ...],
        deadline: float | None = None,
        stop_on_clean: bool = True,
        measures_checks: bool = False,
    ) -> tuple[bool, bool]:
        """Poll a PR until it merges, closes, settles, or times out.

        Registers ``owner/repo#N`` in ``_waiting_prs`` so the parallel
        progress ticker can render an aggregate countdown, then polls
        every ``_merge_recheck_interval`` seconds until one of:

          * ``mergeable_state`` becomes ``clean`` (only when
            ``stop_on_clean`` is True — the default; callers that have
            enabled auto-merge and want to observe the PR actually
            close set it False and include ``"clean"`` in
            ``continue_states`` instead),
          * the PR closes (capturing the ``merged`` flag so the caller
            can tell auto-merge success from closed-without-merge),
          * ``mergeable_state`` leaves ``continue_states`` (the caller
            decides what the new state means), or
          * the deadline passes.

        The total wait is bounded by ``merge_timeout`` unless an
        explicit ``deadline`` is supplied — used to share a single
        budget across sequential waits (e.g. the rebase-then-checks
        phases of conflict recovery).

        Mutates ``pr_info`` in place (``mergeable``,
        ``mergeable_state``, ``head_sha``, ``state``, ``is_draft``).
        Returns ``(closed_during_wait, merged_during_wait)``, or raises
        :class:`DraftPullRequestError` for a PR marked draft meanwhile.
        """
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        if self._github_client is None:
            return False, False

        # Fire-and-forget (``max_wait == 0``): never block.  Returning
        # "not closed" lets callers fall through to the auto-merge-pending
        # path (Step 6 / the conflict handler arms auto-merge and reports
        # AUTO_MERGE_PENDING).  Auto-merge is armed by the caller before
        # this point, so GitHub still completes the merge later.
        if self._no_wait:
            return False, False

        pr_key = f"{owner}/{repo}#{pr_info.number}"
        loop = asyncio.get_running_loop()
        # Drive the wait off a monotonic deadline so the total is
        # bounded even if a single iteration over-sleeps slightly.
        if deadline is None:
            deadline = loop.time() + self._merge_timeout
        # Clamp to the owner-wide global ceiling (when set) so no single
        # PR's wait can push the whole run past ``max_wait``.
        if self._run_deadline is not None:
            deadline = min(deadline, self._run_deadline)

        async with self._waiting_lock:
            self._waiting_prs[pr_key] = deadline

        # Track whether the PR was closed during the wait and, if so,
        # whether it was actually merged.  The REST payload's
        # ``merged`` boolean distinguishes auto-merge success from
        # closed-without-merge (a human closed it, dependabot
        # superseded it, etc.).
        closed_during_wait = False
        merged_during_wait = False
        first_poll = True
        # Set per iteration: whether that poll supplied a usable live
        # ``mergeable_state``.  Guards the head start below.
        live_state_seen = False
        # Bound before the try so the ``finally`` can always read it,
        # even if the parked block is never entered.
        wait_started = loop.time()
        # Set only where a wait is judged to have measured checks; see
        # the assignment at the end of the poll loop.
        wait_ended: float | None = None
        try:
            # The whole poll loop is a wait on an external event
            # (auto-merge / CI / a rebase), so release this worker's
            # concurrency slot for its duration — a parked PR must
            # never starve runnable PRs (see ``slot_lease.py``).  The
            # polling GETs are paced by the HTTP client's own limits.
            async with _mm.parked():
                # The head start is applied after the first live poll,
                # not here: see ``_apply_wait_head_start``.  Sizing it
                # from the fetch-time snapshot could sleep through a PR
                # that had already gone green.
                head_start_pending = measures_checks
                while loop.time() < deadline:
                    if stop_on_clean and pr_info.mergeable_state == "clean":
                        break
                    # Sleep no longer than the time remaining so we don't
                    # overshoot the deadline.  Clamp to non-negative: the
                    # ``while`` check and this ``time()`` call are not
                    # atomic, so a near-deadline crossing could otherwise
                    # pass ``asyncio.sleep`` a tiny negative value.  The
                    # first poll uses a much shorter delay (see
                    # ``MERGE_WAIT_FIRST_POLL_SECONDS``) so a PR that
                    # resolved the moment the wait started is detected
                    # promptly instead of a full interval late.
                    interval = (
                        min(
                            _mm.MERGE_WAIT_FIRST_POLL_SECONDS,
                            self._merge_recheck_interval,
                        )
                        if first_poll
                        else self._merge_recheck_interval
                    )
                    first_poll = False
                    remaining = max(0.0, deadline - loop.time())
                    await asyncio.sleep(min(interval, remaining))
                    try:
                        refreshed_wait = await self._fetch_pr_state(
                            owner, repo, pr_info.number
                        )
                    except Exception as wait_exc:
                        self.log.debug(
                            "Failed to refresh PR state during auto-merge "
                            "wait for %s: %s",
                            pr_key,
                            wait_exc,
                        )
                        continue
                    if isinstance(refreshed_wait, dict):
                        # ``_apply_wait_refresh`` deliberately keeps the
                        # previous value when GitHub answers ``null`` /
                        # ``""`` / ``"unknown"`` while recomputing, and
                        # ``_fetch_pr_state`` can return ``None`` without
                        # raising when a per-PR fallback fails.  Either
                        # way ``pr_info`` still holds the *snapshot*
                        # value, so the head start must not be sized
                        # from it --- that is the stale read this change
                        # exists to remove.
                        live_state_seen = refreshed_wait.get("mergeable_state") not in (
                            None,
                            "",
                            "unknown",
                        )
                        if self._apply_wait_refresh(pr_info, refreshed_wait):
                            closed_during_wait = True
                            merged_during_wait = bool(
                                refreshed_wait.get("merged", False)
                            )
                            break
                    if (
                        pr_info.mergeable_state == "unstable"
                        and pr_info.mergeable is True
                    ):
                        break
                    # Continue waiting only while the PR is in a state the
                    # caller still considers rescuable; any other value
                    # means it became mergeable, closed, or hit a terminal
                    # state, so exit and let the caller decide.
                    if pr_info.mergeable_state not in continue_states:
                        break

                    # Reaching here means the PR has been read live and
                    # genuinely still needs waiting through: it is not
                    # clean, not closed, and still in a continue state.
                    # Only now is a head start worth taking.  Pending
                    # rather than tied to the first iteration, so a
                    # transient fetch failure --- or a reading GitHub
                    # could not supply --- defers it instead of
                    # discarding it or spending it on a stale value.
                    if head_start_pending and live_state_seen:
                        head_start_pending = False
                        if await self._apply_wait_head_start(
                            pr_info,
                            pr_key,
                            max(0.0, deadline - loop.time()),
                            continue_states,
                            stop_on_clean,
                            measures_checks,
                        ):
                            # The head start is sized to land as the
                            # checks finish, so read promptly afterwards.
                            # Without this the next sleep would be the
                            # full steady-state interval, adding it to
                            # every wait the optimisation applies to.
                            first_poll = True
                # Stop the clock inside the park: leaving it re-acquires
                # this worker's slot, and that queue is the scheduler's
                # time, not the repository's.  Only a wait that polled
                # and saw the checks resolve measured anything.
                if not first_poll and self._checks_resolved(pr_info):
                    wait_ended = loop.time()
        finally:
            async with self._waiting_lock:
                self._waiting_prs.pop(pr_key, None)
            # Whatever was true before the wait is not necessarily true
            # after it: that is the point of waiting.  Drop the memo
            # rather than rely on its expiry, which a short
            # ``--merge-timeout`` can outlast.
            if self._github_client is not None:
                self._github_client.invalidate_block_reason(owner, repo, pr_info.number)
            # Record the latency only when the wait produced a result and
            # actually measured checks (see ``_record_check_wait``).
            if measures_checks:
                self._record_check_wait(pr_info, wait_started, wait_ended, deadline)

        return closed_during_wait, merged_during_wait

    @staticmethod
    def _apply_wait_refresh(
        pr_info: PullRequestInfo, refreshed: dict[str, Any]
    ) -> bool:
        """Fold a polled snapshot into *pr_info*; report whether it closed.

        Extracted from :meth:`_wait_for_auto_merge` to keep that method
        within the complexity budget.

        Each field is overwritten only when present *and* usable.
        GitHub returns ``null`` for ``mergeable`` and ``null`` / ``""``
        / ``"unknown"`` for ``mergeable_state`` while it recomputes
        mergeability, and letting those through would push the state out
        of the caller's ``continue_states`` and end the wait early --- a
        ``blocked`` PR briefly reading ``unknown`` would exit and trigger
        a premature manual merge.

        The head is kept current because it can change mid-wait (rebase,
        force-push), and later block-reason analysis must query the
        commit the PR is actually on.

        Raises:
            DraftPullRequestError: When the open PR is now a draft.  No
                check can make it mergeable until it is marked ready, so
                waiting on would time out and report auto-merge pending.
        """
        if "mergeable" in refreshed:
            refreshed_mergeable = refreshed.get("mergeable")
            if refreshed_mergeable is not None:
                pr_info.mergeable = refreshed_mergeable
        if "mergeable_state" in refreshed:
            refreshed_state = refreshed.get("mergeable_state")
            if refreshed_state not in (None, "", "unknown"):
                pr_info.mergeable_state = refreshed_state
        refreshed_head = (refreshed.get("head") or {}).get("sha")
        if refreshed_head:
            pr_info.head_sha = refreshed_head
        if refreshed.get("state") == "closed":
            pr_info.state = "closed"
            return True
        if refreshed.get("draft") is True:
            pr_info.is_draft = True
            raise DraftPullRequestError(
                f"{pr_info.repository_full_name}#{pr_info.number}"
            )
        return False
