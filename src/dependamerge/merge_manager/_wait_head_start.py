# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The adaptive head start for waits on required checks.

Repositories whose checks are consistently slow have the first stretch
of the wait skipped, which saves API calls without materially delaying
the outcome.  The poll loop itself, which applies it, is ``_wait``.
"""

from __future__ import annotations

import asyncio

from ..models import PullRequestInfo
from ._base import _MergeManagerBase


class _WaitHeadStartMixin(_MergeManagerBase):
    """Sizing, recording and applying the head start of a wait."""

    # A head start is only worth taking when a repository's checks are
    # slow relative to the poll cadence; below this multiple of the
    # recheck interval the normal rhythm already costs little.
    _HEAD_START_MIN_INTERVALS = 3.0

    # Fraction of the observed median to skip.  Deliberately short of
    # 1.0 so a repository that speeds up is still caught promptly.
    _HEAD_START_FRACTION = 0.8

    def _record_wait_duration(self, repo_full_name: str, seconds: float) -> None:
        """Remember how long a repository's checks took to resolve.

        Only durations from waits that *ended in a result* are recorded;
        a timeout says nothing about latency except that it exceeded the
        budget.
        """
        if seconds <= 0:
            return
        self._repo_wait_seconds.setdefault(repo_full_name, []).append(seconds)

    def _wait_head_start(self, repo_full_name: str, budget: float) -> float:
        """Seconds to skip before the first poll of a wait.

        Polling a repository every ten seconds from t=0 when its checks
        reliably take four minutes spends around twenty requests learning
        nothing.  Once one PR in the repository has shown how long its
        checks take, its siblings can sleep most of that time first ---
        the striped scheduler runs them one after another, so by the
        second PR the observation already exists.

        Returns ``0.0`` when nothing is known, when the repository is
        quick enough that the normal cadence is cheap, or when the
        remaining budget is too small to gamble on a single sleep.
        """
        observations = self._repo_wait_seconds.get(repo_full_name)
        if not observations:
            return 0.0
        ordered = sorted(observations)
        mid = len(ordered) // 2
        # True median: average the middle pair for an even count.  Taking
        # the upper middle instead lets a single slow outlier set the
        # figure --- [60, 900] would yield 900 rather than 480 --- which
        # is exactly what the median is chosen to avoid.
        median = (
            ordered[mid]
            if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0
        )
        if median < self._HEAD_START_MIN_INTERVALS * self._merge_recheck_interval:
            return 0.0
        # Never spend more than half the remaining budget asleep: a
        # repository that has become faster must still be observed.
        return max(0.0, min(median * self._HEAD_START_FRACTION, budget / 2.0))

    async def _apply_wait_head_start(
        self,
        pr_info: PullRequestInfo,
        pr_key: str,
        remaining: float,
        continue_states: tuple[str, ...],
        stop_on_clean: bool,
        measures_checks: bool,
    ) -> bool:
        """Sleep past the latency this repository has already demonstrated.

        Returns whether it actually slept, so the caller can restore the
        short first-poll cadence: the head start is sized to land as the
        checks finish, and following it with the full steady-state
        interval would add that interval to every wait it applies to.

        Applies only to waits that measure checks.  The figure describes
        check latency, so using it to skip ahead in a *rebase* wait would
        sleep on an unrelated measurement --- and a rebase often lands in
        seconds, so a check-sized head start could sleep clean through it.

        Skipped, too, unless the PR is currently in a state this wait
        intends to sit through.  The conflict path calls back with
        ``stop_on_clean`` after a rebase may already have left the PR
        ``clean``; sleeping first would burn up to half the shared
        deadline before the loop noticed it should return immediately.

        Called **after** the first live poll rather than before it, so
        ``pr_info`` carries state the loop has just read.  Sizing it from
        the run's opening snapshot meant that in a striped run --- where
        a repository's PRs are deliberately serialised --- a sibling
        could sleep up to half its remaining budget having *already* gone
        green, because its snapshot still said ``blocked`` from minutes
        earlier.  The head start exists to save time on slow
        repositories, so spending it on a resolved PR is the exact
        opposite of the intent, and most likely on precisely the
        repositories it was added to help.

        Extracted from :meth:`_wait_for_auto_merge` to keep that method
        within the complexity budget; see :meth:`_wait_head_start` for
        the sizing decision.
        """
        if not measures_checks:
            return False
        if stop_on_clean and pr_info.mergeable_state == "clean":
            return False
        if continue_states and pr_info.mergeable_state not in continue_states:
            # Already outside the states this call waits through, so the
            # loop is about to exit; there is nothing to skip ahead to.
            return False
        head_start = self._wait_head_start(pr_info.repository_full_name, remaining)
        if head_start <= 0:
            return False
        self.log.debug(
            "Head start of %.0fs for %s: this repository's checks have "
            "taken that long already",
            head_start,
            pr_key,
        )
        await asyncio.sleep(head_start)
        return True

    @staticmethod
    def _checks_resolved(pr_info: PullRequestInfo) -> bool:
        """Whether the PR's state says its checks have finished.

        Used to decide whether a completed wait measured anything.  The
        loop can also end because the PR turned ``dirty``, was closed,
        or went ``behind`` --- none of which timed a check run, and all
        of which tend to resolve in seconds, so recording them would
        drag the repository's median down and quietly disable the head
        start for its siblings.

        ``unstable`` counts when the PR is mergeable: that is every
        required check finished, with a non-required one failing.
        """
        if pr_info.mergeable_state == "clean":
            return True
        return pr_info.mergeable_state == "unstable" and pr_info.mergeable is True

    def _record_check_wait(
        self,
        pr_info: PullRequestInfo,
        started: float,
        ended: float | None,
        deadline: float,
    ) -> None:
        """Note how long this repository's checks took, for its siblings.

        Only a wait that ended of its own accord counts.  ``ended`` is
        ``None`` when the wait recorded no measurement --- an exception
        cut it short, it never polled, or it ended for a reason that
        timed no check run --- and a wait that reached the deadline
        bounds the latency from below without measuring it.

        ``ended`` is also sampled *before* the caller's parked block
        exits, since leaving it re-acquires a concurrency slot that on a
        busy run queues behind other work; charging that scheduler delay
        to the repository's checks would teach every sibling to sleep
        through it too.

        The caller decides whether its wait measured checks at all:
        :meth:`_wait_for_auto_merge` also waits for dependabot rebases
        and for an armed auto-merge to close, and recording those would
        let a rebase turnaround masquerade as check latency and hand the
        same PR's next phase a head start worth half its remaining
        budget.
        """
        if ended is None or ended >= deadline:
            return
        self._record_wait_duration(pr_info.repository_full_name, ended - started)
