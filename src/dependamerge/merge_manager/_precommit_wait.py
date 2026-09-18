# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Reading the commit status, and waiting for pre-commit.ci to answer.

The two belong together: waiting *is* reading the same endpoint on a
loop, and both need the same view of what "the status" is --- every
context on the head commit, across every page.

Separated from the retrigger decision in :mod:`_precommit_ci`, which
asks whether a reading means the run has stalled and nudges it when it
has.  Nothing here decides anything; it only reports what GitHub says
and how long it is prepared to keep asking.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._precommit_status import precommit_outcome

#: Status contexts fetched per request.  The combined-status endpoint
#: defaults to 30, so a repository with a moderate integration set can
#: push ``pre-commit.ci - pr`` onto a second page --- where reading one
#: page misreads an errored run as one that never reported.
_STATUSES_PER_PAGE = 100

#: How many pages to read.  A cap keeps a pathological commit from
#: costing the run unbounded requests.  Filling it leaves the reading
#: partial unless something else proves it whole --- see
#: :meth:`_PrecommitWaitMixin._combined_statuses`.
_MAX_STATUS_PAGES = 10


class _PrecommitWaitMixin(_MergeManagerBase):
    """Reading pre-commit.ci's status, and polling until it settles."""

    async def _combined_statuses(
        self, repo_owner: str, repo_name: str, pr_info: PullRequestInfo
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Every status context on the PR's head commit, across pages.

        Returns ``(complete, statuses)``.  ``complete`` says the reading
        accounts for every status on the commit, so **absence** from it
        is evidence.  A partial reading still carries what it found:
        finding a context is conclusive however the read ended, and only
        not finding one depends on having read it all.

        Paginated because the endpoint defaults to 30 entries.  On a
        repository with more integrations than that, a single page can
        omit ``pre-commit.ci - pr`` entirely, and this path reads absence
        as "never reported": an errored run would then look missing,
        leaving an ``unknown`` PR unrepaired and a ``blocked`` one nudged
        without the incident timestamp that scopes the suppression.

        The page cap reintroduces that misreading at a larger threshold,
        so filling it leaves ``complete`` False.  Two things prove the
        read whole before then: a short page, and the payload's own
        ``total_count`` --- the latter only when every row carried a
        distinct id, since offset paging over a list that grows beneath
        it can return one row twice and miss another.
        """
        if not self._github_client:
            return False, []
        statuses: list[dict[str, Any]] = []
        identified: set[int] = set()
        total: int | None = None
        page = 1
        while page <= _MAX_STATUS_PAGES:
            try:
                data = await self._github_client.get(
                    f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}"
                    f"/status?per_page={_STATUSES_PER_PAGE}&page={page}"
                )
            except Exception as e:
                self.log.debug(
                    "Failed to fetch commit status for pre-commit.ci check on %s#%s "
                    "(sha=%s); the reading stops here and is incomplete: %s",
                    pr_info.repository_full_name,
                    pr_info.number,
                    pr_info.head_sha,
                    e,
                )
                return False, statuses
            if not isinstance(data, dict):
                return False, statuses
            reported = data.get("total_count")
            if isinstance(reported, int) and not isinstance(reported, bool):
                # The largest count seen, not the first.  Statuses are
                # returned newest first, so one posted between requests
                # shifts every later entry down a slot --- and can push
                # one past the cap while the page-one count still says
                # the read is whole.  Keeping the stale figure would
                # then certify exactly the truncation this guards.
                total = reported if total is None else max(total, reported)
            raw = data.get("statuses")
            if not isinstance(raw, list):
                # A shape we cannot read is not an empty page, and a
                # missing or null field is not an empty one either.  The
                # endpoint always carries the key, so its absence is a
                # response we did not understand --- and falling through
                # would make it the shortest page of all and certify the
                # absence of everything.
                return False, statuses
            batch = [s for s in raw if isinstance(s, dict)]
            statuses.extend(batch)
            for entry in batch:
                entry_id = entry.get("id")
                if isinstance(entry_id, int) and not isinstance(entry_id, bool):
                    identified.add(entry_id)
            if len(batch) != len(raw):
                # Entries we could not parse were dropped, so this page
                # is short for a reason that says nothing about how many
                # statuses exist.  Keep what parsed --- a target already
                # found is still evidence --- but do not call it whole.
                return False, statuses
            if len(batch) < _STATUSES_PER_PAGE:
                return True, statuses
            if (
                total is not None
                and len(identified) == len(statuses)
                and len(identified) >= total
            ):
                # Distinct rows, counted by identity rather than by how
                # many arrived.  Offset paging over a list that can grow
                # returns the same row twice when something is inserted
                # ahead of it, so a length alone can reach the total
                # while a row we never saw sits past the cap ---
                # certifying an absence that was never established.
                # Requiring every row to have carried a distinct id
                # makes the arithmetic mean what it says, and a payload
                # without ids simply falls through to the page rules.
                return True, statuses
            page += 1
        self.log.debug(
            "Commit status for %s#%s (sha=%s) filled the %d-page budget; "
            "absence from the reading is not evidence",
            pr_info.repository_full_name,
            pr_info.number,
            pr_info.head_sha,
            _MAX_STATUS_PAGES,
        )
        return False, statuses

    async def _await_precommit_ci(
        self,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
        since: datetime | None = None,
    ) -> bool:
        """Poll the head commit until pre-commit.ci reports a result.

        pre-commit.ci can take up to five minutes to run and report
        back, so we need a generous timeout to avoid prematurely marking
        PRs as unmergeable when the check simply hasn't finished yet.
        The whole poll is a wait on an external service, so the worker's
        concurrency slot is released for its duration (``parked()``).

        The poll honours the run-wide ceiling ``--max-wait`` sets, in
        the same way as the auto-merge and required-workflow waits: a
        stale pre-commit status previously parked the worker for the
        full ``merge_timeout`` even under ``--max-wait 0``, which
        promises never to block.
        """
        # Resolved through the package at call time rather than bound at
        # import time, so that a test rebinding the constant on
        # ``dependamerge.merge_manager`` is observed here.
        from dependamerge import merge_manager as _mm

        if self._no_wait:
            self.log.debug(
                "Not waiting for pre-commit.ci on %s#%s (--max-wait 0)",
                pr_info.repository_full_name,
                pr_info.number,
            )
            return False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._merge_timeout
        if self._run_deadline is not None:
            deadline = min(deadline, self._run_deadline)

        max_polls = self._merge_poll_max_attempts
        async with _mm.parked():
            for attempt in range(max_polls):
                # Sleep no longer than the time remaining, matching
                # ``_wait_for_auto_merge`` and the required-workflow
                # wait.  Checking the deadline without clamping would
                # still overshoot the ceiling by up to a full interval.
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(self._merge_recheck_interval, remaining))
                outcome = await self._poll_precommit_status(
                    repo_owner, repo_name, pr_info, since=since
                )
                if outcome is not None:
                    return outcome

                if attempt == max_polls - 1:
                    self.log.debug(
                        f"Still waiting for pre-commit.ci on "
                        f"{pr_info.repository_full_name}#{pr_info.number} "
                        f"({(attempt + 1) * self._merge_recheck_interval:.0f}s elapsed)"
                    )

        self.log.warning(
            f"Timed out waiting for pre-commit.ci on "
            f"{pr_info.repository_full_name}#{pr_info.number}"
        )
        return False

    async def _poll_precommit_status(
        self,
        repo_owner: str,
        repo_name: str,
        pr_info: PullRequestInfo,
        since: datetime | None = None,
    ) -> bool | None:
        """Take one reading of pre-commit.ci, or None while still pending.

        *since* is the reading the retrigger was posted against; a
        terminal status no newer than it has not been replaced yet and
        reads as pending.
        """
        if not self._github_client:
            return None
        _complete, statuses = await self._combined_statuses(
            repo_owner, repo_name, pr_info
        )
        # Whatever was read is consulted, complete or not: a terminal
        # status found in a partial reading settles the wait, and one
        # absent from it reads as pending --- which is what an
        # incomplete reading should mean here anyway.
        outcome = precommit_outcome({"statuses": statuses}, since)
        if outcome is True:
            self._pr_status(
                f"✅ pre-commit.ci passed: {pr_info.html_url}",
                level="info",
            )
            return True
        if outcome is False:
            self._pr_status(
                f"❌ pre-commit.ci failed: {pr_info.html_url}",
                level="warning",
            )
            return False
        return None
