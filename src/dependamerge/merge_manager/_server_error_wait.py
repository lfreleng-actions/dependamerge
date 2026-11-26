# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Waiting out a server error between attempts at a merge.
"""

from __future__ import annotations

import asyncio

from .. import github_async as _gh
from ..models import PullRequestInfo
from ..slot_lease import parked_off_dispatch
from ._base import _MergeManagerBase

#: The pause before retrying a merge after an inconclusive error.
_SHORT_RETRY_WAIT = 1.0


class _ServerErrorWaitMixin(_MergeManagerBase):
    """The backoff between merge attempts GitHub failed to answer."""

    async def _wait_after_server_error(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        failures: int,
        attempt: int,
        deadline: float | None = None,
    ) -> bool:
        """Wait out a server error before another attempt; ``False`` to stop.

        Each merge names its commit, and the loop re-reads the PR before
        every retry, so repeating a merge that may have landed is safe.
        The waits (``_SERVER_ERROR_RETRY_WAITS``) give a short outage
        time to pass.  They stop at the run's deadline, checked again
        once a wait ends, and free the worker's slot, and the
        repository's dispatch lock, while they last.  Under
        ``--max-wait 0`` the short wait of any other error is kept, and
        *attempt* (counted from 0 over every error) is held to
        ``max_retries`` as any other retry is.  *deadline* replaces the
        run's for a caller whose own budget is already clamped to it.
        """
        pr_key = f"{owner}/{repo}#{pr_info.number}"
        if deadline is None:
            deadline = self._run_deadline
        if self._no_wait:
            if attempt >= self.max_retries:
                return False
            await asyncio.sleep(_SHORT_RETRY_WAIT)
            return True
        waits = _gh._SERVER_ERROR_RETRY_WAITS
        if failures > len(waits):
            return False
        delay = waits[failures - 1]
        if deadline is not None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            delay = min(delay, remaining)
        self.log.info(
            "Server error merging %s (attempt %d/%d); retrying in %.0fs",
            pr_key,
            failures,
            len(waits) + 1,
            delay,
        )
        async with parked_off_dispatch():
            await asyncio.sleep(delay)
        # The wait, or taking the slot and lock back, may spend the budget.
        return self._server_error_budget_left(deadline)

    def _server_error_budget_left(self, deadline: float | None = None) -> bool:
        """Whether a merge may still be retried after a server error.

        Asked once the PR has been re-read before the retry, since a slow
        read can spend what the wait left.  *deadline* is as in
        :meth:`_wait_after_server_error`; under ``--max-wait 0`` the
        short waits keep no deadline, so neither does this.
        """
        if self._no_wait:
            return True
        if deadline is None:
            deadline = self._run_deadline
        return deadline is None or asyncio.get_running_loop().time() < deadline
