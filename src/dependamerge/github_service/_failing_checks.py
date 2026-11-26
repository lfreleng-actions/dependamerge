# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""The failing checks ``blocked`` reports for a pull request.

The scan's page holds the first ``SCAN_CHECKS_PAGE_SIZE`` entries of a
commit's check rollup, and the rollup's per-state counts cover all of
it.  The counts decide whether anything fails; only when something does
and the page is short are the commit's check runs and statuses read in
full, so a failing check past the page is never missed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from ..check_runs import failing_check_names, failing_status_names
from ._pr_extract import scan_check_page, split_check_contexts

if TYPE_CHECKING:
    from ..github_async import GitHubAsync

log = logging.getLogger(__name__)


class FailingChecks(NamedTuple):
    """Names of the failing checks, and whether every check was read.

    ``complete`` is False only when something is known to fail and the
    checks could not all be read, so more may be failing than named.
    """

    names: list[str]
    complete: bool


def _failing_names(
    runs: list[dict[str, Any]], statuses: list[dict[str, Any]]
) -> list[str]:
    return failing_check_names(runs) + failing_status_names(statuses)


def _latest_statuses(*readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Statuses by context, a later reading's state replacing an earlier one."""
    by_context: dict[str, dict[str, Any]] = {}
    for reading in readings:
        for status in reading:
            by_context[(status.get("context") or "").strip()] = status
    return list(by_context.values())


async def read_failing_checks(
    api: GitHubAsync, repo_full_name: str, pr: dict[str, Any]
) -> FailingChecks:
    """The failing checks on a PR node's latest commit.

    The latest run per check name decides, and a status context fails
    in the ``failure`` or ``error`` state, as for a page that holds them
    all.  When the full read fails part-way, the names found in the page
    and in what was read are reported, and ``complete`` is False.
    """
    page = scan_check_page(pr)
    if page.failing == 0:
        return FailingChecks([], True)
    page_runs, page_statuses = split_check_contexts(page.contexts)
    page_names = _failing_names(page_runs, page_statuses)
    if page.complete:
        return FailingChecks(page_names, True)

    owner, _, repo = repo_full_name.partition("/")
    ref = page.oid or pr.get("headRefOid")
    if isinstance(ref, str) and ref and repo:
        runs, statuses = await asyncio.gather(
            api.read_check_runs_for_ref(owner, repo, ref),
            api.read_commit_statuses(owner, repo, ref),
        )
        if runs.complete and statuses.complete:
            return FailingChecks(_failing_names(runs.entries, statuses.entries), True)
        names = _failing_names(
            page_runs + runs.entries, _latest_statuses(page_statuses, statuses.entries)
        )
    else:
        names = page_names

    if not names and not page.failing:
        # Nothing is known to fail: the counts are unreadable, and no
        # check read so far is failing.
        return FailingChecks([], True)
    log.warning(
        "Checks on %s#%s could not all be read; more may be failing than %s",
        repo_full_name,
        pr.get("number", "?"),
        ", ".join(names) or "reported",
    )
    return FailingChecks(names, False)
