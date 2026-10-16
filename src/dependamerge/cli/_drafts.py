# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Draft pull requests: listed, recorded as skipped, never acted on.

A draft cannot merge until its author marks it ready, and was made a
draft for a reason, so a run sets drafts aside as soon as its scope is
settled.  Approving one would act on work its author has not finished,
and attempting to merge one only produces GitHub's refusal.

Neither API's merge state says a PR is a draft any more; ``is_draft``
(GraphQL ``isDraft``, REST ``draft``) is the only reliable signal.
"""

from __future__ import annotations

from ..merge_manager import MergeResult, MergeStatus
from ..models import PullRequestInfo
from ._app import console
from ._context import _MergeContext

__all__ = ["DRAFT_REASON", "_set_aside_drafts"]

#: How a draft is recorded in the run's results.
DRAFT_REASON = "draft PR"


def _set_aside_drafts(
    ctx: _MergeContext, prs: list[PullRequestInfo]
) -> list[PullRequestInfo]:
    """Return ``prs`` without drafts, listing and recording the drafts.

    The drafts are recorded on the context as skipped, so the run's
    results document reports them without them taking any part in the
    merge, its counts or its confirmation values.
    """
    drafts = [pr for pr in prs if pr.is_draft]
    if not drafts:
        return prs
    console.print(f"📝 Draft PRs skipped: {len(drafts)}")
    for pr in drafts:
        console.print(
            f"   {pr.repository_full_name}#{pr.number} (by {pr.author})",
            markup=False,
        )
    ctx.draft_results.extend(
        MergeResult(pr_info=pr, status=MergeStatus.SKIPPED, error=DRAFT_REASON)
        for pr in drafts
    )
    return [pr for pr in prs if not pr.is_draft]
