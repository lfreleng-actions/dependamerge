# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The preview (dry-run) simulation of a merge's outcome.

It answers the question the not-mergeable handling answers, without
touching the repository.
"""

from __future__ import annotations

from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._types import MergeResult, MergeStatus


class _PreviewSimulationMixin(_MergeManagerBase):
    """Simulating the merge step in preview mode."""

    def _simulate_preview_merge(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> None:
        """Simulate the Step 6 merge outcome for preview mode.

        Mutates ``result`` in place.  No console output: preview
        progress is conveyed by the Rich tracker counters (with
        preview-accurate labels such as "Mergeable") and the per-PR
        outcomes/reasons are reported in the end-of-run summary, so
        per-PR lines here go to the log only.
        """
        if pr_info.mergeable_state == "behind" and not self.fix_out_of_date:
            result.status = MergeStatus.SKIPPED
            result.error = "PR is behind base branch and --no-fix option is set"
            self._pr_status(
                f"\u23ed\ufe0f Skipped: {pr_info.html_url} [behind, rebase disabled]",
                level="debug",
            )
        elif pr_info.mergeable_state == "behind" and self.fix_out_of_date:
            # Behind PRs merge directly unless branch protection
            # requires up-to-date heads, in which case the real run
            # refreshes the branch first — either way the PR counts
            # as mergeable.
            result.status = MergeStatus.MERGED
            # Use ``warning`` (not ``error``) so the MERGED result
            # does not carry a contradictory error message.
            result.warning = "behind base branch"
            self._pr_status(
                f"\u2611\ufe0f Approve/merge: {pr_info.html_url} [behind base branch]",
                level="debug",
            )
        elif pr_info.mergeable_state == "dirty":
            result.status = MergeStatus.BLOCKED
            result.error = "PR has merge conflicts"
            self._pr_status(
                f"\U0001f6d1 Blocked: {pr_info.html_url} [merge conflicts]",
                level="debug",
            )
        elif pr_info.mergeable is False and pr_info.mergeable_state == "blocked":
            result.status = MergeStatus.BLOCKED
            result.error = "PR blocked by failing checks"
            self._pr_status(
                f"\U0001f6d1 Blocked: {pr_info.html_url} [blocked by failing checks]",
                level="debug",
            )
        else:
            # Simulate successful merge in preview mode
            result.status = MergeStatus.MERGED
            self._pr_status(
                f"\u2611\ufe0f Approve/merge: {pr_info.html_url}",
                level="debug",
            )
