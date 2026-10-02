# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Recovery for a merge GitHub refused because the pull request's head is stale.

Under GitHub Actions a runner has no identity to sign a rebase with, so a
stale head gets the ``@dependabot rebase`` macro where one can land, or,
when the base requires signed commits, a recorded refusal that reports
the PR as needing a manual rebase.  Shared by the merge's retry loop and
by the retry after required workflows finish, which dispatches its own
merge call and hands any new rejection back unhandled.
"""

from __future__ import annotations

import re

from ..bot_identity import is_dependabot
from ..ci_report import in_github_actions
from ..models import PullRequestInfo
from ._base import _MergeManagerBase
from ._types import _github_error_detail

#: GitHub's sentence for a head that needs updating.  The REST answer is
#: "Head branch is out of date"; the other two cover phrasings of the
#: same state.  Anchored to the start of GitHub's detail, optionally
#: after the ruleset preamble, because that is the one place no quoted
#: check or workflow name can occupy: a name such as 'CI: head branch is
#: behind' always follows the preamble's own wording first.
_STALE_HEAD_RE = re.compile(
    r"\A(?:Repository rule violations found\s+)?Head branch is "
    r"(?:out of date|not up to date|behind)\b",
    re.IGNORECASE,
)


def _says_head_is_stale(error_msg: str) -> bool:
    """Whether GitHub's own detail says the PR's head needs updating.

    Read from the detail alone, so neither a repository name nor the
    PR-state suffix the merge layer appends can trigger a rebase.
    """
    return _STALE_HEAD_RE.search(_github_error_detail(error_msg)) is not None


class _StaleHeadRecoveryMixin(_MergeManagerBase):
    """Bringing a PR GitHub called stale up to date, the Actions way."""

    def _stale_head_recovery_applies(self, error_msg: str) -> bool:
        """Whether a merge rejection gets the Actions stale-head recovery."""
        return (
            self.fix_out_of_date
            and in_github_actions()
            and "405" in error_msg
            and _says_head_is_stale(error_msg)
        )

    async def _recover_stale_head(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Request the macro, or record the refusal, for a stale head.

        Returns:
            True when retrying the merge now cannot help: a dependabot
            rebase is under way with auto-merge armed to finish it, or the
            rebase was refused.  False when nothing conclusive happened (a
            macro that could not be posted, or auto-merge that could not
            be armed), so a bounded back-off should decide instead of
            reporting a running rebase as failed.
        """
        # GitHub says the PR is behind, so correct the shared snapshot
        # first: later classification (auto-merge pending) reads it too.
        pr_info.mergeable_state = "behind"
        if is_dependabot(pr_info.author) and (
            self._dependabot_is_rebasing(pr_info.body)
            or await self._request_dependabot_rebase(pr_info, owner, repo)
        ):
            self._rebase_requested.add(f"{owner}/{repo}#{pr_info.number}")
            return await self._enable_auto_merge_with_approval(pr_info, owner, repo)
        return await self._refuse_unsigned_rebase(pr_info, owner, repo)

    async def _recover_stale_head_rejection(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Apply the recovery when the last stored rejection names a stale head.

        Returns:
            True when the recovery ran but settled nothing (a macro not
            posted, or auto-merge not armed for an accepted one), so a
            bounded retry of the merge may still land it.  False when it
            does not apply or retrying cannot help; its outcome (an armed
            auto-merge, a recorded refusal) then stays in the shared
            state, where failure reporting reads it.
        """
        last = self._last_merge_exception.get(f"{owner}/{repo}#{pr_info.number}")
        if last is None or not self._stale_head_recovery_applies(str(last)):
            return False
        return not await self._recover_stale_head(pr_info, owner, repo)
