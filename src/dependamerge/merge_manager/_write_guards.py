# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
The checks made immediately before a write to a pull request.

The run settles its scope once, after the scan, but a pull request can
reach a write by more than one route, and its author can mark it not
ready while it waits in the queue.  So the writes that matter most,
approving and dispatching a merge, check again here: the bot policy, the
repository's own policy, and the draft flag, read live where the
snapshot may be stale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..bot_identity import is_automation_author
from ..bot_policy import BotPolicy
from ..merge_signals import DraftPullRequestError
from ..models import PullRequestInfo

if TYPE_CHECKING:
    from ._base import _MergeManagerBase

__all__ = ["draft_state_now", "is_draft_now", "refuse_out_of_scope_approval"]


async def draft_state_now(
    manager: _MergeManagerBase, owner: str, repo: str, number: int
) -> bool | None:
    """The PR's live draft flag, or ``None`` when the read fails.

    A read through the manager's batched state poller.  Best effort: the
    caller keeps its snapshot on a failed read, and GitHub still refuses
    to merge a draft.
    """
    try:
        data = await manager._fetch_pr_state(owner, repo, number)
    except Exception as exc:
        manager.log.debug(
            "Draft check failed for %s/%s#%s: %s", owner, repo, number, exc
        )
        return None
    draft = data.get("draft") if isinstance(data, dict) else None
    return draft if isinstance(draft, bool) else None


async def is_draft_now(
    manager: _MergeManagerBase, pr_info: PullRequestInfo, owner: str, repo: str
) -> bool:
    """Whether the PR is a draft now, recording it on ``pr_info``."""
    draft = await draft_state_now(manager, owner, repo, pr_info.number)
    if draft is not None:
        pr_info.is_draft = draft
    return pr_info.is_draft


def refuse_out_of_scope_approval(
    policy: BotPolicy, pr_info: PullRequestInfo, owner: str, repo: str
) -> None:
    """Raise unless a run under ``policy`` may approve ``pr_info``.

    Entry to the merge flow already refuses all three cases; this guards
    the approval itself, whatever route reached it.

    Raises:
        RuntimeError: When the PR's App author is outside the run's bot
            policy, or its repository's policy is invalid.
        DraftPullRequestError: When the snapshot says the PR is a draft;
            a conversion since the scan is caught by the live read in
            ``_approve_pr``.
    """
    repository = f"{owner}/{repo}"
    refusal = f"refusing to approve {repository}#{pr_info.number}"
    if is_automation_author(pr_info.author) and not policy.admits(
        pr_info.author, repository
    ):
        raise RuntimeError(f"{refusal}: {pr_info.author} is not in this run's scope")
    if policy.repository_error(repository) is not None:
        raise RuntimeError(f"{refusal}: the repository's bot policy is invalid")
    if pr_info.is_draft:
        raise DraftPullRequestError(f"{repository}#{pr_info.number}")
