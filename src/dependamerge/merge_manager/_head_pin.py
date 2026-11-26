# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
The commit every merge and approval names.

GitHub merges and approves whatever the head is when the request lands
unless the request names a commit.  So every write here names one, by
two rules:

* A pull request nobody confirmed merges at the head the run last read
  (its snapshot).  When GitHub refuses because the head moved, the
  snapshot is refreshed from one live read and the bounded retry tries
  again at the new head.
* A pull request a person confirmed (a major bump at the interactive
  prompt, a gated single-PR source, a human PR covered by
  ``--override``) is approved and merged only at the commit they
  confirmed.  When the head moves, or the run itself asks for a rebase
  or recreate, the pull request stops as skipped so a later run can ask
  again about the new commit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, NoReturn

from ..github_async import ApprovalError
from ..merge_signals import (
    ConfirmedHeadSignal,
    DraftPullRequestError,
    DraftStateUnknownError,
)
from ..models import PullRequestInfo
from ._types import _github_error_detail, _merged_from_payload
from ._write_guards import draft_from_state, pr_state_now

if TYPE_CHECKING:
    from ._base import _MergeManagerBase

__all__ = [
    "ApprovalHeadMovedError",
    "ConfirmedHead",
    "approval_head",
    "approval_says_head_moved",
    "approve_at_head",
    "HEAD_CHANGED_DURING_MERGE",
    "check_confirmed_pr",
    "check_payload_head",
    "confirmed_head",
    "head_key",
    "merge_head",
    "payload_head_stop",
    "raise_head_moved",
    "record_live_head",
    "refresh_head",
    "says_head_moved",
    "stop_after_head_move",
    "stop_after_refused_head",
]

#: The failure reason for an unconfirmed PR whose head kept moving.
HEAD_CHANGED_DURING_MERGE = "head changed during the merge; re-run"

_HEAD_MOVE_REASONS = {
    "rebase": "rebase requested; re-run to confirm the rebased commit",
    "recreate": ("recreate requested; re-run to confirm the replacement pull request"),
}

_HEAD_UNKNOWN = "head unknown when it was confirmed; re-run to confirm it"
_STATE_UNREADABLE = "head and auto-merge state could not be read; re-run"
_CLOSED_HEAD_UNKNOWN = (
    "closed, but the read showed no head, so the commit merged is unknown"
)
_AUTO_MERGE_ARMED = (
    "auto-merge is enabled, so GitHub could merge a later commit; disable it and re-run"
)


#: The HTTP status GitHub answers a merge naming a commit that is no
#: longer the head with: ``409 Conflict``.
_MERGE_409_RE = re.compile(r"'409 [^']*' for url")

#: GitHub's own sentence for that refusal, anchored to the start of its
#: detail: a quoted check or workflow name can never stand there.
_MERGE_HEAD_MOVED_RE = re.compile(r"\AHead branch was modified\b", re.IGNORECASE)

#: An approval refused because its ``commit_id`` is not the PR's head:
#: the start of one entry of GitHub's detail, never a phrase inside one.
_APPROVAL_HEAD_MOVED_RE = re.compile(
    r"(?:\A|;\s*)(?:head sha moved|commit_id is not part of the pull request)\b",
    re.IGNORECASE,
)


def says_head_moved(error_msg: str) -> bool:
    """Whether GitHub refused a merge because the head is not the one named.

    Read from the status line, ahead of GitHub's detail, and from the
    start of that detail alone, as ``_stale_head`` reads its verdict:
    neither a quoted check or workflow name in the detail nor the
    PR-state suffix the merge layer appends can match.
    """
    status_part = error_msg.partition("GitHub: ")[0].split(" (PR state:", 1)[0]
    if _MERGE_409_RE.search(status_part):
        return True
    return _MERGE_HEAD_MOVED_RE.search(_github_error_detail(error_msg)) is not None


def approval_says_head_moved(refusal: ApprovalError) -> bool:
    """Whether GitHub refused an approval for naming a commit not the head."""
    return (
        refusal.status_code == 422
        and _APPROVAL_HEAD_MOVED_RE.search(refusal.detail or "") is not None
    )


class ApprovalHeadMovedError(ApprovalError):
    """GitHub kept refusing an approval because the head moved.

    Raised once the one retry at the re-read head is spent, so the PR
    reports :data:`HEAD_CHANGED_DURING_MERGE` as a refusal a later
    reading can settle, not as the run failing.
    """


@dataclass(frozen=True)
class ConfirmedHead:
    """The commit a person confirmed for one pull request.

    Attributes:
        sha: The head commit shown when the person confirmed.
        label: A reason to keep in front of a skip, such as the major
            gate's, or ``None``.
    """

    sha: str
    label: str | None = None


def head_key(repository_full_name: str, number: int) -> tuple[str, int]:
    """The key a confirmed head is recorded under."""
    return (repository_full_name.lower(), number)


def confirmed_head(
    manager: _MergeManagerBase, pr_info: PullRequestInfo
) -> ConfirmedHead | None:
    """The confirmed head for ``pr_info``, or ``None`` when unconfirmed."""
    heads: dict[tuple[str, int], ConfirmedHead] = (
        getattr(manager, "_confirmed_heads", None) or {}
    )
    return heads.get(head_key(pr_info.repository_full_name, pr_info.number))


def _with_label(confirmed: ConfirmedHead, reason: str) -> str:
    return f"{confirmed.label}; {reason}" if confirmed.label else reason


def _require_known(confirmed: ConfirmedHead) -> None:
    """Stop a confirmed PR recorded without a head: nothing can pin it."""
    if not confirmed.sha:
        raise ConfirmedHeadSignal(_with_label(confirmed, _HEAD_UNKNOWN))


def _moved_reason(confirmed: ConfirmedHead, live_sha: str | None) -> str:
    now = live_sha[:7] if live_sha else "unknown"
    return _with_label(
        confirmed,
        f"head changed since it was confirmed (was {confirmed.sha[:7]}, "
        f"now {now}); re-run to confirm the new commit",
    )


def raise_head_moved(confirmed: ConfirmedHead, live_sha: str | None) -> NoReturn:
    """Stop a confirmed PR whose head is no longer the confirmed commit."""
    raise ConfirmedHeadSignal(_moved_reason(confirmed, live_sha))


def stop_after_refused_head(confirmed: ConfirmedHead, live_sha: str | None) -> NoReturn:
    """Stop a confirmed PR GitHub refused a write for, as naming an old head.

    GitHub's refusal settles that the confirmed commit is no longer the
    head, so the PR stops as skipped whatever the read after it found:
    a failed or headless read, or one still showing the confirmed
    commit, reports the new head as unknown.
    """
    raise_head_moved(confirmed, live_sha if live_sha != confirmed.sha else None)


def payload_head_stop(
    manager: _MergeManagerBase, pr_info: PullRequestInfo, data: Any
) -> str | None:
    """Why a live payload stops a confirmed PR, or ``None``.

    Read before a closed or merged reading counts as this run's merge:
    someone else may have merged a later commit.  The payload's own head
    is compared, never the snapshot's, which a headless read leaves at
    the old commit; a closed reading with no head stops the PR too.
    """
    confirmed = confirmed_head(manager, pr_info)
    if confirmed is None or not isinstance(data, dict):
        return None
    if not confirmed.sha:
        return _with_label(confirmed, _HEAD_UNKNOWN)
    head = _payload_head(data)
    if head:
        return None if head == confirmed.sha else _moved_reason(confirmed, head)
    if data.get("state") == "closed" or _merged_from_payload(data):
        return _with_label(confirmed, _CLOSED_HEAD_UNKNOWN)
    return None


def check_payload_head(
    manager: _MergeManagerBase, pr_info: PullRequestInfo, data: Any
) -> None:
    """Raise :func:`payload_head_stop`'s reason as a ``ConfirmedHeadSignal``."""
    reason = payload_head_stop(manager, pr_info, data)
    if reason is not None:
        raise ConfirmedHeadSignal(reason)


def stop_after_head_move(
    manager: _MergeManagerBase, pr_info: PullRequestInfo, kind: str
) -> None:
    """Stop a confirmed PR once the run has asked for a new head.

    Called after the request is made and counted.  A no-op for a PR
    nobody confirmed, which goes on to merge at its new head.
    """
    confirmed = confirmed_head(manager, pr_info)
    if confirmed is not None:
        raise ConfirmedHeadSignal(_with_label(confirmed, _HEAD_MOVE_REASONS[kind]))


def _payload_head(data: dict[str, Any]) -> str | None:
    head = data.get("head")
    sha = head.get("sha") if isinstance(head, dict) else None
    return sha if isinstance(sha, str) and sha else None


def record_live_head(pr_info: PullRequestInfo, data: Any) -> str | None:
    """Record the head in a live PR payload on the snapshot.

    When the head differs from the snapshot, the mergeable fields read
    with it replace the snapshot's too, since they describe the old
    commit.  Returns the live head, or ``None`` when the payload has
    none.
    """
    if not isinstance(data, dict):
        return None
    sha = _payload_head(data)
    if sha is None:
        return None
    if sha != pr_info.head_sha:
        pr_info.head_sha = sha
        pr_info.mergeable = data.get("mergeable")
        pr_info.mergeable_state = data.get("mergeable_state")
    return sha


async def refresh_head(
    manager: _MergeManagerBase, pr_info: PullRequestInfo, owner: str, repo: str
) -> str | None:
    """Refresh the snapshot's head from one live read.

    Returns the live head, or ``None`` when the read fails or carries
    none; the snapshot is then left as it was.
    """
    try:
        data = await manager._fetch_pr_state(owner, repo, pr_info.number)
    except Exception as exc:
        manager.log.debug(
            "Head read failed for %s/%s#%s: %s", owner, repo, pr_info.number, exc
        )
        return None
    return record_live_head(pr_info, data)


async def merge_head(
    manager: _MergeManagerBase, pr_info: PullRequestInfo, owner: str, repo: str
) -> str:
    """The commit a merge of ``pr_info`` must name.

    Raises:
        ConfirmedHeadSignal: When a confirmed PR's snapshot head is no
            longer the confirmed commit.
        RuntimeError: When no head is known and none can be read.
    """
    confirmed = confirmed_head(manager, pr_info)
    if confirmed is not None:
        _require_known(confirmed)
        if pr_info.head_sha and pr_info.head_sha != confirmed.sha:
            raise_head_moved(confirmed, pr_info.head_sha)
        return confirmed.sha
    if pr_info.head_sha:
        return pr_info.head_sha
    sha = await refresh_head(manager, pr_info, owner, repo)
    if not sha:
        raise RuntimeError(
            f"no head commit known for {owner}/{repo}#{pr_info.number}; "
            "refusing to write without one"
        )
    return sha


def approval_head(
    manager: _MergeManagerBase, pr_info: PullRequestInfo, data: Any
) -> str:
    """The commit an approval of ``pr_info`` must name.

    ``data`` is the live state read just before the approval; its head
    is recorded on the snapshot.  A PR nobody confirmed is approved at
    the live head, the one any merge after it names.

    Raises:
        ConfirmedHeadSignal: When a confirmed PR's head is no longer the
            confirmed commit, so no approval is given.
        RuntimeError: When no head is known.
    """
    head = record_live_head(pr_info, data) or pr_info.head_sha
    confirmed = confirmed_head(manager, pr_info)
    if confirmed is not None:
        _require_known(confirmed)
        if head and head != confirmed.sha:
            raise_head_moved(confirmed, head)
        return confirmed.sha
    if not head:
        raise RuntimeError(
            f"no head commit known for {pr_info.repository_full_name}"
            f"#{pr_info.number}; refusing to approve without one"
        )
    return head


async def approve_at_head(
    manager: _MergeManagerBase,
    owner: str,
    repo: str,
    pr_info: PullRequestInfo,
    commit_id: str,
) -> None:
    """Post this run's approval of ``pr_info`` at ``commit_id``.

    When GitHub refuses the commit as no longer the head, the head is
    read once and the approval tried again at it, provided that same
    read says explicitly the PR is not a draft.

    Raises:
        ConfirmedHeadSignal: When GitHub refuses a confirmed PR's
            commit, whatever the read after it finds.
        DraftPullRequestError: When the read before the retry finds a
            draft.
        DraftStateUnknownError: When that read carries no draft flag.
        ApprovalHeadMovedError: When GitHub refuses the retry for the
            same reason, or the read shows no head to retry at.
        ApprovalError: When GitHub refuses for another reason.
    """
    client = manager._github_client
    if client is None:
        raise RuntimeError("GitHub client not initialized")
    body = "🤖 Dependamerge\nApproved this pull request ✅"
    approve = partial(client.approve_pull_request, owner, repo, pr_info.number, body)
    deadline, no_wait = manager._run_deadline, manager._no_wait
    try:
        await approve(commit_id=commit_id, deadline=deadline, short_waits=no_wait)
    except ApprovalError as refusal:
        if not approval_says_head_moved(refusal):
            raise
        state = await pr_state_now(manager, owner, repo, pr_info.number)
        live = record_live_head(pr_info, state)
        confirmed = confirmed_head(manager, pr_info)
        if confirmed is not None:
            stop_after_refused_head(confirmed, live)
        retry_at = approval_head(manager, pr_info, None)
        target = f"{owner}/{repo}#{pr_info.number}"
        if not live or retry_at == commit_id:
            raise ApprovalHeadMovedError(
                target, refusal.status_code, refusal.detail
            ) from refusal
        # The guard ``_approve_pr`` applies, for the commit retried at.
        draft = draft_from_state(state)
        if draft is None:
            raise DraftStateUnknownError(target) from refusal
        if draft:
            raise DraftPullRequestError(target) from refusal
        try:
            await approve(commit_id=retry_at, deadline=deadline, short_waits=no_wait)
        except ApprovalError as again:
            if not approval_says_head_moved(again):
                raise
            raise ApprovalHeadMovedError(
                target, again.status_code, again.detail
            ) from again


async def check_confirmed_pr(
    manager: _MergeManagerBase, pr_info: PullRequestInfo, owner: str, repo: str
) -> None:
    """Stop a confirmed PR that cannot be merged only at its commit.

    One live read, before anything is approved or awaited.  Auto-merge
    already armed, by an earlier run or the author, would let GitHub
    merge whatever head passes its checks, so such a PR is skipped
    rather than left to it.  A no-op for a PR nobody confirmed.

    Raises:
        ConfirmedHeadSignal: When the confirmed head is unknown, the
            read fails, the head has moved, or auto-merge is armed.
    """
    confirmed = confirmed_head(manager, pr_info)
    if confirmed is None:
        return
    _require_known(confirmed)
    client = manager._github_client
    if client is None:
        raise RuntimeError("GitHub client not initialized")
    try:
        data = await client.get(f"/repos/{owner}/{repo}/pulls/{pr_info.number}")
    except Exception as exc:
        manager.log.debug(
            "Confirmed head read failed for %s/%s#%s: %s",
            owner,
            repo,
            pr_info.number,
            exc,
        )
        data = None
    live = record_live_head(pr_info, data)
    if not isinstance(data, dict) or live is None:
        raise ConfirmedHeadSignal(_with_label(confirmed, _STATE_UNREADABLE))
    if live != confirmed.sha:
        raise_head_moved(confirmed, live)
    if data.get("auto_merge") is not None:
        raise ConfirmedHeadSignal(_with_label(confirmed, _AUTO_MERGE_ARMED))
