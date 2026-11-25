# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Signals that end the processing of one pull request.

Raised from the checks made immediately before a write, and caught only
by the merge manager's single-PR handler, which turns each into that
PR's outcome.  Every layer in between --- recovery paths, auto-merge
wrappers, the Step 5 rebase callbacks --- catches broad exceptions to
keep a run going, so each must re-raise :class:`PullRequestStopSignal`
first: swallowed, a signal would let the very write it guards go ahead.

A module of its own, depending on nothing, so ``rebase`` and
``merge_manager`` can both use it without importing each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import PullRequestInfo

__all__ = [
    "ConfirmedHeadSignal",
    "DraftPullRequestError",
    "DraftStateUnknownError",
    "PullRequestStopSignal",
    "ReviewsUnreadableError",
]


class PullRequestStopSignal(Exception):
    """Stop acting on this pull request; never to be swallowed."""

    #: The pull request the signal concerns when it is not the one the
    #: merge began with: dependabot's replacement for it.  Set by the code
    #: acting on the replacement, so the outcome names that pull request.
    pull_request: PullRequestInfo | None = None


class DraftPullRequestError(PullRequestStopSignal):
    """The pull request is a draft, so it is skipped, not approved.

    Raised from the last reads before approving or dispatching a merge,
    so a pull request its author marked not ready since the scan ends as
    skipped rather than approved or refused a merge.
    """


class DraftStateUnknownError(PullRequestStopSignal):
    """Whether the pull request is a draft could not be read.

    An approval waits for an explicit "not a draft": an approval given to
    a draft would satisfy required reviews once it is marked ready, and
    refusing the later merge cannot undo it.  So the pull request fails
    without the approval, and a re-run tries again.
    """


class ReviewsUnreadableError(PullRequestStopSignal):
    """The reviews could not all be read before an approval or its retry.

    Only a whole reading shows the pull request lacks this run's own
    approval, and a failed POST may still have created one.  So the pull
    request is skipped, neither approved again nor merged, and a re-run
    tries again.
    """


class ConfirmedHeadSignal(PullRequestStopSignal):
    """A confirmed pull request cannot be merged at its confirmed commit.

    Its head moved since a person confirmed it, or the run asked for a
    rebase or recreate that will move it.  So the pull request is
    skipped, neither approved at nor merged at a commit nobody
    confirmed, and a re-run asks again.

    Attributes:
        reason: The skip reason shown for the pull request.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
