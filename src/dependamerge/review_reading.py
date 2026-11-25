# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
A pull request's complete review list, as :class:`ReviewInfo` entries.

The scan reads only the first page of a pull request's reviews.  A
decision that rests on a review being absent (no change request, no
Copilot review left to dismiss) reads them all first, through the
paginated REST reader on ``GitHubAsync``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import PullRequestInfo, ReviewInfo

if TYPE_CHECKING:
    from .github_async import GitHubAsync


def review_info_from_rest(review: dict[str, Any]) -> ReviewInfo | None:
    """Convert a REST review object, or ``None`` when it carries no state.

    The GraphQL node ID is kept as the review's ``id``, as the scan
    keeps it, because dismissing a review is a GraphQL mutation that
    takes that ID; the numeric REST ``id`` is only a fallback.
    """
    state = review.get("state")
    if not isinstance(state, str) or not state:
        return None
    node_id = review.get("node_id")
    review_id = node_id if isinstance(node_id, str) and node_id else review.get("id")
    login = (review.get("user") or {}).get("login")
    return ReviewInfo(
        id="" if review_id is None else str(review_id),
        user=login if isinstance(login, str) and login else "unknown",
        state=state,
        submitted_at=review.get("submitted_at") or "",
        body=review.get("body"),
    )


async def read_reviews(
    api: GitHubAsync, owner: str, repo: str, number: int
) -> tuple[list[ReviewInfo], bool]:
    """Every review on a pull request, and whether all of them were read.

    An entry that does not convert is a review whose verdict is unknown,
    so dropping one leaves the list incomplete.
    """
    reading = await api.read_pull_request_reviews(owner, repo, number)
    converted = [review_info_from_rest(review) for review in reading.entries]
    reviews = [info for info in converted if info is not None]
    return reviews, reading.complete and len(reviews) == len(converted)


async def complete_pr_reviews(
    api: GitHubAsync, pr_info: PullRequestInfo, owner: str, repo: str
) -> bool:
    """Make ``pr_info.reviews`` the full list; return whether it is.

    Costs nothing when the list is already whole.  Otherwise reads every
    review and, only when that reading is whole, replaces the list; a
    partial reading leaves ``pr_info`` untouched and returns False.
    """
    if pr_info.reviews_complete:
        return True
    reviews, complete = await read_reviews(api, owner, repo, pr_info.number)
    if not complete:
        return False
    pr_info.reviews = reviews
    pr_info.reviews_complete = True
    return True
