# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Whether a pull request's reviews make a new approval unnecessary."""

from __future__ import annotations

from typing import Any


def already_sufficiently_approved(
    pr_data: dict[str, Any],
    reviews_data: list[Any],
    current_user: str,
) -> tuple[bool, str | None]:
    """Return ``(skip, approvers)`` when the PR needs no new approval.

    ``skip`` is True when the current user has already approved, or when
    other reviewers have approved a ``clean`` PR.  ``approvers`` names the
    relevant approver(s) for the debug log.
    """
    for review in reviews_data:
        if not isinstance(review, dict):
            continue
        if (review.get("user") or {}).get("login") == current_user and review.get(
            "state"
        ) == "APPROVED":
            return True, current_user

    approved_reviews = [
        review
        for review in reviews_data
        if isinstance(review, dict)
        and review.get("state") == "APPROVED"
        and (review.get("user") or {}).get("login") != current_user
    ]
    if approved_reviews and pr_data.get("mergeable_state") == "clean":
        # A review may carry ``"user": null`` or ``{"login": null}``;
        # coerce both to a string so join() cannot see a None login.
        approvers = [
            (review.get("user") or {}).get("login") or "unknown"
            for review in approved_reviews
        ]
        return True, ", ".join(approvers)
    return False, None
