# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Review, comment and pull-request mutation calls.

Approving a pull request (including the duplicate-approval check that
makes the outer retry safe), posting comments, retitling, updating a
branch from its base, and closing a pull request.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from typing import (
    Any,
)

# ``_APPROVE_RETRY_BASE_DELAY`` and ``_is_transient_server_error`` stay
# attributes of the package rather than names bound here: they were
# module-level attributes of ``dependamerge.github_async`` before the
# split, and callers substitute them there.
import dependamerge.github_async as _pkg

from ._base import _GitHubAsyncBase
from ._errors import (
    _APPROVE_MAX_ATTEMPTS,
    _APPROVE_THROTTLE_MAX_ATTEMPTS,
    ApprovalError,
    _http_status,
    _is_throttled_detail,
    _retry_after_seconds,
    github_error_detail,
)


class _ReviewsMixin(_GitHubAsyncBase):
    """Review and pull-request mutation calls for ``GitHubAsync``."""

    async def update_pull_request_title(
        self, owner: str, repo: str, number: int, title: str
    ) -> None:
        """Set a pull request's title.

        REST: PATCH /repos/{owner}/{repo}/pulls/{pull_number}

        GitHub emits a ``pull_request.edited`` event, which re-runs any
        workflow listening for it --- the mechanism this is used for (see
        ``semantic_title``).  Note that ruleset-*injected* required
        workflows do **not** appear to honour ``edited``, so this is only
        useful for checks the repository or org wires up conventionally.
        """
        await self.patch(
            f"/repos/{owner}/{repo}/pulls/{number}",
            json={"title": title},
        )

    async def approve_pull_request(
        self, owner: str, repo: str, number: int, body: str
    ) -> None:
        """
        Approve a pull request.

        REST: POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews

        This endpoint returns a transient ``500`` with some regularity.
        ``500`` is deliberately absent from ``_is_retryable_status`` --- a
        blanket retry of every failed POST is unsafe --- so the retry is
        handled here, where the operation's semantics are known.

        Crucially, a ``500`` does **not** imply the review was not
        created: in the run analysed in
        ``docs/BULK_RUN_PERFORMANCE_AUDIT.md``, 4 of the 6 PRs whose
        approval "failed" this way went on to merge, which requires the
        approval to have landed.  So before each *retry* --- and once
        more after the final attempt --- the review list is re-read, and
        an approval already present counts as success rather than
        stacking a duplicate review.  The first attempt skips that check,
        so the common path costs exactly one request.

        A ``422`` is classified by GitHub's response body.  One saying
        the review was "submitted too quickly" is GitHub's
        content-creation throttle: it is retried through the same
        duplicate-safe loop, waiting as GitHub advises for secondary
        limits.  One saying the approval is not allowed becomes a
        :class:`PermissionError`; any other is an :class:`ApprovalError`
        carrying GitHub's message.

        Raises:
            PermissionError: If GitHub says the approval is not permitted
            ApprovalError: If GitHub refused the approval for another reason
        """
        target = f"{owner}/{repo}#{number}"
        last_exc: Exception | None = None
        server_failures = 0
        throttled = 0
        while True:
            if last_exc is not None and await self._has_own_approval(
                owner, repo, number
            ):
                self.log.debug(
                    "Approval for %s already present after an error; "
                    "treating as success",
                    target,
                )
                # Reporting success means the approval landed, so the
                # memo must go here too --- not only on the path where
                # the POST itself returned cleanly.
                self.invalidate_block_reason(owner, repo, number)
                return
            try:
                await self.post(
                    f"/repos/{owner}/{repo}/pulls/{number}/reviews",
                    json={"event": "APPROVE", "body": body},
                )
                # The PR's block reason has just changed by construction.
                self.invalidate_block_reason(owner, repo, number)
                return
            except Exception as e:
                perm_error = self._parse_permission_error(e, "approve", owner, repo)
                if perm_error:
                    raise perm_error from e
                status = _http_status(e)
                if status == 422:
                    detail = github_error_detail(e)
                    refusal = ApprovalError(target, status, detail)
                    if not _is_throttled_detail(detail):
                        raise refusal from e
                    refusal.__cause__ = e
                    last_exc = refusal
                    throttled += 1
                    if throttled >= _APPROVE_THROTTLE_MAX_ATTEMPTS:
                        break
                    retry_after = _retry_after_seconds(e)
                    self.content_limiter.throttled(retry_after)
                    delay = (
                        retry_after
                        if retry_after is not None
                        else _pkg._APPROVE_THROTTLE_BASE_DELAY * (2 ** (throttled - 1))
                    )
                    self.log.warning(
                        "GitHub throttled the approval of %s (attempt %d/%d): "
                        "%s; retrying in %.0fs",
                        target,
                        throttled,
                        _APPROVE_THROTTLE_MAX_ATTEMPTS,
                        detail,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                if not _pkg._is_transient_server_error(e):
                    raise
                last_exc = e
                server_failures += 1
                if server_failures >= _APPROVE_MAX_ATTEMPTS:
                    break
                delay = _pkg._APPROVE_RETRY_BASE_DELAY * (2 ** (server_failures - 1))
                self.log.warning(
                    "Transient error approving %s (attempt %d/%d); retrying in %.1fs",
                    target,
                    server_failures,
                    _APPROVE_MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)

        # Attempts exhausted.  One last look: the final POST may have
        # created the review despite reporting failure.
        if await self._has_own_approval(owner, repo, number):
            self.log.info(
                "Approval for %s landed despite a reported error",
                target,
            )
            self.invalidate_block_reason(owner, repo, number)
            return
        assert last_exc is not None
        raise last_exc

    async def _has_own_approval(self, owner: str, repo: str, number: int) -> bool:
        """Whether the authenticated user already has an APPROVED review.

        Paginates: a single default page caps at 30 reviews, and missing
        an existing approval on a busy PR would defeat the
        duplicate-suppression this exists for and post another review.
        """
        try:
            login = await self.get_authenticated_user_login()
            if not login:
                # The lookup returns ``None`` on failure rather than
                # raising.  Without this guard a review carrying
                # ``user: null`` yields ``None == None`` and reports an
                # approval that does not exist --- which would stop
                # ``approve_pull_request`` retrying and let it report
                # success having approved nothing.
                self.log.debug(
                    "Cannot confirm existing approval on %s/%s#%s: "
                    "authenticated user unknown",
                    owner,
                    repo,
                    number,
                )
                return False
            async for page in self.get_paginated(
                f"/repos/{owner}/{repo}/pulls/{number}/reviews",
                per_page=100,
            ):
                if not isinstance(page, list):
                    continue
                for review in page:
                    if not isinstance(review, dict):
                        continue
                    if review.get("state") != "APPROVED":
                        continue
                    user = review.get("user") or {}
                    reviewer = user.get("login")
                    if reviewer and reviewer == login:
                        return True
        except Exception as exc:
            self.log.debug(
                "Could not read reviews for %s/%s#%s: %s", owner, repo, number, exc
            )
            return False
        return False

    async def get_pull_request_review_comments(
        self, owner: str, repo: str, number: int
    ) -> list[dict[str, Any]]:
        """
        Get review comments for a pull request.

        REST: GET /repos/{owner}/{repo}/pulls/{pull_number}/comments
        """
        try:
            data = await self.get(f"/repos/{owner}/{repo}/pulls/{number}/comments")
            return data if isinstance(data, list) else []
        except Exception as e:
            # If we can't get review comments, return empty list
            self.log.debug(f"Could not fetch review comments for PR {number}: {e}")
            return []

    async def post_issue_comment(
        self, owner: str, repo: str, number: int, body: str
    ) -> dict[str, Any]:
        """
        Post a comment on an issue or pull request.

        REST: POST /repos/{owner}/{repo}/issues/{issue_number}/comments

        Raises:
            PermissionError: If token lacks required permissions
        """
        try:
            data = await self.post(
                f"/repos/{owner}/{repo}/issues/{number}/comments",
                json={"body": body},
            )
        except Exception as e:
            perm_error = self._parse_permission_error(
                e, f"post a comment on issue or pull request #{number}", owner, repo
            )
            if perm_error:
                raise perm_error from e
            raise
        return data if isinstance(data, dict) else {}

    async def update_branch(self, owner: str, repo: str, number: int) -> None:
        """
        Update a pull request branch (rebase).

        REST: PUT /repos/{owner}/{repo}/pulls/{pull_number}/update-branch

        Raises:
            PermissionError: If token lacks required permissions
        """
        try:
            await self.put(f"/repos/{owner}/{repo}/pulls/{number}/update-branch")
        except Exception as e:
            perm_error = self._parse_permission_error(e, "update_branch", owner, repo)
            if perm_error:
                raise perm_error from e
            raise

    async def close_pull_request(
        self, owner: str, repo: str, number: int
    ) -> dict[str, Any]:
        """
        Close a pull request.

        Args:
            owner: Repository owner
            repo: Repository name
            number: Pull request number

        Returns:
            Updated pull request data

        Raises:
            PermissionError: If token lacks required permissions
        """
        try:
            return await self.patch(
                f"/repos/{owner}/{repo}/pulls/{number}", json={"state": "closed"}
            )
        except Exception as e:
            perm_error = self._parse_permission_error(e, "close", owner, repo)
            if perm_error:
                raise perm_error from e
            raise
