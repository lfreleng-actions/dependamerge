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
from dataclasses import dataclass, field
from typing import (
    Any,
)

# The retry waits and ``_is_transient_server_error`` stay
# attributes of the package rather than names bound here: they were
# module-level attributes of ``dependamerge.github_async`` before the
# split, and callers substitute them there.
import dependamerge.github_async as _pkg

from ..merge_signals import ReviewsUnreadableError
from ..slot_lease import parked_off_dispatch
from ._base import _GitHubAsyncBase
from ._errors import (
    _APPROVE_THROTTLE_MAX_ATTEMPTS,
    ApprovalError,
    _approval_out_of_time,
    _http_status,
    _is_throttled_detail,
    _retry_after_seconds,
    github_error_detail,
)

#: Page size and page ceiling for :meth:`_ReviewsMixin.list_issue_comments`.
ISSUE_COMMENTS_PAGE_SIZE = 100
ISSUE_COMMENTS_MAX_PAGES = 10


@dataclass(frozen=True)
class IssueComments:
    """A pull request's issue comments, and whether all of them were read.

    ``complete`` is False when a request failed, a payload or comment was
    malformed, or the page ceiling was reached, so ``comments`` may lack
    entries.
    """

    comments: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = True


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
        self,
        owner: str,
        repo: str,
        number: int,
        body: str,
        *,
        commit_id: str,
        deadline: float | None = None,
        short_waits: bool = False,
    ) -> None:
        """
        Approve a pull request at a named commit.

        REST: POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews

        ``commit_id`` is the head commit the caller evaluated, so the
        approval names the commit it vouches for rather than whatever
        the head is when GitHub records it.

        This endpoint returns a transient ``500`` with some regularity.
        The transport does not retry a ``500`` on a write --- a blanket
        retry of every failed POST is unsafe --- so a server error (see
        :func:`server_failure`) is retried here, where the operation's
        semantics are known: four attempts, waiting as
        ``_SERVER_ERROR_RETRY_WAITS`` says between them.  ``deadline``,
        the run's ceiling on the event loop's clock, cuts a wait short
        and stops the retries once it has passed, the review lookup
        included.  ``short_waits``, for a run that may not wait
        (``--max-wait 0``), keeps ``_APPROVE_SHORT_RETRY_WAITS``.

        Crucially, a server error does **not** imply the review was not
        created: in the run analysed in
        ``docs/BULK_RUN_PERFORMANCE_AUDIT.md``, 4 of the 6 PRs whose
        approval "failed" this way went on to merge, which requires the
        approval to have landed.  So before each *retry* --- and once
        more after the final attempt --- the review list is re-read, and
        an approval already present counts as success rather than
        stacking a duplicate review.  Only an approval of ``commit_id``
        counts: one of ours at an earlier commit does not vouch for this
        one.  The first attempt skips that check, so the common path
        costs exactly one request.  When that re-read cannot identify the
        authenticated user, or cannot account for every review and finds
        no approval of ours at ``commit_id``, the approval stops with
        :class:`ReviewsUnreadableError` rather than risk a duplicate.

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
            ReviewsUnreadableError: If, after a failed POST, the
                authenticated user is unknown, or the reviews cannot all
                be read and none read is our approval of ``commit_id``
        """
        if not commit_id:
            raise ValueError(
                f"Refusing to approve PR #{number} in {owner}/{repo} "
                "without a head commit"
            )
        target = f"{owner}/{repo}#{number}"
        last_exc: Exception | None = None
        server_failures = 0
        throttled = 0
        while True:
            if last_exc is not None and await self._has_own_approval(
                owner, repo, number, commit_id=commit_id
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
            if last_exc is not None and _approval_out_of_time(last_exc, deadline):
                raise last_exc  # the lookup above found no approval
            try:
                await self.post(
                    f"/repos/{owner}/{repo}/pulls/{number}/reviews",
                    json={
                        "event": "APPROVE",
                        "body": body,
                        "commit_id": commit_id,
                    },
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
                if not await self._wait_after_approval_server_error(
                    target, server_failures, e, deadline, short_waits
                ):
                    break

        # Attempts exhausted.  One last look: the final POST may have
        # created the review despite reporting failure.
        if await self._has_own_approval(owner, repo, number, commit_id=commit_id):
            self.log.info(
                "Approval for %s landed despite a reported error",
                target,
            )
            self.invalidate_block_reason(owner, repo, number)
            return
        assert last_exc is not None
        raise last_exc

    async def _wait_after_approval_server_error(
        self,
        target: str,
        failures: int,
        error: Exception,
        deadline: float | None,
        short_waits: bool,
    ) -> bool:
        """Wait out a server error before another approval; ``False`` to stop."""
        waits = (
            _pkg._APPROVE_SHORT_RETRY_WAITS
            if short_waits
            else _pkg._SERVER_ERROR_RETRY_WAITS
        )
        if failures > len(waits):
            return False
        delay = waits[failures - 1]
        if deadline is not None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            delay = min(delay, remaining)
        self.log.warning(
            "Server error approving %s (attempt %d/%d); retrying in %.1fs",
            target,
            failures,
            len(waits) + 1,
            delay,
        )
        self.log.debug("Approval error for %s: %s", target, error)
        # Waiting on GitHub, so the worker's slot is free meanwhile, and
        # the dispatch lock when a merge retry reached this approval.
        async with parked_off_dispatch():
            await asyncio.sleep(delay)
        # The wait, or taking the slot and lock back, may spend the budget.
        return deadline is None or asyncio.get_running_loop().time() < deadline

    async def _has_own_approval(
        self, owner: str, repo: str, number: int, *, commit_id: str
    ) -> bool:
        """Whether the authenticated user already approved ``commit_id``.

        An approval of ours at another commit does not count: the head
        may have moved since, and that review vouches for a different
        commit than the one this request names.

        Reads every page: missing an existing approval on a busy PR would
        defeat the duplicate-suppression this exists for and post another
        review.  An approval found in a partial reading still counts.

        Raises:
            ReviewsUnreadableError: When the authenticated user cannot be
                identified, or the reading is partial and holds no
                approval of ours at ``commit_id``: the failed POST may have created one
                that cannot be recognised, so retrying could post a
                duplicate.
        """
        try:
            login = await self.get_authenticated_user_login()
        except Exception as exc:
            self.log.debug(
                "Could not read the authenticated user for %s/%s#%s: %s",
                owner,
                repo,
                number,
                exc,
            )
            raise ReviewsUnreadableError(f"{owner}/{repo}#{number}") from exc
        if not login:
            # The lookup returns ``None`` on failure rather than
            # raising.  Matching on it would let a review carrying
            # ``user: null`` yield ``None == None`` and report an
            # approval that does not exist; reporting none instead
            # would let a retry post a duplicate.  Neither is safe.
            self.log.debug(
                "Cannot confirm existing approval on %s/%s#%s: "
                "authenticated user unknown",
                owner,
                repo,
                number,
            )
            raise ReviewsUnreadableError(f"{owner}/{repo}#{number}")
        reading = await self.read_pull_request_reviews(owner, repo, number)
        if any(
            review.get("state") == "APPROVED"
            and (review.get("user") or {}).get("login") == login
            and review.get("commit_id") == commit_id
            for review in reading.entries
        ):
            return True
        if not reading.complete:
            raise ReviewsUnreadableError(f"{owner}/{repo}#{number}")
        return False

    async def list_issue_comments(
        self, owner: str, repo: str, number: int
    ) -> IssueComments:
        """Read every issue comment on a pull request, oldest first.

        REST: GET /repos/{owner}/{repo}/issues/{issue_number}/comments

        The endpoint has no ``direction`` parameter: it always answers in
        ascending order, so the newest comments sit on the *last* page.
        Reading stops at ``ISSUE_COMMENTS_MAX_PAGES``; a full final page
        there, a failed request, an unexpected payload or a comment that
        is not an object leaves the reading incomplete, and callers
        decide what an unknown tail means.
        """
        comments: list[dict[str, Any]] = []
        pages = 0
        last_full = False
        malformed = False
        try:
            async for page in self.get_paginated(
                f"/repos/{owner}/{repo}/issues/{number}/comments",
                per_page=ISSUE_COMMENTS_PAGE_SIZE,
                max_pages=ISSUE_COMMENTS_MAX_PAGES,
                yield_malformed=True,
            ):
                if not isinstance(page, list):
                    return IssueComments(comments, complete=False)
                pages += 1
                for comment in page:
                    if isinstance(comment, dict):
                        comments.append(comment)
                    else:
                        malformed = True
                last_full = len(page) >= ISSUE_COMMENTS_PAGE_SIZE
        except Exception as exc:
            self.log.debug(
                "Could not read every comment on %s/%s#%s: %s",
                owner,
                repo,
                number,
                exc,
            )
            return IssueComments(comments, complete=False)
        complete = not malformed and not (
            pages >= ISSUE_COMMENTS_MAX_PAGES and last_full
        )
        return IssueComments(comments, complete=complete)

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
