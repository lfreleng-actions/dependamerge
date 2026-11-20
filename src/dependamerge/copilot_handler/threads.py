# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Review-thread inspection and resolution for Copilot feedback.

:class:`_CopilotThreadMixin` carries the thread-level half of
``CopilotCommentHandler``.  GitHub refuses to dismiss a COMMENTED review,
so the only way to clear that feedback is to walk the pull request's
review threads, decide which ones are Copilot's and safe to close, and
resolve them individually.

It is a mixin rather than a separate collaborator so the handler's method
surface stays exactly as it was before this module existed.  Every
attribute it reads is established by ``CopilotCommentHandler.__init__``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
from typing import Any

from ..bot_identity import is_copilot

# Phrases typical of automation suggestions that are safe to resolve
SAFE_PATTERNS = (
    "use: ubuntu-24.04",
    "consider using",
    "you might want to",
    "suggestion:",
    "performance:",
    "style:",
    "formatting",
    "indentation",
    "whitespace",
)

# Keywords that might need human attention; any one keeps a thread open
UNSAFE_PATTERNS = (
    "security",
    "vulnerability",
    "critical",
    "error",
    "bug",
    "broken",
    "incorrect",
)


class _CopilotThreadMixin:
    """Thread-level Copilot handling shared into ``CopilotCommentHandler``."""

    # Established by CopilotCommentHandler.__init__.
    github_client: Any
    preview_mode: bool
    log: logging.Logger

    async def get_pr_review_threads(
        self, owner: str, repo: str, pr_number: int
    ) -> list[dict[str, Any]]:
        """
        Get all review threads for a pull request.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            List of review thread data
        """
        from ..github_graphql import GET_PR_REVIEW_THREADS

        threads = []
        cursor = None
        has_next = True

        while has_next:
            variables = {
                "owner": owner,
                "name": repo,
                "number": pr_number,
                "cursor": cursor,
            }

            result = await self.github_client.graphql(GET_PR_REVIEW_THREADS, variables)

            if (
                not result
                or not result.get("repository")
                or not result["repository"].get("pullRequest")
            ):
                self.log.error(
                    f"❌ Invalid GraphQL response structure for threads: {result}"
                )
                break

            pr_data = result["repository"]["pullRequest"]

            review_threads = pr_data["reviewThreads"]
            nodes = review_threads["nodes"]
            threads.extend(nodes)

            page_info = review_threads["pageInfo"]
            has_next = page_info["hasNextPage"]
            cursor = page_info["endCursor"]

        return threads

    def is_copilot_thread(self, thread: dict[str, Any]) -> bool:
        """
        Check if Copilot opened a review thread.

        Only the author of the thread's first comment counts: a reply by
        Copilot, or a body that mentions Copilot, does not make a human
        reviewer's thread Copilot's.

        Args:
            thread: Review thread data

        Returns:
            True if Copilot wrote the thread's first comment
        """
        comments = (thread.get("comments") or {}).get("nodes") or []
        if not comments:
            return False
        author = (comments[0] or {}).get("author") or {}
        return is_copilot(author.get("login"))

    async def read_all_thread_comments(self, thread: dict[str, Any]) -> bool:
        """
        Read every page of a review thread's comments into ``thread``.

        The comments fetched with the thread are only the first page.  The
        remaining pages are appended to ``thread["comments"]["nodes"]`` and
        its ``pageInfo`` is updated to match, so
        :meth:`is_safe_copilot_thread_to_resolve` can tell a complete read
        from a partial one.

        Args:
            thread: Review thread data, updated in place

        Returns:
            True if every comment of the thread has been read
        """
        from ..github_graphql import GET_REVIEW_THREAD_COMMENTS

        thread_id = thread.get("id")
        comments = thread.get("comments")
        if not isinstance(comments, dict):
            self.log.debug(f"Thread {thread_id} carries no comments object")
            return False
        nodes = comments.get("nodes")
        page_info = comments.get("pageInfo")
        if not isinstance(nodes, list) or not isinstance(page_info, dict):
            self.log.debug(f"Thread {thread_id} carries no comment page data")
            return False

        seen_cursors: set[str] = set()
        while page_info.get("hasNextPage"):
            cursor = page_info.get("endCursor")
            if not thread_id or not cursor or cursor in seen_cursors:
                self.log.debug(
                    f"Cannot read further comments of thread {thread_id}: "
                    f"no usable cursor ({cursor!r})"
                )
                return False
            seen_cursors.add(cursor)
            try:
                result = await self.github_client.graphql(
                    GET_REVIEW_THREAD_COMMENTS,
                    {"threadId": thread_id, "cursor": cursor},
                )
                page = result["node"]["comments"]
                page_nodes = page["nodes"]
                next_info = page["pageInfo"]
                if not isinstance(page_nodes, list) or not isinstance(next_info, dict):
                    raise TypeError("malformed comments page")
            except Exception as e:
                self.log.debug(
                    f"Failed to read further comments of thread {thread_id}: {e}"
                )
                return False
            nodes.extend(page_nodes)
            page_info = next_info
            comments["pageInfo"] = page_info

        return True

    def is_safe_copilot_thread_to_resolve(self, thread: dict[str, Any]) -> bool:
        """
        Check if a Copilot thread is safe to auto-resolve.

        A thread is not safe when it is already resolved, when its
        comments have not all been read, when any comment is not an object
        with a string body, or when any comment contains an unsafe keyword;
        the keyword test applies to outdated threads too.
        Every other thread is safe.

        Args:
            thread: Review thread data

        Returns:
            True if safe to resolve automatically
        """
        thread_id = thread.get("id")

        if thread.get("isResolved", False):
            return False  # Already resolved

        comments_data = thread.get("comments") or {}
        page_info = comments_data.get("pageInfo")
        comments = comments_data.get("nodes")
        if (
            not isinstance(page_info, dict)
            or page_info.get("hasNextPage") is not False
            or not isinstance(comments, list)
        ):
            self.log.debug(
                f"Thread {thread_id} not safe to resolve: its comments were not all read"
            )
            return False

        bodies: list[str] = []
        for comment in comments:
            body = comment.get("body") if isinstance(comment, dict) else None
            if not isinstance(body, str):
                self.log.debug(
                    f"Thread {thread_id} not safe to resolve: a comment has no readable body"
                )
                return False
            bodies.append(body.lower())

        for body in bodies:
            unsafe = [p for p in UNSAFE_PATTERNS if p in body]
            if unsafe:
                self.log.debug(
                    f"Thread {thread_id} not safe to resolve: unsafe keyword {unsafe[0]!r}"
                )
                return False

        if thread.get("isOutdated", False):
            return True  # Outdated threads are usually safe to resolve

        if any(p in body for body in bodies for p in SAFE_PATTERNS):
            self.log.debug(f"Thread {thread_id} matches a safe Copilot phrase")

        # Default to safe for general Copilot suggestions
        return True

    async def resolve_review_thread(self, thread_id: str, pr_context: str = "") -> bool:
        """
        Resolve a specific review thread.

        Args:
            thread_id: GraphQL ID of the thread to resolve

        Returns:
            True if successfully resolved
        """
        if self.preview_mode:
            context = f" for {pr_context}" if pr_context else ""
            self.log.info(
                f"🔍 PREVIEW: Would resolve review thread {thread_id}{context}"
            )
            return True

        from ..github_graphql import RESOLVE_REVIEW_THREAD

        try:
            variables = {"threadId": thread_id}
            result = await self.github_client.graphql(RESOLVE_REVIEW_THREAD, variables)

            if result and result.get("resolveReviewThread"):
                thread_data = result["resolveReviewThread"]["thread"]
                if thread_data.get("isResolved"):
                    context = f" for {pr_context}" if pr_context else ""
                    self.log.info(f"✅ Resolved review thread {thread_id}{context}")
                    return True
                else:
                    context = f" for {pr_context}" if pr_context else ""
                    self.log.error(
                        f"❌ Thread {thread_id}{context} not marked as resolved in response: {thread_data}"
                    )

            context = f" for {pr_context}" if pr_context else ""
            self.log.error(
                f"❌ Failed to resolve review thread {thread_id}{context}. Full response: {result}"
            )
            if result and result.get("errors"):
                self.log.error(
                    f"❌ GraphQL errors for {thread_id}{context}: {result['errors']}"
                )
            return False

        except Exception as e:
            context = f" for {pr_context}" if pr_context else ""
            self.log.error(
                f"❌ Error resolving review thread {thread_id}{context}: {e}"
            )
            return False

    async def resolve_copilot_threads_for_commented_review(
        self, owner: str, repo: str, pr_number: int, review_id: str
    ) -> tuple[int, int]:
        """
        For COMMENTED reviews that can't be dismissed, try to resolve individual threads.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            review_id: Review ID (for logging context)

        Returns:
            Tuple of (resolved_count, total_copilot_threads)
        """
        self.log.info(
            f"🧵 Attempting thread-level resolution for COMMENTED review {review_id}"
        )

        all_threads = await self.get_pr_review_threads(owner, repo, pr_number)

        # Filter for unresolved Copilot threads that are safe to resolve
        copilot_threads = []
        for thread in all_threads:
            if not self.is_copilot_thread(thread) or thread.get("isResolved", False):
                continue
            if not await self.read_all_thread_comments(thread):
                self.log.debug(
                    f"Skipping thread {thread.get('id')}: its comments could not all be read"
                )
                continue
            if self.is_safe_copilot_thread_to_resolve(thread):
                copilot_threads.append(thread)

        if not copilot_threads:
            self.log.warning(
                f"⚠️ Failed to resolve comment/review thread {review_id} in {owner}/{repo}#{pr_number} (no resolvable Copilot threads)"
            )
            return 0, len(all_threads)

        self.log.info(
            f"🎯 Found {len(copilot_threads)} resolvable Copilot threads out of {len(all_threads)} total for {owner}/{repo}#{pr_number}"
        )

        resolved_count = 0
        for i, thread in enumerate(copilot_threads, 1):
            thread_id = thread["id"]

            path = thread.get("path", "unknown")
            line = thread.get("line", "unknown")
            self.log.info(
                f"🔍 Resolving thread {i}/{len(copilot_threads)} in {owner}/{repo}#{pr_number}: {thread_id} on {path}:{line}"
            )

            if await self.resolve_review_thread(
                thread_id, f"{owner}/{repo}#{pr_number}"
            ):
                resolved_count += 1
                self.log.info(
                    f"✅ Successfully resolved thread {i}/{len(copilot_threads)} in {owner}/{repo}#{pr_number}"
                )
            else:
                self.log.error(
                    f"❌ Failed to resolve thread {i}/{len(copilot_threads)} in {owner}/{repo}#{pr_number}: {thread_id}"
                )

        self.log.info(
            f"📊 Resolved {resolved_count}/{len(copilot_threads)} Copilot threads for review {review_id} in {owner}/{repo}#{pr_number}"
        )
        return resolved_count, len(copilot_threads)
