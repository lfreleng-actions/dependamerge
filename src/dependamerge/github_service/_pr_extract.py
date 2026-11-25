# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Field-level extraction from PR GraphQL nodes."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import Any

from ..check_runs import failing_check_names
from ..github_graphql.pull_requests import (
    SCAN_COMMENTS_PAGE_SIZE,
    SCAN_FILES_PAGE_SIZE,
    SCAN_REVIEWS_PAGE_SIZE,
)
from ..models import (
    CopilotComment,
    FileChange,
    ReviewInfo,
)
from ._base import _GitHubServiceBase


class ListNotSelectedError(LookupError):
    """A scan read a nested list its page query did not select.

    The page query omits an unselected list from each node.  Reading
    that absence as an empty list would be a silent wrong answer, so the
    extractors raise instead.
    """


def _selected(pr: dict[str, Any], key: str) -> Any:
    """The node's ``key`` field, which the page query must have selected."""
    if key not in pr:
        raise ListNotSelectedError(
            f"Pull request node #{pr.get('number', '?')} carries no {key!r}: "
            "the scan did not select that list"
        )
    return pr[key]


def _int_or_none(value: Any) -> int | None:
    """``value`` when it is an integer (not a bool), else ``None``."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _page_complete(read: int, total: int | None, page_size: int) -> bool:
    """Whether a page of ``read`` entries is the whole list.

    The list's total decides when there is one, since a page can come
    back short of it with an entry that could not be read.  Without a
    total, a page short of the size the query selects is the whole list.
    """
    if total is not None:
        return read >= total
    return read < page_size


def _scan_list(
    pr: dict[str, Any], key: str, page_size: int
) -> tuple[list[dict[str, Any]], bool]:
    """A selected connection's entries, and whether they are the whole list.

    A connection GitHub answered with ``null`` holds nothing that was
    read, so it is not known to be whole.
    """
    connection = _selected(pr, key)
    if not isinstance(connection, dict):
        return [], False
    nodes = [node for node in connection.get("nodes") or [] if isinstance(node, dict)]
    total = _int_or_none(connection.get("totalCount"))
    return nodes, _page_complete(len(nodes), total, page_size)


def scan_files(pr: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """A PR node's changed files, and whether they are all of them."""
    return _scan_list(pr, "files", SCAN_FILES_PAGE_SIZE)


def scan_comments(pr: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """A PR node's comments, and whether they are all of them."""
    return _scan_list(pr, "comments", SCAN_COMMENTS_PAGE_SIZE)


def scan_reviews(pr: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """A PR node's reviews, and whether they are all of them."""
    return _scan_list(pr, "reviews", SCAN_REVIEWS_PAGE_SIZE)


def scan_reviews_complete(pr: dict[str, Any]) -> bool:
    """Whether a PR node's ``reviews`` connection holds every review."""
    return scan_reviews(pr)[1]


def scan_check_contexts(pr: dict[str, Any]) -> list[dict[str, Any]]:
    """The check runs and status contexts on a PR node's latest commit page.

    An empty list when the commit carries no rollup.
    """
    commits = (_selected(pr, "commits") or {}).get("nodes") or []
    if not commits:
        return []
    commit = (commits[0] or {}).get("commit") or {}
    rollup = commit.get("statusCheckRollup") or {}
    return (rollup.get("contexts") or {}).get("nodes") or []


class _PullRequestExtractMixin(_GitHubServiceBase):
    """PR-node field extraction for ``GitHubService``."""

    def _map_mergeable_enum(self, value: str | None) -> bool | None:
        # GraphQL mergeable: "MERGEABLE" | "CONFLICTING" | "UNKNOWN"
        self.log.debug(f"Mapping mergeable enum: '{value}'")
        if not value:
            self.log.debug("mergeable value is falsy (None, empty, etc.)")
            return None
        v = value.upper()
        if v == "MERGEABLE":
            self.log.debug("Mapped to True (mergeable)")
            return True
        if v == "CONFLICTING":
            self.log.debug("Mapped to False (conflicting)")
            return False
        if v == "UNKNOWN":
            # GitHub is still calculating - treat as potentially mergeable
            self.log.debug("Mapped UNKNOWN to None (still calculating)")
            return None
        self.log.warning(f"Unexpected mergeable value from GraphQL: {value}")
        return None

    def _safe_get_merge_state(self, merge_state_status: str | None) -> str | None:
        """Safely extract and normalize mergeStateStatus from GraphQL."""
        if not merge_state_status:
            # Log when we get null/missing mergeStateStatus for debugging
            self.log.debug("GraphQL mergeStateStatus is null or missing")
            return None

        normalized = merge_state_status.lower().strip()
        if not normalized:
            self.log.debug("GraphQL mergeStateStatus is empty string")
            return None

        # Valid states: clean, dirty, blocked, behind, draft, unstable, unknown
        valid_states = {
            "clean",
            "dirty",
            "blocked",
            "behind",
            "draft",
            "unstable",
            "unknown",
        }
        if normalized not in valid_states:
            self.log.warning(
                f"Unexpected mergeStateStatus from GraphQL: {merge_state_status}"
            )

        return normalized

    def _extract_file_changes(self, pr: dict[str, Any]) -> list[FileChange]:
        files, _ = scan_files(pr)
        result: list[FileChange] = []
        for f in files:
            additions = int(f.get("additions") or 0)
            deletions = int(f.get("deletions") or 0)
            result.append(
                FileChange(
                    filename=f.get("path") or "",
                    additions=additions,
                    deletions=deletions,
                    changes=additions + deletions,
                    status="modified",  # GraphQL 'files' doesn't include a status; best-effort
                )
            )
        return result

    def _extract_reviews(self, pr: dict[str, Any]) -> list[ReviewInfo]:
        """Extract PR reviews from GraphQL node."""
        reviews, _ = scan_reviews(pr)
        result: list[ReviewInfo] = []

        for review in reviews:
            author = (review.get("author") or {}).get("login") or "unknown"
            result.append(
                ReviewInfo(
                    # NOTE: GraphQL returns string node IDs (e.g., "PRR_kwDOGBtQpc4-u-zD")
                    # NOT numeric IDs. Do not convert to int() - it will cause runtime errors.
                    id=review.get("id", ""),
                    user=author,
                    state=review.get("state") or "",
                    submitted_at=review.get("createdAt") or "",
                    body=review.get("body"),
                )
            )
        return result

    def _extract_copilot_comments(self, pr: dict[str, Any]) -> list[CopilotComment]:
        comments, _ = scan_comments(pr)
        result: list[CopilotComment] = []
        for c in comments:
            author = ((c.get("author") or {}).get("login") or "").lower()
            if author in ("github-copilot[bot]", "copilot"):
                result.append(
                    CopilotComment(
                        id=0,  # GraphQL doesn't provide numeric IDs in this selection; not critical for reporting
                        body=c.get("body") or "",
                        created_at=c.get("createdAt") or "",
                        state="open",
                    )
                )
        return result

    @staticmethod
    def _extract_failing_checks(pr: dict[str, Any]) -> list[str]:
        """
        Extract failing checks from the statusCheckRollup on the latest commit.

        A commit can carry several check runs sharing one name, typically
        when a duplicate workflow event causes ``concurrency`` to cancel a
        superseded run.  Only the latest run per name decides the outcome,
        matching GitHub's own merge protection; see ``check_runs``.
        """
        failing: list[str] = []
        contexts = scan_check_contexts(pr)

        check_runs: list[dict[str, Any]] = []

        for ctx in contexts:
            typ = ctx.get("__typename")
            if typ == "CheckRun":
                # Collected for deduplication rather than judged here.
                check_runs.append(ctx)
            elif typ == "StatusContext":
                # Commit statuses are already latest-per-context: GitHub
                # collapses repeated posts to the same context itself.
                state = (ctx.get("state") or "").upper()
                if state in ("FAILURE", "ERROR"):
                    name = ctx.get("context") or ""
                    if name:
                        failing.append(name)

        return failing_check_names(check_runs) + failing
