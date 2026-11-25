# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Field-level extraction from PR GraphQL nodes."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import Any, NamedTuple

from ..bot_identity import is_copilot
from ..check_runs import (
    FAILING_CONCLUSIONS,
    FAILING_STATUS_STATES,
    failing_check_names,
    failing_status_names,
)
from ..github_graphql.pull_requests import (
    SCAN_CHECKS_PAGE_SIZE,
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


def scan_files_total(pr: dict[str, Any]) -> int | None:
    """How many files a PR node's ``files`` connection says it changes."""
    connection = _selected(pr, "files")
    if not isinstance(connection, dict):
        return None
    return _int_or_none(connection.get("totalCount"))


def scan_comments(pr: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """A PR node's comments, and whether they are all of them."""
    return _scan_list(pr, "comments", SCAN_COMMENTS_PAGE_SIZE)


def scan_reviews(pr: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """A PR node's reviews, and whether they are all of them."""
    return _scan_list(pr, "reviews", SCAN_REVIEWS_PAGE_SIZE)


def scan_reviews_complete(pr: dict[str, Any]) -> bool:
    """Whether a PR node's ``reviews`` connection holds every review."""
    return scan_reviews(pr)[1]


class CheckPage(NamedTuple):
    """The check rollup page of a PR node's latest commit.

    ``failing`` counts the check runs and status contexts in a failing
    state across the whole rollup, from its per-state counts; ``None``
    when they are unreadable.  ``complete`` says the page holds every
    entry, and ``oid`` names the commit.
    """

    contexts: list[dict[str, Any]]
    failing: int | None
    complete: bool
    oid: str | None


def _failing_count(connection: dict[str, Any]) -> int | None:
    """Rollup entries in a failing state, from the connection's counts."""
    failing = 0
    for key, states in (
        ("checkRunCountsByState", FAILING_CONCLUSIONS),
        ("statusContextCountsByState", FAILING_STATUS_STATES),
    ):
        entries = connection.get(key)
        if not isinstance(entries, list):
            return None
        for entry in entries:
            state = entry.get("state") if isinstance(entry, dict) else None
            count = _int_or_none(entry.get("count")) if state else None
            if not isinstance(state, str) or count is None:
                return None
            if state.lower() in states:
                failing += count
    return failing


def scan_check_page(pr: dict[str, Any]) -> CheckPage:
    """A PR node's check rollup page, and what its counts say of the rest.

    A commit with no rollup has no checks, and no failing one.
    """
    commits = (_selected(pr, "commits") or {}).get("nodes") or []
    commit = ((commits[0] or {}).get("commit") or {}) if commits else {}
    oid = commit.get("oid") if isinstance(commit.get("oid"), str) else None
    rollup = commit.get("statusCheckRollup")
    if not isinstance(rollup, dict):
        return CheckPage([], 0, True, oid)
    connection = rollup.get("contexts") or {}
    nodes = [node for node in connection.get("nodes") or [] if isinstance(node, dict)]
    runs = _int_or_none(connection.get("checkRunCount"))
    statuses = _int_or_none(connection.get("statusContextCount"))
    total = None if runs is None or statuses is None else runs + statuses
    complete = _page_complete(len(nodes), total, SCAN_CHECKS_PAGE_SIZE)
    return CheckPage(nodes, _failing_count(connection), complete, oid)


def split_check_contexts(
    contexts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A rollup page's check runs and its status contexts, apart."""
    runs = [ctx for ctx in contexts if ctx.get("__typename") == "CheckRun"]
    statuses = [ctx for ctx in contexts if ctx.get("__typename") == "StatusContext"]
    return runs, statuses


def find_copilot_comments(comments: list[dict[str, Any]]) -> list[CopilotComment]:
    """The Copilot comments among page-shaped comment nodes."""
    result: list[CopilotComment] = []
    for c in comments:
        if is_copilot((c.get("author") or {}).get("login")):
            comment_id = _int_or_none(c.get("id", c.get("databaseId")))
            result.append(
                CopilotComment(
                    id=comment_id or 0,
                    body=c.get("body") or "",
                    created_at=c.get("createdAt") or "",
                    state="open",
                )
            )
    return result


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
        return find_copilot_comments(comments)

    @staticmethod
    def _extract_failing_checks(pr: dict[str, Any]) -> list[str]:
        """
        Extract failing checks from the statusCheckRollup on the latest commit.

        A commit can carry several check runs sharing one name, typically
        when a duplicate workflow event causes ``concurrency`` to cancel a
        superseded run.  Only the latest run per name decides the outcome,
        matching GitHub's own merge protection; see ``check_runs``.
        """
        runs, statuses = split_check_contexts(scan_check_page(pr).contexts)
        return failing_check_names(runs) + failing_status_names(statuses)
