# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Completing a scan page's short files or comments list for one PR.

The open pull request page cuts its nested lists to a fixed size.  Before
a short list is used, the pull request's whole list is read over REST;
when that read fails part-way, what the page and the read hold together
is used, and a warning names the pull request.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..file_reading import read_pr_files
from ..models import FileChange, PullRequestInfo
from ._pr_extract import scan_comments, scan_files, scan_files_total

if TYPE_CHECKING:
    from ..github_async import GitHubAsync

log = logging.getLogger(__name__)


def _union(read: list[FileChange], page: list[FileChange]) -> list[FileChange]:
    """The files read, then those only the page holds."""
    names = {change.filename for change in read}
    return read + [change for change in page if change.filename not in names]


async def complete_pr_files(api: GitHubAsync, pr: PullRequestInfo) -> PullRequestInfo:
    """*pr* with every changed file, when its scan page held only some."""
    if pr.files_complete:
        return pr
    owner, _, repo = pr.repository_full_name.partition("/")
    reading = await read_pr_files(api, owner, repo, pr.number, pr.files_total)
    files = (
        reading.files if reading.complete else _union(reading.files, pr.files_changed)
    )
    completed: PullRequestInfo = pr.model_copy(
        update={"files_changed": files, "files_complete": reading.complete}
    )
    return completed


async def complete_scan_files(
    api: GitHubAsync, owner: str, repo: str, pr: dict[str, Any]
) -> list[dict[str, Any]]:
    """A PR node's changed files as ``{"path": ...}`` nodes, completed."""
    nodes, complete = scan_files(pr)
    if complete:
        return nodes
    number = pr.get("number")
    if not isinstance(number, int):
        return nodes
    reading = await read_pr_files(api, owner, repo, number, scan_files_total(pr))
    paths = [change.filename for change in reading.files]
    if reading.complete:
        return [{"path": path} for path in paths]
    seen = set(paths)
    paths += [
        path
        for node in nodes
        if isinstance(path := node.get("path"), str) and path not in seen
    ]
    return [{"path": path} for path in paths]


def _comment_keys(
    comment: dict[str, Any],
) -> tuple[int | None, tuple[Any, Any, Any]]:
    """A page-shaped comment's numeric ID, and its author, time and body.

    The page selects ``databaseId``, which is the REST ``id``; the second
    key matches a page comment that carries no ID.
    """
    number = comment.get("databaseId", comment.get("id"))
    content = (
        (comment.get("author") or {}).get("login"),
        comment.get("createdAt"),
        comment.get("body"),
    )
    numeric = isinstance(number, int) and not isinstance(number, bool)
    return (number if numeric else None), content


def _page_shaped(
    comments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """REST comments in the page's shape, each ID once, and whether all were.

    A comment without a numeric ``id`` or a ``user`` object with a login,
    or one whose ``id`` was already read, cannot be trusted or matched to
    the page: it is left out and the read counts as incomplete.
    """
    read: list[dict[str, Any]] = []
    seen: set[int] = set()
    whole = True
    for comment in comments:
        number = comment.get("id")
        user = comment.get("user")
        login = user.get("login") if isinstance(user, dict) else None
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not isinstance(login, str)
            or not login
            or number in seen
        ):
            whole = False
            continue
        seen.add(number)
        read.append(
            {
                "id": number,
                "author": {"login": login},
                "body": comment.get("body"),
                "createdAt": comment.get("created_at"),
            }
        )
    return read, whole


async def complete_scan_comments(
    api: GitHubAsync, repo_full_name: str, pr: dict[str, Any]
) -> list[dict[str, Any]]:
    """A PR node's comments in the page's shape, completed.

    REST reads the oldest comments first and the page holds the most
    recently updated, so a partial read keeps both, each comment once.
    A read holding a malformed or repeated comment counts as partial.
    """
    nodes, complete = scan_comments(pr)
    number = pr.get("number")
    if complete or not isinstance(number, int):
        return nodes
    owner, _, repo = repo_full_name.partition("/")
    reading = await api.list_issue_comments(owner, repo, number)
    read, whole = _page_shaped(reading.comments)
    if reading.complete and whole:
        return read
    log.warning(
        "Comments on %s#%d could not all be read (%d read)",
        repo_full_name,
        number,
        len(read),
    )
    seen_ids: set[int] = set()
    seen_content: set[tuple[Any, Any, Any]] = set()
    for comment in read:
        comment_id, content = _comment_keys(comment)
        if comment_id is not None:
            seen_ids.add(comment_id)
        seen_content.add(content)
    for node in nodes:
        node_id, content = _comment_keys(node)
        if node_id is not None:
            duplicate = node_id in seen_ids
        else:
            duplicate = content in seen_content
        if not duplicate:
            read.append(node)
    return read
