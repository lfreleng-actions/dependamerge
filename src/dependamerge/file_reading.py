# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Reading a pull request's changed files in full over REST."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from .models import FileChange

if TYPE_CHECKING:
    from .github_async import GitHubAsync

# The pull request files endpoint returns at most this many files.
PR_FILES_CAP = 3000


def _distinct(changes: list[FileChange]) -> list[FileChange]:
    """*changes* with the first entry for each filename, in order."""
    first: dict[str, FileChange] = {}
    for change in changes:
        first.setdefault(change.filename, change)
    return list(first.values())


class FileReading(NamedTuple):
    """A pull request's changed files, and whether all of them were read."""

    files: list[FileChange]
    complete: bool


async def read_pr_files(
    api: GitHubAsync,
    owner: str,
    repo: str,
    pr_number: int,
    expected: int | None = None,
) -> FileReading:
    """Read a pull request's changed files, tolerating pagination failure.

    A failed read, or a page or entry of an unexpected shape (an entry
    without a filename among them), keeps the files read before it, logs
    a warning naming the pull request, and is marked incomplete.  So is a
    read that reached the endpoint's 3,000-file cap holding fewer
    distinct files than ``expected``, the pull request's known file
    count: at the cap the endpoint ends its pagination as if the list
    were whole.  Below the cap a whole read stands even when ``expected``
    is larger, since that count may predate a force-push that removed
    files.  A filename read twice means the list shifted between pages,
    so another file may have gone unread: the read keeps the first entry
    for each filename, warns, and is incomplete.
    """
    files_changed: list[FileChange] = []
    try:
        async for page in api.get_paginated(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/files",
            per_page=100,
            yield_malformed=True,
        ):
            if not isinstance(page, list):
                raise ValueError(f"unexpected files page: {type(page).__name__}")
            for file_data in page:
                if not isinstance(file_data, dict):
                    raise ValueError(
                        f"unexpected file entry: {type(file_data).__name__}"
                    )
                filename = file_data.get("filename")
                if not isinstance(filename, str) or not filename:
                    raise ValueError(f"file entry without a filename: {filename!r}")
                files_changed.append(
                    FileChange(
                        filename=filename,
                        additions=int(file_data.get("additions", 0)),
                        deletions=int(file_data.get("deletions", 0)),
                        changes=int(
                            file_data.get(
                                "changes",
                                (file_data.get("additions", 0) or 0)
                                + (file_data.get("deletions", 0) or 0),
                            )
                        ),
                        status=file_data.get("status", "modified"),
                    )
                )
    except Exception as exc:
        # The transport has already retried; keep what was read, but say
        # the list is incomplete.
        api.log.warning(
            "File list for %s/%s#%d is incomplete (%d read): %s",
            owner,
            repo,
            pr_number,
            len(files_changed),
            exc,
        )
        api.log.debug("File pagination failure detail", exc_info=True)
        return FileReading(_distinct(files_changed), False)
    files = _distinct(files_changed)
    if len(files) < len(files_changed):
        api.log.warning(
            "File list for %s/%s#%d changed while it was read (%d repeated)",
            owner,
            repo,
            pr_number,
            len(files_changed) - len(files),
        )
        return FileReading(files, False)
    read = len(files)
    capped = len(files_changed) >= PR_FILES_CAP
    if capped and expected is not None and read < expected:
        api.log.warning(
            "File list for %s/%s#%d is incomplete (%d of %d read)",
            owner,
            repo,
            pr_number,
            read,
            expected,
        )
        return FileReading(files_changed, False)
    return FileReading(files_changed, True)
