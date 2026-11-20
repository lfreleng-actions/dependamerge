# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Paginated reads of a commit's check runs and status contexts.

One paging loop serves both listings, in a strict form that raises
unless the reading is whole and a tolerant form that keeps a partial
reading and says so.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import Any, NamedTuple

from ._base import _GitHubAsyncBase


class CommitReading(NamedTuple):
    """Entries read from a commit's check-run or status listing.

    ``complete`` says the reading accounts for every entry on the
    commit, so **absence** from it is evidence.  A partial reading still
    carries what it found: finding an entry is conclusive however the
    read ended, and only not finding one depends on having read it all.
    """

    entries: list[dict[str, Any]]
    complete: bool


class IncompleteReadingError(RuntimeError):
    """A commit listing could not be read whole.

    Raised by the strict readers, whose callers reason about absence:
    returning a partial listing there would read as an all-clear for
    whatever went unread.
    """


# Fields each listing's consumers classify by: those an entry must carry
# as a nonblank string, and those that may also be null (a pending run
# has no conclusion yet).  An entry that breaks either rule is discarded,
# and the reading marked incomplete, rather than let it abort
# classification or stand in for a check nobody can identify.
_LISTING_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "check_runs": (("name", "status"), ("conclusion",)),
    "statuses": (("context", "state"), ()),
}


def _usable_entry(entry: Any, fields: tuple[tuple[str, ...], tuple[str, ...]]) -> bool:
    """Whether *entry* is a dict whose classified fields are usable."""
    if not isinstance(entry, dict):
        return False
    required, nullable = fields
    if not all(
        isinstance(value := entry.get(field), str) and value.strip()
        for field in required
    ):
        return False
    return all(isinstance(entry.get(field), (str, type(None))) for field in nullable)


class _CommitListingsMixin(_GitHubAsyncBase):
    """Commit check-run and status listing readers for ``GitHubAsync``."""

    async def get_check_runs_for_ref(
        self, owner: str, repo: str, ref: str
    ) -> list[dict[str, Any]]:
        """Check runs reported against *ref*.

        Returns the raw runs, including superseded duplicates: deciding
        which run is authoritative for a given name belongs to
        :mod:`dependamerge.check_runs`, not here.

        Paginated.  A busy commit carries more than one page, and every
        caller reasons about *absence* --- "nothing is failing", "only
        semantic checks are failing" --- so a truncated page reads as an
        all-clear for whatever sat on page two.  For the same reason a
        reading that is not whole raises :class:`IncompleteReadingError`
        rather than returning what it found.
        """
        return await self._read_commit_listing_strictly(
            f"/repos/{owner}/{repo}/commits/{ref}/check-runs", "check_runs"
        )

    async def get_commit_statuses(
        self, owner: str, repo: str, ref: str
    ) -> list[dict[str, Any]]:
        """Every entry of the combined commit status for *ref*.

        Each entry carries its ``context`` and ``state``.  GitHub's
        combined-status endpoint already collapses each context to its
        latest state, so no deduplication is needed here.  Paginated, and
        strict about completeness, for the reasons
        :meth:`get_check_runs_for_ref` is.
        """
        return await self._read_commit_listing_strictly(
            f"/repos/{owner}/{repo}/commits/{ref}/status", "statuses"
        )

    async def read_check_runs_for_ref(
        self, owner: str, repo: str, ref: str
    ) -> CommitReading:
        """Check runs on *ref*, with whether the reading is whole.

        Unlike :meth:`get_check_runs_for_ref`, a failed or partial read
        does not raise: the runs read are kept and the reading is marked
        incomplete.
        """
        return await self._read_commit_listing_tolerantly(
            f"/repos/{owner}/{repo}/commits/{ref}/check-runs", "check_runs"
        )

    async def read_commit_statuses(
        self, owner: str, repo: str, ref: str
    ) -> CommitReading:
        """Commit status entries on *ref*, with whether the reading is whole.

        The tolerant counterpart of :meth:`get_commit_statuses`.
        """
        return await self._read_commit_listing_tolerantly(
            f"/repos/{owner}/{repo}/commits/{ref}/status", "statuses"
        )

    async def _read_commit_listing_strictly(
        self, path: str, key: str
    ) -> list[dict[str, Any]]:
        """Read a commit listing whole, or raise."""
        entries: list[dict[str, Any]] = []
        if not await self._read_commit_listing(path, key, entries):
            raise IncompleteReadingError(
                f"Reading {path} is incomplete after {len(entries)} entries"
            )
        return entries

    async def _read_commit_listing_tolerantly(
        self, path: str, key: str
    ) -> CommitReading:
        """Read a commit listing, keeping what was read if a request fails."""
        entries: list[dict[str, Any]] = []
        try:
            complete = await self._read_commit_listing(path, key, entries)
        except Exception as exc:
            self.log.debug(
                "Reading %s stopped after %d entries and is incomplete: %s",
                path,
                len(entries),
                exc,
            )
            return CommitReading(entries, False)
        if not complete:
            self.log.debug(
                "Reading %s is incomplete (%d entries read)",
                path,
                len(entries),
            )
        return CommitReading(entries, complete)

    async def _read_commit_listing(
        self, path: str, key: str, into: list[dict[str, Any]]
    ) -> bool:
        """Append every usable entry of a paginated commit listing to *into*.

        The one paging loop behind the check-run and commit-status
        readers.  Returns whether the reading is whole.  It is not when
        no page arrives, a page or its listing has an unexpected shape,
        an entry is discarded as unusable, an entry ``id`` repeats (a
        page boundary shifted under the read, which can push an unseen
        entry past it), an entry without a usable ``id`` arrives in a
        reading of more than one page (a repeat of it could not be
        told apart), a page lacks a non-negative integer
        ``total_count``, or fewer distinct entries arrive than the
        largest ``total_count`` seen.  A repeated ``id`` keeps its later
        observation, in the earlier one's place, since the entry may
        have changed state between pages.  Request failures propagate,
        and *into* then holds what was read before them.
        """
        fields = _LISTING_FIELDS[key]
        complete = True
        total: int | None = None
        pages = 0
        anonymous = 0
        position: dict[int | str, int] = {}
        async for page in self.get_paginated(path, per_page=100):
            pages += 1
            if not isinstance(page, dict):
                complete = False
                continue
            reported = page.get("total_count")
            if (
                isinstance(reported, int)
                and not isinstance(reported, bool)
                and reported >= 0
            ):
                # The largest count seen: an entry posted between pages
                # raises it, and the first figure would then certify a
                # reading that missed one.
                total = reported if total is None else max(total, reported)
            else:
                # GitHub sends the count on every page of both listings.
                # Without it nothing shows the reading reached the end:
                # the transport ends silently on an empty page.
                complete = False
            batch = page.get(key)
            if not isinstance(batch, list):
                complete = False
                break
            if not batch:
                break
            for entry in batch:
                if not _usable_entry(entry, fields):
                    complete = False
                    continue
                entry_id = entry.get("id")
                if not isinstance(entry_id, (int, str)) or isinstance(entry_id, bool):
                    anonymous += 1
                elif entry_id in position:
                    into[position[entry_id]] = entry
                    complete = False
                    continue
                else:
                    position[entry_id] = len(into)
                into.append(entry)
        if pages == 0 or (total is not None and len(into) < total):
            complete = False
        if pages > 1 and anonymous:
            complete = False
        return complete
