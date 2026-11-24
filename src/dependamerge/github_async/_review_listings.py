# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Paginated reads of a pull request's reviews and review comments.

Both REST endpoints answer 30 entries by default, oldest first, so an
unpaged read misses the newest reviews: the ones a decision most often
turns on.  One paging loop reads either listing whole and reports, as
the commit listings do, whether the reading accounts for every entry.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from typing import Any, NamedTuple

from ._base import _GitHubAsyncBase

#: Page size and page ceiling for the review and review-comment readers.
REVIEW_LISTING_PAGE_SIZE = 100
REVIEW_LISTING_MAX_PAGES = 30


class ReviewReading(NamedTuple):
    """Entries read from a pull request's review or review-comment listing.

    ``complete`` says the reading accounts for every entry, so
    **absence** from it is evidence.  A partial reading still carries
    what it found: finding an entry is conclusive however the read
    ended, and only not finding one depends on having read it all.
    """

    entries: list[dict[str, Any]]
    complete: bool


class _ReviewListingsMixin(_GitHubAsyncBase):
    """Review and review-comment listing readers for ``GitHubAsync``."""

    async def read_pull_request_reviews(
        self, owner: str, repo: str, number: int
    ) -> ReviewReading:
        """Every review on a pull request, oldest first.

        A review without a state has an unknown verdict, so it keeps its
        entry but leaves the reading incomplete.

        REST: GET /repos/{owner}/{repo}/pulls/{pull_number}/reviews
        """
        reading = await self._read_review_listing(
            f"/repos/{owner}/{repo}/pulls/{number}/reviews"
        )
        if reading.complete and not all(
            isinstance(review.get("state"), str) and review.get("state")
            for review in reading.entries
        ):
            self.log.debug(
                "Reviews on %s/%s#%s include one without a state; "
                "the reading is incomplete",
                owner,
                repo,
                number,
            )
            return ReviewReading(reading.entries, False)
        return reading

    async def read_pull_request_review_comments(
        self, owner: str, repo: str, number: int
    ) -> ReviewReading:
        """Every review comment on a pull request, oldest first.

        REST: GET /repos/{owner}/{repo}/pulls/{pull_number}/comments
        """
        return await self._read_review_listing(
            f"/repos/{owner}/{repo}/pulls/{number}/comments"
        )

    async def _read_review_listing(self, path: str) -> ReviewReading:
        """Read a paginated list of objects, keeping what a failure leaves.

        The reading is incomplete when a request fails, a page is not a
        list, an entry is not an object, an entry ``id`` repeats (a page
        boundary shifted under the read, which can push an unseen entry
        past it; the later observation replaces the earlier one), or the
        page ceiling is reached on a full page.
        """
        entries: list[dict[str, Any]] = []
        position: dict[int | str, int] = {}
        complete = True
        pages = 0
        last_full = False
        try:
            async for page in self.get_paginated(
                path,
                per_page=REVIEW_LISTING_PAGE_SIZE,
                max_pages=REVIEW_LISTING_MAX_PAGES,
                yield_malformed=True,
            ):
                pages += 1
                if not isinstance(page, list):
                    complete = False
                    break
                for entry in page:
                    if not isinstance(entry, dict):
                        complete = False
                        continue
                    entry_id = entry.get("id")
                    if isinstance(entry_id, (int, str)) and not isinstance(
                        entry_id, bool
                    ):
                        if entry_id in position:
                            entries[position[entry_id]] = entry
                            complete = False
                            continue
                        position[entry_id] = len(entries)
                    entries.append(entry)
                last_full = len(page) >= REVIEW_LISTING_PAGE_SIZE
        except Exception as exc:
            self.log.debug(
                "Reading %s stopped after %d entries and is incomplete: %s",
                path,
                len(entries),
                exc,
            )
            return ReviewReading(entries, False)
        if pages >= REVIEW_LISTING_MAX_PAGES and last_full:
            complete = False
        if not complete:
            self.log.debug(
                "Reading %s is incomplete (%d entries read)", path, len(entries)
            )
        return ReviewReading(entries, complete)
