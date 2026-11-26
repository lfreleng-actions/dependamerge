# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""The nested lists each scan asks the open pull request page for.

``REPO_OPEN_PRS_PAGE`` switches each nested list on or off; a scan
selects only the lists it reads, and the extractors refuse a list that
was not selected rather than read its absence as an empty list.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanLists:
    """Which nested lists a page of open pull requests carries."""

    files: bool = False
    reviews: bool = False
    checks: bool = False
    comments: bool = False

    def variables(self) -> dict[str, bool]:
        """The page query's ``@include`` switches for this selection."""
        return {
            "withFiles": self.files,
            "withReviews": self.reviews,
            "withChecks": self.checks,
            "withComments": self.comments,
        }


#: The merge scans and ``close``: files feed the similarity score, and
#: reviews the merge decisions.
MERGE_SCAN_LISTS = ScanLists(files=True, reviews=True)

#: ``status``: files, to count pull requests touching actions or workflows.
STATUS_SCAN_LISTS = ScanLists(files=True)

#: ``blocked``: checks, for the failing-check reason, and comments, for
#: the Copilot comment count.
BLOCKED_SCAN_LISTS = ScanLists(checks=True, comments=True)
