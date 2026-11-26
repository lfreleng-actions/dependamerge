# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Page-size defaults and the automation-author list."""

from __future__ import annotations

from ..github_graphql.pull_requests import (
    SCAN_CHECKS_PAGE_SIZE,
    SCAN_COMMENTS_PAGE_SIZE,
    SCAN_FILES_PAGE_SIZE,
)

# Pull requests per GraphQL page; halved while rate limited.
DEFAULT_PRS_PAGE_SIZE = 30
# The nested list sizes the page query selects.  They are fixed, so rate
# limiting never changes what a scan sees of a pull request.
DEFAULT_FILES_PAGE_SIZE = SCAN_FILES_PAGE_SIZE
DEFAULT_COMMENTS_PAGE_SIZE = SCAN_COMMENTS_PAGE_SIZE
DEFAULT_CONTEXTS_PAGE_SIZE = SCAN_CHECKS_PAGE_SIZE

# Automation tools recognized for PR categorization.  The first three are
# the built-in bots a merge acts on by default (``bot_policy.BUILTIN_BOTS``);
# any other App counts as automation in reports, but a merge acts on its
# PRs only when a bot policy includes it.
AUTOMATION_TOOLS = [
    "dependabot",
    "renovate",
    "pre-commit",
    "[bot]",
]
