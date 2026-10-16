# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""Page-size defaults and the automation-author list."""

from __future__ import annotations

# GitHub API tuning defaults - optimized for performance and rate limit compliance
DEFAULT_PRS_PAGE_SIZE = 30  # Pull requests per GraphQL page
DEFAULT_FILES_PAGE_SIZE = 50  # Files per pull request
DEFAULT_COMMENTS_PAGE_SIZE = 10  # Comments per pull request
DEFAULT_CONTEXTS_PAGE_SIZE = 20  # Status contexts per pull request

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
