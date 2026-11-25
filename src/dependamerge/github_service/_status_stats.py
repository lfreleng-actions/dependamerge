# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""PR statistics behind the organization status report."""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..bot_identity import (
    canonical_bot_login,
    is_automation_author,
    rest_author_login,
)
from ._base import _GitHubServiceBase

# Bounds the cost of the merged-since read: 10 pages of 100.
_MERGED_PRS_PER_PAGE = 100
_MERGED_PRS_MAX_PAGES = 10


class _StatusStatsMixin(_GitHubServiceBase):
    """PR statistics for ``GitHubService``."""

    async def _gather_pr_statistics(
        self,
        owner: str,
        name: str,
        since_date: str | None,
        *,
        read_open_prs: bool = True,
    ) -> dict[str, int]:
        """
        Gather PR statistics for a repository.

        ``read_open_prs`` false skips the open-PR pages, for a repository
        the listing counts none in; the merged-PR read is unchanged.

        Returns dict with counts for:
        - open_prs_human, open_prs_automation
        - merged_prs_human, merged_prs_automation
        - action_prs_human, action_prs_automation
        - workflow_prs_human, workflow_prs_automation
        """
        stats = {
            "open_prs_human": 0,
            "open_prs_automation": 0,
            "merged_prs_human": 0,
            "merged_prs_automation": 0,
            "action_prs_human": 0,
            "action_prs_automation": 0,
            "workflow_prs_human": 0,
            "workflow_prs_automation": 0,
        }

        try:
            open_prs: list[dict[str, Any]] = []
            page_info: dict[str, Any] = {}
            if read_open_prs:
                first_nodes, page_info = await self._fetch_repo_prs_first_page(
                    owner, name
                )
                open_prs.extend(first_nodes)

            # Get additional pages if needed
            if page_info.get("hasNextPage"):
                async for pr_node in self._iter_repo_open_prs_pages(
                    owner, name, page_info.get("endCursor")
                ):
                    open_prs.append(pr_node)

            # Count open PRs.  The raw GraphQL login is bare for an App,
            # so canonicalise it by account type before classifying.
            for pr in open_prs:
                author_node = pr.get("author") or {}
                is_automation = self._is_automation_author(
                    canonical_bot_login(
                        author_node.get("login"), author_node.get("__typename")
                    )
                )

                if is_automation:
                    stats["open_prs_automation"] += 1
                else:
                    stats["open_prs_human"] += 1

                # Check if PR affects actions or workflows
                files = (pr.get("files") or {}).get("nodes", []) or []
                affects_action = self._affects_action_files(files)
                affects_workflow = self._affects_workflow_files(files)

                if affects_action:
                    if is_automation:
                        stats["action_prs_automation"] += 1
                    else:
                        stats["action_prs_human"] += 1

                if affects_workflow:
                    if is_automation:
                        stats["workflow_prs_automation"] += 1
                    else:
                        stats["workflow_prs_human"] += 1

            # Get merged PRs since the last tag/release
            if since_date:
                merged_prs = await self._get_merged_prs_since(owner, name, since_date)
                for pr in merged_prs:
                    is_automation = self._is_automation_author(
                        rest_author_login(pr.get("user"))
                    )

                    if is_automation:
                        stats["merged_prs_automation"] += 1
                    else:
                        stats["merged_prs_human"] += 1

        except Exception as e:
            self.log.debug(f"Error gathering PR statistics for {owner}/{name}: {e}")

        return stats

    def _is_automation_author(self, author: str) -> bool:
        """Check if author is an automation tool.

        Delegates to the shared :func:`bot_identity.is_automation_author`
        so REST and GraphQL login forms are classified identically.
        """
        return is_automation_author(author)

    def _affects_action_files(self, files: list[dict[str, Any]]) -> bool:
        """Check if files include action definition or implementation files."""
        action_patterns = [
            "action.yaml",
            "action.yml",
            "Dockerfile",  # Action Dockerfiles
        ]

        for file_node in files:
            path = file_node.get("path", "")
            filename = path.split("/")[-1] if "/" in path else path

            if filename.lower() in [p.lower() for p in action_patterns]:
                return True

            # Check for JavaScript action files (in src/ or lib/ directories)
            if path.startswith(("src/", "lib/")) and path.endswith(".js"):
                return True

        return False

    def _affects_workflow_files(self, files: list[dict[str, Any]]) -> bool:
        """Check if files include GitHub workflow or configuration files."""
        for file_node in files:
            path = file_node.get("path", "")

            # Check if file is in .github directory
            if path.startswith(".github/"):
                # Exclude non-workflow files
                if path.endswith((".md", ".txt", ".png", ".jpg", ".gif")):
                    continue

                # Include workflow files and other YAML configs
                if path.startswith(".github/workflows/") or path.endswith(
                    (".yml", ".yaml")
                ):
                    return True

        return False

    async def _get_merged_prs_since(
        self, owner: str, name: str, since_date: str
    ) -> list[dict[str, Any]]:
        """Get merged PRs since a specific date.

        Closed pull requests are read newest-updated first.  A pull
        request's ``merged_at`` is never later than its ``updated_at``,
        so the read stops at the first page whose oldest update precedes
        the since-date.  A page ceiling bounds the cost; when it cuts
        the read short a warning says the merged counts may be low.
        """
        try:
            # Convert date format from YYYY/MM/DD to ISO format
            date_obj = datetime.strptime(since_date, "%Y/%m/%d")
            iso_date = date_obj.strftime("%Y-%m-%dT%H:%M:%SZ")

            merged_prs: list[dict[str, Any]] = []
            pages_read = 0
            last_page_full = False
            reached_since = False

            async for prs_data in self._api.get_paginated(
                f"/repos/{owner}/{name}/pulls",
                params={"state": "closed", "sort": "updated", "direction": "desc"},
                per_page=_MERGED_PRS_PER_PAGE,
                max_pages=_MERGED_PRS_MAX_PAGES,
            ):
                if not isinstance(prs_data, list):
                    break
                pages_read += 1
                last_page_full = len(prs_data) >= _MERGED_PRS_PER_PAGE

                for pr in prs_data:
                    merged_at = pr.get("merged_at")
                    if merged_at and merged_at >= iso_date:
                        merged_prs.append(pr)

                updated = [pr["updated_at"] for pr in prs_data if pr.get("updated_at")]
                if updated and min(updated) < iso_date:
                    reached_since = True
                    break

            if (
                not reached_since
                and last_page_full
                and pages_read >= _MERGED_PRS_MAX_PAGES
            ):
                self.log.warning(
                    "Read the %d most recently updated closed pull requests "
                    "of %s/%s without reaching %s; its merged pull request "
                    "counts may be understated",
                    pages_read * _MERGED_PRS_PER_PAGE,
                    owner,
                    name,
                    since_date,
                )

            return merged_prs

        except Exception as e:
            self.log.debug(f"Error getting merged PRs for {owner}/{name}: {e}")
            return []
