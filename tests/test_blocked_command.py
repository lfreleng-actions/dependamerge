# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.models import (
    OrganizationScanResult,
    UnmergeablePR,
    UnmergeableReason,
)


class TestBlockedCommand:
    """Tests for the blocked command and include-drafts flag."""

    runner: CliRunner = CliRunner()

    def setup_method(self):
        self.runner = CliRunner()

    def _create_unmergeable_pr(
        self,
        repository: str,
        pr_number: int,
        title: str,
        author: str,
        reasons: list[tuple[str, str]],
    ) -> UnmergeablePR:
        """Helper to create an UnmergeablePR with given reasons."""
        return UnmergeablePR(
            repository=repository,
            pr_number=pr_number,
            title=title,
            author=author,
            url=f"https://github.com/{repository}/pull/{pr_number}",
            reasons=[
                UnmergeableReason(type=reason_type, description=reason_desc)
                for reason_type, reason_desc in reasons
            ],
            copilot_comments_count=0,
            copilot_comments=[],
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )

    @patch("dependamerge.github_service.GitHubService")
    def test_blocked_command_default_excludes_drafts(self, mock_service_class):
        """Test that blocked command excludes draft-only PRs by default."""
        # Create mock scan result with mixed PRs
        mock_scan_result = OrganizationScanResult(
            organization="test-org",
            total_repositories=3,
            scanned_repositories=3,
            total_prs=4,
            unmergeable_prs=[
                # This PR should be included (failing checks)
                self._create_unmergeable_pr(
                    "test-org/repo1",
                    1,
                    "Fix bug",
                    "developer",
                    [("failing_checks", "Required status checks are failing")],
                ),
                # This PR should be included (merge conflict)
                self._create_unmergeable_pr(
                    "test-org/repo2",
                    2,
                    "Update dependencies",
                    "dependabot",
                    [("merge_conflict", "Pull request has merge conflicts")],
                ),
                # This PR should be included (behind base)
                self._create_unmergeable_pr(
                    "test-org/repo3",
                    3,
                    "Add feature",
                    "developer",
                    [("behind_base", "Pull request is behind the base branch")],
                ),
            ],
            scan_timestamp="2025-01-01T00:00:00Z",
            errors=[],
        )

        # Setup mock service
        mock_service = AsyncMock()
        mock_service.scan_organization.return_value = mock_scan_result
        mock_service.close = AsyncMock()
        mock_service_class.return_value = mock_service

        # Run command without --include-drafts (default behavior)
        result = self.runner.invoke(
            app, ["blocked", "test-org", "--token", "test_token"]
        )

        # Verify service was called with include_drafts=False (default)
        mock_service.scan_organization.assert_called_once_with(
            "test-org", include_drafts=False
        )

        # Verify exit code
        assert result.exit_code == 0

        # Verify output contains the non-draft PRs
        assert "repo1" in result.stdout
        assert "repo2" in result.stdout
        assert "repo3" in result.stdout

    @patch("dependamerge.github_service.GitHubService")
    def test_blocked_command_include_drafts_flag(self, mock_service_class):
        """Test that --include-drafts flag includes draft PRs."""
        # Create mock scan result with draft PRs included
        mock_scan_result = OrganizationScanResult(
            organization="test-org",
            total_repositories=3,
            scanned_repositories=3,
            total_prs=5,
            unmergeable_prs=[
                # Regular blocked PR
                self._create_unmergeable_pr(
                    "test-org/repo1",
                    1,
                    "Fix bug",
                    "developer",
                    [("failing_checks", "Required status checks are failing")],
                ),
                # Draft PR (should be included when flag is set)
                self._create_unmergeable_pr(
                    "test-org/repo2",
                    2,
                    "WIP: New feature",
                    "developer",
                    [("draft", "Pull request is in draft state")],
                ),
                # Draft PR with other issues (should always be included)
                self._create_unmergeable_pr(
                    "test-org/repo3",
                    3,
                    "WIP: Another feature",
                    "developer",
                    [
                        ("draft", "Pull request is in draft state"),
                        ("failing_checks", "Required status checks are failing"),
                    ],
                ),
            ],
            scan_timestamp="2025-01-01T00:00:00Z",
            errors=[],
        )

        # Setup mock service
        mock_service = AsyncMock()
        mock_service.scan_organization.return_value = mock_scan_result
        mock_service.close = AsyncMock()
        mock_service_class.return_value = mock_service

        # Run command with --include-drafts
        result = self.runner.invoke(
            app, ["blocked", "test-org", "--token", "test_token", "--include-drafts"]
        )

        # Verify service was called with include_drafts=True
        mock_service.scan_organization.assert_called_once_with(
            "test-org", include_drafts=True
        )

        # Verify exit code
        assert result.exit_code == 0

        # Verify output contains all PRs including drafts
        assert "repo1" in result.stdout
        assert "repo2" in result.stdout
        assert "repo3" in result.stdout

    @patch("dependamerge.github_service.GitHubService")
    def test_blocked_command_empty_result(self, mock_service_class):
        """Test blocked command with no unmergeable PRs."""
        mock_scan_result = OrganizationScanResult(
            organization="test-org",
            total_repositories=5,
            scanned_repositories=5,
            total_prs=10,
            unmergeable_prs=[],
            scan_timestamp="2025-01-01T00:00:00Z",
            errors=[],
        )

        # Setup mock service
        mock_service = AsyncMock()
        mock_service.scan_organization.return_value = mock_scan_result
        mock_service.close = AsyncMock()
        mock_service_class.return_value = mock_service

        # Run command
        result = self.runner.invoke(
            app, ["blocked", "test-org", "--token", "test_token"]
        )

        # Verify exit code
        assert result.exit_code == 0

        # Verify success message
        assert "No unmergeable pull requests found" in result.stdout

    @patch("dependamerge.github_service.GitHubService")
    def test_blocked_command_json_output(self, mock_service_class):
        """Test blocked command with JSON output format."""
        mock_scan_result = OrganizationScanResult(
            organization="test-org",
            total_repositories=1,
            scanned_repositories=1,
            total_prs=1,
            unmergeable_prs=[
                self._create_unmergeable_pr(
                    "test-org/repo1",
                    1,
                    "Fix bug",
                    "developer",
                    [("failing_checks", "Required status checks are failing")],
                )
            ],
            scan_timestamp="2025-01-01T00:00:00Z",
            errors=[],
        )

        # Setup mock service
        mock_service = AsyncMock()
        mock_service.scan_organization.return_value = mock_scan_result
        mock_service.close = AsyncMock()
        mock_service_class.return_value = mock_service

        # Run command with JSON format
        result = self.runner.invoke(
            app, ["blocked", "test-org", "--token", "test_token", "--format", "json"]
        )

        # Verify exit code
        assert result.exit_code == 0

        # Verify JSON output
        assert '"organization": "test-org"' in result.stdout
        assert '"total_repositories": 1' in result.stdout
        assert '"failing_checks"' in result.stdout


class TestBlockedCommandServiceIntegration:
    """Integration tests for blocked command with GitHubService."""

    @pytest.mark.asyncio
    async def test_analyze_pr_node_draft_only_excluded_by_default(self):
        """Test that _analyze_pr_node returns None for draft-only PRs by default."""
        from dependamerge.github_service import GitHubService

        service = GitHubService(token="test_token")

        # Mock PR that is only blocked due to draft status
        pr_node = {
            "number": 1,
            "title": "WIP: New feature",
            "author": {"login": "developer"},
            "url": "https://github.com/test-org/repo1/pull/1",
            "isDraft": True,
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "draft",
            "commits": {"nodes": []},
            "comments": {"totalCount": 0, "nodes": []},
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T00:00:00Z",
        }

        # Test with include_drafts=False (default)
        result = await service._analyze_pr_node(
            "test-org/repo1", pr_node, include_drafts=False
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_pr_node_draft_only_included_with_flag(self):
        """Test that _analyze_pr_node includes draft-only PRs when include_drafts=True."""
        from dependamerge.github_service import GitHubService

        service = GitHubService(token="test_token")

        # Mock PR that is only blocked due to draft status
        pr_node = {
            "number": 1,
            "title": "WIP: New feature",
            "author": {"login": "developer"},
            "url": "https://github.com/test-org/repo1/pull/1",
            "isDraft": True,
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "draft",
            "commits": {"nodes": []},
            "comments": {"totalCount": 0, "nodes": []},
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T00:00:00Z",
        }

        # Test with include_drafts=True
        result = await service._analyze_pr_node(
            "test-org/repo1", pr_node, include_drafts=True
        )
        assert result is not None
        assert result.pr_number == 1
        assert len(result.reasons) == 1
        assert result.reasons[0].type == "draft"

    @pytest.mark.asyncio
    async def test_analyze_pr_node_draft_with_other_issues_always_included(self):
        """Test that draft PRs with other blocking issues are always reported."""
        from dependamerge.github_service import GitHubService

        service = GitHubService(token="test_token")

        # Mock PR that is draft AND has merge conflicts
        pr_node = {
            "number": 2,
            "title": "WIP: Update dependencies",
            "author": {"login": "developer"},
            "url": "https://github.com/test-org/repo1/pull/2",
            "isDraft": True,
            "mergeable": "CONFLICTING",
            "mergeStateStatus": "dirty",
            "commits": {"nodes": []},
            "comments": {"totalCount": 0, "nodes": []},
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T00:00:00Z",
        }

        # Test with include_drafts=False - should still return result because of conflicts
        result = await service._analyze_pr_node(
            "test-org/repo1", pr_node, include_drafts=False
        )
        assert result is not None
        assert result.pr_number == 2
        # Should have conflict reason but draft reason should be filtered out
        reason_types = [r.type for r in result.reasons]
        assert "merge_conflict" in reason_types
        assert "draft" not in reason_types

    @pytest.mark.asyncio
    async def test_analyze_pr_node_draft_with_failing_checks_filtered(self):
        """Test that draft status is filtered from PRs with other issues when include_drafts=False."""
        from dependamerge.github_service import GitHubService

        service = GitHubService(token="test_token")

        # Mock PR that is draft AND has failing checks
        # Note: mergeable must be "UNKNOWN" (not "CONFLICTING") for failing checks to be detected
        # when mergeStateStatus is "blocked"
        pr_node = {
            "number": 3,
            "title": "WIP: Fix tests",
            "author": {"login": "developer"},
            "url": "https://github.com/test-org/repo1/pull/3",
            "isDraft": True,
            "mergeable": "UNKNOWN",
            "mergeStateStatus": "blocked",
            "commits": {
                "nodes": [
                    {
                        "commit": {
                            "statusCheckRollup": {
                                "state": "FAILURE",
                                "contexts": {
                                    "nodes": [
                                        {
                                            "__typename": "CheckRun",
                                            "name": "test",
                                            "conclusion": "FAILURE",
                                        }
                                    ]
                                },
                            }
                        }
                    }
                ]
            },
            "comments": {"totalCount": 0, "nodes": []},
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T00:00:00Z",
        }

        # Test with include_drafts=False
        result = await service._analyze_pr_node(
            "test-org/repo1", pr_node, include_drafts=False
        )
        assert result is not None
        assert result.pr_number == 3
        # Should have failing_checks but not draft
        reason_types = [r.type for r in result.reasons]
        assert "failing_checks" in reason_types
        assert "draft" not in reason_types

        # Test with include_drafts=True - should include both
        result_with_drafts = await service._analyze_pr_node(
            "test-org/repo1", pr_node, include_drafts=True
        )
        assert result_with_drafts is not None
        reason_types_with_drafts = [r.type for r in result_with_drafts.reasons]
        assert "failing_checks" in reason_types_with_drafts
        assert "draft" in reason_types_with_drafts

    @pytest.mark.asyncio
    async def test_analyze_pr_node_behind_base_not_draft(self):
        """Test that PRs behind base branch are reported correctly without draft status."""
        from dependamerge.github_service import GitHubService

        service = GitHubService(token="test_token")

        # Mock PR that is behind base branch but not a draft
        pr_node = {
            "number": 4,
            "title": "Update README",
            "author": {"login": "developer"},
            "url": "https://github.com/test-org/repo1/pull/4",
            "isDraft": False,
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "behind",
            "commits": {"nodes": []},
            "comments": {"totalCount": 0, "nodes": []},
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-01T00:00:00Z",
        }

        # Test with include_drafts=False
        result = await service._analyze_pr_node(
            "test-org/repo1", pr_node, include_drafts=False
        )
        assert result is not None
        assert result.pr_number == 4
        assert len(result.reasons) == 1
        assert result.reasons[0].type == "behind_base"
