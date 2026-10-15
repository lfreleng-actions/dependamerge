# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from dependamerge.bot_identity import canonical_bot_login
from dependamerge.models import FileChange, PullRequestInfo
from dependamerge.pr_comparator import PRComparator


class TestPRComparator:
    def test_init_default_threshold(self):
        comparator = PRComparator()
        assert comparator.similarity_threshold == 0.8

    def test_init_custom_threshold(self):
        comparator = PRComparator(0.9)
        assert comparator.similarity_threshold == 0.9

    def test_normalize_title_removes_versions(self):
        comparator = PRComparator()

        original = "Bump dependency from 1.2.3 to 1.2.4"
        normalized = comparator._normalize_title(original)
        assert "1.2.3" not in normalized
        assert "1.2.4" not in normalized
        assert "bump dependency from to" in normalized

    def test_normalize_title_removes_commit_hashes(self):
        comparator = PRComparator()

        original = "Update to commit abc123def456"
        normalized = comparator._normalize_title(original)
        assert "abc123def456" not in normalized
        assert "update to commit" in normalized

    def test_compare_titles_identical(self):
        comparator = PRComparator()

        title1 = "Bump requests from 2.28.0 to 2.28.1"
        title2 = "Bump requests from 2.27.0 to 2.28.1"

        score = comparator._compare_titles(title1, title2)
        assert score > 0.8  # Should be very similar after normalization

    def test_compare_file_changes_identical(self):
        comparator = PRComparator()

        files1 = [
            FileChange(
                filename="requirements.txt",
                additions=1,
                deletions=1,
                changes=2,
                status="modified",
            ),
            FileChange(
                filename="setup.py",
                additions=1,
                deletions=1,
                changes=2,
                status="modified",
            ),
        ]
        files2 = [
            FileChange(
                filename="requirements.txt",
                additions=2,
                deletions=1,
                changes=3,
                status="modified",
            ),
            FileChange(
                filename="setup.py",
                additions=1,
                deletions=2,
                changes=3,
                status="modified",
            ),
        ]

        score = comparator._compare_file_changes(files1, files2)
        assert score == 1.0  # Same files changed

    def test_compare_file_changes_partial_overlap(self):
        comparator = PRComparator()

        files1 = [
            FileChange(
                filename="requirements.txt",
                additions=1,
                deletions=1,
                changes=2,
                status="modified",
            ),
            FileChange(
                filename="setup.py",
                additions=1,
                deletions=1,
                changes=2,
                status="modified",
            ),
        ]
        files2 = [
            FileChange(
                filename="requirements.txt",
                additions=1,
                deletions=1,
                changes=2,
                status="modified",
            ),
            FileChange(
                filename="package.json",
                additions=1,
                deletions=1,
                changes=2,
                status="modified",
            ),
        ]

        score = comparator._compare_file_changes(files1, files2)
        assert 0.3 < score < 0.7  # Partial overlap

    def test_is_automation_pr_dependabot(self):
        comparator = PRComparator()

        pr = PullRequestInfo(
            number=1,
            title="Bump requests from 2.28.0 to 2.28.1",
            body="Bumps requests from 2.28.0 to 2.28.1",
            author="dependabot[bot]",
            head_sha="abc123",
            base_branch="main",
            head_branch="dependabot/pip/requests-2.28.1",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[],
            repository_full_name="owner/repo",
            html_url="https://github.com/owner/repo/pull/1",
        )

        assert comparator._is_automation_pr(pr)

    def test_is_automation_pr_human(self):
        comparator = PRComparator()

        pr = PullRequestInfo(
            number=1,
            title="Fix bug in user authentication",
            body="This PR fixes a critical bug",
            author="human-developer",
            head_sha="abc123",
            base_branch="main",
            head_branch="fix-auth-bug",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[],
            repository_full_name="owner/repo",
            html_url="https://github.com/owner/repo/pull/1",
        )

        assert not comparator._is_automation_pr(pr)

    def test_compare_similar_automation_prs(self):
        comparator = PRComparator(0.7)

        pr1 = PullRequestInfo(
            number=1,
            title="Bump requests from 2.28.0 to 2.28.1",
            body="Bumps requests from 2.28.0 to 2.28.1",
            author="dependabot[bot]",
            head_sha="abc123",
            base_branch="main",
            head_branch="dependabot/pip/requests-2.28.1",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename="requirements.txt",
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="owner/repo1",
            html_url="https://github.com/owner/repo1/pull/1",
        )

        pr2 = PullRequestInfo(
            number=2,
            title="Bump requests from 2.27.0 to 2.28.1",
            body="Bumps requests from 2.27.0 to 2.28.1",
            author="dependabot[bot]",
            head_sha="def456",
            base_branch="main",
            head_branch="dependabot/pip/requests-2.28.1",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename="requirements.txt",
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="owner/repo2",
            html_url="https://github.com/owner/repo2/pull/2",
        )

        result = comparator.compare_pull_requests(pr1, pr2)
        assert result.is_similar
        assert result.confidence_score >= 0.7
        assert len(result.reasons) > 0

    def test_different_packages_not_similar(self):
        """Test that PRs updating different packages are not considered similar."""
        comparator = PRComparator(0.8)

        # PR updating docker/metadata-action
        pr1 = PullRequestInfo(
            number=34,
            title="Chore: Bump docker/metadata-action from 5.7.0 to 5.8.0",
            body="Bumps docker/metadata-action from 5.7.0 to 5.8.0",
            author="dependabot[bot]",
            head_sha="abc123",
            base_branch="main",
            head_branch="dependabot/github_actions/docker/metadata-action-5.8.0",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename=".github/workflows/ci.yml",
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="repo1",
            html_url="https://github.com/repo1/pull/34",
        )

        # PR updating lfreleng-actions/python-build-action (different package)
        pr2 = PullRequestInfo(
            number=72,
            title="Chore: Bump lfreleng-actions/python-build-action from 1.2.0 to 1.3.0",
            body="Bumps lfreleng-actions/python-build-action from 1.2.0 to 1.3.0",
            author="dependabot[bot]",
            head_sha="def456",
            base_branch="main",
            head_branch="dependabot/github_actions/lfreleng-actions/python-build-action-1.3.0",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename=".github/workflows/ci.yml",  # Same filename
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="repo2",
            html_url="https://github.com/repo2/pull/72",
        )

        result = comparator.compare_pull_requests(pr1, pr2)

        # Should NOT be similar despite same filename and author
        assert not result.is_similar
        assert result.confidence_score < 0.8
        # Title score should be 0.0 for different packages
        title_score = comparator._compare_titles(pr1.title, pr2.title)
        assert title_score == 0.0

    def test_same_package_different_versions_similar(self):
        """Test that PRs updating the same package to different versions are similar."""
        comparator = PRComparator(0.8)

        # PR updating docker/metadata-action to 5.8.0
        pr1 = PullRequestInfo(
            number=1,
            title="Chore: Bump docker/metadata-action from 5.7.0 to 5.8.0",
            body="Bumps docker/metadata-action from 5.7.0 to 5.8.0",
            author="dependabot[bot]",
            head_sha="abc123",
            base_branch="main",
            head_branch="dependabot/github_actions/docker/metadata-action-5.8.0",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename=".github/workflows/ci.yml",
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="repo1",
            html_url="https://github.com/repo1/pull/1",
        )

        # PR updating same package (docker/metadata-action) to same version
        pr2 = PullRequestInfo(
            number=2,
            title="Chore: Bump docker/metadata-action from 5.6.0 to 5.8.0",
            body="Bumps docker/metadata-action from 5.6.0 to 5.8.0",
            author="dependabot[bot]",
            head_sha="def456",
            base_branch="main",
            head_branch="dependabot/github_actions/docker/metadata-action-5.8.0",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename=".github/workflows/ci.yml",  # Same filename
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="repo2",
            html_url="https://github.com/repo2/pull/2",
        )

        result = comparator.compare_pull_requests(pr1, pr2)

        # Should be similar - same package, same filename, same author
        assert result.is_similar
        assert result.confidence_score >= 0.8
        # Title score should be 1.0 for same package
        title_score = comparator._compare_titles(pr1.title, pr2.title)
        assert title_score == 1.0

    def test_extract_package_name(self):
        """Test package name extraction from various title formats."""
        comparator = PRComparator()

        # Test various patterns
        test_cases = [
            (
                "Bump docker/metadata-action from 5.7.0 to 5.8.0",
                "docker/metadata-action",
            ),
            (
                "Chore: Bump docker/metadata-action from 5.7.0 to 5.8.0",
                "docker/metadata-action",
            ),
            ("Update requests from 2.28.0 to 2.28.1", "requests"),
            ("Upgrade numpy from 1.21.0 to 1.22.0", "numpy"),
            ("bump pytest from 7.1.0 to 7.2.0", "pytest"),
            ("Random PR title", ""),  # Not a dependency update
            ("Fix bug in authentication", ""),  # Not a dependency update
        ]

        for title, expected_package in test_cases:
            actual_package = comparator._extract_package_name(title)
            assert actual_package == expected_package, f"Failed for title: {title}"

    def test_compare_non_automation_prs(self):
        """Test that non-automation PRs can be compared when only_automation=False."""
        comparator = PRComparator(0.7)

        # Non-automation PR 1 with similar workflow update
        pr1 = PullRequestInfo(
            number=9,
            title="CI: Update tag-push.yaml workflow",
            body="Updates the tag-push workflow configuration",
            author="ModeSevenIndustrialSolutions",
            head_sha="abc123",
            base_branch="main",
            head_branch="fix-workflow",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename=".github/workflows/tag-push.yaml",
                    additions=5,
                    deletions=2,
                    changes=7,
                    status="modified",
                )
            ],
            repository_full_name="org/repo1",
            html_url="https://github.com/org/repo1/pull/9",
        )

        # Non-automation PR 2 with similar workflow update from same author
        pr2 = PullRequestInfo(
            number=15,
            title="CI: Update tag-push.yaml workflow configuration",
            body="Updates the tag-push workflow for better performance",
            author="ModeSevenIndustrialSolutions",
            head_sha="def456",
            base_branch="main",
            head_branch="update-workflow",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename=".github/workflows/tag-push.yaml",
                    additions=3,
                    deletions=1,
                    changes=4,
                    status="modified",
                )
            ],
            repository_full_name="org/repo2",
            html_url="https://github.com/org/repo2/pull/15",
        )

        # Test with only_automation=False (non-automation mode)
        result = comparator.compare_pull_requests(pr1, pr2, only_automation=False)
        assert result.is_similar
        assert result.confidence_score >= 0.7
        assert any("Similar titles" in reason for reason in result.reasons)
        assert any("Similar file changes" in reason for reason in result.reasons)

        # Test with only_automation=True (should fail for non-automation PRs)
        result_automation = comparator.compare_pull_requests(
            pr1, pr2, only_automation=True
        )
        assert not result_automation.is_similar
        assert result_automation.confidence_score == 0.0
        assert (
            "One or both PRs are not from automation tools" in result_automation.reasons
        )

    def test_github_actions_workflow_file_similarity(self):
        """Test that GitHub Actions workflow files get partial similarity even with different names."""
        comparator = PRComparator(0.8)

        # Create PR with semantic-pull-request.yaml workflow
        pr1 = PullRequestInfo(
            number=15,
            title="Chore: Bump amannn/action-semantic-pull-request from 6.0.1 to 6.1.1",
            body="Bumps [amannn/action-semantic-pull-request](https://github.com/amannn/action-semantic-pull-request) from 6.0.1 to 6.1.1.",
            author="dependabot[bot]",
            head_sha="abc123",
            base_branch="main",
            head_branch="dependabot/github_actions/amannn/action-semantic-pull-request-6.1.1",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename=".github/workflows/semantic-pull-request.yaml",
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="org/repo1",
            html_url="https://github.com/org/repo1/pull/15",
        )

        # Create PR with ci.yml workflow (different filename)
        pr2 = PullRequestInfo(
            number=23,
            title="Chore: Bump amannn/action-semantic-pull-request from 6.0.1 to 6.1.1",
            body="Bumps [amannn/action-semantic-pull-request](https://github.com/amannn/action-semantic-pull-request) from 6.0.1 to 6.1.1.",
            author="dependabot[bot]",
            head_sha="def456",
            base_branch="main",
            head_branch="dependabot/github_actions/amannn/action-semantic-pull-request-6.1.1",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename=".github/workflows/ci.yml",
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="org/repo2",
            html_url="https://github.com/org/repo2/pull/23",
        )

        # Test file comparison directly
        file_score = comparator._compare_file_changes(
            pr1.files_changed, pr2.files_changed
        )
        assert file_score == 0.5, (
            f"Expected 0.5 for different workflow files, got {file_score}"
        )

        # Test overall comparison - should be similar despite different workflow filenames
        result = comparator.compare_pull_requests(pr1, pr2, only_automation=True)
        assert result.is_similar
        assert result.confidence_score >= 0.8
        assert any("Similar titles" in reason for reason in result.reasons)

        # Test with non-workflow files - should still get 0.0
        pr3 = PullRequestInfo(
            number=24,
            title="Chore: Bump some-package from 1.0.0 to 1.1.0",
            body="Bumps some-package from 1.0.0 to 1.1.0.",
            author="dependabot[bot]",
            head_sha="ghi789",
            base_branch="main",
            head_branch="dependabot/npm_and_yarn/some-package-1.1.0",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename="package.json",
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="org/repo3",
            html_url="https://github.com/org/repo3/pull/24",
        )

        pr4 = PullRequestInfo(
            number=25,
            title="Chore: Bump some-package from 1.0.0 to 1.1.0",
            body="Bumps some-package from 1.0.0 to 1.1.0.",
            author="dependabot[bot]",
            head_sha="jkl012",
            base_branch="main",
            head_branch="dependabot/npm_and_yarn/some-package-1.1.0",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename="requirements.txt",
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="org/repo4",
            html_url="https://github.com/org/repo4/pull/25",
        )

        # Non-workflow files with different names should still get 0.0
        non_workflow_score = comparator._compare_file_changes(
            pr3.files_changed, pr4.files_changed
        )
        assert non_workflow_score == 0.0, (
            f"Expected 0.0 for different non-workflow files, got {non_workflow_score}"
        )

    def test_author_normalization_bot_names(self):
        """Test that author normalization handles bot name differences between REST and GraphQL APIs."""
        comparator = PRComparator(0.8)

        # Create PR with REST API style author (dependabot[bot])
        pr1 = PullRequestInfo(
            number=15,
            title="Chore: Bump amannn/action-semantic-pull-request from 6.0.1 to 6.1.1",
            body="Bumps [amannn/action-semantic-pull-request] from 6.0.1 to 6.1.1.",
            author="dependabot[bot]",  # REST API format
            head_sha="abc123",
            base_branch="main",
            head_branch="dependabot/github_actions/amannn/action-semantic-pull-request-6.1.1",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename=".github/workflows/semantic-pull-request.yaml",
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="org/repo1",
            html_url="https://github.com/org/repo1/pull/15",
        )

        # Create PR as the GraphQL boundary builds it: GraphQL reports the
        # App as a bare ``dependabot`` Bot node, canonicalised on the way in.
        pr2 = PullRequestInfo(
            number=18,
            title="Chore: Bump amannn/action-semantic-pull-request from 6.0.1 to 6.1.1",
            body="Bumps [amannn/action-semantic-pull-request] from 6.0.1 to 6.1.1.",
            author=canonical_bot_login("dependabot", "Bot"),
            head_sha="def456",
            base_branch="main",
            head_branch="dependabot/github_actions/amannn/action-semantic-pull-request-6.1.1",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=0,
            files_changed=[
                FileChange(
                    filename=".github/workflows/semantic-pull-request.yaml",
                    additions=1,
                    deletions=1,
                    changes=2,
                    status="modified",
                )
            ],
            repository_full_name="org/repo2",
            html_url="https://github.com/org/repo2/pull/18",
        )

        # Test that normalization makes the raw REST and GraphQL forms
        # equivalent, then that the canonical forms compare as one author
        normalized1 = comparator._normalize_author(pr1.author)
        normalized2 = comparator._normalize_author("dependabot")
        assert normalized1 == normalized2, (
            f"Expected normalized authors to match: '{normalized1}' vs '{normalized2}'"
        )
        assert normalized1 == "dependabot", (
            f"Expected normalized author to be 'dependabot', got '{normalized1}'"
        )

        # Test that the overall comparison considers them as same author
        result = comparator.compare_pull_requests(pr1, pr2, only_automation=True)
        assert result.is_similar
        assert result.confidence_score >= 0.8
        assert "Same automation author" in result.reasons

        # Test with other bot types
        test_cases = [
            ("pre-commit-ci[bot]", "pre-commit-ci"),
            ("renovate[bot]", "renovate"),
            ("github-actions[bot]", "github-actions"),
        ]

        for bot_with_suffix, bot_without_suffix in test_cases:
            norm_with = comparator._normalize_author(bot_with_suffix)
            norm_without = comparator._normalize_author(bot_without_suffix)
            assert norm_with == norm_without, (
                f"Bot normalization failed for {bot_with_suffix} vs {bot_without_suffix}"
            )
            assert norm_with == bot_without_suffix.lower(), (
                f"Expected normalized to be '{bot_without_suffix.lower()}', got '{norm_with}'"
            )
