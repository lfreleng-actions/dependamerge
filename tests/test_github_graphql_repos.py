# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for repository-listing GraphQL documents."""

from __future__ import annotations

import pytest

from dependamerge.github_graphql import (
    ORG_REPOS_ONLY,
    OWNER_PROFILE,
    REPO_OPEN_PRS_PAGE,
    USER_REPOS_ONLY,
)

_POLICY_READ = 'object(expression: "HEAD:.github/dependamerge/config.json")'
_BLOB_FIELDS = "... on Blob { text byteSize isTruncated isBinary }"


def test_user_repos_query_limits_owner_affiliations_to_owner() -> None:
    assert "user(login:" in USER_REPOS_ONLY
    assert "ownerAffiliations: OWNER" in USER_REPOS_ONLY


def test_org_repos_query_does_not_set_owner_affiliations() -> None:
    assert "organization(login:" in ORG_REPOS_ONLY
    assert "ownerAffiliations" not in ORG_REPOS_ONLY


class TestPolicyFileReads:
    """Where the bot policy file is read, and where it must not be."""

    def test_owner_profile_tells_the_account_type(self) -> None:
        assert "repositoryOwner(login: $owner)" in OWNER_PROFILE
        assert "__typename" in OWNER_PROFILE

    def test_owner_profile_reads_the_dotgithub_file_and_optionally_a_repo(
        self,
    ) -> None:
        assert 'dotgithub: repository(name: ".github")' in OWNER_PROFILE
        assert "target: repository(name: $repo) @include(if: $withRepo)" in (
            OWNER_PROFILE
        )
        assert OWNER_PROFILE.count(f"policy: {_POLICY_READ}") == 2
        assert OWNER_PROFILE.count(_BLOB_FIELDS) == 2

    def test_only_a_first_pull_request_page_reads_the_repository_file(
        self,
    ) -> None:
        assert "$withPolicy: Boolean!" in REPO_OPEN_PRS_PAGE
        assert f"policy: {_POLICY_READ} @include(if: $withPolicy)" in (
            REPO_OPEN_PRS_PAGE
        )
        assert _BLOB_FIELDS in REPO_OPEN_PRS_PAGE

    @pytest.mark.parametrize("query", [ORG_REPOS_ONLY, USER_REPOS_ONLY])
    def test_repository_listings_never_read_it(self, query: str) -> None:
        # Measured at 1.3-3.9 s per 100-repository page; the first pull
        # request page carries each file for free instead.
        assert "object(" not in query
