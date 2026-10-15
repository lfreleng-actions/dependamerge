# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Repository listing queries.

These are the lightweight "just the repositories" queries: no pull
request nodes, so a single page reveals the owner-wide repository total
and the archived/fork flags needed to decide what to scan.  The
``Organization`` and ``User`` variants are similar, but the user query
pins ``ownerAffiliations: OWNER`` so GitHub does not include repositories
from other owners where the user is a direct collaborator.  See
:mod:`dependamerge.github_graphql.pull_requests` for the heavier queries
that carry PR payloads.

``OWNER_PROFILE`` tells which of the two an owner needs, and carries the
owner's bot policy file with it.
"""

from __future__ import annotations

# Lightweight query to list repositories without PR nodes for accurate counting.
# totalCount is provided by the GitHub GraphQL API for free on connection
# objects, so the first page immediately reveals the org-wide repo total
# without requiring a separate counting pass.
#
# ``isFork`` is included so owner-wide bulk operations can exclude fork
# repositories without a second round-trip; existing consumers that only
# read ``nameWithOwner`` / ``isArchived`` simply ignore the extra field.
ORG_REPOS_ONLY = """
query($org: String!, $reposCursor: String) {
  organization(login: $org) {
    repositories(first: 100, after: $reposCursor, orderBy: { field: NAME, direction: ASC }) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        nameWithOwner
        isArchived
        isFork
      }
    }
  }
}
"""

# User-account counterpart of ORG_REPOS_ONLY.  The ``repositories``
# connection exists on both ``Organization`` and ``User``, but GitHub's
# default User.repositories ownerAffiliations is [OWNER, COLLABORATOR].
# Pinning OWNER prevents owner-wide scans from including direct-collaborator
# repositories that belong to other owners, and keeps totalCount aligned
# with the user's own repositories.  The ``$org`` variable name is retained
# for call-site uniformity even though it carries a user login here.
USER_REPOS_ONLY = """
query($org: String!, $reposCursor: String) {
  user(login: $org) {
    repositories(first: 100, after: $reposCursor, ownerAffiliations: OWNER, orderBy: { field: NAME, direction: ASC }) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        nameWithOwner
        isArchived
        isFork
      }
    }
  }
}
"""

# Whether an owner is an organization or a user, in one inexpensive
# request, with the bot policy files a run needs before it lists anything:
# the owner's, in its ``.github`` repository, and optionally one
# repository's.  Each file is read from the default branch.
#
# ``repositoryOwner`` is null, without an error, for an unknown login.  A
# missing repository is null *with* a NOT_FOUND error at its alias, which
# the caller must tolerate.  ``policy`` is null when the file is absent,
# and has none of the Blob fields when the path is a directory.
OWNER_PROFILE = """
query($owner: String!, $repo: String!, $withRepo: Boolean!) {
  repositoryOwner(login: $owner) {
    __typename
    dotgithub: repository(name: ".github") {
      policy: object(expression: "HEAD:.github/dependamerge/config.json") {
        ... on Blob { text byteSize isTruncated isBinary }
      }
    }
    target: repository(name: $repo) @include(if: $withRepo) {
      policy: object(expression: "HEAD:.github/dependamerge/config.json") {
        ... on Blob { text byteSize isTruncated isBinary }
      }
    }
  }
}
"""
