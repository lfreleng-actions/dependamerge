# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Repository listing queries.

These are the lightweight "just the repositories" queries: no pull
request nodes, only each repository's open pull request count, so a
single page reveals the owner-wide total of non-archived repositories
and the fork flag needed to decide what to scan.  Both variants pass
``isArchived: false`` so GitHub leaves archived repositories out of the
nodes and the total alike.  The
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
# ``isArchived: false`` makes GitHub leave archived repositories out of
# both the nodes and ``totalCount``: nothing is scanned there, so the
# total is the count of repositories a scan can act on.  ``isArchived``
# stays in the selection so the client can still refuse a node that
# reports itself archived.
#
# ``isFork`` is included so owner-wide bulk operations can exclude fork
# repositories without a second round-trip; existing consumers that only
# read ``nameWithOwner`` / ``isArchived`` simply ignore the extra field.
#
# ``pullRequests(states: OPEN) { totalCount }`` lets the scans skip a
# repository with no open pull request instead of asking for its first
# page: one point per listing page instead of two per repository.
# ``$withCounts: false`` leaves the count out, for a page GitHub cannot
# count in: the field is non-null, so an error there nulls the node.
ORG_REPOS_ONLY = """
query($org: String!, $reposCursor: String, $withCounts: Boolean = true) {
  organization(login: $org) {
    repositories(first: 100, after: $reposCursor, isArchived: false, orderBy: { field: NAME, direction: ASC }) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        nameWithOwner
        isArchived
        isFork
        pullRequests(states: OPEN) @include(if: $withCounts) {
          totalCount
        }
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
query($org: String!, $reposCursor: String, $withCounts: Boolean = true) {
  user(login: $org) {
    repositories(first: 100, after: $reposCursor, isArchived: false, ownerAffiliations: OWNER, orderBy: { field: NAME, direction: ASC }) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        nameWithOwner
        isArchived
        isFork
        pullRequests(states: OPEN) @include(if: $withCounts) {
          totalCount
        }
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
# and has none of the Blob fields when the path is a directory.  The
# ``.github`` file is read only ``$withPolicy``: commands that do not act
# on pull requests (``status``, ``blocked``) need no access to it.
OWNER_PROFILE = """
query($owner: String!, $repo: String!, $withRepo: Boolean!, $withPolicy: Boolean!) {
  repositoryOwner(login: $owner) {
    __typename
    dotgithub: repository(name: ".github") @include(if: $withPolicy) {
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
