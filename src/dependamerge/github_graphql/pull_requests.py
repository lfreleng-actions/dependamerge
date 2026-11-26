# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
The open pull request page query.

``REPO_OPEN_PRS_PAGE`` reads one page of a repository's open pull
requests with the payload the scans read — mergeability, base/head refs
and repositories, and four nested lists: changed files, recent comments,
reviews and the latest commit's status check rollup — so a scan does not
have to fall back to per-PR REST calls.

Each nested list has its own ``@include`` switch, so a scan asks only
for the lists it reads, and each carries its total, so a reader can tell
a short page from the whole list.  The nested sizes are fixed: only the
pull request page size is a variable.
"""

from __future__ import annotations

#: The ``files(first: ...)`` page size.
SCAN_FILES_PAGE_SIZE = 50

#: The ``comments(first: ...)`` page size.
SCAN_COMMENTS_PAGE_SIZE = 10

#: The ``reviews(first: ...)`` page size.  The connection's
#: ``totalCount`` rides along, so a reader can tell a pull request with
#: more reviews than this from one with exactly this many.
SCAN_REVIEWS_PAGE_SIZE = 20

#: The status check rollup's ``contexts(first: ...)`` page size.  The
#: connection's per-state counts ride along, exact whatever the length
#: of the list.
SCAN_CHECKS_PAGE_SIZE = 100

# Read one page of a repository's open PRs; pass the previous page's
# cursor to continue.  ``$withFiles``, ``$withComments``, ``$withReviews``
# and ``$withChecks`` select the nested lists; an unselected list is absent
# from each node, not empty.
# The first page also asks for the repository's bot policy file, on its
# default branch, which rides along at no extra cost (``$withPolicy``);
# see OWNER_PROFILE in :mod:`.repos` for the shape of ``policy``.
_REPO_OPEN_PRS_PAGE_TEMPLATE = """
query($owner: String!, $name: String!, $prsCursor: String, $prsPageSize: Int!, $withPolicy: Boolean!, $withFiles: Boolean!, $withComments: Boolean!, $withReviews: Boolean!, $withChecks: Boolean!) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    policy: object(expression: "HEAD:.github/dependamerge/config.json") @include(if: $withPolicy) {
      ... on Blob { text byteSize isTruncated isBinary }
    }
    pullRequests(
      states: OPEN
      first: $prsPageSize
      after: $prsCursor
      orderBy: { field: CREATED_AT, direction: DESC }
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        id
        number
        title
        body
        url
        isDraft
        author { __typename login }
        mergeable
        mergeStateStatus
        baseRefName
        headRefName
        headRefOid
        headRepository { nameWithOwner url isFork }
        baseRepository { nameWithOwner url }
        createdAt
        updatedAt
        files(first: <files>) @include(if: $withFiles) {
          totalCount
          nodes {
            path
            additions
            deletions
          }
        }
        comments(first: <comments>, orderBy: { field: UPDATED_AT, direction: DESC }) @include(if: $withComments) {
          totalCount
          nodes {
            databaseId
            author { login }
            body
            createdAt
          }
        }
        reviews(first: <reviews>, states: [PENDING, COMMENTED, APPROVED, CHANGES_REQUESTED]) @include(if: $withReviews) {
          totalCount
          nodes {
            id
            author { login }
            state
            body
            createdAt
            updatedAt
          }
        }

        commits(last: 1) @include(if: $withChecks) {
          nodes {
            commit {
              oid
              statusCheckRollup {
                state
                contexts(first: <checks>) {
                  checkRunCount
                  checkRunCountsByState { state count }
                  statusContextCount
                  statusContextCountsByState { state count }
                  nodes {
                    __typename
                    ... on CheckRun {
                      name
                      status
                      conclusion
                      startedAt
                      completedAt
                    }
                    ... on StatusContext {
                      context
                      state
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""


def _with_page_sizes(template: str) -> str:
    """``template`` with each ``<list>`` placeholder set to its page size."""
    for name, size in (
        ("files", SCAN_FILES_PAGE_SIZE),
        ("comments", SCAN_COMMENTS_PAGE_SIZE),
        ("reviews", SCAN_REVIEWS_PAGE_SIZE),
        ("checks", SCAN_CHECKS_PAGE_SIZE),
    ):
        template = template.replace(f"<{name}>", str(size))
    return template


REPO_OPEN_PRS_PAGE = _with_page_sizes(_REPO_OPEN_PRS_PAGE_TEMPLATE)
