# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Review thread query and mutation.

Review threads are the unit GitHub uses for unresolved review
conversations, and they are only reachable through GraphQL — there is no
REST equivalent for resolving one.  Fetching threads and resolving them
are two halves of the same operation, so they live together here.

A thread's comments are paged separately from the threads themselves:
:data:`GET_PR_REVIEW_THREADS` returns the first page of each thread's
comments with its ``pageInfo``, and :data:`GET_REVIEW_THREAD_COMMENTS`
reads the pages after it.
"""

from __future__ import annotations

# GraphQL mutation to resolve a review thread
RESOLVE_REVIEW_THREAD = """
mutation ResolveReviewThread($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread {
      id
      isResolved
    }
  }
}
"""

# GraphQL query to get review threads for a pull request
GET_PR_REVIEW_THREADS = """
query GetPullRequestReviewThreads($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 50, after: $cursor) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          isResolved
          isOutdated
          line
          originalLine
          diffSide
          startLine
          originalStartLine
          path
          comments(first: 100) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              id
              author {
                login
              }
              body
              createdAt
            }
          }
        }
      }
    }
  }
}
"""

# GraphQL query to read further pages of one review thread's comments
GET_REVIEW_THREAD_COMMENTS = """
query GetReviewThreadComments($threadId: ID!, $cursor: String) {
  node(id: $threadId) {
    ... on PullRequestReviewThread {
      comments(first: 100, after: $cursor) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          author {
            login
          }
          body
          createdAt
        }
      }
    }
  }
}
"""
