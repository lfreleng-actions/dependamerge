# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Exception types and transient-failure predicates for the async client.

The exception classes, the retry/rate-limit predicates and the retry
budget constants used across the ``GitHubAsync`` mixins.  Split out of
``dependamerge.github_async`` unchanged so the client's transport,
permission and merge code can share them without a cycle.
"""

from __future__ import annotations

import json
import time
from typing import (
    Any,
)


class RateLimitError(Exception):
    """Raised when the primary GitHub API rate limit is reached."""


class SecondaryRateLimitError(Exception):
    """Raised when GitHub's secondary rate limit (abuse detection) triggers."""


class GraphQLError(Exception):
    """Raised for GraphQL errors returned by GitHub."""


class PermissionError(Exception):
    """Raised when GitHub API returns a permission/authorization error.

    Attributes:
        operation: The operation that failed (e.g., 'approve', 'merge', 'close')
        message: Human-readable error message
        token_type_guidance: Guidance for both classic and fine-grained tokens
        status_code: HTTP status of the refusal, when known
    """

    def __init__(
        self,
        operation: str,
        message: str,
        token_type_guidance: dict[str, str] | None = None,
        status_code: int | None = None,
    ):
        self.operation = operation
        self.token_type_guidance = token_type_guidance or {}
        self.status_code = status_code
        super().__init__(message)


class ApprovalError(Exception):
    """GitHub refused an approval for a reason other than permissions.

    Carries GitHub's own explanation verbatim, so the run summary shows
    why the review was rejected instead of a guess about the token.

    Attributes:
        status_code: HTTP status GitHub answered with
        detail: GitHub's message, as read from the response body
    """

    def __init__(self, target: str, status_code: int | None, detail: str):
        self.status_code = status_code
        self.detail = detail
        status = f"HTTP {status_code}" if status_code is not None else "an error"
        reason = detail or "no reason given"
        super().__init__(f"GitHub refused to approve {target} ({status}): {reason}")


class RetryableError(Exception):
    """Internal exception to signal tenacity that a retry should occur."""


# Ceiling of tenacity's ``wait_random_exponential`` on ``_request``.  Named
# so the secondary-rate-limit path can subtract it from a ``Retry-After``
# sleep rather than stacking a full sleep on top of the retry backoff.
_TENACITY_MAX_BACKOFF = 10.0


def _now() -> float:
    return time.time()


def _is_secondary_rate_limited(body_text: str) -> bool:
    text = body_text.lower()
    # GitHub may return messages like:
    # "You have exceeded a secondary rate limit. Please wait a few minutes..."
    # Or "abuse detection mechanism"
    return "secondary rate limit" in text or "abuse detection" in text


def _http_status(error: BaseException) -> int | None:
    """The HTTP status carried by ``error``, or ``None`` when it has none.

    Read from the response rather than the message: httpx puts the
    request URL in the message, so ``/pulls/403/reviews`` would match a
    substring test for ``403``.
    """
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _error_entry_text(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        message = entry.get("message")
        if isinstance(message, str):
            return message
        code = entry.get("code")
        if isinstance(code, str) and code != "custom":
            field = entry.get("field")
            return f"{field}: {code}" if isinstance(field, str) else code
    return ""


def github_error_detail(error: BaseException) -> str:
    """Extract GitHub's explanation from a failed request.

    GitHub puts the actual reason in the response body: ``message``
    plus an ``errors`` list whose entries are strings or objects with a
    ``message`` field.  The ``HTTPStatusError`` text carries only the
    status line, so without this the real cause is lost.  Whitespace is
    collapsed so the result fits on one status line.

    Returns an empty string when no detail could be extracted.
    """
    response = getattr(error, "response", None)
    if response is None:
        return ""
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        parts: list[str] = []
        message = body.get("message")
        if isinstance(message, str) and message.strip():
            parts.append(message)
        errors = body.get("errors")
        if isinstance(errors, list):
            parts.extend(t for t in map(_error_entry_text, errors) if t.strip())
        if parts:
            return "; ".join(" ".join(p.split()) for p in parts)
    try:
        raw = getattr(response, "text", "") or ""
        return " ".join(str(raw).split())[:500]
    except Exception:
        return ""


# Phrases in a 422 body that mean GitHub throttled content creation
# rather than rejected the request on its merits.
_THROTTLE_PHRASES = ("submitted too quickly", "secondary rate limit", "abuse detection")

# Phrases in an approval 422 body that mean the approval is not allowed
# for this token at all, as opposed to failing for this one request.
_APPROVAL_NOT_PERMITTED_PHRASES = (
    "approve your own pull request",
    "review cannot be requested from pull request author",
)


def _is_throttled_detail(detail: str) -> bool:
    text = detail.lower()
    return any(phrase in text for phrase in _THROTTLE_PHRASES)


def _is_approval_not_permitted(detail: str) -> bool:
    text = detail.lower()
    return any(phrase in text for phrase in _APPROVAL_NOT_PERMITTED_PHRASES)


def _retry_after_seconds(error: BaseException) -> float | None:
    """A numeric ``Retry-After`` carried by ``error``'s response, if any."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After")
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_primary_rate_limited(body_text: str) -> bool:
    text = body_text.lower()
    return "api rate limit exceeded" in text


def _is_transient_graphql_error(errors: Any) -> bool:
    try:
        # The structure is usually a list of dicts with "message".
        message_blob = json.dumps(errors).lower()
    except Exception:
        message_blob = str(errors).lower()
    # Heuristics for retryable GraphQL responses
    return any(
        needle in message_blob
        for needle in [
            "rate limit",  # may appear in graphql errors as well
            "something went wrong",  # generic GH error
            "timeout",
            "internal server error",
            "network timeout",
        ]
    )


# Approve-specific retry policy.  ``POST .../reviews`` returns transient
# 500 often enough to matter in bulk runs, but a blanket retry of every
# POST is unsafe, so this is applied only where duplicate-suppression is
# possible (see ``approve_pull_request``).
_APPROVE_MAX_ATTEMPTS = 3
_APPROVE_RETRY_BASE_DELAY = 2.0

# A throttled approval (a 422 saying the review was "submitted too
# quickly") waits as GitHub advises for secondary limits: at least a
# minute, doubling per attempt, unless ``Retry-After`` says otherwise.
_APPROVE_THROTTLE_MAX_ATTEMPTS = 4
_APPROVE_THROTTLE_BASE_DELAY = 60.0

# Server-side statuses worth retrying for an operation that can verify
# its own effect afterwards.  Note 500 is intentionally *not* in
# ``_is_retryable_status``: generic retries must not replay arbitrary
# non-idempotent writes.
# Statuses the outer approve retry handles.  Deliberately **only** 500:
# ``_request`` already retries 429/502/503/504 (``_is_retryable_status``)
# plus transport and rate-limit errors via tenacity, six attempts each.
# Including those here too would nest the loops --- up to 18 requests and
# two sets of backoff sleeps for one approval --- which is precisely the
# API-budget waste this work is trying to remove.  500 is the one status
# ``_request`` does not retry, because a blanket replay of failed POSTs
# is unsafe; it is safe *here* only because this call can verify its own
# effect first (see ``approve_pull_request``).
_TRANSIENT_SERVER_STATUSES = frozenset({500})


def _is_transient_server_error(exc: Exception) -> bool:
    """Whether ``exc`` is a server-side failure the outer retry should own.

    Anything already covered by ``_request``'s tenacity policy returns
    ``False`` here: by the time such an exception surfaces it has been
    retried six times, and trying again adds cost without adding hope.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in _TRANSIENT_SERVER_STATUSES


def _is_retryable_status(status: int) -> bool:
    # Treat common transient statuses as retryable.
    return status in (429, 502, 503, 504)
