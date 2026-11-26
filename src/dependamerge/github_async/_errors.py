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

import asyncio
import json
import re
import time
from collections.abc import Collection
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

import httpx

if TYPE_CHECKING:
    from ..models import PullRequestInfo


class RateLimitError(Exception):
    """Raised when the primary GitHub API rate limit is reached."""


class SecondaryRateLimitError(Exception):
    """Raised when GitHub's secondary rate limit (abuse detection) triggers."""


class GraphQLError(Exception):
    """Raised for GraphQL errors returned by GitHub.

    ``errors`` holds the response's ``errors`` array when there was one,
    so a caller can tell where in the query each error arose.
    """

    def __init__(self, message: str, errors: list[Any] | None = None) -> None:
        super().__init__(message)
        self.errors: list[Any] = errors or []


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


def _approval_out_of_time(error: Exception, deadline: float | None) -> bool:
    """Whether an approval retried after a server error spent *deadline*.

    Asked after the review lookup before a retry, which can spend what
    the wait left.  A throttled approval, stored as an
    :class:`ApprovalError`, keeps its own waits and is never out of time.
    """
    if isinstance(error, ApprovalError) or deadline is None:
        return False
    return asyncio.get_running_loop().time() >= deadline


class RetryableError(Exception):
    """Internal exception to signal tenacity that a retry should occur.

    ``status_code`` is the HTTP status that asked for the retry, when
    one did.  It reaches callers once the retries are exhausted, so they
    can tell a server error from, say, a rate limit that never cleared.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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


def _only_allowed_not_found(errors: Any, allowed: Collection[tuple[str, ...]]) -> bool:
    """Whether every GraphQL error is a ``NOT_FOUND`` at an ``allowed`` path.

    GitHub reports a missing object at a nested field as ``null`` there,
    plus an error naming the field's ``path``, alongside otherwise valid
    data.  Where the caller expects an object may be missing, that is an
    answer rather than a failure.
    """
    if not allowed or not isinstance(errors, list) or not errors:
        return False
    paths = {tuple(path) for path in allowed}
    return all(
        isinstance(error, dict)
        and error.get("type") == "NOT_FOUND"
        and isinstance(error.get("path"), list)
        and tuple(error["path"]) in paths
        for error in errors
    )


# A throttled approval (a 422 saying the review was "submitted too
# quickly") waits as GitHub advises for secondary limits: at least a
# minute, doubling per attempt, unless ``Retry-After`` says otherwise.
_APPROVE_THROTTLE_MAX_ATTEMPTS = 4
_APPROVE_THROTTLE_BASE_DELAY = 60.0

#: Waits before each further attempt of a merge or an approval that
#: failed on a server error: four attempts over about 26 seconds, so a
#: short GitHub outage is outlasted rather than retried inside.
_SERVER_ERROR_RETRY_WAITS: tuple[float, ...] = (2.0, 6.0, 18.0)

#: The approval's waits under ``--max-wait 0``, which allows no long
#: wait: three attempts, as before the longer schedule above.
_APPROVE_SHORT_RETRY_WAITS: tuple[float, ...] = (2.0, 4.0)

#: The HTTP status in an httpx error message, e.g. ``Client error '405
#: Method Not Allowed' for url '…'``.  The one place a status is read
#: out of message text, for errors that arrive without their response.
_HTTP_STATUS_RE = re.compile(r"'(\d{3}) [^']*' for url")

#: Transport failures in which GitHub never answered.
_NO_RESPONSE_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


def status_from_message(text: str) -> int | None:
    """The HTTP status an httpx error message names, or ``None``."""
    match = _HTTP_STATUS_RE.search(text)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class ServerFailure:
    """GitHub failing to answer, as opposed to judging the request.

    ``status`` is the 5xx GitHub sent, or ``None`` when no response
    arrived at all (a timeout or a connection failure).
    """

    status: int | None

    def reason(self, operation: str) -> str:
        """The reason reported for *operation* (``merging``, ``approving``)."""
        label = str(self.status) if self.status is not None else "no response"
        return f"GitHub returned a server error ({label}) while {operation}"


class GitHubServerError(Exception):
    """An operation failed because GitHub did not answer it.

    The message is the reason reported for the pull request: the
    status and the operation, with no URL or client text.
    """

    #: The pull request the error concerns when it is not the one the
    #: merge began with: dependabot's replacement for it.  Set by the code
    #: acting on the replacement, so the outcome names that pull request.
    pull_request: PullRequestInfo | None = None

    def __init__(self, failure: ServerFailure, operation: str) -> None:
        self.failure = failure
        super().__init__(failure.reason(operation))


def _error_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def server_failure(error: BaseException) -> ServerFailure | None:
    """The server error *error* records, or ``None`` when it is not one.

    A server error is any 5xx status, the transport's exhausted retries
    of one, and a timeout or connection failure.  The chain of causes
    is followed, because the merge layer wraps the client's error in
    one of its own; the first status found decides, so a refusal
    wrapped around a failed follow-up read is still a refusal.  Message
    text is read only when no exception in the chain carries a status.
    """
    chain = _error_chain(error)
    for link in chain:
        status = _http_status(link)
        if status is None and isinstance(link, RetryableError):
            status = link.status_code
        if status is not None:
            return ServerFailure(status) if status >= 500 else None
        if isinstance(link, _NO_RESPONSE_ERRORS):
            return ServerFailure(None)
    status = status_from_message(str(error))
    if status is not None and status >= 500:
        return ServerFailure(status)
    return None


def _is_transient_server_error(exc: Exception) -> bool:
    """Whether an approval that raised ``exc`` should be tried again."""
    return server_failure(exc) is not None


def _is_retryable_status(status: int, method: str | None = None) -> bool:
    """Whether the transport retries a response of ``status``.

    A 500 is retried only for a ``GET``: repeating a write that may
    have landed is unsafe, so writes leave a 500 to the operations
    that can check their own effect (the merge and the approval).
    """
    if status == 500:
        return method is not None and method.upper() == "GET"
    return status in (429, 502, 503, 504)
