# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The request and retry machinery behind ``GerritRestClient``.

:class:`_GerritRequestMixin` carries the two methods that issue Gerrit
REST calls: the bounded retry loop and the single-shot request that
dispatches to python-gerrit-api and translates failures into the
:mod:`dependamerge.gerrit._client_errors` hierarchy.  HTTP errors are read
from the response itself, so they keep their exact status and body.

It lives here rather than in ``dependamerge.gerrit.client`` purely to
keep that module reviewable.  Nothing in here calls ``GerritClient`` (it
appears only in annotations) or ``get_credentials_for_host``: those names
are only resolved in ``client``'s own namespace, so that patching them
there stays effective.  Every attribute this mixin reads is established by
``GerritRestClient.__init__``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Final

from requests.exceptions import RequestException

from ._client_errors import (
    _RETRYABLE_HTTP_CODES,
    GerritAuthError,
    GerritNotFoundError,
    GerritRestError,
    _Auth,
    _calculate_backoff,
    _extract_status_code,
    _is_transient_error,
)
from ._client_transport import SessionPool, decode_response

if TYPE_CHECKING:
    from gerrit import GerritClient

log = logging.getLogger("dependamerge.gerrit.client")

_METHODS: Final[frozenset[str]] = frozenset({"GET", "POST", "PUT", "DELETE"})

# Longest stretch of an error body quoted in an exception message; the
# whole body stays available as ``response_body``.
_MAX_ERROR_DETAIL: Final[int] = 500


class _GerritRequestMixin:
    """Request/retry behaviour shared into ``GerritRestClient``."""

    # Established by GerritRestClient.__init__.
    _backends: SessionPool[GerritClient]
    _timeout: float
    _max_attempts: int
    _auth: _Auth | None

    def _request_with_retry(
        self,
        method: str,
        path: str,
        data: Any | None = None,
    ) -> Any:
        """Perform a request with automatic retry on transient failures."""
        last_exception: Exception | None = None

        for attempt in range(self._max_attempts):
            try:
                return self._request(method, path, data)
            except GerritAuthError:
                # Don't retry authentication failures
                raise
            except GerritNotFoundError:
                # Don't retry not found errors
                raise
            except GerritRestError as exc:
                last_exception = exc
                # Check if this is a retryable HTTP error or transient network error
                is_retryable_http = (
                    exc.status_code and exc.status_code in _RETRYABLE_HTTP_CODES
                )
                # An HTTP failure is classified by its status alone: its
                # message now quotes the server's body, which may say anything.
                is_transient = exc.status_code is None and _is_transient_error(exc)

                if is_retryable_http or is_transient:
                    if attempt < self._max_attempts - 1:
                        delay = _calculate_backoff(attempt)
                        if exc.status_code:
                            log.warning(
                                "Gerrit REST %s %s failed (HTTP %d), "
                                "retrying in %.1fs (attempt %d/%d)",
                                method,
                                path,
                                exc.status_code,
                                delay,
                                attempt + 1,
                                self._max_attempts,
                            )
                        else:
                            log.warning(
                                "Gerrit REST %s %s failed (%s), "
                                "retrying in %.1fs (attempt %d/%d)",
                                method,
                                path,
                                exc,
                                delay,
                                attempt + 1,
                                self._max_attempts,
                            )
                        time.sleep(delay)
                        continue
                raise
            except Exception as exc:
                last_exception = exc
                if _is_transient_error(exc):
                    if attempt < self._max_attempts - 1:
                        delay = _calculate_backoff(attempt)
                        log.warning(
                            "Gerrit REST %s %s failed (%s), "
                            "retrying in %.1fs (attempt %d/%d)",
                            method,
                            path,
                            exc,
                            delay,
                            attempt + 1,
                            self._max_attempts,
                        )
                        time.sleep(delay)
                        continue
                raise GerritRestError(
                    f"Gerrit REST {method} {path} failed: {exc}"
                ) from exc

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise GerritRestError(f"Gerrit REST {method} {path} failed unexpectedly")

    def _request(
        self,
        method: str,
        path: str,
        data: Any | None = None,
    ) -> Any:
        """Perform a single HTTP request (no retry) using python-gerrit-api."""
        if not path:
            raise GerritRestError("path is required")
        if method not in _METHODS:
            raise GerritRestError(f"Unsupported HTTP method: {method}")

        # One leading slash, so the endpoint always lands under the base
        # path (and the /a/ prefix) rather than replacing it.
        api_path = "/" + path.lstrip("/")

        log.debug(
            "Gerrit REST %s %s (auth=%s)",
            method,
            api_path,
            "yes" if self._auth else "no",
        )

        headers = {"Accept": "application/json"}
        kwargs: dict[str, Any] = {}
        if data is not None and method in ("POST", "PUT"):
            # Serialised here, not handed over as a dict: the library would
            # form-encode a dict, and drop an empty one entirely.
            kwargs["data"] = json.dumps(data)
            headers["Content-Type"] = "application/json; charset=UTF-8"

        try:
            with self._backends.acquire() as backend:
                url = backend.get_endpoint_url(api_path)
                send = getattr(backend.requester, method.lower())
                response = send(url, headers=headers, raise_for_status=False, **kwargs)
        except RequestException as exc:
            raise GerritRestError(
                f"Gerrit REST {method} {path} failed: {exc}",
                status_code=_extract_status_code(exc),
            ) from exc

        if response.status_code >= 400:
            raise _http_error(method, path, response.status_code, response.text)

        try:
            return decode_response(response)
        except ValueError as exc:
            raise GerritRestError(
                f"Gerrit REST {method} {path} returned invalid JSON: {exc}",
                status_code=response.status_code,
            ) from exc


def _http_error(method: str, path: str, status: int, body: str) -> GerritRestError:
    """Map an HTTP error status to the client's exception hierarchy."""
    if status == 401:
        return GerritAuthError(
            f"Authentication failed for {path}",
            status_code=status,
            response_body=body,
        )
    if status == 403:
        return GerritAuthError(
            f"Access forbidden for {path}",
            status_code=status,
            response_body=body,
        )
    if status == 404:
        return GerritNotFoundError(
            f"Resource not found: {path}",
            status_code=status,
            response_body=body,
        )
    detail = body.strip()
    if len(detail) > _MAX_ERROR_DETAIL:
        detail = detail[:_MAX_ERROR_DETAIL] + "..."
    message = f"Gerrit REST {method} {path} failed: HTTP {status}"
    if detail:
        message = f"{message}: {detail}"
    return GerritRestError(message, status_code=status, response_body=body)
