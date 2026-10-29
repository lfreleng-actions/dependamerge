# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Transport helpers behind ``GerritRestClient``.

The Gerrit REST transport is python-gerrit-api's ``GerritClient`` running
on a Requests session.  This module holds the pieces that adapt it to the
contracts ``GerritRestClient`` has always offered:

- :class:`GerritSession`: a Requests session that keeps proxy and CA
  bundle discovery from the environment, but never picks up credentials
  implicitly from a netrc file, neither on the first request nor when it
  follows a redirect.  Credentials are resolved once, by
  ``build_client``, so ``is_authenticated`` always describes what is sent.
- :class:`SessionPool`: hands each concurrent request its own backend, so
  parallel submissions never share a Requests session between threads.
- :func:`decode_response`: Gerrit's XSSI-guarded JSON decoding, accepting
  both LF and CRLF after the guard.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Generic, TypeVar

import requests
from requests.models import PreparedRequest, Request, Response

_XSSI_GUARD = re.compile(r"\A\)\]\}'\r?\n")

_T = TypeVar("_T")


def _no_auth(request: PreparedRequest) -> PreparedRequest:
    """Requests auth hook that adds nothing to the request."""
    return request


class GerritSession(requests.Session):
    """Requests session that never reads credentials from netrc itself.

    Requests looks up ``~/.netrc`` (or ``$NETRC``) whenever a request
    carries no auth, and again on every redirect.  Turning ``trust_env``
    off would stop that, but would also drop the proxy and CA bundle
    settings taken from the environment.  This session keeps
    ``trust_env`` and blocks only the two netrc lookups.
    """

    def prepare_request(self, request: Request) -> PreparedRequest:
        """Prepare a request, without the implicit netrc lookup."""
        if not request.auth and not self.auth:
            request.auth = _no_auth
        return super().prepare_request(request)

    def rebuild_auth(
        self, prepared_request: PreparedRequest, response: Response
    ) -> None:
        """Strip credentials on an unsafe redirect; never add netrc ones."""
        headers = prepared_request.headers
        url = prepared_request.url
        previous = response.request.url
        if (
            "Authorization" in headers
            and url is not None
            and previous is not None
            and self.should_strip_auth(previous, url)
        ):
            del headers["Authorization"]


class SessionPool(Generic[_T]):
    """Pool of backends, one per concurrent request.

    A backend is taken for the duration of one request and returned
    afterwards, so no two threads ever use the same Requests session at
    once.  The pool grows to the peak concurrency and no further.
    """

    def __init__(self, factory: Callable[[], _T], closer: Callable[[_T], None]):
        self._factory = factory
        self._closer = closer
        self._lock = threading.Lock()
        self._idle: list[_T] = []
        self._all: list[_T] = []
        self._closed = False

    @contextmanager
    def acquire(self) -> Iterator[_T]:
        """Lend a backend to the caller for one request."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Gerrit client is closed")
            backend = self._idle.pop() if self._idle else None
        if backend is None:
            backend = self._factory()
            with self._lock:
                self._all.append(backend)
        try:
            yield backend
        finally:
            with self._lock:
                if self._closed:
                    self._closer(backend)
                else:
                    self._idle.append(backend)

    def __len__(self) -> int:
        with self._lock:
            return len(self._all)

    def close(self) -> None:
        """Close every idle backend; busy ones close when returned."""
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
        for backend in idle:
            self._closer(backend)


def decode_response(response: Response) -> Any:
    """Decode a Gerrit response body.

    JSON responses lose Gerrit's ``)]}'`` XSSI guard, followed by LF or
    CRLF, and are parsed.  Empty bodies and non-JSON content types come
    back as stripped text, as they always have; the services check shapes
    and reject text where they expect a list or an object.

    Raises:
        ValueError: The response claims to be JSON but is not.
    """
    content = response.content.decode(response.encoding or "utf-8").strip()
    if not content:
        return content
    content_type = response.headers.get("content-type", "")
    if content_type.split(";")[0].strip().lower() != "application/json":
        return content
    return json.loads(_XSSI_GUARD.sub("", content, count=1))


__all__ = ["GerritSession", "SessionPool", "decode_response"]
