# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
# Narrow stub for python-gerrit-api 3.1.0; see ../__init__.pyi.

from typing import Any

from requests import Response, Session

class Requester:
    session: Session
    timeout: float
    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_redirects: bool = True,
        stream: bool = False,
        raise_for_status: bool = True,
        **kwargs: Any,
    ) -> Response: ...
    def post(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        data: Any = None,
        json: Any = None,
        files: Any = None,
        headers: dict[str, str] | None = None,
        allow_redirects: bool = True,
        raise_for_status: bool = True,
        **kwargs: Any,
    ) -> Response: ...
    def put(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        data: Any = None,
        json: Any = None,
        files: Any = None,
        headers: dict[str, str] | None = None,
        allow_redirects: bool = True,
        raise_for_status: bool = True,
        **kwargs: Any,
    ) -> Response: ...
    def delete(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        allow_redirects: bool = True,
        raise_for_status: bool = True,
        **kwargs: Any,
    ) -> Response: ...
