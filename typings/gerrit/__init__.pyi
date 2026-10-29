# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
# Narrow stub for python-gerrit-api 3.1.0, which ships no py.typed marker.
# It declares only the surface dependamerge uses.

from typing import Any

from requests import Session

from gerrit.utils.requester import Requester

class GerritClient:
    session: Session
    requester: Requester
    auth_suffix: str
    def __init__(
        self,
        base_url: str,
        username: str | None = None,
        password: str | None = None,
        use_netrc: bool = False,
        ssl_verify: bool = True,
        cert: Any = None,
        cookies: Any = None,
        cookie_jar: Any = None,
        timeout: float = 60,
        max_retries: Any = None,
        session: Session | None = None,
        auth_suffix: str = "/a",
    ) -> None: ...
    def get_endpoint_url(self, endpoint: str) -> str: ...
