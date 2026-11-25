# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The state-changing GitHub API calls behind ``GitHubClient``.

:class:`_GitHubActionMixin` carries the one synchronous operation that
writes: updating an out-of-date branch.  It swallows failures, logs a
warning and reports a boolean, so that a bulk run is never derailed by
one repository refusing it.  Merges and approvals go through the async
client only, where every write names the commit it acts on.

``GitHubAsync`` is imported inside each method, not at module scope, so
that patching it in its own module stays effective.  Every attribute this
mixin reads is established by ``GitHubClient.__init__``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from ..github_async import GitHubAsync

logger = logging.getLogger("dependamerge.github_client")


class _GitHubActionMixin:
    """State-changing GitHub operations shared into ``GitHubClient``."""

    # Established by GitHubClient.__init__.
    token: str
    host: str
    # Provided by GitHubClient: builds a transport client aimed at
    # ``host``, so an Enterprise run does not silently fall back to
    # github.com.  Annotated rather than defined, to avoid shadowing
    # the real method through the MRO.
    _new_async: Callable[..., GitHubAsync]

    def fix_out_of_date_pr(self, owner: str, repo: str, pr_number: int) -> bool:
        """Fix an out-of-date PR by updating the branch."""
        try:

            async def _run():
                async with self._new_async() as api:
                    await api.update_branch(owner, repo, pr_number)
                    return True

            return bool(asyncio.run(_run()))
        except Exception as e:
            logger.warning("Failed to update PR %s: %s", pr_number, e, exc_info=True)
            return False
