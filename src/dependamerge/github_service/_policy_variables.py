# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
The ``DEPENDAMERGE_BOTS`` Actions variables a run reads.

A repository or owner may hold its bot policy in that variable as well
as in a file.  GraphQL does not expose variables, so each costs a REST
request: the owner's is read once, through the organisation endpoint,
and a repository's when it holds an App's pull request, since otherwise
its policy decides nothing.

A token may not be allowed to read them, and may be refused one kind
yet allowed the other: a classic token without ``admin:org`` cannot read
an organisation's variables but can read a repository's.  The first
refusal of a kind stops every later read of that kind, and
:attr:`policy_files` records it so the run can say so once: such a
variable's inclusions do not apply, and its exclusions cannot be
enforced.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from collections.abc import Iterable

import httpx

from ..bot_identity import is_automation_author
from ..bot_policy_files import POLICY_VARIABLE, PolicyFiles
from ..github_async import RateLimitError, SecondaryRateLimitError
from ..models import PullRequestInfo
from ._base import _GitHubServiceBase

# The statuses that say this token may not read variables.
_REFUSED = frozenset({401, 403})


class _PolicyVariablesMixin(_GitHubServiceBase):
    """Policy-variable reads for ``GitHubService``."""

    async def fetch_policy_variables(
        self, owner: str, prs: Iterable[PullRequestInfo] = ()
    ) -> None:
        """Read the owner's variable, and those of repositories with App PRs.

        For a run that has its pull requests: a repository holding no
        App's pull request has no use for its policy.  The repositories'
        are read concurrently, after the owner's, and nothing is read
        twice.  What this reads joins :attr:`policy_files`.
        """
        repositories = {
            pr.repository_full_name for pr in prs if is_automation_author(pr.author)
        }
        await self._read_policy_variables(owner, sorted(repositories))

    def adopt_policy_variables(self, files: PolicyFiles) -> None:
        """Take the variables a run has read already as read here.

        So a service the run starts later reads none of them again, and
        none at all once the run found that it cannot.
        """
        self._owner_policy_variables.update(files.owner_variables)
        self._repository_policy_variables.update(files.repository_variables)
        self._policy_variables_refused |= files.variables_refused

    async def _read_policy_variables(
        self, owner: str, repositories: Iterable[str] = ()
    ) -> None:
        """Read ``owner``'s variable, then each of ``repositories``'."""
        key = owner.lower()
        root = self._owner_root_cache.get(owner)
        # A user account has no organisation variables to read.
        if key not in self._owner_policy_variables and (
            root is None or root[0] == "organization"
        ):
            await self._read_variable(
                f"/orgs/{owner}", self._owner_policy_variables, key
            )
        await asyncio.gather(
            *(self._read_repository_variable(name) for name in repositories)
        )

    async def _read_repository_variable(self, repository: str) -> None:
        """Read ``repository``'s variable, unless the run has already."""
        key = repository.lower()
        if key in self._repository_policy_variables:
            return
        # Bounded, and the refusal checked once a slot is free, so reads
        # still queued when the first refusal arrives are never sent.
        async with self._page_semaphore:
            await self._read_variable(
                f"/repos/{repository}", self._repository_policy_variables, key
            )

    async def _read_variable(
        self, scope: str, read: dict[str, str | None], key: str
    ) -> None:
        """Keep the variable ``scope`` holds as ``read[key]``.

        ``None`` when there is none, or when reading it fails for any
        reason but a refusal or a rate limit.  A refusal records that the
        run cannot read this kind of variable, and keeps nothing.
        """
        kind = scope.split("/", 2)[1]  # "orgs" or "repos"
        if kind in self._policy_variables_refused:
            return
        path = f"{scope}/actions/variables/{POLICY_VARIABLE}"
        try:
            body = await self._api.get(path)
        except (RateLimitError, SecondaryRateLimitError):
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in _REFUSED:
                self._policy_variables_refused.add(kind)
                return
            if status != 404:
                self.log.debug("Taking %s as unset: HTTP %s", path, status)
            read[key] = None
            return
        except Exception as exc:
            self.log.debug("Taking %s as unset: %s", path, exc)
            read[key] = None
            return
        value = body.get("value") if isinstance(body, dict) else None
        read[key] = value if isinstance(value, str) else None
