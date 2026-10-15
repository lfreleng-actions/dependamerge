# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
The bot policy files a run reads, at no cost of their own.

An owner's file arrives with the profile query that tells an
organization from a user, and a repository's with the first page of
its open pull requests.  Both are kept as GraphQL returned them and
offered through :attr:`policy_files`; making a policy of them is
:mod:`dependamerge.bot_policy_files`' concern.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..bot_policy import BotPolicy
from ..bot_policy_files import PolicyFiles, apply_policy_files
from ..github_graphql import ORG_REPOS_ONLY, OWNER_PROFILE, USER_REPOS_ONLY
from ._base import _GitHubServiceBase

# Where the profile query may name a repository that does not exist: an
# owner need not have a ``.github`` repository, nor the target exist.
_MAY_BE_MISSING = (("repositoryOwner", "dotgithub"), ("repositoryOwner", "target"))


def _policy_blob(repository: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The policy file a repository node carries, or ``None`` for none."""
    policy = (repository or {}).get("policy")
    return dict(policy) if isinstance(policy, Mapping) else None


class _PolicyFilesMixin(_GitHubServiceBase):
    """Policy-file collection for ``GitHubService``."""

    @property
    def policy_files(self) -> PolicyFiles:
        """Every policy file read so far, as an immutable snapshot."""
        return PolicyFiles(
            owners=MappingProxyType(dict(self._owner_policy_files)),
            repositories=MappingProxyType(dict(self._repository_policy_files)),
        )

    async def fetch_policy_files(
        self, owner: str, repo: str | None = None
    ) -> PolicyFiles:
        """Read ``owner``'s policy file, and ``repo``'s too when given.

        One request, for a run that needs the files before it lists or
        fetches anything; listing ``owner``'s repositories afterwards
        repeats none of it.  A missing owner or repository adds nothing.

        Returns:
            :attr:`policy_files`, including what this read.
        """
        await self._read_owner_profile(owner, repo)
        return self.policy_files

    def _record_repository_policy(
        self, repository: str, node: Mapping[str, Any] | None
    ) -> None:
        """Keep the policy file in ``repository``'s node, if it asked for one."""
        if node is not None and "policy" in node:
            self._repository_policy_files[repository.lower()] = _policy_blob(node)

    def _with_repository_policy(self, policy: BotPolicy, repository: str) -> BotPolicy:
        """``policy``, also consulting ``repository``'s own file.

        The file is the one its first page of pull requests carried;
        ``policy`` is returned unchanged when no such page was read.
        """
        key = repository.lower()
        if key not in self._repository_policy_files:
            return policy
        files = PolicyFiles(repositories={key: self._repository_policy_files[key]})
        applied, _ = apply_policy_files(policy, files)
        return applied

    async def _read_owner_profile(self, owner: str, repo: str | None = None) -> None:
        """Run the owner profile query, and keep everything it says.

        Caches the owner's account type for :meth:`_resolve_owner_root`,
        taking an owner that does not resolve for a user, as before, so
        the user listing reports it.  Records the owner's policy file,
        and ``repo``'s when given and it exists.
        """
        variables = {"owner": owner, "repo": repo or "", "withRepo": repo is not None}
        data = await self._api.graphql(
            OWNER_PROFILE, variables, allow_not_found=_MAY_BE_MISSING
        )
        profile = (data or {}).get("repositoryOwner") or {}
        if profile.get("__typename") == "Organization":
            self._owner_root_cache[owner] = ("organization", ORG_REPOS_ONLY)
        else:
            self._owner_root_cache[owner] = ("user", USER_REPOS_ONLY)
        if not profile:
            return
        self._owner_policy_files[owner.lower()] = _policy_blob(profile.get("dotgithub"))
        target = profile.get("target")
        if repo is not None and target is not None:
            self._record_repository_policy(f"{owner}/{repo}", target)
