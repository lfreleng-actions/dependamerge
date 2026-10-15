# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
The bot policy files a run reads, at no cost of their own.

An owner's file arrives with the profile query that tells an
organization from a user, and a repository's with the first page of
its open pull requests.  Both are kept as GraphQL returned them and
offered through :attr:`policy_files`, with the ``DEPENDAMERGE_BOTS``
variables read alongside (see :mod:`._policy_variables`); making a
policy of them is :mod:`dependamerge.bot_policy_files`' concern.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, TypeVar

from ..bot_policy import BotPolicy
from ..bot_policy_files import PolicyFiles, apply_policy_files
from ..github_graphql import ORG_REPOS_ONLY, OWNER_PROFILE, USER_REPOS_ONLY
from ._base import _GitHubServiceBase

# Where the profile query may name a repository that does not exist: an
# owner need not have a ``.github`` repository, nor the target exist.
_MAY_BE_MISSING = (("repositoryOwner", "dotgithub"), ("repositoryOwner", "target"))

_V = TypeVar("_V")


def _policy_blob(repository: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The policy file a repository node carries, or ``None`` for none."""
    policy = (repository or {}).get("policy")
    return dict(policy) if isinstance(policy, Mapping) else None


def _entry(values: Mapping[str, _V], key: str) -> dict[str, _V]:
    """``key`` and its value alone, or nothing when ``values`` lacks it."""
    return {key: values[key]} if key in values else {}


class _PolicyFilesMixin(_GitHubServiceBase):
    """Policy-file collection for ``GitHubService``."""

    @property
    def policy_files(self) -> PolicyFiles:
        """Every policy file and variable read so far, as a snapshot."""
        return PolicyFiles(
            owners=MappingProxyType(dict(self._owner_policy_files)),
            repositories=MappingProxyType(dict(self._repository_policy_files)),
            owner_variables=MappingProxyType(dict(self._owner_policy_variables)),
            repository_variables=MappingProxyType(
                dict(self._repository_policy_variables)
            ),
            variables_refused=frozenset(self._policy_variables_refused),
        )

    async def fetch_policy_files(
        self, owner: str, repo: str | None = None
    ) -> PolicyFiles:
        """Read ``owner``'s policy file and variable, and ``repo``'s too.

        One request reads both files, for a run that needs them before it
        lists or fetches anything; listing ``owner``'s repositories
        afterwards repeats none of it.  Each variable costs one more.  A
        missing owner or repository adds nothing.

        Returns:
            :attr:`policy_files`, including what this read.
        """
        await self._read_owner_profile(owner, repo, with_policy=True)
        repository = f"{owner}/{repo}"
        exists = (
            repo is not None and repository.lower() in self._repository_policy_files
        )
        await self._read_policy_variables(owner, [repository] if exists else [])
        return self.policy_files

    def _record_repository_policy(
        self, repository: str, node: Mapping[str, Any] | None
    ) -> None:
        """Keep the policy file in ``repository``'s node, if it asked for one."""
        if node is not None and "policy" in node:
            self._repository_policy_files[repository.lower()] = _policy_blob(node)

    def _with_repository_policy(self, policy: BotPolicy, repository: str) -> BotPolicy:
        """``policy``, also consulting ``repository``'s own file and variable.

        The file is the one its first page of pull requests carried, and
        the variable whatever has been read of it; ``policy`` is returned
        unchanged when neither has.
        """
        key = repository.lower()
        files = PolicyFiles(
            repositories=_entry(self._repository_policy_files, key),
            repository_variables=_entry(self._repository_policy_variables, key),
        )
        if not files.repositories and not files.repository_variables:
            return policy
        applied, _ = apply_policy_files(policy, files)
        return applied

    async def _read_owner_profile(
        self, owner: str, repo: str | None = None, *, with_policy: bool | None = None
    ) -> None:
        """Run the owner profile query, and keep everything it says.

        Caches the owner's account type for :meth:`_resolve_owner_root`,
        taking an owner that does not resolve for a user, as before, so
        the user listing reports it.  Records the owner's policy file
        when ``with_policy`` (by default, when the service reads policy),
        and ``repo``'s when given and it exists.
        """
        if with_policy is None:
            with_policy = self._read_policies
        variables = {
            "owner": owner,
            "repo": repo or "",
            "withRepo": repo is not None,
            "withPolicy": with_policy,
        }
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
        if with_policy:
            self._owner_policy_files[owner.lower()] = _policy_blob(
                profile.get("dotgithub")
            )
        target = profile.get("target")
        if repo is not None and target is not None:
            self._record_repository_policy(f"{owner}/{repo}", target)
