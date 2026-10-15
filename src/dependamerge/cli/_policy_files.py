# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Repository and owner bot policy files, as a run applies them.

The files arrive with requests the run makes anyway: an owner's with the
query that tells an organisation from a user, a repository's with the
first page of its open pull requests.  A run with a single source pull
request reads both in one request before deciding whether that pull
request is in scope.  See ``docs/BOT_POLICY.md``.

An invalid owner file stops the run before anything merges.  An invalid
repository file puts none of that repository's pull requests in scope;
the run reports it and carries on with the others.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..bot_policy import BotPolicy, PolicyLevel, app_slug
from ..bot_policy_files import PolicyFiles, apply_policy_files
from ..error_codes import exit_for_configuration_error
from . import _bot_scope
from ._app import console

__all__ = [
    "_capture_policy_files",
    "_load_source_policy_files",
    "_policy_with_files",
]

_FILE_LEVELS = (PolicyLevel.REPOSITORY, PolicyLevel.OWNER)


def _combined(first: PolicyFiles, second: PolicyFiles) -> PolicyFiles:
    return PolicyFiles(
        owners={**first.owners, **second.owners},
        repositories={**first.repositories, **second.repositories},
    )


def _capture_policy_files(ctx: Any, svc: Any) -> None:
    """Keep the policy files a service has read on the run's context."""
    files = getattr(svc, "policy_files", None)
    if isinstance(files, PolicyFiles):
        ctx.policy_files = _combined(ctx.policy_files, files)


def _check_file_apps(policy: BotPolicy, token: str | None, host: str) -> BotPolicy:
    """Confirm the Apps the files name exist, once each.

    Apps the run's flags or the user's file name were looked up before
    anything was fetched.  An unknown exclusion in a repository's file
    makes that file invalid; in the owner's file it stops the run.  An
    unknown inclusion includes nothing, so it only warns.
    """
    listed = policy.listed()
    checked = {item.login for item in listed if item.level not in _FILE_LEVELS}
    from_files = [item for item in listed if item.level in _FILE_LEVELS]
    logins = sorted({item.login for item in from_files} - checked)
    if not logins:
        return policy
    found = _bot_scope._lookup_apps(logins, token, host)
    invalid = dict(policy.invalid_repositories)
    for item in from_files:
        if found.get(item.login) is not False:
            continue
        problem = f"no GitHub App is named {app_slug(item.login)}"
        if not item.excluded:
            console.print(f"⚠️ {item.label}: {problem}; its inclusion has no effect")
        elif item.level is PolicyLevel.OWNER:
            exit_for_configuration_error(
                message=f"❌ Invalid bot policy in {item.label}",
                details=f"{problem}. Nothing was merged.",
            )
        elif item.repository is not None:
            invalid[item.repository.lower()] = problem
    return replace(policy, invalid_repositories=invalid)


def _policy_with_files(
    policy: BotPolicy, files: PolicyFiles, token: str | None, host: str
) -> BotPolicy:
    """Return ``policy`` also consulting ``files``, checked.

    Exits with a configuration error when an owner's file is invalid.
    """
    applied, owner_errors = apply_policy_files(policy, files)
    if owner_errors:
        exit_for_configuration_error(
            message=f"❌ Invalid bot policy in {owner_errors[0].source}",
            details="\n".join(
                f"{error.source}: {problem}"
                for error in owner_errors
                for problem in error.problems
            ),
        )
    return _check_file_apps(applied, token, host)


def _load_source_policy_files(ctx: Any) -> None:
    """Apply the owner's and the source repository's files to the run.

    Read in one request before the source pull request's author is
    judged.  A source repository whose file is invalid has nothing in
    scope, so the run stops with a configuration error.
    """
    files = ctx.github_client.get_policy_files(ctx.owner, ctx.repo_name)
    if isinstance(files, PolicyFiles):
        ctx.policy_files = _combined(ctx.policy_files, files)
    ctx.bot_policy = _policy_with_files(
        ctx.bot_policy, ctx.policy_files, ctx.token, ctx.host
    )
    repository = f"{ctx.owner}/{ctx.repo_name}"
    problem = ctx.bot_policy.repository_error(repository)
    if problem is not None:
        exit_for_configuration_error(
            message=f"❌ Invalid bot policy in {repository}",
            details=f"{problem}. Nothing in the repository is in scope.",
        )
