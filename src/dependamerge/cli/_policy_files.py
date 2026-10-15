# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Repository and owner bot policies, as a run applies them.

The files arrive with requests the run makes anyway: an owner's with the
query that tells an organisation from a user, a repository's with the
first page of its open pull requests.  The ``DEPENDAMERGE_BOTS``
variables cost a request each, so a run reads the owner's and those of
repositories holding an App's pull request, once it knows which.  A run
with a single source pull request reads its owner's and repository's
before deciding whether that pull request is in scope.  See
``docs/BOT_POLICY.md``.

An invalid owner policy stops the run before anything merges.  An
invalid repository policy puts none of that repository's pull requests
in scope; the run reports it and carries on with the others.  A token
that cannot read variables is reported once.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..bot_policy import BotPolicy, PolicyLevel, app_slug
from ..bot_policy_files import POLICY_VARIABLE, PolicyFiles, apply_policy_files
from ..error_codes import exit_for_configuration_error
from . import _bot_scope
from ._app import console

__all__ = [
    "_apply_policy_files",
    "_capture_policy_files",
    "_load_source_policy_files",
    "_policy_with_files",
    "_warn_unreadable_variables",
]

_FILE_LEVELS = (PolicyLevel.REPOSITORY, PolicyLevel.OWNER)

_KIND_NAMES = {"orgs": "the organisation's", "repos": "repositories'"}


def _unreadable_variables_message(kinds: frozenset[str]) -> str:
    """The one-time warning naming which variables the token cannot read."""
    which = " or ".join(_KIND_NAMES.get(kind, kind) for kind in sorted(kinds))
    return (
        f"⚠️ This token cannot read {which} {POLICY_VARIABLE} Actions "
        "variables: their inclusions do not apply and their exclusions cannot "
        "hold. Keep exclusions that must hold in policy files."
    )


def _combined(first: PolicyFiles, second: PolicyFiles) -> PolicyFiles:
    return PolicyFiles(
        owners={**first.owners, **second.owners},
        repositories={**first.repositories, **second.repositories},
        owner_variables={**first.owner_variables, **second.owner_variables},
        repository_variables={
            **first.repository_variables,
            **second.repository_variables,
        },
        variables_refused=first.variables_refused | second.variables_refused,
    )


def _capture_policy_files(ctx: Any, svc: Any) -> None:
    """Keep what a service has read of the policies on the run's context."""
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

    Exits with a configuration error when an owner's policy is invalid.
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


def _warn_unreadable_variables(ctx: Any, refused_before: frozenset[str]) -> None:
    """Say which variables the token cannot read, unless already said.

    ``refused_before`` is what the run knew before its latest read, so
    each kind is reported once.
    """
    newly = ctx.policy_files.variables_refused - refused_before
    if newly:
        console.print(_unreadable_variables_message(newly))


def _apply_policy_files(
    ctx: Any, *, refused_before: frozenset[str] = frozenset()
) -> None:
    """Apply the files and variables the run has read to its policy.

    Says, once per run, that the token cannot read the variables (see
    :func:`_warn_unreadable_variables`).  Exits with a configuration
    error when an owner's policy is invalid.
    """
    _warn_unreadable_variables(ctx, refused_before)
    ctx.bot_policy = _policy_with_files(
        ctx.bot_policy, ctx.policy_files, ctx.token, ctx.host
    )


def _load_source_policy_files(ctx: Any) -> None:
    """Apply the owner's and the source repository's policies to the run.

    Read before the source pull request's author is judged: both files
    in one request, and both variables.  A source repository whose
    policy is invalid has nothing in scope, so the run stops with a
    configuration error.
    """
    refused_before = ctx.policy_files.variables_refused
    files = ctx.github_client.get_policy_files(ctx.owner, ctx.repo_name)
    if isinstance(files, PolicyFiles):
        ctx.policy_files = _combined(ctx.policy_files, files)
    _apply_policy_files(ctx, refused_before=refused_before)
    repository = f"{ctx.owner}/{ctx.repo_name}"
    problem = ctx.bot_policy.repository_error(repository)
    if problem is not None:
        exit_for_configuration_error(
            message=f"❌ Invalid bot policy in {repository}",
            details=f"{problem}. Nothing in the repository is in scope.",
        )
