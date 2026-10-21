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
that cannot read a kind of variable is reported once, and only when that
kind could decide something no file covers: see
:func:`_warn_unreadable_variables`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from ..bot_identity import is_automation_author
from ..bot_policy import BUILTIN_BOTS, BotPolicy, PolicyLevel, app_slug
from ..bot_policy_files import POLICY_VARIABLE, PolicyFiles, apply_policy_files
from ..error_codes import exit_for_configuration_error
from . import _bot_scope
from ._app import console

__all__ = [
    "_apply_policy_files",
    "_capture_policy_files",
    "_captured_app_prs",
    "_keep_admitted_candidates",
    "_load_source_policy_files",
    "_policy_with_files",
    "_warn_unreadable_variables",
]

_FILE_LEVELS = (PolicyLevel.REPOSITORY, PolicyLevel.OWNER)

_KIND_NAMES = {"orgs": "the organisation", "repos": "repository"}

_KIND_NOUNS = {"orgs": "variable", "repos": "variables"}

_POLICY_FILE = ".github/dependamerge/config.json"


def _unreadable_variables_message(kind: str) -> str:
    """The warning that the token cannot read ``kind``'s variables."""
    which = _KIND_NAMES.get(kind, kind)
    noun = _KIND_NOUNS.get(kind, "variables")
    return (
        f"⚠️ Token cannot read {which} {POLICY_VARIABLE} {noun}\n"
        "   Revise the token permission, or create a configuration file\n"
        f"   to suppress this warning: {_POLICY_FILE}"
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


def _captured_app_prs(svc: Any) -> list[Any]:
    """The App pull requests a service's searches saw, before any filter.

    They decide whether an unreadable variable is worth a warning (see
    :func:`_warn_unreadable_variables`), whether or not they match.
    """
    seen = getattr(svc, "seen_app_prs", None)
    return list(seen) if isinstance(seen, tuple | list) else []


def _check_file_apps(policy: BotPolicy, token: str | None, host: str) -> BotPolicy:
    """Confirm the Apps the files name exist, once each.

    Apps the run's flags or the user's file name were looked up before
    anything was fetched.  An unknown exclusion in a repository's file
    makes that file invalid; in the owner's file it stops the run.  An
    unknown inclusion includes nothing, so it only warns.
    """
    listed = policy.listed()
    from_files = [item for item in listed if item.level in _FILE_LEVELS]
    # Inclusions the flags or the user's file named were looked up
    # already.  Exclusions in files are looked up regardless: a name the
    # run only warned about as an inclusion may be a misspelt exclusion
    # here, which must invalidate the file.
    checked = {item.login for item in listed if item.level not in _FILE_LEVELS}
    logins = sorted(
        {
            item.login
            for item in from_files
            if item.excluded or item.login not in checked
        }
    )
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


def _needs_variables(files: PolicyFiles, kind: str, prs: Iterable[Any]) -> bool:
    """Whether a refused ``kind`` of variable could decide one of ``prs``.

    Only an App outside :data:`BUILTIN_BOTS` depends on a policy to be in
    scope, and a policy file at the refused level covers it: the owner's
    for ``orgs``, the pull request's repository's for ``repos``.
    """
    for pr in prs:
        author = (getattr(pr, "author", None) or "").lower()
        if not is_automation_author(author) or author in BUILTIN_BOTS:
            continue
        repository = (getattr(pr, "repository_full_name", None) or "").lower()
        if kind == "orgs":
            blob = files.owners.get(repository.split("/", 1)[0])
        else:
            blob = files.repositories.get(repository)
        if not blob:
            return True
    return False


def _warn_unreadable_variables(ctx: Any, prs: Iterable[Any] = ()) -> None:
    """Say which variables the token cannot read, where it matters.

    A refused kind is reported only when one of ``prs``, the pull
    requests the run has seen, is an App's outside the built-in ones and
    no policy file covers that level; each kind at most once per run.
    """
    files: PolicyFiles = ctx.policy_files
    warned: frozenset[str] = getattr(ctx, "policy_variables_warned", frozenset())
    pending = sorted(files.variables_refused - warned)
    if not pending:
        return
    prs = list(prs)
    for kind in pending:
        if _needs_variables(files, kind, prs):
            console.print(_unreadable_variables_message(kind))
            warned |= {kind}
    ctx.policy_variables_warned = warned


def _apply_policy_files(ctx: Any, prs: Iterable[Any] = ()) -> None:
    """Apply the files and variables the run has read to its policy.

    ``prs`` are the pull requests the run has seen, which decide whether
    an unreadable variable is worth a warning (see
    :func:`_warn_unreadable_variables`).  Exits with a configuration
    error when an owner's policy is invalid.
    """
    _warn_unreadable_variables(ctx, prs)
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
    files = ctx.github_client.get_policy_files(ctx.owner, ctx.repo_name)
    if isinstance(files, PolicyFiles):
        ctx.policy_files = _combined(ctx.policy_files, files)
    source = getattr(ctx, "source_pr", None)
    _apply_policy_files(ctx, [source] if source is not None else [])
    repository = f"{ctx.owner}/{ctx.repo_name}"
    problem = ctx.bot_policy.repository_error(repository)
    if problem is not None:
        exit_for_configuration_error(
            message=f"❌ Invalid bot policy in {repository}",
            details=f"{problem}. Nothing in the repository is in scope.",
        )


def _keep_admitted_candidates(
    policy: BotPolicy, candidates: list[tuple[Any, Any]]
) -> list[tuple[Any, Any]]:
    """``candidates`` the checked ``policy`` admits, the rest reported.

    The search judged each repository by its unchecked file; checking can
    still invalidate one (an exclusion naming no App), and every PR in an
    invalid repository, a person's included, is then out of scope.
    """
    kept = {
        id(pr)
        for pr in _bot_scope._remove_out_of_scope_bots(
            policy, [pr for pr, _ in candidates]
        )
    }
    return [entry for entry in candidates if id(entry[0]) in kept]
