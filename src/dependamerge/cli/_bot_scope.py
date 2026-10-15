# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The run's bot policy: which GitHub Apps it may act on.

Built from the run's ``--include-bots`` / ``--exclude-bots`` flags and the
user's ``config.json`` (see :mod:`dependamerge.bot_policy`), checked so a
misspelt App cannot quietly leave a bot in scope, and applied to every
pull request the run finds.  Pull requests from Apps outside the policy
are listed with the reason, never silently dropped.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Iterable
from urllib.parse import quote

import typer

# Reached through the module, never bound here: tests substitute
# ``GitHubAsync`` elsewhere in this package (tests/test_patch_targets.py).
import dependamerge.github_async as _github_async

from ..bot_identity import is_automation_author
from ..bot_policy import (
    BotPolicy,
    BotPolicyError,
    app_slug,
    load_user_policy,
    run_policy_source,
)
from ..error_codes import exit_for_configuration_error
from ..models import PullRequestInfo
from ..url_parser import default_github_host, derive_api_urls
from ._app import console

__all__ = [
    "EXCLUDE_BOTS_OPTION",
    "INCLUDE_BOTS_OPTION",
    "_build_bot_policy",
    "_check_listed_bots",
    "_lookup_apps",
    "_out_of_scope_hint",
    "_remove_out_of_scope_bots",
]

#: The ``--include-bots`` option, shared by every command that applies a
#: bot policy so they cannot describe it differently.
INCLUDE_BOTS_OPTION = typer.Option(
    None,
    "--include-bots",
    help=(
        "Act on PRs from these GitHub Apps as well as the built-in "
        "Dependabot, Renovate and pre-commit.ci. Comma-separated and/or "
        "repeatable, each 'name', 'name[bot]' or 'app/name'; '*' "
        "includes every App"
    ),
)

#: The ``--exclude-bots`` option.
EXCLUDE_BOTS_OPTION = typer.Option(
    None,
    "--exclude-bots",
    help=(
        "Never act on PRs from these GitHub Apps, built-in ones included. "
        "Comma-separated and/or repeatable. A name matching no App stops "
        "the run before anything merges"
    ),
)


def _build_bot_policy(
    include_bots: Iterable[str] | None,
    exclude_bots: Iterable[str] | None,
) -> BotPolicy:
    """Build the run's policy from its flags and the user's file.

    Exits with a configuration error, before anything is fetched, when
    either source is invalid.
    """
    # A direct Python call to a command (as some tests make) passes
    # Typer's unresolved ``OptionInfo`` default, which must read as "not
    # given", as ``_resolve_repo_selection`` treats its lists.
    if not isinstance(include_bots, list):
        include_bots = None
    if not isinstance(exclude_bots, list):
        exclude_bots = None
    try:
        run = run_policy_source(include_bots, exclude_bots)
        user = load_user_policy()
    except BotPolicyError as exc:
        exit_for_configuration_error(
            message=f"❌ Invalid bot policy in {exc.source}",
            details="\n".join(exc.problems),
        )
    return BotPolicy(run=run, user=user)


async def _app_exists(api: _github_async.GitHubAsync, login: str) -> bool | None:
    """Whether a GitHub App with ``login`` exists; ``None`` when unknown."""
    try:
        data = await api.get(f"/users/{quote(login, safe='')}")
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return False if status == 404 else None
    return isinstance(data, dict) and data.get("type") == "Bot"


def _lookup_apps(
    logins: list[str], token: str | None, host: str
) -> dict[str, bool | None]:
    """Look each login up once; the value is ``None`` when unknown."""

    async def _run() -> dict[str, bool | None]:
        api_url, graphql_url = derive_api_urls(host or default_github_host())
        async with _github_async.GitHubAsync(
            token=token, api_url=api_url, graphql_url=graphql_url
        ) as api:
            found = await asyncio.gather(*(_app_exists(api, login) for login in logins))
        return dict(zip(logins, found, strict=True))

    return asyncio.run(_run())


def _check_listed_bots(policy: BotPolicy, token: str | None, host: str) -> None:
    """Confirm every App the policy names exists.

    An unknown exclusion stops the run, as an unmatched ``--exclude-repos``
    name does: a misspelt exclusion leaves in scope the very App it was
    meant to keep out.  An unknown inclusion includes nothing, so it only
    warns.  A lookup that fails for any other reason is not held against
    the name.
    """
    listed = policy.listed()
    if not listed:
        return
    found = _lookup_apps(sorted({login for login, _, _ in listed}), token, host)
    unknown_exclusions = []
    for login, excluded, label in listed:
        if found.get(login) is not False:
            continue
        if excluded:
            unknown_exclusions.append(f"{app_slug(login)} (in {label})")
        else:
            console.print(
                f"⚠️ No GitHub App is named {app_slug(login)}; "
                f"its inclusion in {label} has no effect"
            )
    if unknown_exclusions:
        exit_for_configuration_error(
            message="❌ Excluded bot(s) match no GitHub App",
            details=(
                "Correct or remove: "
                + ", ".join(unknown_exclusions)
                + ". Nothing was merged."
            ),
        )


def _out_of_scope_hint(login: str) -> str:
    """How to bring an App into scope, for messages."""
    return (
        f"Include it with --include-bots {app_slug(login)}, "
        "or in a policy file (see docs/BOT_POLICY.md)"
    )


def _remove_out_of_scope_bots(
    policy: BotPolicy, prs: list[PullRequestInfo]
) -> list[PullRequestInfo]:
    """Return ``prs`` without those from Apps outside the policy.

    The removed pull requests are listed, grouped by App and reason, so
    an App that needs including is visible rather than silently skipped.
    People are left in place: whether their pull requests are acted on
    is decided separately.
    """
    kept: list[PullRequestInfo] = []
    skipped: Counter[tuple[str, str]] = Counter()
    for pr in prs:
        if is_automation_author(pr.author):
            decision = policy.decide(pr.author, pr.repository_full_name)
            if not decision.in_scope:
                skipped[(pr.author, decision.reason)] += 1
                continue
        kept.append(pr)

    if skipped:
        console.print(f"🚫 Bot PRs out of scope: {sum(skipped.values())}")
        for (author, reason), count in sorted(skipped.items()):
            console.print(f"   {author} ×{count} ({reason})")
        console.print(
            "   Include an App with --include-bots <name>, or in a policy file"
        )
    return kept
