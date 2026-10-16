# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Shared helpers for reconciling GitHub bot/App actor identities.

GitHub reports the *same* App actor under two different login forms
depending on the API surface:

- The REST API returns the suffixed form, e.g. ``dependabot[bot]``.
- The GraphQL API returns the bare form for a ``Bot`` actor, e.g.
  ``dependabot`` (the ``[bot]`` suffix is dropped).

This split has historically been worked around ad-hoc in several
places (the PR comparator, the Gerrit comparator, repo-merge automation
detection).  Centralising it here gives every consumer one canonical
form and one set of predicates, so a downstream equality check can never
again silently misclassify a bot just because it arrived via GraphQL.

Two layers are provided:

- :func:`canonical_bot_login` and :func:`rest_author_login` normalise an
  actor login *to the REST form* at the data boundary (where an API
  payload is converted to an internal model).  They are driven by the
  account type --- GraphQL ``__typename``, REST ``user.type`` --- so they
  apply uniformly to *every* App actor, not just one identity.
- :func:`normalize_bot_login` and :func:`is_dependabot` compare logins
  irrespective of the ``[bot]`` suffix, so a stray non-canonical value
  cannot disable identity-specific handling.

Whether an author is a bot is decided by account type alone, never by
name: a User account called ``renovate`` is a person.  Canonical logins
make that cheap to check afterwards, because GitHub user logins cannot
contain brackets, so only an App can carry the ``[bot]`` suffix.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BOT_SUFFIX = "[bot]"

# GraphQL ``__typename`` and REST ``user.type`` value for an App actor.
_BOT_TYPENAME = "Bot"

# REST names a few Apps by a display login rather than their account:
# Copilot's coding agent and its code reviewer both arrive as ``Copilot``.
# Only the account ID tells them apart; GraphQL reports their real
# logins, so these map REST onto the same canonical form.
_REST_LOGIN_BY_BOT_ID: Mapping[int, str] = {
    198982749: "copilot-swe-agent[bot]",
    175728472: "copilot-pull-request-reviewer[bot]",
}

# Known GitHub Copilot actor logins (normalised).  Copilot reports under
# several identities depending on the surface: the App author
# (``Copilot`` / ``copilot[bot]`` / ``github-copilot[bot]``), the coding
# agent that raises automated code fixes (``copilot-swe-agent[bot]``), and
# the review actor (``copilot-pull-request-reviewer``).
_COPILOT_NAMES = frozenset(
    {
        "copilot",
        "github-copilot",
        "copilot-swe-agent",
        "copilot-pull-request-reviewer",
    }
)


def canonical_bot_login(login: str | None, typename: str | None = None) -> str:
    """Return ``login`` in the canonical REST form.

    When ``typename`` identifies a GraphQL ``Bot`` actor whose login lacks
    the ``[bot]`` suffix (the bare form GraphQL returns), append it so the
    value matches the REST API form used throughout the codebase.  Every
    other input is returned unchanged.

    Args:
        login: The actor login as returned by the API (may be ``None``).
        typename: The GraphQL ``__typename`` for the actor, when known.
            Only ``"Bot"`` triggers suffixing; this keeps the
            normalisation actor-type driven rather than name-matching a
            specific bot.

    Returns:
        The canonical login, or ``"unknown"`` when ``login`` is empty.
    """
    if not login:
        return "unknown"
    if typename == _BOT_TYPENAME and not login.endswith(BOT_SUFFIX):
        return f"{login}{BOT_SUFFIX}"
    return login


def rest_author_login(user: Mapping[str, Any] | None) -> str:
    """Return the canonical login for a REST ``user`` object.

    REST already names most Apps ``<slug>[bot]``, but not every one, so
    the ``type`` field decides rather than the name, and an App REST
    reports by display login is resolved through its account ID.  A User
    account's login is returned unchanged, so it can never acquire the
    suffix that marks an App.

    Args:
        user: The ``user`` object of a REST pull request payload.

    Returns:
        The canonical login, or an empty string when there is none.
    """
    if not user:
        return ""
    login = user.get("login") or ""
    if not login or user.get("type") != _BOT_TYPENAME:
        return login
    return _REST_LOGIN_BY_BOT_ID.get(user.get("id") or 0) or canonical_bot_login(
        login, _BOT_TYPENAME
    )


def normalize_bot_login(login: str | None) -> str:
    """Return a lower-cased login with any ``[bot]`` suffix removed.

    Use this for comparisons that must treat ``dependabot`` and
    ``dependabot[bot]`` (and any other suffixed/bare bot pair) as equal,
    regardless of which API surface produced the value.
    """
    if not login:
        return ""
    normalized = login.lower()
    if normalized.endswith(BOT_SUFFIX):
        normalized = normalized[: -len(BOT_SUFFIX)]
    return normalized


def is_dependabot(author: str | None) -> bool:
    """Return True when ``author`` is dependabot, in either login form.

    Robust to the REST (``dependabot[bot]``) and GraphQL (``dependabot``)
    forms so dependabot-specific recovery (rebase / recreate macros) is
    never skipped because of a non-canonical login.
    """
    return normalize_bot_login(author) == "dependabot"


def is_copilot(author: str | None) -> bool:
    """Return True when ``author`` is a GitHub Copilot actor.

    Matches every Copilot identity (App author and review actor) in both
    the REST and GraphQL login forms, so Copilot-specific handling (review
    dismissal, block-reason analysis) cannot be bypassed by a
    non-canonical login.
    """
    return normalize_bot_login(author) in _COPILOT_NAMES


def is_automation_author(author: str | None) -> bool:
    """Return True when ``author`` is a GitHub App.

    Expects a canonical login, as :func:`canonical_bot_login` and
    :func:`rest_author_login` produce, in which every App carries the
    ``[bot]`` suffix.  A bare login is a person whatever it is called:
    names such as ``renovate`` or ``copilot-swe-agent`` can belong to
    User accounts, and one of those exists.
    """
    if not author:
        return False
    return author.lower().endswith(BOT_SUFFIX)
