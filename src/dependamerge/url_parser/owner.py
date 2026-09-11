# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
What may be an owner, and how to say so when it may not.

One decision and one wording, reached by every route an owner arrives
on: a bare argument, an owner URL, a repository URL, and both pull
request parsers.  They did not agree before this module existed ---
``not a url`` was refused as a bare owner and accepted through a URL,
then sent to the API as ``orgs/not%20a%20url`` and answered with a 404,
where the identical input given bare named the problem at once.

Separated from :mod:`repos`, which answers "what does this URL address".
This answers the narrower question the parsers ask of one segment, and
is pure, so both the rule and its guidance are directly testable.
"""

from __future__ import annotations

from urllib.parse import unquote

from .host_config import DEFAULT_GITHUB_HOST
from .hosts import _host_matches
from .models import UrlParseError
from .shorthand import looks_like_owner

__all__ = ["require_owner", "require_owner_from_path"]

#: Longest a GitHub login can be.  Enterprise directories are not bound
#: by the dotcom grammar, but they are bound by the column this maps to.
_MAX_OWNER_LENGTH = 39

#: Characters that change what a URL *means* when a segment carrying one
#: is interpolated into a path unescaped.  ``/`` separates path
#: components, ``?`` begins a query, ``#`` a fragment, ``\`` is folded to
#: a separator by some parsers, and ``%`` reintroduces an escape --- the
#: route a doubly-encoded separator would take to arrive intact after
#: one round of decoding.
#:
#: An owner is interpolated into REST paths throughout the client, so a
#: name carrying one of these does not merely fail: ``acme?admin=true``
#: turns ``/orgs/{owner}/repos`` into ``/orgs/acme`` with a query, which
#: is a different request rather than a failing one.  github.com cannot
#: reach this, its grammar being alphanumerics and hyphens; a directory
#: that issues arbitrary names can.
_URL_STRUCTURAL = frozenset("/?#%\\")


def _is_safe_path_segment(name: str) -> bool:
    """Whether *name* can be placed in a URL path as it stands."""
    return not any(
        character in _URL_STRUCTURAL
        or character.isspace()
        or ord(character) < 0x20
        or ord(character) == 0x7F
        for character in name
    )


def _acceptable_owner(name: str, host: str) -> bool:
    """Whether *name* could be a login **on that host**.

    github.com gets the full grammar, which is what makes the bare and
    URL forms of a dotcom target answer identically --- the point of the
    gate.

    An Enterprise host gets only the checks no directory can fail.
    Accounts there are provisioned over LDAP or SAML and need not follow
    the dotcom grammar at all, as the comment beside ``_OWNER_RE``
    records, so applying it would report ``team_name`` as invalid on an
    install where it is somebody's actual login.  What is refused is
    over-length input and anything that would not survive being placed
    in a URL path --- neither of which a directory produces, and both of
    which change the request rather than failing it.
    """
    if _host_matches(host, DEFAULT_GITHUB_HOST):
        return looks_like_owner(name)
    return bool(name) and len(name) <= _MAX_OWNER_LENGTH and _is_safe_path_segment(name)


def _owner_guidance(host: str) -> str:
    """What to tell an operator whose owner was refused, on *that* host.

    One message for both hosts would have to describe the stricter rule,
    and would then contradict the validator: an Enterprise name refused
    for whitespace or length would be told to remove underscores that
    its own directory issues.  Guidance that names the wrong fault sends
    the reader to edit something that was never the problem.
    """
    if _host_matches(host, DEFAULT_GITHUB_HOST):
        return "Logins are alphanumerics and hyphens, at most 39 characters."
    return (
        f"Logins on {host} follow that install's own directory, but cannot "
        "contain whitespace, or characters that would change what a URL "
        "means, and are at most 39 characters."
    )


def require_owner(name: str, host: str) -> str:
    """Return *name*, or refuse it as a login on *host*.

    The single decision and the single wording, so the bare and URL
    forms of the same argument cannot answer differently.  They did:
    ``lfreleng-actions`` and ``https://github.com/lfreleng-actions`` are
    the same request, but only the first was checked, so ``not a url``
    was refused as a bare owner and accepted through a URL --- reaching
    the API as ``orgs/not%20a%20url`` and coming back 404, where the
    operator saw a not-found error instead of the fail-fast message.

    Surrounding whitespace is refused rather than trimmed.
    :func:`looks_like_owner` strips before matching, which suits the
    shorthand expansion it was written for, but accepting the stripped
    form here and returning the original would send ``" acme"`` to the
    API --- the same bypass by a quieter route.  A bare argument is
    already stripped by its caller, so this only bites a URL path
    segment, where whitespace is never meant.

    Echoing the segment is safe.  An owner is a *path* segment, so it
    cannot carry the userinfo or query string a credential would hide
    in, and the host has already been checked by the time this runs.
    """
    if name != name.strip() or not _acceptable_owner(name, host):
        raise UrlParseError(
            f"Not a valid GitHub owner name: {name!r}. {_owner_guidance(host)}"
        )
    return name


def require_owner_from_path(segment: str, host: str) -> str:
    """Decode a URL path segment, then gate it as a login.

    A path segment is percent-encoded, so ``not%20a%20url`` carries no
    literal whitespace and slipped through a check written for the
    decoded form --- the reported bypass, merely spelled differently.

    Decoding also keeps ``lfreleng%2Dactions`` working.  Nothing in this
    codebase decodes an owner, so the escape reaches GitHub verbatim and
    GitHub resolves it; gating the raw segment would refuse a URL that
    does address a real account.

    The *decoded* value is returned, which is the login itself.  The
    transport re-encodes what needs it, so the request is unchanged for
    an ordinary name and corrected for an encoded one.

    Bare CLI arguments deliberately do not come through here.  A shell
    can pass a literal ``%``, and decoding it would silently retarget
    the run.
    """
    return require_owner(unquote(segment), host)
