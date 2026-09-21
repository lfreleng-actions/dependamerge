# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Routing a target between GitHub and Gerrit by what the operator declared.

:mod:`gerrit_hosts` reads the Gerrit declarations and :mod:`hosts` the
GitHub ones; this module is where the two meet.  A declaration is a
statement about a host and outranks any guess about a URL's path shape,
and a host declared as both is refused rather than settled by precedence.
"""

from __future__ import annotations

from .gerrit_hosts import iter_gerrit_hosts, malformed_declaration_for
from .hosts import _host_matches, is_supported_github_host
from .models import HostDeclarationError, UrlParseError
from .redaction import redact_target

__all__ = [
    "is_declared_gerrit_host",
    "reject_conflicting_host_declaration",
    "reject_gerrit_on_github_host",
]


def is_declared_gerrit_host(host: str) -> bool:
    """Report whether the operator has declared ``host`` as Gerrit.

    The Gerrit half of :func:`is_supported_github_host`, with one
    difference: there is no equivalent of github.com, no hostname a
    Gerrit server is known by, so a declaration is the *only* way to
    answer yes.  Everything else falls back to the structural inference
    the parsers already perform on the URL path.

    Subdomains do not qualify, matching the treatment of a declared
    Enterprise host: an arbitrary hostname says nothing about what its
    subdomains run.

    An unusable Gerrit declaration is ignored rather than raised, the
    mirror of :func:`_declared_as_github`.  This is consulted while
    *routing*, which happens for every target --- so propagating the
    error would fail an ordinary GitHub pull request URL over a stale
    ``DEPENDAMERGE_GERRIT_HOSTS`` entry it never reads.  The value is
    still reported where it is genuinely needed: ``--gerrit-host`` is
    validated eagerly by ``apply_gerrit_host``, and
    :func:`~dependamerge.url_parser.gerrit_hosts.gerrit_hosts` raises
    for a caller that wants the whole declared set.

    Args:
        host: The hostname to check.

    Returns:
        True when the operator has named this host as a Gerrit server.
    """
    host = (host or "").strip().lower()
    if not host:
        return False
    try:
        return any(
            _host_matches(host, declared, allow_subdomains=False)
            for declared in iter_gerrit_hosts()
        )
    except UrlParseError:
        return False


def _declared_as_github(host: str) -> bool:
    """Whether *host* is a GitHub host, ignoring unusable configuration.

    :func:`is_supported_github_host` raises when a declaration names a
    port or is not a hostname.  Asked during *routing*, that would abort
    a Gerrit target over GitHub configuration it never consults --- the
    same reasoning ``local_repo._looks_like_gerrit_remote`` records for
    the local-checkout path, which swallows the identical error.  A
    GitHub target still reports the setting through the parsers that do
    need it.
    """
    try:
        return is_supported_github_host(host)
    except UrlParseError:
        return False


def reject_conflicting_host_declaration(host: str) -> None:
    """Refuse a host the operator has declared as both platforms.

    Reported rather than resolved by precedence.  Whichever way a
    precedence rule fell, it would silently ignore one of two explicit
    statements about where the operator's credentials may go, and the
    one ignored would be invisible at the point it mattered.  A
    contradiction in trusted configuration is a question for the
    operator, not something to settle with a rule.

    Only an *explicit* Gerrit declaration can conflict.  github.com is
    a GitHub host without anyone saying so, so declaring it as Gerrit is
    a contradiction the operator did author; a merely inferred Gerrit
    path shape is not a declaration and does not reach here.

    A malformed Gerrit declaration of this same host is reported here
    too, by :func:`malformed_declaration_for`: it is the same question,
    what the operator declared about this host, and reporting it only
    through routing would have ignored it.

    Raises:
        UrlParseError: The host is declared as both GitHub and Gerrit,
            or a Gerrit declaration of it is malformed.
    """
    malformed = malformed_declaration_for(host)
    if malformed is not None:
        raise HostDeclarationError(str(malformed)) from None
    if not is_declared_gerrit_host(host):
        return
    if not _declared_as_github(host):
        return
    raise HostDeclarationError(
        f"Host {host} is declared as both a GitHub host and a Gerrit host. "
        "Remove it from one of them: a host serves one platform, and "
        "guessing which declaration to honour would quietly ignore the "
        "other. GitHub declarations come from --github-host, "
        "DEPENDAMERGE_GITHUB_HOSTS, DEPENDAMERGE_GITHUB_HOST or GH_HOST; "
        "Gerrit declarations from --gerrit-host, DEPENDAMERGE_GERRIT_HOSTS "
        "or DEPENDAMERGE_GERRIT_HOST."
    )


def reject_gerrit_on_github_host(host: str, target: str) -> None:
    """Refuse Gerrit routing for a host declared as GitHub.

    The gap #478 reports: routing consulted the path shape alone, so a
    host declared as GitHub Enterprise was still handed to the Gerrit
    client whenever a URL happened to carry ``/c/…/+/`` or ``/q/topic:``
    --- and the operator saw a Gerrit credential error for a host they
    had explicitly named as GitHub.

    A declaration outranks a guess about the shape of a path, which is
    the rule ``local_repo._looks_like_gerrit_remote`` already applies to
    remotes.  Stronger Gerrit evidence still wins: an explicit Gerrit
    declaration is checked before this is reached.

    Args:
        host: The hostname the target names.
        target: The target, for the message.

    Raises:
        UrlParseError: The host is declared as a GitHub host.
    """
    if not _declared_as_github(host):
        return
    if _host_matches(host, "github.com"):
        # Intrinsic rather than declared, so there is nothing to remove.
        # Offering the usual remedy here sent the operator to delete a
        # declaration that does not exist --- and adding the Gerrit one
        # it suggests would only produce the "declared as both" refusal.
        raise HostDeclarationError(
            f"Refusing to treat {host} as a Gerrit server for "
            f"{redact_target(target)}: github.com is a GitHub host, and "
            "cannot be routed to Gerrit. Check the hostname --- a Gerrit "
            "change URL names its own server, as in "
            "gerrit.example.org/c/project/+/12345."
        )
    raise HostDeclarationError(
        f"Refusing to treat {host} as a Gerrit server for "
        f"{redact_target(target)}: it is a GitHub host. Declare it with "
        "--gerrit-host, DEPENDAMERGE_GERRIT_HOST or "
        "DEPENDAMERGE_GERRIT_HOSTS if it really is Gerrit, and remove it "
        "from the GitHub declarations. Hosts are declared rather than "
        "inferred so a mistyped URL cannot send your token somewhere "
        "unintended."
    )
