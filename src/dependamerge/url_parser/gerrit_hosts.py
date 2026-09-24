# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Which Gerrit hosts the operator has declared.

GitHub Enterprise hosts are declared before they are addressed; Gerrit
hosts were only ever *inferred*, from the shape of a URL path.  That
left the two halves of the same question answered by different
mechanisms, and a host declared as GitHub Enterprise was still routed
to Gerrit whenever the path happened to look Gerrit-shaped.

Declaration answers it with no network contact.  Probing was considered
and rejected in #478: an unauthenticated probe still contacts a host the
operator may have mistyped, which is the thing the declaration rule
exists to prevent, and an authenticated one leaks the token in order to
decide whether it should have been sent.

Deliberately **not** read here: the bare ``GERRIT_HOST`` variable.  It
is already in use by ``merge_manager/_gerrit_submit.py`` for a different
question --- which Gerrit server a GitHub2Gerrit change should be
submitted on, resolved from ``.gitreview`` first --- so consuming it
here would silently reinterpret an existing setting as a routing
declaration.  The ``DEPENDAMERGE_`` names carry no such history.

Separated from :mod:`host_config`, which answers the same question for
GitHub and additionally resolves a *default* host for shorthand.  There
is no Gerrit shorthand to resolve, so there is no default to pick: a
Gerrit target always names its own server.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from .host_config import _clean_host, _strip_scheme
from .models import UrlParseError

__all__ = [
    "gerrit_host_override",
    "gerrit_hosts",
    "iter_gerrit_hosts",
    "malformed_declaration_for",
    "set_gerrit_host",
]

#: Environment variable naming a single Gerrit host.
_GERRIT_HOST_ENV = "DEPENDAMERGE_GERRIT_HOST"

#: Environment variable naming several, comma-separated.
_GERRIT_HOSTS_ENV = "DEPENDAMERGE_GERRIT_HOSTS"

#: Host named by ``--gerrit-host`` on the command line, if any.
#:
#: Process-wide for the reason ``_HOST_OVERRIDE`` is: the flag is one
#: more source of the same setting the environment provides, and the
#: parsers that consult it are pure functions reached long
#: before any per-run context object exists.
_GERRIT_HOST_OVERRIDE: str | None = None


def set_gerrit_host(host: str | None) -> None:
    """Record the host named by ``--gerrit-host``.

    Adds to the environment declarations rather than replacing them.
    ``--github-host`` outranks the environment because it also picks
    the *default* host for shorthand; Gerrit has no default to pick, so
    every declared host is simply declared.

    Anything that is not a string is treated as absent, which tolerates
    direct Python calls to the commands where Typer's ``OptionInfo``
    default arrives unresolved.

    Args:
        host: The hostname, or None to clear a previous value.
    """
    global _GERRIT_HOST_OVERRIDE
    if not isinstance(host, str):
        _GERRIT_HOST_OVERRIDE = None
        return
    cleaned = _clean_host(host, label="Gerrit")
    _GERRIT_HOST_OVERRIDE = cleaned or None


def gerrit_host_override() -> str | None:
    """Return the host named by ``--gerrit-host``, if any."""
    return _GERRIT_HOST_OVERRIDE


def iter_gerrit_hosts() -> Iterator[str]:
    """Yield declared Gerrit hosts lazily: the flag, then the environment.

    Lazily, and deferring a malformed value, for the reasons
    :func:`~dependamerge.url_parser.host_config.iter_enterprise_hosts`
    records: a caller asking whether *some* host matches should not be
    refused by a stale later setting that the target never needed.

    Yields:
        Lowercased hostnames, without duplicates.

    Raises:
        UrlParseError: A value reached during iteration is unusable.
    """
    seen: set[str] = set()

    def _fresh(host: str) -> bool:
        if not host or host in seen:
            return False
        seen.add(host)
        return True

    if _GERRIT_HOST_OVERRIDE and _fresh(_GERRIT_HOST_OVERRIDE):
        yield _GERRIT_HOST_OVERRIDE

    deferred: UrlParseError | None = None

    def _clean(value: str) -> str:
        nonlocal deferred
        try:
            return _clean_host(value, label="Gerrit")
        except UrlParseError as exc:
            if deferred is None:
                deferred = exc
            return ""

    host = _clean(os.environ.get(_GERRIT_HOST_ENV) or "")
    if _fresh(host):
        yield host
    raw = os.environ.get(_GERRIT_HOSTS_ENV) or ""
    for candidate in raw.split(","):
        host = _clean(candidate)
        if _fresh(host):
            yield host

    if deferred is not None:
        raise deferred


def gerrit_hosts() -> tuple[str, ...]:
    """Return the Gerrit hosts the operator has declared.

    Declared by ``--gerrit-host`` on the command line, by
    ``DEPENDAMERGE_GERRIT_HOST``, and by ``DEPENDAMERGE_GERRIT_HOSTS``
    (comma-separated).

    Reads every source, so any malformed value is reported.  A caller
    asking only whether one host is declared should use
    :func:`iter_gerrit_hosts`, which stops at the first match.

    Returns:
        A tuple of lowercased hostnames, without duplicates.
    """
    return tuple(iter_gerrit_hosts())


def malformed_declaration_for(host: str) -> UrlParseError | None:
    """Return the error a malformed Gerrit declaration *of host* raises.

    Routing ignores an unusable declaration, so a stale entry for some
    other server cannot fail an unrelated target.  One naming the
    target's own host is different: ignoring
    ``DEPENDAMERGE_GERRIT_HOST=review.example.com:8443`` routed
    ``review.example.com`` to GitHub guidance, although the operator had
    plainly meant it as Gerrit.

    Only a port is matched, as the one fault that leaves the intended
    hostname unambiguous: the value must be exactly ``host:digits``.
    Anything more is ignored rather than guessed at, because the extra
    syntax changes which host is meant ---
    ``https://review.example.com:8443@evil.example`` names
    ``evil.example``, and blaming it on ``review.example.com`` would
    refuse that host over configuration aimed elsewhere.
    """
    host = (host or "").strip().lower()
    if not host:
        return None
    raw_values = [os.environ.get(_GERRIT_HOST_ENV) or ""]
    raw_values += (os.environ.get(_GERRIT_HOSTS_ENV) or "").split(",")
    for raw in raw_values:
        value = _strip_scheme(raw.strip()).strip("/").lower()
        name, _, port = value.rpartition(":")
        if name != host or not port.isdigit():
            continue
        try:
            _clean_host(raw, label="Gerrit")
        except UrlParseError as exc:
            return exc
    return None
