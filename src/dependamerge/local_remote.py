# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
What a git remote says about the server it names.

Split from :mod:`local_repo`, which answers the wider question of what
the current checkout points at.  These are the pure string decisions
underneath it --- is this a remote at all, which host does it name,
does that host look like Gerrit --- and keeping them apart makes them
testable without a working tree, which is how most of them are
exercised.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit

from .url_parser import (
    UrlParseError,
    is_declared_gerrit_host,
    is_supported_github_host,
    normalize_target,
    redact_target,
)

log = logging.getLogger("dependamerge.local_remote")

#: Gerrit's SSH daemon port, which is definitive when it appears in the
#: authority of a remote.
_GERRIT_SSH_PORT = "29418"

#: A remote carrying an explicit scheme.
_SCHEME_RE = re.compile(r"\A[A-Za-z][A-Za-z0-9+.-]*://")

#: An scp-style remote: ``[user@]host:path``.
_SCP_REMOTE_RE = re.compile(r"\A(?:(?P<user>[^@/]+)@)?(?P<host>[^@/:]+):(?!//)[^\s]+\Z")


def host_suggests_gerrit(host: str) -> bool:
    """Report whether a hostname reads like a Gerrit server.

    A weak, last-resort hint used only after ``.gitreview`` and the SSH
    port have had their say.  Matching is on whole dot-separated labels
    so that ``gerrit.example.org`` and ``review.gerrit.example.org``
    qualify while ``notgerrit.example.org`` does not.

    NOT a security check.  It decides which parser to try on the
    operator's own checkout; it never authorises sending anything
    anywhere.  Host authorisation lives in
    :func:`~dependamerge.url_parser.hosts.is_supported_github_host`.

    Args:
        host: The hostname from a git remote.

    Returns:
        True when the name suggests Gerrit.
    """
    labels = (host or "").strip().lower().split(".")
    return any(label == "gerrit" or label.startswith("gerrit-") for label in labels)


def _remote_hostname(normalized: str) -> str:
    """The hostname a normalised web remote names.

    Userinfo and a port belong to the authority, not to the host, and
    comparing them along with it silently defeated every declaration:
    ``review.example.com:8443`` matches no declared host, so a remote
    naming a port fell through to the name guess.

    Reads it with :func:`urlsplit` rather than by hand.  Both callers
    pass a URL :func:`_remote_web_url` has already normalised, so the
    scp-style ambiguity that motivated a hand-rolled version --- where
    the text after a colon is a *path* --- has been resolved by then,
    and ``urlsplit`` handles the cases stripping a trailing ``:digits``
    does not: a bracketed IPv6 authority, and a malformed port, which
    left ``host:notaport`` matching no declaration at all.
    """
    try:
        return (urlsplit(normalized).hostname or "").lower()
    except ValueError:
        # An authority this unparsable names no host we can compare.
        return ""


def _looks_like_gerrit_remote(url: str) -> bool:
    """Report whether a remote URL is a Gerrit one."""
    raw = url.strip()
    # The port is definitive: Gerrit's SSH daemon owns 29418.  It has to
    # come from the *authority*, though.  An scp-style remote puts the
    # path after the colon, so a substring test would read
    # ``git@github.com:29418/widget.git`` --- an owner named 29418 ---
    # as a Gerrit server.
    #
    # Parsed rather than split by hand, and under guard: a textual
    # ``rpartition`` read ``ssh://[bad]:29418/project`` as definitive
    # Gerrit evidence, and extracting its identity then raised.  A port
    # that cannot be parsed is no evidence, so classification falls
    # through to the hostname --- still readable in ``host:notaport``,
    # and read under the same guard below.
    if _SCHEME_RE.match(raw):
        try:
            if urlsplit(raw).port == int(_GERRIT_SSH_PORT):
                return True
        except ValueError as exc:
            log.debug("remote %s has no usable port: %s", _safe_for_log(raw), exc)

    normalized = _remote_web_url(raw)
    if normalized is None:
        return False
    host = _remote_hostname(normalized)
    # A Gerrit declaration settles it, the way the SSH port does.  The
    # URL parsers honour one, so a checkout of the same server must
    # too: a plain HTTPS remote on a declared Gerrit host carries none
    # of the other evidence --- no 29418, and an arbitrary hostname ---
    # so without this an omitted target was classified as GitHub and
    # then failed asking the operator to declare a host they had
    # already declared, for the other platform.
    if is_declared_gerrit_host(host):
        return True
    # An explicit declaration outranks a guess about the name.
    # Enterprise hostnames are arbitrary, so an operator may well have
    # declared one carrying a ``gerrit`` label, and treating it as
    # Gerrit anyway would make that declaration unusable.  The stronger
    # Gerrit evidence still wins: the SSH port above, the declaration
    # just checked, and ``.gitreview`` which the caller consults first.
    try:
        if is_supported_github_host(host):
            return False
    except UrlParseError as exc:
        # A malformed *GitHub* host setting says nothing about whether
        # this remote is Gerrit.  Raising here would abort inference
        # for a Gerrit checkout over configuration it never consults,
        # and this runs before the merge command's error guard, so it
        # would surface as a traceback.  A GitHub target still reports
        # the same setting through the parsers.
        log.debug("ignoring unusable GitHub host configuration: %s", exc)
    return host_suggests_gerrit(host)


def _names_a_server(url: str) -> bool:
    """Report whether a git remote addresses a server at all.

    A remote is a URL, an scp-style address, or a filesystem path.  It
    is never *shorthand*: that is a convenience for what a human types,
    and applying it here is actively dangerous --- a relative remote
    like ``mirror/widget.git`` would expand to a real, unrelated GitHub
    repository and an omitted-target merge would act on it.

    Args:
        url: The remote URL as git reports it.

    Returns:
        True when the remote names a host rather than a local path.
    """
    raw = url.strip()
    if _SCHEME_RE.match(raw):
        return True
    scp = _SCP_REMOTE_RE.match(raw)
    if scp is None:
        return False
    # Userinfo settles it.  ``git@ghe:acme/widget.git`` addresses a
    # server whose DNS name has a single label, which an internal
    # Enterprise installation may well have, and no filesystem path
    # carries a ``user@`` prefix.
    if scp.group("user"):
        return True
    # ``C:/repos/widget.git`` is a Windows drive, not a host, and the
    # scp pattern cannot tell them apart on its own.  Without userinfo
    # a real remote host is dotted, or is localhost --- or is one the
    # operator declared, which settles the ambiguity the way userinfo
    # does: ``--gerrit-host review`` names ``review`` as a server.  A
    # single letter stays a drive whatever is declared, so the fallback
    # cannot turn ``C:/repos/widget.git`` into a network target.
    authority = scp.group("host").lower()
    return (
        "." in authority
        or authority == "localhost"
        or (len(authority) > 1 and is_declared_gerrit_host(authority))
    )


def _safe_for_log(url: str) -> str:
    """Redact any credentials from a remote before it reaches a log.

    Delegates to :func:`~dependamerge.url_parser.redact_target` rather
    than keeping a second implementation.  This module had its own,
    which then needed the *same* correction independently: both
    anchored on ``scheme://`` and so left ``//user:pw@host`` untouched.
    Two copies of one rule means fixing it twice and discovering that
    fact the hard way.

    ``git_ops.redact_text`` is not the shared one to use here: it only
    recognises http(s), and a remote this module *declines* may carry
    any scheme.

    Args:
        url: The remote URL as git reports it.

    Returns:
        The URL with credentials removed from every position.
    """
    return redact_target(url)


def _remote_web_url(url: str) -> str | None:
    """Normalise a git remote into a URL safe to show and to parse.

    Credentials reach a remote in two ways.  ``normalize_target``
    removes URL userinfo, but a query string can carry one too ---
    a remote ending ``/owner/repo.git?token=SECRET`` --- and this URL
    is printed back to the operator when a target is inferred.  A git
    remote never needs a query or a fragment, so both are dropped.

    Args:
        url: The remote URL as git reports it.

    Returns:
        A web URL with no credentials in any position, or None when
        the remote is not one a target can be derived from.  A local
        ``file://`` mirror is a perfectly valid remote that names no
        server to merge against, so it is an ordinary "cannot infer"
        answer rather than an error.
    """
    raw = url.strip()
    if not _names_a_server(raw):
        # A filesystem path, relative or absolute.  Nothing to target,
        # and emphatically not something to run through shorthand
        # expansion.
        log.debug("remote %s is a local path, not a server", _safe_for_log(raw))
        return None
    try:
        normalized = normalize_target(raw)
    except UrlParseError as exc:
        log.debug("remote %s is not a usable target: %s", _safe_for_log(raw), exc)
        return None
    stripped = normalized.split("?", 1)[0].split("#", 1)[0]
    if not stripped.startswith(("http://", "https://")):
        log.debug("remote %s does not name a server", _safe_for_log(raw))
        return None
    return stripped


def _gerrit_identity_from_remote(url: str) -> tuple[str, str]:
    """Extract ``(host, project)`` from a Gerrit remote URL.

    Gerrit remotes name the project in their path, so a checkout with
    no ``.gitreview`` still identifies itself.

    Args:
        url: The remote URL.

    Returns:
        The host and project, either of which may be empty.
    """
    normalized = _remote_web_url(url)
    if normalized is None:
        return ("", "")
    # The same reduction the classification uses.  Reporting the raw
    # authority here left ``LocalTarget.host`` carrying a port that the
    # classification had already stripped, so the conflict check the
    # CLI runs against it missed declarations for the bare host.
    try:
        project = urlsplit(normalized).path.strip("/")
    except ValueError:
        # Unparsable, as :func:`_remote_hostname` already tolerates.
        return ("", "")
    return (_remote_hostname(normalized), project)
