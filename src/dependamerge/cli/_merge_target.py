# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Working out what the ``merge`` command was pointed at.

A target may be a single change, a Gerrit topic search, a repository or
an owner, and the four shapes are told apart by trying their parsers in
an order that matters (see :func:`_parse_merge_target`).  This module
answers *what was asked for*; :mod:`._merge_dispatch` acts on the
answer.

Separated so neither outgrows a reviewable size, and so the parse-order
reasoning sits next to the failure reporting that depends on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NoReturn
from urllib.parse import urlparse

import typer

from ..local_repo import LocalTarget, detect_local_target
from ..url_parser import (
    HostDeclarationError,
    ParsedGerritTopicUrl,
    ParsedOrgUrl,
    ParsedRepoUrl,
    ParsedUrl,
    UrlParseError,
    _host_matches,
    _is_gerrit_url,
    is_declared_gerrit_host,
    looks_like_topic_search,
    normalize_target,
    parse_change_url,
    parse_gerrit_topic_url,
    parse_org_url,
    parse_repo_url,
    reject_conflicting_host_declaration,
)
from ._app import console

#: A path that is *shaped* like a pull request, whether or not it is a
#: valid one.  Used to pick the right remedy: a fault in this shape is
#: the shape's, so declaring a host cannot repair it.
_PR_SHAPED_PATH_RE = re.compile(r"\A/[^/]+/[^/]+/pull(?:/.*)?\Z")


@dataclass
class _MergeTarget:
    """The URL shape ``merge`` was given; at most one field is set."""

    url: ParsedUrl | None = None
    topic: ParsedGerritTopicUrl | None = None
    repo: ParsedRepoUrl | None = None
    org: ParsedOrgUrl | None = None


def _target_host(target: _MergeTarget) -> str:
    """Return the host of whichever shape was parsed.

    Every parsed shape carries its own host, but the caller holds a
    union of four.  Reading it in one place keeps the GitHub Enterprise
    base URLs derived from the host the operator actually named, rather
    than from a default that happens to be right on github.com and
    silently wrong everywhere else.

    Args:
        target: The parsed target.

    Returns:
        The hostname, or an empty string when nothing was parsed.
    """
    for shape in (target.url, target.topic, target.repo, target.org):
        if shape is not None:
            return shape.host
    return ""


def _resolve_target_url(pr_url: str) -> str:
    """Return the URL to act on, inferring it when none was given.

    Omitting the argument means "this repository".  The checkout's
    remote supplies a repository URL, which then flows through the same
    parsing as a typed one --- so the inference adds a source of URLs
    rather than a second code path.

    Args:
        pr_url: The argument as given, possibly empty.

    Returns:
        A URL for :func:`_parse_merge_target`.

    Raises:
        typer.Exit: When nothing was given and nothing could be
            inferred, or when the checkout is a Gerrit one, whose
            changes are not addressable from the checkout alone.
    """
    if pr_url.strip():
        return pr_url

    target = detect_local_target()
    if target is None:
        console.print(
            "❌ No URL given, and this is not a git repository with a usable remote."
        )
        console.print(
            "   Pass a target, or run from a checkout. Shorthand is "
            "accepted: 'owner', 'owner/repo', 'owner/repo/pull/7'."
        )
        raise typer.Exit(1)

    if target.is_gerrit:
        # A contradiction about the host has to be reported before the
        # checkout guidance, not after: this branch exits without
        # reaching any parser, so it is the only place the omitted
        # target ever gets checked.  ``_looks_like_gerrit_remote``
        # cannot do it --- it returns a bool and runs before this
        # command's error guard, so raising there would print a
        # traceback rather than a message.
        try:
            reject_conflicting_host_declaration(target.host)
        except HostDeclarationError as exc:
            console.print(f"❌ {exc}")
            raise typer.Exit(1) from None

        # Stop here rather than letting a Gerrit checkout fall through
        # to the GitHub path, which would fail somewhere far less
        # informative.  Gerrit changes are addressed by change or topic,
        # neither of which the checkout alone determines.
        where = _describe_gerrit_checkout(target)
        console.print(f"ℹ️ Detected a Gerrit repository{where}.")
        console.print(
            "   Gerrit changes are not addressable as a repository, so "
            "a target is required here:"
        )
        console.print("     a change URL   https://HOST/c/PROJECT/+/12345")
        console.print("     a topic URL    https://HOST/q/topic:my-topic")
        console.print("     or --topic with either of the above")
        raise typer.Exit(1)

    # The URL goes on its own line, indented to sit under the text
    # rather than the marker.  It is the longest part of the message and
    # the part worth reading, so wrapping it mid-path buries it.  The
    # directory name is left out because the URL already carries the
    # repository, and the two disagreeing --- a clone in a renamed
    # directory --- reads as a contradiction.
    console.print(
        f"📍 No URL given; using the '{target.remote}' remote of "
        "current Git repository:"
    )
    console.print(f"   {target.url}")
    return target.url


def _describe_gerrit_checkout(target: LocalTarget) -> str:
    """Describe where a detected Gerrit checkout lives, if known.

    Reads the identity off the target rather than off ``.gitreview``
    alone, so a checkout recognised by its remote is named too --- the
    detail that makes the guidance actionable.
    """
    if not target.host:
        return ""
    if target.project:
        return f" (host {target.host}, project {target.project})"
    return f" (host {target.host})"


def _report_unparsable_url(
    pr_url: str,
    change_err: UrlParseError,
    org_err: UrlParseError,
    repo_err: UrlParseError,
) -> NoReturn:
    """Report the most relevant parse failure and exit.

    If the URL targets a non-github.com host the original
    ``parse_change_url`` error gives host-appropriate guidance (e.g.
    Gerrit tips), whereas ``parse_repo_url`` only talks about github.com.

    Args:
        pr_url: The URL as the operator typed it.
        change_err: Failure from ``parse_change_url``.
        org_err: Failure from ``parse_org_url``.
        repo_err: Failure from ``parse_repo_url``.

    Raises:
        typer.Exit: Always.
    """

    # Normalise the same way the parsers did, so a scheme-less target
    # yields a hostname rather than being read as a bare path.  A bad
    # host *configuration* makes this raise the same error the parsers
    # already recorded; falling back to the raw string lets that
    # recorded message be printed instead of escaping from here, where
    # nothing is catching it.
    try:
        _norm = normalize_target(pr_url)
    except UrlParseError:
        _norm = pr_url
    try:
        parsed = urlparse(_norm)
        host = parsed.hostname or ""
        path = parsed.path
    except Exception:
        host, path = "", ""

    if host and not _host_matches(host.lower(), "github.com"):
        segs = [s for s in path.split("/") if s]
        if _is_gerrit_url(host.lower(), path.rstrip("/")):
            # Structurally a Gerrit change, so the platform-agnostic
            # guidance from parse_change_url is the useful one.
            console.print(f"❌ Invalid URL: {change_err}")
        elif segs and (segs[0] == "orgs" or len(segs) == 1):
            console.print(f"❌ Invalid URL: {org_err}")
        elif _PR_SHAPED_PATH_RE.match(path):
            # PR-shaped but rejected, so the fault is the shape rather
            # than the host --- a non-numeric number, or a stray
            # ``.git``.  Declaring the host cannot repair either, and
            # on an *already* declared Enterprise host the
            # repository-mode guidance is doubly misleading.
            console.print(f"❌ Invalid URL: {change_err}")
        elif len(segs) >= 2:
            # An ordinary owner/repo shape on an undeclared host.  Its
            # rejection carries the instructions for declaring that
            # host; "cannot determine platform" does not, and reporting
            # that instead left the operator with no way forward.
            console.print(f"❌ Invalid URL: {repo_err}")
        else:
            # A host with no path at all names no target on any host,
            # so declaring it cannot help.  Keep the platform-agnostic
            # message rather than sending the operator to configure
            # something that will not change the outcome.
            console.print(f"❌ Invalid URL: {change_err}")
    else:
        # github.com.  A PR-shaped path that was rejected failed on
        # something *within* the URL --- an impossible owner or
        # repository name, a non-numeric number --- so the change
        # parser's complaint is the specific one.  The repository
        # parser's fallback for this shape advises passing the full PR
        # URL, which the operator has just done: guidance that
        # contradicts the input reads as the tool not having looked.
        if _PR_SHAPED_PATH_RE.match(path):
            console.print(f"❌ Invalid URL: {change_err}")
        else:
            console.print(f"❌ Invalid URL: {repo_err}")
    raise typer.Exit(1) from None


def _parse_merge_target(pr_url: str) -> _MergeTarget:
    """Resolve which of the accepted URL shapes ``pr_url`` is.

    Tries a specific PR/change URL first, then a Gerrit topic search URL,
    then an owner-wide URL (bare owner / orgs/owner), then a single
    repository URL.

    A refusal about the *host* stops the cascade rather than joining it.
    Every other failure says "not this shape, try the next"; a
    declaration refusal says "not this host, whatever the shape", so
    continuing asks three more parsers a question already answered --- and
    the operator was then shown whichever of them complained last.  A
    Gerrit topic URL on a host declared as GitHub reported an invalid
    *repository name* of ``topic:``, which describes neither the fault
    nor its remedy.

    Args:
        pr_url: The URL as the operator typed it.

    Returns:
        The target, with exactly one field set.

    Raises:
        typer.Exit: The URL matches none of the accepted shapes, or
            names a host whose declarations refuse it.
    """
    try:
        return _classify_merge_target(pr_url)
    except HostDeclarationError as exc:
        console.print(f"❌ {exc}")
        raise typer.Exit(1) from None


def _target_host_is_declared_gerrit(target: str) -> bool:
    """Whether *target* names a host the operator declared as Gerrit.

    Best-effort and deliberately silent: it decides which *error* to
    report, so a target too malformed to yield a host simply falls
    through to the ordinary cascade.
    """
    try:
        parsed = urlparse(normalize_target(target))
    except (UrlParseError, ValueError):
        return False
    return is_declared_gerrit_host((parsed.hostname or "").lower())


def _classify_merge_target(pr_url: str) -> _MergeTarget:
    """Try each accepted URL shape in turn.

    Raises:
        HostDeclarationError: A declaration refuses the host outright.
        typer.Exit: The URL matches none of the accepted shapes.
    """
    target = _MergeTarget()
    change_err: UrlParseError | None = None
    # ``HostDeclarationError`` is re-raised at each step rather than
    # joining ``UrlParseError``, which it subclasses.  Catching it here
    # would put an authoritative answer about the host back into the
    # cascade it is meant to stop.
    try:
        target.url = parse_change_url(pr_url)
    except HostDeclarationError:
        raise
    except UrlParseError as e:
        change_err = e
        # Not a PR/change URL — try a Gerrit topic search URL next, so
        # pasted dashboard URLs like /q/topic:some-topic work directly.
        try:
            target.topic = parse_gerrit_topic_url(pr_url)
        except HostDeclarationError:
            raise
        except UrlParseError as topic_err:
            target.topic = None
            if looks_like_topic_search(pr_url):
                # The input is unmistakably a topic search, so the topic
                # parser's complaint is the one worth reading.  Letting
                # it join the cascade printed GitHub repository guidance
                # for a Gerrit dashboard URL --- and buried real faults
                # the parser had found, an unsupported port among them.
                console.print(f"❌ Invalid URL: {topic_err}")
                raise typer.Exit(1) from None
        if target.topic is None and _target_host_is_declared_gerrit(pr_url):
            # The host is a declared Gerrit server and neither Gerrit
            # shape fitted, so the change parser's complaint about the
            # format is the only useful answer.  Letting the cascade
            # continue ended with the repository parser telling the
            # operator to declare a GitHub host --- for a host they had
            # declared as Gerrit.
            #
            # After the topic attempt, not before it: on a declared
            # host ``parse_change_url`` routes every path to the change
            # parser, so a perfectly good ``/q/topic:`` URL arrives
            # here carrying a change-format error.  Exiting first
            # refused it.
            console.print(f"❌ Invalid URL: {e}")
            raise typer.Exit(1) from None
    if target.url is None and target.topic is None and change_err is not None:
        # Not a PR URL — try owner-wide before repository.  parse_org_url
        # is strict (only a bare owner or the canonical orgs/owner forms),
        # so a two-segment owner/repo URL falls through to parse_repo_url.
        # Trying owner-wide first is required so /orgs/owner is not
        # mis-parsed by parse_repo_url as owner="orgs", repo="owner".
        try:
            target.org = parse_org_url(pr_url)
        except HostDeclarationError:
            raise
        except UrlParseError as org_err:
            # Not an owner URL — try as a repository URL
            try:
                target.repo = parse_repo_url(pr_url)
            except HostDeclarationError:
                raise
            except UrlParseError as repo_err:
                _report_unparsable_url(pr_url, change_err, org_err, repo_err)
    return target
