# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Parsing for individual change URLs: GitHub pull requests and Gerrit changes.

Supported URL formats:

GitHub:
    https://github.com/owner/repo/pull/123
    https://github.enterprise.com/owner/repo/pull/456

Gerrit:
    https://gerrit.linuxfoundation.org/infra/c/project/name/+/12345
    https://gerrit.example.org/c/project/+/67890
"""

from __future__ import annotations

import re
from urllib.parse import ParseResult, urlparse

from .gerrit_routing import (
    _declared_as_github,
    is_declared_gerrit_host,
    reject_conflicting_host_declaration,
    reject_gerrit_on_github_host,
)
from .git_suffix import has_stray_git_suffix
from .hosts import _host_matches, reject_path_parameters, reject_port_bearing_host
from .models import ChangeSource, ParsedUrl, UrlParseError
from .names import require_owner_from_path, require_repo_from_path
from .redaction import redact_target
from .shorthand import is_shorthand, normalize_target

# aislop-ignore-file ai-slop/hardcoded-url -- This module parses and builds
# GitHub/Gerrit URLs, so URL literals here are the subject matter, not
# stray configuration: example URLs in error/usage messages and
# docstrings, plus the canonical https://api.github.com endpoints for
# GitHub.  Enterprise hosts are always derived from the caller's input.


def parse_change_url(url: str) -> ParsedUrl:
    """
    Parse a GitHub PR URL or Gerrit change URL.

    Args:
        url: The URL to parse.

    Returns:
        A ParsedUrl instance with the extracted components.

    Raises:
        UrlParseError: If the URL format is not recognized or invalid.
    """
    url = url.strip()
    if not url:
        raise UrlParseError("URL cannot be empty")
    from_url = not is_shorthand(url)

    # Expand shorthand ("owner", "owner/repo"), git remote forms, and a
    # missing scheme into an absolute URL.  Centralised so every parser
    # understands the same set of abbreviations.
    url = normalize_target(url)

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise UrlParseError(f"Invalid URL format: {exc}") from exc

    if not parsed.hostname:
        raise UrlParseError("URL must include a hostname")

    host = parsed.hostname.lower()
    path = parsed.path.rstrip("/")

    # First, as in :func:`detect_source` and ``GitHubClient.parse_pr_url``:
    # a contradiction about the host is authoritative, and the shape
    # check below would otherwise answer for it.
    reject_conflicting_host_declaration(host)

    if has_stray_git_suffix(path):
        # A change is never a clone URL, so normalisation preserved the
        # suffix to mark this malformed.  Refusing it here is what makes
        # that stick: both shapes below accept trailing segments, so
        # ``/pull/7/files.git`` matched pull request 7 regardless.
        raise UrlParseError(
            f"Not a change URL: {redact_target(url)}. The trailing '.git' belongs to a "
            "clone URL, not to a pull request or change."
        )

    # Declarations are consulted before the path shape.  Routing used
    # to read the path alone, so a host the operator had named as
    # GitHub Enterprise was still handed to the Gerrit client whenever a
    # URL carried a Gerrit-shaped path --- and a host they had named as
    # Gerrit could not be reached at all unless its path happened to
    # match.  A declaration is a statement about the host; a path shape
    # is a guess about the URL.
    if is_declared_gerrit_host(host):
        # A port is refused here as it is on every GitHub boundary.
        # ``urlparse`` reports ``hostname`` without it, so a target
        # naming ``:8443`` would be parsed for the bare host and the
        # Gerrit client would then address the default port --- a
        # different server than the operator named.
        _refuse_malformed_gerrit_target(parsed, url)
        return _parse_gerrit_url(host, path, url)

    # The definitive Gerrit shape is tested before the GitHub
    # heuristic.  A Gerrit project may contain slashes, so
    # ``/c/team/pull/123/+/456`` is a valid change whose project is
    # ``team/pull/123`` --- and the ``/pull/`` heuristic claimed it
    # first, parsing it as the GitHub pull request ``c/team#123``.  That
    # is a wrong target rather than a failed parse, and it slipped past
    # the refusal below as well.
    #
    # Not on a host that *is* GitHub, though.  The two grammars overlap
    # --- ``/c/team/pull/123/+/456`` satisfies both, the PR regex
    # tolerating trailing segments --- and on github.com the Gerrit
    # reading is meaningless.  Preferring it there refused a URL that
    # ``GitHubClient.parse_pr_url`` still resolves, leaving two public
    # entry points disagreeing about one address.
    if _is_definitive_gerrit_change(path) and not _github_pull_request_wins(host, path):
        reject_gerrit_on_github_host(host, url)
        _refuse_malformed_gerrit_target(parsed, url)
        return _parse_gerrit_url(host, path, url)

    # Detect platform based on URL characteristics
    if _is_github_url(host, path):
        reject_path_parameters(parsed, url)
        return _parse_github_url(host, path, url, from_url=from_url)
    elif _is_gerrit_url(host, path):
        reject_gerrit_on_github_host(host, url)
        _refuse_malformed_gerrit_target(parsed, url)
        return _parse_gerrit_url(host, path, url)
    else:
        raise UrlParseError(
            f"Cannot determine platform for URL: {redact_target(url)}. "
            "Expected GitHub PR URL (containing /pull/) or "
            "Gerrit change URL (containing /c/.../+/)."
        )


#: Gerrit's change route, anchored.  The shape :func:`_parse_gerrit_url`
#: accepts, kept identical so the test that outranks the ``/pull/``
#: heuristic cannot admit a path that parser would then reject.
#: Substring tests for ``/c/`` and ``/+/`` were not enough: they also
#: matched a *GitHub* pull request URL carrying those markers in its
#: trailing segments, which the PR regex tolerates --- so
#: ``/owner/repo/pull/7/c/foo/+/1`` stopped resolving pull request 7.
#: GitHub's pull request route, anchored.  Shared so the overlap
#: rule and the parser agree about what a pull request path is.
_PR_PATH_RE = re.compile(r"^/([^/]+)/([^/]+)/pull/(\d+)(?:/.*)?$")

_GERRIT_CHANGE_PATH_RE = re.compile(r"^(?:/([^/]+))?/c/(.+)/\+/(\d+)(?:/.*)?$")


def _github_pull_request_wins(host: str, path: str) -> bool:
    """Whether a GitHub reading beats the Gerrit change shape here.

    The two grammars overlap: the pull request regex tolerates trailing
    segments, so ``/c/team/pull/123/+/456`` satisfies both.  On a host
    that is GitHub the Gerrit reading of such a path is meaningless, and
    preferring it refused a URL ``GitHubClient.parse_pr_url`` resolves.

    The path has to actually be a pull request, though.  Deferring on
    the host alone sent ``github.com/c/project/+/123`` --- Gerrit-shaped
    and nothing else --- to the pull request parser, which answered with
    a generic format error instead of the refusal that names the real
    problem: github.com cannot be a Gerrit server.
    """
    return _PR_PATH_RE.match(path) is not None and _declared_as_github(host)


def _is_definitive_gerrit_change(path: str) -> bool:
    """Whether *path* carries Gerrit's unambiguous change shape.

    ``/c/<project>/+/<number>`` cannot occur in a GitHub pull request
    URL, which makes it the one Gerrit signal strong enough to outrank
    the ``/pull/`` heuristic --- and it has to, because a Gerrit project
    may contain ``pull/<digits>`` as path components of its own.

    Anchored rather than tested by substring, so the claim holds: a
    GitHub pull request URL may carry trailing segments, and those can
    spell ``/c/`` and ``/+/`` without the path ever being a change.
    """
    return _GERRIT_CHANGE_PATH_RE.match(path) is not None


def _is_github_url(host: str, path: str) -> bool:
    """Check if the URL is a GitHub URL using secure host comparison.

    SECURITY: Uses exact hostname matching via _host_matches(), not
    substring checks, to prevent bypass attacks via crafted hostnames.
    See CodeQL rule py/incomplete-url-substring-sanitization.

    Detection heuristics:
    - Host matches 'github.com' (exact or subdomain)
    - Path contains '/pull/' (for GitHub Enterprise with unknown hosts)
    """
    # SECURITY: Use _host_matches() — never use `"github.com" in host`
    if _host_matches(host, "github.com"):
        return True

    # Path-based detection for GitHub Enterprise with unknown hosts
    if "/pull/" in path:
        return True

    return False


def _is_gerrit_url(host: str, path: str) -> bool:
    """Check if the URL is a Gerrit URL using structural validation.

    SECURITY: Uses Gerrit's distinctive URL path structure rather than
    hostname substring matching. See CodeQL rule
    py/incomplete-url-substring-sanitization.

    Detection heuristics:
    - Path contains '/c/' and '/+/' (Gerrit change URL pattern)
    - Path starts with '/changes/' (Gerrit REST API pattern)
    """
    # Primary: Gerrit change URL structure is definitive
    if "/c/" in path and "/+/" in path:
        return True

    # Secondary: Gerrit REST API pattern
    if path.startswith("/changes/"):
        return True

    return False


def _parse_github_url(
    host: str, path: str, original_url: str, *, from_url: bool
) -> ParsedUrl:
    """
    Parse a GitHub pull request URL.

    Expected format: https://github.com/owner/repo/pull/123
    """
    # Pattern: /owner/repo/pull/number
    match = _PR_PATH_RE.match(path)
    if not match:
        raise UrlParseError(
            f"Invalid GitHub PR URL format. Expected: "
            f"https://{host}/owner/repo/pull/123"
        )

    owner = require_owner_from_path(match.group(1), host)
    repo = require_repo_from_path(match.group(2), from_url=from_url)
    pr_number = int(match.group(3))

    return ParsedUrl(
        source=ChangeSource.GITHUB,
        host=host,
        base_path=None,
        project=f"{owner}/{repo}",
        change_number=pr_number,
        original_url=original_url,
    )


def _parse_gerrit_url(host: str, path: str, original_url: str) -> ParsedUrl:
    """
    Parse a Gerrit change URL.

    Expected formats:
        https://gerrit.example.org/c/project/+/12345
        https://gerrit.example.org/infra/c/project/name/+/12345

    The base_path (e.g., "infra") is optional and appears before /c/.
    """
    # Pattern: optional_base_path/c/project_path/+/number
    # The project path can contain multiple segments (e.g., releng/tool)
    match = _GERRIT_CHANGE_PATH_RE.match(path)

    if not match:
        # Try alternative pattern without base path
        match = re.match(r"^/c/(.+)/\+/(\d+)(?:/.*)?$", path)
        if match:
            base_path = None
            project = match.group(1)
            change_number = int(match.group(2))
        else:
            raise UrlParseError(
                f"Invalid Gerrit change URL format. Expected: "
                f"https://{host}/c/project/+/12345 or "
                f"https://{host}/base/c/project/+/12345"
            )
    else:
        base_path = match.group(1)  # May be None
        project = match.group(2)
        change_number = int(match.group(3))

    if not project:
        raise UrlParseError("Gerrit URL must include a project name")

    if change_number <= 0:
        raise UrlParseError("Gerrit change number must be positive")

    return ParsedUrl(
        source=ChangeSource.GERRIT,
        host=host,
        base_path=base_path,
        project=project,
        change_number=change_number,
        original_url=original_url,
    )


def _refuse_malformed_gerrit_target(parsed: ParseResult, url: str) -> None:
    """Refuse a port or a path parameter on a target answered as Gerrit.

    Travels with every Gerrit answer, in the parser and the detector
    alike, and only *after* the declaration checks, so an authoritative
    host conflict is never masked by a malformed path.  ``urlparse``
    reports ``hostname`` without a port, so a target naming one is
    answered for the bare host and the client then addresses the
    default port --- a different server than the operator named.  It
    also strips a final ``;suffix`` into ``params``, so ``/+/1;other``
    read as change 1.  Leaving either out of :func:`detect_source` had
    the detector reporting GERRIT for a target the parser refuses.
    """
    reject_port_bearing_host(parsed.netloc.lower(), "Gerrit change")
    reject_path_parameters(parsed, url)


def detect_source(url: str) -> ChangeSource:
    """
    Detect the source platform from a URL without full parsing.

    This is a convenience function for quick platform detection.

    Args:
        url: The URL to analyze.

    Returns:
        The detected ChangeSource.

    Raises:
        UrlParseError: If the platform cannot be determined, or the
            host's declarations refuse it.
    """
    url = url.strip()
    if not url:
        raise UrlParseError("URL cannot be empty")

    # Expand shorthand ("owner", "owner/repo"), git remote forms, and a
    # missing scheme into an absolute URL.  Centralised so every parser
    # understands the same set of abbreviations.
    url = normalize_target(url)

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise UrlParseError(f"Invalid URL format: {exc}") from exc

    host = parsed.hostname.lower() if parsed.hostname else ""
    path = parsed.path.rstrip("/")

    # The same order :func:`parse_change_url` applies, so the two cannot
    # disagree about one URL.  They did: this reported GITHUB for a
    # pull-request-shaped URL on a declared Gerrit host that parsing
    # routes to Gerrit, and GERRIT for a Gerrit-shaped URL on a declared
    # GitHub host that parsing refuses outright.
    reject_conflicting_host_declaration(host)
    if is_declared_gerrit_host(host):
        _refuse_malformed_gerrit_target(parsed, url)
        return ChangeSource.GERRIT

    # The definitive Gerrit shape outranks the ``/pull/`` heuristic
    # here for the reason it does in :func:`parse_change_url`, and is
    # skipped on a GitHub host for the same reason too, so the two
    # cannot disagree about one address.
    if _is_definitive_gerrit_change(path) and not _github_pull_request_wins(host, path):
        reject_gerrit_on_github_host(host, url)
        _refuse_malformed_gerrit_target(parsed, url)
        return ChangeSource.GERRIT

    if _is_github_url(host, path):
        reject_path_parameters(parsed, url)
        return ChangeSource.GITHUB
    elif _is_gerrit_url(host, path):
        reject_gerrit_on_github_host(host, url)
        _refuse_malformed_gerrit_target(parsed, url)
        return ChangeSource.GERRIT
    else:
        raise UrlParseError(f"Cannot determine platform for URL: {redact_target(url)}")
