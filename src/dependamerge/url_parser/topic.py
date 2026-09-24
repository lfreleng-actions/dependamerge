# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Parsing for Gerrit topic search URLs.

Supported formats (see :func:`parse_gerrit_topic_url` for the full
contract):
    https://gerrit.example.org/q/topic:some-topic
    https://gerrit.onap.org/r/q/topic:some-topic
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

from .gerrit_routing import (
    reject_conflicting_host_declaration,
    reject_gerrit_on_github_host,
)
from .hosts import reject_path_parameters, reject_port_bearing_host
from .models import ChangeSource, ParsedGerritTopicUrl, UrlParseError
from .redaction import redact_target
from .shorthand import looks_like_host, normalize_target

_SCHEME_RE = re.compile(r"\A[A-Za-z][A-Za-z0-9+.-]*://")


def _names_a_host(value: str) -> bool:
    """Report whether a target names a server of its own.

    A scheme settles it.  Without one, the first path segment does:
    ``gerrit.example.org/q/topic:x`` is a scheme-less URL, whereas
    ``q/topic:x`` is owner shorthand that would resolve against the
    GitHub default host.

    Args:
        value: The target as the operator typed it.

    Returns:
        True when the target carries its own host.
    """
    raw = value.strip()
    if _SCHEME_RE.match(raw):
        return True
    # ``//host/path`` names an authority without a scheme, and
    # ``normalize_target`` recognises it as a web URL, so refusing it
    # here made this parser disagree with every other one about the
    # same input.
    if raw.startswith("//"):
        return True
    return looks_like_host(raw.split("/", 1)[0])


# aislop-ignore-file ai-slop/hardcoded-url -- This module parses and builds
# GitHub/Gerrit URLs, so URL literals here are the subject matter, not
# stray configuration: example URLs in error/usage messages and
# docstrings, plus the canonical https://api.github.com endpoints for
# GitHub.  Enterprise hosts are always derived from the caller's input.


#: Gerrit search URLs separate terms with ``+`` (rendered as a space),
#: so ``topic:`` has to sit at a term boundary.  Shared with
#: :func:`looks_like_topic_search` rather than approximated there: three
#: review rounds found the approximation missing supported searches
#: (``status:open+topic:release``) and claiming unsupported ones
#: (``notopic:release``), which is the argument for one grammar.
#:
#: The value may be empty, but only up to a term boundary.  Requiring a
#: non-empty one meant ``/q/topic:`` matched nothing, so the "cannot be
#: empty" refusal below was unreachable and the CLI fell through to
#: unrelated GitHub guidance.  The boundary is what keeps a malformed
#: term such as ``topic:"abc`` or ``topic:a"b`` from matching a prefix.
_TOPIC_TERM_RE = re.compile(r'(?:^|[+\s])topic:(?:"([^"]*)"|([^+\s"]*))(?=[+\s]|$)')


def _locate_query(parsed: object) -> tuple[list[str], str] | None:
    """Split a parsed URL into its base segments and query expression.

    The legacy UI keeps the query in the fragment (``/#/q/...``); the
    PolyGerrit UI keeps it in the path (``/q/...``).  Returns None when
    the URL names no ``/q/`` segment at all.
    """
    path = unquote(getattr(parsed, "path", "")).rstrip("/")
    fragment = unquote(getattr(parsed, "fragment", "")).rstrip("/")
    if fragment.startswith("/q/"):
        return ([s for s in path.split("/") if s], fragment[len("/q/") :])
    segments = [s for s in path.split("/") if s]
    if "q" not in segments:
        return None
    q_index = segments.index("q")
    return (segments[:q_index], "/".join(segments[q_index + 1 :]))


def looks_like_topic_search(url: str) -> bool:
    """Whether *url* is unmistakably asking for a Gerrit topic search.

    The recognition :func:`parse_gerrit_topic_url` performs, without the
    parsing --- for a caller that needs to know which *error* to report
    and must not reach a different conclusion about the same input.

    The host requirement comes first, and before normalisation, exactly
    as the parser applies it.  Normalising ``q/topic:release`` expands
    it as *owner shorthand* against the GitHub default host, which then
    reads as a topic search on github.com --- so the selector claimed a
    target the parser refuses, and the cascade stopped on a Gerrit error
    for input that was never Gerrit.

    Best-effort: a target too malformed to parse is simply not one.
    """
    raw = url.strip()
    if not _names_a_host(raw):
        return False
    try:
        parsed = urlparse(normalize_target(raw))
    except (UrlParseError, ValueError):
        return False
    located = _locate_query(parsed)
    if located is None:
        return False
    return _TOPIC_TERM_RE.search(located[1]) is not None


def parse_gerrit_topic_url(url: str) -> ParsedGerritTopicUrl:
    """
    Parse a Gerrit topic search URL.

    Supported formats (the optional base path, e.g. "r", precedes /q/):
        https://gerrit.example.org/q/topic:some-topic
        https://gerrit.onap.org/r/q/topic:some-topic
        https://gerrit.example.org/#/q/topic:some-topic  (legacy UI)

    Additional search operators in the query (separated by '+' or
    whitespace) are tolerated; only the ``topic:`` term is extracted.
    Quoted topics (``topic:"some topic"``) and percent-encoded
    characters are handled.

    Args:
        url: The URL to parse.

    Returns:
        A ParsedGerritTopicUrl with the host, base path, and topic.

    Raises:
        UrlParseError: If the URL is not a Gerrit topic search URL.
    """
    original_url = url.strip()
    if not original_url:
        raise UrlParseError("URL cannot be empty")

    # Expand git remote forms and a missing scheme into an absolute
    # URL.  Centralised so every parser understands the same set of
    # abbreviations.
    #
    # Owner shorthand is excluded, because it is a *GitHub* convenience
    # and resolves against the GitHub default host.  Letting it through
    # manufactured a Gerrit target on the wrong server: ``q/topic:x``
    # expanded to ``https://github.com/q/topic:x``, whose path this
    # parser then accepted, so ``merge`` dispatched a Gerrit topic run
    # against github.com.  A Gerrit search has to name its own host.
    if not _names_a_host(original_url):
        raise UrlParseError(
            f"Not a Gerrit search URL (no host): {redact_target(original_url)}. "
            "A topic search must name the Gerrit server, as in "
            "gerrit.example.org/q/topic:release."
        )
    normalized = normalize_target(original_url)

    try:
        parsed = urlparse(normalized)
    except Exception as exc:
        raise UrlParseError(f"Invalid URL format: {exc}") from exc

    if not parsed.hostname:
        raise UrlParseError("URL must include a hostname")

    host = parsed.hostname.lower()

    located = _locate_query(parsed)
    if located is None:
        raise UrlParseError(
            f"Not a Gerrit search URL (no /q/ segment): {redact_target(original_url)}"
        )
    base_segments, query_expr = located

    if not query_expr:
        raise UrlParseError(
            f"Gerrit search URL contains no query expression: {redact_target(original_url)}"
        )

    match = _TOPIC_TERM_RE.search(query_expr)
    if not match:
        raise UrlParseError(
            "Only topic searches are supported for Gerrit query URLs. "
            f"Expected: https://{host}/q/topic:some-topic "
            f"(got: {redact_target(original_url)})"
        )

    topic = (match.group(1) or match.group(2) or "").strip()
    if not topic:
        raise UrlParseError("Gerrit topic cannot be empty")

    # Only now, with the target established as a topic search, are the
    # declarations consulted.  A topic search is the second Gerrit
    # shape, and it lives here rather than in ``change.py``, so the
    # rules have to be applied in both places --- but applying them at
    # the top of this function refused every non-PR URL on a declared
    # GitHub host, because the CLI tries this parser before the owner
    # and repository ones and an ordinary ``owner/repo`` URL reaches it
    # on the way past.  The refusal belongs to a URL that really is
    # asking for Gerrit, not to one merely being tested against it.
    reject_conflicting_host_declaration(host)
    reject_gerrit_on_github_host(host, original_url)
    # As on every other boundary: ``urlparse`` drops the port, so a
    # target naming one would be parsed for the bare host and then
    # addressed on the default port instead.
    reject_port_bearing_host(parsed.netloc.lower(), "Gerrit topic")
    # The same guard every GitHub parser applies, for the same reason:
    # ``urlparse`` splits a ``;suffix`` off into ``params``, so
    # ``/q/topic:release;other`` read as the topic ``release`` --- a
    # different search, returned without complaint.  This parser is
    # public, so it has to guard itself rather than rely on the merge
    # cascade trying ``parse_change_url`` first.  After the host checks
    # above, so a mistyped host is never reported as a malformed path.
    reject_path_parameters(parsed, original_url)

    base_path = "/".join(base_segments) if base_segments else None

    return ParsedGerritTopicUrl(
        source=ChangeSource.GERRIT,
        host=host,
        base_path=base_path,
        topic=topic,
        original_url=original_url,
    )
