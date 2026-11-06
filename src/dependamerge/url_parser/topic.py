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
from urllib.parse import unquote, unquote_plus, urlparse

from .gerrit_routing import (
    reject_conflicting_host_declaration,
    reject_gerrit_on_github_host,
)
from .hosts import reject_path_parameters, reject_port_bearing_host
from .models import ChangeSource, ParsedGerritTopicUrl, UrlParseError
from .redaction import redact_target
from .shorthand import names_a_host, normalize_target

# aislop-ignore-file ai-slop/hardcoded-url -- This module parses and builds
# GitHub/Gerrit URLs, so URL literals here are the subject matter, not
# stray configuration: example URLs in error/usage messages and
# docstrings, plus the canonical https://api.github.com endpoints for
# GitHub.  Enterprise hosts are always derived from the caller's input.


#: Gerrit search URLs separate terms with ``+`` (rendered as a space),
#: so ``topic:`` has to sit at a term boundary.  The expression it runs
#: over is already decoded (:func:`_locate_query`), so every separator
#: is whitespace there and a ``+`` left is a literal one (``%2B``).
#: Shared with :func:`looks_like_topic_search` rather than approximated
#: there: three review rounds found the approximation missing supported
#: searches (``status:open+topic:release``) and claiming unsupported
#: ones (``notopic:release``), which is the argument for one grammar.
#:
#: The value may be empty, but only up to a term boundary.  Requiring a
#: non-empty one meant ``/q/topic:`` matched nothing, so the "cannot be
#: empty" refusal below was unreachable and the CLI fell through to
#: unrelated GitHub guidance.  The boundary is what keeps a malformed
#: term such as ``topic:"abc`` or ``topic:a"b`` from matching a prefix.
#:
#: It runs over :func:`_mask_braced` output, so a ``topic:`` inside a
#: brace-quoted value (``message:{see topic:x}``) is text, not a term.
_TOPIC_TERM_RE = re.compile(r'(?:^|\s)topic:(?:"([^"]*)"|([^\s"]*))(?=\s|$)')


def _mask_braced(expr: str) -> str:
    """Empty every brace-quoted value in *expr*: ``{a b}`` becomes ``{}``.

    Gerrit quotes a value in braces as well as in double quotes, and
    neither nests.  Braces inside double quotes are text, as are double
    quotes inside braces.  An unterminated brace is left for
    :func:`_tokenise_query` to refuse.
    """
    out: list[str] = []
    in_quote = escaped = False
    index = 0
    while index < len(expr):
        ch = expr[index]
        if in_quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_quote = False
        elif ch == '"':
            in_quote = True
        elif ch == "{":
            end = expr.find("}", index + 1)
            if end != -1:
                out.append("{}")
                index = end + 1
                continue
        out.append(ch)
        index += 1
    return "".join(out)


def _tokenise_query(expr: str) -> list[str]:
    """Split a query expression into terms and parentheses.

    Terms are separated by whitespace outside quotes (the URL's ``+``
    separators are spaces by now, and a ``+`` left is literal);
    parentheses outside quotes are tokens of their own, as in Gerrit's
    own grammar.  A value quoted in double quotes or in braces
    (``file:{^src/(a|b)/}``) is opaque: its spaces and parentheses stay
    inside the term.  A backslash escapes the next character inside
    double quotes; braces do not nest and take no escapes.

    Raises:
        UrlParseError: On an unterminated quote or brace, a stray or
            nested brace, or unbalanced parentheses, any of which would
            let the expression escape the ``(...) status:open`` grouping
            it is sent in.
    """
    tokens: list[str] = []
    current: list[str] = []
    in_quote = False
    in_brace = False
    escaped = False
    depth = 0

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for ch in expr:
        if in_quote:
            current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_quote = False
        elif in_brace:
            if ch == "{":
                raise UrlParseError("Gerrit query has a '{' inside braces")
            current.append(ch)
            in_brace = ch != "}"
        elif ch == '"':
            in_quote = True
            current.append(ch)
        elif ch == "{":
            in_brace = True
            current.append(ch)
        elif ch == "}":
            raise UrlParseError("Gerrit query has an unmatched '}'")
        elif ch.isspace():
            flush()
        elif ch in "()":
            flush()
            depth += 1 if ch == "(" else -1
            if depth < 0:
                raise UrlParseError("Gerrit query has an unmatched ')'")
            tokens.append(ch)
        else:
            current.append(ch)
    if in_quote:
        raise UrlParseError("Gerrit query has an unterminated quote")
    if in_brace:
        raise UrlParseError("Gerrit query has an unterminated '{'")
    if depth:
        raise UrlParseError("Gerrit query has an unmatched '('")
    flush()
    return tokens


def _topic_value(token: str) -> str:
    """The topic a ``topic:`` token names, without its quotes or braces."""
    value = token[len("topic:") :]
    if len(value) >= 2 and (value[0], value[-1]) in (('"', '"'), ("{", "}")):
        value = value[1:-1]
    return value.strip()


def _normalise_query(expr: str) -> tuple[str, str]:
    """Return *expr* as one space-separated query, and the topic bounding it.

    The expression is sent as ``(<expr>) status:open``, so it must bound
    the result set by its topic on its own: a top-level ``OR`` or a
    negated topic term would let other changes in.  The topic is the
    first top-level ``topic:`` term's value; a ``topic:`` inside a quoted
    or brace-quoted value is part of that value, not a term.

    Raises:
        UrlParseError: When the expression is malformed, or its
            ``topic:`` term does not constrain the whole query.
    """
    tokens = _tokenise_query(expr)
    depth = 0
    topic = ""
    empty_topic = False
    previous = ""
    for token in tokens:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0:
            if token.upper() == "OR":
                raise UrlParseError(
                    "A top-level OR in a Gerrit query URL could match changes "
                    "outside the topic; group it in parentheses beside the "
                    "topic: term"
                )
            if token.startswith("topic:") and previous.upper() != "NOT" and not topic:
                topic = _topic_value(token)
                empty_topic = empty_topic or not topic
        previous = token if depth == 0 else ""
    if not topic:
        if empty_topic:
            raise UrlParseError("Gerrit topic cannot be empty")
        raise UrlParseError(
            "The topic: term in a Gerrit query URL must apply to the whole "
            "query (not negated, and not inside parentheses)"
        )

    parts: list[str] = []
    for token in tokens:
        if parts and parts[-1] != "(" and token != ")":
            parts.append(" ")
        parts.append(token)
    return "".join(parts), topic


def _locate_query(parsed: object) -> tuple[list[str], str] | None:
    """Split a parsed URL into its base segments and query expression.

    The legacy UI keeps the query in the fragment (``/#/q/...``); the
    PolyGerrit UI keeps it in the path (``/q/...``).  Returns None when
    the URL names no ``/q/`` segment at all.

    The expression is decoded as Gerrit encodes it: a raw ``+`` is a
    space between terms and ``%2B`` a literal plus.  Percent-decoding
    first would read ``label:Verified%2B1`` as two terms, a different
    query from the one the URL names.

    Everything after ``/q/`` is the expression, kept as written: a
    repeated, trailing or encoded ``/`` is part of a search value
    (``file:^src/``, ``message:"https://..."``), and dropping one would
    send a different query.  Only the base path is normalised.
    """
    path = getattr(parsed, "path", "")
    fragment = unquote_plus(getattr(parsed, "fragment", ""))
    if fragment.startswith("/q/"):
        return (_segments(unquote(path)), fragment[len("/q/") :])
    raw_segments = path.split("/")
    for index, segment in enumerate(raw_segments):
        if unquote(segment) == "q":
            base = _segments(unquote("/".join(raw_segments[:index])))
            return (base, unquote_plus("/".join(raw_segments[index + 1 :])))
    return None


def _segments(path: str) -> list[str]:
    """The non-empty ``/``-separated segments of a decoded *path*."""
    return [s for s in path.rstrip("/").split("/") if s]


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
    if not names_a_host(raw):
        return False
    try:
        parsed = urlparse(normalize_target(raw))
    except (UrlParseError, ValueError):
        return False
    located = _locate_query(parsed)
    if located is None:
        return False
    return _TOPIC_TERM_RE.search(_mask_braced(located[1])) is not None


def parse_gerrit_topic_url(url: str) -> ParsedGerritTopicUrl:
    """
    Parse a Gerrit topic search URL.

    Supported formats (the optional base path, e.g. "r", precedes /q/):
        https://gerrit.example.org/q/topic:some-topic
        https://gerrit.onap.org/r/q/topic:some-topic
        https://gerrit.example.org/#/q/topic:some-topic  (legacy UI)

    The query must hold a top-level ``topic:`` term, so it can never be
    unbounded.  Additional search operators (separated by '+' or
    whitespace) are kept: the whole expression is carried on
    ``query`` and narrows the batch as it does in the Gerrit UI.
    Quoted topics (``topic:"some topic"``, ``topic:{some topic}``) and
    percent-encoded characters are handled; a brace-quoted value is
    kept as written.

    Args:
        url: The URL to parse.

    Returns:
        A ParsedGerritTopicUrl with the host, base path, topic, and the
        normalised query expression.

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
    if not names_a_host(original_url):
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

    match = _TOPIC_TERM_RE.search(_mask_braced(query_expr))
    if not match:
        raise UrlParseError(
            "Only topic searches are supported for Gerrit query URLs. "
            f"Expected: https://{host}/q/topic:some-topic "
            f"(got: {redact_target(original_url)})"
        )

    if not (match.group(1) or match.group(2) or "").strip():
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

    query, topic = _normalise_query(query_expr)

    base_path = "/".join(base_segments) if base_segments else None

    return ParsedGerritTopicUrl(
        source=ChangeSource.GERRIT,
        host=host,
        base_path=base_path,
        topic=topic,
        original_url=original_url,
        query=query,
    )
