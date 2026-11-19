# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Fuzz the ``url_parser`` package: targets and URLs an operator types.

Run from the repository root::

    uv run --with atheris python tests/fuzz/fuzz_url_parser.py -max_total_time=60

Contracts checked, in the package's own words:

* ``UrlParseError`` is the documented failure of ``parse_change_url``,
  ``detect_source``, ``normalize_target``, ``require_owner`` and
  ``require_repo``; any other exception escapes and fails the run.
* ``detect_source`` agrees with ``parse_change_url`` on a URL the
  latter accepts.  A Gerrit change it returns has a positive number:
  "Gerrit change number must be positive".  (The GitHub parser states
  no such rule, so ``pull/0`` is not checked.)
* ``require_repo``: "Return *name*" when it is 1-100 characters of
  ``[A-Za-z0-9._-]``, not ``.`` or ``..``, and not ending ``.wiki``.
* ``require_owner`` on github.com: "Logins are alphanumerics and
  hyphens, at most 39 characters, and do not begin with a hyphen."  On
  an Enterprise host it refuses "over-length input and anything that
  would not survive being placed in a URL path".
* ``strip_git_suffix``: "A path that does not end in ``.git`` is
  returned untouched"; otherwise that suffix and any trailing slashes
  go, unless the final segment is ``.git`` alone, which has no name in
  front of it and is kept.
  ``has_stray_git_suffix``: "True when the last segment carries a
  leftover ``.git``", in any case.
* ``redact_target`` removes "URL userinfo, path parameters, the query
  string and the fragment", and a ``;`` or ``%`` that remains is
  "marked with an ellipsis".  A credential planted in the userinfo,
  a ``;jsessionid=``, the query or the fragment never reaches its
  output, nor the output of ``normalize_target`` (for userinfo) or an
  error message from the change parsers.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(_ROOT / "src"), str(_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from dependamerge.url_parser import (  # noqa: E402
    ChangeSource,
    ParsedUrl,
    UrlParseError,
    detect_source,
    has_stray_git_suffix,
    normalize_target,
    parse_change_url,
    redact_target,
    require_owner,
    require_repo,
    strip_git_suffix,
)
from tests.fuzz._harness import (  # noqa: E402
    fields,
    host_environment,
    run,
    violation,
)

#: Host declarations an input can run under, picked by its first byte.
ENVIRONMENTS: tuple[dict[str, str], ...] = (
    {},
    {"DEPENDAMERGE_GITHUB_HOST": "ghe.example.com"},
    {"DEPENDAMERGE_GERRIT_HOST": "gerrit.example.org"},
    {
        "DEPENDAMERGE_GITHUB_HOST": "both.example.net",
        "DEPENDAMERGE_GERRIT_HOST": "both.example.net",
    },
)

#: Marks a planted credential.  No other part of a planted URL holds it.
SECRET = "~"

SCHEMES = (
    "https://",
    "http://",
    "ssh://",
    "git://",
    "git+ssh://",
    "//",
    "HTTPS://",
    "ftp://",
)

_ENTERPRISE_UNSAFE = frozenset("/?#%;\\")
_REPO_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
)
_OWNER_CHARACTERS = _REPO_CHARACTERS - frozenset("._")


def _could_be_repo(name: str) -> bool:
    return (
        0 < len(name) <= 100
        and set(name) <= _REPO_CHARACTERS
        and name not in (".", "..")
        and not name.lower().endswith(".wiki")
    )


def _could_be_dotcom_owner(name: str) -> bool:
    return (
        0 < len(name) <= 39
        and set(name) <= _OWNER_CHARACTERS
        and not name.startswith("-")
    )


def _could_be_enterprise_owner(name: str) -> bool:
    return 0 < len(name) <= 39 and not any(
        c in _ENTERPRISE_UNSAFE or c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F
        for c in name
    )


def _check_redacted(value: str, shown: str) -> None:
    if "?" in shown or "#" in shown:
        raise violation("redact_target kept a query or fragment", value, shown)
    for index, character in enumerate(shown):
        if character in ";%" and shown[index + 1 :] != "\u2026":
            raise violation("redact_target kept an unmarked ';' or '%'", value, shown)


def _check_free(
    text: str,
    parse: Callable[[str], ParsedUrl],
    redact: Callable[[str], str],
) -> None:
    """Contracts that hold for any text at all."""
    try:
        parsed = parse(text)
    except UrlParseError:
        parsed = None
    try:
        source: ChangeSource | None = detect_source(text)
    except UrlParseError:
        source = None
    if parsed is not None:
        if source is not parsed.source:
            raise violation(
                "detect_source disagrees with parse_change_url", text, source, parsed
            )
        if parsed.source is ChangeSource.GERRIT and parsed.change_number <= 0:
            raise violation("Gerrit change has no positive number", text, parsed)
    try:
        normalize_target(text)
    except UrlParseError:
        pass

    for check_name, could_be, accept in (
        ("require_repo", _could_be_repo, require_repo),
        (
            "require_owner(github.com)",
            _could_be_dotcom_owner,
            lambda n: require_owner(n, "github.com"),
        ),
        (
            "require_owner(ghe)",
            _could_be_enterprise_owner,
            lambda n: require_owner(n, "ghe.example.com"),
        ),
    ):
        try:
            accepted = accept(text)
        except UrlParseError:
            continue
        if accepted != text or not could_be(text):
            raise violation(
                f"{check_name} accepted a name its grammar refuses", text, accepted
            )

    trimmed = text.rstrip("/")
    last = trimmed.split("/")[-1]
    # A final segment of ".git" alone has no name in front to keep.
    named_suffix = last.endswith(".git") and last != ".git"
    want = trimmed[: -len(".git")] if named_suffix else text
    if strip_git_suffix(text) != want:
        raise violation("strip_git_suffix misread the suffix", text, want)
    segments = [segment for segment in text.split("/") if segment]
    expected = bool(segments) and segments[-1].lower().endswith(".git")
    if has_stray_git_suffix(text) != expected:
        raise violation("has_stray_git_suffix misread the last segment", text)

    _check_redacted(text, redact(text))


def _clean(value: str, refused: str) -> str:
    return "".join(
        c for c in value if c not in refused and c != SECRET and not c.isspace()
    )


def _check_planted(
    scheme_byte: int,
    user: str,
    password: str,
    host: str,
    path: str,
    parse: Callable[[str], ParsedUrl],
    redact: Callable[[str], str],
) -> None:
    """A credential planted where URLs carry one never comes back out."""
    scheme = SCHEMES[scheme_byte % len(SCHEMES)]
    user = _clean(user, "/@:")
    # Marked between every character, so a partial leak shows too.
    password = SECRET + "".join(c + SECRET for c in _clean(password, "/"))
    host = _clean(host, "/@?#;%:") or "github.com"
    # Without a path, nothing but a "?" or "#" ends the authority.
    path = "" if scheme_byte & 0x08 else "/" + _clean(path, "?#;%").lstrip("/")
    userinfo_url = f"{scheme}{user}:{password}@{host}{path}"

    target = userinfo_url
    if scheme_byte & 0x10:
        target += f";jsessionid={SECRET}a"
    if scheme_byte & 0x20:
        target += f"?token={SECRET}b@{SECRET}d"
    if scheme_byte & 0x40:
        target += f"#{SECRET}c@{SECRET}e"
    shown = redact(target)
    if SECRET in shown:
        raise violation("redact_target showed a planted credential", target, shown)
    _check_redacted(target, shown)

    try:
        normalised = normalize_target(userinfo_url)
    except UrlParseError as exc:
        normalised = str(exc)
    if SECRET in normalised:
        raise violation(
            "normalize_target kept planted userinfo", userinfo_url, normalised
        )

    for name, parser in (("parse_change_url", parse), ("detect_source", detect_source)):
        try:
            parser(target)
        except UrlParseError as exc:
            if SECRET in str(exc):
                raise violation(
                    f"{name} echoed a planted credential", target, str(exc)
                ) from None


def check(
    data: bytes,
    *,
    parse: Callable[[str], ParsedUrl] = parse_change_url,
    redact: Callable[[str], str] = redact_target,
) -> None:
    """Run every URL contract on one input.

    The first byte picks the host declarations and, by its low bit,
    whether the rest is free text or the parts of a planted URL.
    """
    if not data:
        return
    mode, rest = data[0], data[1:]
    with host_environment(ENVIRONMENTS[(mode >> 1) % len(ENVIRONMENTS)]):
        if mode & 1:
            scheme_byte = rest[0] if rest else 0
            user, password, host, path = fields(rest[1:], 4)
            _check_planted(scheme_byte, user, password, host, path, parse, redact)
        else:
            _check_free(fields(rest, 1)[0], parse, redact)


def _seed(mode: int, *parts: str, scheme: int | None = None) -> bytes:
    head = bytes([mode]) + (bytes([scheme]) if scheme is not None else b"")
    return head + "\x00".join(parts).encode("utf-8")


SEEDS: tuple[bytes, ...] = (
    _seed(0, "https://github.com/lfreleng-actions/dependamerge/pull/42"),
    _seed(0, "https://gerrit.linuxfoundation.org/infra/c/releng/builder/+/12345"),
    _seed(4, "https://gerrit.example.org/c/project/+/67890"),
    _seed(2, "https://ghe.example.com/team_name/widget/pull/7"),
    _seed(0, "acme/widget/pull/7"),
    _seed(0, "git@github.com:acme/widget.git"),
    _seed(0, "https://github.com/acme/widget.git/"),
    _seed(0, "https://github.com/acme/.git"),
    _seed(0, "https://github.com/acme/widget/pull/7/files.GIT"),
    _seed(0, "https://gerrit.example.org/q/topic:GH-widget-7"),
    _seed(6, "https://both.example.net/acme/widget/pull/1"),
    _seed(0, "https://github.com/team%3Bjsessionid=SECRET/widget"),
    _seed(0, "lfreleng-actions"),
    _seed(0, "neovim.wiki"),
    # The fix in this change: the password holds an "@".
    _seed(0, "https://user:p@ss@host/x"),
    _seed(1, "user", "p@ss", "github.com", "acme/widget/pull/7", scheme=0x70),
    _seed(1, "", "tok", "gerrit.example.org", "c/p/+/1", scheme=0x35),
    # The fix in this change: an "@" in the query or fragment of a
    # URL with no path, and a password holding a "?".
    _seed(1, "user", "pw", "example.com", "", scheme=0x28),
    _seed(1, "user", "pw", "example.com", "", scheme=0x48),
    _seed(1, "user", "p?w", "example.com", "a", scheme=0x00),
    _seed(1, "git", "pw", "github.com", "acme/widget.git", scheme=0x02),
    # The fix in this change: change numbers past Python's digit limit.
    # libFuzzer sizes inputs from the largest seed.
    _seed(0, "https://github.com/acme/widget/pull/" + "9" * 5000),
    _seed(4, "https://gerrit.example.org/c/widget/+/" + "9" * 5000),
    _seed(0, "https://github.com/" + "a/" * 600 + "pull/1"),
)


def main() -> None:
    """Fuzz :func:`check` from :data:`SEEDS`."""
    import atheris

    run(atheris, check, SEEDS)


if __name__ == "__main__":
    main()
