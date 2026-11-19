# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Fuzz the parsers that read content other people control.

Run from the repository root::

    uv run --with atheris python tests/fuzz/fuzz_untrusted_content.py -max_total_time=60

The first byte of an input picks the target (its low two bits) and
whether the rest is raw text or the parts of a well-formed document
(bit 2).  Contracts checked, in each module's own words:

* ``NetrcParser``: "Supports the standard netrc format with machine,
  login, password, and default tokens", with "quoted strings with
  escape sequences".  A file written from known entries, quoting where
  a value needs it, reads back as those entries, the last of a repeated
  machine winning.  Malformed text raises ``NetrcParseError`` alone.
* ``parse_gitreview``: "Keys are matched case-insensitively.  Optional
  whitespace around ``=`` is tolerated."  ``project`` is "**without**
  the ``.git`` suffix", a ``port=`` that is not digits or is "too long
  for ``int()``" falls back to the default, and the result exists only
  "if at least ``host`` is present and non-empty".  It never raises.
* ``parse_policy_blob``: "Raises: BotPolicyError: When the path holds
  anything but a policy document small enough to read whole", and
  nothing else.  Every App it returns is a canonical ``slug[bot]`` login
  the decoded document names as a whole entry, no App is both included
  and excluded, and an ``include_bots`` entry that is ``*`` alone is
  needed to include them all.
* ``detect_github2gerrit_comments``: a mapping it reports has "the
  minimum required fields", a topic, a mode and a Change-ID, checked
  here field by field; it holds "Gerrit Change-IDs (I-prefixed SHA-1)",
  each a whole token of "the full comment body the mapping was
  extracted from", which is the latest of the comments it lists.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(_ROOT / "src"), str(_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from dependamerge.bot_policy import BotPolicyError, PolicyLevel, PolicySource  # noqa: E402
from dependamerge.bot_policy_files import parse_policy_blob  # noqa: E402
from dependamerge.github2gerrit_detector import (  # noqa: E402
    GitHub2GerritDetectionResult,
    detect_github2gerrit_comments,
)
from dependamerge.gitreview import DEFAULT_GERRIT_PORT, GitReviewInfo, parse_gitreview  # noqa: E402
from dependamerge.netrc import NetrcParseError, NetrcParser  # noqa: E402
from tests.fuzz._harness import fields, run, violation  # noqa: E402

_PRINTABLE = frozenset(chr(code) for code in range(0x20, 0x7F))
_NEEDS_QUOTES = frozenset(' "#\\')
_LOGIN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,98}[a-z0-9])?\[bot\]")
_CHANGE_ID = re.compile(r"I[0-9a-f]{40}")
_POLICY_LIMIT = 16 * 1024
_BOT = "github-actions[bot]"
_START = "<!-- github2gerrit:change-id-map v1 -->"
_END = "<!-- end github2gerrit:change-id-map -->"


# --- netrc ---------------------------------------------------------------


def _netrc_value(value: str) -> str:
    if value and not set(value) & _NEEDS_QUOTES:
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _check_netrc_round_trip(
    flags: int, parts: list[str], parser: type[NetrcParser]
) -> None:
    printable = ["".join(c for c in part if c in _PRINTABLE) for part in parts]
    expected: dict[str, tuple[str, str]] = {}
    lines: list[str] = []
    separator = "\n" if flags & 0x08 else " "
    for machine, login, password in zip(
        printable[0::3], printable[1::3], printable[2::3], strict=False
    ):
        machine = machine.strip()
        if not (machine and login and password):
            continue
        expected[machine.lower()] = (login, password)
        tokens = ["machine", _netrc_value(machine), "login", _netrc_value(login)]
        tokens += ["password", _netrc_value(password)]
        lines.append(separator.join(tokens) + (" # note" if flags & 0x10 else ""))
    default = None
    if flags & 0x20 and printable and printable[-1]:
        default = (printable[-1], printable[-1])
        lines.append(
            f"default login {_netrc_value(default[0])} password {_netrc_value(default[1])}"
        )
    content = "\n".join(lines) + "\n"

    netrc = parser(content)
    if set(netrc.machines) != set(expected):
        raise violation("netrc read different machines", content, netrc.machines)
    for machine, (login, password) in expected.items():
        found = netrc.get_credentials(machine)
        if found is None or (found.login, found.password) != (login, password):
            raise violation("netrc read an entry back wrongly", content, machine, found)
    fallback = netrc.get_credentials("\x01")
    got = None if fallback is None else (fallback.login, fallback.password)
    if got != default:
        raise violation("netrc read the default entry wrongly", content, got)


def _check_netrc_raw(text: str, parser: type[NetrcParser]) -> None:
    try:
        parser(text)
    except NetrcParseError:
        pass


# --- .gitreview ----------------------------------------------------------


def _expected_port(value: str) -> int:
    value = value.strip(" \t")
    if not (value and value.isdecimal()):
        return DEFAULT_GERRIT_PORT
    try:
        return int(value)
    except ValueError:
        return DEFAULT_GERRIT_PORT


def _check_gitreview_structured(
    flags: int, parts: list[str], parse: Callable[[str], GitReviewInfo | None]
) -> None:
    host, port, project = (p.replace("\n", "").replace("\r", "") for p in parts[:3])
    gap = " \t"[: flags >> 3 & 3]

    def key(name: str) -> str:
        return name.upper() if flags & 0x20 else name

    text = (
        "[gerrit]\n"
        f"{key('host')}{gap}={gap}{host}\n"
        f"{key('port')}{gap}={gap}{port}\n"
        f"{key('project')}{gap}={gap}{project}\n"
    )
    info = parse(text)
    if not host.strip():
        if info is not None:
            raise violation(".gitreview without a host gave a result", text, info)
        return
    if info is None:
        raise violation(".gitreview with a host gave no result", text)
    want = (host.strip(), _expected_port(port), project.strip().removesuffix(".git"))
    if (info.host, info.port, info.project) != want:
        raise violation(".gitreview read wrongly", text, info, want)
    lf = info.host.lower() == "gerrit.linuxfoundation.org"
    if (info.base_path == "infra") != lf:
        raise violation(".gitreview base path wrong for host", text, info)


def _check_gitreview_raw(
    text: str, parse: Callable[[str], GitReviewInfo | None]
) -> None:
    info = parse(text)
    if info is not None and (not info.host or "\n" in info.host):
        raise violation(".gitreview result has no usable host", text, info)


# --- policy files --------------------------------------------------------


def _policy_text(flags: int, parts: list[str]) -> str:
    if not flags & 0x04:
        return parts[0]
    half = len(parts) // 2
    document: dict[str, Any] = {
        "include_bots": parts[:half],
        "exclude_bots": parts[half:],
    }
    if flags & 0x40:
        document["exlude_bots"] = []
    return json.dumps(document)


def _check_policy(
    flags: int, parts: list[str], parse: Callable[..., PolicySource | None]
) -> None:
    text = _policy_text(flags, parts)
    size_choice = flags >> 3 & 3
    size: object = (len(text.encode("utf-8")), _POLICY_LIMIT + 1, "12", len(text))[
        size_choice
    ]
    blob = {
        "byteSize": size,
        "isBinary": bool(flags & 0x80),
        "isTruncated": bool(flags & 0x20) and not flags & 0x04,
        "text": text,
    }
    try:
        source = parse(blob, label="acme/widget", level=PolicyLevel.REPOSITORY)
    except BotPolicyError:
        return
    if source is None:
        raise violation("a policy file gave no policy", blob)
    refused = (
        not isinstance(size, int)
        or size > _POLICY_LIMIT
        or blob["isBinary"]
        or blob["isTruncated"]
        or len(text.encode("utf-8")) > _POLICY_LIMIT
    )
    if refused:
        raise violation("an unreadable policy file was accepted", blob, source)
    document = json.loads(text)
    if not isinstance(document, dict) or not set(document) <= {
        "include_bots",
        "exclude_bots",
    }:
        raise violation(
            "a policy that is not a known object was accepted", text, source
        )
    # Compare against the decoded strings, not the JSON spelling, so an
    # escaped name such as "\u0064ependabot" still counts as named.
    included = _policy_strings(document.get("include_bots"))
    named = _named_slugs(included + _policy_strings(document.get("exclude_bots")))
    for login in source.include | source.exclude:
        if not _LOGIN.fullmatch(login) or login.removesuffix("[bot]") not in named:
            raise violation("a policy returned an App it does not name", text, login)
    if source.include & source.exclude:
        raise violation("a policy both includes and excludes an App", text, source)
    if source.include_all and "*" not in _entries(included):
        raise violation("a policy includes every App without '*'", text, source)


def _named_slugs(values: list[str]) -> set[str]:
    """The App slugs *values* name, as whole entries.

    Entries are split on commas and whitespace and read in the three
    forms the module documents, "``name``, ``name[bot]`` and
    ``app/name`` ..., case-insensitively".  A substring test let
    ``notdependabot`` stand for ``dependabot``.
    """
    return {
        entry.lower().removeprefix("app/").removesuffix("[bot]")
        for entry in _entries(values)
    }


def _entries(values: list[str]) -> set[str]:
    """The whole entries in *values*, split on commas and whitespace.

    Entries "may be separated by commas or whitespace", and "``*`` in
    an inclusion list includes every App" (docs/BOT_POLICY.md) only as
    an entry of its own: ``*x`` is not the wildcard.
    """
    return {entry for value in values for entry in re.split(r"[,\s]+", value)}


def _policy_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


# --- GitHub2Gerrit comments ----------------------------------------------


def _comments(flags: int, parts: list[str]) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    for index, body in enumerate(parts):
        bot = flags >> (index % 5 + 3) & 1
        if flags & 0x04 and index % 2 == 0:
            body = f"{_START}\n{body}\n{_END}"
        if index % 3 == 2:
            comments.append(
                {"body": body, "user": {"login": _BOT if bot else "someone"}}
            )
        else:
            comments.append(
                {"body": body, "author": {"login": _BOT if bot else "someone"}}
            )
    return comments


def _check_github2gerrit(
    flags: int,
    parts: list[str],
    detect: Callable[[list[dict[str, Any]]], GitHub2GerritDetectionResult],
) -> None:
    comments = _comments(flags, parts)
    result = detect(comments)
    if not result.has_mapping:
        if result.mapping is not None or result.comment_indices:
            raise violation("no mapping reported, yet one returned", comments, result)
        return
    mapping = result.mapping
    # The required fields are checked here rather than through
    # mapping.is_valid, the predicate the detector itself relies on.
    if mapping is None or not (mapping.topic and mapping.change_ids and mapping.mode):
        raise violation("an invalid mapping was reported", comments, result)
    if result.detection_source not in ("marker", "heuristic"):
        raise violation("unknown detection source", result)
    if not result.comment_indices or not all(
        0 <= i < len(comments) for i in result.comment_indices
    ):
        raise violation("mapping names no comment it came from", comments, result)
    latest = comments[result.comment_indices[-1]]["body"].strip()
    if mapping.raw_comment_body != latest:
        raise violation(
            "mapping body is not the latest listed comment", comments, result
        )
    # A Change-ID is a whole token of the comment, not a prefix of one.
    # The marker parser reads only between the markers, so for its
    # result they delimit a token as whitespace does.
    text = latest
    if result.detection_source == "marker":
        text = text.replace(_START, " ").replace(_END, " ")
    tokens = {token.strip("_*") for token in text.split()}
    for change_id in mapping.change_ids:
        if not _CHANGE_ID.fullmatch(change_id) or change_id not in tokens:
            raise violation(
                "mapping holds something other than a Change-ID", change_id, latest
            )


# --- entry point ---------------------------------------------------------


def check(
    data: bytes,
    *,
    netrc_parser: type[NetrcParser] = NetrcParser,
    gitreview: Callable[[str], GitReviewInfo | None] = parse_gitreview,
    policy: Callable[..., PolicySource | None] = parse_policy_blob,
    github2gerrit: Callable[
        [list[dict[str, Any]]], GitHub2GerritDetectionResult
    ] = detect_github2gerrit_comments,
) -> None:
    """Run the contract the first byte selects on the rest of *data*."""
    if not data:
        return
    flags, rest = data[0], data[1:]
    target, structured = flags & 3, bool(flags & 0x04)
    if target == 0:
        if structured:
            _check_netrc_round_trip(flags, fields(rest, 64), netrc_parser)
        else:
            _check_netrc_raw(fields(rest, 1)[0], netrc_parser)
    elif target == 1:
        if structured:
            _check_gitreview_structured(flags, fields(rest, 3), gitreview)
        else:
            _check_gitreview_raw(fields(rest, 1)[0], gitreview)
    elif target == 2:
        parts = fields(rest, 32) if structured else fields(rest, 1)
        _check_policy(flags, [p for p in parts if p or not structured], policy)
    else:
        _check_github2gerrit(flags, fields(rest, 8), github2gerrit)


def _seed(flags: int, *parts: str) -> bytes:
    return bytes([flags]) + "\x00".join(parts).encode("utf-8")


_CID = "I" + "0123456789abcdef" * 2 + "01234567"
_BLOCK = f"PR: https://github.com/acme/widget/pull/7\nMode: squash\nTopic: GH-widget-7\nChange-Ids:\n  {_CID}\nGitHub-Hash: abc123"

SEEDS: tuple[bytes, ...] = (
    _seed(
        0x04, "github.com", "user", "ghp_token", "Gerrit.Example.org", "me", 'p w#\\"x'
    ),
    _seed(0x3C, "a b", "x\\", "p#q", "a B", "l", "last"),
    # The fix in this change: an escaped backslash ends the login.
    _seed(0x00, 'machine h login "x\\\\" password "p#q"\n'),
    _seed(0x00, "machine h\nmacdef init\ncd /tmp\n\nmachine k login a password b\n"),
    _seed(0x00, "machine"),
    _seed(0x05, "gerrit.linuxfoundation.org", "29418", "releng/builder.git"),
    _seed(0x3D, " gerrit.example.org ", "２９４１８", "project "),
    # The fix in this change: more digits than int() reads.
    _seed(0x0D, "gerrit.example.org", "9" * 5000, "p"),
    _seed(0x01, "[gerrit]\nhost=h # primary\nport=29418 # ssh\nproject=a.git.git\n"),
    _seed(0x06, "dependabot", "app/Renovate[bot]", "*", "pre-commit-ci"),
    _seed(0x46, "dependabot", "dependabot[bot]"),
    _seed(0x02, '{"include_bots": ["*"], "exclude_bots": "dependabot, renovate"}'),
    _seed(
        0x02, '{"include_bots": ["\\u0064ependabot"], "exclude_bots": "\\u0072enovate"}'
    ),
    _seed(0x02, '{"include_bots": "\\u002a"}'),
    # The fix in this change: nesting that exhausts Python 3.11's decoder.
    _seed(0x02, "[" * 5000),
    _seed(0x02, "[" * 8000 + "]" * 8000),
    _seed(0x82, "{}"),
    _seed(0x07, _BLOCK, "plain comment", _BLOCK.replace("squash", "multi-commit")),
    _seed(0xFB, f"Topic: GH-widget-7\nChange-Id: {_CID}\nGitHub-Hash: abc", "", "x"),
    _seed(0x07, _BLOCK.replace(_CID, _CID + "-extra")),
    # The fix in this change: bot-authored, so the heuristic sees it too.
    _seed(0x0F, _BLOCK.replace(_CID, _CID + "-extra")),
    _seed(0x02, '{"include_bots": ["notdependabot", "app/Renovate[bot]"]}'),
    _seed(0x02, '{"include_bots": ["*x", "dependabot,*"]}'),
    # The closing marker straight after the Change-ID, with no newline.
    _seed(0x03, f"{_START}\nMode: squash\nTopic: GH-w-7\nChange-Ids:\n  {_CID}{_END}"),
)


def main() -> None:
    """Fuzz :func:`check` from :data:`SEEDS`."""
    import atheris

    run(atheris, check, SEEDS)


if __name__ == "__main__":
    main()
