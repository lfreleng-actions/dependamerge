# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Classify the version bumps a dependency update makes.

:func:`classify` reads a pull request's title and a text (the pull
request body, or a commit message) and returns the dependency updates
it found, each with a level, plus an overall level:

``major``
    The leading version number changes (``1.3.3`` to ``2.0.0``, ``0.0.2``
    to ``1.0.0``); a downgrade counts too.
``zero-minor``
    Both versions lead with ``0`` and the second number changes
    (``0.7.0`` to ``0.8.0``).
``other``
    Any other readable pair.
``unknown``
    No version pair could be read.

The overall level is the highest among the updates, in that order, so a
grouped update is ``major`` when any one dependency is.

Versions come from, in turn:

1. Lines of the text that *start* with ``Bumps <name> from <old> to
   <new>``, ``Updates `<name>` from <old> to <new>`` or the pre-commit.ci
   ``- [<repository>: <old> → <new>](…)``, read only after every
   ``<details>`` block is removed.  A dependabot body embeds the bumped
   dependency's release notes and commits in such blocks, and those
   routinely mention that dependency's own bumps; reading them more than
   doubles the ``major`` count with false positives.
2. Dependabot's metadata, when the text carries it (a commit message
   does): each ``update-type: version-update:semver-major`` line makes
   the level ``major``, and never lowers one.
3. The title's ``from <old> to <new>``, only when the text yielded no
   update.

A token is a version when, after an optional ``v`` and an optional
PEP 440 epoch (``1!``), it starts with digits and is dotted or all
digits.  An epoch change reads as ``major``.  A token of seven or more
hexadecimal characters with no dot is a commit or a digest, and makes
its update ``unknown``.

Known limit: a calendar-versioned dependency (``2025.12`` to
``2026.1``) reads as ``major`` when its year rolls over.

Everything here is pure: no I/O, no client, no logging.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

MAJOR = "major"
ZERO_MINOR = "zero-minor"
OTHER = "other"
UNKNOWN = "unknown"

#: Levels from lowest to highest.
_RANK = {UNKNOWN: 0, OTHER: 1, ZERO_MINOR: 2, MAJOR: 3}

#: Outcomes under which a major bump counts as landed (or, in a
#: preview, as about to land).
COUNTED_STATUSES = frozenset({"merged", "auto_merge_pending"})

_DETAILS_TAG = re.compile(r"<details\b|(?P<close></details\s*>)", re.IGNORECASE)

_ANCHORED = (
    re.compile(
        r"^Bumps (?:\[(?P<link>[^\]\n]+)\]\([^)\n]*\)|(?P<bare>[^\s\[]\S*))"
        r" from (?P<old>\S+) to (?P<new>\S+)",
        re.MULTILINE,
    ),
    re.compile(
        r"^Updates `(?P<name>[^`\n]+)` from (?P<old>\S+) to (?P<new>\S+)",
        re.MULTILINE,
    ),
    re.compile(
        r"^- \[(?P<name>[^\]\n]+?): (?P<old>\S+) \u2192 (?P<new>[^\s\]]+)\]\(",
        re.MULTILINE,
    ),
)

_METADATA = re.compile(
    r"^[ \t]*(?:-[ \t]*)?(?:dependency-name:[ \t]*(?P<name>\S+)"
    r"|update-type:[ \t]*version-update:semver-(?P<type>[a-z]+))[ \t]*$",
    re.MULTILINE,
)

# The name starts at a token boundary so that a long token is not
# rescanned from every position inside it.
_TITLE_PAIR = re.compile(
    r"(?:(?<!\S)(?P<name>\S+)\s+)?\bfrom\s+(?P<old>\S+)\s+to\s+(?P<new>\S+)",
    re.IGNORECASE,
)

_DIGEST = re.compile(r"[0-9a-fA-F]{7,}")
# The rest cannot start with a digit, so the leading number cannot be
# redistributed between the groups when a long token fails to match.
_VERSION = re.compile(
    r"[vV]?(?:(?P<epoch>\d+)!)?"
    r"(?P<first>\d+)(?P<rest>(?:[A-Za-z.+_\-][0-9A-Za-z.+_\-]*)?)"
)
_SECOND = re.compile(r"[^.]*\.(?P<second>\d+)")


@dataclass(frozen=True)
class Update:
    """One dependency moving from one version to another."""

    name: str
    old: str
    new: str
    level: str

    def as_dict(self) -> dict[str, str]:
        """The update as JSON-ready values."""
        return {
            "name": self.name,
            "from": self.old,
            "to": self.new,
            "level": self.level,
        }


@dataclass(frozen=True)
class Classification:
    """The updates a pull request makes and their overall level."""

    level: str
    updates: tuple[Update, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """The classification as JSON-ready values."""
        return {
            "level": self.level,
            "updates": [update.as_dict() for update in self.updates],
        }

    def major_updates(self) -> tuple[Update, ...]:
        """The updates to name for a major bump.

        The ``major`` updates; failing those (the level came from
        metadata naming no update read here), every update.
        """
        majors = tuple(u for u in self.updates if u.level == MAJOR)
        return majors or self.updates


def _token(raw: str) -> str:
    return raw.strip("`").rstrip(".,;:").strip("`")


def _number(digits: str) -> str:
    # Compared as text, not int(): Python refuses to convert a string of
    # more than 4,300 digits, and PR text may carry one.
    return digits.lstrip("0") or "0"


def _numbers(token: str) -> tuple[str, str, str | None] | None:
    """The epoch and first two numbers of a version token, or ``None``.

    A version without a PEP 440 epoch (``1!2.0.0``) has epoch ``0``.
    """
    if _DIGEST.fullmatch(token):
        return None
    match = _VERSION.fullmatch(token)
    if not match or (match["rest"] and "." not in match["rest"]):
        return None
    second = _SECOND.match(match["rest"])
    epoch = _number(match["epoch"] or "0")
    first = _number(match["first"])
    return epoch, first, _number(second["second"]) if second else None


def pair_level(old: str, new: str) -> str:
    """The level of a move from version ``old`` to version ``new``."""
    before, after = _numbers(_token(old)), _numbers(_token(new))
    if before is None or after is None:
        return UNKNOWN
    # A new epoch restarts the version scheme, so it counts as major.
    if before[:2] != after[:2]:
        return MAJOR
    if before[1] == "0" and before[2] != after[2]:
        return ZERO_MINOR
    return OTHER


def _update(name: str, old: str, new: str) -> Update:
    old, new = _token(old), _token(new)
    return Update(name.strip("`"), old, new, pair_level(old, new))


def _strip_details(text: str) -> str:
    # One pass with a nesting depth: removing the innermost block and
    # rescanning made nested blocks quadratic.
    kept: list[str] = []
    depth = 0
    start = 0
    for tag in _DETAILS_TAG.finditer(text):
        if not tag["close"]:
            if depth == 0:
                kept.append(text[start : tag.start()])
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                start = tag.end()
    # An unclosed block hides everything after it: release notes cut
    # short must not be read as the pull request's own lines.
    if depth == 0:
        kept.append(text[start:])
    return "".join(kept)


def _anchored_updates(text: str) -> list[Update]:
    found: list[tuple[int, Update]] = []
    for pattern in _ANCHORED:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            name = groups.get("name") or groups.get("link") or groups.get("bare") or ""
            found.append((match.start(), _update(name, match["old"], match["new"])))
    found.sort(key=lambda item: item[0])
    unique: list[Update] = []
    for _, update in found:
        if update not in unique:
            unique.append(update)
    return unique


def _semver_major_names(text: str) -> tuple[bool, set[str]]:
    """Whether metadata marks any update major, and the names it marks."""
    any_major = False
    names: set[str] = set()
    current: str | None = None
    for match in _METADATA.finditer(text):
        if match["name"]:
            # YAML quotes names such as "@scope/pkg"; the update lines
            # do not.
            current = match["name"].strip("\"'")
        elif match["type"] == "major":
            any_major = True
            if current:
                names.add(current)
    return any_major, names


def _title_update(title: str) -> Update | None:
    match = _TITLE_PAIR.search(title)
    if not match:
        return None
    return _update(match["name"] or "", match["old"], match["new"])


def classify(title: str | None, text: str | None) -> Classification:
    """Classify the version bumps named by ``title`` and ``text``."""
    body = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    body = _strip_details(body)
    updates = _anchored_updates(body)
    if not updates:
        title_update = _title_update(" ".join((title or "").split()))
        if title_update:
            updates = [title_update]
    any_major, major_names = _semver_major_names(body)
    if major_names:
        updates = [
            Update(u.name, u.old, u.new, MAJOR) if u.name in major_names else u
            for u in updates
        ]
    level = max((u.level for u in updates), key=_RANK.__getitem__, default=UNKNOWN)
    if any_major:
        level = MAJOR
    return Classification(level, tuple(updates))


class _HasPullRequest(Protocol):
    @property
    def pr_info(self) -> Any: ...


def classify_result(result: _HasPullRequest) -> Classification:
    """Classify a merge result from its pull request's title and body."""
    title = getattr(result.pr_info, "title", None)
    body = getattr(result.pr_info, "body", None)
    return classify(
        title if isinstance(title, str) else None,
        body if isinstance(body, str) else None,
    )


def counts_as_major(level: str, status: str) -> bool:
    """Whether a pull request at ``level`` ending ``status`` is reported."""
    return level == MAJOR and status in COUNTED_STATUSES


def major_bump_entries(
    entries: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The results-document entries whose major bump counts."""
    selected = []
    for entry in entries:
        bump = entry.get("bump")
        level = bump.get("level") if isinstance(bump, Mapping) else None
        if counts_as_major(str(level), str(entry.get("status"))):
            selected.append(entry)
    return selected
