# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Limiting an owner-wide merge to named repositories.

An owner-wide run enumerates every non-archived, non-fork repository of
an organisation or user account.  ``--include-repos`` narrows that to
the repositories it names; ``--exclude-repos`` removes the ones it
names.  The two are mutually exclusive, so a run states its scope one
way and there is no precedence rule to remember.

Every name must match a repository the run enumerates, or the run stops
before anything merges.  In an inclusion list an unmatched name is
almost always a typo, and carrying on would quietly do less than asked.
In an exclusion list it matters more: a misspelt exclusion leaves in
scope the very repository it was meant to protect, and nothing short of
stopping can tell those two cases apart.  Archived repositories and
forks are never enumerated, so naming one is refused the same way.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from .url_parser import UrlParseError, require_repo

__all__ = ["RepoSelection", "parse_repo_selection"]

INCLUDE_FLAG = "--include-repos"
EXCLUDE_FLAG = "--exclude-repos"

# Repository names cannot contain commas or whitespace, so either
# separates entries: a single comma-separated value, repeated flags, and
# a multi-line value from a workflow input all read the same way.
_SEPARATORS = re.compile(r"[,\s]+")


@dataclass
class RepoSelection:
    """The repositories an owner-wide run may act on.

    ``names`` maps each lower-cased repository name to the spelling the
    operator gave, so matching is case-insensitive (as GitHub's is) while
    messages repeat the operator's own words back to them.

    ``admits`` is consulted once per enumerated repository and remembers
    every name it was asked about, which is what lets ``unmatched``
    report the names that matched nothing once enumeration finishes.
    """

    owner: str
    exclude: bool
    names: dict[str, str]
    _enumerated: set[str] = field(default_factory=set, init=False, repr=False)

    @property
    def flag(self) -> str:
        """The option this selection came from, for messages."""
        return EXCLUDE_FLAG if self.exclude else INCLUDE_FLAG

    def admits(self, repo_full_name: str) -> bool:
        """Whether the run may act on ``repo_full_name``."""
        name = repo_full_name.rsplit("/", 1)[-1].lower()
        self._enumerated.add(name)
        return (name in self.names) != self.exclude

    def unmatched(self) -> list[str]:
        """Named repositories that no enumerated repository matched."""
        return [
            shown for key, shown in self.names.items() if key not in self._enumerated
        ]

    def describe(self) -> str:
        """One line saying how this run is scoped."""
        verb = "Excluding" if self.exclude else "Limited to"
        return f"{verb}: {', '.join(self.names.values())}"


def _split(values: Iterable[str]) -> list[str]:
    """Flatten repeated and comma- or whitespace-separated values."""
    return [token for value in values for token in _SEPARATORS.split(value) if token]


def _bare_name(token: str, owner: str, flag: str) -> str:
    """Return the repository name ``token`` gives, checked against ``owner``.

    Accepts ``name`` or ``owner/name``.  An ``owner/name`` for a
    different owner is refused rather than ignored: the run cannot act
    on it, and silently dropping it would hide the mistake.
    """
    parts = token.split("/")
    if len(parts) == 2:
        named_owner, name = parts
        if named_owner.lower() != owner.lower():
            raise ValueError(
                f"{flag}: '{token}' belongs to '{named_owner}', but this run "
                f"covers '{owner}'"
            )
    elif len(parts) == 1:
        name = parts[0]
    else:
        raise ValueError(f"{flag}: '{token}' is not 'name' or 'owner/name'")
    try:
        return require_repo(name)
    except UrlParseError as exc:
        raise ValueError(f"{flag}: {exc}") from None


def _parse_names(values: list[str], owner: str, flag: str) -> dict[str, str]:
    """Parse one option's values into a lower-cased-name lookup."""
    tokens = _split(values)
    if not tokens:
        # An explicit narrowing that names nothing must never widen to
        # every repository, so this is an error rather than a no-op.
        raise ValueError(f"{flag}: names no repository; omit it to act on every one")
    names: dict[str, str] = {}
    for token in tokens:
        name = _bare_name(token, owner, flag)
        names.setdefault(name.lower(), name)
    return names


def parse_repo_selection(
    owner: str,
    include: list[str] | None,
    exclude: list[str] | None,
) -> RepoSelection | None:
    """Build the selection the two options describe.

    Args:
        owner: The organisation or user the run covers.
        include: ``--include-repos`` values, or ``None`` when not given.
        exclude: ``--exclude-repos`` values, or ``None`` when not given.

    Returns:
        The selection, or ``None`` when neither option was given.

    Raises:
        ValueError: When both options are given, when one names nothing,
            or when a name is malformed or belongs to another owner.
    """
    if include is not None and exclude is not None:
        raise ValueError(
            f"{INCLUDE_FLAG} and {EXCLUDE_FLAG} are mutually exclusive; "
            "pass one or the other"
        )
    if include is not None:
        return RepoSelection(owner, False, _parse_names(include, owner, INCLUDE_FLAG))
    if exclude is not None:
        return RepoSelection(owner, True, _parse_names(exclude, owner, EXCLUDE_FLAG))
    return None
