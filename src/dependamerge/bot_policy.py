# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Which GitHub Apps a run may act on.

Dependamerge exists to land the steady stream of version bumps that
dependency and hook-update bots raise.  Other Apps raise pull requests
too, and some of those carry code changes that need human review, so
being a bot is not enough to be in scope:

- the **built-in** Apps below are in scope unless a policy excludes them;
- **every other** App is out of scope unless a policy includes it.

A policy names Apps in two lists, ``include_bots`` and ``exclude_bots``,
and can come from several sources.  The most specific source that names
an App decides for it:

1. the run's own ``--include-bots`` / ``--exclude-bots`` flags;
2. the repository's policy file (added in a later change);
3. the owner's policy file (likewise);
4. the user's ``config.json`` under ``$XDG_CONFIG_HOME/dependamerge``;
5. the built-in list.

Personal defaults therefore never override what a repository or owner
declares; only a run's explicit flags can.

Authors reach this module in canonical form (see
:mod:`dependamerge.bot_identity`), where every App, and nothing else,
carries the ``[bot]`` suffix.  People are never in scope here: whether a
human-authored pull request may be acted on is a separate decision.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from .bot_identity import BOT_SUFFIX, is_automation_author

__all__ = [
    "BUILTIN_BOTS",
    "WILDCARD",
    "BotDecision",
    "BotPolicy",
    "BotPolicyError",
    "PolicyLevel",
    "PolicySource",
    "app_slug",
    "load_user_policy",
    "parse_policy_document",
    "run_policy_source",
    "user_config_path",
]

#: The Apps in scope unless a policy excludes them: third-party tools
#: whose pull requests are dependency or hook-version updates.
BUILTIN_BOTS: frozenset[str] = frozenset(
    {"dependabot[bot]", "renovate[bot]", "pre-commit-ci[bot]"}
)

#: In ``include_bots``, includes every App.
WILDCARD = "*"

#: Keys a policy document may hold.
_POLICY_KEYS = ("include_bots", "exclude_bots")

#: A policy document larger than this is refused rather than parsed.
_MAX_DOCUMENT_BYTES = 16 * 1024

# Entries may be separated by commas or whitespace, so a single string, a
# repeated flag and a multi-line value read the same way.
_SEPARATORS = re.compile(r"[,\s]+")

# GitHub App slugs: lower-case letters, digits and hyphens.
_SLUG = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,98}[a-z0-9])?\Z")


class BotPolicyError(ValueError):
    """A policy is invalid; ``problems`` lists every reason at once."""

    def __init__(self, source: str, problems: Iterable[str]) -> None:
        self.source = source
        self.problems = list(problems)
        super().__init__(f"{source}: " + "; ".join(self.problems))


class PolicyLevel(IntEnum):
    """Where a policy came from; a lower value is more specific."""

    RUN = 1
    REPOSITORY = 2
    OWNER = 3
    USER = 4


@dataclass(frozen=True)
class PolicySource:
    """The Apps one source includes and excludes.

    ``include_label`` and ``exclude_label`` name the source in decisions,
    so a run's flags can be told apart from each other while a file is
    named once for both lists.
    """

    level: PolicyLevel
    include_label: str
    exclude_label: str
    include: frozenset[str] = frozenset()
    exclude: frozenset[str] = frozenset()
    include_all: bool = False

    @property
    def listed(self) -> frozenset[str]:
        """Every App this source names, in either list."""
        return self.include | self.exclude


@dataclass(frozen=True)
class BotDecision:
    """Whether an author is in scope, and the reason, for reporting."""

    in_scope: bool
    reason: str


def app_slug(login: str) -> str:
    """Return an App login without its ``[bot]`` suffix, for messages."""
    return login[: -len(BOT_SUFFIX)] if login.lower().endswith(BOT_SUFFIX) else login


def _normalise_entry(entry: str) -> str:
    """Return ``entry`` as a canonical App login, or raise ``ValueError``.

    Accepts ``name``, ``name[bot]`` and ``app/name`` (the form GitHub's
    search and ``gh`` use), case-insensitively.
    """
    slug = entry.strip().lower()
    slug = slug.removeprefix("app/")
    slug = slug.removesuffix(BOT_SUFFIX)
    if not _SLUG.match(slug):
        raise ValueError(f"'{entry}' is not a GitHub App name")
    return f"{slug}{BOT_SUFFIX}"


def _split_entries(values: Iterable[str]) -> list[str]:
    return [item for value in values for item in _SEPARATORS.split(value) if item]


def _parse_lists(
    include_values: Iterable[str],
    exclude_values: Iterable[str],
    problems: list[str],
) -> tuple[frozenset[str], frozenset[str], bool]:
    """Normalise both lists, recording every problem found."""
    include: set[str] = set()
    include_all = False
    for entry in _split_entries(include_values):
        if entry == WILDCARD:
            include_all = True
            continue
        try:
            include.add(_normalise_entry(entry))
        except ValueError as exc:
            problems.append(f"include_bots: {exc}")

    exclude: set[str] = set()
    for entry in _split_entries(exclude_values):
        if entry == WILDCARD:
            problems.append("exclude_bots: '*' is only valid in include_bots")
            continue
        try:
            exclude.add(_normalise_entry(entry))
        except ValueError as exc:
            problems.append(f"exclude_bots: {exc}")

    for both in sorted(include & exclude):
        problems.append(f"{app_slug(both)} is both included and excluded")
    return frozenset(include), frozenset(exclude), include_all


def run_policy_source(
    include_bots: Iterable[str] | None,
    exclude_bots: Iterable[str] | None,
) -> PolicySource | None:
    """Build the run's own source from ``--include-bots``/``--exclude-bots``.

    Each flag may repeat, and each value may hold several entries
    separated by commas or whitespace.

    Returns:
        The source, or ``None`` when neither flag was given.

    Raises:
        BotPolicyError: When any entry is invalid.
    """
    include_values = list(include_bots or [])
    exclude_values = list(exclude_bots or [])
    if not include_values and not exclude_values:
        return None
    problems: list[str] = []
    include, exclude, include_all = _parse_lists(
        include_values, exclude_values, problems
    )
    if problems:
        raise BotPolicyError("--include-bots/--exclude-bots", problems)
    return PolicySource(
        level=PolicyLevel.RUN,
        include_label="--include-bots",
        exclude_label="--exclude-bots",
        include=include,
        exclude=exclude,
        include_all=include_all,
    )


def _list_value(
    document: Mapping[str, Any], key: str, problems: list[str]
) -> list[str]:
    value = document.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    problems.append(f"{key}: must be a string or a list of strings")
    return []


def parse_policy_document(text: str, *, label: str, level: PolicyLevel) -> PolicySource:
    """Parse a JSON policy document.

    Unknown keys are refused rather than ignored: a misspelt
    ``exlude_bots`` must not quietly leave a bot in scope.

    Args:
        text: The document.
        label: How decisions name this source, e.g. a path.
        level: Where the document came from.

    Raises:
        BotPolicyError: When the document is invalid, with every problem.
    """
    if len(text.encode("utf-8")) > _MAX_DOCUMENT_BYTES:
        raise BotPolicyError(label, [f"larger than {_MAX_DOCUMENT_BYTES // 1024} KiB"])
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise BotPolicyError(label, [f"not valid JSON ({exc})"]) from None
    if not isinstance(document, dict):
        raise BotPolicyError(label, ["must be a JSON object"])

    problems: list[str] = []
    unknown = sorted(set(document) - set(_POLICY_KEYS))
    if unknown:
        problems.append(
            f"unknown key(s) {', '.join(unknown)}; "
            f"expected only {', '.join(_POLICY_KEYS)}"
        )
    include_values = _list_value(document, "include_bots", problems)
    exclude_values = _list_value(document, "exclude_bots", problems)
    include, exclude, include_all = _parse_lists(
        include_values, exclude_values, problems
    )
    if problems:
        raise BotPolicyError(label, problems)
    return PolicySource(
        level=level,
        include_label=label,
        exclude_label=label,
        include=include,
        exclude=exclude,
        include_all=include_all,
    )


def user_config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Return the user's policy file path.

    ``$XDG_CONFIG_HOME/dependamerge/config.json``, defaulting to
    ``~/.config/dependamerge/config.json`` on every platform.
    """
    env = os.environ if environ is None else environ
    base = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "dependamerge" / "config.json"


def _display_path(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def load_user_policy(path: Path | None = None) -> PolicySource | None:
    """Load the user's policy file, if there is one.

    Returns:
        The source, or ``None`` when the file does not exist.

    Raises:
        BotPolicyError: When the file cannot be read or is invalid.
    """
    path = path or user_config_path()
    label = _display_path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise BotPolicyError(label, [f"cannot be read ({exc})"]) from None
    return parse_policy_document(text, label=label, level=PolicyLevel.USER)


@dataclass(frozen=True)
class BotPolicy:
    """The Apps a run may act on, resolved per App and repository.

    A policy with no sources admits exactly the built-in Apps, so code
    that forgets to supply one fails safe.
    """

    run: PolicySource | None = None
    user: PolicySource | None = None
    # Keyed by lower-cased owner login and ``owner/name`` respectively.
    owners: Mapping[str, PolicySource] = field(default_factory=dict)
    repositories: Mapping[str, PolicySource] = field(default_factory=dict)

    def _sources(self, repository: str | None) -> list[PolicySource]:
        """The sources that apply to ``repository``, most specific first."""
        repository = (repository or "").lower()
        owner = repository.split("/", 1)[0] if "/" in repository else ""
        candidates = (
            self.run,
            self.repositories.get(repository),
            self.owners.get(owner),
            self.user,
        )
        return [source for source in candidates if source is not None]

    def decide(self, author: str | None, repository: str | None = None) -> BotDecision:
        """Decide whether ``author`` is in scope in ``repository``.

        Args:
            author: A canonical login.
            repository: ``owner/name``; ``None`` consults only the run's
                flags, the user's file and the built-in list.
        """
        if not is_automation_author(author):
            return BotDecision(False, "not a GitHub App")
        assert author is not None
        login = author.lower()
        for source in self._sources(repository):
            if login in source.exclude:
                return BotDecision(False, f"excluded by {source.exclude_label}")
            if login in source.include or source.include_all:
                return BotDecision(True, f"included by {source.include_label}")
        if login in BUILTIN_BOTS:
            return BotDecision(True, "built-in")
        return BotDecision(False, "not included by any policy")

    def admits(self, author: str | None, repository: str | None = None) -> bool:
        """Whether ``author`` is an App in scope in ``repository``."""
        return self.decide(author, repository).in_scope

    def listed(self) -> list[tuple[str, bool, str]]:
        """Every App a source names, for checking that each one exists.

        Returns:
            ``(login, excluded, label)`` triples, exclusions first, one
            per App and list, in a stable order.
        """
        seen: dict[tuple[str, bool], str] = {}
        sources = [
            source
            for source in (
                self.run,
                *self.repositories.values(),
                *self.owners.values(),
                self.user,
            )
            if source is not None
        ]
        for source in sources:
            for login in source.exclude:
                seen.setdefault((login, True), source.exclude_label)
            for login in source.include:
                seen.setdefault((login, False), source.include_label)
        return sorted(
            ((login, excluded, label) for (login, excluded), label in seen.items()),
            key=lambda item: (not item[1], item[0]),
        )
