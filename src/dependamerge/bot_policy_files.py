# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Bot policy files: the user's, and those a repository or owner holds.

:mod:`dependamerge.bot_policy` decides which Apps are in scope from its
sources; this module turns files into those sources.

- The user's file is ``config.json`` under ``$XDG_CONFIG_HOME/dependamerge``.
- A repository's is ``.github/dependamerge/config.json`` on its default
  branch, and an owner's is the same file in its ``.github`` repository.
  They arrive as GraphQL ``Blob`` objects with queries the run makes
  anyway (see :mod:`dependamerge.github_service`), collected as
  :class:`PolicyFiles`.
- A repository or owner may also hold the same document in its
  ``DEPENDAMERGE_BOTS`` Actions variable.  At one level the file and the
  variable combine: each list is the union of both, and an App one
  includes while the other excludes is an error.

An invalid repository policy takes that repository out of scope, because
it cannot say what should be in scope there.  An invalid owner policy
governs every repository the owner has, so it is returned for the run
to stop on.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .bot_policy import (
    _MAX_DOCUMENT_BYTES,
    BotPolicy,
    BotPolicyError,
    PolicyLevel,
    PolicySource,
    app_slug,
    parse_policy_document,
)

__all__ = [
    "POLICY_VARIABLE",
    "PolicyFiles",
    "apply_policy_files",
    "load_user_policy",
    "parse_policy_blob",
    "user_config_path",
]

#: The Actions variable that may hold a repository's or owner's policy.
POLICY_VARIABLE = "DEPENDAMERGE_BOTS"


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
class PolicyFiles:
    """The repository and owner policy files and variables a run has read.

    Keyed by lower-cased ``owner/name`` and owner login.  Each file is
    what GraphQL returned for the path: the Blob's ``text``,
    ``byteSize``, ``isTruncated`` and ``isBinary``; ``{}`` when the path
    is not a file; or ``None`` when nothing is there.  Each variable is
    the value of :data:`POLICY_VARIABLE`, or ``None`` when there is none.
    ``variables_refused`` holds the kinds of variable, ``orgs`` or
    ``repos``, the token proved unable to read, after which the run read
    no more of that kind: a classic token without ``admin:org`` is
    refused an organisation's variables yet reads repositories'.
    """

    owners: Mapping[str, Mapping[str, Any] | None] = field(default_factory=dict)
    repositories: Mapping[str, Mapping[str, Any] | None] = field(default_factory=dict)
    owner_variables: Mapping[str, str | None] = field(default_factory=dict)
    repository_variables: Mapping[str, str | None] = field(default_factory=dict)
    variables_refused: frozenset[str] = frozenset()

    @property
    def variables_unreadable(self) -> bool:
        """Whether the token could not read some kind of variable."""
        return bool(self.variables_refused)


def parse_policy_blob(
    blob: Mapping[str, Any] | None, *, label: str, level: PolicyLevel
) -> PolicySource | None:
    """Parse a policy file as GraphQL returns it.

    Args:
        blob: A :class:`PolicyFiles` value.
        label: How decisions name this source, e.g. ``acme/widget``.
        level: Where the file came from.

    Returns:
        The source, or ``None`` when there is no file.

    Raises:
        BotPolicyError: When the path holds anything but a policy
            document small enough to read whole.
    """
    if blob is None:
        return None
    size = blob.get("byteSize")
    if not isinstance(size, int):
        raise BotPolicyError(label, ["is not a file"])
    if blob.get("isBinary"):
        raise BotPolicyError(label, ["is not a text file"])
    if blob.get("isTruncated") or size > _MAX_DOCUMENT_BYTES:
        raise BotPolicyError(label, [f"larger than {_MAX_DOCUMENT_BYTES // 1024} KiB"])
    text = blob.get("text")
    if not isinstance(text, str):
        raise BotPolicyError(label, ["has no text"])
    return parse_policy_document(text, label=label, level=level)


def _combine(
    file: PolicySource | None, variable: PolicySource | None, label: str
) -> PolicySource | None:
    """One level's source from its file and its variable, either absent.

    Each list is the union of both, labelled by the source that filled
    it, or by ``label`` when both did.

    Raises:
        BotPolicyError: Naming every App one includes and the other
            excludes.
    """
    if file is None or variable is None:
        return file or variable
    problems = [
        f"{app_slug(login)} is included by {including.include_label} "
        f"and excluded by {excluding.exclude_label}"
        for including, excluding in ((file, variable), (variable, file))
        for login in sorted(including.include & excluding.exclude)
    ]
    if problems:
        raise BotPolicyError(label, problems)
    including = [s for s in (file, variable) if s.include or s.include_all]
    excluding = [s for s in (file, variable) if s.exclude]
    return PolicySource(
        level=file.level,
        include_label=including[0].include_label if len(including) == 1 else label,
        exclude_label=excluding[0].exclude_label if len(excluding) == 1 else label,
        include=file.include | variable.include,
        exclude=file.exclude | variable.exclude,
        include_all=file.include_all or variable.include_all,
    )


def _level_source(
    key: str,
    blob: Mapping[str, Any] | None,
    value: str | None,
    level: PolicyLevel,
    suffix: str,
) -> tuple[PolicySource | None, list[BotPolicyError]]:
    """``key``'s source from its file and variable, and why it is invalid.

    The file is labelled by ``key`` and ``suffix``, the variable by
    ``key`` and :data:`POLICY_VARIABLE`.
    """
    file_label = f"{key}{suffix}"
    variable_label = f"{key} {POLICY_VARIABLE}"
    errors: list[BotPolicyError] = []
    file = variable = None
    try:
        file = parse_policy_blob(blob, label=file_label, level=level)
    except BotPolicyError as exc:
        errors.append(exc)
    try:
        if value is not None:
            variable = parse_policy_document(value, label=variable_label, level=level)
    except BotPolicyError as exc:
        errors.append(exc)
    if errors:
        return None, errors
    try:
        return _combine(file, variable, f"{file_label} and {variable_label}"), []
    except BotPolicyError as exc:
        return None, [exc]


def _level_sources(
    blobs: Mapping[str, Mapping[str, Any] | None],
    variables: Mapping[str, str | None],
    level: PolicyLevel,
    suffix: str,
) -> tuple[dict[str, PolicySource | None], dict[str, list[BotPolicyError]]]:
    """Every key's source at ``level``, from its file and its variable.

    Returns:
        Every key's source, ``None`` where there is none or it is
        invalid, and the invalid keys' errors.
    """
    blobs = {key.lower(): blob for key, blob in blobs.items()}
    variables = {key.lower(): value for key, value in variables.items()}
    sources: dict[str, PolicySource | None] = {}
    errors: dict[str, list[BotPolicyError]] = {}
    for key in [*blobs, *(key for key in variables if key not in blobs)]:
        sources[key], problems = _level_source(
            key, blobs.get(key), variables.get(key), level, suffix
        )
        if problems:
            errors[key] = problems
    return sources, errors


def _repository_problem(key: str, errors: list[BotPolicyError]) -> str:
    """An invalid repository policy's problems, the variable's named as such."""
    variable_label = f"{key} {POLICY_VARIABLE}"
    return "; ".join(
        f"{POLICY_VARIABLE}: {problem}" if error.source == variable_label else problem
        for error in errors
        for problem in error.problems
    )


def _merged(
    existing: Mapping[str, PolicySource], read: Mapping[str, PolicySource | None]
) -> dict[str, PolicySource]:
    """``existing``, with every key just read replaced by what was read."""
    kept = {key: source for key, source in existing.items() if key not in read}
    return kept | {key: source for key, source in read.items() if source is not None}


def apply_policy_files(
    policy: BotPolicy, files: PolicyFiles
) -> tuple[BotPolicy, list[BotPolicyError]]:
    """Return a copy of ``policy`` that also consults ``files``.

    Each file is labelled by where it lives: ``acme/widget`` for a
    repository's, ``acme/.github`` for its owner's.  Each variable is
    labelled ``acme/widget DEPENDAMERGE_BOTS`` or ``acme
    DEPENDAMERGE_BOTS``, and a list both fill by both labels joined with
    ``and``.  What ``files`` holds for a key, its file and its variable
    alike, replaces whatever ``policy`` held for it.

    Returns:
        The new policy, recording each invalid repository policy in its
        ``invalid_repositories``, and every invalid owner policy's
        errors, which the caller must not ignore.
    """
    owners, owner_errors = _level_sources(
        files.owners, files.owner_variables, PolicyLevel.OWNER, "/.github"
    )
    repositories, repository_errors = _level_sources(
        files.repositories, files.repository_variables, PolicyLevel.REPOSITORY, ""
    )
    invalid = {
        key: problems
        for key, problems in policy.invalid_repositories.items()
        if key not in repositories
    }
    invalid |= {
        key: _repository_problem(key, errors)
        for key, errors in repository_errors.items()
    }
    applied = replace(
        policy,
        owners=_merged(policy.owners, owners),
        repositories=_merged(policy.repositories, repositories),
        invalid_repositories=invalid,
    )
    return applied, [error for errors in owner_errors.values() for error in errors]
