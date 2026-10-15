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

An invalid repository file takes that repository out of scope, because
it cannot say what should be in scope there.  An invalid owner file
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
    parse_policy_document,
)

__all__ = [
    "PolicyFiles",
    "apply_policy_files",
    "load_user_policy",
    "parse_policy_blob",
    "user_config_path",
]


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
    """The repository and owner policy files a run has read.

    Keyed by lower-cased ``owner/name`` and owner login.  Each value is
    what GraphQL returned for the path: the Blob's ``text``,
    ``byteSize``, ``isTruncated`` and ``isBinary``; ``{}`` when the path
    is not a file; or ``None`` when nothing is there.
    """

    owners: Mapping[str, Mapping[str, Any] | None] = field(default_factory=dict)
    repositories: Mapping[str, Mapping[str, Any] | None] = field(default_factory=dict)


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


def _parse_each(
    blobs: Mapping[str, Mapping[str, Any] | None], level: PolicyLevel, suffix: str
) -> tuple[dict[str, PolicySource | None], dict[str, BotPolicyError]]:
    """Parse every file, labelled by its lower-cased key and ``suffix``.

    Returns:
        Every key's source, ``None`` where there is none or it is
        invalid, and the invalid keys' errors.
    """
    sources: dict[str, PolicySource | None] = {}
    errors: dict[str, BotPolicyError] = {}
    for key, blob in blobs.items():
        key = key.lower()
        sources[key] = None
        try:
            sources[key] = parse_policy_blob(blob, label=f"{key}{suffix}", level=level)
        except BotPolicyError as exc:
            errors[key] = exc
    return sources, errors


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
    repository's, ``acme/.github`` for its owner's.  What ``files``
    holds for a key replaces whatever ``policy`` held for it.

    Returns:
        The new policy, recording each invalid repository file in its
        ``invalid_repositories``, and every invalid owner file's error,
        which the caller must not ignore.
    """
    owners, owner_errors = _parse_each(files.owners, PolicyLevel.OWNER, "/.github")
    repositories, repository_errors = _parse_each(
        files.repositories, PolicyLevel.REPOSITORY, ""
    )
    invalid = {
        key: problems
        for key, problems in policy.invalid_repositories.items()
        if key not in repositories
    }
    invalid |= {key: "; ".join(exc.problems) for key, exc in repository_errors.items()}
    applied = replace(
        policy,
        owners=_merged(policy.owners, owners),
        repositories=_merged(policy.repositories, repositories),
        invalid_repositories=invalid,
    )
    return applied, list(owner_errors.values())
