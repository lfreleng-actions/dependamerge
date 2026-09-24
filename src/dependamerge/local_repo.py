# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Inferring the target from the repository the operator is standing in.

Running ``dependamerge merge`` with no URL should mean "this
repository".  Working that out is a matter of asking git where it is
and what it points at, then deciding whether the answer describes a
GitHub repository or a Gerrit project.

Gerrit needs identifying separately rather than being treated as an
odd-looking GitHub remote.  Its changes are not addressable as
``/owner/repo``, so a Gerrit checkout that silently fell through to the
GitHub path would fail somewhere far less informative than here.

The Gerrit heuristics in this module rank evidence rather than trusting
a name.  None of them is a trust decision --- they choose which parser
to use on the operator's own checkout, never whether to send a
credential somewhere.  Host *authorisation* stays in
``url_parser.hosts``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .git_ops import GitError, run_git
from .gitreview import GitReviewInfo, parse_gitreview
from .local_remote import (
    _gerrit_identity_from_remote,
    _looks_like_gerrit_remote,
    _remote_web_url,
)

# ``host_suggests_gerrit`` is public and was importable from here before
# the remote-string helpers moved to ``local_remote``.  Re-exported with
# an explicit alias so the split stays a refactor rather than a breaking
# change for anything outside this repository.
from .local_remote import host_suggests_gerrit as host_suggests_gerrit
from .url_parser import (
    ChangeSource,
)

log = logging.getLogger("dependamerge.local_repo")

#: Gerrit's default SSH port.  A remote using it is Gerrit; nothing
#: else conventionally listens there.
_GERRIT_SSH_PORT = "29418"

#: Bound on every git call.  ``run_git`` has no default timeout, and
#: these run before the operator has seen any output, so a hung git
#: would look like the tool itself hanging on startup.
_GIT_TIMEOUT = 10.0

#: Remotes consulted, in order.  ``origin`` is the conventional
#: upstream; ``upstream`` is the fork convention, where ``origin`` is
#: the operator's own fork and not what they mean to merge.
_REMOTE_PREFERENCE = ("upstream", "origin")

#: Credentials in a URL, for any scheme.
#: Any ``scheme://`` prefix.
_SCHEME_RE = re.compile(r"\A[A-Za-z][A-Za-z0-9+.-]*://")

#: scp-style remote: ``[user@]host:path``.
_SCP_REMOTE_RE = re.compile(r"\A(?:(?P<user>[^@/]+)@)?(?P<host>[^@/:]+):(?!//)[^\s]+\Z")


@dataclass(frozen=True)
class LocalTarget:
    """What the current checkout points at.

    Attributes:
        source: Which platform the checkout belongs to.
        url: A URL the existing parsers understand.  Empty for Gerrit,
            whose changes are not addressable from the checkout alone.
        remote: The git remote the answer came from, for reporting.
        root: The repository's working tree root.
        gitreview: Gerrit's ``.gitreview``, when the checkout has one.
        host: The server the checkout belongs to, however it was
            determined.  Reported back to the operator, so it is
            populated for a Gerrit checkout recognised by its remote
            as well as one carrying a ``.gitreview``.
        project: The Gerrit project, when it is known.  A remote gives
            this away in its path even without a ``.gitreview``.
    """

    source: ChangeSource
    url: str
    remote: str
    root: Path
    gitreview: GitReviewInfo | None = None
    host: str = ""
    project: str = ""

    @property
    def is_gerrit(self) -> bool:
        """Check whether the checkout belongs to a Gerrit server."""
        return self.source == ChangeSource.GERRIT


def _git(args: list[str], cwd: Path | None) -> str | None:
    """Run a read-only git command, returning stripped stdout or None.

    Every failure mode here --- not a repository, no such remote, git
    missing entirely, an unreadable working directory --- is an
    ordinary "cannot infer" answer rather than an error, because the
    caller always has the option of asking the operator for a URL.

    ``OSError`` is caught alongside ``GitError`` deliberately.
    ``run_git`` converts timeouts and non-zero exits, but process
    creation failures --- a missing ``git``, a working directory that
    does not exist --- surface as ``FileNotFoundError`` and would
    otherwise crash the command with a traceback instead of the
    guidance.
    """
    try:
        result = run_git(args, cwd=cwd, check=True, timeout=_GIT_TIMEOUT)
    except (GitError, OSError) as exc:
        log.debug("git %s failed: %s", " ".join(args[1:]), exc)
        return None
    output = result.stdout.strip()
    return output or None


def repository_root(cwd: Path | None = None) -> Path | None:
    """Return the working tree root, or None outside a repository."""
    root = _git(["git", "rev-parse", "--show-toplevel"], cwd)
    return Path(root) if root else None


def remote_url(
    root: Path, *, preference: tuple[str, ...] = _REMOTE_PREFERENCE
) -> tuple[str, str] | None:
    """Return the first *usable* remote from ``preference``.

    Usable, not merely configured.  A checkout may have ``upstream``
    pointing at a local mirror or an unsupported transport while
    ``origin`` is a perfectly good hosted remote; stopping at the first
    name configured then reported that the repository had no usable
    remote at all, when the answer was one line further down.

    Args:
        root: The repository root.
        preference: Remote names to try, most preferred first.

    Returns:
        A ``(remote_name, url)`` pair whose URL yields a target, or
        None when no configured remote does.
    """
    fallback: tuple[str, str] | None = None
    for name in preference:
        url = _git(["git", "remote", "get-url", name], root)
        if not url:
            continue
        if _remote_web_url(url) is not None:
            return (name, url)
        if fallback is None:
            # Remember the first configured-but-unusable remote so the
            # caller can still name it when nothing resolves --- a
            # Gerrit checkout is identified from the remote even when
            # no web URL can be derived from it.
            fallback = (name, url)
    return fallback


def _read_gitreview(root: Path) -> GitReviewInfo | None:
    """Parse ``.gitreview`` from the working tree, if it has one.

    The existing parser is pure, but the only fetch path was the GitHub
    contents API; a checkout has the file on disk.
    """
    path = root / ".gitreview"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.debug("could not read %s: %s", path, exc)
        return None
    return parse_gitreview(text)


def detect_local_target(cwd: Path | None = None) -> LocalTarget | None:
    """Work out what the current checkout points at.

    Evidence is ranked, strongest first: a ``.gitreview`` file is
    Gerrit's own declaration and settles it; then the remote's SSH
    port; then the shape of its hostname.  Anything else is treated as
    GitHub, which is what the URL parsers already assume.

    Args:
        cwd: Directory to inspect.  Defaults to the process's own.

    Returns:
        The inferred target, or None when the directory is not a git
        repository or has no usable remote.
    """
    root = repository_root(cwd)
    if root is None:
        return None

    found = remote_url(root)
    remote_name, url = found if found else ("", "")

    gitreview = _read_gitreview(root)
    if gitreview is not None and gitreview.is_valid:
        # Gerrit's own declaration of where the project lives, and the
        # reason .gitreview exists at all.  Trusted over the remote,
        # which may point at a replica or a personal mirror.
        return LocalTarget(
            source=ChangeSource.GERRIT,
            url="",
            remote=remote_name,
            root=root,
            gitreview=gitreview,
            host=gitreview.host,
            project=gitreview.project,
        )

    if not url:
        return None

    if _looks_like_gerrit_remote(url):
        # Recognised from the remote alone.  Its host and path are the
        # only identity available, and reporting them is the difference
        # between actionable guidance and a bare refusal.
        host, project = _gerrit_identity_from_remote(url)
        return LocalTarget(
            source=ChangeSource.GERRIT,
            url="",
            remote=remote_name,
            root=root,
            host=host,
            project=project,
        )

    normalized = _remote_web_url(url)
    if normalized is None:
        # A remote git can use but this tool cannot target, such as a
        # local mirror.  Nothing to infer, and the caller can still ask
        # the operator for a URL.
        return None
    return LocalTarget(
        source=ChangeSource.GITHUB,
        url=normalized,
        remote=remote_name,
        root=root,
        host=normalized.split("://", 1)[-1].split("/", 1)[0],
    )
