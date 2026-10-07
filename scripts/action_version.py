#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Resolve the dependamerge version a composite-action run is executing.

The action builds dependamerge from its own source, and hatch-vcs takes
the version from git.  A runner fetches an action as a bare tree with no
``.git``, so without help every run would report a placeholder, and a
results file or Slack digest could not say which release produced it.

The action instead passes the ref it was fetched at.  A release tag
gives the version directly.  A commit SHA, the form this organisation
pins to, is matched against the repository's release tags with ``git
ls-remote``; tags here are annotated, so the peeled ``^{}`` entry is the
one naming the commit.  An untagged commit resolves to ``0.0.0+g`` and
its short SHA, which is both an honest PEP 440 version and traceable.

Nothing here may fail the run: whatever goes wrong, the fallback is
printed and the problem reported as a workflow warning.

Standard library only, as it runs before the project is installed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Sequence

#: A release tag, with or without the leading ``v``.
_RELEASE_TAG = re.compile(r"\Av?(\d+)\.(\d+)\.(\d+)\Z")
_FULL_SHA = re.compile(r"\A[0-9a-f]{40}\Z")

LsRemote = Callable[[str], str]


def _ls_remote(url: str) -> str:
    """Return ``git ls-remote --tags`` output for ``url``."""
    return subprocess.run(
        ["git", "ls-remote", "--tags", url],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    ).stdout


def _release_key(tag: str) -> tuple[int, int, int] | None:
    match = _RELEASE_TAG.match(tag)
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def tags_for_commit(listing: str, sha: str) -> list[str]:
    """Release tags in ``ls-remote`` output that point at commit ``sha``.

    A lightweight tag's own line names the commit; an annotated tag's
    line names the tag object, and its peeled ``^{}`` line the commit.
    Matching either form covers both.
    """
    tags = set()
    for line in listing.splitlines():
        object_sha, _, ref = line.partition("\t")
        if object_sha.strip().lower() != sha:
            continue
        name = ref.strip().removeprefix("refs/tags/").removesuffix("^{}")
        if _release_key(name) is not None:
            tags.add(name)
    return sorted(tags, key=lambda name: _release_key(name) or (0, 0, 0))


def resolve(
    ref: str,
    repository_url: str | Sequence[str],
    sha: str = "",
    ls_remote: LsRemote = _ls_remote,
    warn: Callable[[str], None] = lambda message: None,
) -> str:
    """Return the version for an action fetched at ``ref``.

    Args:
        ref: The ref the action was fetched at (``github.action_ref``):
            a tag, a branch, a commit SHA, or empty.
        repository_url: Where the action's tags can be listed; several
            are tried in order, since an Enterprise runner may fetch the
            action from github.com rather than its own server.
        sha: The commit, when known separately from ``ref`` (a reusable
            workflow passes its own commit).  Preferred over ``ref``.
        ls_remote: Lists a repository's tags; replaced in tests.
        warn: Receives a message when resolution falls back.
    """
    urls = (
        [repository_url]
        if isinstance(repository_url, str)
        else list(dict.fromkeys(repository_url))
    )
    ref = ref.strip()
    sha = sha.strip()
    # A separate commit is authoritative; a tag-shaped ref is trusted
    # only when there is none, since the two can disagree.
    if not sha and _release_key(ref) is not None:
        return ref.removeprefix("v")

    commit = sha.lower() or (ref.lower() if _FULL_SHA.match(ref.lower()) else "")
    if not _FULL_SHA.match(commit):
        warn(f"cannot resolve a commit from ref {ref!r}; using a placeholder version")
        return "0.0.0+unknown"

    failures = []
    answered = False
    for url in urls:
        try:
            tags = tags_for_commit(ls_remote(url), commit)
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"{url}: {exc}")
            continue
        answered = True
        if tags:
            return tags[-1].removeprefix("v")
    if not answered:
        # Only worth a warning when no source answered: a repository the
        # workflow's own server does not host is expected on Enterprise,
        # and an untagged commit is no fault.
        warn("could not list tags of " + "; ".join(failures))
    return f"0.0.0+g{commit[:7]}"


def command_data(text: str) -> str:
    """Escape ``text`` for the message of a workflow command.

    A warning can quote the repository URL, which comes from an action
    input; a line break in it would end the command and start a forged
    one.
    """
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve the dependamerge version an action run executes."
    )
    parser.add_argument("--ref", default="", help="github.action_ref")
    parser.add_argument("--sha", default="", help="commit SHA, when known")
    parser.add_argument(
        "--repository-url",
        action="append",
        required=True,
        help="URL to list the tags of; repeat to try several in order",
    )
    args = parser.parse_args(argv)

    def warn(message: str) -> None:
        print(
            f"::warning title=dependamerge version::{command_data(message)}",
            file=sys.stderr,
        )

    print(resolve(args.ref, args.repository_url, args.sha, warn=warn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
