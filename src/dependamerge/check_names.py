# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Presenting check names in a single line of terminal output.

A matrix job reports one check run per cell, and GitHub names each after
the parameters that produced it::

    Build/Test (org/test-docker-monorepo, v0.0.1, make images TAG=verify,
    TAG=verify tes... / Grype Audit SBOM

Eight cells of that is 827 characters, comma-joined into a paragraph the
terminal wraps mid-name --- and because the names themselves contain
commas, the reader cannot even tell where one ends and the next begins.
The information an operator wants from that paragraph is "Grype Audit
SBOM failed, seven times".

So names are grouped by what they *are* rather than listed by which cell
produced them.  Which cell is one click away on the pull request; the
report's job is to say what kind of thing is failing, in a line that can
be read.

Pure and free of I/O, so the wording can be tested directly.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["summarise_check_names"]

#: How many distinct checks to name before the rest become a count.
#: Four fits a terminal line alongside the label that introduces them.
_MAX_NAMES = 4

#: Longest a single name may be before it is elided.  GitHub truncates
#: its own matrix parameters at roughly this width, so a name still over
#: it after grouping is one no line will hold anyway.
_MAX_NAME_LENGTH = 48


def _leaf(name: str) -> str:
    """The part of a check name that says what the check *is*.

    GitHub joins a job name to the check it reports with ``" / "``, and
    for a matrix job everything left of that separator is the cell's
    parameters --- the part that differs between siblings, and the part
    GitHub itself truncates when it is long.  The tail is the name an
    operator recognises.

    A name with no separator is already a leaf.  A name whose tail is
    empty falls back to the head rather than to the raw string, so a
    stray trailing separator does not become part of the reported name.
    """
    head, separator, tail = name.rpartition(" / ")
    tail = tail.strip()
    if tail:
        return tail
    return (head if separator else name).strip()


def _elide(name: str) -> str:
    """Shorten *name* to something a line can hold."""
    if len(name) <= _MAX_NAME_LENGTH:
        return name
    return name[: _MAX_NAME_LENGTH - 1].rstrip() + "…"


def summarise_check_names(names: Sequence[str]) -> str:
    """Render *names* as one readable, comma-separated line.

    Names sharing a leaf are collapsed to that leaf with a count, so the
    seven matrix cells of one check read as ``Grype Audit SBOM (×7)``
    rather than as seven wrapped lines of parameters.  First-seen order
    is preserved, because the order the API reported them in is the only
    ordering available and re-sorting would make consecutive runs of the
    same repository disagree for no reason.

    Beyond :data:`_MAX_NAMES` distinct checks the remainder becomes a
    count.  A report naming twenty checks is not more useful than one
    naming four and saying there are sixteen others; both send the
    reader to the pull request, and only one of them is readable.
    """
    counts: dict[str, int] = {}
    for name in names:
        leaf = _leaf(name)
        if not leaf:
            continue
        counts[leaf] = counts.get(leaf, 0) + 1

    shown = list(counts.items())[:_MAX_NAMES]
    rendered = [
        f"{_elide(leaf)} (×{count})" if count > 1 else _elide(leaf)
        for leaf, count in shown
    ]
    remaining = len(counts) - len(shown)
    if remaining > 0:
        rendered.append(f"+{remaining} more")
    return ", ".join(rendered)
