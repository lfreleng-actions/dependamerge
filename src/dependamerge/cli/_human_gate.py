# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The gate in front of human-authored pull requests.

Dependamerge acts on automation by default.  Bringing a person's pull
requests into scope takes two deliberate steps, the same in every
command and mode:

1. ``--include-human-prs`` says the run may act on them;
2. ``--override`` confirms exactly which ones, with a value derived from
   them that a run without it prints.

A value printed by one run cannot authorise a different set of pull
requests, and ``--override`` alone authorises nothing.  Refusals exit
non-zero so a scripted caller can tell them apart from "nothing to do":
``CONFIGURATION_ERROR`` for an override without the flag,
``VALIDATION_ERROR`` for a human source without the flag or a wrong
value, and ``OVERRIDE_REQUIRED`` when the value is all that is missing.
"""

from __future__ import annotations

import hashlib
from typing import Any

import typer

from ..error_codes import ExitCode, exit_for_configuration_error, exit_with_error
from ..models import PullRequestInfo
from ._app import console
from ._context import _MergeContext
from ._merge_report import _conclude_run

__all__ = [
    "INCLUDE_HUMAN_PRS_OPTION",
    "OVERRIDE_OPTION",
    "_confirm_human_override",
    "_confirm_human_prs_in_scope",
    "_given_override",
    "_human_set_override",
    "_refuse_human_source",
    "_require_override_pairing",
]

#: The ``--include-human-prs`` option, shared by every command with the
#: gate so they cannot describe it differently.
INCLUDE_HUMAN_PRS_OPTION = typer.Option(
    False,
    "--include-human-prs",
    help=(
        "Act on human-authored PRs too. Needs a matching --override, "
        "which a run without it prints before stopping with exit code 9"
    ),
)

#: The ``--override`` option.
OVERRIDE_OPTION = typer.Option(
    None,
    "--override",
    help=(
        "Confirm the human-authored PRs --include-human-prs brings into "
        "scope, with the value a run without it prints"
    ),
)


def _given_override(value: Any) -> str | None:
    """The ``--override`` value, or ``None`` when none was given.

    A direct Python call to a command (as some tests make) passes Typer's
    unresolved ``OptionInfo`` default, which must read as "not given".
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _require_override_pairing(override: Any, include_human_prs: Any) -> None:
    """Refuse ``--override`` without ``--include-human-prs``.

    An override confirms the human-authored pull requests the flag brings
    into scope; on its own there is nothing for it to confirm.
    """
    if _given_override(override) and include_human_prs is not True:
        exit_for_configuration_error(
            message="❌ --override needs --include-human-prs",
            details=(
                "--override confirms the human-authored PRs that "
                "--include-human-prs brings into scope; it does not replace "
                "the flag. Run again with both."
            ),
        )


def _refuse_human_source(author: str) -> None:
    """Stop a run whose source PR a person wrote, without the flag."""
    exit_with_error(
        ExitCode.VALIDATION_ERROR,
        message=(
            f"❌ Source PR is human-authored (by {author}); dependamerge acts "
            "on automation PRs by default"
        ),
        details=(
            "To act on human-authored PRs, run again with --include-human-prs; "
            "the run then prints the --override value that confirms them."
        ),
    )


def _human_set_override(scope: str, prs: list[PullRequestInfo]) -> str:
    """The ``--override`` value for a set of human-authored PRs.

    Derived from the owner or repository and each PR's number and head
    commit, so it confirms exactly the PRs listed, as they stand: a new
    push, or another human PR appearing, changes it.
    """
    members = sorted(
        f"{pr.repository_full_name.lower()}#{pr.number}@{pr.head_sha}" for pr in prs
    )
    combined = f"humans:{scope.lower()}:{','.join(members)}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _confirm_human_override(
    provided: str | None,
    expected: str,
    *,
    dry_run: bool,
    subject: str,
    derivation: str,
) -> None:
    """Require the override that confirms acting on human-authored PRs.

    Args:
        provided: The ``--override`` value given, if any.
        expected: The value this run's human-authored PRs derive.
        dry_run: A preview acts on nothing, so it reports the value a real
            run needs rather than requiring it.
        subject: What the override confirms, for messages.
        derivation: What the value derives from, for messages.

    Raises:
        SystemExit: ``VALIDATION_ERROR`` for a wrong value, and
            ``OVERRIDE_REQUIRED`` for a missing one outside a dry run.
    """
    if provided is not None:
        if provided.lower() != expected:
            exit_with_error(
                ExitCode.VALIDATION_ERROR,
                message=f"❌ The --override value does not confirm {subject}",
                details=(
                    f"Expected: --override {expected}\n"
                    f"The value derives from {derivation}, so it changes "
                    "when they do."
                ),
            )
        console.print(f"👤 --override confirms {subject}.")
        return

    if dry_run:
        console.print(
            f"🧪 Dry run: a real run needs --override {expected} to act on {subject}."
        )
        return

    exit_with_error(
        ExitCode.OVERRIDE_REQUIRED,
        message=f"❌ --include-human-prs needs --override to act on {subject}",
        details=(
            f"Run again with: --include-human-prs --override {expected}\n"
            f"The value derives from {derivation}, so it changes when they do."
        ),
    )


def _confirm_human_prs_in_scope(
    ctx: _MergeContext, scope: str, human_prs: list[PullRequestInfo]
) -> None:
    """Confirm the human-authored PRs an owner or repository run found.

    The override derives from ``scope`` and exactly these PRs.  A run the
    gate stops is still a finished run, so its record is published first,
    as one with nothing to merge would be.
    """
    if not human_prs:
        return
    noun = "PR" if len(human_prs) == 1 else "PRs"
    try:
        _confirm_human_override(
            _given_override(ctx.override),
            _human_set_override(scope, human_prs),
            dry_run=ctx.dry_run,
            subject=f"the {len(human_prs)} human-authored {noun} listed above",
            derivation=f"{scope} and each of those PRs' number and head commit",
        )
    except SystemExit:
        _conclude_run(ctx, [], preview=True)
        raise
