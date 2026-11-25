# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The gate in front of major version bumps, for Gerrit changes.

The Gerrit counterpart of ``_major_gate``.  A change is gated when
:mod:`dependamerge.version_bumps` reads its subject and commit message as
``major``, or as ``zero-minor`` under ``--gate-zero-minor``:

- an unattended run (``--no-confirm``) holds it back: it gets no vote
  and is not submitted;
- an interactive run lists the gated changes after its preview and asks
  for a value derived from exactly those changes and their revisions;
  anything else holds them back and the rest are submitted;
- a change URL whose source change is gated warns instead of asking,
  since the operator named the bump;
- ``--allow-major`` turns the gate off, in every mode.

A held-back change is neither skipped nor failed, so it leaves a run's
exit code alone.  The changes a gate is applied to are those that would
otherwise be submitted: a change Gerrit already reports as blocked is
skipped as blocked first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import typer
from rich.console import Console

from ..gerrit import (
    GerritChangeInfo,
    GerritComparisonResult,
    GerritSubmitResult,
)
from ..version_bumps import MAJOR, ZERO_MINOR, Classification, Update, classify
from ._major_gate import ALLOW_MAJOR_HINT
from ._sha import _generate_gerrit_majors_sha

#: A change to submit, paired as the submit manager expects.
_Entry = tuple[GerritChangeInfo, GerritComparisonResult | None]

#: A change left out of submission, with its reason.
_Held = tuple[GerritChangeInfo, str]


class _GateOptions(Protocol):
    """The run options the gate reads."""

    @property
    def no_confirm(self) -> bool: ...

    @property
    def dry_run(self) -> bool: ...

    @property
    def allow_major(self) -> bool: ...

    @property
    def gate_zero_minor(self) -> bool: ...


@dataclass(frozen=True)
class _GateOutcome:
    """What the gate left to submit.

    Attributes:
        kept: The changes still to submit, in the order given.
        held: The gated changes held back, each with its reason.
        bind_revisions: Whether gated changes were confirmed, so each
            must be submitted only at the revision shown.
        skipped: The gated changes Gerrit reports as blocked, each with
            its reason; skipped as blocked rather than held back.
    """

    kept: list[_Entry]
    held: list[_Held]
    bind_revisions: bool = False
    skipped: list[_Held] = field(default_factory=list)


def _gated_levels(run: _GateOptions) -> frozenset[str]:
    """The ``version_bumps`` levels this run gates; none under ``--allow-major``."""
    if run.allow_major:
        return frozenset()
    if run.gate_zero_minor:
        return frozenset({MAJOR, ZERO_MINOR})
    return frozenset({MAJOR})


def _gated_bump(run: _GateOptions, change: GerritChangeInfo) -> Classification | None:
    """The classification of ``change`` when the gate applies to it, else ``None``."""
    levels = _gated_levels(run)
    if not levels:
        return None
    bump = classify(change.subject, change.message)
    return bump if bump.level in levels else None


def _gated_updates(run: _GateOptions, bump: Classification) -> tuple[Update, ...]:
    """The updates at a gated level; failing any, every update."""
    levels = _gated_levels(run)
    return tuple(u for u in bump.updates if u.level in levels) or bump.updates


def _describe_update(update: Update) -> str:
    name = f"{update.name} " if update.name else ""
    return f"{name}{update.old} → {update.new}"


def _gate_reason(run: _GateOptions, bump: Classification) -> str:
    """How a gated change is recorded when held back.

    Names every gated update, so a grouped change says which of its
    dependencies put it behind the gate.
    """
    label = "major version bump" if bump.level == MAJOR else "0.x minor version bump"
    described = ", ".join(_describe_update(u) for u in _gated_updates(run, bump))
    return f"{label}: {described}" if described else label


def _label(change: GerritChangeInfo) -> str:
    return change.url or f"{change.project} #{change.number}"


def _without(entries: list[_Entry], numbers: set[int]) -> list[_Entry]:
    return [entry for entry in entries if entry[0].number not in numbers]


def _print_gated(
    run: _GateOptions,
    gated: list[tuple[GerritChangeInfo, Classification]],
    console: Console,
) -> None:
    console.print(
        "\n⚠️ Major version bumps among the changes to submit: "
        f"{len(gated)} (submitted only when confirmed)",
        markup=False,
    )
    for change, bump in gated:
        console.print(f"   • {_label(change)}", markup=False)
        for update in _gated_updates(run, bump):
            console.print(f"     {_describe_update(update)}", markup=False)


def _ask_for_value(value: str, console: Console) -> str | None:
    """The value the operator entered, or ``None`` for EOF or an interrupt."""
    console.print()
    console.print(f"To submit them too enter: {value}")
    try:
        entered: str = typer.prompt(
            "Enter the string above to include them "
            "(or press Enter to submit the rest without them)",
            default="",
            show_default=False,
        )
    except (KeyboardInterrupt, EOFError, typer.Abort):
        return None
    return entered.strip()


def _hold_back_unattended(
    run: _GateOptions,
    entries: list[_Entry],
    gated: list[tuple[GerritChangeInfo, Classification]],
    console: Console,
) -> _GateOutcome:
    held = [(change, _gate_reason(run, bump)) for change, bump in gated]
    verb = "a real run holds back" if run.dry_run else "held back"
    console.print(f"\n⏸️ Major version bumps {verb}: {len(held)}")
    for change, reason in held:
        # markup=False: names and versions come from commit text.
        console.print(f"   {change.project} #{change.number}: {reason}", markup=False)
    if run.dry_run:
        console.print(f"   💡 {ALLOW_MAJOR_HINT}")
    return _GateOutcome(_without(entries, {c.number for c, _ in held}), held)


def _settle_major_bumps(
    run: _GateOptions,
    host: str,
    entries: list[_Entry],
    console: Console,
    *,
    source: GerritChangeInfo | None = None,
) -> _GateOutcome:
    """Decide which gated changes among ``entries`` are submitted.

    Args:
        run: The run's options.
        host: The Gerrit host, from which the confirming value derives.
        entries: The changes that would otherwise be submitted.
        console: Rich console for output.
        source: The change a change URL names.  When it is itself gated
            and the run is interactive, a warning replaces the prompt.

    Returns:
        The changes left to submit and those held back.  A dry run holds
        back what an unattended run would, and never prompts.
    """
    gated: list[tuple[GerritChangeInfo, Classification]] = []
    for change, _ in entries:
        bump = _gated_bump(run, change)
        if bump is not None:
            gated.append((change, bump))
    if not gated:
        return _GateOutcome(entries, [])

    if run.no_confirm:
        return _hold_back_unattended(run, entries, gated, console)

    _print_gated(run, gated, console)

    source_bump = _gated_bump(run, source) if source is not None else None
    if source_bump is not None:
        outcome = (
            "a real run submits it and the similar changes once confirmed"
            if run.dry_run
            else "confirming below submits it and the similar changes with it"
        )
        console.print(
            f"\n⚠️ The source change is a {_gate_reason(run, source_bump)}; {outcome}.",
            markup=False,
        )
        return _GateOutcome(entries, [], bind_revisions=True)

    if run.dry_run:
        console.print(
            "🧪 Dry run: a real run asks for a value to submit these, and "
            "without it submits the rest."
        )
        return _GateOutcome(entries, [])

    value = _generate_gerrit_majors_sha(host, [change for change, _ in gated])
    entered = _ask_for_value(value, console)
    if entered == value:
        console.print("✅ Major version bumps included.")
        return _GateOutcome(entries, [], bind_revisions=True)
    if entered is None:
        console.print("\n⏭️ No value entered: major version bumps held back.")
    elif entered == "":
        console.print("⏭️ Major version bumps held back.")
    else:
        console.print("❌ The value does not match: major version bumps held back.")

    held = [(change, _gate_reason(run, bump)) for change, bump in gated]
    kept = _without(entries, {change.number for change, _ in held})
    console.print(f"Changes to submit without them: {len(kept)}/{len(entries)}")
    return _GateOutcome(kept, held)


def _print_held_back(held: list[_Held], console: Console) -> None:
    """Recap the held-back changes, each with its reason, then the hint."""
    if not held:
        return
    console.print("\n⏸️ Held back changes:")
    for change, reason in held:
        console.print(f"   • {_label(change)}\n     {reason}", markup=False)
    console.print(f"   💡 {ALLOW_MAJOR_HINT}")


def _print_submitted_major_bumps(
    results: list[GerritSubmitResult],
    all_changes: list[_Entry],
    console: Console,
) -> None:
    """List the submitted changes that bump a dependency across a major version.

    Such a change is submitted under ``--allow-major`` or once its value
    is entered, so this is the reader's prompt to check it did not break
    the repository.  Prints nothing when no submitted change counts.
    """
    by_key = {(change.project, change.number): change for change, _ in all_changes}
    counted: list[tuple[GerritChangeInfo, Classification]] = []
    for result in results:
        change = by_key.get((result.project, result.change_number))
        if not result.submitted or change is None:
            continue
        bump = classify(change.subject, change.message)
        if bump.level == MAJOR:
            counted.append((change, bump))
    if not counted:
        return
    console.print(
        f"\n⚠️ Major version bumps submitted: {len(counted)} "
        "(check they did not break the repository)",
        markup=False,
    )
    for change, bump in counted:
        console.print(f"   • {_label(change)}", markup=False)
        for update in bump.major_updates():
            console.print(f"     {_describe_update(update)}", markup=False)
