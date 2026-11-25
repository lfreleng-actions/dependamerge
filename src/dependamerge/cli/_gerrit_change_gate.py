# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The major version bump gate, for a change URL's batch.

A query batch skips its blocked changes and scopes its submission
before the gate in ``_gerrit_major_gate`` runs; a change URL's batch
has neither, so this reads what Gerrit reports as blocking each change,
skips a blocked gated one, holds back what would submit a held-back
change along, and builds the :class:`SubmissionGuard` its submission
carries.
"""

from __future__ import annotations

from dataclasses import replace

from rich.console import Console

from ..gerrit import GerritChangeInfo, GerritService, SubmissionGuard
from ._gerrit_blocked import _blocked_reason
from ._gerrit_major_gate import (
    _Entry,
    _gated_bump,
    _GateOptions,
    _GateOutcome,
    _Held,
    _settle_major_bumps,
    _without,
)


def _hold_back_dependents(
    service: GerritService,
    outcome: _GateOutcome,
    console: Console,
) -> _GateOutcome:
    """Hold back each kept change whose submission takes a held one along.

    Gerrit submits a change's open ancestors with it and, under
    ``change.submitWholeTopic``, every change sharing its topic.  A
    change URL's batch has no scope guard, so Gerrit is asked what each
    kept change would be submitted with; one that would take a held-back
    change, or for which Gerrit does not say, is held back too, until no
    kept change would.

    A change Gerrit reports as blocked stays on the submit manager's
    path instead: held back, it would leave the exit code alone and hide
    the block.  The :class:`SubmissionGuard` the batch then carries
    refuses it before its vote, so it fails the run without submitting
    a held-back change.

    Raises:
        GerritRestError: If Gerrit cannot be asked.
    """
    if not outcome.held:
        return outcome
    held_numbers = {change.number for change, _ in outcome.held}
    together: dict[int, set[int] | None] = {}
    added: list[_Held] = []
    changed = True
    while changed:
        changed = False
        for change, _ in outcome.kept:
            if change.number in held_numbers or _blocked_reason(change):
                continue
            if change.number not in together:
                together[change.number] = service.submitted_together_numbers(
                    change.number
                )
            numbers = together[change.number]
            if numbers is None:
                reason = (
                    "Gerrit did not report which changes it submits with this "
                    "one, so it could submit a held-back change"
                )
            else:
                taken = sorted(numbers & held_numbers)
                if not taken:
                    continue
                listed = ", ".join(f"#{number}" for number in taken)
                reason = f"submitting it would also submit held-back {listed}"
            held_numbers.add(change.number)
            added.append((change, reason))
            changed = True
    if not added:
        return outcome
    console.print(
        f"\n⏸️ Held back with them, as Gerrit submits them together: {len(added)}"
    )
    for change, reason in added:
        console.print(f"   {change.project} #{change.number}: {reason}", markup=False)
    return _GateOutcome(
        _without(outcome.kept, held_numbers),
        [*outcome.held, *added],
        outcome.bind_revisions,
    )


def _holds_source_back(run: _GateOptions, source: GerritChangeInfo) -> bool:
    """Whether an unattended run holds a gated source change back."""
    return run.no_confirm and _gated_bump(run, source) is not None


def _with_blocker_details(
    service: GerritService, run: _GateOptions, entries: list[_Entry]
) -> list[_Entry]:
    """``entries`` with what Gerrit reports as blocking each, once any is gated.

    A change URL's changes are read without submit requirements, and
    its similar changes without detailed labels, so a change blocked by
    an unsatisfied requirement would look submittable.  Only the fields
    :func:`_blocked_reason` reads are taken from the fresh read; a
    change it does not return keeps what it had.

    Raises:
        GerritRestError: If Gerrit cannot be asked.
        GerritServiceError: If Gerrit's answer cannot be parsed.
    """
    if all(_gated_bump(run, change) is None for change, _ in entries):
        return entries
    latest = {
        change.number: change
        for change in service.get_open_changes_by_number(
            [change.number for change, _ in entries]
        )
    }
    refreshed: list[_Entry] = []
    for change, comparison in entries:
        fresh = latest.get(change.number)
        if fresh is not None:
            change = change.model_copy(
                update={
                    "work_in_progress": fresh.work_in_progress,
                    "labels": fresh.labels,
                    "submit_requirements_met": fresh.submit_requirements_met,
                    "submit_requirement_status": fresh.submit_requirement_status,
                    "mergeable": change.mergeable
                    if fresh.mergeable is None
                    else fresh.mergeable,
                }
            )
        refreshed.append((change, comparison))
    return refreshed


def _gate_change_batch(
    service: GerritService,
    run: _GateOptions,
    host: str,
    entries: list[_Entry],
    console: Console,
    source: GerritChangeInfo,
) -> _GateOutcome:
    """Gate a change URL's batch, then hold back what would take a held change.

    Once any change is gated, what Gerrit reports as blocking each change
    is read first (:func:`_with_blocker_details`).  A gated change Gerrit
    reports as blocked is then skipped as blocked before the gate runs,
    as in a query batch, so it gets no vote and the run still reports the
    block in its exit code; a blocked change the gate would not take goes
    on to the submit manager as before.
    """
    entries = _with_blocker_details(service, run, entries)
    skipped: list[_Held] = []
    for change, _ in entries:
        reason = _blocked_reason(change)
        if reason and _gated_bump(run, change) is not None:
            skipped.append((change, f"Blocked: {reason}"))
    skip = {change.number for change, _ in skipped}
    if skipped:
        console.print(f"\n⏭️ Blocked major version bumps skipped: {len(skipped)}")
        for change, reason in skipped:
            console.print(
                f"   {change.project} #{change.number}: {reason}", markup=False
            )
    gate = _settle_major_bumps(
        run,
        host,
        _without(entries, skip),
        console,
        source=None if source.number in skip else source,
    )
    gate = _hold_back_dependents(service, gate, console)
    return replace(gate, skipped=skipped)


def _submission_guard(outcome: _GateOutcome) -> SubmissionGuard | None:
    """What a change URL's batch must keep to once the gate has acted on it.

    :func:`_hold_back_dependents` reads Gerrit's groups before the
    continue prompt; the guard has the submit manager read them again
    before each vote and each submit, so a held-back change that joined
    a kept change's topic since, or a confirmed change's new patchset,
    is not submitted along with another change.  A gated change skipped
    as blocked is kept out the same way: its blocker can clear before a
    sibling sharing its topic submits.  ``None`` when the gate neither
    held a change back, skipped one, nor had one confirmed.
    """
    if not outcome.held and not outcome.skipped and not outcome.bind_revisions:
        return None
    return SubmissionGuard(
        confirmed={
            change.number: change.current_revision for change, _ in outcome.kept
        },
        held=frozenset(
            change.number for change, _ in [*outcome.held, *outcome.skipped]
        ),
    )
