# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
What submitting one Gerrit change would submit with it.

Submitting a change can submit others: its unmerged ancestors, and with
``change.submitWholeTopic`` every change sharing its topic, in any
project.  A query URL's batch must not reach past the changes previewed,
so before a change in it is voted on, Gerrit's ``submitted_together``
set is checked against the batch: every member in it, at the revision
previewed, and none hidden from the caller.  A change URL's batch, which
has no such check, asks for the set by number instead, so a change whose
submission would take a held-back major version bump along is held back
too; once the major version gate has acted on it, a
:class:`SubmissionGuard` checks each change's set again before its vote
and before its submit, refusing one with members hidden from the caller
as the query check does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from dependamerge.gerrit.client import GerritRestClient

_ENDPOINT = (
    "/changes/{number}/submitted_together?o=NON_VISIBLE_CHANGES&o=CURRENT_REVISION"
)


@dataclass(frozen=True)
class SubmissionGuard:
    """What a change URL's batch may not submit as a side effect.

    Unlike a query batch's scope, the set may reach past the batch; it
    may not take a held-back change along, nor a confirmed change at a
    revision other than the one confirmed.

    Attributes:
        confirmed: Each change number the batch submits, mapped to the
            revision confirmed for it.
        held: The change numbers left out of the batch: major version
            bumps held back, or skipped as blocked.
    """

    confirmed: Mapping[int, str]
    held: frozenset[int]


#: A scope (query batch) or a guard (change batch) a submission must meet.
GroupCheck = Mapping[int, str] | SubmissionGuard


def _hidden_problems(change_number: int, data: Mapping[str, object]) -> list[str]:
    """Why Gerrit's answer leaves changes hidden from the caller, if it does.

    A change the caller cannot see, or a count Gerrit left out, could be
    one the batch must not submit; neither can be ruled out by number.
    """
    hidden = data.get("non_visible_changes")
    if not isinstance(hidden, int) or isinstance(hidden, bool):
        return [
            f"#{change_number}: Gerrit did not report whether it submits "
            "changes you cannot see"
        ]
    if hidden:
        return [f"#{change_number} would also submit {hidden} changes you cannot see"]
    return []


def submitted_together_problems(
    client: GerritRestClient,
    change_number: int,
    scope: Mapping[int, str],
) -> list[str]:
    """Why submitting *change_number* could act outside *scope*, if it could.

    Args:
        client: The REST client to ask Gerrit with.
        change_number: The change about to be voted on and submitted.
        scope: Each change number the batch submits, mapped to the
            revision previewed for it.

    Returns:
        One message per problem; empty when Gerrit would submit only
        changes in *scope*, each at its previewed revision.

    Raises:
        GerritRestError: If Gerrit cannot be asked.
    """
    data = client.get(_ENDPOINT.format(number=change_number))
    if not isinstance(data, dict) or not isinstance(data.get("changes"), list):
        return [
            f"#{change_number}: Gerrit did not report which changes it submits "
            "with, so the batch's scope cannot be confirmed"
        ]

    problems = _hidden_problems(change_number, data)
    for item in data["changes"]:
        number = item.get("_number") if isinstance(item, dict) else None
        if not isinstance(number, int) or isinstance(number, bool):
            problems.append(
                f"#{change_number} would also submit a change Gerrit did not number"
            )
            continue
        if number not in scope:
            project = item.get("project", "unknown project")
            problems.append(
                f"#{change_number} would also submit #{number} ({project}), "
                "which is not among the changes to submit"
            )
        elif item.get("current_revision") != scope[number]:
            problems.append(
                f"#{change_number} would submit #{number} at a revision other "
                "than the one previewed"
            )
    return problems


def submitted_together_numbers(
    client: GerritRestClient, change_number: int
) -> set[int] | None:
    """The changes Gerrit would submit with *change_number*, by number.

    Args:
        client: The REST client to ask Gerrit with.
        change_number: The change whose submission is in question.

    Returns:
        The number of every visible change in its ``submitted_together``
        set, *change_number* left out; ``None`` when Gerrit's answer
        does not say which changes they are.

    Raises:
        GerritRestError: If Gerrit cannot be asked.
    """
    data = client.get(_ENDPOINT.format(number=change_number))
    if not isinstance(data, dict) or not isinstance(data.get("changes"), list):
        return None
    numbers: set[int] = set()
    for item in data["changes"]:
        number = item.get("_number") if isinstance(item, dict) else None
        if not isinstance(number, int) or isinstance(number, bool):
            return None
        numbers.add(number)
    numbers.discard(change_number)
    return numbers


def guard_problems(
    client: GerritRestClient,
    change_number: int,
    guard: SubmissionGuard,
) -> list[str]:
    """Why submitting *change_number* could breach *guard*, if it could.

    Args:
        client: The REST client to ask Gerrit with.
        change_number: The change about to be voted on or submitted.
        guard: The held-back changes and confirmed revisions to keep to.

    Returns:
        One message per problem; empty when Gerrit would submit no
        held-back change, every confirmed one at its revision, and no
        change hidden from the caller, which could be a held-back one.

    Raises:
        GerritRestError: If Gerrit cannot be asked.
    """
    data = client.get(_ENDPOINT.format(number=change_number))
    if not isinstance(data, dict) or not isinstance(data.get("changes"), list):
        return [
            f"#{change_number}: Gerrit did not report which changes it submits "
            "with, so it could submit a held-back change"
        ]

    problems = _hidden_problems(change_number, data)
    for item in data["changes"]:
        number = item.get("_number") if isinstance(item, dict) else None
        if not isinstance(number, int) or isinstance(number, bool):
            problems.append(
                f"#{change_number} would also submit a change Gerrit did not number"
            )
        elif number in guard.held:
            problems.append(f"#{change_number} would also submit held-back #{number}")
        elif (
            number in guard.confirmed
            and item.get("current_revision") != guard.confirmed[number]
        ):
            problems.append(
                f"#{change_number} would submit #{number} at a revision other "
                "than the one confirmed"
            )
    return problems


def group_problems(
    client: GerritRestClient, change_number: int, check: GroupCheck
) -> list[str]:
    """Why submitting *change_number* could fail *check*, a scope or a guard."""
    if isinstance(check, SubmissionGuard):
        return guard_problems(client, change_number, check)
    return submitted_together_problems(client, change_number, check)
