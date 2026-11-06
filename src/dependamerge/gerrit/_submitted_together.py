# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
What submitting one Gerrit change would submit with it.

Submitting a change can submit others: its unmerged ancestors, and with
``change.submitWholeTopic`` every change sharing its topic, in any
project.  A query URL's batch must not reach past the changes previewed,
so before a change in it is voted on, Gerrit's ``submitted_together``
set is checked against the batch: every member in it, at the revision
previewed, and none hidden from the caller.
"""

from __future__ import annotations

from collections.abc import Mapping

from dependamerge.gerrit.client import GerritRestClient

_ENDPOINT = (
    "/changes/{number}/submitted_together?o=NON_VISIBLE_CHANGES&o=CURRENT_REVISION"
)


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

    problems: list[str] = []
    hidden = data.get("non_visible_changes")
    if not isinstance(hidden, int) or isinstance(hidden, bool):
        problems.append(
            f"#{change_number}: Gerrit did not report whether it submits "
            "changes you cannot see"
        )
    elif hidden:
        problems.append(
            f"#{change_number} would also submit {hidden} changes you cannot see"
        )

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
