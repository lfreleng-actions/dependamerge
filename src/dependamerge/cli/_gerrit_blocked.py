# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Whether Gerrit already reports a change as unable to submit.

Shared by a query batch, which skips such a change, and the major
version bump gate, which never takes one.
"""

from ..gerrit import GerritChangeInfo

#: Submit requirement statuses that keep a change from being submitted.
_FAILING_REQUIREMENT = {"UNSATISFIED": "unsatisfied", "ERROR": "in error"}

#: The requirement this run's own +2 is about to satisfy, and the one
#: status of it that +2 can clear.  ``ERROR`` means Gerrit could not
#: evaluate the requirement, which no vote fixes.
_REVIEW_LABEL = "Code-Review"
_AWAITING_REVIEW = "UNSATISFIED"


def _blocked_reason(change: GerritChangeInfo) -> str | None:
    """Why Gerrit already reports ``change`` as unable to submit, if it does.

    Work in progress; a vote at a label's minimum (Gerrit's ``=MIN``,
    such as ``Verified -1``) or a legacy ``rejected`` summary; a submit
    requirement that is unsatisfied or in error; or ``mergeable: false``
    where Gerrit reports it.  Code-Review's own requirement blocks only
    when in error: when unsatisfied, the +2 this run gives is what it
    lacks.
    """
    reasons: list[str] = []
    if change.work_in_progress:
        reasons.append("work in progress")
    vetoed: set[str] = set()
    for label in change.labels:
        if label.minimum_vote is not None:
            reasons.append(f"{label.name} {label.minimum_vote:+d}")
            vetoed.add(label.name)
        elif label.rejected:
            vote = f"{label.value:+d}" if label.value is not None else "rejected"
            reasons.append(f"{label.name} {vote}")
            vetoed.add(label.name)
    for name, status in change.submit_requirement_status.items():
        failing = _FAILING_REQUIREMENT.get(status)
        awaiting_review = name == _REVIEW_LABEL and status == _AWAITING_REVIEW
        if failing and not awaiting_review and name not in vetoed:
            reasons.append(f"{name} requirement {failing}")
    if change.mergeable is False:
        reasons.append("merge conflict")
    return ", ".join(reasons) or None
