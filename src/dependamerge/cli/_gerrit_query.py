# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Acting on exactly the open changes a Gerrit query URL matches.

A change URL (``/c/<project>/+/<number>``) anchors on one change and
finds the changes similar to it.  A query URL (``/q/...``) has no anchor:
its batch is every open change the query returns, and nothing else.  No
similarity scoring or owner matching runs, nothing is rebased, and a
change Gerrit already reports as blocked stays in the batch but is
skipped rather than voted on.
"""

import math
from dataclasses import dataclass

import typer
from rich.console import Console

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..error_codes import ExitCode, exit_with_error
from ..gerrit import (
    GerritChangeComparator,
    GerritChangeInfo,
    GerritComparisonResult,
    GerritService,
)
from ..url_parser import ParsedGerritTopicUrl
from ._human_gate import _confirm_human_override, _given_override
from ._sha import _generate_gerrit_query_sha

#: The ``--threshold`` default; any other value was asked for, and
#: earns a note that a query URL does not use it.
_DEFAULT_THRESHOLD = 0.8

#: A change to submit, paired as the submit manager expects.
_Entry = tuple[GerritChangeInfo, GerritComparisonResult | None]


@dataclass(frozen=True)
class _GerritRunOptions:
    """The merge command's flags that steer a Gerrit run.

    Attributes:
        no_confirm: Skip the confirmation prompt.
        dry_run: Preview only; never review or submit.
        override: The ``--override`` value given, if any.
        include_human_prs: Bring non-automation changes into scope.
        show_progress: Drive a live progress tracker during submission.
        topic: ``--topic``; a change URL only.
    """

    no_confirm: bool
    dry_run: bool
    override: str | None
    include_human_prs: bool
    show_progress: bool
    topic: str | None = None


def _check_query_options(
    topic: str | None,
    similarity_threshold: float,
    console: Console,
) -> None:
    """Refuse or note the options a query URL does not take.

    Raises:
        SystemExit: ``VALIDATION_ERROR`` when ``--topic`` is given, since
            it would replace the query the URL names.
    """
    if topic:
        exit_with_error(
            ExitCode.VALIDATION_ERROR,
            message="❌ --topic cannot be combined with a Gerrit query URL",
            details=(
                "A query URL acts on exactly the changes its query matches; "
                "put the topic in the URL's query instead."
            ),
        )
    if isinstance(similarity_threshold, int | float) and not math.isclose(
        similarity_threshold, _DEFAULT_THRESHOLD
    ):
        console.print(
            "ℹ️ --threshold does not apply to a Gerrit query URL; "
            "the batch is exactly the query's result set."
        )


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


def _print_query_batch(
    changes: list[GerritChangeInfo],
    blocked: dict[int, str],
    console: Console,
) -> tuple[int, int]:
    """List every matched change, marking blocked ones and permissions.

    Gerrit offers the submit action only on a change that is already
    submittable, so a change awaiting its +2 says nothing about whether
    the caller may submit it.

    Returns:
        How many changes carry a permission warning, and how many of the
        rest are not yet submittable, so their submit permission is
        unverified.
    """
    console.print(f"\nQuery matched {len(changes)} open changes:")
    lacking = 0
    unverified = 0
    for change in changes:
        console.print(
            f"  • {change.project} #{change.number}: {change.subject} "
            f"({change.owner.strip()})",
            markup=False,
        )
        reason = blocked.get(change.number)
        if reason:
            console.print(f"    ⛔ Blocked ({reason}): will be skipped", markup=False)
        warnings = change.get_permission_warnings()
        if warnings:
            lacking += 1
            for warning in warnings:
                console.print(f"    ⚠️ {warning}")
        elif not change.submittable:
            unverified += 1
    return lacking, unverified


def _print_permission_summary(lacking: int, unverified: int, console: Console) -> None:
    """Say what the listing showed of the caller's permissions."""
    if lacking:
        console.print(f"   ⚠️ You may lack permissions on {lacking} changes (see above)")
    elif unverified:
        console.print(
            "   ✅ You appear to have +2 Code-Review permission on every change"
        )
    else:
        console.print(
            "   ✅ You appear to have required permissions "
            "(+2 Code-Review, submit) on every change"
        )
    if unverified:
        console.print(
            f"   ℹ️ Submit permission not verified on {unverified} changes: "
            "Gerrit shows it only once a change is submittable"
        )


def _gate_human_query_batch(
    parsed_url: ParsedGerritTopicUrl,
    expression: str,
    changes: list[GerritChangeInfo],
    comparator: GerritChangeComparator,
    run: _GerritRunOptions,
) -> None:
    """Require --include-human-prs and --override unless all are automation.

    The override derives from the host, the query and every matched
    change's number and current revision, so it confirms exactly the
    batch listed.

    Raises:
        SystemExit: As ``_confirm_human_override``, or
            ``VALIDATION_ERROR`` without ``--include-human-prs``.
    """
    humans = [c for c in changes if not comparator.is_automation_change(c)]
    if not humans:
        return

    subject = f"the {len(changes)} changes matched by query '{expression}'"
    if not run.include_human_prs:
        exit_with_error(
            ExitCode.VALIDATION_ERROR,
            message=(
                f"❌ {len(humans)} of {subject} are not from a recognized "
                "automation tool"
            ),
            details=(
                "To submit them, run again with --include-human-prs; the run "
                "then prints the --override value that confirms the batch."
            ),
        )
    _confirm_human_override(
        _given_override(run.override),
        _generate_gerrit_query_sha("override", parsed_url.host, expression, changes),
        dry_run=run.dry_run,
        subject=subject,
        derivation=(
            "the host, the query and each matched change's number and current revision"
        ),
    )


def _batch_scope(to_submit: list[_Entry]) -> dict[int, str]:
    """Each change number to submit, mapped to its previewed revision."""
    return {change.number: change.current_revision for change, _ in to_submit}


def _refuse_submission_beyond_batch(
    service: GerritService,
    to_submit: list[_Entry],
    console: Console,
) -> None:
    """Refuse when submitting the batch could submit anything else.

    Gerrit can submit a change's unmerged ancestors with it, and with
    ``change.submitWholeTopic`` every change in its topic, in any
    project; none of those passed this preview, the human gate or the
    revision checks.  So every change's ``submitted_together`` set must
    lie within the batch, at the revisions listed, before any vote.

    Raises:
        SystemExit: ``VALIDATION_ERROR`` when any change's set reaches
            outside the batch, or Gerrit cannot say what it holds.
    """
    scope = _batch_scope(to_submit)
    problems: list[str] = []
    for change, _ in to_submit:
        problems.extend(service.submitted_together_problems(change.number, scope))
    if not problems:
        return
    console.print("\n❌ Submitting this batch could act outside it:")
    for problem in problems:
        console.print(f"   • {problem}", markup=False)
    exit_with_error(
        ExitCode.VALIDATION_ERROR,
        message="❌ Refusing to submit: the batch's exact scope cannot be guaranteed",
        details=(
            "Change the query so it matches every change Gerrit submits "
            "together (a term such as project: or branch: can leave one "
            "out), resolve any blocked member, then run again; or submit "
            "those changes separately."
        ),
    )


def _prepare_gerrit_query_batch(
    service: GerritService,
    parsed_url: ParsedGerritTopicUrl,
    comparator: GerritChangeComparator,
    console: Console,
    run: _GerritRunOptions,
) -> tuple[list[_Entry], list[tuple[GerritChangeInfo, str]]] | None:
    """Fetch, list, gate and confirm the batch a query URL names.

    Returns:
        The changes to submit and the blocked changes to skip, each with
        its reason; or None when the run should stop (a dry run, or the
        user declined).

    Raises:
        typer.Exit: When the query matches no open change.
        SystemExit: When the human gate refuses the batch, or submitting
            it could also submit changes outside it.
        GerritServiceError: When the result set is over the ceiling or
            not wholly parsable.
    """
    expression = parsed_url.query_expression
    console.print(f"📋 Fetching open changes matching query '{expression}'...")
    changes = service.get_open_changes_for_query(expression)
    if not changes:
        console.print(f"❌ No open changes match query '{expression}'")
        raise typer.Exit(1)

    blocked: dict[int, str] = {}
    for change in changes:
        reason = _blocked_reason(change)
        if reason:
            blocked[change.number] = reason

    lacking, unverified = _print_query_batch(changes, blocked, console)

    _gate_human_query_batch(parsed_url, expression, changes, comparator, run)

    to_submit: list[_Entry] = [(c, None) for c in changes if c.number not in blocked]
    skipped = [
        (c, f"Blocked: {blocked[c.number]}") for c in changes if c.number in blocked
    ]

    if not run.no_confirm or run.dry_run:
        label = "Dry run" if run.dry_run else "Preview"
        console.print(
            f"\n📊 {label}: {len(to_submit)} changes would be reviewed and "
            f"submitted; {len(skipped)} blocked would be skipped"
        )
        _print_permission_summary(lacking, unverified, console)

    _refuse_submission_beyond_batch(service, to_submit, console)

    if run.dry_run:
        console.print("\n🧪 Dry run: no changes were reviewed or submitted.")
        return None
    if not run.no_confirm:
        continue_sha = _generate_gerrit_query_sha(
            "continue", parsed_url.host, expression, changes
        )
        if not _pkg._confirm_continue_sha(continue_sha, console):
            return None

    return to_submit, skipped
