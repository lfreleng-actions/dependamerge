# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The gate in front of major version bumps.

A pull request is gated when :mod:`dependamerge.version_bumps` reads it
as ``major``, or as ``zero-minor`` under ``--gate-zero-minor``.  Merging
one is a deliberate act:

- an unattended run (``--no-confirm``) holds it back before the merge
  pass, recording it as skipped, the way drafts are set aside;
- an interactive run lists the mergeable ones after its preview and asks
  for a value derived from exactly those pull requests; anything else
  leaves them out and the rest merge;
- ``--allow-major`` turns the gate off, in every mode.

Held-back pull requests are skipped, never failed, so they leave a run's
exit code alone.
"""

from __future__ import annotations

import hashlib

import typer

from ..merge_manager import MergeResult, MergeStatus
from ..models import PullRequestInfo
from ..version_bumps import MAJOR, ZERO_MINOR, Classification, Update, classify
from ._app import console
from ._context import _MergeContext
from ._merge_report import _print_major_bumps

__all__ = [
    "ALLOW_MAJOR_HINT",
    "ALLOW_MAJOR_OPTION",
    "GATE_ZERO_MINOR_OPTION",
    "_confirm_major_bumps",
    "_gate_reason",
    "_gated_bump",
    "_major_set_value",
    "_recheck_major_heads",
    "_set_aside_major_bumps",
    "_source_held_back",
]

#: The ``--allow-major`` option.
ALLOW_MAJOR_OPTION = typer.Option(
    False,
    "--allow-major",
    help=(
        "Merge major version bumps like any other PR, in every mode. "
        "Without it, --no-confirm holds them back and an interactive run "
        "asks for a value of their own"
    ),
)

#: The ``--gate-zero-minor`` option.
GATE_ZERO_MINOR_OPTION = typer.Option(
    False,
    "--gate-zero-minor",
    help="Gate 0.x minor version bumps (0.7.0 to 0.8.0) as major bumps",
)

#: The hint an unattended run prints below the bumps it held back.
ALLOW_MAJOR_HINT = "re-run with --allow-major to merge them"


def _gated_levels(ctx: _MergeContext) -> frozenset[str]:
    """The ``version_bumps`` levels this run gates; none under ``--allow-major``."""
    if ctx.allow_major:
        return frozenset()
    if ctx.gate_zero_minor:
        return frozenset({MAJOR, ZERO_MINOR})
    return frozenset({MAJOR})


def _gated_bump(ctx: _MergeContext, pr: PullRequestInfo) -> Classification | None:
    """The classification of ``pr`` when the gate applies to it, else ``None``."""
    levels = _gated_levels(ctx)
    if not levels:
        return None
    bump = classify(pr.title, pr.body)
    return bump if bump.level in levels else None


def _gated_updates(ctx: _MergeContext, bump: Classification) -> tuple[Update, ...]:
    """The updates that put ``bump`` behind the gate.

    Failing any at a gated level (the level came from metadata naming no
    update read here), every update.
    """
    levels = _gated_levels(ctx)
    return tuple(u for u in bump.updates if u.level in levels) or bump.updates


def _describe_update(update: Update) -> str:
    name = f"{update.name} " if update.name else ""
    return f"{name}{update.old} → {update.new}"


def _gate_reason(ctx: _MergeContext, bump: Classification) -> str:
    """How a gated PR is recorded when left out of the merge.

    Names every gated update, so a grouped update says which of its
    dependencies put it behind the gate.
    """
    label = "major version bump" if bump.level == MAJOR else "0.x minor version bump"
    described = ", ".join(_describe_update(u) for u in _gated_updates(ctx, bump))
    return f"{label}: {described}" if described else label


def _skipped(pr: PullRequestInfo, reason: str) -> MergeResult:
    return MergeResult(pr_info=pr, status=MergeStatus.SKIPPED, error=reason)


def _set_aside_major_bumps(
    ctx: _MergeContext, prs: list[PullRequestInfo]
) -> list[PullRequestInfo]:
    """Return ``prs`` without the gated ones in an unattended run.

    The gated PRs are listed, with a hint, and recorded on the context
    as skipped, so the results document reports them without them taking
    any part in the merge, its counts or its confirmation values.  An
    interactive run keeps them: its preview decides which reach the
    prompt.
    """
    if not ctx.no_confirm:
        return prs
    held: list[tuple[PullRequestInfo, str]] = []
    for pr in prs:
        bump = _gated_bump(ctx, pr)
        if bump is not None:
            held.append((pr, _gate_reason(ctx, bump)))
    if not held:
        return prs
    console.print(f"⏸️ Major version bumps held back: {len(held)}")
    for pr, reason in held:
        # markup=False: names and versions come from PR text.
        console.print(
            f"   {pr.repository_full_name}#{pr.number}: {reason}", markup=False
        )
    console.print(f"   💡 {ALLOW_MAJOR_HINT}")
    ctx.set_aside_results.extend(_skipped(pr, reason) for pr, reason in held)
    held_ids = {id(pr) for pr, _ in held}
    return [pr for pr in prs if id(pr) not in held_ids]


def _source_held_back(ctx: _MergeContext) -> bool:
    """Whether an unattended run holds a single-PR run's source back."""
    return (
        ctx.no_confirm
        and ctx.source_pr is not None
        and _gated_bump(ctx, ctx.source_pr) is not None
    )


def _major_set_value(scope: str, prs: list[PullRequestInfo]) -> str:
    """The value that confirms merging a set of gated PRs.

    Derived from the scope and each PR's repository, number and head
    commit, as the human gate's ``--override`` value is but with its own
    prefix, so one cannot stand in for the other.  A new push, or another
    gated PR joining the set, changes it.
    """
    members = sorted(
        f"{pr.repository_full_name.lower()}#{pr.number}@{pr.head_sha}" for pr in prs
    )
    combined = f"majors:{scope.lower()}:{','.join(members)}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _pr_key(pr: PullRequestInfo) -> tuple[str, int]:
    return (pr.repository_full_name.lower(), pr.number)


def _record_confirmed_heads(ctx: _MergeContext, prs: list[PullRequestInfo]) -> None:
    ctx.confirmed_major_heads.update({_pr_key(pr): pr.head_sha for pr in prs})


def _live_head(ctx: _MergeContext, pr: PullRequestInfo) -> str | None:
    """The PR's head commit as GitHub reports it now, or ``None`` unread."""
    if ctx.github_client is None:
        return None
    owner, _, repo = pr.repository_full_name.partition("/")
    try:
        head = ctx.github_client.get_pull_request_info(owner, repo, pr.number).head_sha
    except Exception:
        return None
    return head if isinstance(head, str) and head else None


def _recheck_major_heads(
    ctx: _MergeContext, results: list[MergeResult]
) -> list[MergeResult]:
    """Leave out each confirmed gated PR whose head moved since it was shown.

    Run just before the real pass: a push during the prompts would
    otherwise merge a revision the operator never confirmed.  A head
    that cannot be read counts as moved.

    Returns:
        ``results``, with every such PR replaced by a skipped result at
        the same position.
    """
    if not ctx.confirmed_major_heads:
        return results
    rechecked: list[MergeResult] = []
    moved: list[PullRequestInfo] = []
    for result in results:
        confirmed = ctx.confirmed_major_heads.get(_pr_key(result.pr_info))
        bump = _gated_bump(ctx, result.pr_info)
        if (
            result.status.value != "merged"
            or confirmed is None
            or bump is None
            or _live_head(ctx, result.pr_info) == confirmed
        ):
            rechecked.append(result)
            continue
        reason = f"{_gate_reason(ctx, bump)}; head changed since it was confirmed"
        moved.append(result.pr_info)
        rechecked.append(_skipped(result.pr_info, reason))
    if moved:
        console.print(f"⏸️ Major version bumps left out, their head moved: {len(moved)}")
        for pr in moved:
            console.print(f"   {pr.repository_full_name}#{pr.number}", markup=False)
    return rechecked


def _print_gated(
    ctx: _MergeContext, gated: list[tuple[MergeResult, Classification]]
) -> None:
    console.print(
        "\n⚠️ Major version bumps among the mergeable pull requests: "
        f"{len(gated)} (merged only when confirmed)",
        markup=False,
    )
    for result, bump in gated:
        console.print(f"   • {result.pr_info.html_url}", markup=False)
        for update in _gated_updates(ctx, bump):
            console.print(f"     {_describe_update(update)}", markup=False)


def _ask_for_value(value: str) -> str | None:
    """The value the operator entered, or ``None`` for EOF or an interrupt."""
    console.print()
    console.print(f"To merge them too enter: {value}")
    try:
        entered: str = typer.prompt(
            "Enter the string above to include them "
            "(or press Enter to merge the rest without them)",
            default="",
            show_default=False,
        )
    except (KeyboardInterrupt, EOFError, typer.Abort):
        return None
    return entered.strip()


def _confirm_major_bumps(
    ctx: _MergeContext,
    scope: str,
    results: list[MergeResult],
    *,
    source: PullRequestInfo | None = None,
) -> list[MergeResult]:
    """Settle the gated PRs an interactive preview judged mergeable.

    Lists them apart from the rest and asks for the value
    :func:`_major_set_value` derives from ``scope`` and exactly those
    PRs.  The value keeps them in; Enter, anything else, EOF or an
    interrupt leaves them out, recorded as skipped.  The heads confirmed
    are recorded for :func:`_recheck_major_heads`.  A dry run lists them
    and says what a real run would do, without asking.  With nothing
    gated, the mergeable major bumps are listed as before.

    Args:
        ctx: The run's shared state.
        scope: The owner, repository or source PR the run addresses.
        results: The preview's results, in merge-list order.
        source: The PR a single-PR run was given.  When it is itself
            gated, the operator named the bump and the similar PRs make
            the same one, so a warning replaces the prompt.

    Returns:
        ``results``, with every gated PR left out replaced by a skipped
        result at the same position.
    """
    gated: list[tuple[int, MergeResult, Classification]] = []
    for index, result in enumerate(results):
        if result.status.value != "merged":
            continue
        bump = _gated_bump(ctx, result.pr_info)
        if bump is not None:
            gated.append((index, result, bump))

    source_bump = _gated_bump(ctx, source) if source is not None else None
    if not gated or source_bump is not None:
        _print_major_bumps(results, preview=True)
        if source_bump is not None:
            outcome = (
                "a real run merges it and the similar PRs once confirmed"
                if ctx.dry_run
                else "confirming below merges it and the similar PRs with it"
            )
            console.print(
                f"\n⚠️ The source PR is a {_gate_reason(ctx, source_bump)}; {outcome}.",
                markup=False,
            )
            if not ctx.dry_run:
                _record_confirmed_heads(ctx, [result.pr_info for _, result, _ in gated])
        return results

    _print_gated(ctx, [(result, bump) for _, result, bump in gated])

    if ctx.dry_run:
        console.print(
            "🧪 Dry run: a real run asks for a value to merge these, and "
            "without it merges the rest."
        )
        return results

    gated_prs = [result.pr_info for _, result, _ in gated]
    value = _major_set_value(scope, gated_prs)
    entered = _ask_for_value(value)
    if entered == value:
        console.print("✅ Major version bumps included.")
        _record_confirmed_heads(ctx, gated_prs)
        return results
    if entered is None:
        console.print("\n⏭️ No value entered: major version bumps left out.")
    elif entered == "":
        console.print("⏭️ Major version bumps left out.")
    else:
        console.print("❌ The value does not match: major version bumps left out.")

    left_out = {
        index: _skipped(result.pr_info, _gate_reason(ctx, bump))
        for index, result, bump in gated
    }
    kept = [left_out.get(index, result) for index, result in enumerate(results)]
    remaining = sum(1 for r in kept if r.status.value == "merged")
    console.print(f"Mergeable without them: {remaining}/{len(kept)} PRs")
    return kept
