# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Result reporting for a completed merge pass.

Turns the per-PR results into the closing summary, expands the raw
failure reasons into readable guidance, and lists the PRs that could not
be merged.
"""

import typer

from ..ci_report import build_document, in_github_actions, write_results_file
from ..error_codes import ExitCode
from ..error_text import summarise_error
from ..merge_manager import (
    MergeResult,
)
from ..rule_violations import (
    RULE_VIOLATION_MARKER,
    is_rule_violation,
    required_status_check_names,
    required_workflow_names,
    status_check_violation_verb,
    violation_verb,
)
from ..url_parser import redact_target
from ._app import console
from ._context import _MergeContext


def _prs(count: int) -> str:
    """``"1 PR"`` or ``"7 PRs"``.

    A run reporting "Unsettled 1 PRs" reads as a typo, which invites the
    reader to distrust the number beside it.
    """
    return f"{count} PR" if count == 1 else f"{count} PRs"


#: Real-run outcomes a human has to act on.  ``FAILED`` could not merge;
#: ``BLOCKED`` will not merge on its own (conflicts, a recreate that
#: never landed).  The rest are not failures: ``UNSETTLED`` merges on a
#: re-run, ``AUTO_MERGE_PENDING`` completes server-side, and ``SKIPPED``
#: and ``CLOSED`` leave nothing to follow up.
_NEEDS_ATTENTION = frozenset({"failed", "blocked"})


def _exit_if_any_failed(real_results: list[MergeResult]) -> None:
    """Exit with ``MERGE_ERROR`` when a real run left PRs needing a human.

    Called after the final summary of a run that actually merged, never
    after a preview or dry run, whose outcomes are predictions.  Without
    it a run that merged nothing it attempted still exited 0, so a
    script or scheduled workflow could not tell success from failure.

    Raises:
        typer.Exit: With :attr:`ExitCode.MERGE_ERROR` when any result
            is failed or blocked.
    """
    count = sum(1 for r in real_results if r.status.value in _NEEDS_ATTENTION)
    if count == 0:
        return
    code = int(ExitCode.MERGE_ERROR)
    console.print(f"\n❌ {_prs(count)} failed or blocked; exit code {code}")
    raise typer.Exit(code=code)


def _print_final_merge_summary(real_results: list[MergeResult]) -> None:
    """Print the post-run 🚀 Final Results line and per-outcome recap.

    Shared by the org / repo / similar-PR confirmed-merge paths so
    every outcome category (including closed-without-merge) renders
    identically regardless of scope.

    The counts appear once.  Each category previously also printed its
    own line, so a run finishing with anything other than merges said
    the same numbers twice in two different orders --- and the live
    tracker had already said them a third time.
    """
    final_merged = sum(1 for r in real_results if r.status.value == "merged")
    final_failed = sum(1 for r in real_results if r.status.value == "failed")
    final_skipped = sum(1 for r in real_results if r.status.value == "skipped")
    final_blocked = sum(1 for r in real_results if r.status.value == "blocked")
    final_closed = sum(1 for r in real_results if r.status.value == "closed")
    final_unsettled = sum(1 for r in real_results if r.status.value == "unsettled")
    final_auto_merge = sum(
        1 for r in real_results if r.status.value == "auto_merge_pending"
    )
    parts = [f"{final_merged} merged"]
    if final_auto_merge > 0:
        parts.append(f"{final_auto_merge} auto-merge pending")
    parts.append(f"{final_failed} failed")
    if final_skipped > 0:
        parts.append(f"{final_skipped} skipped")
    if final_blocked > 0:
        parts.append(f"{final_blocked} blocked")
    if final_unsettled > 0:
        parts.append(f"{final_unsettled} unsettled")
    if final_closed > 0:
        parts.append(f"{final_closed} closed")
    console.print(f"\n🚀 Final Results: {', '.join(parts)}")
    if final_unsettled > 0:
        console.print("   ⏱️ Unsettled PRs will merge on a re-run")

    _print_failed_pr_details(real_results)


def _format_failure_reason(reason: str) -> list[str]:
    """Expand a failure reason into consistent display lines.

    Failures are rendered in a consistent shape so the final summary is
    actionable at a glance:

      1. the failed PR URL (prepended by the caller),
      2. a single **failure-type** line whose parts are joined with
         `` / `` and carry no trailing colon, e.g.
         ``Repository rule violations found / Required workflows failed``,
      3. one bullet (``• ``) per individual failing condition.

    Repository-ruleset violations arrive from GitHub as one long string
    that crams the offending workflow / status-check names into a
    quoted, comma-separated clause.  We split that into the type line
    plus a bullet per name for both the ``Required workflows`` and
    ``Required status check(s)`` variants.  Reasons we do not recognise
    are reduced to a single line and returned as they are.

    One rejection can name **both** kinds, and each is reported: a
    failing status context such as ``pre-commit.ci - pr`` used to be
    dropped whenever a workflow was named alongside it, leaving the
    summary listing conditions that pass and omitting the one that
    blocks.  Each kind also carries its own verb, because workflows that
    have not finished routinely accompany a context that has failed.
    """
    if not is_rule_violation(reason):
        # The unrecognised reason is whatever an exception said, which
        # for an ``httpx`` error is two lines.  Reduced here rather than
        # where it was composed: every route that reports a reason it
        # could not classify arrives at this branch, including the
        # catch-all that assigns ``str(e)`` for any escaping exception.
        return [summarise_error(reason)]

    workflows = required_workflow_names(reason)
    checks = required_status_check_names(reason)
    if not workflows and not checks:
        return [summarise_error(reason)]

    headings = [RULE_VIOLATION_MARKER]
    if workflows:
        headings.append(f"Required workflows {violation_verb(reason)}")
    if checks:
        headings.append(f"Required status checks {status_check_violation_verb(reason)}")

    # Bullets are labelled only when both kinds are present.  With one
    # kind the heading already says which, so labelling would add noise
    # to every existing report; with both, an unlabelled name does not
    # say which heading it belongs under.
    both = bool(workflows and checks)
    bullets = [
        f"• Required workflow: {name}" if both else f"• {name}" for name in workflows
    ] + [f"• Required status check: {name}" if both else f"• {name}" for name in checks]
    return [" / ".join(headings), *bullets]


def _print_failed_pr_details(
    merge_results: list[MergeResult],
) -> None:
    """Print URL and reason for every non-merged PR in the result list.

    A bare ``Failed: 1`` line in the summary forces the user to
    scroll back through the merge output to find which PR failed
    and why.  During a real merge run the per-PR status lines are
    no longer printed to the console at all (progress is conveyed
    by the live tracker counters), so this end-of-run report is
    the *only* place reasons appear.  It therefore covers every
    non-merged terminal outcome — failed, blocked, unsettled, skipped,
    closed and auto-merge pending — one section per outcome.
    """
    sections: list[tuple[str, str]] = [
        ("failed", "\n❌ Failed PRs:"),
        ("blocked", "\n🛑 Blocked PRs:"),
        ("unsettled", "\n⏱️ Unsettled PRs (re-run to merge):"),
        ("skipped", "\n⏭️ Skipped PRs:"),
        ("closed", "\n🚪 Closed PRs:"),
        ("auto_merge_pending", "\n🤖 Auto-merge pending PRs:"),
    ]
    for status_value, heading in sections:
        matching = [r for r in merge_results if r.status.value == status_value]
        if not matching:
            continue
        console.print(heading)
        for r in matching:
            url = getattr(r.pr_info, "html_url", "<unknown>")
            reason = r.error or "no reason reported"
            body = "\n".join(f"     {line}" for line in _format_failure_reason(reason))
            # markup=False so bracketed reasons are not eaten by Rich.
            console.print(f"   • {url}\n{body}", markup=False)


def _display_merge_results(
    merge_results: list[MergeResult],
    no_confirm: bool,
) -> None:
    """Print the final summary of merge results.

    In preview mode the per-category lines are the only counts, so they
    are printed.  After a real run the ``📈 Final Results`` line carries
    the same numbers, and printing both said everything twice.
    """
    merged_count = sum(1 for r in merge_results if r.status.value == "merged")
    failed_count = sum(1 for r in merge_results if r.status.value == "failed")
    skipped_count = sum(1 for r in merge_results if r.status.value == "skipped")
    blocked_count = sum(1 for r in merge_results if r.status.value == "blocked")
    unsettled_count = sum(1 for r in merge_results if r.status.value == "unsettled")
    closed_count = sum(1 for r in merge_results if r.status.value == "closed")
    auto_merge_count = sum(
        1 for r in merge_results if r.status.value == "auto_merge_pending"
    )

    if not no_confirm:
        if failed_count > 0:
            console.print(f"❌ Would fail to merge {_prs(failed_count)}")
        if skipped_count > 0:
            console.print(f"⏭️ Skipped {_prs(skipped_count)}")
        if blocked_count > 0:
            console.print(f"🛑 Blocked {_prs(blocked_count)}")
        if unsettled_count > 0:
            console.print(f"⏱️ Unsettled {_prs(unsettled_count)}")
        if closed_count > 0:
            console.print(f"🚪 Closed without merging: {_prs(closed_count)}")
        if auto_merge_count > 0:
            console.print(f"⏳ Auto-merge pending for {_prs(auto_merge_count)}")

    if no_confirm:
        parts = [f"{merged_count} merged"]
        if auto_merge_count > 0:
            parts.append(f"{auto_merge_count} auto-merge pending")
        parts.append(f"{failed_count} failed")
        if skipped_count > 0:
            parts.append(f"{skipped_count} skipped")
        if blocked_count > 0:
            parts.append(f"{blocked_count} blocked")
        if unsettled_count > 0:
            parts.append(f"{unsettled_count} unsettled")
        if closed_count > 0:
            parts.append(f"{closed_count} closed")
        console.print(f"📈 Final Results: {', '.join(parts)}")
        if unsettled_count > 0:
            console.print("   ⏱️ Unsettled PRs will merge on a re-run")

    _print_failed_pr_details(merge_results)


def _publish_results(
    ctx: _MergeContext,
    results: list[MergeResult],
    *,
    preview: bool,
) -> None:
    """Write the results document when running under GitHub Actions.

    Outside Actions this does nothing, so an operator at a terminal sees
    no change.  A file that cannot be written is reported and skipped:
    losing the step summary must not fail a run that merged correctly.
    """
    if not in_github_actions():
        return
    document = build_document(
        results,
        explain=_format_failure_reason,
        # The target lands in a file, a step summary and possibly Slack,
        # so it gets the same credential redaction as any displayed
        # target: no userinfo, query, fragment or path parameters.
        target=redact_target(ctx.pr_url),
        scope=ctx.scope,
        preview=preview,
        dry_run=ctx.dry_run,
        selection=ctx.repo_selection.describe() if ctx.repo_selection else None,
        scan_errors=ctx.scan_errors,
    )
    try:
        path = write_results_file(document)
    except OSError as exc:
        console.print(f"⚠️ Could not write the results file: {exc}", markup=False)
        return
    console.print(f"📄 Results file: {path}", markup=False)


def _conclude_run(
    ctx: _MergeContext,
    results: list[MergeResult],
    *,
    preview: bool,
    attempted: list[MergeResult] | None = None,
) -> None:
    """Publish a finished run's results, then apply its exit code.

    Every route that reports final results ends here, including an
    owner or repository with nothing to merge, so a scheduled run always
    leaves a results document behind.  Publishing comes first because
    the exit for a failed run would otherwise skip it.

    Args:
        ctx: The run's shared state.
        results: One result per PR evaluated or merged; empty when there
            was nothing to do.
        preview: The results are predictions (preview or dry run), which
            never fail the run.
        attempted: The PRs the run actually tried to merge, when that is
            narrower than ``results``.  A confirmed run records the PRs
            its preview rejected but never attempted them, so its exit
            code comes from this list alone, as it always has.
    """
    _publish_results(ctx, results, preview=preview)
    if not preview:
        _exit_if_any_failed(results if attempted is None else attempted)


def _confirmed_run_results(
    preview_results: list[MergeResult],
    real_results: list[MergeResult],
) -> list[MergeResult]:
    """The full record of a confirmed run: what it merged, and what it never tried.

    A confirmed pass re-runs only the PRs the preview judged mergeable,
    so its own results omit every PR the preview rejected.  Those still
    need a human, and a record built from the real pass alone would
    report a clean run that left them behind.
    """
    rejected = [r for r in preview_results if r.status.value != "merged"]
    return [*real_results, *rejected]
