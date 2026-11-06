# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Reviewing and submitting a batch of Gerrit changes.

Prints the changes in scope, previews what submission would do, and ---
once confirmed --- votes and submits them.  A change URL scopes the
batch to the changes similar to it; a query URL scopes it to exactly
the open changes its query matches (see ``_gerrit_query``).
"""

from pathlib import Path

import typer
from rich.console import Console

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..error_codes import ExitCode
from ..gerrit import (
    GerritAuthError,
    GerritChangeComparator,
    GerritChangeInfo,
    GerritComparisonResult,
    GerritRestError,
    GerritService,
    GerritServiceError,
    GerritSubmitResult,
)
from ..netrc import (
    GerritCredentials,
)
from ..progress_tracker import MergeProgressTracker
from ..url_parser import (
    ParsedGerritTopicUrl,
    ParsedUrl,
)
from ._gerrit_query import (
    _batch_scope,
    _check_query_options,
    _GerritRunOptions,
    _prepare_gerrit_query_batch,
)
from ._gerrit_setup import (
    _maybe_rebase_gerrit_change,
    _print_gerrit_final_summary,
    _resolve_gerrit_candidates,
    _resolve_gerrit_credentials_or_exit,
    _resolve_gerrit_only_automation,
    _resolve_gerrit_source_change,
)
from ._similarity import _format_gerrit_similarity


def _find_and_print_similar_changes(
    service: GerritService,
    comparator: GerritChangeComparator,
    source_change: GerritChangeInfo,
    candidates: list[GerritChangeInfo] | None,
    only_automation: bool,
    console: Console,
) -> list[tuple[GerritChangeInfo, GerritComparisonResult]]:
    """Score candidates against the source change and print the matches."""
    similar_changes = service.find_similar_changes(
        source_change,
        comparator,
        only_automation=only_automation,
        candidates=candidates,
    )

    console.print(f"Found {len(similar_changes)} similar changes:")
    for change, comparison in similar_changes:
        console.print(f"  • {change.project} #{change.number}: {change.subject}")
        console.print(f"    {_format_gerrit_similarity(comparison)}")

    return similar_changes


def _preview_gerrit_submission(
    source_change: GerritChangeInfo,
    all_changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
    no_confirm: bool,
    dry_run: bool,
    console: Console,
) -> bool:
    """Warn about permissions and preview the run, then decide to proceed.

    Permissions are per-project in Gerrit, so this checks the source
    change and warns if the user may lack sufficient permissions. It
    then either stops (dry run), prompts for confirmation
    (interactive), or proceeds straight to submission (--no-confirm).

    Returns:
        True when submission should proceed, False when the caller
        should stop (dry run, or the user declined confirmation).
    """
    permission_warnings = source_change.get_permission_warnings()
    if permission_warnings:
        console.print("\n⚠️ Permission warnings:")
        for warning in permission_warnings:
            console.print(f"   • {warning}")
        console.print(
            "\n   Note: Permissions vary by project. The operation may still "
            "succeed on some changes."
        )

    if not no_confirm or dry_run:
        label = "Dry run" if dry_run else "Preview"
        console.print(
            f"\n📊 {label}: {len(all_changes)} changes would be reviewed and submitted"
        )
        if source_change.has_required_permissions():
            console.print(
                "   ✅ You appear to have required permissions (+2 Code-Review, submit)"
            )
        else:
            console.print(
                "   ⚠️ You may not have all required permissions (see warnings above)"
            )
        if dry_run:
            console.print("\n🧪 Dry run: no changes were reviewed or submitted.")
            return False
        if not _pkg._confirm_gerrit_submission(source_change, console):
            return False

    return True


def _run_gerrit_submission(
    parsed_url: ParsedUrl | ParsedGerritTopicUrl,
    credentials: GerritCredentials,
    all_changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
    show_progress: bool,
    console: Console,
    skipped: list[tuple[GerritChangeInfo, str]] | None = None,
    scope: dict[int, str] | None = None,
) -> None:
    """Submit all changes in parallel and print the final summary.

    ``skipped`` holds the changes of a query batch left out of
    submission, each with its reason; they are counted in the final
    tally and, like failures, make the run exit non-zero.
    ``scope`` maps each change of a query batch to its previewed
    revision: each is reviewed and submitted only at that revision, so a
    patchset uploaded since is never approved, and only while Gerrit
    would submit nothing outside ``scope`` with it.

    Live in-place progress mirrors the GitHub merge path: the submit
    manager records each change's transitory and terminal states
    against the tracker while the parallel submission runs, and
    failures are recapped afterwards by _print_gerrit_final_summary
    instead of interleaved lines. The tracker is created (unstarted)
    before the submit manager so it can be handed over, but only
    started inside the try/finally so it is always stopped, even when
    submission setup or the run itself raises.
    """
    console.print(f"\n🚀 Submitting {len(all_changes)} changes...")
    if skipped and not all_changes:
        # Every change of the query batch is blocked: nothing to submit.
        _finish_gerrit_submission([], all_changes, console, skipped)
        return

    progress_tracker: MergeProgressTracker | None = None
    if show_progress:
        progress_tracker = MergeProgressTracker(
            parsed_url.host,
            operation_label="Submitting changes",
            operation_icon="▶️",
            unit_label="changes",
        )
        progress_tracker.set_total_prs(len(all_changes))

    submit_manager = _pkg.create_submit_manager(
        host=parsed_url.host,
        base_path=parsed_url.base_path,
        username=credentials.username,
        password=credentials.password,
        progress_tracker=progress_tracker,
    )

    try:
        if progress_tracker is not None:
            progress_tracker.start()
        results = submit_manager.submit_changes_parallel(
            all_changes, bind_revisions=scope is not None, scope=scope
        )
    finally:
        if progress_tracker is not None:
            progress_tracker.stop()

    _finish_gerrit_submission(results, all_changes, console, skipped)


def _finish_gerrit_submission(
    results: list[GerritSubmitResult],
    all_changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
    console: Console,
    skipped: list[tuple[GerritChangeInfo, str]] | None,
) -> None:
    """Print the final summary and exit non-zero if anything was left.

    As for a GitHub run: a change left unsubmitted needs a human, and an
    exit of 0 would tell a script or scheduled workflow otherwise.
    """
    _print_gerrit_final_summary(results, all_changes, console, skipped)

    code = int(ExitCode.MERGE_ERROR)
    failed = sum(1 for result in results if not result.success)
    if skipped:
        console.print(
            f"\n❌ {failed + len(skipped)} changes not submitted "
            f"({failed} failed, {len(skipped)} skipped); exit code {code}"
        )
        raise typer.Exit(code=code)
    if failed:
        noun = "change" if failed == 1 else "changes"
        console.print(f"\n❌ {failed} {noun} failed; exit code {code}")
        raise typer.Exit(code=code)


def _merge_gerrit_query(
    service: GerritService,
    parsed_url: ParsedGerritTopicUrl,
    credentials: GerritCredentials,
    comparator: GerritChangeComparator,
    console: Console,
    run: _GerritRunOptions,
) -> None:
    """Submit exactly the open changes a query URL matches, bar blocked ones."""
    batch = _prepare_gerrit_query_batch(service, parsed_url, comparator, console, run)
    if batch is None:
        return
    to_submit, skipped = batch
    _run_gerrit_submission(
        parsed_url,
        credentials,
        to_submit,
        run.show_progress,
        console,
        skipped=skipped,
        scope=_batch_scope(to_submit),
    )


def _merge_gerrit_change(
    service: GerritService,
    parsed_url: ParsedUrl,
    credentials: GerritCredentials,
    comparator: GerritChangeComparator,
    console: Console,
    run: _GerritRunOptions,
) -> None:
    """Submit a change URL's change with the open changes similar to it."""
    source_change = _resolve_gerrit_source_change(
        service, parsed_url, credentials, console
    )
    only_automation = _resolve_gerrit_only_automation(
        source_change,
        comparator,
        run.override,
        console,
        include_human_prs=run.include_human_prs,
        dry_run=run.dry_run,
    )

    source_change = _maybe_rebase_gerrit_change(
        service, source_change, credentials, console, dry_run=run.dry_run
    )

    candidates = _resolve_gerrit_candidates(
        service, source_change, parsed_url, run.topic, console
    )
    similar_changes = _find_and_print_similar_changes(
        service, comparator, source_change, candidates, only_automation, console
    )

    # Prepare list of changes to submit (similar + source)
    source_entry: tuple[GerritChangeInfo, GerritComparisonResult | None] = (
        source_change,
        None,
    )
    all_changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]] = [
        *similar_changes,
        source_entry,
    ]

    if not _preview_gerrit_submission(
        source_change, all_changes, run.no_confirm, run.dry_run, console
    ):
        return

    _run_gerrit_submission(
        parsed_url, credentials, all_changes, run.show_progress, console
    )


def _handle_gerrit_merge(
    parsed_url: ParsedUrl | ParsedGerritTopicUrl,
    no_confirm: bool,
    similarity_threshold: float,
    verbose: bool,
    console: Console,
    no_netrc: bool = False,
    netrc_file: Path | None = None,
    netrc_optional: bool = True,
    dry_run: bool = False,
    override: str | None = None,
    topic: str | None = None,
    show_progress: bool = True,
    include_human_prs: bool = False,
) -> None:
    """
    Handle merge operation for a Gerrit change or query URL.

    A change URL anchors on that change and submits it with the changes
    similar to it.  A query URL submits exactly the open changes its
    query matches: no anchor, no similarity scoring, no rebase.

    Args:
        parsed_url: Parsed Gerrit change URL (host, project, change
            number) or query URL (host, topic, query expression).
        no_confirm: If True, skip confirmation prompt.
        similarity_threshold: Threshold for matching similar changes;
            a change URL only.
        verbose: Enable verbose output.
        console: Rich console for output.
        no_netrc: If True, skip .netrc credential lookup.
        netrc_file: Explicit path to a .netrc file.
        netrc_optional: If True, don't fail if netrc not found.
        dry_run: If True, preview only and never review or submit any
            change, even when ``no_confirm`` is also set.
        override: Confirms a non-automation source change, or a query
            batch holding one, with the value a run without it prints;
            needs ``include_human_prs``.
        topic: Explicit topic to scope a change URL's similar-change
            search to; when omitted, the source change's own topic.
            Refused with a query URL, whose query it would replace.
        show_progress: If True, drive a live Rich progress tracker
            during submission (in-place counters, matching the GitHub
            merge path) instead of printing per-change status lines.
        include_human_prs: Bring a non-automation source change, and the
            changes similar to it, into scope once ``override`` confirms.
    """
    is_query = isinstance(parsed_url, ParsedGerritTopicUrl)
    if is_query:
        _check_query_options(topic, similarity_threshold, console)

    credentials = _resolve_gerrit_credentials_or_exit(
        parsed_url, no_netrc, netrc_file, verbose, console
    )

    kind = "query" if is_query else "change"
    console.print(f"🔍 Examining Gerrit {kind} on {parsed_url.host}...")

    try:
        service = _pkg.create_gerrit_service(
            host=parsed_url.host,
            base_path=parsed_url.base_path,
            username=credentials.username,
            password=credentials.password,
        )

        if not service.is_authenticated:
            console.print("⚠️ Warning: Service created but may not be authenticated")

        comparator = _pkg.create_gerrit_comparator(
            similarity_threshold=similarity_threshold
        )
        run = _GerritRunOptions(
            no_confirm=no_confirm,
            dry_run=dry_run,
            override=override,
            include_human_prs=include_human_prs,
            show_progress=show_progress,
            topic=topic,
        )
        if isinstance(parsed_url, ParsedGerritTopicUrl):
            _merge_gerrit_query(
                service, parsed_url, credentials, comparator, console, run
            )
        else:
            _merge_gerrit_change(
                service, parsed_url, credentials, comparator, console, run
            )

    except typer.Exit:
        # Re-raise typer.Exit without treating it as an error
        raise
    except GerritAuthError as e:
        console.print(f"❌ Gerrit authentication failed: {e}")
        console.print("   Check your GERRIT_USERNAME and GERRIT_PASSWORD")
        raise typer.Exit(1) from None
    except GerritRestError as e:
        console.print(f"❌ Gerrit API error: {e}")
        raise typer.Exit(1) from None
    except GerritServiceError as e:
        # Not a transport failure: the request succeeded but the
        # response could not be used.  Distinct from the generic
        # handler below, which would label a diagnosis we made
        # deliberately as unexpected.
        console.print(f"❌ Gerrit service error: {e}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ Error during Gerrit merge operation: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1) from None
