# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Credential, change and candidate resolution for Gerrit.

Everything a Gerrit submission needs before the first vote is cast:
resolving credentials, locating the source change of a change URL,
honouring an override, rebasing when the change is behind, and
reading every open change it is compared against.  A query URL takes none of the
source-change steps: ``_gerrit_query`` handles it.
"""

from pathlib import Path
from typing import NamedTuple

import typer
from rich.console import Console

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..error_codes import (
    ExitCode,
    exit_with_error,
)
from ..error_text import summarise_error
from ..gerrit import (
    DEFAULT_LIST_OPTIONS,
    QUERY_BATCH_CEILING,
    SIMILAR_SCAN_CEILING,
    GerritChangeComparator,
    GerritChangeInfo,
    GerritComparisonResult,
    GerritService,
    GerritServiceError,
    GerritSubmitResult,
    topic_query_term,
)
from ..netrc import (
    GerritCredentials,
    NetrcParseError,
)
from ..url_parser import (
    ParsedGerritTopicUrl,
    ParsedUrl,
)
from ._human_gate import _confirm_human_override, _given_override
from ._sha import _generate_gerrit_override_sha
from ._similarity import _display_change_info


def _print_gerrit_final_summary(
    results: list[GerritSubmitResult],
    all_changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
    console: Console,
    skipped: list[tuple[GerritChangeInfo, str]] | None = None,
) -> None:
    """Print the post-run 🚀 Final Results line and failure recap.

    Mirrors the GitHub ``_print_final_merge_summary`` /
    ``_print_failed_pr_details`` pair: per-change status lines are
    not printed while the submission runs (progress is conveyed by
    the live tracker counters), so this end-of-run report is the
    only place failure reasons appear.

    Args:
        results: Submit results, one per attempted change.
        all_changes: The (change, comparison) tuples that were
            submitted; used to recover each change's web URL for the
            failure recap (``GerritSubmitResult`` carries only the
            project and number).
        console: Rich console to print to.
        skipped: Changes left out of submission, each with its reason.
            When given (a query batch), the tally counts them and the
            recap lists them; None leaves both out.
    """
    submitted = sum(1 for r in results if r.submitted)
    failed = [r for r in results if not r.success]
    reviewed_only = sum(1 for r in results if r.reviewed and not r.submitted)

    parts = [f"{submitted} submitted", f"{len(failed)} failed"]
    if skipped is not None:
        parts.append(f"{len(skipped)} skipped")
    if reviewed_only > 0:
        parts.append(f"{reviewed_only} reviewed (not submitted)")
    console.print(f"\n🚀 Final Results: {', '.join(parts)}")

    if skipped:
        console.print("\n⏭️ Skipped changes:")
        for change, reason in skipped:
            label = change.url or f"{change.project} #{change.number}"
            console.print(f"   • {label}\n     {reason}", markup=False)

    if not failed:
        return
    url_by_key = {
        (change.project, change.number): change.url for change, _ in all_changes
    }
    console.print("\n❌ Failed changes:")
    for result in failed:
        url = url_by_key.get((result.project, result.change_number)) or (
            f"{result.project} #{result.change_number}"
        )
        reason = summarise_error(result.error) if result.error else "no reason reported"
        # markup=False so bracketed reasons are not eaten by Rich.
        console.print(f"   • {url}\n     {reason}", markup=False)


def _resolve_gerrit_credentials_or_exit(
    parsed_url: ParsedUrl | ParsedGerritTopicUrl,
    no_netrc: bool,
    netrc_file: Path | None,
    verbose: bool,
    console: Console,
) -> GerritCredentials:
    """Resolve Gerrit credentials, or print guidance and exit.

    Args:
        parsed_url: Parsed Gerrit change or topic URL (for the host).
        no_netrc: If True, skip .netrc credential lookup.
        netrc_file: Explicit path to a .netrc file.
        verbose: When True, report which auth method was used.
        console: Rich console for output.

    Returns:
        Valid Gerrit credentials.

    Raises:
        typer.Exit: When no valid credentials can be resolved.
    """
    try:
        credentials = _pkg.resolve_gerrit_credentials(
            host=parsed_url.host,
            use_netrc=not no_netrc,
            netrc_file=netrc_file,
        )
    except NetrcParseError as e:
        console.print(f"⚠️ Error parsing .netrc file: {e}")
        credentials = None

    if credentials is None or not credentials.is_valid:
        console.print("❌ Gerrit credentials not found.")
        console.print("   Options:")
        console.print("   1. Create a ~/.netrc file with Gerrit credentials")
        console.print(
            "   2. Set GERRIT_USERNAME and GERRIT_PASSWORD environment variables"
        )
        console.print(
            "   Tip: Source your .secrets.gerrit file and run use_lf or use_onap"
        )
        raise typer.Exit(1)

    if verbose:
        console.print(f"🔑 Using credentials from {credentials.auth_method_display()}")

    return credentials


def _resolve_gerrit_source_change(
    service: GerritService,
    parsed_url: ParsedUrl,
    credentials: GerritCredentials,
    console: Console,
) -> GerritChangeInfo:
    """Fetch and display the change a change URL names, validating it.

    Returns:
        The source change.

    Raises:
        typer.Exit: When no change is found, or the source change is
            already merged or abandoned.
    """
    console.print(f"📋 Fetching change {parsed_url.change_number}...")
    source_change = service.get_change_info(parsed_url.change_number)

    if source_change is None:
        console.print("❌ Change not found")
        raise typer.Exit(1)

    # Display source change info using Rich table (same style as GitHub)
    _display_change_info(
        source_change,
        console=console,
        auth_method=credentials.auth_method_display(),
    )

    if source_change.status == "MERGED":
        console.print("\n✅ Change is already merged.")
        raise typer.Exit(0)

    if source_change.status == "ABANDONED":
        console.print("\n❌ Change has been abandoned.")
        raise typer.Exit(1)

    return source_change


def _resolve_gerrit_only_automation(
    source_change: GerritChangeInfo,
    comparator: GerritChangeComparator,
    override: str | None,
    console: Console,
    *,
    include_human_prs: bool = False,
    dry_run: bool = False,
) -> bool:
    """Decide whether the batch is automation-only, gating everything else.

    Automation source changes match only other automation changes.  Any
    other source needs ``--include-human-prs`` and the ``--override`` that
    confirms it, as a human-authored GitHub PR does, and then widens
    matching beyond automation.

    Returns:
        True when only automation changes should be matched.

    Raises:
        SystemExit: ``VALIDATION_ERROR`` without ``--include-human-prs``
            or with a wrong override; ``OVERRIDE_REQUIRED`` when only the
            override is missing.  Each used to exit 0, which a script
            could not tell from success.
    """
    if comparator.is_automation_change(source_change):
        return True

    owner = source_change.owner.strip()
    if not include_human_prs:
        exit_with_error(
            ExitCode.VALIDATION_ERROR,
            message=(
                f"❌ Source change {source_change.number} (owner {owner}) is "
                "not from a recognized automation tool"
            ),
            details=(
                "To submit it and similar changes, run again with "
                "--include-human-prs; the run then prints the --override "
                "value that confirms them."
            ),
        )
    _confirm_human_override(
        _given_override(override),
        _generate_gerrit_override_sha(source_change),
        dry_run=dry_run,
        subject=(
            f"change {source_change.number} by {owner} and the changes similar to it"
        ),
        derivation="the change's owner and subject",
    )
    return False


def _maybe_rebase_gerrit_change(
    service: GerritService,
    source_change: GerritChangeInfo,
    credentials: GerritCredentials,
    console: Console,
    *,
    dry_run: bool = False,
) -> GerritChangeInfo:
    """Rebase the source change when it has merge conflicts.

    A dry run only reports the conflict: a rebase uploads a new patchset,
    and a person's change reaches a dry run without the ``--override``
    that would confirm acting on it.

    Returns:
        The (possibly refreshed) source change. Unchanged when the
        change is already mergeable, or in a dry run.

    Raises:
        typer.Exit: When the rebase fails, whether due to conflicts
            needing manual resolution or another error.
    """
    if source_change.mergeable is not False:
        return source_change

    if dry_run:
        console.print(
            "\n🧪 Dry run: change has merge conflicts; a run would try to rebase it."
        )
        return source_change

    console.print("\n⚠️ Change has merge conflicts. Attempting to rebase...")
    rebase_result = service.rebase_change(source_change.number)

    if rebase_result["success"]:
        console.print("✅ Rebase successful! Refreshing change info...")
        source_change = service.get_change_info(source_change.number)
        _display_change_info(
            source_change,
            console=console,
            auth_method=credentials.auth_method_display(),
        )
        return source_change

    if rebase_result["conflict"]:
        console.print("\n❌ Rebase failed due to merge conflicts:")
        if rebase_result["conflicting_files"]:
            console.print("\n   Conflicting files:")
            for file_path in rebase_result["conflicting_files"]:
                console.print(f"   • {file_path}")
        console.print(
            "\n💡 To resolve: manually rebase the change locally and push a new patchset."
        )
        console.print(f"   git review -d {source_change.number}")
        console.print(f"   git rebase origin/{source_change.branch}")
        console.print("   # resolve conflicts, then:")
        console.print("   git review")
        raise typer.Exit(1)

    console.print(f"\n❌ Rebase failed: {rebase_result['error']}")
    raise typer.Exit(1)


class _CandidateScope(NamedTuple):
    """The open changes a change URL compares its source against."""

    expression: str
    """The Gerrit query expression that selects the scope."""
    ceiling: int
    """The most changes to accept before refusing the read."""
    label: str
    """The scope as the run reports it."""


def _gerrit_candidate_scope(
    source_change: GerritChangeInfo,
    topic: str | None,
    comparator: GerritChangeComparator,
    only_automation: bool,
    host: str,
) -> _CandidateScope:
    """Choose the open changes a change URL compares its source against.

    The narrowest scope that holds every change that could match: the
    topic when there is one; else the source owner's changes when a
    change by another account cannot match (owners must already be the
    same, or the threshold is beyond a change whose owner scores 0);
    else every open change on the server.

    An explicit --topic wins over the source change's own topic.
    """
    effective_topic = topic or source_change.topic
    if effective_topic:
        return _CandidateScope(
            topic_query_term(effective_topic),
            QUERY_BATCH_CEILING,
            f"with topic '{effective_topic}'",
        )

    account_id = source_change.owner_account_id
    if account_id is not None and (
        not only_automation or not comparator.other_owners_can_match()
    ):
        return _CandidateScope(
            f"owner:{account_id}",
            SIMILAR_SCAN_CEILING,
            f"owned by {source_change.owner.strip()}",
        )

    return _CandidateScope("status:open", SIMILAR_SCAN_CEILING, f"on {host}")


def _resolve_gerrit_candidates(
    service: GerritService,
    candidate_scope: _CandidateScope,
    console: Console,
) -> list[GerritChangeInfo]:
    """Read every open change in a candidate scope.

    :func:`_gerrit_candidate_scope` chooses the scope.  The read is complete
    or refused: comparing against part of the scope would leave out
    changes that match, and which ones would change from run to run.

    Raises:
        typer.Exit: When the scope holds more changes than its ceiling,
            a change that does not parse, or keeps changing while read.
            Nothing has been submitted at that point.
    """
    expression, ceiling, scope = candidate_scope
    console.print(f"\n🔍 Reading open changes {scope}...")
    try:
        candidates = service.get_open_changes_for_query(
            expression, ceiling=ceiling, options=DEFAULT_LIST_OPTIONS
        )
    except GerritServiceError as exc:
        console.print(f"❌ Could not read every open change {scope}: {exc}")
        console.print(
            "   Nothing was submitted. Narrow the scope with --topic, or "
            "use a query URL (/q/...) to name the changes to submit."
        )
        raise typer.Exit(1) from None

    console.print(f"🔍 Compared against {len(candidates)} open changes {scope}")
    return candidates
