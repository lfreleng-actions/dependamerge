# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The steps behind the ``close`` command.

Authorises the request, finds the PRs similar to the source, reports what
was found, and closes them --- as a dry run, interactively, or straight
away.
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field

import typer

# Names substituted at ``dependamerge.cli.<name>`` are read from
# the package at call time, so one substitution reaches every
# caller rather than only the module that bound the name.
import dependamerge.cli as _pkg

from ..bot_policy import BotPolicy
from ..bot_policy_files import PolicyFiles
from ..close_manager import CloseResult
from ..error_codes import (
    ExitCode,
    exit_for_configuration_error,
    exit_for_pr_state_error,
    exit_with_error,
)
from ..models import ComparisonResult, PullRequestInfo
from ..progress_tracker import MergeProgressTracker
from ._app import console
from ._bot_scope import _out_of_scope_hint
from ._human_gate import (
    _confirm_human_override,
    _given_override,
    _refuse_human_source,
)
from ._policy_files import (
    _apply_policy_files,
    _capture_policy_files,
    _captured_app_prs,
    _keep_admitted_candidates,
    _load_source_policy_files,
)
from ._sha import _generate_continue_sha, _generate_override_sha
from ._similarity import _format_condensed_similarity


@dataclass
class _CloseContext:
    """Shared state for the ``close`` command pipeline."""

    token: str
    github_client: _pkg.GitHubClient
    # The GitHub host this run addresses, taken from the pull request
    # URL.  A GitHub Enterprise Server install serves its API from
    # different base URLs, so a run that loses the host silently
    # addresses github.com instead.
    host: str
    owner: str
    repo_name: str
    pr_number: int
    source_pr: PullRequestInfo
    progress_tracker: MergeProgressTracker | None
    # The GitHub Apps this run may act on; without any flags or user
    # config, only the built-in Apps.
    bot_policy: BotPolicy = field(default_factory=BotPolicy)
    # The repository and owner bot policy files the run has read.
    policy_files: PolicyFiles = field(default_factory=PolicyFiles)
    # The kinds of unreadable variable the run has warned about.
    policy_variables_warned: frozenset[str] = frozenset()


def _run_close_parallel(
    ctx: _CloseContext,
    prs: list[PullRequestInfo],
    preview_mode: bool,
) -> list[CloseResult]:
    """Close (or preview-close) ``prs`` in parallel."""

    async def _close_parallel() -> list[CloseResult]:
        close_manager = _pkg.AsyncCloseManager(
            token=ctx.token,
            host=ctx.host,
            progress_tracker=ctx.progress_tracker,
            preview_mode=preview_mode,
        )
        async with close_manager:
            # Convert to list of tuples (PR, None) for consistency
            pr_tuples: list[tuple[PullRequestInfo, ComparisonResult | None]] = [
                (pr, None) for pr in prs
            ]
            return await close_manager.close_prs_parallel(pr_tuples)

    return asyncio.run(_close_parallel())


def _print_close_debug_matching(
    ctx: _CloseContext,
    comparator: _pkg.PRComparator,
    similarity_threshold: float,
) -> None:
    """Show detailed matching diagnostics for the source PR."""
    source_pr = ctx.source_pr
    console.print("\n🔍 Debug Matching Information")
    console.print(
        "   Source PR automation status: "
        f"{ctx.github_client.is_automation_author(source_pr.author)}"
    )
    console.print(
        f"   Extracted package: '{comparator._extract_package_name(source_pr.title)}'"
    )
    console.print(f"   Similarity threshold: {similarity_threshold}")
    if source_pr.body:
        console.print(f"   Body preview: {source_pr.body[:100]}...")
        console.print(
            f"   Is dependabot body: {comparator._is_dependabot_body(source_pr.body)}"
        )
    else:
        console.print("   ⚠️ Source PR has no body")
    console.print()


def _validate_close_authorization(
    ctx: _CloseContext,
    override: str | None,
    *,
    include_human_prs: bool,
    dry_run: bool,
) -> None:
    """Gate the source PR on who wrote it, as ``merge`` does.

    An App the bot policy admits passes.  A person's PR needs
    ``--include-human-prs`` and the ``--override`` that confirms it.
    Every refusal exits non-zero; returning without closing anything
    used to exit 0, which a script could not tell from success.

    Raises:
        SystemExit: ``VALIDATION_ERROR`` for an App outside the policy, a
            human source without the flag, or a wrong override;
            ``OVERRIDE_REQUIRED`` when only the override is missing.
    """
    author = ctx.source_pr.author
    if ctx.source_pr.is_draft is True:
        exit_for_pr_state_error(
            ctx.pr_number,
            "a draft",
            details=(
                "Drafts are set aside, as merge sets them aside; mark it ready "
                "for review first. Nothing was scanned or closed"
            ),
        )
    # The owner's and repository's files decide which Apps are in scope.
    _load_source_policy_files(ctx)
    if ctx.github_client.is_automation_author(author):
        _require_source_in_scope(ctx)
        return

    if not include_human_prs:
        _refuse_human_source(author)

    commit_messages = ctx.github_client.get_pull_request_commits(
        ctx.owner, ctx.repo_name, ctx.pr_number
    )
    first_commit_line = commit_messages[0].split("\n")[0] if commit_messages else ""
    _confirm_human_override(
        _given_override(override),
        _generate_override_sha(ctx.source_pr, first_commit_line),
        dry_run=dry_run,
        subject=(
            f"closing {ctx.source_pr.html_url} by {author} and the PRs similar to it"
        ),
        derivation="the PR's author and first commit message",
    )


def _require_source_in_scope(ctx: _CloseContext) -> None:
    """Stop when the run's policy puts the source PR out of scope.

    Judged when the request is authorised, and again once the search has
    applied every policy it read: that can still invalidate the source's
    repository or exclude its App, and close has no later guard, while
    the close list always starts with the source.

    Raises:
        SystemExit: ``CONFIGURATION_ERROR`` when the source repository's
            policy is invalid; ``VALIDATION_ERROR`` when the source's
            App is out of scope.
    """
    source = ctx.source_pr
    repository = source.repository_full_name
    problem = ctx.bot_policy.repository_error(repository)
    if problem is not None:
        exit_for_configuration_error(
            message=f"❌ Invalid bot policy in {repository}",
            details=f"{problem}. Nothing in the repository is in scope.",
        )
    if not ctx.github_client.is_automation_author(source.author):
        return
    decision = ctx.bot_policy.decide(source.author, repository)
    if not decision.in_scope:
        exit_with_error(
            ExitCode.VALIDATION_ERROR,
            message=(
                f"❌ Source PR author {source.author} is not in this run's "
                f"scope ({decision.reason})"
            ),
            details=_out_of_scope_hint(source.author),
        )


def _find_similar_prs_for_close(
    ctx: _CloseContext,
    comparator: _pkg.PRComparator,
    debug_matching: bool,
) -> list[tuple[PullRequestInfo, ComparisonResult]]:
    """Search the owner for PRs similar to the source PR."""
    if ctx.progress_tracker:
        console.print()
    else:
        console.print(f"\nChecking owner: {ctx.owner}")

    # Use GitHubService for async PR finding
    from ..github_service import GitHubService

    if ctx.progress_tracker:
        ctx.progress_tracker.update_operation("Listing repositories...")

    # Every App PR the search saw, matching or not: they decide whether
    # an unreadable variable is worth a warning.
    seen: list[PullRequestInfo] = []

    async def _find_similar():
        svc = GitHubService(
            token=ctx.token,
            host=ctx.host,
            progress_tracker=ctx.progress_tracker,
            debug_matching=debug_matching,
            read_policies=True,
        )
        # The variables read with the source's are not read again.
        svc.adopt_policy_variables(ctx.policy_files)
        try:
            only_automation = ctx.github_client.is_automation_author(
                ctx.source_pr.author
            )
            found = await svc.find_similar_prs(
                ctx.owner,
                ctx.source_pr,
                comparator,
                only_automation=only_automation,
                bot_policy=ctx.bot_policy,
            )
            _capture_policy_files(ctx, svc)
            seen.extend(_captured_app_prs(svc))
            return found
        finally:
            await svc.close()

    found: list[tuple[PullRequestInfo, ComparisonResult]] = asyncio.run(_find_similar())
    # Apply and check every candidate repository's policy, as merge does:
    # close has no later recheck, so an invalid repository's PRs must not
    # survive to it, nor the source, which every close list starts with.
    _apply_policy_files(ctx, [ctx.source_pr, *seen, *(pr for pr, _ in found)])
    _require_source_in_scope(ctx)
    return _keep_admitted_candidates(ctx.bot_policy, found)


def _print_close_analysis_summary(
    ctx: _CloseContext,
    all_similar_prs: list[tuple[PullRequestInfo, ComparisonResult]],
) -> None:
    """Report the search outcome and list the matching PRs."""
    if ctx.progress_tracker:
        ctx.progress_tracker.stop()
        summary = ctx.progress_tracker.get_summary()
        elapsed_time = summary.get("elapsed_time")
        total_prs_analyzed = summary.get("total_prs_analyzed")
        completed_repositories = summary.get("completed_repositories")
        similar_prs_found = summary.get("similar_prs_found")
        errors_count = summary.get("errors_count", 0)
        console.print(f"\n✅ Analysis completed in {elapsed_time}")
        console.print(
            f"📊 Analyzed {total_prs_analyzed} PRs across {completed_repositories} repositories"
        )
        console.print(f"🔍 Found {similar_prs_found} similar PRs")
        if errors_count > 0:
            console.print(f"⚠️ {errors_count} errors encountered during analysis")
        console.print()
    else:
        console.print(f"\n🔍 Found {len(all_similar_prs)} similar PRs")

    if not all_similar_prs:
        # Not a failure: the supplied PR is simply the only one to
        # close.  Use a neutral glyph rather than ❌ so the output
        # does not read like an error.
        console.print("⏩ No similar PRs found for this owner")

    for target_pr, comparison in all_similar_prs:
        console.print(f"  • {target_pr.repository_full_name} #{target_pr.number}")
        console.print(f"    {_format_condensed_similarity(comparison)}")


def _run_close_dry_run(
    ctx: _CloseContext,
    all_prs_to_close: list[PullRequestInfo],
) -> None:
    """Evaluate in preview mode and report what *would* be closed."""
    if ctx.progress_tracker:
        ctx.progress_tracker.start()
        console.print()
    else:
        console.print(
            f"\n🧪 Dry run: evaluating {len(all_prs_to_close)} pull requests..."
        )

    close_results = _run_close_parallel(ctx, all_prs_to_close, True)

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()
        console.print()

    closeable_count = sum(1 for r in close_results if r.status.value == "closed")
    console.print(
        f"\n🧪 Dry run: would close {closeable_count}/"
        f"{len(all_prs_to_close)} PRs (no changes made)"
    )


def _run_interactive_close(
    ctx: _CloseContext,
    all_prs_to_close: list[PullRequestInfo],
) -> None:
    """Preview the close, then prompt for SHA-gated confirmation."""
    if ctx.progress_tracker:
        ctx.progress_tracker.start()
        console.print()
    else:
        console.print(f"\n🚀 Evaluating {len(all_prs_to_close)} pull requests...")

    close_results = _run_close_parallel(ctx, all_prs_to_close, True)

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()
        console.print()

    # Count closeable PRs
    closed_count = sum(1 for r in close_results if r.status.value == "closed")
    total_to_close = len(all_prs_to_close)

    console.print(f"\nCloseable {closed_count}/{total_to_close} PRs")

    if closed_count == 0:
        console.print("\n❌ No PRs are eligible for closing")
        return

    # Generate continuation SHA and prompt user
    commit_messages = ctx.github_client.get_pull_request_commits(
        ctx.owner, ctx.repo_name, ctx.pr_number
    )
    first_commit_line = commit_messages[0].split("\n")[0] if commit_messages else ""
    continue_sha_hash = _generate_continue_sha(ctx.source_pr, first_commit_line)
    console.print()
    console.print(f"To proceed with closing enter: {continue_sha_hash}")

    # Check if in test mode (don't prompt during tests)
    if "pytest" in sys.modules or os.getenv("TESTING"):
        console.print("⚠️ Test mode detected - skipping interactive prompt")
        return

    user_input = typer.prompt(
        "\nEnter the string above to continue (or press Enter to cancel)"
    ).strip()

    if user_input != continue_sha_hash:
        console.print("\n❌ Operation cancelled by user")
        return

    # Run actual close on closeable PRs only
    console.print(f"\n🔨 Closing {closed_count} closeable pull requests...")
    closeable_prs = [
        all_prs_to_close[i]
        for i, result in enumerate(close_results)
        # These were preview "closed"
        if result.status.value == "closed"
    ]

    if ctx.progress_tracker:
        ctx.progress_tracker.start()

    final_results = _run_close_parallel(ctx, closeable_prs, False)

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()

    # Count final results
    final_closed = sum(1 for r in final_results if r.status.value == "closed")
    final_failed = sum(1 for r in final_results if r.status.value == "failed")

    console.print(f"\n🚀 Final Results: {final_closed} closed, {final_failed} failed")


def _run_immediate_close(
    ctx: _CloseContext,
    all_prs_to_close: list[PullRequestInfo],
) -> None:
    """Close all candidate PRs without confirmation."""
    if ctx.progress_tracker:
        ctx.progress_tracker.start()
    console.print(f"\n🚀 Closing {len(all_prs_to_close)} pull requests...")

    close_results = _run_close_parallel(ctx, all_prs_to_close, False)

    if ctx.progress_tracker:
        ctx.progress_tracker.stop()

    # Count results
    closed_count = sum(1 for r in close_results if r.status.value == "closed")
    failed_count = sum(1 for r in close_results if r.status.value == "failed")

    console.print(f"\n🚀 Final Results: {closed_count} closed, {failed_count} failed")
