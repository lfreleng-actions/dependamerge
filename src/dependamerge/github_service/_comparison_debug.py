# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""The per-comparison score breakdown the similar scan prints under ``--debug``."""

from __future__ import annotations

from ..models import ComparisonResult, PullRequestInfo


def print_comparison_debug(
    repo_full_name: str,
    source_pr: PullRequestInfo,
    target_pr: PullRequestInfo,
    comparator,
    comparison: ComparisonResult,
) -> None:
    """Print the per-comparison score breakdown for ``--debug`` runs."""
    from rich.console import Console

    debug_console = Console()
    debug_console.print(
        f"\n🔍 [bold]Comparing {repo_full_name}#{target_pr.number}[/bold]"
    )
    debug_console.print(f"   Title: {target_pr.title}")
    debug_console.print(f"   Author: {target_pr.author}")

    # Show individual scores
    title_score = comparator._compare_titles(source_pr.title, target_pr.title)
    body_score = comparator._compare_bodies(source_pr.body, target_pr.body)
    files_score = comparator._compare_file_changes(
        source_pr.files_changed, target_pr.files_changed
    )
    author_score = (
        1.0
        if comparator._normalize_author(source_pr.author)
        == comparator._normalize_author(target_pr.author)
        else 0.0
    )

    debug_console.print(f"   📝 Title score: {title_score:.3f}")
    debug_console.print(f"   📄 Body score: {body_score:.3f}")
    debug_console.print(f"   📁 Files score: {files_score:.3f}")
    debug_console.print(f"   👤 Author score: {author_score:.3f}")
    debug_console.print(
        f"   🎯 Overall: {comparison.confidence_score:.3f} (threshold: 0.8)"
    )

    if comparison.is_similar:
        debug_console.print(
            f"   ✅ [green]SIMILAR[/green] - {', '.join(comparison.reasons)}"
        )
        return

    debug_console.print("   ❌ [red]NOT SIMILAR[/red]")

    # Show why it failed
    if title_score == 0:
        source_pkg = comparator._extract_package_name(source_pr.title)
        target_pkg = comparator._extract_package_name(target_pr.title)
        debug_console.print(f"      📦 Source package: '{source_pkg}'")
        debug_console.print(f"      📦 Target package: '{target_pkg}'")

    if body_score < 0.6:
        if target_pr.body is None:
            debug_console.print("      ⚠️ Target PR has no body")
        elif source_pr.body is None:
            debug_console.print("      ⚠️ Source PR has no body")
        else:
            debug_console.print(
                f"      📄 Body comparison failed (score: {body_score:.3f})"
            )
