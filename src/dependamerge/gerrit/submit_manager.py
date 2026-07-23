# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Gerrit submit manager for parallel review and submit operations.

This module provides the GerritSubmitManager class for handling bulk
approval (+2 Code-Review) and submit operations on Gerrit changes.

It supports:
- Parallel submission of multiple changes
- Review (vote) operations with configurable labels
- Submit with pre-flight checks (submittable status)
- Error handling and result tracking
- Dry-run mode for previewing operations
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, Any

from dependamerge.gerrit.client import (
    GerritAuthError,
    GerritRestError,
    build_client,
)
from dependamerge.gerrit.models import (
    GerritChangeInfo,
    GerritComparisonResult,
    GerritSubmitResult,
)

if TYPE_CHECKING:
    from dependamerge.progress_tracker import MergeProgressTracker


log = logging.getLogger("dependamerge.gerrit.submit_manager")


class SubmitStatus(str, Enum):
    """Status values for submit operations."""

    PENDING = "pending"
    REVIEWING = "reviewing"
    REVIEWED = "reviewed"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class GerritSubmitManager:
    """
    Manages parallel approval and submission of Gerrit changes.

    This class handles the workflow of reviewing changes (applying
    Code-Review +2 votes) and submitting them.
    """

    def __init__(
        self,
        host: str,
        base_path: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 30.0,
        max_workers: int = 5,
        progress_tracker: MergeProgressTracker | None = None,
    ) -> None:
        """
        Initialize the submit manager.

        Args:
            host: Gerrit server hostname.
            base_path: Optional base path (e.g., "infra").
            username: HTTP username for authentication.
            password: HTTP password for authentication.
            timeout: Request timeout in seconds.
            max_workers: Maximum parallel workers for submissions.
            progress_tracker: Optional progress tracker for UI feedback.
        """
        self.host = host
        self.base_path = base_path
        self._max_workers = max_workers
        self._progress_tracker = progress_tracker

        self._client = build_client(
            host,
            base_path=base_path,
            timeout=timeout,
            username=username,
            password=password,
        )

        if not self._client.is_authenticated:
            log.warning(
                "GerritSubmitManager initialized without authentication. "
                "Review and submit operations will fail."
            )

        log.debug(
            "GerritSubmitManager initialized: host=%s, base_path=%s, "
            "max_workers=%d, auth=%s",
            host,
            base_path,
            max_workers,
            "yes" if self._client.is_authenticated else "no",
        )

    @property
    def is_authenticated(self) -> bool:
        """Check if the manager has authentication credentials."""
        return self._client.is_authenticated

    def submit_changes(
        self,
        changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
        review_labels: dict[str, int] | None = None,
        dry_run: bool = False,
    ) -> list[GerritSubmitResult]:
        """
        Submit multiple changes sequentially.

        Args:
            changes: List of (change, comparison_result) tuples.
            review_labels: Labels to apply (default: {"Code-Review": 2}).
            dry_run: If True, simulate operations without making changes.

        Returns:
            List of GerritSubmitResult for each change.
        """
        if review_labels is None:
            review_labels = {"Code-Review": 2}

        results: list[GerritSubmitResult] = []

        for change, _comparison in changes:
            result = self._submit_with_tracking(change, review_labels, dry_run)
            results.append(result)

        return results

    def submit_changes_parallel(
        self,
        changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
        review_labels: dict[str, int] | None = None,
        dry_run: bool = False,
    ) -> list[GerritSubmitResult]:
        """
        Submit multiple changes in parallel.

        Args:
            changes: List of (change, comparison_result) tuples.
            review_labels: Labels to apply (default: {"Code-Review": 2}).
            dry_run: If True, simulate operations without making changes.

        Returns:
            List of GerritSubmitResult for each change.
        """
        if review_labels is None:
            review_labels = {"Code-Review": 2}

        if not changes:
            return []

        # Use ThreadPoolExecutor for parallel execution.  Keep each
        # future paired with its change so an unexpected worker error
        # can still be attributed to the right change in the results
        # (and mapped back to a URL in the final failure recap).
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                (
                    executor.submit(
                        self._submit_with_tracking, change, review_labels, dry_run
                    ),
                    change,
                )
                for change, _comparison in changes
            ]

            results = []
            for future, change in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    log.error(
                        "Unexpected error in parallel submit for %s #%d: %s",
                        change.project,
                        change.number,
                        exc,
                    )
                    results.append(
                        GerritSubmitResult.failure_result(
                            change_number=change.number,
                            project=change.project,
                            error=str(exc),
                        )
                    )

        return results

    def _submit_with_tracking(
        self,
        change: GerritChangeInfo,
        review_labels: dict[str, int],
        dry_run: bool,
    ) -> GerritSubmitResult:
        """Submit a single change while driving the progress tracker.

        Mirrors the GitHub merge pipeline's tracker protocol: the
        change enters a transitory ``submitting`` display state while
        the review + submit round-trips run, then records a terminal
        ``merge_success`` / ``merge_failure`` outcome (which also
        clears the transitory entry and advances completion progress).
        No-op when no tracker was supplied.
        """
        tracker = self._progress_tracker
        change_key = f"{change.project}#{change.number}"
        if tracker is not None:
            tracker.track_pr_state(change_key, "submitting")
        try:
            result = self._submit_single_change(change, review_labels, dry_run)
        except Exception:
            # _submit_single_change catches expected errors; anything
            # escaping is unexpected, but the tracker entry must not
            # be left dangling in the transitory state.
            if tracker is not None:
                tracker.merge_failure(change_key)
            raise
        if tracker is not None:
            if result.success:
                tracker.merge_success(change_key)
            else:
                tracker.merge_failure(change_key)
        return result

    def _precheck_submit(
        self,
        change: GerritChangeInfo,
        dry_run: bool,
        start_time: float,
    ) -> GerritSubmitResult | None:
        """Return a short-circuit result, or None when submission may proceed.

        Rejects changes that cannot be submitted (not open, work in
        progress) and satisfies dry runs without any network calls.
        """
        if not change.is_open:
            return GerritSubmitResult.failure_result(
                change_number=change.number,
                project=change.project,
                error=f"Change is not open (status: {change.status})",
                duration=time.time() - start_time,
            )

        if change.work_in_progress:
            return GerritSubmitResult.failure_result(
                change_number=change.number,
                project=change.project,
                error="Change is marked as Work In Progress",
                duration=time.time() - start_time,
            )

        if dry_run:
            log.info(
                "[DRY RUN] Would review and submit %s #%d",
                change.project,
                change.number,
            )
            # A dry run performs no review or submit, so report the
            # simulated success with reviewed/submitted left False.
            # Callers gate real side effects on ``submitted`` (e.g.
            # closing the corresponding GitHub PR after a Gerrit
            # submit), so a dry run must never claim it submitted.
            return GerritSubmitResult.success_result(
                change_number=change.number,
                project=change.project,
                reviewed=False,
                submitted=False,
                duration=time.time() - start_time,
            )

        return None

    def _submit_error_result(
        self,
        change: GerritChangeInfo,
        exc: Exception,
        reviewed: bool,
        start_time: float,
    ) -> GerritSubmitResult:
        """Map an exception raised during submission to a failure result.

        Classifies the exception, logs it at the matching level (auth
        and REST errors are expected; anything else is logged with a
        traceback), and returns a failure result.
        """
        if isinstance(exc, GerritAuthError):
            log.error(
                "Authentication error for %s #%d: %s",
                change.project,
                change.number,
                exc,
            )
            message = f"Authentication error: {exc}"
        elif isinstance(exc, GerritRestError):
            log.error(
                "REST error for %s #%d: %s",
                change.project,
                change.number,
                exc,
            )
            message = f"REST error: {exc}"
        else:
            log.exception(
                "Unexpected error for %s #%d: %s",
                change.project,
                change.number,
                exc,
            )
            message = f"Unexpected error: {exc}"

        return GerritSubmitResult.failure_result(
            change_number=change.number,
            project=change.project,
            error=message,
            reviewed=reviewed,
            duration=time.time() - start_time,
        )

    def _submit_single_change(
        self,
        change: GerritChangeInfo,
        review_labels: dict[str, int],
        dry_run: bool,
    ) -> GerritSubmitResult:
        """
        Submit a single change (review + submit).

        Args:
            change: The change to submit.
            review_labels: Labels to apply.
            dry_run: If True, simulate without making changes.

        Returns:
            GerritSubmitResult indicating success or failure.
        """
        start_time = time.time()
        reviewed = False

        precheck = self._precheck_submit(change, dry_run, start_time)
        if precheck is not None:
            return precheck

        try:
            review_success = self._review_change(change.number, review_labels)
            if not review_success:
                return GerritSubmitResult.failure_result(
                    change_number=change.number,
                    project=change.project,
                    error="Failed to apply review",
                    reviewed=False,
                    duration=time.time() - start_time,
                )
            reviewed = True
            log.info(
                "Applied review to %s #%d: %s",
                change.project,
                change.number,
                review_labels,
            )

            submit_success = self._submit_change(change.number)
            if not submit_success:
                return GerritSubmitResult.failure_result(
                    change_number=change.number,
                    project=change.project,
                    error="Failed to submit (change may not be submittable)",
                    reviewed=reviewed,
                    duration=time.time() - start_time,
                )
            log.info(
                "Submitted %s #%d",
                change.project,
                change.number,
            )

            return GerritSubmitResult.success_result(
                change_number=change.number,
                project=change.project,
                reviewed=reviewed,
                submitted=True,
                duration=time.time() - start_time,
            )

        except Exception as exc:
            return self._submit_error_result(change, exc, reviewed, start_time)

    def _review_change(
        self,
        change_number: int,
        labels: dict[str, int],
    ) -> bool:
        """
        Apply a review (vote) to a change.

        Args:
            change_number: The change number.
            labels: Labels to apply (e.g., {"Code-Review": 2}).

        Returns:
            True if successful, False otherwise.
        """
        endpoint = f"/changes/{change_number}/revisions/current/review"
        payload = {"labels": labels}

        try:
            self._client.post(endpoint, data=payload)
            return True
        except GerritRestError as exc:
            log.warning("Failed to review change %d: %s", change_number, exc)
            return False

    def _submit_change(self, change_number: int) -> bool:
        """
        Submit a change.

        Args:
            change_number: The change number.

        Returns:
            True if successful, False otherwise.
        """
        endpoint = f"/changes/{change_number}/submit"

        try:
            self._client.post(endpoint)
            return True
        except GerritRestError as exc:
            log.warning("Failed to submit change %d: %s", change_number, exc)
            return False

    def review_only(
        self,
        changes: list[GerritChangeInfo],
        review_labels: dict[str, int] | None = None,
        dry_run: bool = False,
    ) -> list[GerritSubmitResult]:
        """
        Apply reviews without submitting.

        Useful for approving changes that need additional verification.

        Args:
            changes: List of changes to review.
            review_labels: Labels to apply.
            dry_run: If True, simulate without making changes.

        Returns:
            List of results indicating review success/failure.
        """
        if review_labels is None:
            review_labels = {"Code-Review": 2}

        results: list[GerritSubmitResult] = []

        for change in changes:
            start_time = time.time()

            if dry_run:
                log.info(
                    "[DRY RUN] Would review %s #%d with %s",
                    change.project,
                    change.number,
                    review_labels,
                )
                results.append(
                    GerritSubmitResult.success_result(
                        change_number=change.number,
                        project=change.project,
                        reviewed=True,
                        submitted=False,
                        duration=time.time() - start_time,
                    )
                )
                continue

            success = self._review_change(change.number, review_labels)
            if success:
                results.append(
                    GerritSubmitResult.success_result(
                        change_number=change.number,
                        project=change.project,
                        reviewed=True,
                        submitted=False,
                        duration=time.time() - start_time,
                    )
                )
            else:
                results.append(
                    GerritSubmitResult.failure_result(
                        change_number=change.number,
                        project=change.project,
                        error="Failed to apply review",
                        duration=time.time() - start_time,
                    )
                )

        return results

    def get_submit_summary(self, results: list[GerritSubmitResult]) -> dict[str, Any]:
        """
        Generate a summary of submit results.

        Args:
            results: List of submit results.

        Returns:
            Dictionary with summary statistics.
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        reviewed = sum(1 for r in results if r.reviewed)
        submitted = sum(1 for r in results if r.submitted)
        total_duration = sum(r.duration_seconds for r in results)

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "reviewed": reviewed,
            "submitted": submitted,
            "total_duration_seconds": round(total_duration, 2),
            "average_duration_seconds": (
                round(total_duration / total, 2) if total > 0 else 0.0
            ),
        }


def create_submit_manager(
    host: str,
    base_path: str | None = None,
    username: str | None = None,
    password: str | None = None,
    max_workers: int = 5,
    progress_tracker: MergeProgressTracker | None = None,
) -> GerritSubmitManager:
    """
    Factory function to create a GerritSubmitManager.

    Args:
        host: Gerrit server hostname.
        base_path: Optional base path.
        username: HTTP username for authentication.
        password: HTTP password for authentication.
        max_workers: Maximum parallel workers.
        progress_tracker: Optional progress tracker.

    Returns:
        Configured GerritSubmitManager instance.
    """
    return GerritSubmitManager(
        host=host,
        base_path=base_path,
        username=username,
        password=password,
        max_workers=max_workers,
        progress_tracker=progress_tracker,
    )


__all__ = [
    "GerritSubmitManager",
    "SubmitStatus",
    "create_submit_manager",
]
