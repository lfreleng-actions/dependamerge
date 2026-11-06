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

The submit status vocabulary and result construction live in the sibling
module ``_submit_results``, and the per-change review and submit calls
live in ``_submit_operations``; both are mixed back into
``GerritSubmitManager`` here, so this module's surface is unchanged.
``build_client`` is deliberately resolved in *this* module's namespace
only, so that substituting it here is observed by the code that uses it.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypeVar

# The redundant ``as`` aliases below mark deliberate re-exports: every name
# here has always been reachable as
# ``dependamerge.gerrit.submit_manager.<name>``, even where only the sibling
# modules still reference it.
from dependamerge.gerrit.client import GerritAuthError as GerritAuthError
from dependamerge.gerrit.client import GerritRestError as GerritRestError
from dependamerge.gerrit.client import (
    build_client,
)
from dependamerge.gerrit.models import (
    GerritChangeInfo,
    GerritComparisonResult,
    GerritSubmitResult,
)

# The sibling modules below carry the parts of this module that never touch
# build_client.
from ._submit_operations import _GerritSubmitOperationMixin
from ._submit_results import SubmitStatus as SubmitStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from dependamerge.progress_tracker import MergeProgressTracker


log = logging.getLogger("dependamerge.gerrit.submit_manager")

_Result = TypeVar("_Result", GerritSubmitResult, GerritSubmitResult | None)


class GerritSubmitManager(_GerritSubmitOperationMixin):
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
        bind_revisions: bool = False,
        scope: Mapping[int, str] | None = None,
    ) -> list[GerritSubmitResult]:
        """
        Submit multiple changes sequentially.

        Args:
            changes: List of (change, comparison_result) tuples.
            review_labels: Labels to apply (default: {"Code-Review": 2}).
            dry_run: If True, simulate operations without making changes.
            bind_revisions: If True, act on each change only at the
                revision it carries, refusing one with a newer patchset.
            scope: With ``bind_revisions``, each change number the batch
                submits mapped to its revision; a change Gerrit would
                submit with anything else is refused before its vote.
                Every change is then voted on before any is submitted
                (see ``_submit_scoped_batch``).

        Returns:
            List of GerritSubmitResult for each change.
        """
        if review_labels is None:
            review_labels = {"Code-Review": 2}

        if bind_revisions and scope is not None:
            return self._submit_scoped_batch(
                changes, review_labels, dry_run, scope, max_workers=1
            )

        results: list[GerritSubmitResult] = []

        for change, _comparison in changes:
            result = self._submit_with_tracking(
                change, review_labels, dry_run, bind_revisions, scope
            )
            results.append(result)

        return results

    def submit_changes_parallel(
        self,
        changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
        review_labels: dict[str, int] | None = None,
        dry_run: bool = False,
        bind_revisions: bool = False,
        scope: Mapping[int, str] | None = None,
    ) -> list[GerritSubmitResult]:
        """
        Submit multiple changes in parallel.

        Args:
            changes: List of (change, comparison_result) tuples.
            review_labels: Labels to apply (default: {"Code-Review": 2}).
            dry_run: If True, simulate operations without making changes.
            bind_revisions: If True, act on each change only at the
                revision it carries, refusing one with a newer patchset.
            scope: With ``bind_revisions``, each change number the batch
                submits mapped to its revision; a change Gerrit would
                submit with anything else is refused before its vote.
                Every change is then voted on before any is submitted
                (see ``_submit_scoped_batch``).

        Returns:
            List of GerritSubmitResult for each change.
        """
        if review_labels is None:
            review_labels = {"Code-Review": 2}

        if not changes:
            return []

        if bind_revisions and scope is not None:
            return self._submit_scoped_batch(
                changes, review_labels, dry_run, scope, self._max_workers
            )

        # Use ThreadPoolExecutor for parallel execution.  Keep each
        # future paired with its change so an unexpected worker error
        # can still be attributed to the right change in the results
        # (and mapped back to a URL in the final failure recap).
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                (
                    executor.submit(
                        self._submit_with_tracking,
                        change,
                        review_labels,
                        dry_run,
                        bind_revisions,
                        scope,
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

    def _submit_scoped_batch(
        self,
        changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]],
        review_labels: dict[str, int],
        dry_run: bool,
        scope: Mapping[int, str],
        max_workers: int,
    ) -> list[GerritSubmitResult]:
        """Vote on every change of a scoped batch, then submit them.

        Gerrit submits a change's unmerged ancestors with it, and under
        ``change.submitWholeTopic`` its whole topic, so a submit fails
        until every change in that group has its vote.  Reviewing and
        submitting change by change would fail an early submit that a
        later one then completes.  So every change is voted on first;
        then each voted change is submitted, once its revision and its
        submitted-together group are confirmed again against *scope*,
        since either may have moved while the batch was voted on.  Once
        every submit has returned, a change that failed, at its vote or
        its submit, counts as submitted when Gerrit reports it merged at
        its previewed revision: another change of the batch submitted it.

        Returns:
            One result per change, in the order given.
        """
        start_time = time.time()
        batch = [change for change, _comparison in changes]
        tracker = self._progress_tracker
        if tracker is not None:
            for change in batch:
                tracker.track_pr_state(_tracker_key(change), "submitting")

        results = self._run_each(
            lambda c: self._review_in_scope(
                c, review_labels, dry_run, scope, start_time
            ),
            batch,
            max_workers,
        )

        voted = [c for c, r in zip(batch, results, strict=True) if r is None]
        attempts = iter(
            self._run_each(
                lambda c: self._submit_reviewed(c, scope, start_time),
                voted,
                max_workers,
            )
        )
        outcomes = [r if r is not None else next(attempts) for r in results]
        if voted:
            outcomes = [
                self._reconcile_merged(change, outcome, start_time)
                for change, outcome in zip(batch, outcomes, strict=True)
            ]
        self._record_outcomes(batch, outcomes)
        return outcomes

    def _run_each(
        self,
        action: Callable[[GerritChangeInfo], _Result],
        batch: list[GerritChangeInfo],
        max_workers: int,
    ) -> list[_Result | GerritSubmitResult]:
        """Apply *action* to each change on a worker pool, in order.

        An unexpected worker error becomes that change's failure result.
        """
        if not batch:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [(executor.submit(action, change), change) for change in batch]
            results: list[_Result | GerritSubmitResult] = []
            for future, change in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    log.error(
                        "Unexpected error in submit for %s #%d: %s",
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

    def _record_outcomes(
        self,
        batch: list[GerritChangeInfo],
        results: Sequence[GerritSubmitResult | None],
    ) -> None:
        """Record each final result against the progress tracker."""
        tracker = self._progress_tracker
        if tracker is None:
            return
        for change, result in zip(batch, results, strict=True):
            if result is None:
                continue
            if result.success:
                tracker.merge_success(_tracker_key(change))
            else:
                tracker.merge_failure(_tracker_key(change))

    def _submit_with_tracking(
        self,
        change: GerritChangeInfo,
        review_labels: dict[str, int],
        dry_run: bool,
        bind_revision: bool = False,
        scope: Mapping[int, str] | None = None,
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
            result = self._submit_single_change(
                change, review_labels, dry_run, bind_revision, scope
            )
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


def _tracker_key(change: GerritChangeInfo) -> str:
    """The progress tracker's key for *change*."""
    return f"{change.project}#{change.number}"


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
