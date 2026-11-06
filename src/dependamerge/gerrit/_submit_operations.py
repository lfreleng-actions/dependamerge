# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The per-change review and submit calls behind the submit manager.

:class:`_GerritSubmitOperationMixin` carries the REST round-trips that
act on a single change: applying a review (vote), submitting, the
review-then-submit pipeline that combines them, and the review-only bulk
variant.

It lives here rather than in ``dependamerge.gerrit.submit_manager``
purely to keep that module reviewable.  Nothing in here references
``build_client``: that name is only resolved in ``submit_manager``'s own
namespace, so that patching it there stays effective.  Every attribute
this mixin reads is established by ``GerritSubmitManager.__init__``.
The logger name is unchanged so records keep reporting as
``dependamerge.gerrit.submit_manager``.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import logging
import time
from collections.abc import Mapping

from dependamerge.gerrit.client import GerritRestClient, GerritRestError
from dependamerge.gerrit.models import GerritChangeInfo, GerritSubmitResult

from ._submit_results import _GerritSubmitResultMixin
from ._submitted_together import submitted_together_problems

log = logging.getLogger("dependamerge.gerrit.submit_manager")


class _GerritSubmitOperationMixin(_GerritSubmitResultMixin):
    """Review and submit calls shared into ``GerritSubmitManager``."""

    # Established by GerritSubmitManager.__init__.
    _client: GerritRestClient

    def _submit_single_change(
        self,
        change: GerritChangeInfo,
        review_labels: dict[str, int],
        dry_run: bool,
        bind_revision: bool = False,
        scope: Mapping[int, str] | None = None,
    ) -> GerritSubmitResult:
        """
        Submit a single change (review + submit).

        Args:
            change: The change to submit.
            review_labels: Labels to apply.
            dry_run: If True, simulate without making changes.
            bind_revision: If True, act only on ``change.current_revision``,
                the patchset the caller previewed: refuse when the change
                has a newer one, and review and submit that revision
                rather than whatever is current when the call lands.
            scope: When given, each change number the batch submits
                mapped to its previewed revision; the change is refused
                before any vote when Gerrit would submit anything else
                with it (see ``submitted_together_problems``).

        Returns:
            GerritSubmitResult indicating success or failure.
        """
        start_time = time.time()
        reviewed = False

        precheck = self._precheck_submit(change, dry_run, start_time)
        if precheck is not None:
            return precheck

        revision: str | None = None
        try:
            if bind_revision:
                revision = change.current_revision
                refusal = self._bound_refusal(change, scope)
                if refusal is not None:
                    return GerritSubmitResult.failure_result(
                        change_number=change.number,
                        project=change.project,
                        error=refusal,
                        reviewed=False,
                        duration=time.time() - start_time,
                    )

            review_success = self._review_change(
                change.number, review_labels, revision=revision
            )
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

            submit_success = self._submit_change(change.number, revision=revision)
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

    def _bound_refusal(
        self, change: GerritChangeInfo, scope: Mapping[int, str] | None
    ) -> str | None:
        """Why *change* may not be voted on at its previewed revision, if so.

        Refuses a change with a newer patchset and, given *scope*, one
        Gerrit would submit together with anything outside it.
        """
        refusal = self._stale_revision_error(change)
        if refusal is None and scope is not None:
            problems = submitted_together_problems(self._client, change.number, scope)
            refusal = "; ".join(problems) or None
        return refusal

    def _review_in_scope(
        self,
        change: GerritChangeInfo,
        review_labels: dict[str, int],
        dry_run: bool,
        scope: Mapping[int, str],
        start_time: float,
    ) -> GerritSubmitResult | None:
        """Vote on *change* at its previewed revision, without submitting.

        The first phase of a scoped batch (see ``_submit_scoped_batch``).

        Returns:
            None once the vote is in and the change awaits its submit; a
            final result when it stops here (refused, a failed vote, or a
            dry run).
        """
        precheck = self._precheck_submit(change, dry_run, start_time)
        if precheck is not None:
            return precheck
        try:
            refusal = self._bound_refusal(change, scope)
            if refusal is None and not self._review_change(
                change.number, review_labels, revision=change.current_revision
            ):
                refusal = "Failed to apply review"
        except Exception as exc:
            return self._submit_error_result(change, exc, False, start_time)
        if refusal is not None:
            return GerritSubmitResult.failure_result(
                change_number=change.number,
                project=change.project,
                error=refusal,
                reviewed=False,
                duration=time.time() - start_time,
            )
        log.info(
            "Applied review to %s #%d: %s",
            change.project,
            change.number,
            review_labels,
        )
        return None

    def _submit_reviewed(
        self,
        change: GerritChangeInfo,
        scope: Mapping[int, str],
        start_time: float,
    ) -> GerritSubmitResult:
        """Submit a voted change at its previewed revision.

        The second phase of a scoped batch.  The checks made before the
        vote run again first: while the rest of the batch was voted on, a
        change may have joined the group Gerrit submits with this one, or
        a member may have a new patchset, and the revision-bound submit
        guards only this change.  ``_reconcile_merged`` then settles a
        refused submit.
        """
        try:
            refusal = self._bound_refusal(change, scope)
            submitted = refusal is None and self._submit_change(
                change.number, revision=change.current_revision
            )
        except Exception as exc:
            return self._submit_error_result(change, exc, True, start_time)
        if refusal is not None:
            return GerritSubmitResult.failure_result(
                change_number=change.number,
                project=change.project,
                error=f"{refusal}; voted on but not submitted",
                reviewed=True,
                duration=time.time() - start_time,
            )
        if not submitted:
            return GerritSubmitResult.failure_result(
                change_number=change.number,
                project=change.project,
                error="Failed to submit (change may not be submittable)",
                reviewed=True,
                duration=time.time() - start_time,
            )
        log.info("Submitted %s #%d", change.project, change.number)
        return GerritSubmitResult.success_result(
            change_number=change.number,
            project=change.project,
            reviewed=True,
            submitted=True,
            duration=time.time() - start_time,
        )

    def _reconcile_merged(
        self,
        change: GerritChangeInfo,
        result: GerritSubmitResult,
        start_time: float,
    ) -> GerritSubmitResult:
        """Count a failed change as done when it merged anyway.

        Run once every submit of the batch has returned: another change
        of the batch may have submitted this one together with itself,
        and Gerrit then reports it merged at its previewed revision.
        This holds whether or not this run's vote on it landed, so the
        result keeps ``reviewed`` as it was.
        """
        if result.success:
            return result
        try:
            data = self._client.get(f"/changes/{change.number}?o=CURRENT_REVISION")
        except Exception as exc:
            log.warning("Could not recheck change %d: %s", change.number, exc)
            return result
        merged = (
            isinstance(data, dict)
            and data.get("status") == "MERGED"
            and data.get("current_revision") == change.current_revision
        )
        if not merged:
            return result
        log.info("Submitted %s #%d with its group", change.project, change.number)
        return GerritSubmitResult.success_result(
            change_number=change.number,
            project=change.project,
            reviewed=result.reviewed,
            submitted=True,
            duration=time.time() - start_time,
        )

    def _stale_revision_error(self, change: GerritChangeInfo) -> str | None:
        """Why *change* may not be acted on at its previewed revision, if so.

        Returns None when Gerrit still reports ``change.current_revision``
        as the current patchset.
        """
        if not change.current_revision:
            return "No previewed revision to bind the review and submit to"
        data = self._client.get(f"/changes/{change.number}?o=CURRENT_REVISION")
        current = data.get("current_revision") if isinstance(data, dict) else None
        if current != change.current_revision:
            return (
                "Change has a new patchset since the preview "
                f"({change.current_revision[:10]} -> {str(current)[:10]}); "
                "not reviewed or submitted"
            )
        return None

    def _review_change(
        self,
        change_number: int,
        labels: dict[str, int],
        revision: str | None = None,
    ) -> bool:
        """
        Apply a review (vote) to a change.

        Args:
            change_number: The change number.
            labels: Labels to apply (e.g., {"Code-Review": 2}).
            revision: The revision to vote on; the current one when None.

        Returns:
            True if successful, False otherwise.
        """
        target = revision or "current"
        endpoint = f"/changes/{change_number}/revisions/{target}/review"
        payload = {"labels": labels}

        try:
            self._client.post(endpoint, data=payload)
            return True
        except GerritRestError as exc:
            log.warning("Failed to review change %d: %s", change_number, exc)
            return False

    def _submit_change(self, change_number: int, revision: str | None = None) -> bool:
        """
        Submit a change.

        Args:
            change_number: The change number.
            revision: When given, submit through that revision, which
                Gerrit refuses unless it is still the current patchset.

        Returns:
            True if successful, False otherwise.
        """
        if revision:
            endpoint = f"/changes/{change_number}/revisions/{revision}/submit"
        else:
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
