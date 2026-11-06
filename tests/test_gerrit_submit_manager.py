# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Tests for Gerrit submit manager.

This module tests the GerritSubmitManager class for reviewing and
submitting Gerrit changes, including parallel operations, dry-run mode,
and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from dependamerge.gerrit.client import GerritRestError
from dependamerge.gerrit.models import (
    GerritChangeInfo,
    GerritComparisonResult,
    GerritFileChange,
    GerritSubmitResult,
)
from dependamerge.gerrit.submit_manager import (
    GerritSubmitManager,
    SubmitStatus,
    create_submit_manager,
)


@pytest.fixture
def mock_client():
    """Create a mock Gerrit REST client."""
    client = MagicMock()
    client.is_authenticated = True
    client.post.return_value = {}
    return client


@pytest.fixture
def sample_change():
    """Create a sample open Gerrit change."""
    return GerritChangeInfo(
        number=12345,
        change_id="I1234567890abcdef",
        project="my-project",
        subject="Chore: Bump actions/checkout from 4.1.0 to 4.2.0",
        owner="dependabot",
        branch="main",
        status="NEW",
        submittable=True,
        work_in_progress=False,
        files_changed=[
            GerritFileChange(
                filename=".github/workflows/ci.yml",
                status="M",
                lines_inserted=1,
                lines_deleted=1,
            )
        ],
    )


@pytest.fixture
def wip_change():
    """Create a work-in-progress change."""
    return GerritChangeInfo(
        number=12346,
        change_id="I9876543210fedcba",
        project="my-project",
        subject="WIP: Work in progress",
        owner="developer",
        branch="main",
        status="NEW",
        work_in_progress=True,
    )


@pytest.fixture
def merged_change():
    """Create an already merged change."""
    return GerritChangeInfo(
        number=12347,
        change_id="Iabcdef1234567890",
        project="my-project",
        subject="Already merged",
        owner="developer",
        branch="main",
        status="MERGED",
    )


class TestSubmitStatus:
    """Tests for SubmitStatus enum."""

    def test_pending_status(self):
        """Test PENDING status value."""
        assert SubmitStatus.PENDING.value == "pending"

    def test_submitted_status(self):
        """Test SUBMITTED status value."""
        assert SubmitStatus.SUBMITTED.value == "submitted"

    def test_failed_status(self):
        """Test FAILED status value."""
        assert SubmitStatus.FAILED.value == "failed"

    def test_blocked_status(self):
        """Test BLOCKED status value."""
        assert SubmitStatus.BLOCKED.value == "blocked"


class TestGerritSubmitManagerInit:
    """Tests for GerritSubmitManager initialization."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_basic_init(self, mock_build_client, mock_client):
        """Test basic manager initialization."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        assert manager.host == "gerrit.example.org"
        assert manager.is_authenticated is True

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_init_with_base_path(self, mock_build_client, mock_client):
        """Test initialization with base path."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            base_path="infra",
        )

        assert manager.base_path == "infra"

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_init_without_auth(self, mock_build_client):
        """Test initialization without authentication."""
        mock_client = MagicMock()
        mock_client.is_authenticated = False
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(host="gerrit.example.org")

        assert manager.is_authenticated is False


class TestReviewChange:
    """Tests for the _review_change method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_review_success(self, mock_build_client, mock_client):
        """Test successful review operation."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._review_change(12345, {"Code-Review": 2})

        assert result is True
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/changes/12345/revisions/current/review" in call_args[0][0]

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_review_failure(self, mock_build_client, mock_client):
        """Test failed review operation."""
        from dependamerge.gerrit.client import GerritRestError

        mock_build_client.return_value = mock_client
        mock_client.post.side_effect = GerritRestError("Failed", 500)

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._review_change(12345, {"Code-Review": 2})

        assert result is False


class TestSubmitChange:
    """Tests for the _submit_change method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_success(self, mock_build_client, mock_client):
        """Test successful submit operation."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_change(12345)

        assert result is True
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/changes/12345/submit" in call_args[0][0]

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_failure(self, mock_build_client, mock_client):
        """Test failed submit operation."""
        from dependamerge.gerrit.client import GerritRestError

        mock_build_client.return_value = mock_client
        mock_client.post.side_effect = GerritRestError("Not submittable", 409)

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_change(12345)

        assert result is False


class TestSubmitSingleChange:
    """Tests for the _submit_single_change method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_open_change(self, mock_build_client, mock_client, sample_change):
        """Test submitting an open change."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_single_change(
            sample_change, {"Code-Review": 2}, dry_run=False
        )

        assert result.success is True
        assert result.reviewed is True
        assert result.submitted is True
        assert result.change_number == 12345
        assert result.project == "my-project"

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_wip_change(self, mock_build_client, mock_client, wip_change):
        """Test submitting a WIP change fails."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_single_change(
            wip_change, {"Code-Review": 2}, dry_run=False
        )

        assert result.success is False
        assert "Work In Progress" in result.error

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_merged_change(self, mock_build_client, mock_client, merged_change):
        """Test submitting an already merged change fails."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_single_change(
            merged_change, {"Code-Review": 2}, dry_run=False
        )

        assert result.success is False
        assert "not open" in result.error

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_dry_run(self, mock_build_client, mock_client, sample_change):
        """Test dry run mode."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        result = manager._submit_single_change(
            sample_change, {"Code-Review": 2}, dry_run=True
        )

        assert result.success is True
        # A dry run performs no review or submit, so it must not claim
        # either happened: callers gate real side effects (e.g. closing
        # the GitHub PR after a Gerrit submit) on ``submitted``.
        assert result.reviewed is False
        assert result.submitted is False
        # No actual API calls should be made
        mock_client.post.assert_not_called()


class TestSubmitBoundToRevision:
    """A query batch acts on each change only at its previewed revision."""

    @staticmethod
    def _manager(mock_build_client, mock_client) -> GerritSubmitManager:
        mock_build_client.return_value = mock_client
        return GerritSubmitManager(host="gerrit.example.org", username="user")

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_reviews_and_submits_the_previewed_revision(
        self, mock_build_client, mock_client, sample_change
    ):
        """Review and submit name the revision, not 'current'."""
        change = sample_change.model_copy(update={"current_revision": "abc123"})
        mock_client.get.return_value = {"current_revision": "abc123"}
        manager = self._manager(mock_build_client, mock_client)

        results = manager.submit_changes_parallel([(change, None)], bind_revisions=True)

        assert results[0].success is True
        endpoints = [c[0][0] for c in mock_client.post.call_args_list]
        assert endpoints == [
            "/changes/12345/revisions/abc123/review",
            "/changes/12345/revisions/abc123/submit",
        ]

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_new_patchset_is_neither_reviewed_nor_submitted(
        self, mock_build_client, mock_client, sample_change
    ):
        """A patchset uploaded after the preview is refused before voting."""
        change = sample_change.model_copy(update={"current_revision": "abc123"})
        mock_client.get.return_value = {"current_revision": "def456"}
        manager = self._manager(mock_build_client, mock_client)

        results = manager.submit_changes([(change, None)], bind_revisions=True)

        assert results[0].success is False
        assert "new patchset" in (results[0].error or "")
        mock_client.post.assert_not_called()

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_change_without_a_revision_is_refused(
        self, mock_build_client, mock_client, sample_change
    ):
        """Without a previewed revision there is nothing to bind to."""
        manager = self._manager(mock_build_client, mock_client)

        results = manager.submit_changes([(sample_change, None)], bind_revisions=True)

        assert results[0].success is False
        mock_client.get.assert_not_called()
        mock_client.post.assert_not_called()

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_change_submitting_outside_scope_is_not_voted_on(
        self, mock_build_client, mock_client, sample_change
    ):
        """Gerrit's submitted_together set is checked again before voting."""
        change = sample_change.model_copy(update={"current_revision": "abc123"})
        mock_client.get.side_effect = [
            {"current_revision": "abc123"},
            {
                "changes": [
                    {"_number": 12345, "current_revision": "abc123"},
                    {"_number": 999, "project": "other", "current_revision": "x"},
                ],
                "non_visible_changes": 0,
            },
        ]
        manager = self._manager(mock_build_client, mock_client)

        results = manager.submit_changes_parallel(
            [(change, None)], bind_revisions=True, scope={12345: "abc123"}
        )

        assert results[0].success is False
        assert "#999 (other)" in (results[0].error or "")
        assert "submitted_together" in mock_client.get.call_args[0][0]
        mock_client.post.assert_not_called()

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_change_submitting_within_scope_is_submitted(
        self, mock_build_client, mock_client, sample_change
    ):
        """A set inside the batch, before the vote and the submit, goes ahead."""
        change = sample_change.model_copy(update={"current_revision": "abc123"})
        mock_client.get.side_effect = [
            {"current_revision": "abc123"},
            {"changes": [], "non_visible_changes": 0},
        ] * 2
        manager = self._manager(mock_build_client, mock_client)

        results = manager.submit_changes(
            [(change, None)], bind_revisions=True, scope={12345: "abc123"}
        )

        assert results[0].success is True
        assert mock_client.post.call_count == 2

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_unbound_submission_is_unchanged(
        self, mock_build_client, mock_client, sample_change
    ):
        """A change URL's batch still acts on the current revision."""
        manager = self._manager(mock_build_client, mock_client)

        manager.submit_changes([(sample_change, None)])

        endpoints = [c[0][0] for c in mock_client.post.call_args_list]
        assert endpoints == [
            "/changes/12345/revisions/current/review",
            "/changes/12345/submit",
        ]
        mock_client.get.assert_not_called()


class TestSubmitScopedBatchByGroup:
    """Every change of a scoped batch gets its vote before any submit."""

    SCOPE = {101: "rev101", 102: "rev102", 103: "rev103"}

    @staticmethod
    def _batch(
        sample_change,
    ) -> list[tuple[GerritChangeInfo, GerritComparisonResult | None]]:
        return [
            (
                sample_change.model_copy(
                    update={"number": number, "current_revision": revision}
                ),
                None,
            )
            for number, revision in TestSubmitScopedBatchByGroup.SCOPE.items()
        ]

    @classmethod
    def _gerrit(
        cls,
        mock_client,
        merged: set[int],
        refuse_submit: set[int],
        refuse_review: set[int] | frozenset[int] = frozenset(),
    ):
        """Answer as Gerrit does for a submitted-together group."""

        def get(endpoint: str):
            number = int(endpoint.split("/")[2].split("?")[0])
            if "submitted_together" in endpoint:
                return {
                    "changes": [
                        {"_number": n, "current_revision": r}
                        for n, r in cls.SCOPE.items()
                    ],
                    "non_visible_changes": 0,
                }
            status = "MERGED" if number in merged else "NEW"
            return {"status": status, "current_revision": cls.SCOPE[number]}

        def post(endpoint: str, data=None):
            number = int(endpoint.split("/")[2])
            if endpoint.endswith("/review") and number in refuse_review:
                raise GerritRestError("Vote refused", 403)
            if endpoint.endswith("/submit"):
                if number in refuse_submit:
                    raise GerritRestError("Change is merged", 409)
                merged.update(cls.SCOPE)
            return {}

        mock_client.get.side_effect = get
        mock_client.post.side_effect = post

    @pytest.mark.parametrize("parallel", [True, False])
    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_every_vote_lands_before_the_first_submit(
        self, mock_build_client, mock_client, sample_change, parallel
    ):
        """Under submitWholeTopic a submit fails until the group is voted."""
        mock_build_client.return_value = mock_client
        self._gerrit(mock_client, merged=set(), refuse_submit=set())
        manager = GerritSubmitManager(host="gerrit.example.org", username="user")
        submit = manager.submit_changes_parallel if parallel else manager.submit_changes

        submit(self._batch(sample_change), bind_revisions=True, scope=self.SCOPE)

        kinds = [c[0][0].rsplit("/", 1)[1] for c in mock_client.post.call_args_list]
        first_submit = kinds.index("submit")
        assert kinds[:first_submit] == ["review"] * 3

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_change_merged_with_its_group_counts_as_submitted(
        self, mock_build_client, mock_client, sample_change
    ):
        """A submit refused because the group already merged it succeeds."""
        mock_build_client.return_value = mock_client
        self._gerrit(mock_client, merged=set(), refuse_submit={102, 103})
        manager = GerritSubmitManager(host="gerrit.example.org", username="user")

        results = manager.submit_changes(
            self._batch(sample_change), bind_revisions=True, scope=self.SCOPE
        )

        assert [r.success for r in results] == [True, True, True]
        assert all(r.submitted for r in results)

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_refused_submit_is_rechecked_after_every_submit_returns(
        self, mock_build_client, mock_client, sample_change
    ):
        """A submit refused before a later one merges the group still counts."""
        mock_build_client.return_value = mock_client
        self._gerrit(mock_client, merged=set(), refuse_submit={101})
        manager = GerritSubmitManager(host="gerrit.example.org", username="user")

        results = manager.submit_changes(
            self._batch(sample_change), bind_revisions=True, scope=self.SCOPE
        )

        assert [r.success for r in results] == [True, True, True]

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_failed_vote_on_a_member_merged_with_its_group_succeeds(
        self, mock_build_client, mock_client, sample_change
    ):
        """An already-approved member merged by another submit counts."""
        mock_build_client.return_value = mock_client
        self._gerrit(
            mock_client, merged=set(), refuse_submit=set(), refuse_review={102}
        )
        manager = GerritSubmitManager(host="gerrit.example.org", username="user")

        results = manager.submit_changes_parallel(
            self._batch(sample_change), bind_revisions=True, scope=self.SCOPE
        )

        assert [r.success for r in results] == [True, True, True]
        assert results[1].submitted is True
        assert results[1].reviewed is False
        assert results[0].reviewed is True

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_failed_vote_on_an_unmerged_member_stays_a_failure(
        self, mock_build_client, mock_client, sample_change
    ):
        """A member neither voted on nor merged keeps its vote failure."""
        mock_build_client.return_value = mock_client
        self._gerrit(
            mock_client,
            merged=set(),
            refuse_submit=set(self.SCOPE),
            refuse_review={102},
        )
        manager = GerritSubmitManager(host="gerrit.example.org", username="user")

        results = manager.submit_changes(
            self._batch(sample_change), bind_revisions=True, scope=self.SCOPE
        )

        assert [r.success for r in results] == [False, False, False]
        assert results[1].error == "Failed to apply review"
        assert results[1].reviewed is False

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_refused_submit_of_an_unmerged_change_fails(
        self, mock_build_client, mock_client, sample_change
    ):
        """A change Gerrit neither submitted nor merged stays a failure."""
        mock_build_client.return_value = mock_client
        self._gerrit(mock_client, merged=set(), refuse_submit=set(self.SCOPE))
        manager = GerritSubmitManager(host="gerrit.example.org", username="user")

        results = manager.submit_changes_parallel(
            self._batch(sample_change), bind_revisions=True, scope=self.SCOPE
        )

        assert [r.success for r in results] == [False, False, False]
        assert all(r.reviewed for r in results)

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_change_refused_before_its_vote_is_never_submitted(
        self, mock_build_client, mock_client, sample_change
    ):
        """A change with a newer patchset gets neither a vote nor a submit."""
        mock_build_client.return_value = mock_client
        self._gerrit(mock_client, merged=set(), refuse_submit=set())
        batch = self._batch(sample_change)
        batch[1] = (batch[1][0].model_copy(update={"current_revision": "old"}), None)
        manager = GerritSubmitManager(host="gerrit.example.org", username="user")

        results = manager.submit_changes_parallel(
            batch, bind_revisions=True, scope=self.SCOPE
        )

        assert results[1].success is False
        assert "new patchset" in (results[1].error or "")
        acted_on = {c[0][0].split("/")[2] for c in mock_client.post.call_args_list}
        assert "102" not in acted_on

    @classmethod
    def _gerrit_moving_after_votes(cls, mock_client, group_after, revisions_after):
        """Answer as Gerrit does when the group moves once every vote is in."""
        state = {"votes": 0}

        def moved() -> bool:
            return bool(state["votes"] >= len(cls.SCOPE))

        def get(endpoint: str):
            number = int(endpoint.split("/")[2].split("?")[0])
            revisions = revisions_after if moved() else cls.SCOPE
            if "submitted_together" in endpoint:
                group = group_after if moved() else cls.SCOPE
                return {
                    "changes": [
                        {"_number": n, "current_revision": revisions.get(n, "revX")}
                        for n in group
                    ],
                    "non_visible_changes": 0,
                }
            return {"status": "NEW", "current_revision": revisions[number]}

        def post(endpoint: str, data=None):
            if endpoint.endswith("/review"):
                state["votes"] += 1
            return {}

        mock_client.get.side_effect = get
        mock_client.post.side_effect = post

    @pytest.mark.parametrize("parallel", [True, False])
    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_change_joining_the_group_after_the_votes_blocks_every_submit(
        self, mock_build_client, mock_client, sample_change, parallel
    ):
        """A change entering the topic between vote and submit is not merged."""
        mock_build_client.return_value = mock_client
        self._gerrit_moving_after_votes(
            mock_client,
            group_after=[*self.SCOPE, 999],
            revisions_after=self.SCOPE,
        )
        manager = GerritSubmitManager(host="gerrit.example.org", username="user")
        submit = manager.submit_changes_parallel if parallel else manager.submit_changes

        results = submit(
            self._batch(sample_change), bind_revisions=True, scope=self.SCOPE
        )

        kinds = [c[0][0].rsplit("/", 1)[1] for c in mock_client.post.call_args_list]
        assert kinds == ["review"] * 3
        assert [r.success for r in results] == [False, False, False]
        assert all(r.reviewed for r in results)
        assert all("#999" in (r.error or "") for r in results)

    @pytest.mark.parametrize("parallel", [True, False])
    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_new_patchset_after_the_votes_blocks_every_submit(
        self, mock_build_client, mock_client, sample_change, parallel
    ):
        """A member's new patchset between vote and submit is not merged."""
        mock_build_client.return_value = mock_client
        self._gerrit_moving_after_votes(
            mock_client,
            group_after=list(self.SCOPE),
            revisions_after={**self.SCOPE, 103: "rev103-new"},
        )
        manager = GerritSubmitManager(host="gerrit.example.org", username="user")
        submit = manager.submit_changes_parallel if parallel else manager.submit_changes

        results = submit(
            self._batch(sample_change), bind_revisions=True, scope=self.SCOPE
        )

        kinds = [c[0][0].rsplit("/", 1)[1] for c in mock_client.post.call_args_list]
        assert kinds == ["review"] * 3
        assert [r.success for r in results] == [False, False, False]
        assert "new patchset" in (results[2].error or "")
        assert "#103" in (results[0].error or "")

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_group_is_checked_again_before_each_submit(
        self, mock_build_client, mock_client, sample_change
    ):
        """submitted_together is read after the last vote, before a submit."""
        mock_build_client.return_value = mock_client
        self._gerrit(mock_client, merged=set(), refuse_submit=set())
        manager = GerritSubmitManager(host="gerrit.example.org", username="user")
        calls: list[str] = []
        mock_client.get.side_effect = _recording(mock_client.get.side_effect, calls)
        mock_client.post.side_effect = _recording(mock_client.post.side_effect, calls)

        manager.submit_changes(
            self._batch(sample_change), bind_revisions=True, scope=self.SCOPE
        )

        first_submit = next(i for i, c in enumerate(calls) if c.endswith("/submit"))
        last_vote = max(i for i, c in enumerate(calls) if c.endswith("/review"))
        between = calls[last_vote + 1 : first_submit]
        assert any("/changes/101/submitted_together" in c for c in between)


def _recording(answer, calls: list[str]):
    """Wrap a mock side effect so each endpoint it answers is recorded."""

    def record(endpoint: str, *args, **kwargs):
        calls.append(endpoint)
        return answer(endpoint, *args, **kwargs)

    return record


class TestSubmitChanges:
    """Tests for the submit_changes method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_multiple_changes(self, mock_build_client, mock_client):
        """Test submitting multiple changes sequentially."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]] = [
            (
                GerritChangeInfo(
                    number=i,
                    change_id=f"I{i:040d}",
                    project="proj",
                    subject="Test",
                    owner="bot",
                    branch="main",
                    status="NEW",
                ),
                None,
            )
            for i in range(3)
        ]

        results = manager.submit_changes(changes)

        assert len(results) == 3
        assert all(r.success for r in results)

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_submit_empty_list(self, mock_build_client, mock_client):
        """Test submitting empty list."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = manager.submit_changes([])

        assert results == []


class TestSubmitChangesParallel:
    """Tests for the submit_changes_parallel method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_parallel_submit(self, mock_build_client, mock_client):
        """Test parallel submission of changes."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
            max_workers=3,
        )

        changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]] = [
            (
                GerritChangeInfo(
                    number=i,
                    change_id=f"I{i:040d}",
                    project="proj",
                    subject="Test",
                    owner="bot",
                    branch="main",
                    status="NEW",
                ),
                None,
            )
            for i in range(5)
        ]

        results = manager.submit_changes_parallel(changes)

        assert len(results) == 5
        # All should succeed
        assert all(r.success for r in results)

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_parallel_empty_list(self, mock_build_client, mock_client):
        """Test parallel submit with empty list."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = manager.submit_changes_parallel([])

        assert results == []


class TestProgressTracking:
    """Tests for progress tracker integration during submission.

    The submit manager drives the same tracker protocol as the
    GitHub merge pipeline: a transitory ``submitting`` state while
    the review + submit round-trips run, then a terminal
    ``merge_success`` / ``merge_failure`` outcome keyed by
    ``project#number``.
    """

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_success_records_submitting_then_success(
        self, mock_build_client, mock_client, sample_change
    ):
        """Test a successful submit drives the tracker states."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}
        tracker = MagicMock()

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
            progress_tracker=tracker,
        )

        results = manager.submit_changes([(sample_change, None)])

        assert results[0].success
        tracker.track_pr_state.assert_called_once_with("my-project#12345", "submitting")
        tracker.merge_success.assert_called_once_with("my-project#12345")
        tracker.merge_failure.assert_not_called()

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_failure_records_merge_failure(
        self, mock_build_client, mock_client, sample_change
    ):
        """Test a failed review drives merge_failure on the tracker."""
        mock_build_client.return_value = mock_client
        mock_client.post.side_effect = GerritRestError("boom")
        tracker = MagicMock()

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
            progress_tracker=tracker,
        )

        results = manager.submit_changes([(sample_change, None)])

        assert not results[0].success
        tracker.merge_failure.assert_called_once_with("my-project#12345")
        tracker.merge_success.assert_not_called()

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_parallel_submission_drives_tracker_per_change(
        self, mock_build_client, mock_client
    ):
        """Test parallel submission records one outcome per change."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}
        tracker = MagicMock()

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
            max_workers=2,
            progress_tracker=tracker,
        )

        changes: list[tuple[GerritChangeInfo, GerritComparisonResult | None]] = [
            (
                GerritChangeInfo(
                    number=i,
                    change_id=f"I{i:040d}",
                    project="proj",
                    subject="Test",
                    owner="bot",
                    branch="main",
                    status="NEW",
                ),
                None,
            )
            for i in range(3)
        ]

        results = manager.submit_changes_parallel(changes)

        assert all(r.success for r in results)
        assert tracker.merge_success.call_count == 3
        assert tracker.track_pr_state.call_count == 3

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_no_tracker_is_a_no_op(self, mock_build_client, mock_client, sample_change):
        """Test submission works unchanged without a tracker."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = manager.submit_changes([(sample_change, None)])

        assert results[0].success


class TestReviewOnly:
    """Tests for the review_only method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_review_only_success(self, mock_build_client, mock_client, sample_change):
        """Test review-only operation."""
        mock_build_client.return_value = mock_client
        mock_client.post.return_value = {}

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = manager.review_only([sample_change])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].reviewed is True
        assert results[0].submitted is False

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_review_only_dry_run(self, mock_build_client, mock_client, sample_change):
        """Test review-only with dry run."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = manager.review_only([sample_change], dry_run=True)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].reviewed is True
        assert results[0].submitted is False
        mock_client.post.assert_not_called()


class TestGetSubmitSummary:
    """Tests for the get_submit_summary method."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_summary_all_success(self, mock_build_client, mock_client):
        """Test summary with all successful results."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = [
            GerritSubmitResult.success_result(
                change_number=i,
                project="proj",
                reviewed=True,
                submitted=True,
                duration=1.0,
            )
            for i in range(5)
        ]

        summary = manager.get_submit_summary(results)

        assert summary["total"] == 5
        assert summary["successful"] == 5
        assert summary["failed"] == 0
        assert summary["reviewed"] == 5
        assert summary["submitted"] == 5
        assert summary["total_duration_seconds"] == 5.0
        assert summary["average_duration_seconds"] == 1.0

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_summary_mixed_results(self, mock_build_client, mock_client):
        """Test summary with mixed success/failure results."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        results = [
            GerritSubmitResult.success_result(
                change_number=1,
                project="proj",
                reviewed=True,
                submitted=True,
                duration=1.0,
            ),
            GerritSubmitResult.failure_result(
                change_number=2,
                project="proj",
                error="Failed",
                reviewed=True,
                duration=0.5,
            ),
            GerritSubmitResult.failure_result(
                change_number=3,
                project="proj",
                error="Failed",
                duration=0.5,
            ),
        ]

        summary = manager.get_submit_summary(results)

        assert summary["total"] == 3
        assert summary["successful"] == 1
        assert summary["failed"] == 2
        assert summary["reviewed"] == 2
        assert summary["submitted"] == 1

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_summary_empty_results(self, mock_build_client, mock_client):
        """Test summary with empty results."""
        mock_build_client.return_value = mock_client

        manager = GerritSubmitManager(
            host="gerrit.example.org",
            username="user",
            password="pass",
        )

        summary = manager.get_submit_summary([])

        assert summary["total"] == 0
        assert summary["successful"] == 0
        assert summary["average_duration_seconds"] == 0.0


class TestCreateSubmitManager:
    """Tests for the create_submit_manager factory function."""

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_create_with_defaults(self, mock_build_client, mock_client):
        """Test factory with default parameters."""
        mock_build_client.return_value = mock_client

        manager = create_submit_manager(host="gerrit.example.org")

        assert isinstance(manager, GerritSubmitManager)
        assert manager.host == "gerrit.example.org"

    @patch("dependamerge.gerrit.submit_manager.build_client")
    def test_create_with_all_params(self, mock_build_client, mock_client):
        """Test factory with all parameters."""
        mock_build_client.return_value = mock_client

        manager = create_submit_manager(
            host="gerrit.example.org",
            base_path="infra",
            username="user",
            password="pass",
            max_workers=10,
        )

        assert isinstance(manager, GerritSubmitManager)
        assert manager.host == "gerrit.example.org"
        assert manager.base_path == "infra"
