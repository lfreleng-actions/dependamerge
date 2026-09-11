# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for compacting check names into one line of output.

A 60-PR owner-wide run against ``lfreleng-actions`` produced this for a
single pull request, wrapped by the terminal mid-name:

    blocked by required status check: pre-commit.ci - pr; also failing:
    Build/Test (lfreleng-actions/test-docker-monorepo, v0.0.1, make
    images TAG=verify, TAG=verify tes... / Grype Audit SBOM, Build/Test
    (lfreleng-actions/test-docker-monorepo, v0.0.1, TAG=verify
    tests/scripts/test_images.s... / Grype Audit SBOM, …

827 characters, and unsplittable by eye because the names contain the
same commas used to join them.  Every entry but one is the same check on
a different matrix cell.
"""

from __future__ import annotations

import pytest

from dependamerge.check_names import summarise_check_names

# The names as GitHub reported them, from the run that prompted this.
MATRIX_CELLS = [
    "Build/Test (lfreleng-actions/test-docker-monorepo, v0.0.1, make images "
    "TAG=verify, TAG=verify tes... / Grype Audit SBOM",
    "Build/Test (lfreleng-actions/test-docker-monorepo, v0.0.1, TAG=verify "
    "tests/scripts/test_images.s... / Grype Audit SBOM",
    "Build/Test (lfreleng-actions/test-docker-monorepo, v0.0.1, make images "
    "TAG=verify, base-alpine:ve... / Grype Audit SBOM",
    "Build/Test (lfreleng-actions/test-docker-project, v0.1.0, false, false) "
    "/ Grype Audit SBOM",
    'Build/Test (lfreleng-actions/test-docker-monorepo, v0.0.1, [{"name": '
    '"base-alpine", "context": "b... / Grype Audit SBOM',
    "Build/Test (lfreleng-actions/test-docker-monorepo, v0.0.1, tests, printf "
    '"FROM alpine:3.22@sha256... / Grype Audit SBOM',
    'Build/Test (lfreleng-actions/test-docker-monorepo, v0.0.1, [{"name": '
    '"util-echo", "context": "uti... / Grype Audit SBOM',
    "Testing 🧪",
]


class TestMatrixCellsCollapseToTheCheck:
    """The reported run, which is what made this unreadable."""

    def test_the_observed_names_become_one_line(self) -> None:
        assert summarise_check_names(MATRIX_CELLS) == (
            "Grype Audit SBOM (×7), Testing 🧪"
        )

    def test_the_result_fits_a_terminal_line(self) -> None:
        # The label that introduces it costs about 68 characters, so the
        # summary has to leave room for that and still fit 120 columns.
        assert len(summarise_check_names(MATRIX_CELLS)) < 50

    def test_a_single_cell_is_not_given_a_count(self) -> None:
        """``(×1)`` reads as noise, not information."""
        assert summarise_check_names(["Build (x) / Lint"]) == "Lint"


class TestOrdinaryNamesAreLeftAlone:
    """Most failures are a couple of plainly-named checks."""

    def test_names_without_a_job_prefix_pass_through(self) -> None:
        assert summarise_check_names(["Audit Workflows", "Zizmor Scan 🌈"]) == (
            "Audit Workflows, Zizmor Scan 🌈"
        )

    def test_an_empty_list_is_empty(self) -> None:
        assert summarise_check_names([]) == ""


class TestManyDistinctChecksAreCounted:
    """Twenty names is not more useful than four and a count."""

    def test_the_tail_becomes_a_count(self) -> None:
        assert summarise_check_names([f"check-{i}" for i in range(9)]) == (
            "check-0, check-1, check-2, check-3, +5 more"
        )

    def test_exactly_the_limit_needs_no_count(self) -> None:
        assert summarise_check_names([f"check-{i}" for i in range(4)]) == (
            "check-0, check-1, check-2, check-3"
        )

    def test_grouping_happens_before_the_limit(self) -> None:
        """Otherwise one matrix would exhaust the budget on its own."""
        names = [f"Build ({i}) / Lint" for i in range(20)] + ["Typecheck"]

        assert summarise_check_names(names) == "Lint (×20), Typecheck"


class TestALongNameIsElided:
    """A name no line can hold is cut rather than allowed to wrap."""

    def test_it_is_shortened_with_an_ellipsis(self) -> None:
        out = summarise_check_names(["x" * 100])

        assert len(out) < 60
        assert out.endswith("…")

    def test_a_name_within_the_limit_is_untouched(self) -> None:
        name = "y" * 48

        assert summarise_check_names([name]) == name


class TestFirstSeenOrderIsKept:
    """Re-sorting would make consecutive runs disagree for no reason."""

    def test_order_follows_the_input(self) -> None:
        assert summarise_check_names(["Zebra", "Apple", "Mango"]) == (
            "Zebra, Apple, Mango"
        )


class TestOddInputIsSurvivable:
    """The names come from an API, so they are not guaranteed sensible."""

    @pytest.mark.parametrize("name", ["", "   ", " / "])
    def test_a_nameless_entry_is_dropped(self, name: str) -> None:
        assert summarise_check_names([name, "Real Check"]) == "Real Check"

    def test_a_trailing_separator_keeps_the_job_name(self) -> None:
        """Nothing after the separator means the job name is all there is."""
        assert summarise_check_names(["Build / "]) == "Build"
