# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests that a finished run does not claim to still be working.

The live display is non-transient, so the last frame stays on the
terminal after the run ends.  A 16-PR owner-wide run left this behind:

    ▶️ Merging PRs in lfreleng-actions (16/16 PRs, 100%)
       Merging PR 162 in lfreleng-actions/python-dynamic-version-action
       ⬆️ Rebased: 3 | 📣 Retriggered: 1 | ✅ Merged: 7 | ❌ Failed: 9

100% complete, every outcome counted, and still naming a pull request
as being merged.

It appeared intermittently because two writers race for the field: the
per-PR merge label never clears, and the wait ticker clears only when it
wrote last.  Whether a finished run told the truth was therefore a
matter of timing, which is what makes ``stop`` --- the one place that
knows the run has ended --- the right place to settle it.
"""

from __future__ import annotations

import pytest

from dependamerge.progress_tracker import MergeProgressTracker, ProgressTracker


def _rendered(tracker) -> str:
    """The text of the frame the terminal is left showing."""
    renderable = tracker._generate_display_text()
    return getattr(renderable, "plain", None) or str(renderable)


class TestStoppingClearsTheOperation:
    """Whoever wrote last, a stopped run advertises no work."""

    def test_the_merge_label_does_not_survive(self) -> None:
        tracker = MergeProgressTracker("lfreleng-actions")
        tracker.set_total_prs(16)
        tracker.update_operation("Merging PR 162 in org/python-dynamic-version-action")
        tracker.merge_success("org/repo#1")

        tracker.stop()

        assert tracker.current_operation == ""
        assert "Merging PR 162" not in _rendered(tracker)

    def test_the_scan_label_does_not_survive(self) -> None:
        tracker = ProgressTracker("lfreleng-actions")
        tracker.total_repositories = 5
        tracker.completed_repositories = 5
        tracker.update_operation("Scanning some-repo...")

        tracker.stop()

        assert tracker.current_operation == ""
        assert "Scanning some-repo" not in _rendered(tracker)

    def test_the_counts_are_untouched(self) -> None:
        """Clearing the label must not disturb what the run achieved."""
        tracker = MergeProgressTracker("lfreleng-actions")
        tracker.set_total_prs(2)
        tracker.merge_success("org/repo#1")
        tracker.merge_failure("org/repo#2")
        tracker.update_operation("Merging PR 2 in org/repo")

        tracker.stop()

        assert tracker.prs_merged == 1
        assert tracker.prs_failed == 1
        assert tracker.completed_prs == 2


class TestEitherWriterLeavesItClean:
    """The bug was a race, so both orderings are pinned.

    The ticker's own shutdown clear stays as it is -- it keeps the
    countdown from sticking *during* a run -- but it can no longer be
    the only thing standing between a finished run and a false claim.
    """

    @pytest.mark.parametrize(
        "last_write",
        ["Merging PR 7 in org/repo", "", "Waiting on checks for org/repo#7"],
    )
    def test_whatever_wrote_last(self, last_write: str) -> None:
        tracker = MergeProgressTracker("lfreleng-actions")
        tracker.set_total_prs(1)
        tracker.update_operation("Merging PR 7 in org/repo")
        tracker.update_operation(last_write)

        tracker.stop()

        assert tracker.current_operation == ""


class TestStoppingTwiceIsSafe:
    """Teardown runs from a ``finally``, so it can be reached twice."""

    def test_a_second_stop_does_not_raise(self) -> None:
        tracker = MergeProgressTracker("lfreleng-actions")
        tracker.update_operation("Merging PR 1 in org/repo")

        tracker.stop()
        tracker.stop()

        assert tracker.current_operation == ""


class TestTheFallbackRepaints:
    """Clearing the field does not erase what is already on screen.

    Without Rich the display is drawn in place with a carriage return,
    so the operation text stays on the terminal until something
    overwrites it. Teardown has to repaint, not just forget.
    """

    @staticmethod
    def _plain_tracker(monkeypatch) -> ProgressTracker:
        tracker = ProgressTracker("lfreleng-actions")
        tracker.rich_available = False
        tracker.live = None
        tracker.total_repositories = 5
        tracker.completed_repositories = 5
        monkeypatch.setattr(tracker, "_stdout_is_tty", lambda: True)
        return tracker

    def test_the_final_paint_omits_the_operation(self, monkeypatch, capsys) -> None:
        tracker = self._plain_tracker(monkeypatch)
        tracker.update_operation("Merging PR 162 in org/repo")
        tracker._fallback_display()
        assert "Merging PR 162" in capsys.readouterr().out

        tracker.stop()

        # The repaint is the last thing written, so the terminal ends
        # showing progress without a live operation.
        final = capsys.readouterr().out
        assert final
        assert "Merging PR 162" not in final
        assert "repos (100%)" in final

    def test_nothing_is_repainted_if_nothing_was_drawn(
        self, monkeypatch, capsys
    ) -> None:
        """A tracker that never displayed must not emit on teardown."""
        tracker = self._plain_tracker(monkeypatch)

        tracker.stop()

        assert capsys.readouterr().out == ""
