# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for reducing an error to one line of report output.

An owner-wide run against ``lfreleng-actions`` reported this, and the
continuation line is not the terminal wrapping --- it starts at column
zero because the reason carries a newline the report never split on::

       • https://github.com/lfreleng-actions/hw-bom-javascript/pull/323
         Failed to approve PR lfreleng-actions/hw-bom-javascript#323: ...
    For more information check: https://developer.mozilla.org/.../500

The failure was real: ``POST .../reviews`` returned 500 on all three
attempts of the approve retry.  Only the second line is at issue, and it
is httpx explaining what 500 means.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from dependamerge.cli._merge_report import _format_failure_reason
from dependamerge.close_manager import AsyncCloseManager, CloseStatus
from dependamerge.error_text import summarise_error
from dependamerge.models import PullRequestInfo

#: The approval failure as the run reported it, reconstructed through
#: the same two layers that built it: httpx's message, wrapped by
#: ``_approve_pr``'s ``RuntimeError``.
REVIEWS_URL = (
    "https://api.github.com/repos/lfreleng-actions/hw-bom-javascript/pulls/323/reviews"
)


def _httpx_500() -> httpx.HTTPStatusError:
    """The exception httpx raises, with the message it builds itself."""
    request = httpx.Request("POST", REVIEWS_URL)
    response = httpx.Response(status_code=500, request=request)
    return httpx.HTTPStatusError(
        f"Server error '500 Internal Server Error' for url '{REVIEWS_URL}'\n"
        "For more information check: "
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500",
        request=request,
        response=response,
    )


def _approval_reason() -> str:
    """What ``_approve_pr`` hands on, exactly as the run reported it."""
    return (
        f"Failed to approve PR lfreleng-actions/hw-bom-javascript#323: {_httpx_500()}"
    )


class TestTheObservedFailureBecomesOneLine:
    """The run that prompted this."""

    def test_the_reported_reason_is_a_single_line(self) -> None:
        assert "\n" not in summarise_error(_approval_reason())

    def test_the_mdn_link_is_gone(self) -> None:
        assert "developer.mozilla.org" not in summarise_error(_approval_reason())
        assert "For more information check" not in summarise_error(_approval_reason())

    def test_the_status_line_survives_whole(self) -> None:
        """Everything the reader or a classifier might want is kept."""
        assert summarise_error(_approval_reason()) == (
            "Failed to approve PR lfreleng-actions/hw-bom-javascript#323: "
            f"Server error '500 Internal Server Error' for url '{REVIEWS_URL}'"
        )

    def test_the_report_indents_every_line_it_prints(self) -> None:
        """The defect as the operator met it, at the sink that showed it.

        ``_print_failed_pr_details`` indents per element it is handed, so
        a reason that is one element containing a newline had only its
        first line indented.
        """
        lines = _format_failure_reason(_approval_reason())

        body = "\n".join(f"     {line}" for line in lines)
        assert all(line.startswith("     ") for line in body.splitlines())


class TestWhatIsKept:
    """Only the trailer is disposable."""

    def test_an_exception_may_be_passed_instead_of_its_text(self) -> None:
        assert summarise_error(_httpx_500()) == (
            f"Server error '500 Internal Server Error' for url '{REVIEWS_URL}'"
        )

    def test_a_message_without_a_trailer_is_untouched(self) -> None:
        reason = "blocked by required status check: pre-commit.ci - pr"
        assert summarise_error(reason) == reason

    def test_internal_spacing_is_left_alone(self) -> None:
        """Only newlines are collapsed, not every whitespace run."""
        assert summarise_error("GitHub:  Base branch  was modified") == (
            "GitHub:  Base branch  was modified"
        )

    def test_githubs_own_body_is_kept(self) -> None:
        """The most specific thing a run learns about a rejection."""
        summary = summarise_error(
            "Client error '405 Method Not Allowed' for url 'https://api/x'\n"
            "For more information check: "
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/405\n"
            "GitHub: Base branch was modified. Review and try the merge again."
        )

        assert summary == (
            "Client error '405 Method Not Allowed' for url 'https://api/x' "
            "GitHub: Base branch was modified. Review and try the merge again."
        )

    def test_the_older_httpx_trailer_goes_too(self) -> None:
        """httpx pointed at httpstatuses.com before 0.21."""
        summary = summarise_error(
            "Server error '502 Bad Gateway' for url 'https://api/x'\n"
            "For more information check: https://httpstatuses.com/502"
        )

        assert summary == "Server error '502 Bad Gateway' for url 'https://api/x'"


class TestTheEdgesDoNotMisbehave:
    """Inputs that are not the observed one."""

    def test_a_bare_newline_becomes_a_space(self) -> None:
        assert summarise_error("first\nsecond") == "first second"

    def test_repeated_newlines_do_not_multiply_spaces(self) -> None:
        assert summarise_error("first\n\n  second") == "first second"

    def test_it_is_idempotent(self) -> None:
        """A reason may be composed from an exception already reduced."""
        once = summarise_error(_approval_reason())
        assert summarise_error(once) == once

    @pytest.mark.parametrize("reason", ["", "\n", "   \n  "])
    def test_an_empty_reason_stays_empty(self, reason: str) -> None:
        assert summarise_error(reason) == ""

    def test_prose_mentioning_information_is_not_eaten(self) -> None:
        """The trailer is recognised by its own wording, not loosely."""
        reason = "for more information see the run log"
        assert summarise_error(reason) == reason

    def test_the_phrase_without_a_url_keeps_every_word(self) -> None:
        """A URL is required, or the phrase eats the word after it.

        Matching an optional ``\\S*`` would delete ``run`` here and leave
        ``logs for details`` behind --- a partial deletion, which is
        worse than either keeping the line or dropping it whole.
        """
        assert summarise_error(
            "Something failed\nFor more information check: run logs for details"
        ) == ("Something failed For more information check: run logs for details")


class TestTheRuleViolationPathIsUnaffected:
    """The formatter's parsing branch must keep its behaviour."""

    def test_a_ruleset_rejection_still_becomes_bullets(self) -> None:
        lines = _format_failure_reason(
            "Repository rule violations found: Required status check "
            '"pre-commit.ci - pr" is failing.'
        )

        assert len(lines) > 1
        assert any(line.startswith("• ") for line in lines)


class TestTheClosePathReportsItsOwnReason:
    """The sinks the merge report does not cover.

    The close path reports its own failures and stores the same string,
    and nothing later passes a ``CloseResult`` through
    ``_format_failure_reason``.  It has two such sinks: the retry loop
    reports the attempt that exhausted it, and ``_record_unexpected_error``
    reports anything raised outside that loop.  Both are driven through
    ``_close_single_pr`` rather than called directly, so the tests fail
    if the route to either changes.
    """

    @staticmethod
    def _pr() -> PullRequestInfo:
        repo = "lfreleng-actions/hw-bom-javascript"
        return PullRequestInfo(
            number=323,
            title="t",
            body=None,
            author="dependabot[bot]",
            head_sha="a" * 40,
            base_branch="main",
            head_branch="x",
            state="open",
            mergeable=True,
            mergeable_state="clean",
            behind_by=None,
            files_changed=[],
            repository_full_name=repo,
            html_url=f"https://github.com/{repo}/pull/323",
            reviews=[],
            review_comments=[],
        )

    @staticmethod
    def _mgr() -> tuple[AsyncCloseManager, MagicMock]:
        """One attempt, so the retry loop exhausts without sleeping."""
        mgr = AsyncCloseManager(token="test-token", max_retries=1)
        console = MagicMock()
        mgr._console = console
        return mgr, console

    @staticmethod
    def _assert_clean(reason: str | None, console: MagicMock) -> None:
        assert reason is not None
        assert "\n" not in reason
        assert "developer.mozilla.org" not in reason
        assert "500 Internal Server Error" in reason

        printed = " ".join(
            str(call.args[0]) for call in console.print.call_args_list if call.args
        )
        assert "developer.mozilla.org" not in printed

    @pytest.mark.asyncio
    async def test_an_exhausted_retry_reports_one_line(self) -> None:
        """The close request itself failing, which is the usual route."""
        mgr, console = self._mgr()
        client = AsyncMock()
        client.close_pull_request = AsyncMock(side_effect=_httpx_500())
        mgr._github_client = client

        result = await mgr._close_single_pr(self._pr())

        assert result.status is CloseStatus.FAILED
        # The retry loop reports its own, so the outer handler's prefix
        # must be absent --- otherwise this test proves nothing the one
        # below does not.
        assert result.error is not None
        assert not result.error.startswith("Unexpected error:")
        self._assert_clean(result.error, console)

    @pytest.mark.asyncio
    async def test_an_error_outside_the_retry_loop_reports_one_line(self) -> None:
        """The outer handler, reached by anything the loop cannot catch."""
        mgr, console = self._mgr()
        mgr._github_client = AsyncMock()
        mgr._close_with_retries = AsyncMock(side_effect=_httpx_500())  # type: ignore[method-assign]

        result = await mgr._close_single_pr(self._pr())

        assert result.status is CloseStatus.FAILED
        assert result.error is not None
        assert result.error.startswith("Unexpected error:")
        self._assert_clean(result.error, console)
