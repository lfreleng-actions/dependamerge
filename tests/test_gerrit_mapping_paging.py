# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""GitHub2Gerrit mapping detection over every page of comments.

A mapping beyond the first page is found, and a reading that could not
cover every page leaves the answer unknown: in ``submit`` and ``skip``
modes the PR is skipped rather than merged on GitHub.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dependamerge.github_async import GitHubAsync, IssueComments
from dependamerge.merge_manager import AsyncMergeManager, MergeStatus
from dependamerge.models import PullRequestInfo

MAPPING_BODY = (
    "<!-- github2gerrit:change-id-map v1 -->\n"
    "PR: https://github.com/lfit/releng-gerrit_to_platform/pull/41\n"
    "Mode: squash\n"
    "Topic: GH-releng-gerrit_to_platform-41\n"
    "Change-Ids:\n"
    "  I6a9987bd1b1cf1e4975dd5da2fb26b6b35ee0048\n"
    "GitHub-Hash: 41b89b8d5055be4e\n"
    "<!-- end github2gerrit:change-id-map -->"
)


def _pr() -> PullRequestInfo:
    return PullRequestInfo(
        number=41,
        title="Bump foo from 1.0 to 2.0",
        body="Dependabot bump body",
        author="dependabot[bot]",
        head_sha="abc123",
        base_branch="main",
        head_branch="dependabot/pip/foo-2.0",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name="lfit/releng-gerrit_to_platform",
        html_url="https://github.com/lfit/releng-gerrit_to_platform/pull/41",
    )


def _manager(mode: str, reading: IssueComments, *, preview: bool) -> AsyncMergeManager:
    mgr = AsyncMergeManager(
        token="test-token", github2gerrit_mode=mode, preview_mode=preview
    )
    client = AsyncMock()
    client.list_issue_comments = AsyncMock(return_value=reading)
    mgr._github_client = client
    mgr._console = MagicMock()
    mgr._check_merge_requirements = AsyncMock(  # type: ignore[method-assign]
        return_value=(True, "mocked")
    )
    return mgr


def test_mapping_found_on_second_page() -> None:
    gh = GitHubAsync(token="fake-token")
    filler = [{"body": f"comment {i}"} for i in range(100)]

    async def pages(path: str, **_: Any) -> AsyncIterator[Any]:
        yield filler
        yield [{"user": {"login": "github-actions[bot]"}, "body": MAPPING_BODY}]

    gh.get_paginated = pages  # type: ignore[method-assign]
    mgr = AsyncMergeManager(token="test-token")
    mgr._github_client = gh

    result = asyncio.run(mgr._detect_github2gerrit("o", "r", 41))

    assert result.has_mapping is True
    assert result.mapping is not None
    assert result.mapping.topic == "GH-releng-gerrit_to_platform-41"


def test_complete_reading_without_mapping_merges() -> None:
    mgr = _manager(
        "submit", IssueComments([{"body": "Bumps foo"}], complete=True), preview=True
    )

    result = asyncio.run(mgr._merge_single_pr(_pr()))

    assert result.status == MergeStatus.MERGED


@pytest.mark.parametrize("mode", ["submit", "skip"])
@pytest.mark.parametrize(
    "reading",
    [
        IssueComments([{"body": "Bumps foo"}], complete=False),
        None,
    ],
    ids=["incomplete", "raised"],
)
def test_incomplete_reading_skips(
    mode: str, reading: IssueComments | None, caplog: pytest.LogCaptureFixture
) -> None:
    mgr = _manager(mode, reading or IssueComments([]), preview=False)
    if reading is None:
        mgr._github_client.list_issue_comments = AsyncMock(  # type: ignore[union-attr,method-assign]
            side_effect=RuntimeError("502")
        )
    mgr._merge_pr_with_retry = AsyncMock(return_value=True)  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(mgr._merge_single_pr(_pr()))

    assert result.status == MergeStatus.SKIPPED
    assert "rule out a GitHub2Gerrit mapping" in (result.error or "")
    mgr._merge_pr_with_retry.assert_not_called()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("rule out a GitHub2Gerrit mapping" in r.getMessage() for r in warnings)


def test_ignore_mode_does_not_consult_comments() -> None:
    mgr = _manager("ignore", IssueComments([], complete=False), preview=True)

    result = asyncio.run(mgr._merge_single_pr(_pr()))

    assert result.status == MergeStatus.MERGED
    mgr._github_client.list_issue_comments.assert_not_called()  # type: ignore[union-attr]


def _mapping_then_failed_page() -> GitHubAsync:
    """A client whose first page holds a mapping and whose second fails."""
    gh = GitHubAsync(token="fake-token")
    first: list[dict[str, Any]] = [{"body": f"comment {i}"} for i in range(99)]
    first.append({"user": {"login": "github-actions[bot]"}, "body": MAPPING_BODY})

    async def pages(path: str, **_: Any) -> AsyncIterator[Any]:
        yield first
        raise RuntimeError("502 Bad Gateway")

    gh.get_paginated = pages  # type: ignore[method-assign]
    return gh


def test_mapping_in_incomplete_reading_is_flagged() -> None:
    mgr = AsyncMergeManager(token="test-token")
    mgr._github_client = _mapping_then_failed_page()

    result = asyncio.run(mgr._detect_github2gerrit("o", "r", 41))

    assert result.has_mapping is True
    assert result.complete is False


@pytest.mark.parametrize("preview", [False, True], ids=["live", "preview"])
def test_submit_mode_does_not_submit_mapping_from_incomplete_reading(
    preview: bool, caplog: pytest.LogCaptureFixture
) -> None:
    mgr = _manager("submit", IssueComments([]), preview=preview)
    mgr._github_client = _mapping_then_failed_page()
    mgr._submit_gerrit_change = AsyncMock(return_value=True)  # type: ignore[method-assign]
    mgr._merge_pr_with_retry = AsyncMock(return_value=True)  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(mgr._merge_single_pr(_pr()))

    assert result.status == MergeStatus.SKIPPED
    assert "confirm the latest GitHub2Gerrit mapping" in (result.error or "")
    mgr._submit_gerrit_change.assert_not_called()
    mgr._merge_pr_with_retry.assert_not_called()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("latest GitHub2Gerrit mapping" in r.getMessage() for r in warnings)


def test_skip_mode_skips_mapping_from_incomplete_reading() -> None:
    mgr = _manager("skip", IssueComments([]), preview=False)
    mgr._github_client = _mapping_then_failed_page()
    mgr._merge_pr_with_retry = AsyncMock(return_value=True)  # type: ignore[method-assign]

    result = asyncio.run(mgr._merge_single_pr(_pr()))

    assert result.status == MergeStatus.SKIPPED
    mgr._merge_pr_with_retry.assert_not_called()
