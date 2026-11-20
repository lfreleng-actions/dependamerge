# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""The ``@dependabot recreate`` guard reads every page of comments."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.github_async import GitHubAsync, IssueComments
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager


def _pr() -> PullRequestInfo:
    return PullRequestInfo(
        number=7,
        title="Bump foo",
        body="bump",
        author="dependabot[bot]",
        head_sha="abc",
        base_branch="main",
        head_branch="dependabot/pip/foo",
        state="open",
        mergeable=True,
        mergeable_state="blocked",
        behind_by=0,
        files_changed=[],
        repository_full_name="o/r",
        html_url="https://github.com/o/r/pull/7",
    )


@pytest.mark.asyncio
async def test_recreate_comment_on_second_page_suppresses() -> None:
    gh = GitHubAsync(token="fake-token")

    async def pages(path: str, **_: Any) -> AsyncIterator[Any]:
        yield [{"body": f"comment {i}"} for i in range(100)]
        yield [{"body": "@dependabot recreate"}]

    gh.get_paginated = pages  # type: ignore[method-assign]
    mgr, _client = make_merge_manager()
    mgr._github_client = gh

    assert await mgr._recreate_already_requested("o", "r", _pr()) is True


@pytest.mark.asyncio
async def test_complete_reading_without_recreate_allows() -> None:
    mgr, client = make_merge_manager()
    client.list_issue_comments = AsyncMock(
        return_value=IssueComments([{"body": "hello"}])
    )

    assert await mgr._recreate_already_requested("o", "r", _pr()) is False


@pytest.mark.asyncio
async def test_incomplete_reading_suppresses() -> None:
    mgr, client = make_merge_manager()
    client.list_issue_comments = AsyncMock(
        return_value=IssueComments([{"body": "hello"}], complete=False)
    )

    assert await mgr._recreate_already_requested("o", "r", _pr()) is True
