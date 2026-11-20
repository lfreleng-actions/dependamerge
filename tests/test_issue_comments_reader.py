# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for ``GitHubAsync.list_issue_comments``, the one complete reader.

The issue comments endpoint answers oldest first and ignores any
``direction`` parameter, so a decision about the newest comment has to
read every page.  The reader stops at a ten-page ceiling and says
whether what it returns is the whole story.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from dependamerge.github_async import GitHubAsync, IssueComments


def _page(start: int, size: int) -> list[dict[str, Any]]:
    return [{"id": i, "body": f"comment {i}"} for i in range(start, start + size)]


def _reader(pages: list[Any], fail_after: int | None = None) -> GitHubAsync:
    """A client whose paginator yields *pages*, raising after *fail_after*."""
    gh = GitHubAsync(token="fake-token")
    calls: list[dict[str, Any]] = []

    async def fake_paginated(path: str, **kwargs: Any) -> AsyncIterator[Any]:
        calls.append({"path": path, **kwargs})
        for index, page in enumerate(pages):
            if fail_after is not None and index == fail_after:
                raise RuntimeError("502 Bad Gateway")
            yield page

    gh.get_paginated = fake_paginated  # type: ignore[method-assign]
    gh.calls = calls  # type: ignore[attr-defined]
    return gh


@pytest.mark.asyncio
async def test_three_pages_concatenated_in_order() -> None:
    gh = _reader([_page(0, 100), _page(100, 100), _page(200, 7)])

    reading = await gh.list_issue_comments("o", "r", 5)

    assert isinstance(reading, IssueComments)
    assert reading.complete is True
    assert [c["id"] for c in reading.comments] == list(range(207))
    assert gh.calls == [  # type: ignore[attr-defined]
        {
            "path": "/repos/o/r/issues/5/comments",
            "per_page": 100,
            "max_pages": 10,
            "yield_malformed": True,
        }
    ]


@pytest.mark.asyncio
async def test_failing_second_page_returns_first_incomplete() -> None:
    gh = _reader([_page(0, 100), _page(100, 100)], fail_after=1)

    reading = await gh.list_issue_comments("o", "r", 5)

    assert reading.complete is False
    assert [c["id"] for c in reading.comments] == list(range(100))


@pytest.mark.asyncio
async def test_unexpected_payload_is_incomplete() -> None:
    gh = _reader([_page(0, 100), {"message": "Not Found"}])

    reading = await gh.list_issue_comments("o", "r", 5)

    assert reading.complete is False
    assert len(reading.comments) == 100


@pytest.mark.asyncio
async def test_full_tenth_page_is_incomplete() -> None:
    gh = _reader([_page(i * 100, 100) for i in range(10)])

    reading = await gh.list_issue_comments("o", "r", 5)

    assert reading.complete is False
    assert len(reading.comments) == 1000


@pytest.mark.asyncio
async def test_short_tenth_page_is_complete() -> None:
    gh = _reader([_page(i * 100, 100) for i in range(9)] + [_page(900, 3)])

    reading = await gh.list_issue_comments("o", "r", 5)

    assert reading.complete is True
    assert len(reading.comments) == 903


@pytest.mark.asyncio
async def test_transport_stops_at_ceiling_without_direction() -> None:
    """Through the real paginator: ten requests, no ``direction``."""
    gh = GitHubAsync(token="fake-token")
    seen: list[dict[str, Any]] = []
    response = MagicMock()
    response.headers = {"Link": '<https://api.github.com/x?page=99>; rel="next"'}

    async def fake_fetch(method: str, url: str, params: Any = None, **_: Any):
        seen.append(dict(params or {}))
        return response, _page(len(seen) * 100, 100)

    gh._fetch_json = fake_fetch  # type: ignore[method-assign]

    reading = await gh.list_issue_comments("o", "r", 5)

    assert len(seen) == 10
    assert all("direction" not in params for params in seen)
    assert [params["page"] for params in seen] == list(range(1, 11))
    assert reading.complete is False


def _real_paginator(payloads: list[Any]) -> tuple[GitHubAsync, list[int]]:
    """A client whose real paginator fetches *payloads*, then ``[]``."""
    gh = GitHubAsync(token="fake-token")
    pages: list[int] = []
    response = MagicMock()
    response.headers = {"Link": '<https://api.github.com/x?page=99>; rel="next"'}

    async def fake_fetch(method: str, url: str, params: Any = None, **_: Any):
        pages.append(params["page"])
        index = len(pages) - 1
        return response, payloads[index] if index < len(payloads) else []

    gh._fetch_json = fake_fetch  # type: ignore[method-assign]
    return gh, pages


@pytest.mark.asyncio
async def test_empty_object_page_is_incomplete() -> None:
    """``{}`` is not the end of the collection: the tail is unknown."""
    gh, pages = _real_paginator([_page(0, 100), {}])

    reading = await gh.list_issue_comments("o", "r", 5)

    assert pages == [1, 2]
    assert reading.complete is False
    assert len(reading.comments) == 100


@pytest.mark.asyncio
async def test_empty_list_page_ends_a_complete_reading() -> None:
    gh, pages = _real_paginator([_page(0, 100)])

    reading = await gh.list_issue_comments("o", "r", 5)

    assert pages == [1, 2]
    assert reading.complete is True
    assert len(reading.comments) == 100


@pytest.mark.asyncio
async def test_non_object_comment_is_incomplete() -> None:
    page = _page(0, 3)
    page.insert(1, None)  # type: ignore[arg-type]
    gh, pages = _real_paginator([page])

    reading = await gh.list_issue_comments("o", "r", 5)

    assert pages == [1, 2]
    assert reading.complete is False
    assert [c["id"] for c in reading.comments] == [0, 1, 2]
