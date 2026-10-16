# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""
Tests for tolerating an expected ``NOT_FOUND`` in a GraphQL response.

GitHub answers a nested field that names a missing object, such as an
owner's absent ``.github`` repository, with ``null`` there and an error
naming the field's path, alongside valid data.  A caller that expects
the object may be missing names the path; every other error still
raises exactly as before.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from dependamerge.github_async import GitHubAsync, GraphQLError

_DOTGITHUB = ("repositoryOwner", "dotgithub")
_DATA = {"repositoryOwner": {"__typename": "Organization", "dotgithub": None}}


def _not_found(*path: str, login: str = "acme") -> dict[str, Any]:
    return {
        "type": "NOT_FOUND",
        "path": list(path),
        "message": f"Could not resolve to a Repository with the name '{login}/.github'.",
    }


async def _ask(
    body: dict[str, Any], allow: tuple[tuple[str, ...], ...] = ()
) -> tuple[dict[str, Any], int]:
    """Send one query answered with ``body``; return the data and attempts."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(
            200,
            content=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )

    api = GitHubAsync(token="t")
    api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        data = await api.graphql("query { x }", {}, allow_not_found=allow)
    finally:
        await api._client.aclose()
    return data, attempts["n"]


@pytest.mark.asyncio
async def test_an_allowed_not_found_returns_the_data() -> None:
    body = {"data": _DATA, "errors": [_not_found(*_DOTGITHUB)]}
    data, attempts = await _ask(body, (_DOTGITHUB,))
    assert data == _DATA
    assert attempts == 1


@pytest.mark.asyncio
async def test_without_an_allowance_it_still_raises() -> None:
    body = {"data": _DATA, "errors": [_not_found(*_DOTGITHUB)]}
    with pytest.raises(GraphQLError, match="NOT_FOUND"):
        await _ask(body)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "errors",
    [
        # The same kind of error, somewhere the caller did not expect it.
        [_not_found("repositoryOwner")],
        [_not_found("repositoryOwner", "dotgithub", "policy")],
        # An allowed error does not excuse one beside it.
        [
            _not_found(*_DOTGITHUB),
            {"type": "FORBIDDEN", "path": ["repositoryOwner"], "message": "no"},
        ],
        # Not a NOT_FOUND at all, though at an allowed path.
        [{"type": "FORBIDDEN", "path": list(_DOTGITHUB), "message": "no"}],
    ],
)
async def test_any_other_error_still_raises(errors: list[dict[str, Any]]) -> None:
    with pytest.raises(GraphQLError):
        await _ask({"data": _DATA, "errors": errors}, (_DOTGITHUB,))


@pytest.mark.asyncio
async def test_an_allowed_not_found_without_data_still_raises() -> None:
    body = {"data": None, "errors": [_not_found(*_DOTGITHUB)]}
    with pytest.raises(GraphQLError):
        await _ask(body, (_DOTGITHUB,))


@pytest.mark.asyncio
async def test_an_allowed_not_found_is_never_mistaken_for_a_transient_error() -> None:
    # The transient heuristic matches wording, and this owner's message
    # says "timeout"; the expected error must not be retried five times.
    error = _not_found(*_DOTGITHUB, login="timeout-org")
    data, attempts = await _ask({"data": _DATA, "errors": [error]}, (_DOTGITHUB,))
    assert data == _DATA
    assert attempts == 1
