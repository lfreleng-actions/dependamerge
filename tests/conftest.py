# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""Shared pytest fixtures for dependamerge tests.

Typed Mock Client Pattern
=========================

Problem
-------
``AsyncMergeManager`` (and ``AsyncCloseManager``) declare their internal HTTP
client as an optional type::

    self._github_client: GitHubAsync | None = None

The client is only populated inside ``__aenter__()`` (the async context
manager).  In tests we routinely bypass the context manager and inject an
``AsyncMock`` directly::

    mgr = AsyncMergeManager(token="t")
    mgr._github_client = AsyncMock()
    mgr._github_client.get = AsyncMock(return_value=...)  # ← warning!

Because the *declared* type is ``GitHubAsync | None``, basedpyright cannot
prove the value is non-``None`` after assignment and flags every subsequent
attribute access as ``reportOptionalMemberAccess``.

Solution
--------
The ``make_merge_manager`` helper (and any similar helpers in individual test
modules) returns a **tuple** ``(manager, client)`` where ``client`` is typed
as ``AsyncMock`` — a concrete, non-optional reference to the same object
stored in ``manager._github_client``.  All subsequent mock configuration
should go through the ``client`` variable::

    mgr, client = make_merge_manager(token="t")
    client.get = AsyncMock(return_value=...)          # ✓ no warning
    client.post_issue_comment = AsyncMock()            # ✓ no warning
    client.post_issue_comment.assert_called_once()     # ✓ no warning

This eliminates basedpyright ``reportOptionalMemberAccess`` warnings without
changing any production code or adding ``assert ... is not None`` boilerplate
to every test.

Guidelines for New Tests
------------------------
1. **Always** use the ``make_merge_manager`` helper (or a module-local
   wrapper around it) when you need an ``AsyncMergeManager`` with a mocked
   GitHub client outside of ``async with``.

2. Hold on to the returned ``client`` variable and use it — *not*
   ``mgr._github_client`` — for all mock setup and assertions.

3. If a test intentionally sets ``_github_client = None`` to exercise the
   "no client" code path, do that *after* unpacking the tuple::

       mgr, _client = make_merge_manager()
       mgr._github_client = None   # intentional for this test

4. If you use ``async with AsyncMergeManager(...) as mgr:`` (which calls
   ``__aenter__`` and sets the real client), you can safely replace the
   client inside the block because basedpyright already narrowed the type.
   You do **not** need this helper in that case.

5. If instead you patch ``dependamerge.merge_manager.GitHubAsync`` and
   drive the CLI end to end, use ``make_github_async_mock`` rather than
   hand-rolling the ``__aenter__`` plumbing.  It mirrors the real
   client's synchronous methods *and* its dict-returning coroutines,
   both of which a blanket ``AsyncMock`` gets wrong.

See Also
--------
- ``tests/test_dependabot_recreate.py`` — ``_make_manager()`` wraps this
  helper with module-specific defaults.
- ``tests/test_precommit_ci_trigger.py`` — same pattern.
- ``tests/test_github2gerrit_detector.py`` — same pattern for
  ``_make_mgr_with_no_gitreview``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dependamerge.github_async import GitHubAsync
from dependamerge.merge_manager import AsyncMergeManager
from dependamerge.models import PullRequestInfo
from dependamerge.url_parser import set_gerrit_host, set_github_host

_RUN_INTEGRATION_ENV = "DEPENDAMERGE_RUN_INTEGRATION"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-integration`` opt-in flag.

    Live integration tests (see ``tests/integration``) hit real GitHub /
    Gerrit servers.  They already self-skip when credentials are absent,
    but they must not run as part of the ordinary unit-test suite even
    when a token happens to be present in the environment, because they
    are slow and network-dependent.  They run only when explicitly
    requested via ``--run-integration`` or the
    ``DEPENDAMERGE_RUN_INTEGRATION`` environment variable.
    """
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run live GitHub/Gerrit integration tests (marked 'integration').",
    )


def _integration_enabled(config: pytest.Config) -> bool:
    if config.getoption("--run-integration"):
        return True
    return os.environ.get(_RUN_INTEGRATION_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``integration`` tests unless explicitly opted in."""
    if _integration_enabled(config):
        return
    skip_integration = pytest.mark.skip(
        reason="integration tests disabled (pass --run-integration or set "
        f"{_RUN_INTEGRATION_ENV}=1)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def _reset_github_host_override(monkeypatch):
    """Detach every test from ambient host configuration.

    The resolved host comes from a process-wide override *and* several
    environment variables.  Clearing only the override leaves the rest:
    a developer who legitimately has ``GH_HOST`` set for their
    Enterprise installation would run the otherwise-dotcom suite
    against that host and see failures nobody else can reproduce.

    The same hazard as the ambient git configuration in
    ``tests/test_local_repo.py``, and worth the same treatment.  Tests
    that want a host set do so explicitly, after this has run.

    Gerrit declarations are cleared alongside, and for a sharper
    reason: they decide *routing*, so one left set would send a target
    to the wrong platform's parser entirely rather than merely to the
    wrong host.  ``GERRIT_HOST`` is deliberately absent --- this tool
    does not read it as a declaration, and clearing a variable it never
    consults would hide that from anyone reading this list.
    """
    for name in (
        "DEPENDAMERGE_GITHUB_HOST",
        "DEPENDAMERGE_GITHUB_HOSTS",
        "GH_HOST",
        "DEPENDAMERGE_GERRIT_HOST",
        "DEPENDAMERGE_GERRIT_HOSTS",
    ):
        monkeypatch.delenv(name, raising=False)
    set_github_host(None)
    set_gerrit_host(None)
    yield
    set_github_host(None)
    set_gerrit_host(None)


@pytest.fixture(autouse=True)
def _detach_from_github_actions(monkeypatch):
    """Run every test as if outside GitHub Actions.

    Under Actions a merge run writes a results file and appends its path
    to ``GITHUB_OUTPUT``.  The suite itself runs under Actions in CI, so
    without this every CLI test there would write into the CI job's own
    step outputs, and behave differently from the same test run locally.
    Tests of that behaviour set the variables explicitly.
    """
    for name in ("GITHUB_ACTIONS", "GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY"):
        monkeypatch.delenv(name, raising=False)


class FakeCommitListings:
    """Serve a commit's check runs and statuses the way GitHub pages them.

    ``get`` answers an unpaginated request with the first page at
    GitHub's default size of 30; ``get_paginated`` honours ``per_page``
    and stops after the last page.  ``fail_on`` names ``(listing,
    page)`` pairs, with listing ``"check-runs"`` or ``"status"``, whose
    request raises.  ``requests`` records each listing request made.
    Any other path is answered from ``other``, keyed by path suffix, as
    one page when read through ``get_paginated``.
    """

    def __init__(
        self,
        check_runs: list[dict[str, Any]] | None = None,
        statuses: list[dict[str, Any]] | None = None,
        *,
        fail_on: set[tuple[str, int]] | None = None,
        other: dict[str, Any] | None = None,
    ) -> None:
        self.listings = {
            "check-runs": ("check_runs", self._numbered(check_runs)),
            "status": ("statuses", self._numbered(statuses)),
        }
        self.fail_on = fail_on or set()
        self.other = other or {}
        self.requests: list[tuple[str, int]] = []

    @staticmethod
    def _numbered(entries: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Give each entry an ``id``, as GitHub does, unless it has one."""
        return [{"id": index, **entry} for index, entry in enumerate(entries or [], 1)]

    def install(self, api: Any) -> None:
        """Route *api*'s ``get`` and ``get_paginated`` through this fake."""
        api.get = self.get
        api.get_paginated = self.get_paginated

    def _page(self, listing: str, page: int, per_page: int) -> dict[str, Any]:
        self.requests.append((listing, page))
        if (listing, page) in self.fail_on:
            raise RuntimeError(f"502 Bad Gateway on {listing} page {page}")
        key, entries = self.listings[listing]
        start = (page - 1) * per_page
        return {
            "total_count": len(entries),
            key: entries[start : start + per_page],
        }

    @staticmethod
    def _listing(path: str) -> str | None:
        for listing in ("check-runs", "status"):
            if path.endswith(f"/{listing}"):
                return listing
        return None

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        listing = self._listing(path)
        if listing is not None:
            return self._page(listing, 1, 30)
        for suffix, value in self.other.items():
            if path.endswith(suffix):
                return value
        return {}

    async def get_paginated(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        per_page: int = 100,
        max_pages: int | None = None,
        yield_malformed: bool = False,
    ):
        listing = self._listing(path)
        if listing is None:
            page = await self.get(path)
            if page:
                yield page
            return
        total = len(self.listings[listing][1])
        page = 1
        while True:
            yield self._page(listing, page, per_page)
            if page * per_page >= total:
                return
            page += 1


@pytest.fixture
def listings_via_get(monkeypatch):
    """Serve ``GitHubAsync.get_paginated`` as one page from its ``get``.

    For tests that route a mocked ``get`` by URL.  The check-run,
    commit-status, review and review-comment listings are read through
    ``get_paginated``, which would otherwise reach the real transport.
    """
    from dependamerge.github_async import GitHubAsync

    async def _one_page(self, path, **_kwargs):
        page = await self.get(path)
        if page:
            yield page

    monkeypatch.setattr(GitHubAsync, "get_paginated", _one_page)


@pytest.fixture(scope="session")
def _empty_config_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return Path(tmp_path_factory.mktemp("config-home"))


@pytest.fixture(autouse=True)
def _detach_from_user_config(monkeypatch, _empty_config_home: Path):
    """Run every test without the developer's own bot policy.

    A ``~/.config/dependamerge/config.json`` that includes or excludes
    an App changes which pull requests a run acts on, so a suite that
    read it would pass or fail by whose machine it ran on.  Tests of the
    user policy point ``XDG_CONFIG_HOME`` elsewhere themselves.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(_empty_config_home))


@pytest.fixture
def paginate_through_get(monkeypatch):
    """Serve ``GitHubAsync.get_paginated`` from the instance's ``get``.

    Tests that route every request through a ``gh.get`` mock predate the
    paged ruleset list.  This answers a paged read with the single
    ``{path}?per_page=100`` request they already expect, so their routes
    and request counts stay as written.
    """

    async def get_paginated(
        self: GitHubAsync, path: str, **_kwargs: Any
    ) -> AsyncIterator[dict[str, Any] | list[Any]]:
        page = await self.get(f"{path}?per_page=100")
        if page:
            yield page

    monkeypatch.setattr(GitHubAsync, "get_paginated", get_paginated)


def make_merge_manager(**overrides: Any) -> tuple[AsyncMergeManager, AsyncMock]:
    """Build an ``AsyncMergeManager`` with a pre-injected ``AsyncMock`` client.

    Returns a ``(manager, client)`` tuple.  The ``client`` reference is typed
    as ``AsyncMock`` (never ``None``), so attribute access on it will not
    trigger basedpyright ``reportOptionalMemberAccess`` warnings.

    All keyword arguments are forwarded to ``AsyncMergeManager.__init__``.
    A ``token`` default of ``"test-token"`` is provided if not supplied.

    Usage::

        mgr, client = make_merge_manager(preview_mode=True)
        client.get = AsyncMock(return_value={...})
        result = await mgr._some_method(pr)
        client.get.assert_called_once()

    Parameters
    ----------
    **overrides:
        Keyword arguments forwarded to ``AsyncMergeManager()``.

    Returns
    -------
    tuple[AsyncMergeManager, AsyncMock]
        The manager instance and its mock GitHub client.
    """
    defaults: dict[str, Any] = {"token": "test-token"}
    defaults.update(overrides)
    mgr = AsyncMergeManager(**defaults)
    client = AsyncMock()
    # Mirror the real client's shape for the handful of *synchronous*
    # methods on it.  A blanket ``AsyncMock`` turns these into
    # coroutines that production code never awaits, producing
    # "never awaited" warnings that obscure genuine ones.
    client.invalidate_block_reason = MagicMock()
    client.clear_block_reasons = MagicMock()
    mgr._github_client = client
    return mgr, client


def make_pr_info(**overrides: Any) -> PullRequestInfo:
    """Build an open, clean dependabot ``PullRequestInfo`` for ``o/r#1``.

    Keyword arguments replace any field.
    """
    fields: dict[str, Any] = {
        "number": 1,
        "title": "Chore: Bump foo from 1.0.0 to 1.0.1",
        "body": "bump",
        "author": "dependabot[bot]",
        "head_sha": "a" * 40,
        "base_branch": "main",
        "head_branch": "dependabot/foo",
        "state": "open",
        "mergeable": True,
        "mergeable_state": "clean",
        "behind_by": None,
        "files_changed": [],
        "repository_full_name": "o/r",
        "html_url": "https://github.com/o/r/pull/1",
    }
    fields.update(overrides)
    return PullRequestInfo(**fields)


def make_github_async_mock() -> AsyncMock:
    """Build a faithful stand-in for the ``GitHubAsync`` class's instance.

    For tests that patch ``dependamerge.merge_manager.GitHubAsync`` and
    drive the CLI end to end, rather than injecting a client directly as
    :func:`make_merge_manager` does.  Assign the result to the patched
    class's ``return_value``.

    A *single* mock is returned, and its ``__aenter__`` yields that same
    object, because the real ``GitHubAsync.__aenter__`` returns ``self``.
    Splitting it into a separate "instance" and "client" pair --- the
    shape this replaced --- is the trap: ``AsyncMergeManager`` keeps the
    object it constructed and calls ``__aenter__`` only for its side
    effect, then shares that object with ``GitHubService``.  Configuring
    the ``__aenter__`` result therefore left the object actually in use
    bare, and the divergence was invisible because both are mocks.

    Two further details keep the double honest where a blanket
    ``AsyncMock`` silently diverges:

    *Synchronous methods.*  ``invalidate_block_reason`` and
    ``clear_block_reasons`` do not return awaitables, and production
    code calls them without ``await``.  Mocked as coroutine functions
    they leak "never awaited" warnings.

    *Dict-returning coroutines.*  ``graphql`` is declared to return
    ``dict[str, Any]``, but awaiting a bare ``AsyncMock`` yields another
    ``AsyncMock``.  Production code then calls ``.get(...)`` on it,
    which manufactures a coroutine nobody awaits.  That warning is
    reported against whichever line happens to be running when the
    garbage collector reclaims the object, so it points nowhere near the
    cause --- see #420 for how long that misdirection held.  Returning a
    real empty dict keeps the "no repository data" branch honest;
    override it when a test cares about the payload.

    Usage::

        with patch("dependamerge.merge_manager.GitHubAsync") as cls:
            client = make_github_async_mock()
            cls.return_value = client
            client.merge_pull_request = AsyncMock(return_value=True)

    Returns
    -------
    AsyncMock
        The client, which is also its own async-context-manager result.
    """
    client = AsyncMock()
    client.invalidate_block_reason = MagicMock()
    client.clear_block_reasons = MagicMock()
    client.graphql = AsyncMock(return_value={})
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client
