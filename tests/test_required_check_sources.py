# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests that required checks come from both places GitHub enforces.

Rulesets and classic branch protection are **cumulative** — GitHub
enforces every requirement either one declares. The collector consulted
protection only when the ruleset lookup came back empty, so a branch
guarded by both reported a partial set, and reported it reliable
because the skipped lookup could not report a failure it was never
asked to perform.

The sharpest consequence was not the reporting: ``_precommit_ci_required``
asks whether ``pre-commit.ci - pr`` is required, so a repository
enforcing it through protection while a ruleset also applied had its
retrigger silently disabled.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from dependamerge.github_async import GitHubAsync

RULESET_LIST = "/repos/org/repo/rulesets?per_page=100"
RULESET_DETAIL = "/repos/org/repo/rulesets/1"
PROTECTION = "/repos/org/repo/branches/main/protection/required_status_checks"


def _estate_router(repos: int):
    """A ``gh.get`` stand-in for *repos* identically-configured repos."""

    async def get(url: str, **_kwargs: Any) -> Any:
        if url.endswith("/rulesets?per_page=100"):
            return [{"id": 1}]
        if "/rulesets/" in url:
            return {
                "conditions": {},
                "rules": [
                    {
                        "type": "required_status_checks",
                        "parameters": {"required_status_checks": [{"context": "DCO"}]},
                    }
                ],
            }
        if url.endswith("/protection/required_status_checks"):
            return {"contexts": ["pre-commit.ci - pr"]}
        return {"default_branch": "main"}

    return get


def _router(
    *,
    ruleset_contexts: list[str] | None = None,
    protection_contexts: list[str] | None = None,
    protection_error: Exception | None = None,
):
    """A ``gh.get`` stand-in describing one repository's configuration."""

    async def get(url: str, **_kwargs: Any) -> Any:
        if url == "/repos/org/repo":
            return {"default_branch": "main"}
        if url == RULESET_LIST:
            return [{"id": 1}] if ruleset_contexts else []
        if url == RULESET_DETAIL:
            return {
                "conditions": {},
                "rules": [
                    {
                        "type": "required_status_checks",
                        "parameters": {
                            "required_status_checks": [
                                {"context": c} for c in ruleset_contexts or []
                            ]
                        },
                    }
                ],
            }
        if url == PROTECTION:
            if protection_error is not None:
                raise protection_error
            return {"contexts": list(protection_contexts or [])}
        raise AssertionError(f"unexpected request: {url}")

    return get


async def _contexts(gh: GitHubAsync) -> tuple[list[str], bool]:
    checks, reliable = await gh.get_required_status_checks_reliable(
        "org", "repo", "main"
    )
    return [c["context"] for c in checks], reliable


class TestBothSourcesAreCombined:
    """A branch may be guarded by a ruleset and by protection at once."""

    @pytest.mark.asyncio
    async def test_a_protection_context_is_not_lost_to_a_ruleset(self) -> None:
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_router(
                    ruleset_contexts=["DCO"],
                    protection_contexts=["pre-commit.ci - pr"],
                )
            )

            contexts, reliable = await _contexts(gh)

        assert contexts == ["DCO", "pre-commit.ci - pr"]
        assert reliable is True

    @pytest.mark.asyncio
    async def test_a_context_required_by_both_appears_once(self) -> None:
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_router(
                    ruleset_contexts=["DCO"], protection_contexts=["DCO"]
                )
            )

            contexts, _ = await _contexts(gh)

        assert contexts == ["DCO"]

    @pytest.mark.asyncio
    async def test_protection_alone_still_works(self) -> None:
        """The path that worked before: no ruleset, protection only."""
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_router(protection_contexts=["pre-commit.ci - pr"])
            )

            contexts, reliable = await _contexts(gh)

        assert contexts == ["pre-commit.ci - pr"]
        assert reliable is True


class TestReliabilityFoldsBothLookups:
    """An empty set means nothing unless both sources were readable."""

    @pytest.mark.asyncio
    async def test_a_404_is_a_reliable_none(self) -> None:
        """An unprotected branch must not read as an unreadable one."""
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_router(
                    ruleset_contexts=["DCO"],
                    protection_error=Exception("404 Not Found"),
                )
            )

            contexts, reliable = await _contexts(gh)

        assert contexts == ["DCO"]
        assert reliable is True

    @pytest.mark.asyncio
    async def test_any_other_protection_failure_is_unreliable(self) -> None:
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_router(
                    ruleset_contexts=["DCO"],
                    protection_error=Exception("403 Forbidden"),
                )
            )

            contexts, reliable = await _contexts(gh)

        assert contexts == ["DCO"]
        assert reliable is False

    @pytest.mark.asyncio
    async def test_an_unreliable_result_is_not_cached(self) -> None:
        """Pinning an error-derived verdict would outlive the outage."""
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_router(
                    ruleset_contexts=["DCO"],
                    protection_error=Exception("403 Forbidden"),
                )
            )

            await _contexts(gh)
            before = gh.get.await_count
            await _contexts(gh)

        assert gh.get.await_count > before


class TestTheAddedCostIsOneRequestPerBranch:
    """Measured, because the issue asked for it rather than an assurance.

    The extra call is per ``owner/repo@branch`` and cached for the
    session, so a run over many pull requests in one repository pays it
    once — not once per pull request.
    """

    @pytest.mark.asyncio
    async def test_a_ruleset_guarded_branch_costs_one_more(self) -> None:
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_router(
                    ruleset_contexts=["DCO"], protection_contexts=["lint"]
                )
            )

            await _contexts(gh)
            requests = [call.args[0] for call in gh.get.await_args_list]

        # repo metadata, ruleset list, ruleset detail, protection.
        assert requests == [
            "/repos/org/repo",
            RULESET_LIST,
            RULESET_DETAIL,
            PROTECTION,
        ]

    @pytest.mark.asyncio
    async def test_a_second_read_of_the_same_branch_is_free(self) -> None:
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(  # type: ignore[method-assign]
                side_effect=_router(
                    ruleset_contexts=["DCO"], protection_contexts=["lint"]
                )
            )

            await _contexts(gh)
            first = gh.get.await_count
            await _contexts(gh)

        assert gh.get.await_count == first


class TestTheCostScalesWithRepositoriesNotPullRequests:
    """The issue asked for a multi-repository measurement, not a promise.

    A 60-PR owner-wide run over 32 repositories is the shape that
    matters: the added lookup is keyed on ``owner/repo@branch``, so it
    should be paid once per repository however many pull requests each
    one holds.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("repos", "prs_each"), [(1, 1), (32, 2), (32, 10)])
    async def test_one_added_request_per_repository(
        self, repos: int, prs_each: int
    ) -> None:
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(side_effect=_estate_router(repos))  # type: ignore[method-assign]

            for index in range(repos):
                for _ in range(prs_each):
                    await gh.get_required_status_checks_reliable(
                        "org", f"repo-{index}", "main"
                    )

            requests = [call.args[0] for call in gh.get.await_args_list]

        protection = [
            r for r in requests if r.endswith("/protection/required_status_checks")
        ]
        # Once per repository, whatever the pull request count.
        assert len(protection) == repos
        # And the whole lookup is 4 requests per repository, not per PR.
        assert len(requests) == repos * 4

    @pytest.mark.asyncio
    async def test_a_slash_named_branch_is_encoded(self) -> None:
        """``release/v2`` must not become extra path segments.

        Unencoded, the request addresses a path that cannot exist, and
        the 404 is indistinguishable from the definitive "this branch
        has no protection" -- so protection-required checks would be
        silently omitted on every slash-named base.
        """
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(side_effect=_estate_router(1))  # type: ignore[method-assign]

            await gh.get_required_status_checks_reliable("org", "repo", "release/v2")
            requests = [call.args[0] for call in gh.get.await_args_list]

        assert (
            "/repos/org/repo/branches/release%2Fv2/protection/required_status_checks"
            in requests
        )
        assert not any("/branches/release/v2/" in r for r in requests)

    @pytest.mark.asyncio
    async def test_each_branch_is_counted_separately(self) -> None:
        """The cache key includes the branch, so a second base costs again."""
        async with GitHubAsync(token="t") as gh:
            gh.get = AsyncMock(side_effect=_estate_router(1))  # type: ignore[method-assign]

            await gh.get_required_status_checks_reliable("org", "repo", "main")
            await gh.get_required_status_checks_reliable("org", "repo", "release/v2")

            protection = [
                call.args[0]
                for call in gh.get.await_args_list
                if call.args[0].endswith("/protection/required_status_checks")
            ]

        assert len(protection) == 2
