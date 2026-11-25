# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
The cross-organization similar-PR search.

``find_similar_prs`` walks an owner's repositories; the per-repository
sweep, the candidate filter and the ``--debug`` comparison dump live in
helpers so each stays reviewable.
"""

# pyright: reportUninitializedInstanceVariable=false

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import NamedTuple

from ..bot_identity import is_automation_author
from ..bot_policy import BotPolicy
from ..models import ComparisonResult, PullRequestInfo
from ._base import _GitHubServiceBase
from ._comparison_debug import print_comparison_debug
from ._repositories import _listed_without_open_prs
from ._scan_lists import MERGE_SCAN_LISTS


class _Fetched(NamedTuple):
    """One repository's open PRs, as a worker fetched them.

    ``prs`` is ``None`` when the repository's own policy file is invalid;
    ``seen`` holds the App PRs read either way.
    """

    name: str
    prs: list[PullRequestInfo] | None
    seen: list[PullRequestInfo]


def _is_source(source_pr: PullRequestInfo, pr: PullRequestInfo) -> bool:
    """Whether ``pr`` is ``source_pr`` itself."""
    return (pr.number, pr.repository_full_name) == (
        source_pr.number,
        source_pr.repository_full_name,
    )


def _other_app_pr(source_pr: PullRequestInfo, pr: PullRequestInfo) -> bool:
    """Whether ``pr`` is an App's, other than ``source_pr`` itself."""
    return is_automation_author(pr.author) and not _is_source(source_pr, pr)


def _app_prs(prs: Iterable[PullRequestInfo]) -> list[PullRequestInfo]:
    """The App PRs among ``prs``."""
    return [pr for pr in prs if is_automation_author(pr.author)]


def _variable_matters(
    source_pr: PullRequestInfo,
    prs: list[PullRequestInfo],
    *,
    only_automation: bool,
) -> bool:
    """Whether a repository's ``DEPENDAMERGE_BOTS`` variable bears on a search.

    Read where an App's PR is open, as every run reads it.  The policy
    judges no person's PR, but an invalid one keeps the search from
    acting on any PR in the repository; so for a person's source PR it is
    read where that person has another PR open, the only PRs the search
    could act on there.
    """
    if not any(_other_app_pr(source_pr, pr) for pr in prs):
        return False
    author = source_pr.author or ""
    return only_automation or any(
        (pr.author or "") == author and not _is_source(source_pr, pr) for pr in prs
    )


class _SimilarPullRequestsMixin(_GitHubServiceBase):
    """Similar-PR discovery for ``GitHubService``."""

    @property
    def seen_app_prs(self) -> tuple[PullRequestInfo, ...]:
        """Every App's pull request the searches have seen, as a snapshot.

        Before any filter: an App the policy does not admit, or whose pull
        request does not match, was still seen, and a policy variable the
        token cannot read might have admitted it.
        """
        return tuple(self._seen_app_prs)

    async def find_similar_prs(
        self,
        org: str,
        source_pr: PullRequestInfo,
        comparator,
        *,
        only_automation: bool,
        bot_policy: BotPolicy | None = None,
    ) -> list[tuple[PullRequestInfo, ComparisonResult]]:
        """
        Find PRs across an organization that are similar to the provided source PR.

        This integrates progress updates:
        - Updates total repositories
        - Starts/completes repository sections
        - Increments PR analysis count per PR
        - Tracks similar PRs found

        Repositories are read concurrently, at most ``max_repo_tasks`` at
        once; their variables are read, and their PRs judged, in listing
        order, so the results come back in that order and a variable the
        token is refused stops the same reads a serial search would.  A
        repository that cannot be read fails the whole search.  A repository the
        listing counts no open PR in is not read; it still counts as
        processed.  The search is a snapshot: a PR opened after the
        listing, in a repository listed with none, is not seen.

        Args:
            org: Organization login.
            source_pr: The PR to compare against.
            comparator: Provides compare_pull_requests(source, target) -> ComparisonResult.
            only_automation: If True, restrict candidates to automation PRs; otherwise, same author as source.
            bot_policy: The Apps whose PRs count as automation candidates;
                without one, only the built-in Apps do.

        Returns:
            List of (PullRequestInfo, ComparisonResult) tuples for similar PRs.
        """
        # A bounded pipeline, as in the owner-wide scan: a fixed pool of
        # workers drains a queue the listing feeds, and a failure in any
        # repository, or in the listing, tears the search down at once.
        # The workers only fetch; one judge then reads each repository's
        # variable and decides on its PRs in listing order, as a serial
        # search would, since one refused read stops every later one.
        # ``window`` caps the repositories listed but not yet judged, so a
        # slow early one cannot leave the whole owner's PRs in ``fetched``.
        worker_count = max(1, self._max_repo_tasks)
        queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
        window = asyncio.Semaphore(worker_count * 2)
        # Fetched repositories by listing position; ``None`` past the last.
        fetched: dict[int, asyncio.Future[_Fetched | None]] = {}
        loop = asyncio.get_running_loop()

        def slot(position: int) -> asyncio.Future[_Fetched | None]:
            if position not in fetched:
                fetched[position] = loop.create_future()
            return fetched[position]

        results: list[tuple[PullRequestInfo, ComparisonResult]] = []
        seen: list[PullRequestInfo] = []

        async def producer() -> None:
            position = 0
            # Repo total is set automatically by _iter_org_repositories
            # on the first GraphQL page via totalCount.
            async for repo in self._iter_org_repositories_with_open_prs(org):
                repo_full_name = repo.get("nameWithOwner") or ""
                if not repo_full_name or "/" not in repo_full_name:
                    if self._progress:
                        self._progress.add_error()
                    continue
                if _listed_without_open_prs(repo):
                    if self._progress:
                        self._progress.complete_repository(0)
                    continue
                await window.acquire()
                await queue.put((position, repo_full_name))
                position += 1
            slot(position).set_result(None)
            for _ in range(worker_count):
                await queue.put(None)

        async def worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    return
                position, repo_full_name = item
                slot(position).set_result(
                    await self._fetch_listed_repo(
                        repo_full_name, bot_policy=bot_policy or BotPolicy()
                    )
                )

        async def judge() -> None:
            position = 0
            while (repo := await slot(position)) is not None:
                del fetched[position]
                results.extend(
                    await self._find_similar_prs_in_repo(
                        repo,
                        source_pr,
                        comparator,
                        only_automation=only_automation,
                        bot_policy=bot_policy or BotPolicy(),
                    )
                )
                seen.extend(repo.seen)
                window.release()
                position += 1

        pipeline = [
            asyncio.create_task(producer()),
            asyncio.create_task(judge()),
            *(asyncio.create_task(worker()) for _ in range(worker_count)),
        ]
        try:
            # No return_exceptions: one repository that cannot be read
            # fails the search, which must not act on a smaller batch.
            await asyncio.gather(*pipeline)
        except BaseException:
            for task in pipeline:
                task.cancel()
            await asyncio.gather(*pipeline, return_exceptions=True)
            raise

        self._seen_app_prs.extend(seen)
        return results

    async def _fetch_listed_repo(
        self, repo_full_name: str, *, bot_policy: BotPolicy
    ) -> _Fetched:
        """A repository's open PRs, before its variable is read.

        Runs in the repository pool, so at most ``max_repo_tasks``
        repositories are read at once.  The policy file its first page
        carries is checked here, as it concerns that repository alone;
        when it is invalid, the later pages are not read and the
        fetched PRs are ``None``, after the first page's App PRs join
        ``seen``: none is in scope, but whether a variable the token
        cannot read is worth a warning still counts them.
        """
        async with self._repo_semaphore:
            if self._progress:
                self._progress.start_repository(repo_full_name)
                self._progress.update_operation(
                    f"Getting open PRs from {repo_full_name}"
                )
            owner_n, name_n = repo_full_name.split("/", 1)
            first_nodes, page_info = await self._fetch_repo_prs_first_page(
                owner_n, name_n, lists=MERGE_SCAN_LISTS
            )
            if self._valid_repository_policy(bot_policy, repo_full_name) is None:
                return _Fetched(
                    repo_full_name,
                    None,
                    _app_prs(
                        self.to_pull_request_info(repo_full_name, node)
                        for node in first_nodes
                    ),
                )
            nodes = list(first_nodes)
            if page_info.get("hasNextPage"):
                end_cursor = page_info.get("endCursor") or None
                async for pr_node in self._iter_repo_open_prs_pages(
                    owner_n, name_n, end_cursor, lists=MERGE_SCAN_LISTS
                ):
                    nodes.append(pr_node)
            prs = [self.to_pull_request_info(repo_full_name, node) for node in nodes]
            return _Fetched(repo_full_name, prs, _app_prs(prs))

    async def _find_similar_prs_in_repo(
        self,
        repo: _Fetched,
        source_pr: PullRequestInfo,
        comparator,
        *,
        only_automation: bool,
        bot_policy: BotPolicy,
    ) -> list[tuple[PullRequestInfo, ComparisonResult]]:
        """Compare every open PR of one fetched repository against *source_pr*.

        Candidates are judged with the repository's own bot policy (see
        :meth:`_repository_policy_for`); when that is invalid, nothing
        there is a candidate.
        """
        repo_full_name = repo.name
        policy = await self._repository_policy_for(
            repo, source_pr, only_automation=only_automation, bot_policy=bot_policy
        )
        matching_prs_in_repo: list[tuple[PullRequestInfo, ComparisonResult]] = []

        for target_pr in repo.prs if policy is not None and repo.prs else []:
            if not self._is_similar_pr_candidate(
                source_pr,
                target_pr,
                only_automation=only_automation,
                bot_policy=policy,
            ):
                continue

            if self._progress:
                self._progress.analyze_pr(target_pr.number, repo_full_name)

            comparison: ComparisonResult = comparator.compare_pull_requests(
                source_pr, target_pr, only_automation
            )

            if self._debug_matching:
                print_comparison_debug(
                    repo_full_name, source_pr, target_pr, comparator, comparison
                )

            if comparison.is_similar:
                matching_prs_in_repo.append((target_pr, comparison))
                if self._progress:
                    # We can reuse 'found_similar_pr' if using MergeProgressTracker,
                    # otherwise this call will be a no-op for ProgressTracker.
                    try:
                        self._progress.found_similar_pr()  # type: ignore[attr-defined]
                    except Exception:
                        # No-op when the tracker lacks this method or
                        # the display update fails; counting is
                        # cosmetic only.
                        pass

        if self._progress:
            self._progress.complete_repository(len(matching_prs_in_repo))
        return matching_prs_in_repo

    async def _repository_policy_for(
        self,
        repo: _Fetched,
        source_pr: PullRequestInfo,
        *,
        only_automation: bool,
        bot_policy: BotPolicy,
    ) -> BotPolicy | None:
        """The policy that judges a fetched repository's PRs.

        It also consults the repository's own file, which its first page
        carried, and its ``DEPENDAMERGE_BOTS`` variable, read where it
        bears on the search (see :func:`_variable_matters`).  ``None``
        when that policy is invalid.
        """
        if repo.prs is None:
            return None
        if not _variable_matters(source_pr, repo.prs, only_automation=only_automation):
            return self._with_repository_policy(bot_policy, repo.name)
        await self._read_repository_variable(repo.name)
        return self._valid_repository_policy(bot_policy, repo.name)

    def _valid_repository_policy(
        self, bot_policy: BotPolicy, repository: str
    ) -> BotPolicy | None:
        """``bot_policy`` with ``repository``'s own; ``None`` when invalid."""
        policy = self._with_repository_policy(bot_policy, repository)
        problem = policy.repository_error(repository)
        if problem is None:
            return policy
        self.log.warning(
            "Skipping %s: its bot policy is invalid (%s)", repository, problem
        )
        return None

    @staticmethod
    def _is_similar_pr_candidate(
        source_pr: PullRequestInfo,
        target_pr: PullRequestInfo,
        *,
        only_automation: bool,
        bot_policy: BotPolicy | None = None,
    ) -> bool:
        """Whether *target_pr* is worth comparing against *source_pr*."""
        # Skip the source PR itself
        if _is_source(source_pr, target_pr):
            return False

        # Candidate filtering: an App the policy admits when the source is
        # automation (a person whose login merely contains "bot" never
        # qualifies), the same author otherwise.
        if only_automation:
            return (bot_policy or BotPolicy()).admits(
                target_pr.author, target_pr.repository_full_name
            )
        return (target_pr.author or "") == (source_pr.author or "")
