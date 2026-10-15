# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""The ``DEPENDAMERGE_BOTS`` Actions variables as a bot policy source.

GraphQL does not expose variables, so each costs a REST request: the
owner's is read once, a repository's when it holds an App's pull
request.  A token may be refused one kind and allowed the other (a
classic token without ``admin:org`` is refused an organisation's but
reads a repository's), so a refusal stops further reads of its kind only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from dependamerge.bot_policy import BotPolicy, PolicyLevel
from dependamerge.bot_policy_files import PolicyFiles, apply_policy_files
from dependamerge.cli._policy_files import _warn_unreadable_variables
from dependamerge.github_graphql import ORG_REPOS_ONLY, USER_REPOS_ONLY
from dependamerge.github_service import GitHubService
from dependamerge.models import PullRequestInfo

CODE_MONKEY = "lf-releng-code-monkey-bot[bot]"


def _error(status: int, path: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"https://api.github.com{path}")
    return httpx.HTTPStatusError(
        "error", request=request, response=httpx.Response(status, request=request)
    )


class _Api:
    """Answers variable reads from a table: a value, or an HTTP status."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.paths: list[str] = []

    async def get(self, path: str, params: Any = None) -> dict[str, Any]:
        self.paths.append(path)
        answer = self.answers.get(path.split("/actions/")[0], 404)
        if isinstance(answer, int):
            raise _error(answer, path)
        return {"name": "DEPENDAMERGE_BOTS", "value": answer}


def _service(answers: dict[str, Any], *, organisation: bool = True) -> tuple[Any, _Api]:
    svc = GitHubService(token="t")
    api = _Api(answers)
    svc._api = api  # type: ignore[assignment]
    root = (
        ("organization", ORG_REPOS_ONLY) if organisation else ("user", USER_REPOS_ONLY)
    )
    svc._owner_root_cache["acme"] = root
    return svc, api


def _pr(author: str, repo: str) -> PullRequestInfo:
    return PullRequestInfo(
        number=1,
        title="t",
        body=None,
        author=author,
        head_sha="abc",
        base_branch="main",
        head_branch="topic",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name=repo,
        html_url=f"https://github.com/{repo}/pull/1",
    )


DOCUMENT = json.dumps({"include_bots": ["lf-releng-code-monkey-bot"]})


class TestReads:
    @pytest.mark.asyncio
    async def test_values_and_absences_are_recorded(self) -> None:
        svc, _api = _service({"/orgs/acme": DOCUMENT})
        await svc.fetch_policy_variables("acme", [_pr(CODE_MONKEY, "acme/widget")])
        files = svc.policy_files
        assert files.owner_variables == {"acme": DOCUMENT}
        assert files.repository_variables == {"acme/widget": None}
        assert not files.variables_unreadable

    @pytest.mark.asyncio
    async def test_only_repositories_with_app_prs_are_read(self) -> None:
        svc, api = _service({})
        prs = [_pr("octocat", "acme/people"), _pr(CODE_MONKEY, "acme/widget")]
        await svc.fetch_policy_variables("acme", prs)
        assert "/repos/acme/people/actions/variables/DEPENDAMERGE_BOTS" not in api.paths
        assert "/repos/acme/widget/actions/variables/DEPENDAMERGE_BOTS" in api.paths

    @pytest.mark.asyncio
    async def test_the_owner_variable_is_read_once(self) -> None:
        svc, api = _service({})
        await svc.fetch_policy_variables("acme")
        await svc.fetch_policy_variables("acme")
        assert api.paths == ["/orgs/acme/actions/variables/DEPENDAMERGE_BOTS"]

    @pytest.mark.asyncio
    async def test_a_user_account_has_no_organisation_variable(self) -> None:
        svc, api = _service({}, organisation=False)
        await svc.fetch_policy_variables("acme")
        assert api.paths == []

    @pytest.mark.asyncio
    async def test_an_organisation_refusal_still_reads_repositories(self) -> None:
        svc, _api = _service({"/orgs/acme": 403, "/repos/acme/widget": DOCUMENT})
        await svc.fetch_policy_variables("acme", [_pr(CODE_MONKEY, "acme/widget")])
        files = svc.policy_files
        assert files.variables_refused == frozenset({"orgs"})
        assert files.repository_variables == {"acme/widget": DOCUMENT}

    @pytest.mark.asyncio
    async def test_a_repository_refusal_stops_later_repository_reads(self) -> None:
        svc, api = _service({"/repos/acme/a": 403})
        await svc.fetch_policy_variables("acme", [_pr(CODE_MONKEY, "acme/a")])
        await svc.fetch_policy_variables("acme", [_pr(CODE_MONKEY, "acme/b")])
        assert not any("/repos/acme/b/" in path for path in api.paths)
        assert svc.policy_files.variables_refused == frozenset({"repos"})


def _blob(document: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(document)
    return {
        "text": text,
        "byteSize": len(text),
        "isTruncated": False,
        "isBinary": False,
    }


class TestCombining:
    def test_a_variable_alone_applies(self) -> None:
        files = PolicyFiles(owner_variables={"acme": DOCUMENT})
        policy, errors = apply_policy_files(BotPolicy(), files)
        assert not errors
        decision = policy.decide(CODE_MONKEY, "acme/widget")
        assert decision.in_scope
        assert decision.reason == "included by acme DEPENDAMERGE_BOTS"

    def test_file_and_variable_combine(self) -> None:
        files = PolicyFiles(
            repositories={"acme/widget": _blob({"exclude_bots": ["renovate"]})},
            repository_variables={"acme/widget": DOCUMENT},
        )
        policy, _ = apply_policy_files(BotPolicy(), files)
        assert policy.admits(CODE_MONKEY, "acme/widget")
        assert not policy.admits("renovate[bot]", "acme/widget")

    def test_a_repository_conflict_invalidates_its_policy(self) -> None:
        files = PolicyFiles(
            repositories={"acme/widget": _blob({"exclude_bots": [CODE_MONKEY]})},
            repository_variables={"acme/widget": DOCUMENT},
        )
        policy, errors = apply_policy_files(BotPolicy(), files)
        assert not errors
        assert policy.repository_error("acme/widget") is not None

    def test_an_owner_conflict_is_returned_to_stop_the_run(self) -> None:
        files = PolicyFiles(
            owners={"acme": _blob({"exclude_bots": [CODE_MONKEY]})},
            owner_variables={"acme": DOCUMENT},
        )
        _, errors = apply_policy_files(BotPolicy(), files)
        assert errors
        assert "included by" in "; ".join(errors[0].problems)

    def test_policy_files_stay_backward_compatible(self) -> None:
        files = PolicyFiles(owners={}, repositories={})
        assert files.variables_refused == frozenset()
        assert not files.variables_unreadable
        assert PolicyLevel.OWNER  # levels are unchanged


class TestWarning:
    def test_each_refused_kind_is_reported_once(self, capsys) -> None:
        ctx = SimpleNamespace(
            policy_files=PolicyFiles(variables_refused=frozenset({"orgs"}))
        )
        _warn_unreadable_variables(ctx, frozenset())
        first = capsys.readouterr().out
        assert "the organisation's DEPENDAMERGE_BOTS" in first
        _warn_unreadable_variables(ctx, frozenset({"orgs"}))
        assert capsys.readouterr().out == ""
