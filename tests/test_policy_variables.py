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
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dependamerge.bot_identity import is_automation_author
from dependamerge.bot_policy import BotPolicy, PolicyLevel
from dependamerge.bot_policy_files import PolicyFiles, apply_policy_files
from dependamerge.cli._policy_files import (
    _load_source_policy_files,
    _warn_unreadable_variables,
)
from dependamerge.github_graphql import ORG_REPOS_ONLY, USER_REPOS_ONLY
from dependamerge.github_service import GitHubService
from dependamerge.github_service._policy_variables import PolicyVariableError
from dependamerge.models import PullRequestInfo

CODE_MONKEY = "lf-releng-code-monkey-bot[bot]"


def _error(status: int, path: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"https://api.github.com{path}")
    return httpx.HTTPStatusError(
        "error", request=request, response=httpx.Response(status, request=request)
    )


class _Api:
    """Answers variable reads from a table.

    An HTTP status, a raw body (a dict or a list), or the value of a
    well-formed answer.
    """

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.paths: list[str] = []

    async def get(self, path: str, params: Any = None) -> Any:
        self.paths.append(path)
        answer = self.answers.get(path.split("/actions/")[0], 404)
        if isinstance(answer, int):
            raise _error(answer, path)
        if isinstance(answer, dict | list):
            return answer
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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [500, 502, 422])
    async def test_other_failures_stop_the_run(self, status: int) -> None:
        # Taken for absent, a variable that excludes an App could not
        # protect that App's PRs (Copilot review on #567).
        svc, _api = _service({"/repos/acme/widget": status})
        with pytest.raises(PolicyVariableError, match="acme/widget"):
            await svc.fetch_policy_variables("acme", [_pr(CODE_MONKEY, "acme/widget")])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [{}, {"value": None}, {"value": 5}, []])
    async def test_an_answer_without_a_value_stops_the_run(self, body: Any) -> None:
        # Only a 404 says there is no variable: read as absent, a garbled
        # answer would drop any exclusion it held (Copilot review on #567).
        svc, _api = _service({"/repos/acme/widget": body})
        with pytest.raises(PolicyVariableError, match="held no value"):
            await svc.fetch_policy_variables("acme", [_pr(CODE_MONKEY, "acme/widget")])
        assert "acme/widget" not in svc.policy_files.repository_variables


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


ORG_WARNING = (
    "⚠️ Token cannot read the organisation DEPENDAMERGE_BOTS variable\n"
    "   Revise the token permission, or create a configuration file\n"
    "   to suppress this warning: .github/dependamerge/config.json\n"
)

REPO_WARNING = (
    "⚠️ Token cannot read repository DEPENDAMERGE_BOTS variables\n"
    "   Revise the token permission, or create a configuration file\n"
    "   to suppress this warning: .github/dependamerge/config.json\n"
)


def _refused(*kinds: str, **files: Any) -> SimpleNamespace:
    return SimpleNamespace(
        policy_files=PolicyFiles(variables_refused=frozenset(kinds), **files)
    )


class TestWarning:
    """The warning appears only where a refused variable could matter.

    That is, for an App outside the built-in ones, at a level no policy
    file covers; and each kind at most once per run.
    """

    @pytest.mark.parametrize(
        "author",
        ["dependabot[bot]", "renovate[bot]", "pre-commit-ci[bot]", "octocat"],
    )
    def test_no_warning_for_built_in_apps_or_people(self, capsys, author) -> None:
        ctx = _refused("orgs", "repos")
        _warn_unreadable_variables(ctx, [_pr(author, "acme/widget")])
        assert capsys.readouterr().out == ""

    def test_no_warning_without_pull_requests(self, capsys) -> None:
        _warn_unreadable_variables(_refused("orgs", "repos"), [])
        assert capsys.readouterr().out == ""

    def test_an_organisation_warning_for_another_app(self, capsys) -> None:
        ctx = _refused("orgs")
        _warn_unreadable_variables(ctx, [_pr(CODE_MONKEY, "acme/widget")])
        out = capsys.readouterr().out
        assert out == ORG_WARNING
        assert all(len(line) <= 80 for line in out.splitlines())

    def test_a_repository_warning_for_another_app(self, capsys) -> None:
        ctx = _refused("repos")
        _warn_unreadable_variables(ctx, [_pr(CODE_MONKEY, "acme/widget")])
        out = capsys.readouterr().out
        assert out == REPO_WARNING
        assert all(len(line) <= 80 for line in out.splitlines())

    def test_one_block_per_refused_kind(self, capsys) -> None:
        ctx = _refused("orgs", "repos")
        _warn_unreadable_variables(ctx, [_pr(CODE_MONKEY, "acme/widget")])
        assert capsys.readouterr().out == ORG_WARNING + REPO_WARNING

    def test_an_owner_file_silences_the_organisation_warning(self, capsys) -> None:
        ctx = _refused("orgs", owners={"acme": _blob({"exclude_bots": []})})
        _warn_unreadable_variables(ctx, [_pr(CODE_MONKEY, "acme/widget")])
        assert capsys.readouterr().out == ""

    def test_a_repository_file_silences_the_repository_warning(self, capsys) -> None:
        ctx = _refused(
            "repos", repositories={"acme/widget": _blob({"exclude_bots": []})}
        )
        _warn_unreadable_variables(ctx, [_pr(CODE_MONKEY, "Acme/Widget")])
        assert capsys.readouterr().out == ""

    def test_each_repository_without_a_file_needs_its_variable(self, capsys) -> None:
        ctx = _refused(
            "repos",
            owners={"acme": _blob({})},
            repositories={"acme/widget": _blob({}), "acme/gadget": None},
        )
        prs = [_pr(CODE_MONKEY, "acme/widget"), _pr(CODE_MONKEY, "acme/gadget")]
        _warn_unreadable_variables(ctx, prs)
        assert capsys.readouterr().out == REPO_WARNING

    def test_an_owner_file_leaves_the_repository_warning(self, capsys) -> None:
        ctx = _refused("orgs", "repos", owners={"acme": _blob({})})
        _warn_unreadable_variables(ctx, [_pr(CODE_MONKEY, "acme/widget")])
        assert capsys.readouterr().out == REPO_WARNING

    def test_each_refused_kind_is_reported_once(self, capsys) -> None:
        ctx = _refused("orgs")
        prs = [_pr(CODE_MONKEY, "acme/widget")]
        _warn_unreadable_variables(ctx, prs)
        assert capsys.readouterr().out == ORG_WARNING
        _warn_unreadable_variables(ctx, prs)
        assert capsys.readouterr().out == ""

    def test_a_kind_held_back_is_reported_once_it_matters(self, capsys) -> None:
        ctx = _refused("orgs")
        _warn_unreadable_variables(ctx, [_pr("dependabot[bot]", "acme/widget")])
        assert capsys.readouterr().out == ""
        _warn_unreadable_variables(ctx, [_pr(CODE_MONKEY, "acme/gadget")])
        assert capsys.readouterr().out == ORG_WARNING

    def test_the_source_pull_request_decides_on_load(self, capsys) -> None:
        refused = PolicyFiles(variables_refused=frozenset({"orgs"}))
        client = MagicMock()
        client.get_policy_files.return_value = refused
        ctx = SimpleNamespace(
            github_client=client,
            owner="acme",
            repo_name="widget",
            token="t",
            host="",
            policy_files=PolicyFiles(),
            bot_policy=BotPolicy(),
            source_pr=_pr(CODE_MONKEY, "acme/widget"),
        )
        _load_source_policy_files(ctx)
        assert capsys.readouterr().out == ORG_WARNING
        ctx.source_pr = _pr("dependabot[bot]", "acme/widget")
        ctx.policy_variables_warned = frozenset()
        _load_source_policy_files(ctx)
        assert capsys.readouterr().out == ""


class TestSearchWarning:
    """merge and close judge the warning by every App PR their search saw.

    Not only the matches: an App no policy admits, or whose PR does not
    match, is still one an unreadable variable might have decided
    (Copilot review on #585).
    """

    @staticmethod
    def _service(seen: list[PullRequestInfo]) -> MagicMock:
        svc = MagicMock()
        svc.find_similar_prs = AsyncMock(return_value=[])
        svc.close = AsyncMock()
        svc.policy_files = PolicyFiles(variables_refused=frozenset({"orgs"}))
        svc.seen_app_prs = tuple(seen)
        return svc

    @classmethod
    def _merge(cls, seen: list[PullRequestInfo]) -> None:
        from dependamerge.cli._context import _MergeContext
        from dependamerge.cli._merge_scan import _scan_and_find_similar

        ctx = _MergeContext(
            pr_url="https://github.com/acme/home/pull/1",
            no_confirm=True,
            similarity_threshold=0.8,
            merge_method="merge",
            token="t",
            override=None,
            no_fix=False,
            merge_timeout=300.0,
            show_progress=False,
            debug_matching=False,
            dismiss_copilot=False,
            force="none",
            verbose=False,
            no_netrc=False,
            netrc_file=None,
            netrc_optional=True,
            github2gerrit_mode="ignore",
        )
        ctx.owner = "acme"
        ctx.github_client = MagicMock()
        ctx.github_client.is_automation_author.side_effect = is_automation_author
        ctx.source_pr = _pr("dependabot[bot]", "acme/home")
        ctx.comparator = MagicMock()
        with patch(
            "dependamerge.github_service.GitHubService",
            return_value=cls._service(seen),
        ):
            _scan_and_find_similar(ctx)

    @classmethod
    def _close(cls, seen: list[PullRequestInfo]) -> None:
        from dependamerge.cli import _close

        client = MagicMock()
        client.is_automation_author.side_effect = is_automation_author
        ctx = _close._CloseContext(
            token="t",
            github_client=client,
            host="github.com",
            owner="acme",
            repo_name="home",
            pr_number=1,
            source_pr=_pr("dependabot[bot]", "acme/home"),
            progress_tracker=None,
        )
        with patch(
            "dependamerge.github_service.GitHubService",
            return_value=cls._service(seen),
        ):
            _close._find_similar_prs_for_close(ctx, MagicMock(), debug_matching=False)

    @pytest.mark.parametrize("run", ["_merge", "_close"])
    def test_an_unmatched_app_pr_warns(self, capsys, run) -> None:
        getattr(self, run)([_pr(CODE_MONKEY, "acme/gadget")])
        assert ORG_WARNING in capsys.readouterr().out

    @pytest.mark.parametrize("run", ["_merge", "_close"])
    def test_only_built_in_app_prs_stay_silent(self, capsys, run) -> None:
        getattr(self, run)([_pr("renovate[bot]", "acme/gadget")])
        assert "Token cannot read" not in capsys.readouterr().out
