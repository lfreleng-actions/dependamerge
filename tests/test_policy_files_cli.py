# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""How a run applies repository and owner bot policy files."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dependamerge.bot_policy import BotPolicy, run_policy_source
from dependamerge.bot_policy_files import PolicyFiles
from dependamerge.cli._bot_scope import _remove_out_of_scope_bots
from dependamerge.cli._policy_files import (
    _capture_policy_files,
    _load_source_policy_files,
    _policy_with_files,
)
from dependamerge.error_codes import ExitCode
from dependamerge.models import PullRequestInfo

CODE_MONKEY = "lf-releng-code-monkey-bot[bot]"
LOOKUP = "dependamerge.cli._bot_scope._lookup_apps"


def _blob(document: Any) -> dict[str, Any]:
    text = document if isinstance(document, str) else json.dumps(document)
    return {
        "text": text,
        "byteSize": len(text),
        "isTruncated": False,
        "isBinary": False,
    }


def _files(
    owner: Any = None, **repositories: Any
) -> PolicyFiles:  # repositories: name -> document
    return PolicyFiles(
        owners={} if owner is None else {"acme": _blob(owner)},
        repositories={f"acme/{name}": _blob(doc) for name, doc in repositories.items()},
    )


def _pr(author: str, number: int = 1, repo: str = "acme/widget") -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
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
        html_url=f"https://github.com/{repo}/pull/{number}",
    )


@pytest.fixture
def apps_exist():
    """Every App a file names exists, unless a test says otherwise."""
    with patch(LOOKUP, side_effect=lambda logins, *_: dict.fromkeys(logins, True)) as m:
        yield m


class TestPrecedence:
    def test_owner_file_brings_an_app_into_scope(self, apps_exist) -> None:
        policy = _policy_with_files(
            BotPolicy(),
            _files({"include_bots": ["lf-releng-code-monkey-bot"]}),
            "t",
            "",
        )
        decision = policy.decide(CODE_MONKEY, "acme/widget")
        assert decision.in_scope
        assert decision.reason == "included by acme/.github"

    def test_repository_file_wins_for_its_repository(self, apps_exist) -> None:
        files = _files(
            {"include_bots": [CODE_MONKEY]}, widget={"exclude_bots": [CODE_MONKEY]}
        )
        policy = _policy_with_files(BotPolicy(), files, "t", "")
        assert not policy.admits(CODE_MONKEY, "acme/widget")
        assert policy.admits(CODE_MONKEY, "acme/other")

    def test_no_files_changes_nothing(self) -> None:
        with patch(LOOKUP) as lookup:
            policy = _policy_with_files(BotPolicy(), PolicyFiles(), "t", "")
        assert policy == BotPolicy()
        lookup.assert_not_called()


class TestInvalidFiles:
    def test_invalid_owner_file_stops_the_run(self) -> None:
        with pytest.raises(SystemExit) as raised:
            _policy_with_files(BotPolicy(), _files({"exlude_bots": []}), "t", "")
        assert raised.value.code == ExitCode.CONFIGURATION_ERROR

    def test_invalid_repository_file_takes_its_prs_out_of_scope(self, capsys) -> None:
        policy = _policy_with_files(BotPolicy(), _files(widget="{not json"), "t", "")
        prs = [
            _pr("dependabot[bot]", 1),
            _pr("octocat", 2),
            _pr("dependabot[bot]", 3, repo="acme/other"),
        ]
        kept = _remove_out_of_scope_bots(policy, prs)
        assert [pr.number for pr in kept] == [3]
        assert "Skipped 2 PR(s) in acme/widget" in capsys.readouterr().out


class TestUnknownApps:
    def test_unknown_exclusion_invalidates_a_repository_file(self) -> None:
        files = _files(widget={"exclude_bots": ["renovat"]})
        with patch(LOOKUP, return_value={"renovat[bot]": False}):
            policy = _policy_with_files(BotPolicy(), files, "t", "")
        assert policy.repository_error("acme/widget") is not None

    def test_unknown_exclusion_in_the_owner_file_stops_the_run(self) -> None:
        with patch(LOOKUP, return_value={"renovat[bot]": False}):
            with pytest.raises(SystemExit) as raised:
                _policy_with_files(
                    BotPolicy(), _files({"exclude_bots": ["renovat"]}), "t", ""
                )
        assert raised.value.code == ExitCode.CONFIGURATION_ERROR

    def test_apps_the_flags_named_are_not_looked_up_again(self) -> None:
        policy = BotPolicy(run=run_policy_source(["renovate"], None))
        files = _files({"include_bots": ["renovate", "other-app"]})
        with patch(LOOKUP, return_value={"other-app[bot]": True}) as lookup:
            _policy_with_files(policy, files, "t", "")
        lookup.assert_called_once_with(["other-app[bot]"], "t", "")


class TestOverrides:
    def test_flags_overruling_a_file_are_reported(self, apps_exist, capsys) -> None:
        policy = BotPolicy(run=run_policy_source([CODE_MONKEY], None))
        policy = _policy_with_files(
            policy, _files({"exclude_bots": [CODE_MONKEY]}), "t", ""
        )
        kept = _remove_out_of_scope_bots(policy, [_pr(CODE_MONKEY)])
        assert len(kept) == 1
        assert f"overrule acme/.github for {CODE_MONKEY}" in capsys.readouterr().out


class TestCapture:
    def test_a_service_files_are_kept(self) -> None:
        ctx = SimpleNamespace(policy_files=PolicyFiles())
        svc = SimpleNamespace(policy_files=_files({"include_bots": ["x"]}))
        _capture_policy_files(ctx, svc)
        assert "acme" in ctx.policy_files.owners

    def test_a_mocked_service_adds_nothing(self) -> None:
        ctx = SimpleNamespace(policy_files=PolicyFiles())
        _capture_policy_files(ctx, MagicMock())
        assert ctx.policy_files == PolicyFiles()


class TestSourceFiles:
    def _ctx(self, files: Any) -> SimpleNamespace:
        client = MagicMock()
        client.get_policy_files.return_value = files
        return SimpleNamespace(
            github_client=client,
            owner="acme",
            repo_name="widget",
            token="t",
            host="",
            policy_files=PolicyFiles(),
            bot_policy=BotPolicy(),
        )

    def test_files_apply_before_the_source_is_judged(self, apps_exist) -> None:
        ctx = self._ctx(_files(widget={"include_bots": [CODE_MONKEY]}))
        _load_source_policy_files(ctx)
        assert ctx.bot_policy.admits(CODE_MONKEY, "acme/widget")
        ctx.github_client.get_policy_files.assert_called_once_with("acme", "widget")

    def test_invalid_source_repository_file_stops_the_run(self) -> None:
        ctx = self._ctx(_files(widget=["not", "an", "object"]))
        with pytest.raises(SystemExit) as raised:
            _load_source_policy_files(ctx)
        assert raised.value.code == ExitCode.CONFIGURATION_ERROR
