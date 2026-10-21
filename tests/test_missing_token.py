# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""A missing GitHub token is a configuration error, never a traceback.

Every GitHub ``merge`` and ``close`` path, and ``blocked --fix``, report
it the same way, whether or not the run's bot policy names an App.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.cli._blocked_cmd import _run_blocked_fix
from dependamerge.cli._bot_scope import _lookup_apps
from dependamerge.error_codes import ExitCode

PR_URL = "https://github.com/acme/widget/pull/7"
MESSAGE = "GitHub token required"
DETAILS = "Provide --token or set the GITHUB_TOKEN environment variable"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


def _assert_reported(result) -> None:
    assert result.exit_code == ExitCode.CONFIGURATION_ERROR, result.output
    assert MESSAGE in result.output
    assert DETAILS in result.output
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, ValueError)


@pytest.mark.parametrize(
    "args",
    [
        [PR_URL],
        ["--include-bots", "some-app", PR_URL],
        ["--exclude-bots", "some-app", PR_URL],
        ["acme/widget"],
        ["--include-bots", "some-app", "acme/widget"],
        ["acme"],
        ["--include-bots", "some-app", "acme"],
    ],
)
def test_merge_without_a_token_is_a_configuration_error(args: list[str]) -> None:
    _assert_reported(runner.invoke(app, ["merge", "--dry-run", *args]))


def test_merge_with_only_a_user_policy_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "dependamerge").mkdir()
    (tmp_path / "dependamerge" / "config.json").write_text(
        json.dumps({"include_bots": ["some-app"]}), encoding="utf-8"
    )
    _assert_reported(runner.invoke(app, ["merge", "--dry-run", PR_URL]))


@pytest.mark.parametrize("args", [[PR_URL], ["--include-bots", "some-app", PR_URL]])
def test_close_without_a_token_is_a_configuration_error(args: list[str]) -> None:
    _assert_reported(runner.invoke(app, ["close", "--dry-run", *args]))


def test_gerrit_merge_needs_no_github_token() -> None:
    with patch("dependamerge.cli._merge_cmd._dispatch_gerrit") as dispatch:
        result = runner.invoke(
            app,
            [
                "merge",
                "--gerrit-host",
                "gerrit.example.org",
                "https://gerrit.example.org/r/c/acme/widget/+/123",
            ],
        )
    assert result.exit_code == 0, result.output
    dispatch.assert_called_once()


def test_blocked_fix_command_without_a_token_is_a_configuration_error() -> None:
    # The scan needs the token before the fix does, so it has to be
    # resolved before scanning or the scan reports it as exit 3.
    _assert_reported(runner.invoke(app, ["blocked", "acme", "--fix", "--no-progress"]))


def test_blocked_fix_without_a_token_is_a_configuration_error(capsys) -> None:
    scan = SimpleNamespace(
        unmergeable_prs=[
            SimpleNamespace(
                repository="acme/widget",
                pr_number=7,
                reasons=[SimpleNamespace(type="merge_conflict")],
            )
        ]
    )
    with pytest.raises(SystemExit) as raised:
        _run_blocked_fix(scan, None, None, None, None)
    assert raised.value.code == ExitCode.CONFIGURATION_ERROR
    out = capsys.readouterr().out
    assert MESSAGE in out
    assert DETAILS in out


def test_lookup_without_a_client_leaves_every_login_unknown() -> None:
    logins = ["renovate[bot]", "some-app[bot]"]
    with patch(
        "dependamerge.github_async.GitHubAsync",
        side_effect=ValueError("GitHub token is required. Set GITHUB_TOKEN."),
    ):
        assert _lookup_apps(logins, None, "github.com") == dict.fromkeys(logins)
