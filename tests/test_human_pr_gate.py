# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the human-authored source PR gate.

Regression cover for lfreleng-actions/dependamerge#395, where pointing
dependamerge at a human-authored PR exited 0 having done nothing, and for
#565, where ``--include-human-prs`` and ``--override`` each authorised a
human-authored source alone, so the operator met a hex string whichever
was given.  The rule now: the flag brings human PRs into scope, the
override confirms them, and each refusal has its own exit code.
"""

from unittest.mock import MagicMock

import pytest
import typer

from dependamerge.bot_policy import BotPolicy
from dependamerge.cli import _generate_override_sha, _validate_automation_author
from dependamerge.cli._human_gate import (
    _human_set_override,
    _require_override_pairing,
)
from dependamerge.error_codes import ExitCode
from dependamerge.models import PullRequestInfo

FIRST_COMMIT_LINE = "Fix: Env-mediate job status in run block"
PERSON = "ModeSevenIndustrialSolutions"


def _ctx(*, author, include_human_prs=False, override=None, dry_run=False):
    """A merge context whose source PR is authored by *author*."""
    source_pr = MagicMock()
    source_pr.author = author
    source_pr.title = "Fix: Env-mediate job status in run block"
    source_pr.body = "body"
    source_pr.html_url = "https://github.com/lfreleng-actions/x/pull/63"
    source_pr.repository_full_name = "lfreleng-actions/x"

    github_client = MagicMock()
    github_client.is_automation_author.side_effect = lambda a: a.endswith("[bot]")
    github_client.get_pull_request_commits.return_value = [FIRST_COMMIT_LINE]

    ctx = MagicMock()
    ctx.github_client = github_client
    ctx.source_pr = source_pr
    ctx.owner = "lfreleng-actions"
    ctx.repo_name = "tailscale-openstack-bastion-action"
    ctx.pr_number = 63
    ctx.include_human_prs = include_human_prs
    ctx.override = override
    ctx.dry_run = dry_run
    ctx.bot_policy = BotPolicy()
    return ctx


def _expected_sha(ctx):
    return _generate_override_sha(ctx.source_pr, FIRST_COMMIT_LINE)


class TestAutomationSource:
    """Apps the bot policy admits are unaffected by the gate."""

    def test_bot_author_passes_without_flags(self):
        _validate_automation_author(_ctx(author="pre-commit-ci[bot]"))

    def test_bot_author_does_not_consult_commits(self):
        # The gate should short-circuit before fetching commit messages;
        # an extra API call per PR is wasteful at organisation scale.
        ctx = _ctx(author="dependabot[bot]")
        _validate_automation_author(ctx)
        ctx.github_client.get_pull_request_commits.assert_not_called()


class TestHumanSourceWithoutTheFlag:
    """Without --include-human-prs the run stops, naming the flag."""

    def test_exits_validation_error(self):
        with pytest.raises(SystemExit) as excinfo:
            _validate_automation_author(_ctx(author=PERSON))
        # #395: this once exited 0, indistinguishable from success.
        assert excinfo.value.code == ExitCode.VALIDATION_ERROR

    def test_names_the_flag_and_offers_no_override(self, capsys):
        ctx = _ctx(author=PERSON)
        with pytest.raises(SystemExit):
            _validate_automation_author(ctx)
        output = capsys.readouterr().out
        assert "--include-human-prs" in output
        assert _expected_sha(ctx) not in output

    def test_costs_no_api_call(self):
        ctx = _ctx(author=PERSON)
        with pytest.raises(SystemExit):
            _validate_automation_author(ctx)
        ctx.github_client.get_pull_request_commits.assert_not_called()

    def test_an_override_alone_does_not_authorise(self):
        # The CLI refuses this pairing up front; the gate holds anyway.
        ctx = _ctx(author=PERSON)
        ctx.override = _expected_sha(ctx)
        with pytest.raises(SystemExit) as excinfo:
            _validate_automation_author(ctx)
        assert excinfo.value.code == ExitCode.VALIDATION_ERROR


class TestHumanSourceWithTheFlag:
    """--include-human-prs needs the override that confirms the PR."""

    def test_missing_override_exits_override_required(self, capsys):
        ctx = _ctx(author=PERSON, include_human_prs=True)
        with pytest.raises(SystemExit) as excinfo:
            _validate_automation_author(ctx)
        assert excinfo.value.code == ExitCode.OVERRIDE_REQUIRED
        assert f"--override {_expected_sha(ctx)}" in capsys.readouterr().out

    def test_matching_override_proceeds(self, capsys):
        ctx = _ctx(author=PERSON, include_human_prs=True)
        ctx.override = _expected_sha(ctx)
        _validate_automation_author(ctx)
        assert "--override confirms" in capsys.readouterr().out

    def test_override_matches_whatever_its_case_and_spacing(self):
        ctx = _ctx(author=PERSON, include_human_prs=True)
        ctx.override = f"  {_expected_sha(ctx).upper()} "
        _validate_automation_author(ctx)

    def test_wrong_override_exits_validation_error(self, capsys):
        ctx = _ctx(author=PERSON, include_human_prs=True, override="0" * 16)
        with pytest.raises(SystemExit) as excinfo:
            _validate_automation_author(ctx)
        assert excinfo.value.code == ExitCode.VALIDATION_ERROR
        assert _expected_sha(ctx) in capsys.readouterr().out

    def test_dry_run_reports_the_value_instead_of_requiring_it(self, capsys):
        ctx = _ctx(author=PERSON, include_human_prs=True, dry_run=True)
        _validate_automation_author(ctx)
        assert f"--override {_expected_sha(ctx)}" in capsys.readouterr().out

    def test_no_prompt_is_issued(self, monkeypatch):
        # The gate must never block on input: with --no-confirm the run
        # has to proceed unattended.  Any prompt here would hang CI.
        def _explode(*args, **kwargs):
            raise AssertionError("the author gate must not prompt")

        monkeypatch.setattr(typer, "prompt", _explode)
        monkeypatch.setattr(typer, "confirm", _explode)
        ctx = _ctx(author=PERSON, include_human_prs=True)
        ctx.override = _expected_sha(ctx)
        _validate_automation_author(ctx)


class TestOverridePairing:
    def test_override_without_the_flag_is_a_configuration_error(self):
        with pytest.raises(SystemExit) as excinfo:
            _require_override_pairing("0123456789abcdef", False)
        assert excinfo.value.code == ExitCode.CONFIGURATION_ERROR

    @pytest.mark.parametrize(
        ("override", "include"),
        [(None, False), (None, True), ("0123456789abcdef", True), ("  ", False)],
    )
    def test_other_pairings_pass(self, override, include):
        _require_override_pairing(override, include)


def _pr(number: int, head_sha: str = "a" * 40) -> PullRequestInfo:
    return PullRequestInfo(
        number=number,
        title="t",
        body=None,
        author="a-person",
        head_sha=head_sha,
        base_branch="main",
        head_branch="topic",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name="acme/widget",
        html_url=f"https://github.com/acme/widget/pull/{number}",
    )


class TestHumanSetOverride:
    """Owner and repository runs confirm exactly the PRs they found."""

    def test_order_does_not_matter(self):
        assert _human_set_override("acme", [_pr(1), _pr(2)]) == (
            _human_set_override("acme", [_pr(2), _pr(1)])
        )

    def test_another_pr_changes_it(self):
        assert _human_set_override("acme", [_pr(1)]) != (
            _human_set_override("acme", [_pr(1), _pr(2)])
        )

    def test_a_new_push_changes_it(self):
        assert _human_set_override("acme", [_pr(1, "a" * 40)]) != (
            _human_set_override("acme", [_pr(1, "b" * 40)])
        )

    def test_another_scope_changes_it(self):
        assert _human_set_override("acme", [_pr(1)]) != (
            _human_set_override("acme/widget", [_pr(1)])
        )
