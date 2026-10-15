# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the bot policy: which GitHub Apps a run may act on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dependamerge.bot_policy import (
    BUILTIN_BOTS,
    BotPolicy,
    BotPolicyError,
    ListedBot,
    PolicyLevel,
    parse_policy_document,
    run_policy_source,
)
from dependamerge.bot_policy_files import load_user_policy, user_config_path

CODE_MONKEY = "lf-releng-code-monkey-bot[bot]"


def _document(**lists: object) -> str:
    return json.dumps(lists)


def _owner_source(**lists: object):
    return parse_policy_document(
        _document(**lists), label="acme/.github", level=PolicyLevel.OWNER
    )


def _repository_source(**lists: object):
    return parse_policy_document(
        _document(**lists), label="acme/widget", level=PolicyLevel.REPOSITORY
    )


class TestDefaults:
    """With no sources, exactly the built-in Apps are in scope."""

    @pytest.mark.parametrize("login", sorted(BUILTIN_BOTS))
    def test_builtin_apps_are_in_scope(self, login: str) -> None:
        decision = BotPolicy().decide(login, "acme/widget")
        assert decision.in_scope
        assert decision.reason == "built-in"

    @pytest.mark.parametrize(
        "login",
        [CODE_MONKEY, "github-actions[bot]", "copilot-swe-agent[bot]"],
    )
    def test_other_apps_are_out_of_scope(self, login: str) -> None:
        decision = BotPolicy().decide(login, "acme/widget")
        assert not decision.in_scope
        assert decision.reason == "not included by any policy"

    @pytest.mark.parametrize("login", ["octocat", "renovate", "", None])
    def test_people_are_never_in_scope(self, login: str | None) -> None:
        # Whether a person's PR may be acted on is decided elsewhere.
        decision = BotPolicy().decide(login, "acme/widget")
        assert not decision.in_scope
        assert decision.reason == "not a GitHub App"


class TestEntries:
    @pytest.mark.parametrize(
        "entry",
        [
            "lf-releng-code-monkey-bot",
            "lf-releng-code-monkey-bot[bot]",
            "app/lf-releng-code-monkey-bot",
            "LF-Releng-Code-Monkey-Bot",
        ],
    )
    def test_every_form_names_the_same_app(self, entry: str) -> None:
        source = run_policy_source([entry], None)
        assert source is not None
        assert source.include == frozenset({CODE_MONKEY})

    def test_separators_and_repeats(self) -> None:
        source = run_policy_source(["a, b", "c\nd"], ["e"])
        assert source is not None
        assert source.include == {"a[bot]", "b[bot]", "c[bot]", "d[bot]"}
        assert source.exclude == {"e[bot]"}

    def test_no_flags_is_no_source(self) -> None:
        assert run_policy_source(None, None) is None
        assert run_policy_source([], []) is None

    @pytest.mark.parametrize("entry", ["not@app", "-leading", "trailing-", "a/b"])
    def test_invalid_names_are_refused(self, entry: str) -> None:
        with pytest.raises(BotPolicyError, match="not a GitHub App name"):
            run_policy_source([entry], None)

    def test_both_lists_naming_one_app_is_refused(self) -> None:
        with pytest.raises(BotPolicyError, match="both included and excluded"):
            run_policy_source(["renovate"], ["renovate[bot]"])

    def test_wildcard_only_includes(self) -> None:
        with pytest.raises(BotPolicyError, match="only valid in include_bots"):
            run_policy_source(None, ["*"])

    def test_every_problem_is_reported_at_once(self) -> None:
        with pytest.raises(BotPolicyError) as raised:
            run_policy_source(["bad_name"], ["*"])
        assert len(raised.value.problems) == 2


class TestDocuments:
    def test_lists_may_be_strings(self) -> None:
        source = _owner_source(include_bots="a, b", exclude_bots=["c"])
        assert source.include == {"a[bot]", "b[bot]"}
        assert source.exclude == {"c[bot]"}

    def test_unknown_keys_are_refused(self) -> None:
        # A misspelt exclusion must stop, not quietly leave a bot in scope.
        with pytest.raises(BotPolicyError, match="unknown key"):
            _owner_source(exlude_bots=["renovate"])

    @pytest.mark.parametrize("text", ["[]", '"x"', "{", "null"])
    def test_documents_must_be_json_objects(self, text: str) -> None:
        with pytest.raises(BotPolicyError):
            parse_policy_document(text, label="x", level=PolicyLevel.OWNER)

    def test_list_values_must_be_strings(self) -> None:
        with pytest.raises(BotPolicyError, match="list of strings"):
            _owner_source(include_bots=[1, 2])

    def test_oversized_documents_are_refused(self) -> None:
        text = _document(include_bots=["a"] * 9000)
        with pytest.raises(BotPolicyError, match="larger than"):
            parse_policy_document(text, label="x", level=PolicyLevel.OWNER)


class TestPrecedence:
    """The most specific source that names an App decides for it."""

    def test_run_flags_beat_the_user_file(self) -> None:
        run = run_policy_source(["renovate"], None)
        user = parse_policy_document(
            _document(exclude_bots=["renovate"]), label="~/x", level=PolicyLevel.USER
        )
        decision = BotPolicy(run=run, user=user).decide("renovate[bot]", "acme/w")
        assert decision.in_scope
        assert decision.reason == "included by --include-bots"

    def test_repository_beats_owner_beats_user(self) -> None:
        user = parse_policy_document(
            _document(include_bots=[CODE_MONKEY]), label="~/x", level=PolicyLevel.USER
        )
        owner = _owner_source(exclude_bots=[CODE_MONKEY])
        repository = parse_policy_document(
            _document(include_bots=[CODE_MONKEY]),
            label="acme/widget",
            level=PolicyLevel.REPOSITORY,
        )
        policy = BotPolicy(
            user=user,
            owners={"acme": owner},
            repositories={"acme/widget": repository},
        )
        assert policy.decide(CODE_MONKEY, "acme/widget").reason == (
            "included by acme/widget"
        )
        assert policy.decide(CODE_MONKEY, "acme/other").reason == (
            "excluded by acme/.github"
        )
        assert policy.decide(CODE_MONKEY, "elsewhere/x").reason == "included by ~/x"

    def test_owner_and_repository_match_case_insensitively(self) -> None:
        policy = BotPolicy(owners={"acme": _owner_source(include_bots=[CODE_MONKEY])})
        assert policy.admits(CODE_MONKEY, "ACME/Widget")

    def test_builtin_apps_can_be_excluded(self) -> None:
        run = run_policy_source(None, ["dependabot"])
        decision = BotPolicy(run=run).decide("dependabot[bot]", "acme/w")
        assert not decision.in_scope
        assert decision.reason == "excluded by --exclude-bots"

    def test_wildcard_includes_every_app_but_yields_to_exclusions(self) -> None:
        run = run_policy_source(["*"], ["renovate"])
        policy = BotPolicy(run=run)
        assert policy.admits(CODE_MONKEY, "acme/w")
        assert not policy.admits("renovate[bot]", "acme/w")
        assert not policy.admits("octocat", "acme/w")

    def test_decisions_without_a_repository_use_run_and_user_only(self) -> None:
        policy = BotPolicy(owners={"acme": _owner_source(include_bots=[CODE_MONKEY])})
        assert not policy.admits(CODE_MONKEY)


class TestOverrides:
    """A run's flags overrule a repository's or owner's file, and say so."""

    def test_a_flag_against_the_repository_file_names_it(self) -> None:
        policy = BotPolicy(
            run=run_policy_source(None, [CODE_MONKEY]),
            repositories={
                "acme/widget": _repository_source(include_bots=[CODE_MONKEY])
            },
        )
        decision = policy.decide(CODE_MONKEY, "acme/widget")
        assert not decision.in_scope
        assert decision.reason == "excluded by --exclude-bots"
        assert decision.overrides == "acme/widget"

    def test_a_flag_against_the_owner_file_names_it(self) -> None:
        policy = BotPolicy(
            run=run_policy_source([CODE_MONKEY], None),
            owners={"acme": _owner_source(exclude_bots=[CODE_MONKEY])},
        )
        decision = policy.decide(CODE_MONKEY, "acme/widget")
        assert decision.in_scope
        assert decision.overrides == "acme/.github"

    def test_an_owner_wildcard_names_every_app(self) -> None:
        policy = BotPolicy(
            run=run_policy_source(None, ["renovate"]),
            owners={"acme": _owner_source(include_bots="*")},
        )
        assert policy.decide("renovate[bot]", "acme/w").overrides == "acme/.github"

    @pytest.mark.parametrize(
        ("run", "repository", "owner", "user"),
        [
            # The run agrees with the file that would decide.
            ([CODE_MONKEY], {"include_bots": [CODE_MONKEY]}, None, None),
            # Only the owner disagrees, and the repository outranks it.
            (
                [CODE_MONKEY],
                {"include_bots": [CODE_MONKEY]},
                {"exclude_bots": [CODE_MONKEY]},
                None,
            ),
            # The run does not name the App, so a file decides.
            (["renovate"], {"include_bots": [CODE_MONKEY]}, None, None),
            # Overruling the user's own file is not worth a mention.
            ([CODE_MONKEY], None, None, {"exclude_bots": [CODE_MONKEY]}),
        ],
    )
    def test_nothing_is_overridden_otherwise(
        self,
        run: list[str],
        repository: dict[str, list[str]] | None,
        owner: dict[str, list[str]] | None,
        user: dict[str, list[str]] | None,
    ) -> None:
        policy = BotPolicy(
            run=run_policy_source(run, None),
            repositories=(
                {"acme/widget": _repository_source(**repository)} if repository else {}
            ),
            owners={"acme": _owner_source(**owner)} if owner else {},
            user=(
                parse_policy_document(
                    _document(**user), label="~/x", level=PolicyLevel.USER
                )
                if user
                else None
            ),
        )
        assert policy.decide(CODE_MONKEY, "acme/widget").overrides is None


class TestInvalidRepositoryFile:
    def test_nothing_is_in_scope_there_whatever_the_flags(self) -> None:
        policy = BotPolicy(
            run=run_policy_source(["*"], None),
            invalid_repositories={"acme/widget": "not valid JSON"},
        )
        decision = policy.decide("dependabot[bot]", "Acme/Widget")
        assert not decision.in_scope
        assert decision.reason == "Acme/Widget's bot policy file is invalid"
        assert policy.repository_error("ACME/widget") == "not valid JSON"

    def test_other_repositories_are_unaffected(self) -> None:
        policy = BotPolicy(invalid_repositories={"acme/widget": "not valid JSON"})
        assert policy.admits("dependabot[bot]", "acme/other")
        assert policy.repository_error("acme/other") is None
        assert policy.repository_error(None) is None


class TestListed:
    def test_exclusions_come_first_and_each_names_its_source(self) -> None:
        run = run_policy_source(["b"], ["a"])
        user = parse_policy_document(
            _document(include_bots=["b"]), label="~/x", level=PolicyLevel.USER
        )
        policy = BotPolicy(
            run=run,
            user=user,
            owners={"acme": _owner_source(include_bots=["b"])},
            repositories={"acme/widget": _repository_source(exclude_bots=["c"])},
        )
        assert policy.listed() == [
            ListedBot("a[bot]", True, "--exclude-bots", PolicyLevel.RUN, None),
            ListedBot(
                "c[bot]", True, "acme/widget", PolicyLevel.REPOSITORY, "acme/widget"
            ),
            ListedBot("b[bot]", False, "--include-bots", PolicyLevel.RUN, None),
            ListedBot(
                "b[bot]", False, "acme/.github", PolicyLevel.OWNER, "acme/.github"
            ),
            ListedBot("b[bot]", False, "~/x", PolicyLevel.USER, None),
        ]


class TestUserFile:
    def test_path_follows_xdg_config_home(self, tmp_path: Path) -> None:
        env = {"XDG_CONFIG_HOME": str(tmp_path)}
        assert user_config_path(env) == tmp_path / "dependamerge" / "config.json"

    def test_path_defaults_to_dot_config(self) -> None:
        assert user_config_path({}) == (
            Path.home() / ".config" / "dependamerge" / "config.json"
        )

    def test_missing_file_is_no_source(self, tmp_path: Path) -> None:
        assert load_user_policy(tmp_path / "absent.json") is None

    def test_file_is_loaded_at_user_level(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(_document(include_bots=[CODE_MONKEY]), encoding="utf-8")
        source = load_user_policy(path)
        assert source is not None
        assert source.level is PolicyLevel.USER
        assert BotPolicy(user=source).admits(CODE_MONKEY, "acme/widget")

    def test_invalid_file_names_itself(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(BotPolicyError) as raised:
            load_user_policy(path)
        assert raised.value.source == str(path)

    def test_default_location_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "dependamerge").mkdir()
        user_config_path().write_text(
            _document(exclude_bots=["renovate"]), encoding="utf-8"
        )
        source = load_user_policy()
        assert source is not None
        assert source.exclude == {"renovate[bot]"}
