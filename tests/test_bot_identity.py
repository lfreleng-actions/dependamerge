# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Tests for the shared bot-identity helpers."""

from __future__ import annotations

import pytest

from dependamerge.bot_identity import (
    canonical_bot_login,
    is_automation_author,
    is_copilot,
    is_dependabot,
    normalize_bot_login,
    rest_author_login,
)


class TestCanonicalBotLogin:
    """``canonical_bot_login`` normalises GraphQL ``Bot`` logins to REST form."""

    def test_bot_typename_appends_suffix(self) -> None:
        assert canonical_bot_login("dependabot", "Bot") == "dependabot[bot]"

    def test_bot_typename_applies_to_any_app_actor(self) -> None:
        # Driven by __typename, not a dependabot-specific match: every bot
        # is canonicalised uniformly.
        assert canonical_bot_login("renovate", "Bot") == "renovate[bot]"
        assert canonical_bot_login("pre-commit-ci", "Bot") == "pre-commit-ci[bot]"
        assert canonical_bot_login("github-actions", "Bot") == "github-actions[bot]"

    def test_already_suffixed_is_idempotent(self) -> None:
        assert canonical_bot_login("dependabot[bot]", "Bot") == "dependabot[bot]"

    def test_user_typename_unchanged(self) -> None:
        # A human login that happens to resemble a bot is left untouched.
        assert canonical_bot_login("octocat", "User") == "octocat"

    def test_missing_typename_leaves_login_unchanged(self) -> None:
        # Without __typename (e.g. an older query) the bare login is
        # preserved; predicates below still match it.
        assert canonical_bot_login("dependabot", None) == "dependabot"

    def test_empty_login_returns_unknown(self) -> None:
        assert canonical_bot_login(None, "Bot") == "unknown"
        assert canonical_bot_login("", "Bot") == "unknown"


class TestNormalizeBotLogin:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("dependabot[bot]", "dependabot"),
            ("dependabot", "dependabot"),
            ("Dependabot[bot]", "dependabot"),
            ("RENOVATE[BOT]", "renovate"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_normalize(self, value: str | None, expected: str) -> None:
        assert normalize_bot_login(value) == expected


class TestIsDependabot:
    def test_matches_both_forms(self) -> None:
        assert is_dependabot("dependabot[bot]")
        assert is_dependabot("dependabot")
        assert is_dependabot("Dependabot[bot]")

    def test_rejects_other_actors(self) -> None:
        assert not is_dependabot("renovate[bot]")
        assert not is_dependabot("octocat")
        assert not is_dependabot(None)
        assert not is_dependabot("")


class TestIsCopilot:
    @pytest.mark.parametrize(
        "login",
        [
            "Copilot",
            "copilot",
            "copilot[bot]",
            "github-copilot",
            "github-copilot[bot]",
            "copilot-swe-agent",
            "copilot-swe-agent[bot]",
            "copilot-pull-request-reviewer",
        ],
    )
    def test_matches_every_copilot_identity(self, login: str) -> None:
        assert is_copilot(login)

    def test_rejects_other_actors(self) -> None:
        assert not is_copilot("dependabot[bot]")
        assert not is_copilot("octocat")
        assert not is_copilot(None)
        assert not is_copilot("")


class TestRestAuthorLogin:
    """``rest_author_login`` decides by account type, not by name."""

    def test_app_keeps_its_suffixed_login(self) -> None:
        user = {"login": "dependabot[bot]", "type": "Bot", "id": 49699333}
        assert rest_author_login(user) == "dependabot[bot]"

    @pytest.mark.parametrize(
        ("account_id", "expected"),
        [
            (198982749, "copilot-swe-agent[bot]"),
            (175728472, "copilot-pull-request-reviewer[bot]"),
        ],
    )
    def test_copilot_display_login_resolves_by_account_id(
        self, account_id: int, expected: str
    ) -> None:
        # REST names two different Copilot Apps "Copilot"; GraphQL
        # reports their real logins, which these must match.
        user = {"login": "Copilot", "type": "Bot", "id": account_id}
        assert rest_author_login(user) == expected
        assert rest_author_login(user) == canonical_bot_login(
            expected.removesuffix("[bot]"), "Bot"
        )

    def test_unknown_app_without_suffix_gains_one(self) -> None:
        user = {"login": "Display", "type": "Bot", "id": 1}
        assert rest_author_login(user) == "Display[bot]"

    @pytest.mark.parametrize(
        "login", ["renovate", "github-actions", "copilot-swe-agent", "talbot"]
    )
    def test_user_account_is_never_suffixed(self, login: str) -> None:
        user = {"login": login, "type": "User", "id": 203248971}
        assert rest_author_login(user) == login
        assert not is_automation_author(rest_author_login(user))

    def test_missing_user(self) -> None:
        assert rest_author_login(None) == ""
        assert rest_author_login({}) == ""


class TestIsAutomationAuthor:
    @pytest.mark.parametrize(
        "author",
        [
            "dependabot[bot]",
            "renovate[bot]",
            "pre-commit-ci[bot]",
            "github-actions[bot]",
            "copilot-swe-agent[bot]",
            "lf-releng-code-monkey-bot[bot]",
            "Dependabot[BOT]",
        ],
    )
    def test_any_app_is_automation(self, author: str) -> None:
        assert is_automation_author(author)

    @pytest.mark.parametrize(
        "author",
        [
            "john-doe",
            "some-user",
            None,
            "",
            # Bare logins are people, whatever they are called: these
            # names can belong to User accounts (copilot-swe-agent does).
            "dependabot",
            "renovate",
            "github-actions",
            "copilot-swe-agent",
            # So is a person whose login merely contains "bot".
            "talbot",
            "random-bot",
        ],
    )
    def test_rejects_people(self, author: str | None) -> None:
        assert not is_automation_author(author)

    def test_graphql_app_classifies_once_canonical(self) -> None:
        # The data boundary turns a GraphQL Bot node into the REST form,
        # which is what makes the bare name classify as an App.
        assert is_automation_author(canonical_bot_login("dependabot", "Bot"))
        assert not is_automation_author(canonical_bot_login("dependabot", "User"))
