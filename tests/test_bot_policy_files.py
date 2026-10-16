# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for turning repository and owner policy files into sources."""

from __future__ import annotations

import json
from typing import Any

import pytest

from dependamerge.bot_policy import (
    BotPolicy,
    BotPolicyError,
    PolicyLevel,
    parse_policy_document,
    run_policy_source,
)
from dependamerge.bot_policy_files import (
    PolicyFiles,
    apply_policy_files,
    parse_policy_blob,
)

CODE_MONKEY = "lf-releng-code-monkey-bot[bot]"


def _blob(text: str) -> dict[str, Any]:
    """A file as GraphQL returns it."""
    return {
        "text": text,
        "byteSize": len(text.encode("utf-8")),
        "isTruncated": False,
        "isBinary": False,
    }


def _policy(**lists: object) -> dict[str, Any]:
    return _blob(json.dumps(lists))


def _parse(blob: dict[str, Any] | None):
    return parse_policy_blob(blob, label="acme/widget", level=PolicyLevel.REPOSITORY)


class TestParsePolicyBlob:
    def test_no_file_is_no_source(self) -> None:
        assert _parse(None) is None

    def test_a_file_is_parsed_at_its_level(self) -> None:
        source = _parse(_policy(include_bots=["lf-releng-code-monkey-bot"]))
        assert source is not None
        assert source.level is PolicyLevel.REPOSITORY
        assert source.include == {CODE_MONKEY}
        assert source.include_label == "acme/widget"

    @pytest.mark.parametrize(
        ("blob", "problem"),
        [
            # GraphQL selects no Blob fields from a directory.
            ({}, "is not a file"),
            ({**_policy(), "isBinary": True, "text": None}, "is not a text file"),
            ({**_policy(), "isTruncated": True}, "larger than 16 KiB"),
            ({**_policy(), "byteSize": 16 * 1024 + 1}, "larger than 16 KiB"),
            ({**_policy(), "text": None}, "has no text"),
            (_blob("{not json"), "not valid JSON"),
            (_policy(exlude_bots=["renovate"]), "unknown key"),
        ],
    )
    def test_anything_else_is_refused(self, blob: dict[str, Any], problem: str) -> None:
        with pytest.raises(BotPolicyError, match=problem) as raised:
            _parse(blob)
        assert raised.value.source == "acme/widget"


class TestApplyPolicyFiles:
    def test_files_become_sources_named_where_they_live(self) -> None:
        files = PolicyFiles(
            owners={"acme": _policy(exclude_bots=["renovate"])},
            repositories={"acme/widget": _policy(include_bots=[CODE_MONKEY])},
        )
        policy, errors = apply_policy_files(BotPolicy(), files)
        assert errors == []
        assert policy.decide(CODE_MONKEY, "acme/widget").reason == (
            "included by acme/widget"
        )
        assert not policy.admits(CODE_MONKEY, "acme/other")
        assert policy.decide("renovate[bot]", "acme/widget").reason == (
            "excluded by acme/.github"
        )

    def test_absent_files_add_nothing(self) -> None:
        files = PolicyFiles(owners={"acme": None}, repositories={"acme/widget": None})
        assert apply_policy_files(BotPolicy(), files) == (BotPolicy(), [])

    def test_files_take_their_place_in_the_precedence(self) -> None:
        user = parse_policy_document(
            json.dumps({"include_bots": [CODE_MONKEY]}),
            label="~/x",
            level=PolicyLevel.USER,
        )
        files = PolicyFiles(
            owners={"acme": _policy(exclude_bots=[CODE_MONKEY])},
            repositories={"acme/widget": _policy(include_bots=[CODE_MONKEY])},
        )
        policy, _ = apply_policy_files(BotPolicy(user=user), files)
        assert policy.decide(CODE_MONKEY, "acme/widget").reason == (
            "included by acme/widget"
        )
        assert policy.decide(CODE_MONKEY, "acme/other").reason == (
            "excluded by acme/.github"
        )
        assert policy.decide(CODE_MONKEY, "elsewhere/x").reason == "included by ~/x"

        # Only the run's own flags outrank the files, and they say so.
        run = run_policy_source(None, [CODE_MONKEY])
        overruled, _ = apply_policy_files(BotPolicy(run=run, user=user), files)
        decision = overruled.decide(CODE_MONKEY, "acme/widget")
        assert (decision.in_scope, decision.overrides) == (False, "acme/widget")
        assert overruled.decide(CODE_MONKEY, "elsewhere/x").overrides is None

    def test_an_invalid_repository_file_takes_that_repository_out_of_scope(
        self,
    ) -> None:
        files = PolicyFiles(
            repositories={"Acme/Widget": _blob("{not json"), "acme/other": None}
        )
        policy, errors = apply_policy_files(BotPolicy(), files)
        assert errors == []
        assert "not valid JSON" in (policy.repository_error("acme/widget") or "")
        assert not policy.admits("dependabot[bot]", "acme/widget")
        assert policy.admits("dependabot[bot]", "acme/other")

    def test_an_invalid_owner_file_is_returned_not_raised(self) -> None:
        policy, errors = apply_policy_files(
            BotPolicy(), PolicyFiles(owners={"acme": {}})
        )
        assert [(error.source, error.problems) for error in errors] == [
            ("acme/.github", ["is not a file"])
        ]
        assert policy.owners == {}

    def test_a_file_read_later_replaces_what_was_known(self) -> None:
        first, _ = apply_policy_files(
            BotPolicy(), PolicyFiles(repositories={"acme/widget": {}})
        )
        assert first.repository_error("acme/widget") == "is not a file"
        files = PolicyFiles(repositories={"acme/widget": _policy(include_bots="*")})
        second, _ = apply_policy_files(first, files)
        assert second.repository_error("acme/widget") is None
        assert second.admits(CODE_MONKEY, "acme/widget")
