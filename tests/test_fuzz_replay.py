# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Replay every fuzz seed, and show each oracle rejecting a wrong answer.

The harnesses under ``tests/fuzz`` need Atheris only to search for new
inputs; their checks are plain functions, so the seeds run here in the
ordinary suite.  An oracle that cannot fail proves nothing, so each is
also handed a deliberately wrong implementation and must refuse it.
"""

from __future__ import annotations

from typing import Any

import pytest

from dependamerge.bot_policy import PolicyLevel, PolicySource
from dependamerge.github2gerrit_detector import (
    GitHub2GerritDetectionResult,
    GitHub2GerritMapping,
)
from dependamerge.gitreview import GitReviewInfo
from dependamerge.netrc import NetrcCredentials, NetrcParser
from dependamerge.url_parser import ChangeSource, ParsedUrl
from tests.fuzz import fuzz_redaction, fuzz_untrusted_content, fuzz_url_parser
from tests.fuzz._harness import ContractViolation

HARNESSES = (fuzz_url_parser, fuzz_untrusted_content, fuzz_redaction)


@pytest.mark.parametrize(
    ("harness", "seed"),
    [
        pytest.param(harness, seed, id=f"{harness.__name__.rsplit('.', 1)[-1]}-{index}")
        for harness in HARNESSES
        for index, seed in enumerate(harness.SEEDS)
    ],
)
def test_seed_holds(harness: Any, seed: bytes) -> None:
    harness.check(seed)


def _url_seed(*parts: str, mode: int = 0, scheme: int | None = None) -> bytes:
    head = bytes([mode]) + (bytes([scheme]) if scheme is not None else b"")
    return head + "\x00".join(parts).encode()


def _parsed(source: ChangeSource, number: int) -> ParsedUrl:
    return ParsedUrl(
        source=source,
        host="github.com",
        base_path=None,
        project="acme/widget",
        change_number=number,
        original_url="",
    )


@pytest.mark.parametrize(
    ("data", "wrong"),
    [
        pytest.param(
            _url_seed("u", "pw", "github.com", "acme/widget", mode=1, scheme=0),
            {"redact": lambda value: value},
            id="redact-keeps-userinfo",
        ),
        pytest.param(
            _url_seed("https://h/a;b/c"),
            {"redact": lambda value: value},
            id="redact-keeps-unmarked-semicolon",
        ),
        pytest.param(
            _url_seed("https://github.com/acme/widget/pull/7"),
            {"parse": lambda value: _parsed(ChangeSource.GERRIT, 7)},
            id="parse-disagrees-with-detect",
        ),
        pytest.param(
            _url_seed("https://gerrit.linuxfoundation.org/infra/c/releng/builder/+/1"),
            {"parse": lambda value: _parsed(ChangeSource.GERRIT, 0)},
            id="parse-returns-no-gerrit-number",
        ),
    ],
)
def test_url_oracle_rejects_wrong_result(data: bytes, wrong: dict[str, Any]) -> None:
    with pytest.raises(ContractViolation):
        fuzz_url_parser.check(data, **wrong)


def _unguarded_number(value: str) -> ParsedUrl:
    return _parsed(ChangeSource.GITHUB, int(value.rsplit("/", 1)[-1]))


def test_url_oracle_lets_undocumented_error_escape() -> None:
    data = _url_seed("https://github.com/acme/widget/pull/" + "9" * 5000)
    with pytest.raises(ValueError, match="integer string conversion"):
        fuzz_url_parser.check(data, parse=_unguarded_number)


def _no_strip(path: str) -> str:
    return path


def _strip_bare_suffix(path: str) -> str:
    return path.rstrip("/").removesuffix(".git")


@pytest.mark.parametrize(
    ("data", "helper", "wrong"),
    [
        pytest.param(
            _url_seed("https://github.com/acme/widget/pull/7"),
            "detect_source",
            lambda value: ChangeSource.GERRIT,
            id="detect-source-misreads-platform",
        ),
        pytest.param(
            _url_seed("u", "pw", "github.com", "acme/widget", mode=1, scheme=0),
            "normalize_target",
            lambda value, **_: value,
            id="normalize-keeps-userinfo",
        ),
        pytest.param(
            _url_seed("-acme"),
            "require_owner",
            lambda name, host: name,
            id="require-owner-accepts-leading-hyphen",
        ),
        pytest.param(
            _url_seed("neovim.wiki"),
            "require_repo",
            lambda name: name,
            id="require-repo-accepts-wiki",
        ),
        pytest.param(
            _url_seed("https://github.com/acme/widget.git"),
            "has_stray_git_suffix",
            lambda path: False,
            id="stray-suffix-missed",
        ),
        pytest.param(
            _url_seed("https://github.com/acme/widget.git"),
            "strip_git_suffix",
            _no_strip,
            id="strip-suffix-no-op",
        ),
        pytest.param(
            _url_seed("https://github.com/acme/.git"),
            "strip_git_suffix",
            _strip_bare_suffix,
            id="strip-suffix-takes-bare-git",
        ),
    ],
)
def test_url_oracle_rejects_wrong_helper(
    monkeypatch: pytest.MonkeyPatch, data: bytes, helper: str, wrong: Any
) -> None:
    monkeypatch.setattr(fuzz_url_parser, helper, wrong)
    with pytest.raises(ContractViolation):
        fuzz_url_parser.check(data)


class _ForgetfulNetrc(NetrcParser):
    def get_credentials(self, machine: str) -> NetrcCredentials | None:
        found = super().get_credentials(machine)
        if found is None:
            return None
        return NetrcCredentials(
            machine=found.machine, login=found.password, password=found.login
        )


def _wrong_mapping(comments: list[dict[str, Any]]) -> GitHub2GerritDetectionResult:
    mapping = GitHub2GerritMapping(
        pr_url="",
        mode="squash",
        topic="GH-widget-7",
        change_ids=("I0123456789abcdef0123456789abcdef01234567-extra",),
        raw_comment_body=comments[0]["body"].strip(),
    )
    return GitHub2GerritDetectionResult(
        has_mapping=True,
        mapping=mapping,
        comment_indices=[0],
        detection_source="marker",
    )


def _prefix_mapping(comments: list[dict[str, Any]]) -> GitHub2GerritDetectionResult:
    # The Change-ID the heuristic took from "I<40 hex>-extra".
    mapping = GitHub2GerritMapping(
        pr_url="",
        mode="squash",
        topic="GH-widget-7",
        change_ids=("I0123456789abcdef0123456789abcdef01234567",),
        raw_comment_body=comments[0]["body"].strip(),
    )
    return GitHub2GerritDetectionResult(
        has_mapping=True,
        mapping=mapping,
        comment_indices=[0],
        detection_source="heuristic",
    )


def _empty_topic(comments: list[dict[str, Any]]) -> GitHub2GerritDetectionResult:
    mapping = GitHub2GerritMapping(
        pr_url="",
        mode="squash",
        topic="",
        change_ids=("I0123456789abcdef0123456789abcdef01234567",),
        raw_comment_body=comments[0]["body"].strip(),
    )
    return GitHub2GerritDetectionResult(
        has_mapping=True,
        mapping=mapping,
        comment_indices=[0],
        detection_source="marker",
    )


def test_github2gerrit_oracle_checks_fields_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even when the model's own predicate passes everything, the oracle
    # must refuse a mapping with no topic.
    monkeypatch.setattr(GitHub2GerritMapping, "is_valid", property(lambda self: True))
    body = "I0123456789abcdef0123456789abcdef01234567"
    with pytest.raises(ContractViolation, match="invalid mapping"):
        fuzz_untrusted_content.check(
            _content_seed(0x07, body), github2gerrit=_empty_topic
        )


def _content_seed(flags: int, *parts: str) -> bytes:
    return bytes([flags]) + "\x00".join(parts).encode()


@pytest.mark.parametrize(
    ("data", "wrong"),
    [
        pytest.param(
            _content_seed(0x04, "github.com", "user", "secret"),
            {"netrc_parser": _ForgetfulNetrc},
            id="netrc-swaps-login-and-password",
        ),
        pytest.param(
            _content_seed(0x05, "gerrit.example.org", "29418", "releng/builder.git"),
            {
                "gitreview": lambda text: GitReviewInfo(
                    host="gerrit.example.org", project="releng/builder.git"
                )
            },
            id="gitreview-keeps-git-suffix",
        ),
        pytest.param(
            _content_seed(0x01, "port=29418\n"),
            {"gitreview": lambda text: GitReviewInfo(host="")},
            id="gitreview-returns-no-host",
        ),
        pytest.param(
            _content_seed(0x82, "{}"),
            {
                "policy": lambda blob, **_: PolicySource(
                    level=PolicyLevel.REPOSITORY, include_label="x", exclude_label="x"
                )
            },
            id="policy-accepts-binary-file",
        ),
        pytest.param(
            _content_seed(0x06, "dependabot"),
            {
                "policy": lambda blob, **_: PolicySource(
                    level=PolicyLevel.REPOSITORY,
                    include_label="x",
                    exclude_label="x",
                    include=frozenset({"stranger[bot]"}),
                )
            },
            id="policy-invents-an-app",
        ),
        pytest.param(
            _content_seed(0x02, '{"include_bots": ["\\u0064ependabot"]}'),
            {
                "policy": lambda blob, **_: PolicySource(
                    level=PolicyLevel.REPOSITORY,
                    include_label="x",
                    exclude_label="x",
                    include=frozenset({"u0064ependabot[bot]"}),
                )
            },
            id="policy-reads-escape-literally",
        ),
        pytest.param(
            _content_seed(0x02, '{"include_bots": ["notdependabot"]}'),
            {
                "policy": lambda blob, **_: PolicySource(
                    level=PolicyLevel.REPOSITORY,
                    include_label="x",
                    exclude_label="x",
                    include=frozenset({"dependabot[bot]"}),
                )
            },
            id="policy-reads-a-substring-as-an-app",
        ),
        pytest.param(
            _content_seed(0x02, '{"exclude_bots": ["*x"]}'),
            {
                "policy": lambda blob, **_: PolicySource(
                    level=PolicyLevel.REPOSITORY,
                    include_label="x",
                    exclude_label="x",
                    include_all=True,
                )
            },
            id="policy-wildcard-from-exclude-list",
        ),
        pytest.param(
            _content_seed(0x02, '{"include_bots": ["*x"]}'),
            {
                "policy": lambda blob, **_: PolicySource(
                    level=PolicyLevel.REPOSITORY,
                    include_label="x",
                    exclude_label="x",
                    include_all=True,
                )
            },
            id="policy-reads-a-wildcard-substring",
        ),
        pytest.param(
            _content_seed(0x07, "any comment"),
            {"github2gerrit": _wrong_mapping},
            id="github2gerrit-accepts-suffixed-change-id",
        ),
        pytest.param(
            _content_seed(
                0x0B, "Topic: GH-widget-7\nI0123456789abcdef0123456789abcdef01234567-x"
            ),
            {"github2gerrit": _prefix_mapping},
            id="github2gerrit-takes-a-change-id-prefix",
        ),
        pytest.param(
            # Markers delimit a token only for the marker parser's result.
            _content_seed(
                0x0B,
                "Topic: GH-widget-7\nI0123456789abcdef0123456789abcdef01234567"
                "<!-- end github2gerrit:change-id-map -->",
            ),
            {"github2gerrit": _prefix_mapping},
            id="github2gerrit-heuristic-takes-id-before-marker",
        ),
    ],
)
def test_content_oracle_rejects_wrong_result(
    data: bytes, wrong: dict[str, Any]
) -> None:
    with pytest.raises(ContractViolation):
        fuzz_untrusted_content.check(data, **wrong)


@pytest.mark.parametrize(
    ("data", "wrong"),
    [
        pytest.param(
            bytes([0x02]) + b"\x00\x00user\x00p@ss",
            {
                "redact": lambda text: text.replace(
                    text.split("@", 1)[0], "https://user:***"
                )
            },
            id="redact-stops-at-first-at",
        ),
        pytest.param(
            bytes([0x00]) + b"log: \x00\x00abc",
            {"redact": lambda text: text},
            id="redact-keeps-token",
        ),
        pytest.param(
            bytes([0x80]) + b"one\ntwo",
            {"summarise": lambda message: message},
            id="summary-keeps-newline",
        ),
        pytest.param(
            bytes([0x80]) + b"Client error",
            {"summarise": lambda message: message.replace("\n", " ")},
            id="summary-keeps-trailer",
        ),
        pytest.param(
            bytes([0x80]) + b"x",
            {"summarise": lambda message: message + "!"},
            id="summary-not-idempotent",
        ),
    ],
)
def test_redaction_oracle_rejects_wrong_result(
    data: bytes, wrong: dict[str, Any]
) -> None:
    with pytest.raises(ContractViolation):
        fuzz_redaction.check(data, **wrong)
