# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the version bump classifier."""

import time

import pytest

from dependamerge.version_bumps import (
    MAJOR,
    OTHER,
    UNKNOWN,
    ZERO_MINOR,
    Update,
    classify,
    counts_as_major,
    pair_level,
)

_LINKED = (
    "Bumps [python-test-action](https://github.com/lfreleng-actions/"
    "python-test-action) from 1.3.3 to 2.0.0.\n"
    "- [Release notes](https://github.com/lfreleng-actions/python-test-action/releases)\n"
)

_RELEASE_NOTES = (
    "Bumps [syrupy](https://github.com/syrupy-project/syrupy) from 4.9.1 to 4.9.2.\n"
    "<details>\n<summary>Release notes</summary>\n"
    "<blockquote>\n"
    "Bumps [pytest](https://github.com/pytest-dev/pytest) from 7.4.0 to 8.0.0.\n"
    "Updates `pytest` from 7.4.0 to 8.0.0\n"
    "</blockquote>\n</details>\n"
    "<details>\n<summary>Commits</summary>\n<ul>\n"
    "<li>Bump syrupy from 3.0.0 to 4.0.0</li>\n</ul>\n</details>\n"
)

_GROUPED = (
    "Bumps the pip group with 3 updates: [a](https://x/a), [b](https://x/b) "
    "and [c](https://x/c).\n\n"
    "Updates `a` from 1.2.0 to 1.3.0\n"
    "<details><summary>Commits</summary>Updates `z` from 1 to 9</details>\n"
    "Updates `b` from 2.9.1 to 3.0.0\n"
    "Updates `c` from 0.4.0 to 0.4.1\n"
)

_METADATA = (
    "Bumps [syrupy](https://github.com/syrupy-project/syrupy) from 4.9.1 to 4.10.0.\n"
    "---\n"
    "updated-dependencies:\n"
    "- dependency-name: syrupy\n"
    "  dependency-version: 4.10.0\n"
    "  dependency-type: direct:development\n"
    "  update-type: version-update:semver-major\n"
    "...\n"
)


@pytest.mark.parametrize(
    ("title", "text", "level", "updates"),
    [
        pytest.param(
            "Chore: Bump python-test-action from 1.3.3 to 2.0.0",
            _LINKED,
            MAJOR,
            [("python-test-action", "1.3.3", "2.0.0", MAJOR)],
            id="linked-name",
        ),
        pytest.param(
            "Chore: Bump org.apache.commons:commons-lang3",
            "Bumps org.apache.commons:commons-lang3 from 3.17.0 to 3.18.0.\n",
            OTHER,
            [("org.apache.commons:commons-lang3", "3.17.0", "3.18.0", OTHER)],
            id="bare-name",
        ),
        pytest.param(
            "Chore: Bump the pip group with 3 updates",
            _GROUPED,
            MAJOR,
            [
                ("a", "1.2.0", "1.3.0", OTHER),
                ("b", "2.9.1", "3.0.0", MAJOR),
                ("c", "0.4.0", "0.4.1", OTHER),
            ],
            id="grouped-one-major",
        ),
        pytest.param(
            "Chore: Bump syrupy from 4.9.1 to 4.9.2",
            _RELEASE_NOTES,
            OTHER,
            [("syrupy", "4.9.1", "4.9.2", OTHER)],
            id="details-never-count",
        ),
        pytest.param(
            "Chore: Bump things",
            "This also Bumps [x](https://x) from 1.0.0 to 2.0.0.\n"
            " Updates `y` from 1.0 to 2.0\n",
            UNKNOWN,
            [],
            id="not-at-line-start",
        ),
        pytest.param(
            "Chore: Bump fedora from `43b29f6` to `15da97e`",
            "Bumps fedora from `43b29f6` to `15da97e`.\n",
            UNKNOWN,
            [("fedora", "43b29f6", "15da97e", UNKNOWN)],
            id="digest",
        ),
        pytest.param(
            "Chore: Bump actions/checkout from v4.2.2 to v5.0.0",
            "Bumps [actions/checkout](https://x) from v4.2.2 to v5.0.0.\n",
            MAJOR,
            [("actions/checkout", "v4.2.2", "v5.0.0", MAJOR)],
            id="v-prefix",
        ),
        pytest.param(
            "Chore: Bump x from 0.0.2 to 1.0.0",
            "Bumps x from 0.0.2 to 1.0.0.\n",
            MAJOR,
            [("x", "0.0.2", "1.0.0", MAJOR)],
            id="zero-to-one",
        ),
        pytest.param(
            "Chore: Bump x from 0.7.0 to 0.8.0",
            "Bumps x from 0.7.0 to 0.8.0.\n",
            ZERO_MINOR,
            [("x", "0.7.0", "0.8.0", ZERO_MINOR)],
            id="zero-minor",
        ),
        pytest.param(
            "Chore: Bump x from 0.7.1 to 0.7.2",
            "Bumps x from 0.7.1 to 0.7.2.\n",
            OTHER,
            [("x", "0.7.1", "0.7.2", OTHER)],
            id="zero-patch",
        ),
        pytest.param(
            "Chore: Bump x from 3.1.0 to 2.9.0",
            "Bumps x from 3.1.0 to 2.9.0.\n",
            MAJOR,
            [("x", "3.1.0", "2.9.0", MAJOR)],
            id="downgrade",
        ),
        pytest.param(
            "Chore: pre-commit autoupdate",
            "<!--pre-commit.ci start-->\nupdates:\n"
            "- [github.com/astral-sh/ruff-pre-commit: v0.14.0 \u2192 v0.15.0]"
            "(https://github.com/astral-sh/ruff-pre-commit/compare/v0.14.0...v0.15.0)\n"
            "- [github.com/psf/black: 24.10.0 \u2192 25.1.0]"
            "(https://github.com/psf/black/compare/24.10.0...25.1.0)\n",
            MAJOR,
            [
                (
                    "github.com/astral-sh/ruff-pre-commit",
                    "v0.14.0",
                    "v0.15.0",
                    ZERO_MINOR,
                ),
                ("github.com/psf/black", "24.10.0", "25.1.0", MAJOR),
            ],
            id="pre-commit-tags",
        ),
        pytest.param(
            "Chore: pre-commit autoupdate",
            "updates:\n"
            "- [github.com/psf/black: 8a737e727ac5ab2f1d4cf5876720ed276dc8dc4b"
            " \u2192 25c5ecb5ee9b9e0d1d2e6b1f8c1c7b3f8e1f5a2d]"
            "(https://github.com/psf/black/compare/8a737e7...25c5ecb)\n",
            UNKNOWN,
            [
                (
                    "github.com/psf/black",
                    "8a737e727ac5ab2f1d4cf5876720ed276dc8dc4b",
                    "25c5ecb5ee9b9e0d1d2e6b1f8c1c7b3f8e1f5a2d",
                    UNKNOWN,
                )
            ],
            id="pre-commit-commits",
        ),
        pytest.param(
            "Chore: Bump syrupy from 4.9.1 to 4.10.0",
            _METADATA,
            MAJOR,
            [("syrupy", "4.9.1", "4.10.0", MAJOR)],
            id="semver-major-metadata",
        ),
        pytest.param(
            "Chore: Bump eclipse-temurin from 21.0.8_9-jre to 25.0.1_8-jre",
            "Bumps eclipse-temurin from 21.0.8_9-jre to 25.0.1_8-jre.\n",
            MAJOR,
            [("eclipse-temurin", "21.0.8_9-jre", "25.0.1_8-jre", MAJOR)],
            id="underscore-tag",
        ),
        pytest.param(
            "Chore: Bump foo from 1!1.3.3 to 2!2.0.0",
            "Bumps foo from 1!1.3.3 to 2!2.0.0.\n",
            MAJOR,
            [("foo", "1!1.3.3", "2!2.0.0", MAJOR)],
            id="epoch",
        ),
        pytest.param(
            "Chore: Bump foo from 1!1.3.3 to 1!2.0.0",
            "No anchored line.\n",
            MAJOR,
            [("foo", "1!1.3.3", "1!2.0.0", MAJOR)],
            id="epoch-title-only",
        ),
        pytest.param(
            "Chore(deps): Update dependency foo from 1.4.0 to 2.1.0",
            "This PR contains the following updates: a table.\n",
            MAJOR,
            [("foo", "1.4.0", "2.1.0", MAJOR)],
            id="title-only",
        ),
        pytest.param(
            "Chore: Update the README",
            "Nothing about versions here.\n",
            UNKNOWN,
            [],
            id="nothing-readable",
        ),
        pytest.param(None, None, UNKNOWN, [], id="none"),
        pytest.param("", "", UNKNOWN, [], id="empty"),
        pytest.param(
            "Chore: Bump things",
            "Some text.\r\nBumps [x](https://x) from 1.0.0 to 2.0.0.\r\n",
            MAJOR,
            [("x", "1.0.0", "2.0.0", MAJOR)],
            id="crlf",
        ),
    ],
)
def test_classify(title, text, level, updates):
    result = classify(title, text)
    assert result.level == level
    assert result.updates == tuple(Update(*update) for update in updates)


def test_body_beats_title():
    result = classify("Bump x from 1.0.0 to 9.0.0", "Bumps x from 1.0.0 to 1.1.0.\n")
    assert result.level == OTHER


def test_metadata_never_lowers_a_level():
    text = "Bumps x from 1.0.0 to 2.0.0.\nupdate-type: version-update:semver-patch\n"
    assert classify(None, text).level == MAJOR


def test_quoted_metadata_name_marks_its_update():
    text = (
        "Updates `@scope/pkg` from 1.2.0 to 1.3.0\n"
        "Updates `other` from 2.0.0 to 2.1.0\n"
        "---\n"
        "updated-dependencies:\n"
        '- dependency-name: "@scope/pkg"\n'
        "  update-type: version-update:semver-major\n"
        "- dependency-name: other\n"
        "  update-type: version-update:semver-minor\n"
    )
    result = classify(None, text)
    assert result.level == MAJOR
    assert [u.name for u in result.major_updates()] == ["@scope/pkg"]


def test_metadata_marks_a_title_update():
    text = (
        "---\n"
        "updated-dependencies:\n"
        "- dependency-name: foo\n"
        "  update-type: version-update:semver-major\n"
    )
    result = classify("Chore: Bump foo from 1.2.0 to 1.3.0", text)
    assert result.level == MAJOR
    assert result.updates == (Update("foo", "1.2.0", "1.3.0", MAJOR),)


def test_unclosed_details_hides_the_rest():
    text = "Bumps x from 1.0.0 to 1.0.1.\n<details>\nBumps y from 1.0.0 to 2.0.0.\n"
    assert classify(None, text).level == OTHER


def test_nested_details_hide_only_their_contents():
    text = (
        "<details><details>\nBumps y from 1.0.0 to 2.0.0.\n</details>\n"
        "Bumps z from 1.0.0 to 3.0.0.\n</details>\n"
        "</details>\nBumps x from 1.0.0 to 1.0.1.\n"
        "<details>\nBumps w from 1.0.0 to 2.0.0.\n"
    )
    assert [u.name for u in classify(None, text).updates] == ["x"]


def test_as_dict_shape():
    assert classify(None, "Bumps x from 1.0.0 to 2.0.0.\n").as_dict() == {
        "level": MAJOR,
        "updates": [{"name": "x", "from": "1.0.0", "to": "2.0.0", "level": MAJOR}],
    }


_HUGE = "9" * 5000


@pytest.mark.parametrize(
    ("old", "new", "level"),
    [
        ("1", "2", MAJOR),
        ("1234567", "1.0.0", UNKNOWN),
        ("latest", "1.0.0", UNKNOWN),
        ("1.0.0-rc.1", "1.0.0", OTHER),
        ("01.2.0", "1.3.0", OTHER),
        # Past Python's 4,300-digit int() limit, in either number.
        (f"{_HUGE}.0", f"{_HUGE}.1", OTHER),
        ("1.0", f"{_HUGE}.0", MAJOR),
        (f"0.{_HUGE}", "0.1", ZERO_MINOR),
        (f"0.{_HUGE}", f"0.{_HUGE}.1", OTHER),
        # PEP 440 epochs: a missing one is 0, and a change is major.
        ("1!1.3.3", "1!2.0.0", MAJOR),
        ("1!1.3.3", "1!1.4.0", OTHER),
        ("1!0.7.0", "1!0.8.0", ZERO_MINOR),
        ("1!1.3.3", "2!1.3.3", MAJOR),
        ("1.3.3", "1!1.3.3", MAJOR),
        ("0!1.3.3", "1.4.0", OTHER),
        ("v1!1.0", "v1!1.1", OTHER),
        ("1.0!", "2.0.0", UNKNOWN),
        ("1!", "2.0.0", UNKNOWN),
        ("!1.0", "2.0.0", UNKNOWN),
        ("1!!1.0", "2.0.0", UNKNOWN),
    ],
)
def test_pair_level(old, new, level):
    assert pair_level(old, new) == level


@pytest.mark.parametrize(
    ("level", "status", "counted"),
    [
        (MAJOR, "merged", True),
        (MAJOR, "auto_merge_pending", True),
        (MAJOR, "failed", False),
        (MAJOR, "skipped", False),
        (ZERO_MINOR, "merged", False),
    ],
)
def test_counts_as_major(level, status, counted):
    assert counts_as_major(level, status) is counted


_LONG = "9" * 50_000


@pytest.mark.parametrize(
    ("title", "text"),
    [
        (None, f"Bumps x from {_LONG}.0! to 1.0.0\n"),
        (f"Bump x from {_LONG}.0! to 1.0.0", None),
        (None, f"Bumps x from {_LONG}!{_LONG}! to 1.0.0\n"),
        ("a" * 50_000, None),
        (None, "<details>\n" * 20_000 + "Bumps y from 1.0.0 to 2.0.0.\n"),
        (
            None,
            "<details>" * 20_000
            + "\nBumps y from 1.0.0 to 2.0.0.\n"
            + "</details>" * 20_000,
        ),
    ],
)
def test_long_malformed_input_is_read_in_linear_time(title, text):
    # Quadratic backtracking took seconds here; a linear scan takes
    # milliseconds, so the bound leaves room for a slow runner.
    start = time.monotonic()
    assert classify(title, text).level == UNKNOWN
    assert time.monotonic() - start < 1
