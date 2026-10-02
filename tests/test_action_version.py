# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for scripts/action_version.py, the action's version resolver."""

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "action_version.py"

COMMIT = "609a3915070f3adf13317718c3a4869961eaa368"
TAG_OBJECT = "7805d6c65d354451f292f81aed1b4278c7020922"
OTHER = "11646b48844dd558f745b46e8020c4c6f72c6d8a"

# The shape of ``git ls-remote --tags`` for this repository: annotated
# tags list the tag object, then the peeled commit with ``^{}``.
LISTING = "\n".join(
    [
        "c8e99be1f2715ceb000c0a08f7814a9f2fe61dac\trefs/tags/v0.13.1",
        f"{OTHER}\trefs/tags/v0.13.1^{{}}",
        f"{TAG_OBJECT}\trefs/tags/v0.14.0",
        f"{COMMIT}\trefs/tags/v0.14.0^{{}}",
        f"{COMMIT}\trefs/tags/not-a-release",
    ]
)


@pytest.fixture(scope="module")
def action_version() -> ModuleType:
    spec = importlib.util.spec_from_file_location("action_version", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _never(url: str) -> str:
    raise AssertionError(f"unexpected ls-remote of {url}")


class TestResolve:
    def test_a_release_tag_ref_is_its_own_version(self, action_version):
        assert action_version.resolve("v0.14.0", "u", ls_remote=_never) == "0.14.0"

    def test_a_commit_ref_matches_the_annotated_tag_on_it(self, action_version):
        version = action_version.resolve(COMMIT, "u", ls_remote=lambda url: LISTING)
        assert version == "0.14.0"

    def test_a_separate_sha_wins_over_the_ref(self, action_version):
        version = action_version.resolve(
            "main", "u", sha=OTHER, ls_remote=lambda url: LISTING
        )
        assert version == "0.13.1"

    def test_a_separate_sha_wins_over_a_tag_shaped_ref(self, action_version):
        version = action_version.resolve(
            "v0.14.0", "u", sha=OTHER, ls_remote=lambda url: LISTING
        )
        assert version == "0.13.1"

    def test_a_lightweight_tag_matches_too(self, action_version):
        listing = f"{COMMIT}\trefs/tags/v1.2.3"
        assert action_version.resolve(COMMIT, "u", ls_remote=lambda url: listing) == (
            "1.2.3"
        )

    def test_the_highest_release_wins_when_several_share_a_commit(self, action_version):
        listing = f"{COMMIT}\trefs/tags/v1.2.10\n{COMMIT}\trefs/tags/v1.2.9"
        assert action_version.resolve(COMMIT, "u", ls_remote=lambda url: listing) == (
            "1.2.10"
        )

    def test_an_untagged_commit_names_itself(self, action_version):
        version = action_version.resolve("a" * 40, "u", ls_remote=lambda url: LISTING)
        assert version == "0.0.0+gaaaaaaa"

    def test_a_branch_ref_without_a_sha_falls_back(self, action_version):
        warnings: list[str] = []
        version = action_version.resolve(
            "main", "u", ls_remote=_never, warn=warnings.append
        )
        assert version == "0.0.0+unknown"
        assert warnings

    def test_a_listing_failure_falls_back_with_a_warning(self, action_version):
        def failing(url: str) -> str:
            raise subprocess.CalledProcessError(128, ["git", "ls-remote"])

        warnings: list[str] = []
        version = action_version.resolve(
            COMMIT, "u", ls_remote=failing, warn=warnings.append
        )
        assert version == f"0.0.0+g{COMMIT[:7]}"
        assert any("could not list tags" in warning for warning in warnings)


class TestMain:
    def test_prints_the_version(self, action_version, capsys):
        assert action_version.main(["--ref", "v2.0.0", "--repository-url", "u"]) == 0
        assert capsys.readouterr().out.strip() == "2.0.0"

    def test_warnings_use_workflow_command_syntax(self, action_version, capsys):
        assert action_version.main(["--ref", "main", "--repository-url", "u"]) == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == "0.0.0+unknown"
        assert captured.err.startswith("::warning title=dependamerge version::")
