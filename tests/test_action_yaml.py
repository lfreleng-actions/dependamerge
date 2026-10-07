# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Credential and toolchain rules for the composite action's steps.

The merge token is handed to the dependamerge process alone. Every
other step, and any tool a step runs, must see no GitHub token: not the
input, and not a GITHUB_TOKEN or GH_TOKEN the caller's job exports.
Composite steps inherit the job's environment, so each step masks both.

Which expressions a step may evaluate is an allowlist, not a search for
token references: a blocklist misses whatever it did not foresee, such
as ``toJSON(inputs)``, which serialises the token with every other
input.
"""

import json
import os
import re
import subprocess
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

_ACTION = Path(__file__).resolve().parent.parent / "action.yaml"
_TOKEN_VARIABLES = ("GITHUB_TOKEN", "GH_TOKEN")
# Read by bash before a step's first line runs: BASH_ENV names a file it
# sources, and SHELLOPTS can switch on xtrace, which would log the token.
_SHELL_STARTUP_VARIABLES = ("BASH_ENV", "SHELLOPTS")
# The github context fields a step may read: none carries a credential.
_GITHUB_FIELDS = frozenset(
    {
        "action_path",
        "action_ref",
        "action_repository",
        "repository",
        "run_id",
        "server_url",
    }
)
# A whole expression that is one plain property path; anything else (a
# function call, an index, a bare context, another context) is refused.
_PROPERTY_PATH = re.compile(
    r"\A(?:inputs\.(?P<input>[a-z][a-z0-9_]*)"
    r"|steps\.(?P<step>[a-z][a-z0-9_-]*)\.outputs\.(?P<output>[a-z][a-z0-9_]*)"
    r"|github\.(?P<github>[a-z][a-z0-9_]*))\Z"
)


@pytest.fixture(scope="module")
def action() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(_ACTION.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def steps(action) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = action["runs"]["steps"]
    return loaded


def _label(step: dict[str, Any]) -> str:
    return str(step.get("id") or step.get("name"))


def test_every_step_masks_inherited_tokens(steps) -> None:
    unmasked = [
        f"{_label(step)}: {name}"
        for step in steps
        for name in _TOKEN_VARIABLES + _SHELL_STARTUP_VARIABLES
        if (step.get("env") or {}).get(name) != ""
    ]
    assert unmasked == []


def _strings(node: Any) -> list[str]:
    """Every string in a parsed step, keys and values alike."""
    if isinstance(node, dict):
        return [s for k, v in node.items() for s in _strings(k) + _strings(v)]
    if isinstance(node, list):
        return [s for item in node for s in _strings(item)]
    return [node] if isinstance(node, str) else []


def _expressions(node: Any) -> list[str]:
    """Every ``${{ }}`` expression in a parsed node: env, with and run."""
    return [
        expression.strip()
        for text in _strings(node)
        for expression in re.findall(r"\$\{\{(.*?)\}\}", text, re.DOTALL)
    ]


def _refusal(
    expression: str, *, step: str, inputs: Collection[str], step_ids: Collection[str]
) -> str | None:
    """Why ``expression`` may not appear in ``step``, or None if it may."""
    path = _PROPERTY_PATH.match(expression)
    if path is None:
        return "not a plain inputs, steps or github property"
    if path["input"] is not None:
        if path["input"] not in inputs:
            return "names no declared input"
        if path["input"] == "token" and step != "run":
            return "the token input belongs to the run step alone"
    elif path["step"] is not None:
        if path["step"] not in step_ids:
            return "names no step"
    elif path["github"] not in _GITHUB_FIELDS:
        return "is not an allowed github field"
    return None


def _refusals(action: dict[str, Any], steps: list[dict[str, Any]]) -> list[str]:
    inputs = set(action["inputs"])
    step_ids = {step["id"] for step in steps if "id" in step}
    found = []
    for step in steps:
        for expression in _expressions(step):
            reason = _refusal(
                expression, step=_label(step), inputs=inputs, step_ids=step_ids
            )
            if reason:
                found.append(f"{_label(step)}: {expression}: {reason}")
    return found


def test_every_step_expression_is_allowlisted(action, steps) -> None:
    assert _refusals(action, steps) == []


def test_the_run_step_alone_holds_the_token(steps) -> None:
    holders = [_label(step) for step in steps if "inputs.token" in _expressions(step)]
    assert holders == ["run"]


def test_action_outputs_read_step_outputs_alone(action, steps) -> None:
    # An output is handed to the caller, so one built from an input or
    # a context could pass the token straight out of the action.
    step_ids = {step["id"] for step in steps if "id" in step}
    for name, output in action["outputs"].items():
        (expression,) = _expressions(output["value"])
        path = _PROPERTY_PATH.match(expression)
        assert path is not None and path["step"] in step_ids, name


_DECLARED = ("target", "token")
_STEP_IDS = ("options",)


@pytest.mark.parametrize(
    "expression",
    [
        # Whole contexts, serialised or bare: each carries the token.
        "toJSON(inputs)",
        "toJSON(github)",
        "inputs",
        "github",
        "join(inputs.*, ',')",
        # The token itself, by any route.
        "github.token",
        "github['token']",
        "inputs['token']",
        "format('{0}', github.token)",
        "INPUTS.TOKEN",
        # Contexts no step needs.
        "env.GITHUB_TOKEN",
        "secrets.ANYTHING",
        "github.event.pull_request.title",
        # Paths that name nothing the action declares.
        "inputs.undeclared",
        "steps.nope.outputs.x",
        # The token input outside the run step.
        "inputs.token",
    ],
)
def test_an_expression_outside_the_allowlist_is_refused(expression) -> None:
    reason = _refusal(expression, step="other", inputs=_DECLARED, step_ids=_STEP_IDS)
    assert reason is not None


@pytest.mark.parametrize(
    ("expression", "step"),
    [
        ("inputs.target", "other"),
        ("inputs.token", "run"),
        ("steps.options.outputs.target", "other"),
        ("github.repository", "other"),
    ],
)
def test_an_allowlisted_expression_is_accepted(expression, step) -> None:
    assert _refusal(expression, step=step, inputs=_DECLARED, step_ids=_STEP_IDS) is None


def test_the_run_step_never_exports_the_token(steps) -> None:
    (run,) = [s for s in steps if s.get("id") == "run"]
    script: str = run["run"]
    # A 'token' the caller exported keeps its export attribute when
    # assigned, and allexport exports even a fresh one; both would hand
    # the merge token to uv. The attribute must go before any tool runs.
    first_tool = script.index("uv sync")
    for statement in ("unset -v token", 'token="$MERGE_TOKEN"', "declare +x token"):
        assert -1 < script.find(statement) < first_tool, statement
    assert script.index("unset -v token") < script.index('token="$MERGE_TOKEN"')
    assert script.index('token="$MERGE_TOKEN"') < script.index("declare +x token")


def test_setup_uv_installs_an_exact_uv_without_a_token(steps) -> None:
    (setup,) = [s for s in steps if str(s.get("uses", "")).startswith("astral-sh/")]
    options = setup["with"]
    # Unset, setup-uv reads a version from the caller's workspace or
    # installs the latest release.
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(options["version"]))
    assert options["working-directory"] == "${{ github.action_path }}"
    assert options["github-token"] == ""


def _run_step_environment_seen_by_uv(
    steps: list[dict[str, Any]], tmp_path: Path, inherited: dict[str, str]
) -> list[dict[str, str]]:
    """Run the run step as the runner would; return each uv call's env.

    The step's own ``env`` is laid over the inherited job environment,
    as the runner does, and the script runs under the runner's bash
    options. uv and the installed dependamerge are recorders, so the
    test needs neither a network nor a build.
    """
    (run,) = [s for s in steps if s.get("id") == "run"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    seen = tmp_path / "uv-calls.jsonl"
    recorder = (
        f"#!{sys.executable}\n"
        "import json, os\n"
        f"with open({str(seen)!r}, 'a') as f:\n"
        "    f.write(json.dumps(dict(os.environ)) + '\\n')\n"
    )
    (bin_dir / "uv").write_text(recorder)
    (bin_dir / "uv").chmod(0o755)
    venv_bin = tmp_path / "dependamerge-venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "dependamerge").write_text("#!/bin/sh\nexit 0\n")
    (venv_bin / "dependamerge").chmod(0o755)

    def evaluate(value: Any) -> str:
        # Every expression the step uses names an input or a step output.
        return re.sub(r"\$\{\{.*?\}\}", "x", str(value))

    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "HOME": str(tmp_path),
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_ACTION_PATH": str(_ACTION.parent),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_SERVER_URL": "https://github.com",
        **inherited,
        **{k: evaluate(v) for k, v in (run.get("env") or {}).items()},
    }
    script = tmp_path / "run.sh"
    script.write_text(run["run"])
    subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
        env=env,
        check=True,
        capture_output=True,
    )
    return [json.loads(line) for line in seen.read_text().splitlines()]


_VERSION_PREFIXES = ("SETUPTOOLS_SCM_", "VCS_VERSIONING_")


def _version_settings(call: dict[str, str]) -> dict[str, str]:
    return {
        key: value for key, value in call.items() if key.startswith(_VERSION_PREFIXES)
    }


def test_the_build_is_given_the_resolved_version(steps, tmp_path) -> None:
    calls = _run_step_environment_seen_by_uv(steps, tmp_path, {})
    assert len(calls) == 2  # the dependency sync, then the project build
    for call in calls:
        assert _version_settings(call) == {"SETUPTOOLS_SCM_PRETEND_VERSION": "x"}


@pytest.mark.parametrize(
    "name",
    [
        # Each overrides the release the action resolved, even with
        # SETUPTOOLS_SCM_PRETEND_VERSION set: a metadata 'tag' replaces it.
        "SETUPTOOLS_SCM_PRETEND_METADATA",
        "SETUPTOOLS_SCM_PRETEND_METADATA_FOR_DEPENDAMERGE",
        "VCS_VERSIONING_PRETEND_METADATA",
        "VCS_VERSIONING_PRETEND_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DEPENDAMERGE",
        "SETUPTOOLS_SCM_OVERRIDES_FOR_DEPENDAMERGE",
    ],
)
def test_the_build_sees_only_the_resolved_version(steps, tmp_path, name) -> None:
    inherited = {name: '{tag="9.9.9"}' if "METADATA" in name else "9.9.9"}
    calls = _run_step_environment_seen_by_uv(steps, tmp_path, inherited)
    assert calls, "the run step called uv"
    for call in calls:
        # The resolved release alone: any other setting could change the
        # version built into dependamerge, so it would disagree with the
        # action's version output.
        assert _version_settings(call) == {"SETUPTOOLS_SCM_PRETEND_VERSION": "x"}


def test_the_version_resolver_joins_each_value_to_its_option(steps) -> None:
    (version,) = [s for s in steps if s.get("id") == "version"]
    # A separate value that looks like an option ('--help') would be
    # read as one, and the resolver would exit before its fallback.
    assert re.search(r"--(ref|sha|repository-url) \"", version["run"]) is None
    for option in ("--ref=", "--sha=", "--repository-url="):
        assert option in version["run"]
