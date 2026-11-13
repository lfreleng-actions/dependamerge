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
# The runner context fields a step may read: the action keeps its
# interpreter, uv's tools and its cache under the runner's temp directory.
_RUNNER_FIELDS = frozenset({"temp"})
# A whole expression that is one plain property path; anything else (a
# function call, an index, a bare context, another context) is refused.
_PROPERTY_PATH = re.compile(
    r"\A(?:inputs\.(?P<input>[a-z][a-z0-9_]*)"
    r"|steps\.(?P<step>[a-z][a-z0-9_-]*)\.outputs\.(?P<output>[a-z][a-z0-9_-]*)"
    r"|github\.(?P<github>[a-z][a-z0-9_]*)"
    r"|runner\.(?P<runner>[a-z][a-z0-9_]*))\Z"
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
    elif path["runner"] is not None:
        if path["runner"] not in _RUNNER_FIELDS:
            return "is not an allowed runner field"
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
        "runner.environment",
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
        ("runner.temp", "other"),
        ("steps.options.outputs.uv-path", "other"),
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
    first_tool = script.index('"$UV" sync')
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


def _run_step_uv_calls(
    steps: list[dict[str, Any]], tmp_path: Path, inherited: dict[str, str]
) -> list[dict[str, Any]]:
    """Run the run step as the runner would; return each uv call.

    The step's own ``env`` is laid over the inherited job environment,
    as the runner does, and the script runs under the runner's bash
    options. uv and the installed dependamerge are recorders, so the
    test needs neither a network nor a build. Each call is a mapping of
    its ``argv`` and ``env``.
    """
    (run,) = [s for s in steps if s.get("id") == "run"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    seen = tmp_path / "uv-calls.jsonl"
    recorder = (
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(seen)!r}, 'a') as f:\n"
        "    call = {'argv': sys.argv[1:], 'env': dict(os.environ)}\n"
        "    f.write(json.dumps(call) + '\\n')\n"
    )
    uv = bin_dir / "uv"
    uv.write_text(recorder)
    uv.chmod(0o755)
    venv_bin = tmp_path / "dependamerge-venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "dependamerge").write_text("#!/bin/sh\nexit 0\n")
    (venv_bin / "dependamerge").chmod(0o755)

    def evaluate(value: Any) -> str:
        # Every expression the step uses names an input or a step output;
        # setup-uv's uv is the recorder.
        return re.sub(
            r"\$\{\{\s*(.*?)\s*\}\}",
            lambda m: str(uv) if m[1] == "steps.setup-uv.outputs.uv-path" else "x",
            str(value),
        )

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


def _run_step_environment_seen_by_uv(
    steps: list[dict[str, Any]], tmp_path: Path, inherited: dict[str, str]
) -> list[dict[str, str]]:
    """Run the run step as the runner would; return each uv call's env."""
    return [call["env"] for call in _run_step_uv_calls(steps, tmp_path, inherited)]


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


# Composite steps run in the caller's workspace, which can hold anything
# the caller checked out. None of it may stand in for the action's own
# Python: each file below records that it ran, then fails.
_PLANTED_MODULE = """\
import os
import sys

marker = os.path.join(os.environ["RUNNER_TEMP"], "planted-ran")
with open(marker, "a", encoding="utf-8") as handle:
    handle.write(__name__ + "\\n")
sys.exit(97)
"""
_PLANTED_INTERPRETER = """\
#!/bin/sh
echo "interpreter $0" >> "$RUNNER_TEMP/planted-ran"
exit 97
"""
_STEP_VALUES = {
    "github.action_ref": "v1.2.3",
    "github.action_repository": "lfreleng-actions/dependamerge",
    "inputs.target": "acme",
    "inputs.fail_on_merge_errors": "true",
    "inputs.slack_when": "always",
    "steps.python.outputs.path": sys.executable,
}


def _plant(workspace: Path) -> None:
    """Fill ``workspace`` with files that would take over a Python run."""
    for name in ("json", "re", "sitecustomize", "dependamerge/__init__"):
        module = workspace / f"{name}.py"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text(_PLANTED_MODULE)
    # The interpreter a .python-version, or a .venv, would have uv run.
    for interpreter in ("bin/python3", ".venv/bin/python", ".venv/bin/python3"):
        path = workspace / interpreter
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_PLANTED_INTERPRETER)
        path.chmod(0o755)
    (workspace / ".python-version").write_text("./bin/python3\n")
    (workspace / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")


def _run_step_in_workspace(
    step: dict[str, Any],
    tmp_path: Path,
    inherited: dict[str, str],
    values: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run ``step`` as the runner would, from a workspace of planted files.

    ``values`` supplies expressions beyond the defaults. Returns the
    finished process, the step's output file and the marker any planted
    file writes when it runs.
    """
    values = {**_STEP_VALUES, **(values or {})}
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _plant(workspace)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir(exist_ok=True)
    # The planted interpreters come first, as a caller's $GITHUB_PATH
    # entry would put them.
    path = [
        str(workspace / ".venv" / "bin"),
        str(workspace / "bin"),
        os.path.dirname(sys.executable),
        "/usr/bin",
        "/bin",
    ]

    def evaluate(value: Any) -> str:
        return re.sub(
            r"\$\{\{\s*(.*?)\s*\}\}", lambda m: values.get(m[1], ""), str(value)
        )

    env = {
        "PATH": os.pathsep.join(path),
        "HOME": str(tmp_path),
        "RUNNER_TEMP": str(runner_temp),
        "RUNNER_OS": "Linux",
        "GITHUB_ACTION_PATH": str(_ACTION.parent),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "acme/caller",
        "GITHUB_SHA": "0" * 40,
        **inherited,
        **{k: evaluate(v) for k, v in (step.get("env") or {}).items()},
    }
    script = tmp_path / "step.sh"
    script.write_text(step["run"])
    finished = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return finished, tmp_path / "output", runner_temp / "planted-ran"


# An empty PYTHONPATH entry, as 'X:$PYTHONPATH' leaves when PYTHONPATH
# is unset, names the working directory.
_INHERITED = pytest.mark.parametrize(
    "inherited", [{}, {"PYTHONPATH": ":"}], ids=["workspace", "empty-pythonpath"]
)


@_INHERITED
@pytest.mark.parametrize(
    ("step_id", "expected"),
    [("version", "version=1.2.3"), ("options", "target=acme")],
)
def test_python_steps_run_nothing_from_the_workspace(
    steps, tmp_path, inherited, step_id, expected
) -> None:
    (step,) = [s for s in steps if s.get("id") == step_id]
    finished, output, marker = _run_step_in_workspace(step, tmp_path, inherited)
    assert not marker.exists(), marker.read_text()
    assert finished.returncode == 0, finished.stderr
    assert expected in output.read_text().splitlines()


@_INHERITED
def test_the_summary_imports_the_installed_report(steps, tmp_path, inherited) -> None:
    # The test environment stands in for the one the run step builds.
    probe = subprocess.run(
        [sys.executable, "-I", "-c", "import dependamerge.ci_report"],
        cwd=tmp_path,
        capture_output=True,
    )
    if sys.prefix == sys.base_prefix or probe.returncode != 0:
        pytest.skip("dependamerge is not installed in a virtual environment")
    from dependamerge.ci_report import build_document

    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    (runner_temp / "dependamerge-venv").symlink_to(sys.prefix)
    results = tmp_path / "results.json"
    document = build_document(
        [],
        explain=lambda _: [],
        target="acme",
        scope="owner",
        preview=False,
        dry_run=True,
    )
    results.write_text(json.dumps(document))
    (step,) = [s for s in steps if s.get("id") == "summary"]
    values = {"steps.run.outputs.results_file": str(results)}
    finished, output, marker = _run_step_in_workspace(step, tmp_path, inherited, values)
    assert not marker.exists(), marker.read_text()
    assert finished.returncode == 0, finished.stderr
    assert "total=0" in output.read_text().splitlines()


def test_the_run_step_drops_an_inherited_pythonpath(steps, tmp_path) -> None:
    # The build and dependamerge run from the caller's workspace, so an
    # empty entry would put it on their sys.path.
    calls = _run_step_environment_seen_by_uv(steps, tmp_path, {"PYTHONPATH": ":"})
    assert calls, "the run step called uv"
    for call in calls:
        assert "PYTHONPATH" not in call


# Variables that name or steer an interpreter, as a step earlier in the
# caller's job could set through $GITHUB_ENV. None may reach a uv call
# or a Python start, and none may decide which interpreter runs.
_INTERPRETER_SELECTORS = (
    "UV_PYTHON",
    "UV_PYTHON_PREFERENCE",
    "UV_PYTHON_INSTALL_DIR",
    "UV_SYSTEM_PYTHON",
    "UV_MANAGED_PYTHON",
    "UV_INTERNAL__PARENT_INTERPRETER",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONHOME",
    "PYTHONEXECUTABLE",
    "__PYVENV_LAUNCHER__",
)
# uv's valid python-preference values; any other makes it refuse its
# arguments before it looks for an interpreter.
_PYTHON_PREFERENCES = ("only-managed", "managed", "system", "only-system")


def _setup_uv(steps: list[dict[str, Any]]) -> dict[str, Any]:
    (setup,) = [s for s in steps if str(s.get("uses", "")).startswith("astral-sh/")]
    return setup


def test_setup_uv_cannot_run_a_caller_interpreter(steps) -> None:
    setup = _setup_uv(steps)
    assert setup.get("id") == "setup-uv"
    options = setup["with"]
    env = setup["env"]
    # Its own bin directory, not the caller's HOME or XDG_BIN_HOME.
    assert options["tool-bin-dir"].startswith("${{ runner.temp }}/")
    assert options["enable-cache"] is False
    # python-version exports UV_PYTHON to the caller's later steps, and
    # activate-environment puts a venv on their PATH.
    assert "python-version" not in options
    assert "activate-environment" not in options
    assert env["UV_NO_CONFIG"] == "1"
    assert env["UV_NO_PROJECT"] == "1"
    assert env["UV_PYTHON_PREFERENCE"] not in _PYTHON_PREFERENCES
    assert env["UV_PYTHON_ARCH"] == ""
    assert env["UV_PYTHON_INSTALL_DIR"].startswith("${{ runner.temp }}/")
    # Each would let setup-uv's own lookup, or its bare uv, run a
    # caller's interpreter.
    for name in ("UV_SYSTEM_PYTHON", "UV_MANAGED_PYTHON", "UV_NO_MODIFY_PATH"):
        assert name not in env, name


def _commands(script: str) -> list[str]:
    """The script's logical lines, comments dropped and continuations joined."""
    lines = [line for line in script.splitlines() if not line.lstrip().startswith("#")]
    return re.sub(r"\\\n\s*", " ", "\n".join(lines)).splitlines()


def _arrays(script: str) -> dict[str, str]:
    """Each bash array the script assigns, by name, with its elements."""
    return dict(re.findall(r"\b(\w+)=\((.*?)\)", script, re.DOTALL))


_UV_CALL = re.compile(r'(?:^|[\s;(|&{])("\$UV"|uvx?)\s')
_TRUSTED_PYTHON = '--python "$PYTHON"'


def _uv_call_refusals(step: dict[str, Any]) -> list[str]:
    script = step.get("run") or ""
    arrays = _arrays(script)
    found = []
    for command in _commands(script):
        for call in _UV_CALL.finditer(command):
            if call[1] != '"$UV"':
                found.append(f"{_label(step)}: uv from PATH: {command.strip()}")
                continue
            if "env -i " in command[: call.start(1)]:
                continue
            expanded = command + " ".join(
                arrays[name]
                for name in re.findall(r'"\$\{(\w+)\[@\]\}"', command)
                if name in arrays
            )
            if _TRUSTED_PYTHON not in expanded:
                found.append(f"{_label(step)}: no --python: {command.strip()}")
    return found


def test_every_uv_call_names_the_trusted_interpreter(steps) -> None:
    calls = [_UV_CALL.search(c) for s in steps for c in _commands(s.get("run") or "")]
    assert any(calls), "the action calls uv"
    assert [r for step in steps for r in _uv_call_refusals(step)] == []


@pytest.mark.parametrize(
    "command",
    [
        "uv sync --locked",
        'uvx --python "$PYTHON" tool',
        '"$UV" run --no-project script.py',
        '"$UV" sync --python ">=3.10"',
        '"$UV" pip install --system x',
    ],
)
def test_a_uv_call_without_the_trusted_interpreter_is_refused(command) -> None:
    assert _uv_call_refusals({"id": "x", "run": command}) != []


@pytest.mark.parametrize(
    "script",
    [
        '"$UV" run --python "$PYTHON" script.py',
        'env -i "${clean[@]}" "$UV" python find --managed-python 3.13',
        'args=(--locked\n  --python "$PYTHON")\n"$UV" sync "${args[@]}"',
        '"$UV" sync \\\n  --python "$PYTHON"',
    ],
)
def test_a_uv_call_with_the_trusted_interpreter_is_accepted(script) -> None:
    assert _uv_call_refusals({"id": "x", "run": script}) == []


# A Python start in command position: the trusted interpreter, or the
# environment built from it, named through a variable.
_COMMAND_START = r"(?:^|\$\(|[;{|&]|&&|\|\|)\s*(?:\w+=\S*\s+)*"
_PYTHON_START = re.compile(_COMMAND_START + r'"\$(PYTHON|python)"(?P<rest>.*)')
# A python found on PATH, which the caller's job controls.
_BARE_PYTHON = re.compile(_COMMAND_START + r"python(?:3(?:\.\d+)?)?(?:\s|$)")


def test_every_python_start_is_isolated(steps) -> None:
    # macOS reads both even under -I.
    unset = "unset PYTHONEXECUTABLE __PYVENV_LAUNCHER__"
    starts = 0
    for step in steps:
        commands = [c.strip() for c in _commands(step.get("run") or "")]
        for index, command in enumerate(commands):
            assert _BARE_PYTHON.search(command) is None, command
            for start in _PYTHON_START.finditer(command):
                starts += 1
                assert start["rest"].startswith(" -I "), command
                assert unset in commands[:index], command
    # The version and options resolvers, and the summary's report.
    assert starts >= 3


def test_steps_take_uv_and_python_from_the_action_steps(steps) -> None:
    ids = [step.get("id") for step in steps]
    assert ids.index("setup-uv") < ids.index("python") < ids.index("version")
    for step in steps:
        env = step.get("env") or {}
        script = step.get("run") or ""
        if "$PYTHON" in script and step.get("id") != "python":
            assert env["PYTHON"] == "${{ steps.python.outputs.path }}", _label(step)
        if "$UV" in script:
            assert env["UV"] == "${{ steps.setup-uv.outputs.uv-path }}", _label(step)


def _run_install_step(
    steps: list[dict[str, Any]], tmp_path: Path, found: str
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]], Path]:
    """Run the Install Python step with a recording uv.

    The recorder answers ``python find`` with ``found``, where the
    placeholder ``{install_dir}`` names the step's install directory.
    Returns the finished process, each uv call, and the output file.
    """
    (step,) = [s for s in steps if s.get("id") == "python"]
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    seen = tmp_path / "uv-calls.jsonl"
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(seen)!r}, 'a') as f:\n"
        "    f.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}))\n"
        "    f.write('\\n')\n"
        "if sys.argv[1:3] == ['python', 'find']:\n"
        f"    print({found!r}.format(\n"
        "        install_dir=os.environ['UV_PYTHON_INSTALL_DIR']))\n"
    )
    uv.chmod(0o755)
    values = {
        "steps.setup-uv.outputs.uv-path": str(uv),
        "runner.temp": str(runner_temp),
    }

    def evaluate(value: Any) -> str:
        return re.sub(
            r"\$\{\{\s*(.*?)\s*\}\}", lambda m: values.get(m[1], ""), str(value)
        )

    workspace = tmp_path / "workspace"
    (workspace / ".venv" / "bin").mkdir(parents=True)
    hostile = {
        "UV_PYTHON": str(workspace / ".venv" / "bin" / "python"),
        "UV_PYTHON_INSTALL_DIR": str(workspace / "python"),
        "UV_PYTHON_INSTALL_MIRROR": "https://127.0.0.1:9",
        "UV_PYTHON_PREFERENCE": "only-system",
        "UV_CACHE_DIR": str(workspace / "cache"),
        "UV_INTERNAL__PARENT_INTERPRETER": str(workspace / ".venv/bin/python"),
        "VIRTUAL_ENV": str(workspace / ".venv"),
        "CONDA_PREFIX": str(workspace / ".venv"),
        "PYTHONHOME": str(workspace),
        "XDG_BIN_HOME": str(workspace / "xdg-bin"),
        "HTTPS_PROXY": "http://proxy.example:3128",
    }
    env = {
        "PATH": os.pathsep.join(["/usr/bin", "/bin"]),
        "HOME": str(tmp_path),
        "RUNNER_TEMP": str(runner_temp),
        "RUNNER_OS": "Linux",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        **hostile,
        **{k: evaluate(v) for k, v in (step.get("env") or {}).items()},
    }
    script = tmp_path / "step.sh"
    script.write_text(step["run"])
    finished = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    calls = [json.loads(line) for line in seen.read_text().splitlines()]
    return finished, calls, tmp_path / "output"


# What uv may see when it installs and finds the interpreter: its own
# settings, and the network settings a runner may need.
_INSTALL_ALLOWED = frozenset(
    {
        "UV_NO_CONFIG",
        "UV_NO_PROJECT",
        "UV_PYTHON_INSTALL_DIR",
        "UV_CACHE_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "UV_NATIVE_TLS",
    }
)


def test_the_install_step_hides_the_caller_environment(steps, tmp_path) -> None:
    finished, calls, output = _run_install_step(
        steps, tmp_path, "{install_dir}/cpython-3.13/bin/python3.13"
    )
    assert finished.returncode == 0, finished.stderr
    assert [call["argv"][:2] for call in calls] == [
        ["python", "install"],
        ["python", "find"],
    ]
    runner_temp = tmp_path / "runner-temp"
    for call in calls:
        # The recorder's own interpreter sets LC_CTYPE as it coerces the
        # C locale (PEP 538); it was not in the environment it was given.
        env = {k: v for k, v in call["env"].items() if k != "LC_CTYPE"}
        assert set(env) <= _INSTALL_ALLOWED, sorted(set(env) - _INSTALL_ALLOWED)
        assert env["UV_PYTHON_INSTALL_DIR"].startswith(f"{runner_temp}/")
        assert env["UV_CACHE_DIR"].startswith(f"{runner_temp}/")
        assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
    (path,) = [
        line.removeprefix("path=")
        for line in output.read_text().splitlines()
        if line.startswith("path=")
    ]
    assert path.startswith(f"{runner_temp}/")


def test_the_install_step_refuses_an_interpreter_found_elsewhere(
    steps, tmp_path
) -> None:
    finished, _, output = _run_install_step(
        steps, tmp_path, str(tmp_path / "workspace" / ".venv" / "bin" / "python")
    )
    assert finished.returncode != 0
    assert not output.exists() or "path=" not in output.read_text()


@pytest.mark.parametrize("name", _INTERPRETER_SELECTORS)
def test_the_run_step_builds_with_the_trusted_interpreter(
    steps, tmp_path, name
) -> None:
    inherited = {name: str(tmp_path / "workspace" / ".venv" / "bin" / "python")}
    calls = _run_step_uv_calls(steps, tmp_path, inherited)
    assert len(calls) == 2
    for call in calls:
        assert name not in call["env"]
        argv = call["argv"]
        # "x" is what the test substitutes for the Python step's output.
        assert argv[argv.index("--python") + 1] == "x"
        assert "--no-config" in argv


def test_the_run_step_uses_its_own_cache(steps, tmp_path) -> None:
    inherited = {"UV_CACHE_DIR": str(tmp_path / "workspace" / "cache")}
    calls = _run_step_uv_calls(steps, tmp_path, inherited)
    assert calls, "the run step called uv"
    for call in calls:
        assert call["env"]["UV_CACHE_DIR"].startswith(f"{tmp_path}/")
        assert "/workspace/" not in call["env"]["UV_CACHE_DIR"]


@pytest.mark.parametrize("step_id", ["version", "options"])
@pytest.mark.parametrize("name", _INTERPRETER_SELECTORS)
def test_python_steps_ignore_interpreter_selectors(
    steps, tmp_path, step_id, name
) -> None:
    (step,) = [s for s in steps if s.get("id") == step_id]
    workspace = tmp_path / "workspace"
    value = {
        "VIRTUAL_ENV": workspace / ".venv",
        "CONDA_PREFIX": workspace / ".venv",
        "PYTHONHOME": workspace,
        "UV_PYTHON_INSTALL_DIR": workspace,
        "UV_PYTHON_PREFERENCE": "only-system",
        "UV_SYSTEM_PYTHON": "1",
        "UV_MANAGED_PYTHON": "1",
    }.get(name, workspace / ".venv" / "bin" / "python")
    finished, output, marker = _run_step_in_workspace(
        step, tmp_path, {name: str(value)}
    )
    assert not marker.exists(), marker.read_text()
    assert finished.returncode == 0, finished.stderr
    assert output.read_text().strip(), "the step wrote its outputs"


def _run_refusal_step(
    steps: list[dict[str, Any]], inherited: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    (step,) = [s for s in steps if s.get("id") == "no-modify-path"]
    env = {k: v for k, v in os.environ.items() if k != "UV_NO_MODIFY_PATH"}
    return subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={**env, **inherited},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.xfail(strict=True, reason="no step refuses UV_NO_MODIFY_PATH yet")
def test_uv_no_modify_path_is_refused_before_setup_uv(steps) -> None:
    ids = [step.get("id") for step in steps]
    assert ids.index("no-modify-path") < ids.index("setup-uv")


@pytest.mark.xfail(strict=True, reason="no step refuses UV_NO_MODIFY_PATH yet")
@pytest.mark.parametrize("value", ["1", "0", ""])
def test_an_inherited_uv_no_modify_path_stops_the_action(steps, value) -> None:
    # setup-uv leaves PATH alone when it is set, so a bare uv would then
    # be whichever one the caller's PATH names.
    finished = _run_refusal_step(steps, {"UV_NO_MODIFY_PATH": value})
    assert finished.returncode == 1, finished.stderr
    assert "UV_NO_MODIFY_PATH" in finished.stdout


@pytest.mark.xfail(strict=True, reason="no step refuses UV_NO_MODIFY_PATH yet")
def test_the_action_runs_without_uv_no_modify_path(steps) -> None:
    finished = _run_refusal_step(steps, {})
    assert finished.returncode == 0, finished.stderr
