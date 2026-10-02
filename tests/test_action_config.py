# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for scripts/action_config.py, the action's option resolver."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "action_config.py"


@pytest.fixture(scope="module")
def action_config() -> ModuleType:
    spec = importlib.util.spec_from_file_location("action_config", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(**values: Any) -> str:
    return json.dumps(values)


class TestPrecedence:
    def test_defaults_fill_everything_but_the_target(self, action_config):
        options = action_config.resolve("", {"target": "acme"})
        assert options == {
            "target": "acme",
            "include_repos": "",
            "exclude_repos": "",
            "merge_method": "merge",
            "force": "code-owners",
            "max_wait": "900",
            "fix_out_of_date": "true",
            "dismiss_copilot": "false",
            "dry_run": "false",
        }

    def test_config_supplies_what_inputs_leave_empty(self, action_config):
        options = action_config.resolve(
            _config(target="acme", dry_run=True, max_wait=0),
            {"target": "", "dry_run": ""},
        )
        assert options["target"] == "acme"
        assert options["dry_run"] == "true"
        assert options["max_wait"] == "0"

    def test_an_explicit_input_wins_over_config(self, action_config):
        options = action_config.resolve(
            _config(target="acme", dry_run=False), {"dry_run": "true"}
        )
        assert options["dry_run"] == "true"

    def test_whitespace_input_counts_as_unset(self, action_config):
        options = action_config.resolve(_config(target="acme"), {"target": "  "})
        assert options["target"] == "acme"


class TestValues:
    def test_repository_lists_accept_strings_and_arrays(self, action_config):
        from_list = action_config.resolve(
            _config(target="acme", exclude_repos=["a", "b c"]), {}
        )
        from_text = action_config.resolve(
            "", {"target": "acme", "exclude_repos": "a,\nb"}
        )
        assert from_list["exclude_repos"] == "a,b,c"
        assert from_text["exclude_repos"] == "a,b"

    def test_fractional_seconds_survive(self, action_config):
        options = action_config.resolve(_config(target="acme", max_wait=2.5), {})
        assert options["max_wait"] == "2.5"

    def test_flags_accept_booleans_and_their_spellings(self, action_config):
        options = action_config.resolve(
            _config(target="acme", fix_out_of_date=False), {"dismiss_copilot": "True"}
        )
        assert options["fix_out_of_date"] == "false"
        assert options["dismiss_copilot"] == "true"


class TestProblems:
    def _problems(
        self, action_config, config: str, inputs: dict[str, str]
    ) -> list[str]:
        with pytest.raises(action_config.ConfigError) as excinfo:
            action_config.resolve(config, inputs)
        problems: list[str] = list(excinfo.value.problems)
        return problems

    def test_a_missing_target_is_reported(self, action_config):
        assert self._problems(action_config, "", {}) == [
            "target: required, as an input or in config"
        ]

    def test_an_unknown_key_stops_the_run(self, action_config):
        problems = self._problems(
            action_config, _config(target="acme", exlude_repos=["a"]), {}
        )
        assert len(problems) == 1
        assert "unknown key 'exlude_repos'" in problems[0]

    def test_every_problem_is_reported_at_once(self, action_config):
        problems = self._problems(
            action_config,
            _config(target="acme", merge_method="fast", max_wait=-1, dry_run="yes"),
            {"force": "max"},
        )
        assert len(problems) == 4
        assert any(p.startswith("force (input)") for p in problems)
        assert any(p.startswith("merge_method (config)") for p in problems)

    def test_include_and_exclude_are_mutually_exclusive(self, action_config):
        problems = self._problems(
            action_config,
            _config(target="acme", include_repos="a"),
            {"exclude_repos": "b"},
        )
        assert problems == [
            "include_repos and exclude_repos are mutually exclusive; set one"
        ]

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            ("{not json", "not valid JSON"),
            ("[1, 2]", "must be a JSON object"),
        ],
    )
    def test_malformed_config_is_refused(self, action_config, config, expected):
        problems = self._problems(action_config, config, {"target": "acme"})
        assert expected in problems[0]

    @pytest.mark.parametrize("value", ["", [], ",", [" "]])
    def test_a_list_naming_nothing_is_refused(self, action_config, value):
        problems = self._problems(
            action_config, _config(target="acme", include_repos=value), {}
        )
        assert "names no repository" in problems[0]

    def test_a_multi_line_target_is_refused(self, action_config):
        # A line break would inject a second step output.
        problems = self._problems(action_config, _config(target="acme\nx=y"), {})
        assert "single line" in problems[0]

    @pytest.mark.parametrize("value", [True, "soon", float("inf")])
    def test_max_wait_must_be_seconds(self, action_config, value):
        config = json.dumps({"target": "acme", "max_wait": value})
        assert self._problems(action_config, config, {})


class TestMain:
    def test_writes_outputs(self, action_config, tmp_path, capsys):
        output = tmp_path / "output"
        code = action_config.main(
            {
                "INPUT_CONFIG": _config(target="acme", exclude_repos=["x"]),
                "INPUT_DRY_RUN": "true",
                "GITHUB_OUTPUT": str(output),
            }
        )
        assert code == 0
        lines = output.read_text().splitlines()
        assert "target=acme" in lines
        assert "exclude_repos=x" in lines
        assert "dry_run=true" in lines

    def test_problems_become_workflow_errors(self, action_config, tmp_path, capsys):
        output = tmp_path / "output"
        code = action_config.main({"INPUT_CONFIG": "{}", "GITHUB_OUTPUT": str(output)})
        assert code == 2
        assert not output.exists()
        assert capsys.readouterr().err.startswith(
            "::error title=dependamerge options::target: required"
        )

    def test_a_hostile_key_cannot_forge_a_command(self, action_config, capsys):
        config = json.dumps({"target": "acme", "x\n::warning::forged": 1})
        assert action_config.main({"INPUT_CONFIG": config}) == 2
        err = capsys.readouterr().err
        assert err.count("\n") == 1  # one command, one line
        assert "%0A::warning::forged" in err

    def test_the_target_value_is_never_logged(self, action_config, tmp_path, capsys):
        output = tmp_path / "output"
        target = "https://github.com/acme?token=hunter2"
        env = {"INPUT_TARGET": target, "GITHUB_OUTPUT": str(output)}
        assert action_config.main(env) == 0
        assert "hunter2" not in capsys.readouterr().out
        assert f"target={target}" in output.read_text().splitlines()
