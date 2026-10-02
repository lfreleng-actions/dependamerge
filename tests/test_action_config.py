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

    @pytest.mark.parametrize(
        ("server", "target", "allowed"),
        [
            ("github.com", "acme", True),
            ("github.com", "acme/widget", True),
            ("github.com", "https://github.com/acme", True),
            ("github.com", "git@github.com:acme/widget.git", True),
            ("ghe.acme.com", "acme/widget", True),
            ("ghe.acme.com", "https://GHE.acme.com/acme", True),
            ("ghe.acme.com", "https://github.com/acme/widget", False),
            ("ghe.acme.com", "github.com/acme", False),
            ("github.com", "https://evil.example/x", False),
            # Network-path references name a host too.
            ("github.com", "//evil.example/acme", False),
            ("github.com", "//github.com/acme", True),
            # So does an scp remote, even with an undotted host.
            ("github.com", "git@evil:acme/widget.git", False),
            ("github.com", "evil:acme/widget", False),
            # Only a leading scheme names the host; a later one does not.
            ("github.com", "evil.example/x?u=https://github.com/acme", False),
            ("github.com", "HTTPS://GitHub.com/acme", True),
            # The authority ends at '?' or '#' as well as '/': an '@'
            # after either is query or fragment, never userinfo.
            ("github.com", "https://evil.example?next=@github.com/acme/widget", False),
            ("github.com", "https://evil.example#@github.com/acme", False),
            ("github.com", "//evil.example?@github.com/acme", False),
            ("github.com", "evil.example?x=@github.com/acme", False),
            ("github.com", "https://user@github.com/acme", True),
            ("github.com", "https://github.com/acme?tab=repositories", True),
        ],
    )
    def test_a_target_on_another_host_is_refused(
        self, action_config, server, target, allowed
    ):
        # Config can supply the target too, so the resolver checks it.
        config = _config(target=target)
        if allowed:
            assert action_config.resolve(config, {}, server=server)["target"]
        else:
            problems = self._problems_on(action_config, config, server)
            assert problems == [
                "target: must be on the GitHub server running this workflow"
            ]

    def _problems_on(self, action_config, config: str, server: str) -> list[str]:
        with pytest.raises(action_config.ConfigError) as excinfo:
            action_config.resolve(config, {}, server=server)
        problems: list[str] = list(excinfo.value.problems)
        return problems

    def test_a_duplicate_key_stops_the_run(self, action_config):
        # Last-value-wins would quietly turn this dry run live.
        config = '{"target": "acme", "dry_run": true, "dry_run": false}'
        problems = self._problems(action_config, config, {})
        assert problems == ["config: duplicate key 'dry_run'"]

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
            # Python's json accepts these; JSON does not. Each is refused
            # even where an explicit input overrides the key, since
            # only the winning value is validated.
            ('{"target": NaN}', "not valid JSON (NaN is not a JSON value)"),
            ('{"target": Infinity}', "not valid JSON (Infinity is not a JSON value)"),
            (
                '{"max_wait": -Infinity}',
                "not valid JSON (-Infinity is not a JSON value)",
            ),
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

    @pytest.mark.parametrize("source", ["input", "config"])
    def test_an_option_shaped_target_is_refused(self, action_config, source):
        target = "--include-human-prs"
        config, inputs = ("", {"target": target})
        if source == "config":
            config, inputs = _config(target=target), {}
        problems = self._problems(action_config, config, inputs)
        assert problems == [
            f"target ({source}): must name an owner, repository or pull request"
        ]

    @pytest.mark.parametrize("value", [True, "soon", float("inf")])
    def test_max_wait_must_be_seconds(self, action_config, value):
        config = json.dumps({"target": "acme", "max_wait": value})
        assert self._problems(action_config, config, {})

    @pytest.mark.parametrize("raw", ["-1", "-1e-400", "-0.0", "-0"])
    def test_negative_max_wait_fails_closed(self, action_config, raw):
        # -1e-400 underflows to -0.0, and the integer -0 loses its sign as
        # an int; neither may pass as 0.
        config = '{"target": "acme", "max_wait": ' + raw + "}"
        problems = self._problems(action_config, config, {})
        assert problems == [
            "max_wait (config): must be 0 or a positive number of seconds"
        ]

    @pytest.mark.parametrize("source", ["input", "config"])
    def test_a_tiny_positive_max_wait_is_not_zero(self, action_config, source):
        # 1e-400 rounds to 0.0 as a float, which would mean fire-and-forget.
        if source == "config":
            config, inputs = '{"target": "acme", "max_wait": 1e-400}', {}
        else:
            config, inputs = "", {"target": "acme", "max_wait": "1e-400"}
        problems = self._problems(action_config, config, inputs)
        assert problems == [
            f"max_wait ({source}): is too small to represent; use 0 or a larger value"
        ]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("0", "0"), ("900", "900"), ("0.5", "0.5"), ("1e3", "1000")],
    )
    def test_seconds_render_for_the_cli(self, action_config, raw, expected):
        options = action_config.resolve("", {"target": "acme", "max_wait": raw})
        assert options["max_wait"] == expected

    def test_a_huge_max_wait_is_a_problem_not_a_crash(self, action_config):
        config = '{"target": "acme", "max_wait": 1' + "0" * 400 + "}"
        problems = self._problems(action_config, config, {})
        assert problems == ["max_wait (config): must be a number of seconds"]

    def test_an_out_of_range_exponent_is_a_problem_not_a_crash(self, action_config):
        # Decimal raises InvalidOperation, which is not a ValueError.
        config = '{"target": "acme", "max_wait": 1e1000000000000000000}'
        problems = self._problems(action_config, config, {})
        assert problems[0].startswith("config: cannot be read")

    def test_an_unreadable_integer_is_a_problem_not_a_crash(
        self, action_config, monkeypatch
    ):
        # Python 3.11+ refuses integers past its digit limit with a plain
        # ValueError; simulate that limit on every supported version.
        def refuse(text, **kwargs):
            raise ValueError("Exceeds the limit (4300 digits)")

        monkeypatch.setattr(action_config.json, "loads", refuse)
        problems = self._problems(action_config, '{"target": "acme"}', {})
        assert problems[0].startswith("config: cannot be read")

    def test_both_lists_are_reported_even_when_one_is_malformed(self, action_config):
        problems = self._problems(
            action_config,
            _config(target="acme", include_repos=[1]),
            {"exclude_repos": "b"},
        )
        assert any(p.startswith("include_repos (config)") for p in problems)
        assert "include_repos and exclude_repos are mutually exclusive; set one" in (
            problems
        )


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
