#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Resolve the composite action's options from its inputs and JSON config.

A scheduled caller keeps its scope in one JSON document, typically a
repository or organisation variable, so changing what a run covers is a
settings edit rather than a workflow change:

    {
      "target": "my-org",
      "exclude_repos": ["test-fixture-one", "test-fixture-two"],
      "max_wait": 900,
      "dry_run": false
    }

Each key names an action input.  An input given explicitly wins over the
config, and the config over the default; an empty input counts as not
given.  Every value is validated whichever source it came from, and an
unknown key is an error rather than ignored: a misspelt "exlude_repos"
must stop the run, not quietly widen it to every repository.

Every problem is reported at once.  Standard library only, as it runs
before the project is installed.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_SEPARATORS = re.compile(r"[,\s]+")
_TRUE, _FALSE = "true", "false"


class ConfigError(ValueError):
    """One or more options are invalid; ``problems`` lists them all."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise ValueError("must be a single line")
    return value.strip()


def _repos(value: Any) -> str:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        items = value
    else:
        raise ValueError("must be a string or a list of strings")
    names = [name for item in items for name in _SEPARATORS.split(item) if name]
    if not names:
        raise ValueError("names no repository; omit it to act on every one")
    return ",".join(names)


def _choice(*allowed: str) -> Callable[[Any], str]:
    def parse(value: Any) -> str:
        if value not in allowed:
            raise ValueError(f"must be one of: {', '.join(allowed)}")
        return str(value)

    return parse


def _seconds(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("must be a number of seconds")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("must be a number of seconds") from None
    if number < 0 or math.isnan(number) or math.isinf(number):
        raise ValueError("must be 0 or a positive number of seconds")
    return str(int(number)) if number.is_integer() else str(number)


def _flag(value: Any) -> str:
    if isinstance(value, bool):
        return _TRUE if value else _FALSE
    if isinstance(value, str) and value.strip().lower() in (_TRUE, _FALSE):
        return value.strip().lower()
    raise ValueError("must be true or false")


#: Option name -> (parser, default).  A default of None means required.
OPTIONS: dict[str, tuple[Callable[[Any], str], str | None]] = {
    "target": (_text, None),
    "include_repos": (_repos, ""),
    "exclude_repos": (_repos, ""),
    "merge_method": (_choice("merge", "squash", "rebase"), "merge"),
    "force": (_choice("none", "code-owners", "protection-rules", "all"), "code-owners"),
    "max_wait": (_seconds, "900"),
    "fix_out_of_date": (_flag, _TRUE),
    "dismiss_copilot": (_flag, _FALSE),
    "dry_run": (_flag, _FALSE),
}


def _load_config(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            [f"config: not valid JSON ({exc.msg}, line {exc.lineno})"]
        ) from None
    if not isinstance(config, dict):
        raise ConfigError(["config: must be a JSON object"])
    return config


def resolve(config_text: str, inputs: Mapping[str, str]) -> dict[str, str]:
    """Return every option's final value, as the action passes it on.

    Args:
        config_text: The ``config`` input; empty for none.
        inputs: Explicit input values by option name; empty means unset.

    Raises:
        ConfigError: Listing every problem found.
    """
    config = _load_config(config_text)
    problems = [
        f"config: unknown key '{key}' (known: {', '.join(OPTIONS)})"
        for key in config
        if key not in OPTIONS
    ]
    resolved: dict[str, str] = {}
    for name, (parse, default) in OPTIONS.items():
        given = inputs.get(name, "")
        source, value = (
            ("input", given) if given.strip() else ("config", config.get(name))
        )
        if source == "config" and name not in config:
            if default is None:
                problems.append(f"{name}: required, as an input or in config")
            else:
                resolved[name] = default
            continue
        try:
            resolved[name] = parse(value)
        except ValueError as exc:
            problems.append(f"{name} ({source}): {exc}")
    if resolved.get("include_repos") and resolved.get("exclude_repos"):
        problems.append(
            "include_repos and exclude_repos are mutually exclusive; set one"
        )
    if problems:
        raise ConfigError(problems)
    return resolved


def command_data(text: str) -> str:
    """Escape ``text`` for the message of a workflow command.

    Problems quote config keys verbatim, so a key holding a line break
    could otherwise end the command and start a forged one.
    """
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(environ: Mapping[str, str] | None = None) -> int:
    """Resolve options from ``INPUT_*`` variables into ``GITHUB_OUTPUT``."""
    env = os.environ if environ is None else environ
    inputs = {name: env.get(f"INPUT_{name.upper()}", "") for name in OPTIONS}
    try:
        options = resolve(env.get("INPUT_CONFIG", ""), inputs)
    except ConfigError as exc:
        for problem in exc.problems:
            print(
                f"::error title=dependamerge options::{command_data(problem)}",
                file=sys.stderr,
            )
        return 2
    lines = [f"{name}={value}" for name, value in options.items()]
    output = env.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    # The target is logged as set, never by value: a URL target may
    # carry a credential in its userinfo, query or fragment.
    shown = [
        "target=(set)" if name == "target" else f"{name}={value}"
        for name, value in options.items()
    ]
    print("Resolved options:\n  " + "\n  ".join(shown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
