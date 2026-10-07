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
from decimal import Decimal, InvalidOperation
from typing import Any, NoReturn

_SEPARATORS = re.compile(r"[,\s]+")
_TRUE, _FALSE = "true", "false"
#: A scheme at the start of a target, as the CLI recognises one.
_SCHEME = re.compile(r"\A[A-Za-z][A-Za-z0-9+.-]*://")
# Where an authority ends: an '@' past any of these is path, query or
# fragment, never userinfo.
_AUTHORITY_END = re.compile(r"[/?#]")


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


def _target(value: Any) -> str:
    target = _text(value)
    if target.startswith("-"):
        # Option-shaped, such as '--include-human-prs': refused outright
        # rather than relying on the run step's '--' alone.
        raise ValueError("must name an owner, repository or pull request")
    return target


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
    """Validate a number of seconds, exactly, and render it for the CLI.

    Parsed as a ``Decimal`` (config floats arrive as one too), so nothing
    rounds before it is judged: a negative that would underflow to -0.0
    stays negative, and a positive too small for a float is refused
    rather than silently becoming 0, which means fire-and-forget.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError("must be a number of seconds")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except InvalidOperation:
        raise ValueError("must be a number of seconds") from None
    if not number.is_finite() or number.is_signed():
        raise ValueError("must be 0 or a positive number of seconds")
    seconds = float(number)  # what the CLI will parse
    if math.isinf(seconds):
        raise ValueError("must be a number of seconds")
    if number > 0 and seconds == 0:
        raise ValueError("is too small to represent; use 0 or a larger value")
    return str(int(number)) if number == number.to_integral_value() else str(seconds)


def _flag(value: Any) -> str:
    if isinstance(value, bool):
        return _TRUE if value else _FALSE
    if isinstance(value, str) and value.strip().lower() in (_TRUE, _FALSE):
        return value.strip().lower()
    raise ValueError("must be true or false")


#: Option name -> (parser, default).  A default of None means required.
OPTIONS: dict[str, tuple[Callable[[Any], str], str | None]] = {
    "target": (_target, None),
    "include_repos": (_repos, ""),
    "exclude_repos": (_repos, ""),
    "merge_method": (_choice("merge", "squash", "rebase"), "merge"),
    "force": (_choice("none", "code-owners", "protection-rules", "all"), "code-owners"),
    "max_wait": (_seconds, "900"),
    "fix_out_of_date": (_flag, _TRUE),
    "dismiss_copilot": (_flag, _FALSE),
    "dry_run": (_flag, _FALSE),
}


class _DuplicateKeys(Exception):
    """A JSON object named the same key more than once."""

    def __init__(self, keys: Sequence[str]) -> None:
        super().__init__(", ".join(keys))
        self.keys = list(keys)


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json.loads`` object hook that refuses duplicate keys."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for key, _value in pairs:
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise _DuplicateKeys(duplicates)
    return dict(pairs)


class _NonStandardConstant(ValueError):
    """A JSON document used NaN, Infinity or -Infinity."""


def _no_constant(name: str) -> NoReturn:
    """``json.loads`` constant hook: JSON has no NaN or Infinity.

    Python's decoder accepts them by default, and since only the value
    that wins is validated, one under a key an input overrides would
    otherwise pass in a config that is not valid JSON.
    """
    raise _NonStandardConstant(name)


def _load_config(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        # Decimal for integers and floats alike, so a number is judged
        # exactly as written, sign included ('-0'), see _seconds.
        config = json.loads(
            text,
            parse_float=Decimal,
            parse_int=Decimal,
            parse_constant=_no_constant,
            object_pairs_hook=_unique_keys,
        )
    except _DuplicateKeys as exc:
        # JSON's last-value-wins would let a stray duplicate quietly undo
        # a setting such as dry_run, so any duplicate stops the run.
        raise ConfigError(
            [f"config: duplicate key '{key}'" for key in exc.keys]
        ) from None
    except _NonStandardConstant as exc:
        raise ConfigError(
            [f"config: not valid JSON ({exc} is not a JSON value)"]
        ) from None
    except json.JSONDecodeError as exc:
        raise ConfigError(
            [f"config: not valid JSON ({exc.msg}, line {exc.lineno})"]
        ) from None
    except (ValueError, ArithmeticError) as exc:
        # Valid JSON Python still refuses: a number past the interpreter's
        # digit limit (ValueError), or an exponent beyond Decimal's range
        # (InvalidOperation, an ArithmeticError).
        raise ConfigError([f"config: cannot be read ({exc!r})"]) from None
    if not isinstance(config, dict):
        raise ConfigError(["config: must be a JSON object"])
    return config


def _named_host(target: str) -> str | None:
    """The host a target names, or ``None`` for shorthand.

    The CLI's own reading: a scheme (``https://``) or a network-path
    reference (``//host/...``) at the start introduces an authority, and
    so does a first segment holding a dot or a colon, which no login or
    repository name has (``ghe.example.com/acme``, or an scp remote such
    as ``git@ghe:acme/widget.git``).  Shorthand (``owner``,
    ``owner/repo``) carries no host and resolves to the pinned server.
    """
    scheme = _SCHEME.match(target)
    if scheme:
        rest = target[scheme.end() :]
    elif target.startswith("//"):
        rest = target[2:]
    else:
        rest = target
        segment = target.split("/", 1)[0]
        if "." not in segment and ":" not in segment:
            return None
    authority = _AUTHORITY_END.split(rest, maxsplit=1)[0]
    return authority.rsplit("@", 1)[-1].split(":", 1)[0].lower()


def resolve(
    config_text: str, inputs: Mapping[str, str], server: str = ""
) -> dict[str, str]:
    """Return every option's final value, as the action passes it on.

    Args:
        config_text: The ``config`` input; empty for none.
        inputs: Explicit input values by option name; empty means unset.
        server: Host of the GitHub server running the workflow; when set,
            a target naming any other host is refused.

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
    # Whether each list was supplied, not whether it parsed: a malformed
    # list must not hide that both were given.
    supplied = {
        name
        for name in ("include_repos", "exclude_repos")
        if inputs.get(name, "").strip() or name in config
    }
    if len(supplied) == 2:
        problems.append(
            "include_repos and exclude_repos are mutually exclusive; set one"
        )
    target = resolved.get("target")
    if server and target:
        # The token goes to the server running the workflow and nowhere
        # else: a target naming another host is refused.
        host = _named_host(target)
        if host is not None and host != server.lower():
            problems.append(
                "target: must be on the GitHub server running this workflow"
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
        options = resolve(
            env.get("INPUT_CONFIG", ""),
            inputs,
            server=env.get("GITHUB_SERVER_URL", "").split("://", 1)[-1],
        )
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
