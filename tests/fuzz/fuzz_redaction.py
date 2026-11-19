# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Fuzz the redaction helpers that stand between a secret and a log.

Run from the repository root::

    uv run --with atheris python tests/fuzz/fuzz_redaction.py -max_total_time=60

Contracts checked, in each module's own words:

* ``git_ops.redaction.redact_text``: "redact likely secrets
  (tokens/passwords) from a string", covering "all known GitHub token
  formats, GitLab tokens", the ``http(s)://user:password@host`` password
  "masked whole", and ``x-access-token:<token>@``.  A secret of one of
  those shapes, planted in arbitrary text, never reaches the output.
* ``error_text.summarise_error``: "Render *error* as the single line a
  failure report can show", "Idempotent", and the httpx "For more
  information check:" trailer goes while every word before it stays.

Output is encoded as strict UTF-8, as a log stream would, so a value
that cannot be written fails here rather than in a report.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(_ROOT / "src"), str(_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from dependamerge.error_text import summarise_error  # noqa: E402
from dependamerge.git_ops.redaction import redact_text  # noqa: E402
from tests.fuzz._harness import fields, run, violation  # noqa: E402

#: Marks a planted secret.  The text around it never holds one.
SECRET = "~"

TOKEN_PREFIXES = (
    "ghp_",
    "ghs_",
    "ghu_",
    "gho_",
    "ghr_",
    "github_pat_",
    "glpat-",
    "GHP_",
)
SCHEMES = ("https://", "http://", "HTTPS://")
TRAILER = "For more information check:"


def _clean(value: str, refused: str = "") -> str:
    return "".join(c for c in value if c not in refused and c != SECRET)


def _token(mode: int, body: str) -> str:
    alphanumeric = "".join(c for c in body if c.isascii() and c.isalnum())
    return (
        TOKEN_PREFIXES[(mode >> 2 & 7) % len(TOKEN_PREFIXES)]
        + (alphanumeric + "A" * 36)[:36]
    )


def _check_redact(mode: int, parts: list[str], redact: Callable[[str], str]) -> None:
    prefix, suffix, first, second = parts
    kind = mode & 3
    if kind == 0:
        secret = _token(mode, first)
        text = _clean(prefix) + secret + _clean(suffix)
    elif kind == 1:
        secret = SECRET
        token = SECRET + _clean(first, "@")
        text = f"{_clean(prefix)}https://x-access-token:{token}@github.com/{_clean(suffix)}"
    else:
        secret = SECRET
        user = "".join(c for c in _clean(first, ":/@") if not c.isspace()) or "u"
        # Marked between every character, so a partial leak shows too.
        password = SECRET + "".join(
            c + SECRET for c in _clean(second, "/") if not c.isspace()
        )
        scheme = SCHEMES[(mode >> 2 & 3) % len(SCHEMES)]
        text = f"{_clean(prefix)}{scheme}{user}:{password}@github.com/{_clean(suffix)}"
    shown = redact(text)
    shown.encode("utf-8")
    if secret in shown:
        raise violation("redact_text showed a planted secret", text, shown)


def _check_summary(message: str, summarise: Callable[[str], str]) -> None:
    line = summarise(message)
    line.encode("utf-8")
    if "\n" in line or line != line.strip():
        raise violation("summarise_error gave more than one clean line", message, line)
    if summarise(line) != line:
        raise violation("summarise_error is not idempotent", message, line)
    message = _clean(message)
    if TRAILER in message:
        return
    with_trailer = f"{message}\n{TRAILER} https://{SECRET}x"
    line = summarise(with_trailer)
    if SECRET in line or line.split() != message.split():
        raise violation(
            "summarise_error mishandled the httpx trailer", with_trailer, line
        )


def check(
    data: bytes,
    *,
    redact: Callable[[str], str] = redact_text,
    summarise: Callable[[str], str] = summarise_error,
) -> None:
    """Run the contract the first byte selects on the rest of *data*."""
    if not data:
        return
    mode, rest = data[0], data[1:]
    if mode & 0x80:
        _check_summary(fields(rest, 1)[0], summarise)
    else:
        _check_redact(mode, fields(rest, 4), redact)


def _seed(mode: int, *parts: str) -> bytes:
    return bytes([mode]) + "\x00".join(parts).encode("utf-8")


SEEDS: tuple[bytes, ...] = (
    _seed(0x00, "token ", " in a log", "abcdefghijklmnopqrstuvwxyz0123456789"),
    _seed(0x0C, "git push https://", "/x", "Z9"),
    # 0x0C above plants an OAuth token; these a refresh and a
    # fine-grained token.
    _seed(0x10, "refresh ", "", "x"),
    _seed(0x14, "fine-grained ", "", "x"),
    _seed(0x01, "fatal: ", "acme/widget.git", "secret-token"),
    _seed(0x02, "remote: ", "acme/widget.git", "user", "pass"),
    # The fix in this change: the password holds an "@".
    _seed(0x06, "", "x", "user", "p@ss@word"),
    _seed(0x02, "https://a:b@c ", "", "u", "p@q"),
    _seed(
        0x80,
        "Client error '404 Not Found'\nFor more information check: https://httpstatuses.com/404",
    ),
    _seed(0x80, "line one\n  line two\r\n\n"),
    _seed(0x80, " For more information check: the run log"),
    # libFuzzer sizes inputs from the largest seed.
    _seed(0x80, "word " * 400 + "\n" * 50),
)


def main() -> None:
    """Fuzz :func:`check` from :data:`SEEDS`."""
    import atheris

    run(atheris, check, SEEDS)


if __name__ == "__main__":
    main()
