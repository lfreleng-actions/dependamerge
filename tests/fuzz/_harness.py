# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""What the fuzz harnesses share: decoding, seeds and the Atheris entry point.

Each harness exposes ``check(data)``, which runs the code under test on
one input and raises :class:`ContractViolation` when the result breaks
a documented contract.  ``check`` decodes the bytes itself and needs
nothing beyond the standard library, so ``tests/test_fuzz_replay.py``
can call it on every seed in the ordinary suite.  Atheris is needed only
to *search* for new inputs: each harness imports it in its own ``main``
and hands the module to :func:`run`, so this module never names it.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from types import ModuleType

from dependamerge.url_parser import set_gerrit_host, set_github_host

#: Every variable the URL parsers read to decide which host is which.
HOST_VARIABLES = (
    "DEPENDAMERGE_GITHUB_HOST",
    "DEPENDAMERGE_GITHUB_HOSTS",
    "GH_HOST",
    "DEPENDAMERGE_GERRIT_HOST",
    "DEPENDAMERGE_GERRIT_HOSTS",
)


class ContractViolation(AssertionError):
    """The code under test broke a contract its documentation states."""


def violation(message: str, *values: object) -> ContractViolation:
    """Build a :class:`ContractViolation` naming the values involved."""
    shown = ", ".join(repr(value) for value in values)
    return ContractViolation(f"{message}: {shown}" if shown else message)


def fields(data: bytes, count: int) -> list[str]:
    """Split *data* into exactly *count* text fields on NUL bytes.

    Decoding replaces invalid UTF-8 rather than rejecting it, so every
    input reaches the code under test.  A missing field is empty and any
    surplus stays in the last one.
    """
    parts = data.decode("utf-8", errors="replace").split("\x00", count - 1)
    return parts + [""] * (count - len(parts))


@contextmanager
def host_environment(variables: dict[str, str]) -> Iterator[None]:
    """Run with exactly *variables* as the host declarations, then restore.

    The process-wide overrides are cleared on the way in and out, so one
    input cannot change how the next is parsed.
    """
    saved = {name: os.environ.get(name) for name in HOST_VARIABLES}
    for name in HOST_VARIABLES:
        os.environ.pop(name, None)
    os.environ.update(variables)
    set_github_host(None)
    set_gerrit_host(None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        set_github_host(None)
        set_gerrit_host(None)


def _write_seeds(seeds: Iterable[bytes]) -> str:
    """Write *seeds* to a fresh directory for libFuzzer to start from."""
    directory = tempfile.mkdtemp(prefix="dependamerge-fuzz-seeds-")
    for index, seed in enumerate(seeds):
        with open(os.path.join(directory, f"seed-{index:03d}"), "wb") as handle:
            handle.write(seed)
    return directory


def run(
    atheris: ModuleType, check: Callable[[bytes], None], seeds: Iterable[bytes]
) -> None:
    """Fuzz *check* under the *atheris* module, starting from *seeds*.

    libFuzzer options pass through unchanged.  Unless an argument names
    an existing file, the seeds are written out and their directory goes
    after any directories given, so every fuzzing run starts from the
    known-hard cases.  libFuzzer keeps new inputs in the first directory.
    """
    argv = list(sys.argv)
    if not any(os.path.isfile(arg) for arg in argv[1:]):
        argv.append(_write_seeds(seeds))
    atheris.instrument_all()
    atheris.Setup(argv, check)
    atheris.Fuzz()
