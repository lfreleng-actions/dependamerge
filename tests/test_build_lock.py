# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""The composite action builds with the backend that uv.lock pins.

uv does not lock ``[build-system] requires``, so the action installs the
``build`` dependency group from the lock and builds without isolation.
That only holds while the group names exactly what the build system
requires: a requirement missing from the group would be absent at build
time, and one only in the group would never be used.
"""

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_the_build_group_mirrors_the_build_system() -> None:
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    requires = project["build-system"]["requires"]
    group = project["dependency-groups"]["build"]
    assert sorted(group) == sorted(requires)
