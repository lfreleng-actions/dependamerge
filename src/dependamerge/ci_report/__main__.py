# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Render a results file for the composite action.

Run as ``python -m dependamerge.ci_report COMMAND RESULTS_FILE``, where
``COMMAND`` is one of:

``markdown``
    The step-summary report.
``outputs``
    One ``key=value`` line per outcome count plus ``total``, ready to
    append to ``GITHUB_OUTPUT``.

The result goes to standard output.  Kept out of the ``dependamerge``
CLI because it serves the action rather than an operator.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .document import TERMINAL_STATUSES, load_document
from .markdown import render_markdown

_USAGE = "usage: python -m dependamerge.ci_report {markdown|outputs} RESULTS_FILE"


def render_outputs(document: Mapping[str, Any]) -> str:
    """One ``key=value`` line per outcome count, then ``total``."""
    counts = document.get("counts") or {}
    lines = [f"{status}={int(counts.get(status, 0))}" for status in TERMINAL_STATUSES]
    lines.append(f"total={len(document.get('results') or [])}")
    return "\n".join(lines) + "\n"


_RENDERERS = {"markdown": render_markdown, "outputs": render_outputs}


def main(argv: Sequence[str] | None = None) -> int:
    """Render the named results file; return the process exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in _RENDERERS:
        print(_USAGE, file=sys.stderr)
        return 2
    try:
        document = load_document(Path(args[1]))
    except (OSError, ValueError) as exc:
        print(f"dependamerge.ci_report: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(_RENDERERS[args[0]](document))
    return 0


if __name__ == "__main__":
    sys.exit(main())
