# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Render a results file for the composite action.

Run as ``python -m dependamerge.ci_report markdown RESULTS_FILE``; the
report goes to standard output.  Kept out of the ``dependamerge`` CLI
because it serves the action rather than an operator.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from .document import load_document
from .markdown import render_markdown

_USAGE = "usage: python -m dependamerge.ci_report markdown RESULTS_FILE"


def main(argv: Sequence[str] | None = None) -> int:
    """Render the named results file; return the process exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "markdown":
        print(_USAGE, file=sys.stderr)
        return 2
    try:
        document = load_document(Path(args[1]))
    except (OSError, ValueError) as exc:
        print(f"dependamerge.ci_report: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(render_markdown(document))
    return 0


if __name__ == "__main__":
    sys.exit(main())
