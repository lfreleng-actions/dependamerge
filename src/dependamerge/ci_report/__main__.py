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
``slack --channel ID [--run-url URL]``
    A Slack ``chat.postMessage`` payload, as one line of JSON.

The result goes to standard output.  Kept out of the ``dependamerge``
CLI because it serves the action rather than an operator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .document import TERMINAL_STATUSES, load_document
from .markdown import render_markdown
from .slack import render_payload


def render_outputs(document: Mapping[str, Any]) -> str:
    """One ``key=value`` line per outcome count, then ``total``."""
    counts = document.get("counts") or {}
    lines = [f"{status}={int(counts.get(status, 0))}" for status in TERMINAL_STATUSES]
    lines.append(f"total={len(document.get('results') or [])}")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dependamerge.ci_report",
        description="Render a dependamerge results file.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("markdown", "outputs"):
        commands.add_parser(name).add_argument("results_file", type=Path)
    slack = commands.add_parser("slack")
    slack.add_argument("results_file", type=Path)
    slack.add_argument("--channel", required=True, help="Slack channel ID")
    slack.add_argument("--run-url", default="", help="link to the workflow run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render the named results file; return the process exit status."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        document = load_document(args.results_file)
    except (OSError, ValueError) as exc:
        print(f"dependamerge.ci_report: {exc}", file=sys.stderr)
        return 1
    if args.command == "markdown":
        sys.stdout.write(render_markdown(document))
    elif args.command == "outputs":
        sys.stdout.write(render_outputs(document))
    else:
        payload = render_payload(document, channel=args.channel, run_url=args.run_url)
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
