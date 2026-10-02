# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The results document as GitHub-flavoured Markdown, for a step summary.

Outcomes needing a human come first, so a reader sees what to act on
without scrolling past every merge.  Each outcome gets its own table;
an outcome with no PRs gets nothing, not an empty table.

Values reaching a table cell come from pull requests, which anyone able
to open one can title, so text is escaped to stay text: every ASCII
punctuation mark is backslash-escaped, which keeps ``[x](url)`` and
``![x](url)`` from becoming a link or an image and ``|`` from ending
the cell, and ``&``, ``<`` and ``>`` become entities so no markup
opens.  A link's destination is handled apart from text: only a plain
``http(s)`` URL becomes one, and anything else is shown unlinked.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

#: Outcome order and labels: (status, real-run heading, preview heading,
#: whether the table carries a reason column).
_SECTIONS: tuple[tuple[str, str, str, bool], ...] = (
    ("failed", "\u274c Failed", "\u274c Would fail", True),
    ("blocked", "\U0001f6d1 Blocked", "\U0001f6d1 Blocked", True),
    ("unsettled", "\u23f1\ufe0f Unsettled", "\u23f1\ufe0f Unsettled", True),
    (
        "auto_merge_pending",
        "\U0001f916 Auto-merge pending",
        "\U0001f916 Auto-merge pending",
        True,
    ),
    ("skipped", "\u23ed\ufe0f Skipped", "\u23ed\ufe0f Skipped", True),
    ("closed", "\U0001f6aa Closed", "\U0001f6aa Closed", True),
    ("merged", "\u2705 Merged", "\u2705 Mergeable", False),
)

_SCOPE_LABELS = {
    "owner": "owner-wide",
    "repository": "repository",
    "pull_request": "pull request and similar PRs",
}


_ENTITIES = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
_PUNCTUATION = frozenset("!\"#$%'()*+,-./:;=?@[\\]^_`{|}~")

#: A link destination safe to write as-is: an absolute http(s) URL with
#: nothing that could end the destination or the table cell.
_SAFE_URL = re.compile(r"\Ahttps?://[^\s<>()|\\`]+\Z")


def _cell(value: object) -> str:
    """Escape ``value`` as literal text for one Markdown table cell."""
    text = " ".join(("" if value is None else str(value)).split())
    return "".join(
        _ENTITIES.get(char) or (f"\\{char}" if char in _PUNCTUATION else char)
        for char in text
    )


def _pr_link(entry: Mapping[str, Any]) -> str:
    """``[#7](url)``, or plain ``#7`` when the URL is not a safe one."""
    number = entry.get("number")
    label = f"#{number}" if isinstance(number, int) else _cell(number)
    url = str(entry.get("url") or "")
    return f"[{label}]({url})" if _SAFE_URL.match(url) else label


def _reason(entry: Mapping[str, Any]) -> str:
    """The reason column: the heading, then one line per detail."""
    lines = [_cell(entry.get("reason") or entry.get("warning") or "")]
    lines.extend(f"\u2022 {_cell(detail)}" for detail in entry.get("details") or [])
    return "<br>".join(line for line in lines if line)


def _table(entries: Sequence[Mapping[str, Any]], with_reason: bool) -> list[str]:
    header = "| Repository | Pull request | Title |"
    rule = "| --- | --- | --- |"
    if with_reason:
        header += " Reason |"
        rule += " --- |"
    rows = [header, rule]
    for entry in entries:
        row = (
            f"| {_cell(entry.get('repository'))} | {_pr_link(entry)} "
            f"| {_cell(entry.get('title'))} |"
        )
        if with_reason:
            row += f" {_reason(entry)} |"
        rows.append(row)
    return rows


def render_markdown(document: Mapping[str, Any]) -> str:
    """Render a results document as a step-summary section."""
    preview = bool(document.get("preview"))
    scope = _SCOPE_LABELS.get(str(document.get("scope")), str(document.get("scope")))
    mode = "dry run" if document.get("dry_run") else "preview" if preview else "merge"
    tool = document.get("tool") or {}

    lines = ["## \U0001f916 Dependamerge results", ""]
    lines.append(
        f"**Target:** {_cell(document.get('target'))} ({scope}) \u00b7 "
        f"**Mode:** {mode} \u00b7 **Version:** {_cell(tool.get('version'))}"
    )
    if document.get("selection"):
        lines.extend(["", f"**Scope:** {_cell(document['selection'])}"])

    entries = list(document.get("results") or [])
    counts = document.get("counts") or {}
    if not entries:
        lines.extend(["", "No pull requests to process."])
    else:
        lines.extend(["", "| Outcome | Pull requests |", "| --- | ---: |"])
        for status, heading, preview_heading, _ in _SECTIONS:
            if counts.get(status):
                label = preview_heading if preview else heading
                lines.append(f"| {label} | {counts[status]} |")

    for status, heading, preview_heading, with_reason in _SECTIONS:
        matching = [entry for entry in entries if entry.get("status") == status]
        if matching:
            label = preview_heading if preview else heading
            lines.extend(["", f"### {label}", "", *_table(matching, with_reason)])

    scan_errors = document.get("scan_errors") or []
    if scan_errors:
        lines.extend(["", "### \u26a0\ufe0f Repositories not scanned", ""])
        lines.extend(f"- {_cell(error)}" for error in scan_errors)

    return "\n".join(lines) + "\n"
