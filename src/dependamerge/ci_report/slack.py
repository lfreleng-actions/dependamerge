# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The results document as a Slack ``chat.postMessage`` payload.

A digest for people who follow up: outcomes needing a human are listed
first, each with its reason, then the merges; outcomes that resolve on
their own (auto-merge pending, unsettled, closed) appear only as counts.

Slack rejects a whole message when any one limit is exceeded, so each
section is fitted to the 3,000-character text ceiling by shedding PR
lines and saying how many it shed, measured in UTF-16 code units as
Slack counts them.  The message holds at most a dozen blocks, far below
the 50-block ceiling, whatever the run contained.

Text reaching the payload comes from pull requests, which anyone able to
open one can title, so ``&``, ``<`` and ``>`` are escaped everywhere:
unescaped, a title could open a link or mention such as ``<!channel>``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MAX_TEXT_CHARS = 3000
MAX_HEADER_CHARS = 150
#: Slack's ceiling on the top-level ``text`` of a message.
MAX_FALLBACK_CHARS = 40000

#: Outcomes listed PR by PR, with their reasons:
#: (status, heading, heading in a preview or dry run).
_LISTED: tuple[tuple[str, str, str], ...] = (
    ("failed", "\u274c *Failed*", "\u274c *Would fail*"),
    ("blocked", "\U0001f6d1 *Blocked*", "\U0001f6d1 *Blocked*"),
    ("skipped", "\u23ed\ufe0f *Skipped*", "\u23ed\ufe0f *Skipped*"),
    ("merged", "\u2705 *Merged*", "\u2705 *Mergeable*"),
)

#: Every outcome in the count line, in reading order:
#: (status, label, label in a preview or dry run).
_COUNTED: tuple[tuple[str, str, str], ...] = (
    ("merged", "\u2705 {} merged", "\u2705 {} mergeable"),
    (
        "auto_merge_pending",
        "\U0001f916 {} auto-merge pending",
        "\U0001f916 {} auto-merge pending",
    ),
    ("failed", "\u274c {} failed", "\u274c {} would fail"),
    ("blocked", "\U0001f6d1 {} blocked", "\U0001f6d1 {} blocked"),
    ("unsettled", "\u23f1\ufe0f {} unsettled", "\u23f1\ufe0f {} unsettled"),
    ("skipped", "\u23ed\ufe0f {} skipped", "\u23ed\ufe0f {} skipped"),
    ("closed", "\U0001f6aa {} closed", "\U0001f6aa {} closed"),
)


def text_length(text: str) -> int:
    """Length as Slack measures it: UTF-16 code units."""
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def escape(value: object) -> str:
    """Escape ``value`` for Slack mrkdwn, flattened to one line."""
    text = "" if value is None else str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return " ".join(text.split())


def _ellipsize(text: str, budget: int) -> str:
    if text_length(text) <= budget:
        return text
    cut = text[: max(budget - 1, 0)]
    while cut and text_length(cut) > budget - 1:
        cut = cut[:-1]
    return f"{cut}\u2026"


def _pr_line(entry: Mapping[str, Any], *, with_reason: bool) -> str:
    """``• <url|owner/repo#7> Title — reason``."""
    label = f"{escape(entry.get('repository'))}#{escape(entry.get('number'))}"
    url = str(entry.get("url") or "")
    # A URL carrying a delimiter would break out of the link; GitHub's never
    # do, so this only guards against a malformed document.
    link = label if any(c in url for c in "<>| ") or not url else f"<{url}|{label}>"
    line = f"\u2022 {link} {escape(entry.get('title'))}"
    if with_reason:
        reason = entry.get("reason") or entry.get("warning")
        details = [escape(detail) for detail in entry.get("details") or []]
        explained = "; ".join(filter(None, [escape(reason), *details]))
        if explained:
            line += f" \u2014 _{explained}_"
    return line


def fit_lines(heading: str, lines: Sequence[str], budget: int = MAX_TEXT_CHARS) -> str:
    """``heading`` and as many ``lines`` as fit, noting how many were shed.

    Lines are kept in order, so the first ones listed survive.  A line
    longer than the whole budget is cut rather than dropped.

    Each line is measured once, and the kept prefix grows only while it
    still leaves room for the note, so the work is linear in the number
    of lines rather than re-rendering the list once per line shed.
    """

    def note(hidden: int) -> str:
        return f"\n\u2026 and {hidden} more" if hidden else ""

    used = text_length(heading)
    sizes = [1 + text_length(line) for line in lines]  # 1 for the newline
    if used + sum(sizes) <= budget:
        return "\n".join([heading, *lines])
    shown = 0
    for size in sizes:
        hidden_after = len(lines) - shown - 1
        if used + size + text_length(note(hidden_after)) > budget:
            break
        used += size
        shown += 1
    if shown == 0 and lines:
        # Not even the first line fits whole: cut it rather than drop it,
        # so the section still names a PR.  The link opens the line, so
        # it survives; the cut falls in the title or reason.
        tail = note(len(lines) - 1)
        room = budget - used - 1 - text_length(tail)
        return "\n".join([heading, _ellipsize(lines[0], room)]) + tail
    text = "\n".join([heading, *lines[:shown]]) + note(len(lines) - shown)
    return _ellipsize(text, budget)


def _section(text: str) -> dict[str, Any]:
    # verbatim stops Slack auto-linking bare URLs and #channel names in
    # untrusted titles; the explicit <url|label> links still render.
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": text, "verbatim": True},
    }


def _owner(document: Mapping[str, Any]) -> str:
    target = str(document.get("target") or "")
    return (
        target.removeprefix("https://")
        .removeprefix("http://")
        .removeprefix("github.com/")
    )


#: When a digest is worth posting: always; on any activity (a PR was
#: processed); on attention (a PR failed or was blocked); or never.
SLACK_WHEN = ("always", "activity", "attention", "never")


def should_post(document: Mapping[str, Any], when: str) -> bool:
    """Whether ``document`` warrants a digest under the ``when`` policy.

    Raises:
        ValueError: When ``when`` is not one of :data:`SLACK_WHEN`.
    """
    if when not in SLACK_WHEN:
        raise ValueError(f"slack_when must be one of: {', '.join(SLACK_WHEN)}")
    counts = document.get("counts") or {}
    if when == "activity":
        return bool(document.get("results"))
    if when == "attention":
        return bool(counts.get("failed") or counts.get("blocked"))
    return when == "always"


def render_payload(
    document: Mapping[str, Any],
    *,
    channel: str,
    run_url: str = "",
) -> dict[str, Any]:
    """Build the ``chat.postMessage`` payload for a results document.

    Args:
        document: A loaded results document.
        channel: Slack channel ID to post to.
        run_url: Link to the workflow run, shown beneath the digest.
    """
    counts = document.get("counts") or {}
    entries = list(document.get("results") or [])
    preview = bool(document.get("preview"))
    mode = " (dry run)" if document.get("dry_run") else " (preview)" if preview else ""
    target = escape(_owner(document))

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": _ellipsize(
                    f"\U0001f916 Dependamerge{mode}: {_owner(document)}",
                    MAX_HEADER_CHARS,
                ),
            },
        }
    ]

    tally = [
        (preview_label if preview else label).format(counts[status])
        for status, label, preview_label in _COUNTED
        if counts.get(status)
    ]
    summary = " \u00b7 ".join(tally) if tally else "No pull requests to process."
    if document.get("selection"):
        summary += f"\n_{escape(document['selection'])}_"
    summary = _ellipsize(summary, MAX_TEXT_CHARS)
    blocks.append(_section(summary))

    # The per-outcome listings, kept apart for the fallback text below.
    listings: list[str] = []
    for status, heading, preview_heading in _LISTED:
        matching = [entry for entry in entries if entry.get("status") == status]
        if matching:
            lines = [
                _pr_line(entry, with_reason=status != "merged") for entry in matching
            ]
            title = preview_heading if preview else heading
            listings.append(fit_lines(title, lines))

    scan_errors = [escape(error) for error in document.get("scan_errors") or []]
    if scan_errors:
        heading = "\u26a0\ufe0f *Repositories not scanned*"
        listings.append(fit_lines(heading, [f"\u2022 {e}" for e in scan_errors]))
    blocks.extend(_section(listing) for listing in listings)

    tool = document.get("tool") or {}
    footer = f"dependamerge {escape(tool.get('version'))}"
    if run_url and not any(c in run_url for c in "<>| "):
        footer = f"<{run_url}|View workflow run> \u00b7 {footer}"
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": footer}]})

    # Screen readers and notifications read the top-level text, not the
    # blocks, so it carries the whole digest: the headline, the counts and
    # any scope selection, every listing as fitted (and escaped) for its
    # block, and the footer with the run link.
    headline = f"\U0001f916 Dependamerge{mode}: {target}"
    fallback = "\n\n".join([headline, summary, *listings, footer])
    return {
        "channel": channel,
        "text": _ellipsize(fallback, MAX_FALLBACK_CHARS),
        # The fallback is parsed apart from the blocks, so verbatim does
        # not reach it: parse "none" stops Slack auto-linking a bare URL
        # or #channel from a title there.  unfurl_* only stop previews.
        "parse": "none",
        "unfurl_links": False,
        "unfurl_media": False,
        "blocks": blocks,
    }
