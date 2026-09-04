# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Parsing GitHub's repository rule-violation messages.

When a merge is refused by a ruleset, GitHub returns a single prose
string naming the offending conditions::

    Repository rule violations found
    Required workflows 'AI Slop Scan 🧹, Zizmor Scan 🌈' are not satisfied

Two very different consumers need the names out of that string: the CLI,
to render one bullet per condition, and the merge pipeline, to decide
whether waiting can possibly help.  Parsing it in both places invites
them to drift apart, so the extraction lives here --- pure, shared and
directly testable.

The message is also the *more dependable* source for required-workflow
names than enumerating org rulesets: during the 503-PR run analysed in
``docs/BULK_RUN_PERFORMANCE_AUDIT.md``, ``GET /orgs/{org}/rulesets``
returned 403 while this string was always present on the rejection.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

__all__ = [
    "RULE_VIOLATION_MARKER",
    "is_rule_violation",
    "name_spans",
    "required_workflow_names",
    "required_status_check_names",
    "status_check_violation_verb",
    "violation_verb",
    "workflow_name_fragments",
    "workflow_name_spans",
]

RULE_VIOLATION_MARKER = "Repository rule violations found"

_WORKFLOW_MARKER = "Required workflows "
_STATUS_CHECK_MARKER = "Required status check"

# GitHub states the outcome immediately after the closing quote, as in
# ``' are not satisfied`` or ``' failed``.  Anchoring on that wording is
# what lets the closing delimiter be found without assuming the names
# contain no apostrophe of their own.
_OUTCOME_RE = re.compile(
    r"\s*(?:are|is|was|were|have|has)?\s*(?:not satisfied|failed|fail)",
    re.IGNORECASE,
)


def _workflow_quote_bounds(reason: str) -> tuple[int, int] | None:
    """Absolute ``(open, close)`` quote positions of the workflow list.

    Shared so that every reading of the message agrees on where the
    workflow names *are* --- which is what lets the other clause markers
    be located outside them.  A rule name is arbitrary text and may
    contain the words another clause is recognised by, so searching the
    raw string for a marker finds the name as readily as the clause.

    The workflow marker is itself located outside the double-quoted
    spans, which hold status-check names.  A context called ``Run
    Required workflows Don't Fail`` otherwise makes its own apostrophe
    look like the opening quote of a workflow list, and the parser
    invents a workflow out of the remainder of the sentence.
    """
    if not reason:
        return None
    marker_at = _first_marker_outside(
        reason, _WORKFLOW_MARKER, _double_quoted_spans(reason)
    )
    if marker_at == -1:
        return None
    open_at = reason.find("'", marker_at + len(_WORKFLOW_MARKER))
    if open_at == -1:
        return None
    body = reason[open_at + 1 :]
    # Apostrophes inside a later double-quoted status-check name are not
    # candidates.  ``Required workflows 'Build' are not satisfied
    # Required status check "CI 'failed'" is failing`` otherwise closes
    # the workflow list on the apostrophe inside ``"CI 'failed'"``,
    # swallowing the whole status clause into a single workflow name.
    quoted = _double_quoted_spans(body)
    closing = -1
    for index, char in enumerate(body):
        if char != "'" or any(s <= index <= e for s, e in quoted):
            continue
        if _OUTCOME_RE.match(body, index + 1):
            closing = index
    if closing == -1:
        for index, char in enumerate(body):
            if char == "'" and not any(s <= index <= e for s, e in quoted):
                closing = index
                break
    if closing == -1:
        return open_at, len(reason)
    return open_at, open_at + 1 + closing


def _double_quoted_spans(text: str) -> list[tuple[int, int]]:
    """Ranges covered by ``"…"`` pairs, which hold status-check names."""
    return [(m.start(), m.end() - 1) for m in re.finditer(r'"[^"]*"', text)]


def _first_marker_outside(text: str, marker: str, spans: list[tuple[int, int]]) -> int:
    """Index of the first *marker* not lying inside any of *spans*."""
    at = text.find(marker)
    while at != -1:
        if not any(start <= at <= end for start, end in spans):
            return at
        at = text.find(marker, at + 1)
    return -1


def _split_workflow_names(reason: str) -> tuple[str, str] | None:
    """Split ``Required workflows 'A, B' are not satisfied`` in two.

    Returns ``(names, outcome)`` --- the raw comma-separated list and the
    clause saying what became of it --- or ``None`` when *reason* is not
    that shape.

    A workflow name is an arbitrary Actions ``name:`` value and GitHub
    does not escape the quote it wraps the list in, so a workflow called
    ``Don't Fail`` puts an apostrophe *inside* it.  Treating the first
    apostrophe as the closing delimiter would then yield the name ``Don``
    and the outcome ``t Fail' are not satisfied`` --- read as "failed",
    which would skip the recovery path for a workflow that had merely
    not started yet.

    The closing delimiter is therefore the *last* apostrophe an
    *outcome* follows.  Last rather than first because a name may
    contain a quoted phrase of its own: in ``'CI 'Fail Fast'' are not
    satisfied`` the inner quote is followed by ``Fail Fast``, which the
    outcome pattern matches, so taking the first would cut the name
    short at ``CI`` and read the rest as a failure.  Trailing prose is
    safe from the later choice: an apostrophe in ``GitHub's ruleset``
    has no outcome after it and never qualifies.  When no candidate
    qualifies the first apostrophe is used, preserving the previous
    behaviour for message shapes not seen here.
    """
    bounds = _workflow_quote_bounds(reason)
    if bounds is None:
        return None
    open_at, close_at = bounds
    names = reason[open_at + 1 : close_at]
    outcome = reason[close_at + 1 :] if close_at < len(reason) else ""
    return names, outcome


def is_rule_violation(reason: str) -> bool:
    """Whether *reason* is a ruleset rejection."""
    return RULE_VIOLATION_MARKER in (reason or "")


def violation_verb(reason: str) -> str:
    """``"failed"`` when a condition ran and failed, else ``"not satisfied"``.

    The distinction matters: *failed* means a workflow ran and reported
    failure, which retrying cannot fix, whereas *not satisfied* can mean
    it is still running --- or has never started at all.

    Only the text **after** the quoted condition names is inspected.
    The enclosing exception always begins ``Failed to merge PR …``, so
    scanning the whole string would classify every rejection as
    ``failed``; a condition *name* containing "fail" would do the same.
    """
    clause = _verb_clause(reason)
    return "failed" if "fail" in clause.lower() else "not satisfied"


def _verb_clause(reason: str) -> str:
    """The portion of *reason* that states the outcome.

    For a workflow violation that is whatever follows the closing quote
    of the name list (``' are not satisfied``), located by
    :func:`_split_workflow_names` so an apostrophe inside a name cannot
    be mistaken for it.  For a status-check violation the names are
    individually quoted, so the text after the final quote is used.
    Falls back to the whole string when neither shape is recognised.

    One rejection can name *both* kinds, in which case the workflow
    clause runs on into the status-check one.  The tail is therefore cut
    at the status-check marker, so a status context that has already
    failed cannot report the workflows as failed when they are merely
    unfinished --- the exact pair that arises while required workflows
    are still queued.
    """
    if not reason:
        return ""
    workflow = _split_workflow_names(reason)
    if workflow is not None:
        tail = workflow[1]
        cut = _first_marker_outside(
            tail, _STATUS_CHECK_MARKER, _double_quoted_spans(tail)
        )
        return tail if cut == -1 else tail[:cut]
    clause = _status_check_clause(reason)
    if clause:
        # ``Required status check "X" is failing.`` --- take everything
        # after the last quoted name.
        idx = clause.rfind('"')
        return clause[idx + 1 :] if idx != -1 else clause
    return reason


def workflow_name_fragments(reason: str) -> list[str]:
    """The comma-separated pieces of the quoted list, in order.

    Unlike :func:`required_workflow_names` this keeps duplicates.  A
    workflow name may itself contain a comma, so the pieces are only a
    *guess* at the names --- and reconciling that guess against the runs
    that actually dispatched needs the sequence exactly as GitHub wrote
    it.  Collapsing ``'Build, Build'`` to a single piece makes that name
    impossible to rejoin, and a name that cannot be rejoined reads as
    never dispatched, which stops the wait on a workflow that ran.
    """
    workflow = _split_workflow_names(reason)
    if workflow is None:
        return []
    return [name.strip() for name in workflow[0].split(",") if name.strip()]


def required_workflow_names(reason: str) -> list[str]:
    """Workflow names quoted in a ``Required workflows '…'`` clause.

    GitHub can repeat a name within one violation string, so duplicates
    are collapsed while first-seen order is preserved --- callers render
    these as a bullet list.  Callers that need to *reconcile* the pieces
    against observed runs want :func:`workflow_name_fragments` instead.
    """
    return list(dict.fromkeys(workflow_name_fragments(reason)))


def _status_check_clause(reason: str) -> str:
    """The span of *reason* that belongs to the status-check rule.

    Status-check names are quoted individually, so anything reading them
    from the whole string also reads any double quote a *workflow* name
    happens to contain.  ``Required status check "DCO" is expected.
    Required workflows 'Build "Fast"' failed`` then yields a phantom
    context called ``Fast``, and hands DCO the workflows' ``failed``.

    Both markers are located **outside the quoted name spans**, because
    a rule name is arbitrary text.  A workflow called ``Build Required
    status check "DCO"`` would otherwise invent a status context, and a
    context called ``Run Required workflows fast`` would truncate its own
    clause and return no names at all.  Either rule may come first, so
    the cut is made relative to the marker rather than at a fixed
    position.
    """
    if not reason:
        return ""
    workflow = _workflow_quote_bounds(reason)
    start = _first_marker_outside(
        reason, _STATUS_CHECK_MARKER, [workflow] if workflow else []
    )
    if start == -1:
        return ""
    clause = reason[start:]
    end = _first_marker_outside(clause, _WORKFLOW_MARKER, _double_quoted_spans(clause))
    return clause if end == -1 else clause[:end]


def required_status_check_names(reason: str) -> list[str]:
    """Context names quoted in a ``Required status check \"…\"`` clause."""
    clause = _status_check_clause(reason)
    if not clause:
        return []
    names = [c.strip() for c in re.findall(r'"([^"]+)"', clause) if c.strip()]
    return list(dict.fromkeys(names))


def status_check_violation_verb(reason: str) -> str:
    """The outcome stated for the ``Required status check`` clause.

    :func:`violation_verb` answers for the rejection as a whole and
    resolves a *workflow* clause in preference, so on a rejection naming
    both kinds it reports the workflows' outcome.  Applying that to the
    status checks is wrong whenever the two differ, which is the usual
    case: workflows that have not finished sit alongside a status
    context that has already failed.

    Read from the status-check clause alone, for the reason
    :func:`_status_check_clause` gives.
    """
    clause = _status_check_clause(reason)
    if not clause:
        return "not satisfied"
    closing = clause.rfind('"')
    tail = clause[closing + 1 :] if closing != -1 else clause
    return "failed" if "fail" in tail.lower() else "not satisfied"


def name_spans(fragments: Sequence[str]) -> Iterator[tuple[int, int, str]]:
    """Every ``(start, width, name)`` *fragments* could be stating.

    GitHub joins the required names with a comma and a workflow name may
    itself contain one, so ``'Build, Test, Lint'`` is three fragments
    that might be three workflows, or two (``Build, Test`` and ``Lint``),
    or one.  Nothing in the message settles it, so every contiguous span
    of fragments is offered, rejoined with both spacings --- only the
    separator GitHub emits is known for certain.

    Callers that need to know *which* fragments a match accounts for use
    ``start`` and ``width``; callers only asking whether a given name
    appears can ignore them.  Stated here once so the two readings
    cannot drift apart.
    """
    total = len(fragments)
    for width in range(1, total + 1):
        for start in range(total - width + 1):
            span = fragments[start : start + width]
            for separator in (", ", ","):
                yield start, width, separator.join(span)


def workflow_name_spans(reason: str) -> Iterator[tuple[int, int, str]]:
    """:func:`name_spans` over the names quoted in *reason*."""
    return name_spans(workflow_name_fragments(reason))
