# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Which check runs a later run of the same check supersedes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def parse_ts(value: Any) -> datetime | None:
    """Parse a GitHub timestamp, returning None when it is unusable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        # GitHub returns RFC 3339 with a trailing ``Z``.
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # A timestamp without tz info parses to a naive datetime, which
    # raises ``TypeError`` when subtracted from the tz-aware ``now``.
    # Treat it as unparsable (fail closed) so the detector degrades
    # gracefully instead of aborting the merge run.
    if ts.tzinfo is None:
        return None
    return ts


def run_source(run: dict[str, Any]) -> tuple[Any, str]:
    """The GitHub App and name that together identify a check."""
    app = run.get("app")
    app_id = app.get("id") if isinstance(app, dict) else None
    return app_id, str(run.get("name") or "")


def _int_id(run: dict[str, Any]) -> int | None:
    """A run's ``id`` when it is a usable integer."""
    value = run.get("id")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _started_after(other: dict[str, Any], run: dict[str, Any]) -> bool:
    """Whether *other* provably began after *run*.

    Two keys order runs: ``started_at``, and ``id``, which GitHub assigns
    in creation order.  *other* is later when at least one key the pair
    shares says so and none says the opposite: an older run can sit
    queued and start after a newer one, so start time and id that
    disagree leave the pair unordered.  Completion time is deliberately
    not used: an older run can finish after a newer one starts.  A pair
    where only one run has a start time is unordered too: comparing ids
    alone could contradict a third run's start times and make the order
    cyclic.  When nothing orders the pair the answer is no, so an
    unordered pending run stays judged.
    """
    other_ts = parse_ts(other.get("started_at"))
    run_ts = parse_ts(run.get("started_at"))
    if (other_ts is None) != (run_ts is None):
        return False
    later = earlier = False
    if other_ts is not None and run_ts is not None:
        later, earlier = other_ts > run_ts, other_ts < run_ts
    other_id, run_id = _int_id(other), _int_id(run)
    if other_id is not None and run_id is not None:
        later = later or other_id > run_id
        earlier = earlier or other_id < run_id
    return later and not earlier


def _check_suite_id(run: dict[str, Any]) -> int | None:
    """The id of the check suite *run* belongs to, when it is known."""
    suite = run.get("check_suite")
    return _int_id(suite) if isinstance(suite, dict) else None


def current_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs that no later run of the same App, suite and name supersedes.

    A run from another App never supersedes one: it cannot satisfy a
    requirement bound to the first App.  Nor does a run from another
    check suite: a manually dispatched workflow, for one, cannot satisfy
    the pull request's own run.  A run whose suite is unknown is kept.
    """
    current: list[dict[str, Any]] = []
    by_source: dict[tuple[Any, int, str], list[dict[str, Any]]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        suite = _check_suite_id(run)
        if suite is None:
            current.append(run)
            continue
        app, name = run_source(run)
        by_source.setdefault((app, suite, name), []).append(run)
    return current + [
        run
        for group in by_source.values()
        for run in group
        if not any(_started_after(other, run) for other in group)
    ]
