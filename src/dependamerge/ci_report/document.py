# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
The results document: what a merge run did, in a stable JSON shape.

The document is written only under GitHub Actions, where the runner
sets ``GITHUB_ACTIONS=true``, so nothing changes for anyone running the
tool by hand and no flag is needed to opt in.  The file goes into
``RUNNER_TEMP`` (removed by the runner when the job ends) through
:func:`tempfile.mkstemp`, which creates it readable by its owner alone
under a name nothing else can predict.  Its path is published as the
``results_file`` step output, so the step that renders it never has to
guess where it is.

``schema_version`` changes only when a consumer would misread the new
shape; adding a field does not change it.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._version import __version__
from ..merge_manager import MergeResult

SCHEMA_VERSION = 1

#: The step output naming the results file.
OUTPUT_KEY = "results_file"

#: Terminal outcomes, always present in ``counts`` so a consumer can read
#: any of them without checking for the key.
TERMINAL_STATUSES = (
    "merged",
    "auto_merge_pending",
    "failed",
    "blocked",
    "unsettled",
    "skipped",
    "closed",
)


def in_github_actions(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this process is a GitHub Actions step."""
    env = os.environ if environ is None else environ
    return env.get("GITHUB_ACTIONS") == "true"


def _entry(result: MergeResult, explain: Callable[[str], list[str]]) -> dict[str, Any]:
    """One result, flattened to JSON-ready values."""
    pr = result.pr_info
    status = result.status.value
    # A non-merged outcome always carries a reason, falling back to the
    # console summary's own wording when the error was empty.
    error = result.error or (None if status == "merged" else "no reason reported")
    lines = explain(error) if error else []
    return {
        "repository": pr.repository_full_name,
        "number": pr.number,
        "title": pr.title,
        "url": pr.html_url,
        "author": pr.author,
        "status": status,
        "reason": lines[0] if lines else None,
        # The explanation's bullets, without the bullet: each renderer
        # chooses its own.
        "details": [line.removeprefix("\u2022 ") for line in lines[1:]],
        "warning": result.warning,
    }


def build_document(
    results: Sequence[MergeResult],
    *,
    explain: Callable[[str], list[str]],
    target: str,
    scope: str,
    preview: bool,
    dry_run: bool,
    selection: str | None = None,
    scan_errors: Iterable[str] = (),
) -> dict[str, Any]:
    """Describe one merge run.

    Args:
        results: One result per PR the run evaluated or merged.
        explain: Expands a raw failure reason into a heading line plus
            one bullet per failing condition (the console summary's own
            formatter, so both say the same thing).
        target: The target as the operator gave it.
        scope: ``owner``, ``repository`` or ``pull_request``.
        preview: The outcomes are predictions; nothing was merged.
        dry_run: The run was a ``--dry-run``.
        selection: How ``--include-repos``/``--exclude-repos`` limited
            the run, or ``None``.
        scan_errors: Repositories an owner-wide run could not scan.
    """
    counts: dict[str, int] = dict.fromkeys(TERMINAL_STATUSES, 0)
    for result in results:
        status: str = result.status.value
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "dependamerge", "version": __version__},
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": target,
        "scope": scope,
        "preview": preview,
        "dry_run": dry_run,
        "selection": selection,
        "counts": counts,
        "results": [_entry(result, explain) for result in results],
        "scan_errors": list(scan_errors),
    }


def write_results_file(
    document: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Write ``document`` to a private file and publish its path.

    Raises:
        OSError: When the file or the step output cannot be written.
    """
    env = os.environ if environ is None else environ
    fd, name = tempfile.mkstemp(
        prefix="dependamerge-results-",
        suffix=".json",
        dir=env.get("RUNNER_TEMP") or None,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    output = env.get("GITHUB_OUTPUT")
    if output:
        if "\n" in name or "\r" in name:
            # A line break would end the value early and let the rest of
            # the path inject a second output.
            raise OSError(f"refusing to publish an unsafe path: {name!r}")
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"{OUTPUT_KEY}={name}\n")
    return Path(name)


def load_document(path: Path) -> dict[str, Any]:
    """Read a results file, refusing a schema this version cannot read.

    Raises:
        OSError: When the file cannot be read.
        ValueError: When it is not a results document this code reads.
    """
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: not a results document")
    version = document.get("schema_version")
    # bool and float compare equal to 1, but are not the integer version.
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {version!r} is not {SCHEMA_VERSION}")
    # The renderers read these as containers; a wrong shape here must be
    # reported as a bad file, not crash a renderer with a traceback.
    expected = {
        "tool": dict,
        "counts": dict,
        "results": list,
        "scan_errors": list,
    }
    for key, kind in expected.items():
        if key in document and not isinstance(document[key], kind):
            raise ValueError(f"{path}: '{key}' must be a {kind.__name__}")
    if not all(isinstance(entry, dict) for entry in document.get("results") or []):
        raise ValueError(f"{path}: every entry in 'results' must be an object")
    if not all(
        isinstance(entry.get("details", []), list)
        for entry in document.get("results") or []
    ):
        raise ValueError(f"{path}: every entry's 'details' must be a list")
    counts = document.get("counts") or {}
    if not all(isinstance(n, int) and not isinstance(n, bool) for n in counts.values()):
        raise ValueError(f"{path}: every count must be an integer")
    return document
