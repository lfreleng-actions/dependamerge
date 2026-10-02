# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Machine-readable merge results, and the reports built from them, for CI.

Under GitHub Actions a merge run writes its results as JSON to a private
temporary file in ``RUNNER_TEMP`` and names that file in the step's
``results_file`` output (see :mod:`.document`).  The renderers here turn
the file into reports from a later step; the runner removes
``RUNNER_TEMP`` when the job ends.
"""

from .document import (
    OUTPUT_KEY,
    SCHEMA_VERSION,
    build_document,
    in_github_actions,
    load_document,
    write_results_file,
)
from .markdown import render_markdown

__all__ = [
    "OUTPUT_KEY",
    "SCHEMA_VERSION",
    "build_document",
    "in_github_actions",
    "load_document",
    "render_markdown",
    "write_results_file",
]
