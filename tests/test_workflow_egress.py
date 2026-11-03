# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Egress allow-list wiring in the reusable dependamerge workflow.

The merge job hands every allow-list input to the block action, so a
caller can extend the organisation baseline. The failure notice keeps a
fixed pinned list: a caller's list that broke the merge job's hardening
would otherwise silence the notice of that very failure.
"""

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

_WORKFLOW = (
    Path(__file__).resolve().parent.parent
    / ".github"
    / "workflows"
    / "dependamerge.yaml"
)
_BLOCK_ACTION = "lfreleng-actions/harden-runner-block-action@"
_PINNED_CONFIG = (
    "lfreleng-actions//.github/harden-runner/lfreleng-actions/"
    "allow_list.txt@6d4a65c5601afd6c2fbe52bda4b0024b132f265d"
)


@pytest.fixture(scope="module")
def jobs() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    workflow_jobs: dict[str, Any] = loaded["jobs"]
    return workflow_jobs


def _block_action_with(job: dict[str, Any]) -> dict[str, Any]:
    matches = [
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith(_BLOCK_ACTION)
    ]
    assert len(matches) == 1
    return matches[0].get("with") or {}


def test_merge_job_passes_every_allow_list_input(jobs) -> None:
    assert _block_action_with(jobs["merge"]) == {
        "config": "${{ inputs.harden_runner_allowlist }}",
        "supplemental_config": "${{ inputs.harden_runner_supplemental_allowlist }}",
        "supplemental_required": (
            "${{ inputs.harden_runner_supplemental_allowlist != '' "
            "&& 'true' || 'false' }}"
        ),
        "supplemental_unpinned": (
            "${{ inputs.harden_runner_supplemental_unpinned && 'true' || 'false' }}"
        ),
        "extra_endpoints": "${{ inputs.harden_runner_extra_endpoints }}",
    }


def test_failure_notice_keeps_the_fixed_pinned_list(jobs) -> None:
    assert _block_action_with(jobs["notify-failure"]) == {"config": _PINNED_CONFIG}
