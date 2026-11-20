# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""The ``--rebase-cooldown`` option on ``merge``."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from dependamerge.cli import app
from dependamerge.merge_manager import AsyncMergeManager
from dependamerge.progress_tracker import MergeProgressTracker

URL = "https://github.com/owner/repo/pull/22"


def _invoke(*extra: str) -> tuple[Any, list[Any]]:
    captured: list[Any] = []

    def fake_run_guarded(ctx: Any, _message: str, _action: Any) -> None:
        captured.append(ctx)

    with patch("dependamerge.cli._merge_cmd._run_guarded", fake_run_guarded):
        result = CliRunner().invoke(
            app, ["merge", URL, "--token", "test_token", *extra]
        )
    return result, captured


@pytest.mark.parametrize(
    ("extra", "expected"),
    [((), 8.0), (("--rebase-cooldown", "0"), 0.0), (("--rebase-cooldown", "2.5"), 2.5)],
)
def test_cooldown_reaches_context(extra: tuple[str, ...], expected: float) -> None:
    result, captured = _invoke(*extra)

    assert result.exit_code == 0, result.stdout
    assert [ctx.rebase_cooldown for ctx in captured] == [expected]


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_invalid_cooldown_refused(value: str) -> None:
    result, captured = _invoke("--rebase-cooldown", value)

    assert result.exit_code == 1
    assert "Invalid --rebase-cooldown" in result.stdout
    assert captured == []


@pytest.mark.parametrize(("hours", "seconds"), [(8, 28800.0), (0, 0.0)])
def test_manager_stores_cooldown_in_seconds(hours: float, seconds: float) -> None:
    mgr = AsyncMergeManager(token="t", rebase_cooldown=hours)

    assert mgr._rebase_cooldown_seconds == seconds


def test_manager_falls_back_on_negative_cooldown() -> None:
    mgr = AsyncMergeManager(token="t", rebase_cooldown=-3)

    assert mgr._rebase_cooldown_seconds == 8 * 3600.0


def test_positional_arguments_keep_their_settings() -> None:
    """The new option is appended, so existing positional calls still bind."""
    tracker = MagicMock(spec=MergeProgressTracker)
    mgr = AsyncMergeManager("t", "squash", 2, 5, False, 600.0, tracker, True)

    assert mgr.progress_tracker is tracker
    assert mgr.preview_mode is True
    assert mgr._rebase_cooldown_seconds == 8 * 3600.0
