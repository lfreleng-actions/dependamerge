# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Watching an in-progress merge must not strand a same-repository sibling.

The merge retry loop holds the repository's dispatch lock when GitHub
says a merge is already in progress.  A sibling backing off a server
error resumes by taking a slot and then the lock; if the watch kept
the lock while parked, it would then wait for the slot the sibling
holds, and neither could finish (issue #688).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from dependamerge.slot_lease import (
    holding_dispatch_lock,
    holding_slot,
    parked_off_dispatch,
)
from tests.conftest import make_merge_manager, make_pr_info


@pytest.mark.asyncio
async def test_a_server_error_backoff_resumes_past_an_in_progress_watch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dependamerge.merge_manager.MERGE_IN_PROGRESS_POLL_SECONDS", 0.001
    )
    monkeypatch.setattr(
        "dependamerge.merge_manager.MERGE_IN_PROGRESS_FIRST_POLL_SECONDS", 0.001
    )
    mgr, client = make_merge_manager()
    pr = make_pr_info()
    slots = asyncio.Semaphore(1)
    lock = asyncio.Lock()
    a_parked = asyncio.Event()
    b_watching = asyncio.Event()
    a_has_slot = asyncio.Event()
    finished: list[str] = []

    async def poll(_path: str, **_kwargs: object) -> dict[str, object]:
        if not b_watching.is_set():
            b_watching.set()
            await a_has_slot.wait()
            return {"merged": False, "state": "open"}
        return {"merged": True, "state": "closed"}

    client.get = AsyncMock(side_effect=poll)

    async def server_error_backoff() -> None:
        async with holding_slot(slots), holding_dispatch_lock(lock):
            async with parked_off_dispatch():
                a_parked.set()
                await b_watching.wait()
            finished.append("a")

    async def in_progress_watch() -> None:
        await a_parked.wait()
        async with holding_slot(slots), holding_dispatch_lock(lock):
            assert await mgr._await_in_progress_merge("o", "r", pr, f"o/r#{pr.number}")
            finished.append("b")

    async def slot_taken() -> None:
        # A takes the slot back, and with the lock free may finish at once.
        while not slots.locked() and "a" not in finished:
            await asyncio.sleep(0)
        a_has_slot.set()

    tasks = [
        asyncio.create_task(server_error_backoff()),
        asyncio.create_task(in_progress_watch()),
    ]
    await b_watching.wait()
    watcher = asyncio.create_task(slot_taken())
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
    finally:
        for task in [*tasks, watcher]:
            task.cancel()
        await asyncio.gather(*tasks, watcher, return_exceptions=True)
    assert sorted(finished) == ["a", "b"]
    assert not lock.locked()
    assert not slots.locked()
