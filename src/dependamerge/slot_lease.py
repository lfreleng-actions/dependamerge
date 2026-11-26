# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Park-aware concurrency slots for the production merge path.

This ports the merge engine's core scheduling semantic (see
``docs/MERGE_ENGINE_DESIGN.md``) to the
legacy orchestration without decomposing ``_merge_single_pr`` into
engine phases: **a concurrency slot is held only while a PR is doing
active work; a PR waiting on an external event (a dependabot rebase,
CI checks, auto-merge) holds no slot.**

The legacy behaviour — every waiting loop running inside the worker
task that holds one of the N global semaphore slots — meant a handful
of slow external operations could pin the entire run's capacity: 41
repos needing a ~5-minute rebase at 10 slots drain in 20-25 minutes of
idle waiting.  With parking, all 41 rebases are issued in one
scheduling pass and runnable PRs never queue behind parked ones.

Usage::

    async with holding_slot(semaphore):       # replaces bare acquire
        ...active work...
        async with parked():                  # inside any wait loop
            while not done:
                await poll()                  # slot released here
        ...active work again (slot re-held)...

``parked()`` finds the current task's lease through a
:class:`contextvars.ContextVar`, so deeply nested wait loops (e.g. the
rebase module's post-rebase poll) release the slot without threading a
lease parameter through every call site.  Each ``asyncio`` task gets
its own context, so concurrent workers never see each other's lease.

While parked, the polling GETs run without a slot — API protection is
not the semaphore's job: the shared ``GitHubAsync`` client has its own
concurrency cap, RPS limiter, and adaptive throttle.

Nesting ``parked()`` inside ``parked()`` is a no-op for the inner
block (the slot is already released); ``parked()`` outside any
``holding_slot()`` is also a no-op, so helpers using it remain safe to
call from tests or non-slotted paths.

A task holding a repository's merge dispatch lock must not park with
:func:`parked` alone: a sibling on the same repository can take the
released slot and then wait for the lock, while this task waits for a
slot to come back.  Such a task holds the lock through
:func:`holding_dispatch_lock`, and its waits use
:func:`parked_off_dispatch`, which lets go of the lock as well and takes
the slot back before the lock, the order every other task takes them in.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections.abc import AsyncIterator

__all__ = [
    "SlotLease",
    "current_lease",
    "holding_dispatch_lock",
    "holding_slot",
    "parked",
    "parked_off_dispatch",
]


class SlotLease:
    """A releasable, re-acquirable hold on an ``asyncio.Semaphore``.

    Tracks whether the permit is currently held so ``release`` /
    ``acquire`` are idempotent: releasing an already-released lease or
    re-acquiring a held one is a no-op, never a semaphore imbalance.
    """

    __slots__ = ("_semaphore", "_held")

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    async def acquire(self) -> None:
        """Acquire the underlying permit (no-op when already held)."""
        if not self._held:
            await self._semaphore.acquire()
            self._held = True

    def release(self) -> None:
        """Release the underlying permit (no-op when not held)."""
        if self._held:
            self._semaphore.release()
            self._held = False


_current_lease: contextvars.ContextVar[SlotLease | None] = contextvars.ContextVar(
    "dependamerge_slot_lease", default=None
)


def current_lease() -> SlotLease | None:
    """The current task's slot lease, or ``None`` outside one."""
    return _current_lease.get()


@contextlib.asynccontextmanager
async def holding_slot(semaphore: asyncio.Semaphore) -> AsyncIterator[SlotLease]:
    """Hold one slot of ``semaphore`` for the duration of the block.

    Drop-in replacement for ``async with semaphore:`` that additionally
    publishes a :class:`SlotLease` to the task context so nested
    :func:`parked` blocks can release the slot while waiting.
    """
    lease = SlotLease(semaphore)
    await lease.acquire()
    token = _current_lease.set(lease)
    try:
        yield lease
    finally:
        _current_lease.reset(token)
        # Balanced regardless of what the block did: exactly one
        # release when the lease is still held, none when a parked
        # block was cancelled mid-reacquire.
        lease.release()


@contextlib.asynccontextmanager
async def parked() -> AsyncIterator[None]:
    """Release the current task's slot for the duration of a wait.

    On exit the slot is re-acquired (competing fairly with queued
    work) before active processing resumes.  A no-op when the task
    holds no lease or the lease is already released (nested parks).
    """
    lease = _current_lease.get()
    if lease is None or not lease.held:
        yield
        return
    lease.release()
    try:
        yield
    finally:
        # Re-acquire even when the wait body raised: the caller's
        # exception handling (and eventually ``holding_slot``'s
        # ``finally``) expects the lease to be in its pre-park state.
        # If *this* acquire is cancelled, the lease stays released and
        # ``holding_slot`` correctly skips its final release.
        await lease.acquire()


class _DispatchLease:
    """A dispatch lock held by one task, which alone may release it."""

    __slots__ = ("lock", "owner", "held")

    def __init__(self, lock: asyncio.Lock) -> None:
        self.lock = lock
        self.owner = asyncio.current_task()
        self.held = False


_current_dispatch: contextvars.ContextVar[_DispatchLease | None] = (
    contextvars.ContextVar("dependamerge_dispatch_lease", default=None)
)


@contextlib.asynccontextmanager
async def holding_dispatch_lock(lock: asyncio.Lock) -> AsyncIterator[None]:
    """Hold a merge dispatch lock, so :func:`parked_off_dispatch` finds it."""
    lease = _DispatchLease(lock)
    await lock.acquire()
    lease.held = True
    token = _current_dispatch.set(lease)
    try:
        yield
    finally:
        _current_dispatch.reset(token)
        # Skipped when a park's re-acquire was cancelled: the lock is
        # then not ours to release.
        if lease.held:
            lock.release()


@contextlib.asynccontextmanager
async def parked_off_dispatch() -> AsyncIterator[None]:
    """:func:`parked`, also letting go of the dispatch lock this task holds.

    The lock is taken back after the slot, never while waiting for one,
    so no task holds the lock while it waits for a slot, and a task
    cancelled while parked takes back neither, with or without a lock
    to let go of.  A task that inherited the lock's lease from its
    parent's context does not own it and leaves it alone.
    """
    lease = _current_dispatch.get()
    if lease is not None and (
        not lease.held or lease.owner is not asyncio.current_task()
    ):
        lease = None
    if lease is not None:
        lease.lock.release()
        lease.held = False
    slot = _current_lease.get()
    if slot is not None and not slot.held:
        slot = None
    if slot is not None:
        slot.release()
    # A task cancelled in the wait or while taking its slot back unwinds
    # without the slot and the lock, rather than wait for a sibling's
    # work to finish; holding_slot() and holding_dispatch_lock() then
    # skip their releases.
    cancelled = False
    try:
        yield
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        if not cancelled:
            if slot is not None:
                await slot.acquire()
            if lease is not None:
                await lease.lock.acquire()
                lease.held = True
