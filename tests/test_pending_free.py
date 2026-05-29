"""Deferred-free / pending-free queue tests (HiCache --hicache-overlap).

The async demote/promote path can't return source pages to the free list
until the recording CUDA event fires. Both pools share a duck-typed
``deferred_free(pages, event)`` API; ``allocate`` drains it on each call
and, as a last resort, synchronizes on the oldest pending event so a
caller never sees a spurious OOM when capacity is genuinely available.

These tests use a fake event (just ``.query()`` and ``.synchronize()``)
so they run on a laptop without CUDA.
"""

from __future__ import annotations

from collections import deque

import torch

from miniengine.cpu_kv_pool import CpuKvPool
from miniengine.kv_memory_pool import KVMemoryPool


class FakeEvent:
    """Stand-in for torch.cuda.Event with controllable fire state."""

    def __init__(self, fired: bool = True) -> None:
        self._fired = fired
        self.synchronize_calls = 0

    def query(self) -> bool:
        return self._fired

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self._fired = True

    def fire(self) -> None:
        """Simulate the GPU finishing the recorded work."""
        self._fired = True


# ── KVMemoryPool ───────────────────────────────────────────────────────────


def _gpu_pool(num_pages: int = 4) -> KVMemoryPool:
    return KVMemoryPool(
        num_pages=num_pages, page_size=2, num_layers=1, num_kv_heads=1,
        head_dim=4, dtype=torch.float16, device="cpu",
    )


def test_kv_pool_deferred_free_with_fired_event_drains_on_next_allocate():
    p = _gpu_pool(num_pages=4)
    pages = p.allocate(4)
    assert p.num_free == 0
    p.deferred_free(pages[:2], FakeEvent(fired=True))
    # Pages aren't in free list yet, but the next allocate drains them.
    assert p.num_free == 0
    got = p.allocate(2)
    assert sorted(got) == sorted(pages[:2])


def test_kv_pool_deferred_free_with_unfired_event_stays_pending():
    p = _gpu_pool(num_pages=4)
    pages = p.allocate(4)
    ev = FakeEvent(fired=False)
    p.deferred_free(pages[:2], ev)
    # No pages drained — allocate(1) would need to evict-or-sync. Since
    # there's no cache attached, sync-on-pending should fire the event.
    got = p.allocate(1)
    assert len(got) == 1
    assert ev.synchronize_calls == 1


def test_kv_pool_explicit_drain_returns_release_count():
    p = _gpu_pool(num_pages=4)
    pages = p.allocate(4)
    p.deferred_free(pages[:1], FakeEvent(fired=True))
    p.deferred_free(pages[1:3], FakeEvent(fired=False))
    released = p._drain_pending_free()
    assert released == 1
    # The not-yet-fired one is still pending.
    assert len(p._pending_free) == 1


# ── CpuKvPool ──────────────────────────────────────────────────────────────


def _cpu_pool(num_pages: int = 4) -> CpuKvPool:
    return CpuKvPool(
        num_pages=num_pages, page_size=2, num_layers=1, num_kv_heads=1,
        head_dim=4, dtype=torch.float16,
    )


def test_cpu_pool_deferred_free_with_fired_event():
    p = _cpu_pool(num_pages=4)
    slots = p.allocate(4)
    p.deferred_free(slots[:2], FakeEvent(fired=True))
    got = p.allocate(2)
    assert sorted(got) == sorted(slots[:2])


def test_cpu_pool_deferred_free_syncs_on_oldest_when_starved():
    p = _cpu_pool(num_pages=2)
    slots = p.allocate(2)
    ev = FakeEvent(fired=False)
    p.deferred_free(slots, ev)
    got = p.allocate(2)
    assert sorted(got) == sorted(slots)
    assert ev.synchronize_calls == 1


def test_cpu_pool_drain_separates_fired_from_unfired():
    p = _cpu_pool(num_pages=4)
    s = p.allocate(4)
    p.deferred_free([s[0]], FakeEvent(fired=True))
    p.deferred_free([s[1]], FakeEvent(fired=False))
    p.deferred_free([s[2]], FakeEvent(fired=True))
    released = p._drain_pending_free()
    assert released == 2
    assert len(p._pending_free) == 1
