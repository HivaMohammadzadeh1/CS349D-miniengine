"""HiCache (milestone 4 Track 1) tests for RadixCache.

Exercises demote-on-eviction, promote-on-hit, CPU-tier LRU eviction, lock
protection, and the fall-back drop path — all on CPU tensors so the suite
runs without a GPU.
"""

from __future__ import annotations

from collections import deque

import pytest
import torch

from miniengine.cpu_kv_pool import CpuKvPool
from miniengine.radix_cache import RadixCache


# ── Test doubles ───────────────────────────────────────────────────────────


class FakeGpuPool:
    """KVMemoryPool stand-in that lives on the CPU.

    Implements just the surface ``RadixCache`` touches:
    ``page_size``, ``allocate``, ``free``, ``kv_caches`` (per-layer K/V).
    No eviction-on-allocate hook — that's the cache's job, and our tests
    drive eviction explicitly via :meth:`RadixCache.evict`.
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int = 1,
        num_kv_heads: int = 1,
        head_dim: int = 4,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.num_pages = num_pages
        self.page_size = page_size
        shape = (num_pages, page_size, num_kv_heads, head_dim)
        self._kv = [
            (
                torch.zeros(shape, dtype=dtype),
                torch.zeros(shape, dtype=dtype),
            )
            for _ in range(num_layers)
        ]
        self._free: deque[int] = deque(range(num_pages))

    def allocate(self, n: int) -> list[int]:
        if n <= 0:
            return []
        if len(self._free) < n:
            raise RuntimeError(
                f"FakeGpuPool exhausted: requested {n}, free {len(self._free)}"
            )
        return [self._free.popleft() for _ in range(n)]

    def free(self, pages: list[int]) -> None:
        for p in pages:
            self._free.append(p)

    @property
    def kv_caches(self):
        return self._kv

    @property
    def num_free(self) -> int:
        return len(self._free)


def _make_pools(
    *,
    gpu_pages: int = 4,
    cpu_pages: int = 4,
    page_size: int = 2,
    num_layers: int = 1,
    num_kv_heads: int = 1,
    head_dim: int = 4,
):
    gpu = FakeGpuPool(
        num_pages=gpu_pages, page_size=page_size, num_layers=num_layers,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    cpu = CpuKvPool(
        num_pages=cpu_pages, page_size=page_size, num_layers=num_layers,
        num_kv_heads=num_kv_heads, head_dim=head_dim, dtype=torch.float16,
    )
    return gpu, cpu


def _seed_gpu_kv(gpu: FakeGpuPool, page: int, fill: float) -> None:
    """Write a recognisable value into every layer/K+V slot of one page."""
    for k, v in gpu.kv_caches:
        k[page].fill_(fill)
        v[page].fill_(fill + 0.5)


# ── Demote ─────────────────────────────────────────────────────────────────


def test_evict_with_cpu_pool_demotes_node_in_place():
    """Eviction in HiCache mode moves the node to CPU tier instead of dropping."""
    gpu, cpu = _make_pools(gpu_pages=4, cpu_pages=4, page_size=2)
    cache = RadixCache(gpu, cpu_pool=cpu)

    # Insert a 2-page entry; mark its pages with distinctive values.
    gpu_pages = gpu.allocate(2)   # [0, 1]
    _seed_gpu_kv(gpu, gpu_pages[0], fill=1.0)
    _seed_gpu_kv(gpu, gpu_pages[1], fill=2.0)
    leaf, _ = cache.insert_and_return([10, 11, 12, 13], gpu_pages)
    assert leaf.tier == "gpu"
    assert gpu.num_free == 2

    freed = cache.evict(2)
    assert freed == 2
    # Node still in tree but demoted.
    assert leaf.tier == "cpu"
    assert len(leaf.pages) == 2
    # GPU pages returned, CPU slots consumed.
    assert gpu.num_free == 4
    assert cpu.num_free == cpu.capacity - 2
    # Metrics.
    assert cache.metrics.total_demoted_pages == 2
    assert cache.metrics.total_evicted_pages == 0   # no DROP happened


def test_evict_without_cpu_pool_drops_like_m3():
    """The cpu_pool=None code path is byte-identical to milestone 3."""
    gpu, _ = _make_pools(gpu_pages=4, page_size=2)
    cache = RadixCache(gpu)   # cpu_pool defaults to None

    gpu_pages = gpu.allocate(2)
    leaf, _ = cache.insert_and_return([10, 11, 12, 13], gpu_pages)
    freed = cache.evict(2)
    assert freed == 2
    # Node removed; pages returned to GPU.
    assert leaf.parent is not None   # detached but reference may still see it
    assert cache.root.children == {}
    assert gpu.num_free == 4
    assert cache.metrics.total_evicted_pages == 2
    assert cache.metrics.total_demoted_pages == 0


# ── Promote + bitwise round-trip ───────────────────────────────────────────


def test_match_then_promote_refreshes_pages_to_gpu_with_kv_preserved():
    """KV survives demote→promote, and matched_pages refresh to GPU indices."""
    gpu, cpu = _make_pools(gpu_pages=4, cpu_pages=4, page_size=2)
    cache = RadixCache(gpu, cpu_pool=cpu)

    # Insert two pages with distinct KV.
    gpu_pages = gpu.allocate(2)
    _seed_gpu_kv(gpu, gpu_pages[0], fill=1.0)
    _seed_gpu_kv(gpu, gpu_pages[1], fill=2.0)
    leaf, _ = cache.insert_and_return([10, 11, 12, 13], gpu_pages)

    # Demote: evict to push it to CPU.
    cache.evict(2)
    assert leaf.tier == "cpu"
    cpu_slots = list(leaf.pages)
    # The values must be sitting in the CPU pool now.
    k0, v0 = cpu.k_buffers[0][cpu_slots[0]], cpu.v_buffers[0][cpu_slots[0]]
    assert torch.all(k0 == 1.0) and torch.all(v0 == 1.5)

    # Scribble over the original GPU page slots to prove promote re-fills.
    for layer_k, layer_v in gpu.kv_caches:
        layer_k.zero_()
        layer_v.zero_()

    # Match + promote.
    match = cache.match_prefix([10, 11, 12, 13])
    assert match.matched_tokens == 4
    # Right after match, matched_pages are still CPU slot indices.
    assert match.matched_pages == cpu_slots

    cache.promote_match(match)
    # Now they're GPU page indices (allocated fresh).
    new_gpu_pages = match.matched_pages
    assert len(new_gpu_pages) == 2
    assert all(0 <= p < gpu.num_pages for p in new_gpu_pages)
    assert leaf.tier == "gpu"
    assert leaf.pages == new_gpu_pages

    # And the KV values came back bit-for-bit.
    for layer_k, layer_v in gpu.kv_caches:
        assert torch.all(layer_k[new_gpu_pages[0]] == 1.0)
        assert torch.all(layer_v[new_gpu_pages[0]] == 1.5)
        assert torch.all(layer_k[new_gpu_pages[1]] == 2.0)
        assert torch.all(layer_v[new_gpu_pages[1]] == 2.5)

    # CPU slots were freed back.
    assert cpu.num_free == cpu.capacity
    assert cache.metrics.total_promoted_pages == 2


def test_promote_match_is_noop_when_no_cpu_pool():
    gpu, _ = _make_pools(gpu_pages=4, page_size=2)
    cache = RadixCache(gpu)
    gpu_pages = gpu.allocate(2)
    cache.insert_and_return([10, 11, 12, 13], gpu_pages)
    match = cache.match_prefix([10, 11, 12, 13])
    snapshot = list(match.matched_pages)
    cache.promote_match(match)   # no cpu_pool → no-op
    assert match.matched_pages == snapshot


def test_promote_match_is_noop_when_path_is_all_gpu():
    """Cold path: HiCache enabled but the matched path never touched CPU."""
    gpu, cpu = _make_pools(gpu_pages=4, cpu_pages=4, page_size=2)
    cache = RadixCache(gpu, cpu_pool=cpu)
    gpu_pages = gpu.allocate(2)
    cache.insert_and_return([10, 11, 12, 13], gpu_pages)
    match = cache.match_prefix([10, 11, 12, 13])
    pre = list(match.matched_pages)
    cache.promote_match(match)
    assert match.matched_pages == pre
    assert cache.metrics.total_promoted_pages == 0


# ── CPU-tier LRU + locked protection ───────────────────────────────────────


def test_cpu_overflow_drops_lru_cpu_leaves():
    """When the CPU pool fills, the oldest CPU-tier leaf is dropped to make room."""
    # CPU sized for exactly two demoted leaves (2 pages each = 4 slots). The
    # third demote then forces CPU eviction of the oldest CPU leaf.
    gpu, cpu = _make_pools(gpu_pages=8, cpu_pages=4, page_size=2)
    cache = RadixCache(gpu, cpu_pool=cpu)

    import time as _t
    leaves = []
    for prefix in ([10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]):
        pages = gpu.allocate(2)
        leaf, _ = cache.insert_and_return(prefix, pages)
        leaves.append(leaf)
        # Stagger so the LRU order is well-defined.
        _t.sleep(0.002)

    cache.evict(2)   # demotes leaves[0]
    cache.evict(2)   # demotes leaves[1]
    assert cpu.num_free == 0
    assert leaves[0].tier == "cpu" and leaves[1].tier == "cpu"

    cache.evict(2)   # third demote forces CPU-eviction of the LRU CPU leaf
    assert leaves[2].tier == "cpu"
    assert leaves[0] not in cache.root.children.values()
    assert cache.metrics.total_cpu_evicted_pages == 2


def test_locked_node_is_not_demoted_or_dropped():
    gpu, cpu = _make_pools(gpu_pages=4, cpu_pages=4, page_size=2)
    cache = RadixCache(gpu, cpu_pool=cpu)

    pages = gpu.allocate(2)
    leaf, _ = cache.insert_and_return([10, 11, 12, 13], pages)
    cache.inc_lock_ref(leaf)
    try:
        freed = cache.evict(2)
        assert freed == 0
        assert leaf.tier == "gpu"   # untouched
    finally:
        cache.dec_lock_ref(leaf)


def test_fallback_drop_when_cpu_pool_cannot_grow():
    """All CPU leaves locked → demote can't make room → falls back to drop."""
    gpu, cpu = _make_pools(gpu_pages=4, cpu_pages=2, page_size=2)
    cache = RadixCache(gpu, cpu_pool=cpu)

    # Fill CPU tier with two locked demoted nodes, then try to demote a third.
    locked_leaves = []
    for prefix in ([10, 11], [20, 21]):
        pages = gpu.allocate(1)
        leaf, _ = cache.insert_and_return(prefix, pages)
        locked_leaves.append(leaf)
    cache.evict(2)
    for leaf in locked_leaves:
        assert leaf.tier == "cpu"
        cache.inc_lock_ref(leaf)

    # Now insert a fresh GPU leaf and try to evict it. CPU is full and locked;
    # _cpu_evict can't free anything, so evict must drop the new leaf.
    pages = gpu.allocate(1)
    new_leaf, _ = cache.insert_and_return([30, 31], pages)
    freed = cache.evict(1)
    assert freed == 1
    # Either dropped from tree (parent.children no longer contains it) or its
    # pages went back to GPU as a drop (not a demote).
    assert new_leaf.tier == "gpu"   # still gpu-tier because drop, not demote
    assert cache.metrics.total_evicted_pages >= 1
    # Unlock so reset()/teardown doesn't trip.
    for leaf in locked_leaves:
        cache.dec_lock_ref(leaf)


# ── num_evictable_pages tier filter ───────────────────────────────────────


def test_num_evictable_excludes_cpu_tier_when_hicache_enabled():
    gpu, cpu = _make_pools(gpu_pages=4, cpu_pages=4, page_size=2)
    cache = RadixCache(gpu, cpu_pool=cpu)

    # Two leaves: one stays GPU, one we demote.
    pages_a = gpu.allocate(1)
    leaf_a, _ = cache.insert_and_return([10, 11], pages_a)
    pages_b = gpu.allocate(1)
    leaf_b, _ = cache.insert_and_return([20, 21], pages_b)
    cache.evict(1)   # demotes the LRU leaf

    cpu_leaf = leaf_a if leaf_a.tier == "cpu" else leaf_b
    gpu_leaf = leaf_b if cpu_leaf is leaf_a else leaf_a
    assert cpu_leaf.tier == "cpu" and gpu_leaf.tier == "gpu"

    # Only the GPU-tier leaf's page counts toward num_evictable_pages.
    assert cache.num_evictable_pages() == 1


# ── Split inherits tier (correctness for radix-tree splits over CPU edges) ─


def test_split_inherits_cpu_tier_when_parent_being_split_is_cpu():
    """A mid-edge insertion split must preserve the original tier."""
    gpu, cpu = _make_pools(gpu_pages=4, cpu_pages=4, page_size=2)
    cache = RadixCache(gpu, cpu_pool=cpu)

    pages = gpu.allocate(2)
    leaf, _ = cache.insert_and_return([10, 11, 12, 13], pages)
    cache.evict(2)  # demote to CPU
    assert leaf.tier == "cpu"

    # Now insert a prefix that diverges at the second page — forces a split
    # of the (CPU-tier) leaf. insert_and_return requires len(pages) ==
    # len(tokens) // page_size even when the first page is redundant; the
    # cache returns the redundant page via the second tuple element.
    new_gpu_pages = gpu.allocate(2)
    new_leaf, redundant = cache.insert_and_return([10, 11, 99, 99], new_gpu_pages)
    assert len(redundant) == 1   # the first page is already cached
    # The split node is the parent of both the old (now-tail) node and the new leaf.
    split_parent = new_leaf.parent
    assert split_parent is not None and split_parent is not cache.root
    assert split_parent.tier == "cpu"      # inherited from the original CPU leaf
    assert new_leaf.tier == "gpu"          # freshly inserted


# ── reset routes pages to the owning pool ─────────────────────────────────


def test_reset_returns_pages_to_correct_tier():
    gpu, cpu = _make_pools(gpu_pages=4, cpu_pages=4, page_size=2)
    cache = RadixCache(gpu, cpu_pool=cpu)

    pa = gpu.allocate(1)
    pb = gpu.allocate(1)
    cache.insert_and_return([10, 11], pa)
    cache.insert_and_return([20, 21], pb)
    cache.evict(1)   # demote one
    assert cpu.num_free == cpu.capacity - 1

    cache.reset()
    assert gpu.num_free == gpu.num_pages
    assert cpu.num_free == cpu.capacity
    assert cache.root.children == {}
