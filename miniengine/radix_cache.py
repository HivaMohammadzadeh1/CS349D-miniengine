"""Radix-tree prefix cache — Milestone 3, Part B.

Stores already-computed KV pages keyed by token prefix so a new request whose
prompt starts with a cached prefix can reuse those pages instead of
recomputing them.

The data structure is a radix tree whose nodes own KV pages from the
``KVMemoryPool``.  Pages held by the cache are *not* in the pool's free list;
they return there only when the cache evicts them (LRU) or when an
``insert_and_return`` call discovers them as duplicates of pages already in
the tree.

Invariants
----------
* Every node's edge ``key`` is page-aligned: ``len(key) % page_size == 0``
  and ``len(pages) == len(key) // page_size``.
* Sibling edges (children of the same node) have unique **first pages** —
  the ``children`` dict is keyed by ``tuple(key[:page_size])``. Keying by
  the full first page (not just the first token) is necessary because two
  unrelated prompts can share the first token of the chat template but
  diverge within the first page; with page-aligned matching they must be
  stored as separate siblings.
* ``ref_count > 0`` on a node means the node *and* every ancestor up to root
  are pinned against eviction (an in-flight request is borrowing them).

Performance counters in ``CacheMetrics`` are surfaced via ``/cache_stats``.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miniengine.cpu_kv_pool import CpuKvPool
    from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Aggregate cache statistics — surfaced via ``/cache_stats``."""

    total_lookups: int = 0
    total_query_tokens: int = 0
    total_hit_tokens: int = 0
    total_inserted_pages: int = 0
    total_evicted_pages: int = 0
    # ── Milestone 4 (HiCache) ──────────────────────────────────────────────
    # Pages demoted GPU→CPU (kept in tree, colder tier).
    total_demoted_pages: int = 0
    # Pages promoted CPU→GPU on a hit.
    total_promoted_pages: int = 0
    # CPU-tier nodes dropped entirely when the CPU pool overflows.
    total_cpu_evicted_pages: int = 0
    # Wall time (ms) spent in demote/promote indexed copies. In blocking
    # mode that's the synchronous copy time; in --hicache-overlap mode it's
    # the event-measured GPU copy-stream time.
    total_demote_time_ms: float = 0.0
    total_promote_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_query_tokens == 0:
            return 0.0
        return self.total_hit_tokens / self.total_query_tokens


class RadixNode:
    """A radix-tree node.

    Edge layout: ``key`` carries the tokens on the edge from the parent;
    ``pages`` carries the KV pages corresponding to ``key`` (one page per
    ``page_size`` tokens; both are page-aligned by construction).

    Tier (milestone 4 — HiCache)
        ``tier`` is ``"gpu"`` when ``pages`` are indices into the GPU
        :class:`KVMemoryPool`, ``"cpu"`` when they are slot indices into the
        host-side :class:`CpuKvPool`. A node's pages live entirely in one
        tier (page granularity, single-tier per node — no mixing). The field
        defaults to ``"gpu"``; when ``RadixCache.cpu_pool is None`` it stays
        ``"gpu"`` forever and behavior is byte-identical to milestone 3.
    """

    __slots__ = (
        "parent", "children", "key", "pages", "ref_count", "last_access", "tier",
    )

    def __init__(self) -> None:
        self.parent: RadixNode | None = None
        # Keyed by tuple(key[:page_size]) — see module docstring for why
        # first-token keying is not sufficient.
        self.children: dict[tuple[int, ...], "RadixNode"] = {}
        self.key: list[int] = []
        self.pages: list[int] = []
        self.ref_count: int = 0
        self.last_access: float = time.monotonic()
        self.tier: str = "gpu"


@dataclass
class MatchResult:
    """Result of a prefix lookup.

    ``matched_tokens`` is page-aligned (multiple of ``page_size``);
    ``matched_pages`` carries the KV pages for those tokens.
    ``last_node`` is the deepest node the walk reached — callers lock it
    (``inc_lock_ref``) for the lifetime of the borrowing request.
    """

    matched_pages: list[int] = field(default_factory=list)
    matched_tokens: int = 0
    last_node: "RadixNode | None" = None


class RadixCache:
    """Token-prefix → KV-pages cache backed by a radix tree.

    Page-aligned matching, LRU eviction of unlocked subtrees,
    eviction-on-allocate via ``KVMemoryPool.allocate``, and sglang-style
    ``inc_lock_ref`` / ``dec_lock_ref`` pinning for in-flight matches.
    """

    def __init__(
        self,
        pool: "KVMemoryPool",
        cpu_pool: "CpuKvPool | None" = None,
        copy_stream=None,
        overlap: bool = False,
    ) -> None:
        self.pool = pool
        self.page_size = pool.page_size
        self.root = RadixNode()
        self.metrics = CacheMetrics()
        self._num_cached_pages = 0
        # Tie-breaker for heap entries with equal last_access timestamps.
        self._heap_seq = itertools.count()
        # ── Milestone 4 (HiCache) ──────────────────────────────────────────
        # When ``cpu_pool`` is provided, GPU eviction *demotes* into it
        # instead of dropping; on a hit, the engine calls
        # :meth:`promote_match` to lift CPU-resident nodes back to GPU.
        # ``cpu_pool is None`` keeps every code path byte-identical to m3.
        self.cpu_pool: "CpuKvPool | None" = cpu_pool
        # --hicache-overlap (bonus): when ``overlap`` and a CUDA stream are
        # provided, demote/promote copies run on ``copy_stream`` and the
        # source/destination pages stay reserved (pool ``deferred_free``)
        # until the recording event fires. ``overlap=False`` is the
        # blocking path that lands first.
        self.copy_stream = copy_stream
        self.overlap: bool = bool(overlap and copy_stream is not None
                                  and cpu_pool is not None)

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def num_cached_pages(self) -> int:
        """Total pages currently held by the tree."""
        return self._num_cached_pages

    def num_evictable_pages(self) -> int:
        """GPU pages that an LRU sweep could reclaim right now.

        With HiCache enabled, only *GPU-tier* leaves are counted — CPU-tier
        leaves hold no GPU pages, so they cannot satisfy a GPU shortage
        directly (their slots can be freed in the CPU pool, but that only
        makes room for a subsequent demote). The GPU pool's
        ``num_evictable`` query is exactly this number.
        """
        total = 0
        for node in self._walk():
            if node is self.root:
                continue
            if not self._is_evictable(node):
                continue
            if self.cpu_pool is not None and node.tier != "gpu":
                continue
            total += len(node.pages)
        return total

    def _walk(self):
        """Iterate every node in the tree (including root)."""
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    def _is_evictable(self, node: "RadixNode") -> bool:
        """A node is evictable when nothing in its subtree is locked."""
        return node.ref_count == 0 and not node.children

    # ── Lookup ─────────────────────────────────────────────────────────

    def match_prefix(self, tokens: list[int]) -> MatchResult:
        """Find the longest page-aligned prefix of ``tokens`` in the tree.

        Walks page-by-page: at each level the next child is looked up by
        ``tuple(tokens[offset:offset+ps])``. Along the matched edge we
        continue in ``page_size`` chunks until the edge ends or a page's
        tokens diverge.
        """
        ps = self.page_size
        self.metrics.total_lookups += 1
        self.metrics.total_query_tokens += len(tokens)

        matched_pages: list[int] = []
        matched_tokens = 0
        node = self.root
        offset = 0
        now = time.monotonic()

        while offset + ps <= len(tokens):
            page_key = tuple(tokens[offset : offset + ps])
            child = node.children.get(page_key)
            if child is None:
                break
            # First page guaranteed to match (dict key). Walk further pages
            # along this edge.
            edge_off = ps
            edge_len = len(child.key)
            matched_pages.append(child.pages[0])
            matched_tokens += ps
            offset += ps
            while (
                edge_off + ps <= edge_len
                and offset + ps <= len(tokens)
                and child.key[edge_off : edge_off + ps]
                == tokens[offset : offset + ps]
            ):
                matched_pages.append(child.pages[edge_off // ps])
                matched_tokens += ps
                edge_off += ps
                offset += ps

            if edge_off < edge_len:
                # Partial edge match — stop at the last full page boundary.
                # The child stays where it is; no split is necessary on a
                # pure lookup.
                node = child
                break

            # Full edge consumed; descend.
            child.last_access = now
            node = child

        self.metrics.total_hit_tokens += matched_tokens
        return MatchResult(
            matched_pages=matched_pages,
            matched_tokens=matched_tokens,
            last_node=node,
        )

    # ── Lock ref counting (sglang-style) ───────────────────────────────

    def inc_lock_ref(self, node: "RadixNode | None") -> None:
        """Lock ``node`` (and the path to root) against eviction."""
        if node is None:
            return
        cur = node
        while cur is not None and cur is not self.root:
            cur.ref_count += 1
            cur = cur.parent

    def dec_lock_ref(self, node: "RadixNode | None") -> None:
        """Release a lock.  Refresh ``last_access`` while walking."""
        if node is None:
            return
        now = time.monotonic()
        cur = node
        while cur is not None and cur is not self.root:
            if cur.ref_count > 0:
                cur.ref_count -= 1
            cur.last_access = now
            cur = cur.parent

    # ── Insertion ──────────────────────────────────────────────────────

    def insert_and_return(
        self, tokens: list[int], pages: list[int]
    ) -> tuple["RadixNode", list[int]]:
        """Insert (tokens, pages) into the tree.

        Both arguments must be page-aligned:
            len(tokens) % page_size == 0
            len(pages) == len(tokens) // page_size

        Returns ``(leaf_node, redundant_pages)``: ``redundant_pages`` are
        pages the caller handed in that turned out to be duplicates of pages
        already cached at the same prefix.  The caller should return them to
        the pool.
        """
        ps = self.page_size
        if len(tokens) % ps != 0:
            raise ValueError(
                f"insert_and_return requires page-aligned tokens; "
                f"got len={len(tokens)} ps={ps}"
            )
        if len(pages) != len(tokens) // ps:
            raise ValueError(
                f"insert_and_return: pages/tokens mismatch ({len(pages)} vs "
                f"{len(tokens) // ps})"
            )

        redundant: list[int] = []
        node = self.root
        offset = 0
        now = time.monotonic()

        while offset < len(tokens):
            page_key = tuple(tokens[offset : offset + ps])
            child = node.children.get(page_key)
            if child is None:
                # Fresh edge — append the remainder as a new leaf.
                new_node = RadixNode()
                new_node.parent = node
                new_node.key = list(tokens[offset:])
                new_node.pages = list(pages[offset // ps :])
                new_node.last_access = now
                node.children[page_key] = new_node
                self._num_cached_pages += len(new_node.pages)
                self.metrics.total_inserted_pages += len(new_node.pages)
                return new_node, redundant

            # First page guaranteed to match (dict key). Walk further pages.
            edge_off = ps
            edge_len = len(child.key)
            redundant.append(pages[offset // ps])
            offset += ps
            while (
                edge_off < edge_len
                and offset < len(tokens)
                and child.key[edge_off : edge_off + ps]
                == tokens[offset : offset + ps]
            ):
                # Page already cached — caller's copy is redundant.
                redundant.append(pages[offset // ps])
                edge_off += ps
                offset += ps

            if edge_off == edge_len:
                # Full edge consumed and matched; descend.
                child.last_access = now
                node = child
                continue

            # Mismatch mid-edge at a page boundary — split. edge_off >= ps
            # is guaranteed (first page matched via the dict lookup).
            split = RadixNode()
            split.parent = node
            split.key = child.key[:edge_off]
            split.pages = child.pages[: edge_off // ps]
            split.last_access = now
            # Inherit the original child's tier — split.pages are sliced from
            # child.pages, so they live in the same pool. Without this,
            # CPU-tier child pages would be mis-tagged as GPU and the next
            # GPU eviction would try to free them via the GPU pool. Identity
            # for m3 (every node is "gpu" anyway).
            split.tier = child.tier
            # Re-parent the tail of the original child under the split.
            old_child_first_page = page_key   # before mutating child.key
            child.key = child.key[edge_off:]
            child.pages = child.pages[edge_off // ps :]
            child.parent = split
            split.children[tuple(child.key[:ps])] = child
            # Split inherits the SAME first-page key the old child had
            # under `node`, since split.key starts at child.key's old
            # first page. Overwrite the parent's dict entry.
            node.children[old_child_first_page] = split
            # ref_count on a node is "locked leaves in subtree"; after
            # split, all that lock-weight belongs to descendants, so split
            # inherits child's ref_count.
            split.ref_count = child.ref_count

            if offset == len(tokens):
                return split, redundant

            # Attach the remaining input under the split as a new branch.
            new_node = RadixNode()
            new_node.parent = split
            new_node.key = list(tokens[offset:])
            new_node.pages = list(pages[offset // ps :])
            new_node.last_access = now
            split.children[tuple(tokens[offset : offset + ps])] = new_node
            self._num_cached_pages += len(new_node.pages)
            self.metrics.total_inserted_pages += len(new_node.pages)
            return new_node, redundant

        # Exhausted tokens exactly at a node boundary — return the deepest
        # node we landed on. Everything was redundant.
        return node, redundant

    # ── Eviction ───────────────────────────────────────────────────────

    def evict(self, n_pages_needed: int) -> int:
        """LRU-reclaim at least ``n_pages_needed`` GPU pages (best effort).

        Plain m3 mode (``cpu_pool is None``): evictable leaves are dropped
        and their pages returned to the GPU pool.

        HiCache mode (``cpu_pool`` set): each LRU-selected *GPU-tier* leaf
        is **demoted** to the CPU tier instead — its KV is copied D2H, the
        GPU pages are returned to the pool, and the node stays in the tree
        marked ``tier="cpu"``. If the CPU pool is too full and
        :meth:`_cpu_evict` can't make room either, we fall back to dropping
        the node entirely (m3 behavior) so eviction always makes progress.

        Returns the number of GPU pages actually freed.
        """
        if n_pages_needed <= 0:
            return 0

        hicache = self.cpu_pool is not None

        # Min-heap of (last_access, seq, node) over current GPU-evictable
        # leaves. With HiCache on, CPU-tier nodes hold no GPU pages so
        # they're skipped here (the CPU tier has its own LRU in _cpu_evict).
        heap: list[tuple[float, int, RadixNode]] = []
        for node in self._walk():
            if node is self.root:
                continue
            if not self._is_evictable(node):
                continue
            if hicache and node.tier != "gpu":
                continue
            heapq.heappush(heap, (node.last_access, next(self._heap_seq), node))

        freed = 0
        while heap and freed < n_pages_needed:
            _, _, node = heapq.heappop(heap)
            # Re-validate: a re-pushed parent may have re-acquired
            # children/locks; a node may have been demoted between pushes.
            if not self._is_evictable(node):
                continue
            if hicache and node.tier != "gpu":
                continue
            if not node.pages:
                continue

            page_count = len(node.pages)

            if hicache and self._try_demote(node):
                # Demoted in place: node stays in the tree as CPU-tier,
                # its GPU pages were returned to the pool. Parent does NOT
                # become a leaf (this node is still its child).
                freed += page_count
                continue

            # Fall-back / m3 path: drop the node entirely.
            self.pool.free(node.pages)
            freed += page_count
            self._num_cached_pages -= page_count
            self.metrics.total_evicted_pages += page_count
            parent = node.parent
            if parent is not None:
                # Detach from parent's children dict (keyed by first page).
                parent.children.pop(tuple(node.key[: self.page_size]), None)
                if self._is_evictable(parent) and parent is not self.root:
                    # Only re-push if the parent is itself a GPU eviction
                    # candidate (m3: always; HiCache: must be GPU-tier).
                    if not hicache or parent.tier == "gpu":
                        heapq.heappush(
                            heap, (parent.last_access, next(self._heap_seq), parent)
                        )

        return freed

    # ── HiCache: demote / promote / CPU-tier eviction ──────────────────

    def _try_demote(self, node: "RadixNode") -> bool:
        """Move ``node``'s pages from GPU to CPU; keep the node in the tree.

        Returns ``True`` on success, ``False`` when the CPU pool can't accept
        the pages (even after :meth:`_cpu_evict`). On failure the caller
        falls back to dropping the node.
        """
        assert self.cpu_pool is not None
        assert node.tier == "gpu"

        need = len(node.pages)
        # Ensure CPU room. If we can't make enough, fail back to caller.
        if self.cpu_pool.num_free < need:
            shortfall = need - self.cpu_pool.num_free
            self._cpu_evict(shortfall)
        if self.cpu_pool.num_free < need:
            return False

        # Allocate CPU slots.
        from miniengine.cpu_kv_pool import CpuKvOutOfMemory
        try:
            cpu_slots = self.cpu_pool.allocate(need)
        except CpuKvOutOfMemory:
            return False

        # D2H copy of K and V for every layer.
        gpu_pages = node.pages
        t0 = time.monotonic()
        gpu_kv = self.pool.kv_caches  # list[(K, V)] per layer
        cpu_k = self.cpu_pool.k_buffers
        cpu_v = self.cpu_pool.v_buffers
        if self.overlap:
            # Async on the dedicated copy stream. The PyTorch indexed-copy
            # of a CUDA tensor into a pinned CPU tensor IS a non-blocking
            # DMA when the source is on CUDA and the destination is pinned.
            # Issuing under stream context routes the DMA to copy_stream.
            import torch  # local import keeps the blocking path import-free
            with torch.cuda.stream(self.copy_stream):
                for layer, (k_gpu, v_gpu) in enumerate(gpu_kv):
                    cpu_k[layer][cpu_slots] = k_gpu[gpu_pages].to(
                        cpu_k[layer].device, non_blocking=True
                    )
                    cpu_v[layer][cpu_slots] = v_gpu[gpu_pages].to(
                        cpu_v[layer].device, non_blocking=True
                    )
                event = torch.cuda.Event()
                event.record(self.copy_stream)
            # Defer freeing GPU pages until the D2H event fires; otherwise
            # a fresh allocate could hand out a page that the copy stream
            # is still reading.
            self.pool.deferred_free(gpu_pages, event)
        else:
            # Blocking: synchronous indexed copy on the default stream.
            for layer, (k_gpu, v_gpu) in enumerate(gpu_kv):
                cpu_k[layer][cpu_slots] = k_gpu[gpu_pages].to(cpu_k[layer].device)
                cpu_v[layer][cpu_slots] = v_gpu[gpu_pages].to(cpu_v[layer].device)
            self.pool.free(gpu_pages)
        self.metrics.total_demote_time_ms += (time.monotonic() - t0) * 1000.0

        # Repoint the node — GPU pages have been (or will be) returned.
        node.pages = cpu_slots
        node.tier = "cpu"
        self.metrics.total_demoted_pages += need
        # Demoted pages no longer count as "cached" GPU pages; HiCache
        # tracks them separately. Keeping _num_cached_pages = GPU pages
        # only matches what KVMemoryPool can see.
        self._num_cached_pages -= need
        return True

    def _cpu_evict(self, n_slots_needed: int) -> int:
        """LRU-drop CPU-tier leaves to make room for new demotions.

        No lower tier — evicted CPU nodes are removed from the tree entirely
        (their prefix is lost; a future request matching that prefix will
        re-prefill from scratch). Returns slots freed.
        """
        if n_slots_needed <= 0 or self.cpu_pool is None:
            return 0

        heap: list[tuple[float, int, RadixNode]] = []
        for node in self._walk():
            if node is self.root:
                continue
            if node.tier != "cpu":
                continue
            if not self._is_evictable(node):
                continue
            heapq.heappush(heap, (node.last_access, next(self._heap_seq), node))

        freed = 0
        while heap and freed < n_slots_needed:
            _, _, node = heapq.heappop(heap)
            if not self._is_evictable(node) or node.tier != "cpu":
                continue
            if not node.pages:
                continue

            count = len(node.pages)
            self.cpu_pool.free(node.pages)
            freed += count
            self.metrics.total_cpu_evicted_pages += count
            parent = node.parent
            if parent is not None:
                parent.children.pop(tuple(node.key[: self.page_size]), None)
                # Parent may now be a CPU-tier leaf — push it back.
                if (
                    parent is not self.root
                    and self._is_evictable(parent)
                    and parent.tier == "cpu"
                ):
                    heapq.heappush(
                        heap, (parent.last_access, next(self._heap_seq), parent)
                    )
        return freed

    def _promote_node(self, node: "RadixNode") -> None:
        """Lift one CPU-tier node back to GPU.

        Allocates GPU pages (which may itself trigger demotion of *other*
        cold nodes — that's fine; the path being promoted is locked by the
        caller). Copies CPU→GPU and repoints the node. Caller is
        responsible for locking the path before calling this.
        """
        assert self.cpu_pool is not None
        assert node.tier == "cpu"

        need = len(node.pages)
        gpu_pages = self.pool.allocate(need)   # may evict-and-demote others

        t0 = time.monotonic()
        gpu_kv = self.pool.kv_caches
        cpu_slots = node.pages
        cpu_k = self.cpu_pool.k_buffers
        cpu_v = self.cpu_pool.v_buffers
        if self.overlap:
            import torch
            with torch.cuda.stream(self.copy_stream):
                for layer, (k_gpu, v_gpu) in enumerate(gpu_kv):
                    k_gpu[gpu_pages] = cpu_k[layer][cpu_slots].to(
                        k_gpu.device, non_blocking=True
                    )
                    v_gpu[gpu_pages] = cpu_v[layer][cpu_slots].to(
                        v_gpu.device, non_blocking=True
                    )
                event = torch.cuda.Event()
                event.record(self.copy_stream)
            # The compute stream that runs the request's next forward must
            # wait on the H2D event — flash-attn cannot read half-copied KV.
            # wait_event is non-blocking on the CPU; only the GPU stream
            # serializes.
            torch.cuda.current_stream().wait_event(event)
            # The H2D source (cpu_slots) is still being read by the copy
            # stream — defer-free so a fresh allocate doesn't overwrite it.
            self.cpu_pool.deferred_free(cpu_slots, event)
        else:
            for layer, (k_gpu, v_gpu) in enumerate(gpu_kv):
                k_gpu[gpu_pages] = cpu_k[layer][cpu_slots].to(k_gpu.device)
                v_gpu[gpu_pages] = cpu_v[layer][cpu_slots].to(v_gpu.device)
            self.cpu_pool.free(cpu_slots)
        self.metrics.total_promote_time_ms += (time.monotonic() - t0) * 1000.0

        node.pages = gpu_pages
        node.tier = "gpu"
        self.metrics.total_promoted_pages += need
        # Promoted pages are now GPU-resident; restore the cached-pages
        # accounting that demote subtracted.
        self._num_cached_pages += need

    def promote_match(self, match: MatchResult) -> None:
        """Ensure every page in ``match.matched_pages`` is GPU-resident.

        Walks the matched path root→leaf, promotes any CPU-tier nodes back
        to GPU, and rewrites ``match.matched_pages`` with the refreshed GPU
        page indices. A no-op when HiCache is disabled or when the matched
        path is already all-GPU — that keeps the cold path on m3 exactly
        identical to milestone 3.

        Concurrency: the matched leaf is temp-locked during promotion so
        the allocator (which may run :meth:`evict` mid-promotion) cannot
        demote any node on the path back out from under us.
        """
        if self.cpu_pool is None:
            return
        leaf = match.last_node
        if leaf is None or leaf is self.root:
            return

        # Path from root → leaf, then filter to CPU-tier nodes.
        path: list[RadixNode] = []
        cur: RadixNode | None = leaf
        while cur is not None and cur is not self.root:
            path.append(cur)
            cur = cur.parent
        path.reverse()
        cpu_nodes = [n for n in path if n.tier == "cpu"]
        if not cpu_nodes:
            return   # all-GPU match — fast path, no work

        # How many of leaf.pages were actually used by match_prefix? Plain
        # arithmetic on the original matched_pages: ancestors contributed
        # all their pages; the remainder is the leaf's used prefix.
        ancestors_pages = sum(len(n.pages) for n in path[:-1])
        last_used = len(match.matched_pages) - ancestors_pages
        assert 0 <= last_used <= len(leaf.pages)

        # Temp-lock the leaf — pins the whole ancestor chain against GPU
        # eviction triggered by our own allocations during promotion.
        self.inc_lock_ref(leaf)
        try:
            for node in cpu_nodes:
                self._promote_node(node)
        finally:
            self.dec_lock_ref(leaf)

        # Refresh matched_pages with the new (now-GPU) page indices.
        rebuilt: list[int] = []
        for node in path[:-1]:
            rebuilt.extend(node.pages)
        rebuilt.extend(leaf.pages[:last_used])
        match.matched_pages = rebuilt

    # ── Maintenance ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop the whole tree, return every page to its owning pool."""
        gpu_pages: list[int] = []
        cpu_slots: list[int] = []
        for node in self._walk():
            if node is self.root:
                continue
            if node.tier == "cpu":
                cpu_slots.extend(node.pages)
            else:
                gpu_pages.extend(node.pages)
        if gpu_pages:
            self.pool.free(gpu_pages)
        if cpu_slots and self.cpu_pool is not None:
            self.cpu_pool.free(cpu_slots)
        self.root.children.clear()
        self._num_cached_pages = 0
        self.metrics = CacheMetrics()
