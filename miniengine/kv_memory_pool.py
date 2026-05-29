"""Pre-allocated paged KV cache memory pool — Milestone 2, Part A.

The pool owns a fixed amount of GPU memory, divided into equal-size
**pages**. Each page holds the KV state for `page_size` tokens for one
layer. Requests acquire pages as their KV grows and return them when
they finish; the cache itself never reallocates.

Storage layout
--------------
For each layer we keep one K and one V tensor of shape

    (num_pages, page_size, num_kv_heads, head_dim)

This is exactly the layout `flash_attn_with_kvcache` expects for paged
caches indexed by a `block_table` — so we can hand the same tensor to
flash-attn at decode time with zero copies. K and V are kept as
separate tensors (rather than one fused KV tensor) for the same reason:
flash-attn's API takes them separately.

Free list is a `collections.deque` of free page indices. O(1) pop/push.

Milestone 3 additions
---------------------
* Optional ``RadixCache`` integration. When attached, ``allocate`` asks
  the cache to evict LRU pages before raising on a free-list shortage.
* ``KVOutOfMemory`` — raised when neither free list nor cache eviction
  can satisfy an allocation. Caught by the scheduler's retraction loop.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from miniengine.radix_cache import RadixCache


class KVOutOfMemory(RuntimeError):
    """Raised by ``KVMemoryPool.allocate`` when no pages are available
    even after cache eviction. The scheduler catches this and retracts a
    running request back to the waiting queue (milestone 3 bonus).
    """


class KVMemoryPool:
    """Pre-allocated paged KV cache pool."""

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        if num_pages <= 0:
            raise ValueError(f"num_pages must be positive, got {num_pages}")
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")

        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        shape = (num_pages, page_size, num_kv_heads, head_dim)
        # Two tensors per layer (K, V). torch.empty is fine — pages are
        # only read after they've been written by prefill/decode.
        self._kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = [
            (
                torch.empty(shape, dtype=dtype, device=device),
                torch.empty(shape, dtype=dtype, device=device),
            )
            for _ in range(num_layers)
        ]
        # Initial free list: every page is free.
        self._free: deque[int] = deque(range(num_pages))

        # ── Milestone 3: optional radix-cache hook ─────────────────────
        # When attached, ``allocate`` calls ``cache.evict(...)`` before
        # raising on a free-list shortage. The cache owns pages held in
        # its tree (they are NOT in self._free).
        self.cache: "RadixCache | None" = None

        # ── Milestone 4 (HiCache --hicache-overlap): pending-free queue ──
        # When an async D2H demote is in flight on a dedicated CUDA stream,
        # the GPU pages it reads from CANNOT be returned to the free list
        # immediately — a fresh ``allocate`` could hand them out and
        # overwrite the source mid-copy. Instead, the cache calls
        # :meth:`deferred_free` with a recording event; ``allocate``
        # drains the queue first, returning only pages whose event has
        # fired. In the blocking path this queue stays empty.
        self._pending_free: list[tuple[list[int], object]] = []

    # ── Allocation API ──────────────────────────────────────────────────

    def attach_cache(self, cache: "RadixCache") -> None:
        """Wire a radix cache for eviction-on-allocate (milestone 3)."""
        self.cache = cache

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve `num_pages` pages and return their indices.

        Raises ``KVOutOfMemory`` if the request cannot be satisfied even
        after evicting LRU pages from the radix cache (when attached) and
        draining any pending async demotes whose copy events have fired.
        """
        if num_pages <= 0:
            return []
        # Drain async demotes first — copies whose events have completed
        # can release their source GPU pages back to the free list now.
        if self._pending_free:
            self._drain_pending_free()
        if len(self._free) < num_pages and self.cache is not None:
            need = num_pages - len(self._free)
            self.cache.evict(need)
            # Eviction may have enqueued new async demotes; drain again.
            if self._pending_free:
                self._drain_pending_free()
        # As a last resort, block on the oldest pending event so the caller
        # doesn't hit a spurious OOM when capacity is genuinely available —
        # just not yet released by the copy stream. Runs regardless of
        # whether a cache is attached so the pool stays self-consistent.
        while len(self._free) < num_pages and self._pending_free:
            pages, event = self._pending_free.pop(0)
            if hasattr(event, "synchronize"):
                event.synchronize()
            for p in pages:
                self._free.append(p)
        if len(self._free) < num_pages:
            evictable = self.cache.num_evictable_pages() if self.cache else 0
            raise KVOutOfMemory(
                f"KV pool out of pages: requested {num_pages}, "
                f"free {len(self._free)}, evictable {evictable}"
            )
        return [self._free.popleft() for _ in range(num_pages)]

    def free(self, page_indices: list[int]) -> None:
        """Return the listed pages to the free pool."""
        for idx in page_indices:
            self._free.append(idx)

    def deferred_free(self, page_indices: list[int], event: object) -> None:
        """Return pages once ``event.query()`` fires (HiCache async demote).

        ``event`` is duck-typed: anything with ``query() -> bool`` (and
        optionally ``synchronize()``). Pages stay reserved — neither in
        the free list nor in any tree node — until the next
        :meth:`allocate` drains the queue and finds the event has fired.
        """
        if not page_indices:
            return
        self._pending_free.append((list(page_indices), event))

    def _drain_pending_free(self) -> int:
        """Release pages whose async-copy events have fired. Returns count."""
        if not self._pending_free:
            return 0
        still_pending: list[tuple[list[int], object]] = []
        released = 0
        for pages, event in self._pending_free:
            ready = True
            if hasattr(event, "query"):
                try:
                    ready = bool(event.query())
                except Exception:
                    ready = True   # treat broken events as best-effort done
            if ready:
                for p in pages:
                    self._free.append(p)
                released += len(pages)
            else:
                still_pending.append((pages, event))
        self._pending_free = still_pending
        return released

    def pages_needed(self, seq_len: int) -> int:
        """How many pages are required to store `seq_len` tokens."""
        if seq_len <= 0:
            return 0
        return (seq_len + self.page_size - 1) // self.page_size

    @property
    def num_free(self) -> int:
        """Pages currently available for allocation."""
        return len(self._free)

    @property
    def num_evictable(self) -> int:
        """Pages held by the cache that an LRU sweep could free now."""
        return self.cache.num_evictable_pages() if self.cache is not None else 0

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (K, V) cache tensors, shape (num_pages, page_size, kv_heads, head_dim)."""
        return self._kv_caches

    @classmethod
    def from_budget(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        bytes_budget: int,
    ) -> "KVMemoryPool":
        """Derive `num_pages` from a memory budget and build the pool."""
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        # Bytes for one page across ALL layers and both K and V:
        #   2 (K, V) * num_layers * page_size * num_kv_heads * head_dim * elem_bytes
        bytes_per_page = (
            2 * num_layers * page_size * num_kv_heads * head_dim * elem_bytes
        )
        if bytes_per_page <= 0:
            raise ValueError("derived bytes_per_page is non-positive")
        num_pages = max(1, bytes_budget // bytes_per_page)
        return cls(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
