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

    # ── Allocation API ──────────────────────────────────────────────────

    def attach_cache(self, cache: "RadixCache") -> None:
        """Wire a radix cache for eviction-on-allocate (milestone 3)."""
        self.cache = cache

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve `num_pages` pages and return their indices.

        Raises ``KVOutOfMemory`` if the request cannot be satisfied even
        after evicting LRU pages from the radix cache (when attached).
        """
        if num_pages <= 0:
            return []
        if len(self._free) < num_pages and self.cache is not None:
            need = num_pages - len(self._free)
            self.cache.evict(need)
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
