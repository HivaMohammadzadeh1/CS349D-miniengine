"""CPU-tier KV cache pool — Milestone 4 (HiCache, Track 1).

Pinned host-memory mirror of :class:`~miniengine.kv_memory_pool.KVMemoryPool`.
GPU pages evicted from the radix cache are *demoted* into this pool instead
of dropped; on a later hit the cache *promotes* them back to HBM.

Layout
------
For each transformer layer we keep two pinned host tensors of shape

    (num_pages, page_size, num_kv_heads, head_dim)

— identical to the GPU pool so a copy is a straight per-page indexed write
``cpu.k[layer][cpu_slots] = gpu.k[layer][gpu_pages]`` (and the reverse on
promote). Pinned memory is what lets those copies run non-blocking on a
dedicated CUDA stream (``--hicache-overlap``).

Why pinned host memory?
    Async ``cudaMemcpyAsync`` only overlaps with compute when the host buffer
    is page-locked. Pageable memory falls back to a synchronous copy and a
    bounce buffer — defeats the whole overlap mechanism.

Capacity
--------
``num_cpu_pages`` is derived from ``--cpu-cache-size-gb`` and the per-page
byte size, the same way :meth:`KVMemoryPool.from_budget` derives the GPU
pool. ``CpuKvPool.from_budget`` is the recommended constructor.

Pending-free queue
------------------
In ``--hicache-overlap`` mode a demoted GPU page must remain reserved until
its D2H copy completes, and a CPU slot must stay reserved while its H2D copy
is in flight. We do not track that here — the cache owns the pending-free
queues and calls :meth:`free` only once the recording event has fired. Keeps
this module a pure allocator.
"""

from __future__ import annotations

from collections import deque

import torch


class CpuKvOutOfMemory(RuntimeError):
    """Raised when the CPU tier cannot satisfy an allocation.

    The cache catches this, runs CPU-tier LRU eviction, and either retries or
    falls back to dropping the GPU node entirely (m3 behavior). Distinct from
    :class:`~miniengine.kv_memory_pool.KVOutOfMemory` so the cache can tell
    the two tiers' shortages apart without inspecting the error message.
    """


class CpuKvPool:
    """Pinned host-memory paged KV pool, sized in slots-of-page-size."""

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        pin: bool = True,
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

        shape = (num_pages, page_size, num_kv_heads, head_dim)
        # pin_memory=True is only meaningful when CUDA is available; PyTorch
        # silently ignores it on CPU-only builds, which is exactly what we
        # want for Mac-side unit tests. We pass it unconditionally.
        use_pin = pin and torch.cuda.is_available()
        self._k_buffers: list[torch.Tensor] = [
            torch.empty(shape, dtype=dtype, device="cpu", pin_memory=use_pin)
            for _ in range(num_layers)
        ]
        self._v_buffers: list[torch.Tensor] = [
            torch.empty(shape, dtype=dtype, device="cpu", pin_memory=use_pin)
            for _ in range(num_layers)
        ]
        self._free: deque[int] = deque(range(num_pages))
        self._pinned = use_pin

    # ── Allocator API (mirrors KVMemoryPool's surface) ─────────────────────

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve ``num_pages`` slots and return their indices.

        Raises :class:`CpuKvOutOfMemory` on shortage. The cache catches this,
        runs CPU-tier LRU eviction, and retries (or falls back).
        """
        if num_pages <= 0:
            return []
        if len(self._free) < num_pages:
            raise CpuKvOutOfMemory(
                f"CPU KV pool out of slots: requested {num_pages}, "
                f"free {len(self._free)}, capacity {self.num_pages}"
            )
        return [self._free.popleft() for _ in range(num_pages)]

    def free(self, slot_indices: list[int]) -> None:
        """Return the listed CPU slots to the free pool."""
        for idx in slot_indices:
            self._free.append(idx)

    # ── Introspection ──────────────────────────────────────────────────────

    @property
    def num_free(self) -> int:
        """Slots currently available for allocation."""
        return len(self._free)

    @property
    def capacity(self) -> int:
        """Total slot count (constant after construction)."""
        return self.num_pages

    @property
    def k_buffers(self) -> list[torch.Tensor]:
        """Per-layer K tensor, shape ``(num_pages, page_size, kv_heads, head_dim)``."""
        return self._k_buffers

    @property
    def v_buffers(self) -> list[torch.Tensor]:
        """Per-layer V tensor, shape ``(num_pages, page_size, kv_heads, head_dim)``."""
        return self._v_buffers

    @property
    def is_pinned(self) -> bool:
        """True when underlying host buffers are page-locked (CUDA available)."""
        return self._pinned

    # ── Sizing helpers ─────────────────────────────────────────────────────

    @staticmethod
    def bytes_per_page(
        num_layers: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
        """Bytes of host memory consumed by one page across all layers, K+V."""
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        return 2 * num_layers * page_size * num_kv_heads * head_dim * elem_bytes

    @classmethod
    def from_budget(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        bytes_budget: int,
        pin: bool = True,
    ) -> "CpuKvPool":
        """Build a pool sized to ``bytes_budget`` bytes of host memory.

        The number of pages is ``bytes_budget // bytes_per_page``, floored at 1
        so a positive budget always produces at least one slot.
        """
        bpp = cls.bytes_per_page(num_layers, page_size, num_kv_heads, head_dim, dtype)
        if bpp <= 0:
            raise ValueError("derived bytes_per_page is non-positive")
        num_pages = max(1, bytes_budget // bpp)
        return cls(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            pin=pin,
        )
