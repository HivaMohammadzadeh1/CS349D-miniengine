"""
CUDA-graph capture for paged decode — Milestone 2 extra credit.

Captures the paged-decode forward at a set of *bucket* batch sizes and
replays at runtime. The actual live batch B is rounded up to the
smallest captured bucket ≥ B; padded entries [B, bucket) point at a
reserved **scratch page** so the kernel's K/V writes for those entries
land in a safe scratchpad rather than corrupting any real request's
cache.

Invariants the captured region satisfies:
  - Stable tensor identities (input_ids, position_ids, cache_seqlens,
    block_table, logits_buf, plus the KV pool's per-layer K and V
    tensors).
  - No CPU↔GPU sync inside the graph (no `.item()`, no Python branches
    on tensor values, no allocations).
  - Fixed batch size per graph (one graph per bucket).

Sampling lives OUTSIDE the graph: replay returns logits, the engine
samples per-request with multinomial / .item().
"""

from __future__ import annotations

from typing import Callable

import torch


class CudaGraphRunner:
    """Captures and replays paged decode at bucket batch sizes."""

    def __init__(
        self,
        decode_fn: Callable,
        kv_pool,
        vocab_size: int,
        dtype: torch.dtype,
        device: str,
        bucket_batch_sizes: list[int],
        max_blocks: int,
        scratch_page_idx: int,
        warmup_iters: int = 3,
    ):
        if not bucket_batch_sizes:
            raise ValueError("bucket_batch_sizes must be non-empty")

        self.decode_fn = decode_fn
        self.kv_pool = kv_pool
        self.bucket_batch_sizes = sorted(set(bucket_batch_sizes))
        self.max_bs = self.bucket_batch_sizes[-1]
        self.max_blocks = max_blocks
        self.scratch_page_idx = scratch_page_idx
        self.device = device

        # Static input/output buffers — same address forever.
        self.input_ids = torch.zeros((self.max_bs, 1), dtype=torch.long, device=device)
        self.position_ids = torch.zeros((self.max_bs, 1), dtype=torch.long, device=device)
        self.cache_seqlens = torch.zeros((self.max_bs,), dtype=torch.int32, device=device)
        # Pre-fill block_table with scratch — any unused slot stays harmless.
        self.block_table = torch.full(
            (self.max_bs, max_blocks),
            scratch_page_idx,
            dtype=torch.int32,
            device=device,
        )
        self.logits_buf = torch.zeros((self.max_bs, vocab_size), dtype=dtype, device=device)

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._capture_all(warmup_iters=warmup_iters)

    # ── Capture ────────────────────────────────────────────────────────

    def _capture_all(self, warmup_iters: int) -> None:
        """Run warmup forwards, then capture one graph per bucket size."""
        # Pre-warm the RoPE cos/sin cache to the largest position any
        # replay can ever hit. _lookup_rope cannot grow the cache while
        # capturing (would need `.item()` = CPU↔GPU sync, banned in
        # capture mode), so we force the growth here while still in
        # eager mode. 65,536 covers any reasonable workload — well
        # above the bench harness's --input-len 1024 + --output-len 512
        # and even comfortably above page_size * max_blocks.
        PREWARM_MAX_POS = 65536
        self.position_ids[0, 0] = PREWARM_MAX_POS - 1
        _ = self.decode_fn(
            input_ids=self.input_ids[: self.max_bs],
            position_ids=self.position_ids[: self.max_bs],
            cache_seqlens=self.cache_seqlens[: self.max_bs],
            block_table=self.block_table[: self.max_bs],
            kv_pool=self.kv_pool,
        )
        self.position_ids[0, 0] = 0  # restore

        # Warmup at EVERY bucket size, not just the largest. With
        # --torch-compile, dynamo's shape-specialization cache is keyed
        # on input shape; if we only warm up at max_bs, the FIRST call
        # at each smaller bucket inside `torch.cuda.graph(...)` will
        # trigger a recompile — and recompile internally calls
        # torch.cuda.get_rng_state(), which is banned during capture:
        #
        #   RuntimeError: Cannot call CUDAGeneratorImpl::current_seed
        #   during CUDA graph capture.
        #
        # Pre-warming at each bucket forces dynamo to compile + cache
        # all shapes in eager mode, so the capture-time forwards just
        # replay cached compiled code with zero new compiles.
        for bs in self.bucket_batch_sizes:
            for _ in range(warmup_iters):
                _ = self.decode_fn(
                    input_ids=self.input_ids[:bs],
                    position_ids=self.position_ids[:bs],
                    cache_seqlens=self.cache_seqlens[:bs],
                    block_table=self.block_table[:bs],
                    kv_pool=self.kv_pool,
                )
        torch.cuda.synchronize()

        for bs in self.bucket_batch_sizes:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits = self.decode_fn(
                    input_ids=self.input_ids[:bs],
                    position_ids=self.position_ids[:bs],
                    cache_seqlens=self.cache_seqlens[:bs],
                    block_table=self.block_table[:bs],
                    kv_pool=self.kv_pool,
                )
                # Copy into the persistent logits buffer so callers can
                # read it without holding the per-replay output tensor.
                self.logits_buf[:bs].copy_(logits)
            self.graphs[bs] = graph
        torch.cuda.synchronize()

    # ── Replay ─────────────────────────────────────────────────────────

    def _bucket_for(self, batch_size: int) -> int:
        for b in self.bucket_batch_sizes:
            if b >= batch_size:
                return b
        raise RuntimeError(
            f"batch size {batch_size} exceeds max captured bucket {self.max_bs}; "
            f"increase --cuda-graph-batch-sizes"
        )

    def replay(
        self,
        input_ids: torch.Tensor,        # (B, 1)
        position_ids: torch.Tensor,     # (B, 1)
        cache_seqlens: torch.Tensor,    # (B,)
        block_table: torch.Tensor,      # (B, blocks)
    ) -> torch.Tensor:
        """Run paged decode by replaying the bucketed graph. Returns
        logits of shape (B, vocab_size)."""
        B = input_ids.shape[0]
        bucket = self._bucket_for(B)

        bt_blocks = block_table.shape[1]
        if bt_blocks > self.max_blocks:
            raise RuntimeError(
                f"request page table has {bt_blocks} blocks but runner was "
                f"built for max {self.max_blocks}. Bump CudaGraphRunner.max_blocks."
            )

        # Copy the live batch into the static buffers.
        self.input_ids[:B].copy_(input_ids)
        self.position_ids[:B].copy_(position_ids)
        self.cache_seqlens[:B].copy_(cache_seqlens)
        self.block_table[:B, :bt_blocks].copy_(block_table)
        # Slots [bt_blocks, max_blocks) for live rows: leave scratch
        # default — flash-attn won't read past cache_seqlens anyway.
        if bt_blocks < self.max_blocks:
            self.block_table[:B, bt_blocks:].fill_(self.scratch_page_idx)

        # Pad rows [B, bucket): cache_seqlens=0 + block_table=scratch
        # makes the kernel read no real data and write to scratch only.
        if B < bucket:
            self.cache_seqlens[B:bucket].fill_(0)
            self.block_table[B:bucket].fill_(self.scratch_page_idx)
            # input_ids / position_ids for pad rows can stay anything;
            # their compute is discarded.

        self.graphs[bucket].replay()
        # Clone the live slice so callers can mutate without aliasing
        # the static buffer (the next replay overwrites it).
        return self.logits_buf[:B].clone()
