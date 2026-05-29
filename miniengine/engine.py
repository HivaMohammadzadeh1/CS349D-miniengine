"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Two decode paths:
  - decode_step(req)        : one request, used by baseline scheduler
  - batched_decode(reqs)    : many requests, one forward pass with padded
                              KV + attention mask, used by batched mode

Prefill stays per-request — variable prompt lengths make batched prefill
complex, and decode is where the throughput gain lives.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.cpu_kv_pool import CpuKvPool
from miniengine.cuda_graph_runner import CudaGraphRunner
from miniengine.kv_memory_pool import KVMemoryPool, KVOutOfMemory
from miniengine.model import CausalLM, ModelConfig, load_weights
from miniengine.paged_model import PagedCausalLM
from miniengine.radix_cache import RadixCache
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


class Engine:
    """Model wrapper supporting baseline (per-request) and batched decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        mode: str = "batched",
        page_size: int = 32,
        mem_fraction_static: float = 0.85,
        torch_compile: bool = False,
        cuda_graph: bool = False,
        cuda_graph_batch_sizes: list[int] | None = None,
        cuda_graph_max_blocks: int = 256,
        disable_radix_cache: bool = False,
        cpu_cache_size_gb: float = 0.0,
        hicache_overlap: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.page_size = page_size
        self.mem_fraction_static = mem_fraction_static
        self.torch_compile_enabled = torch_compile
        self.cuda_graph_enabled = cuda_graph
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes or [1, 2, 4, 8, 16, 32]
        self.cuda_graph_max_blocks = cuda_graph_max_blocks
        self.disable_radix_cache = disable_radix_cache
        # ── Milestone 4 (HiCache) ──────────────────────────────────────────
        # 0.0 disables HiCache entirely; the cache stays GPU-only and behavior
        # is byte-identical to milestone 3. Overlap flag is plumbed through but
        # the async copy stream is wired up in cpu_kv_pool/radix_cache task #11.
        self.cpu_cache_size_gb = cpu_cache_size_gb
        self.hicache_overlap = hicache_overlap
        self.kv_pool: KVMemoryPool | None = None
        self.cpu_kv_pool: CpuKvPool | None = None
        self.radix_cache: RadixCache | None = None
        self.cuda_graph_runner: CudaGraphRunner | None = None
        self.scratch_page_idx: int | None = None

        # ── Tokenizer (still from HF — it's just a tokenizer) ──────────
        logger.info("Loading tokenizer from %s …", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ── Model (bare-bone PyTorch, loaded from safetensors) ──────────
        logger.info("Loading model config from %s …", model_path)
        config = ModelConfig.from_pretrained(model_path)
        logger.info(
            "Config: layers=%d, hidden=%d, heads=%d, kv_heads=%d, head_dim=%d, "
            "intermediate=%d, vocab=%d, tie_embed=%s",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.vocab_size,
            config.tie_word_embeddings,
        )

        # Build on meta device — load_weights replaces parameters with
        # GPU tensors directly, so we never allocate a CPU fp32 copy.
        # Paged mode uses PagedCausalLM (flash-attn paged kernels);
        # baseline/batched modes use the M1 CausalLM. Both expose the
        # same parameter hierarchy so the same checkpoint loads into
        # either one.
        with torch.device("meta"):
            if self.mode == "paged":
                self.model = PagedCausalLM(config)
            else:
                self.model = CausalLM(config)
        load_weights(self.model, model_path, dtype=dtype, device=device)
        self.model.eval()
        self.config = config

        # ── Paged-mode setup: build KV pool, optional torch.compile ────
        if self.mode == "paged":
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            weights_bytes = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            budget = int(total_mem * mem_fraction_static) - weights_bytes
            if budget <= 0:
                raise RuntimeError(
                    f"mem-fraction-static={mem_fraction_static} leaves no room "
                    f"for the KV pool after weights ({weights_bytes/1e9:.2f}GB)."
                )
            self.kv_pool = KVMemoryPool.from_budget(
                num_layers=config.num_hidden_layers,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                page_size=page_size,
                dtype=dtype,
                device=device,
                bytes_budget=budget,
            )
            logger.info(
                "KV pool: %d pages × %d tokens (%.2f GB)  free=%d",
                self.kv_pool.num_pages,
                page_size,
                budget / 1e9,
                self.kv_pool.num_free,
            )

            # Milestone 3: attach radix cache (unless explicitly disabled).
            # Milestone 4: optionally build a CPU-tier pool (HiCache).
            if not self.disable_radix_cache:
                if self.cpu_cache_size_gb > 0:
                    cpu_bytes = int(self.cpu_cache_size_gb * 1e9)
                    self.cpu_kv_pool = CpuKvPool.from_budget(
                        num_layers=config.num_hidden_layers,
                        num_kv_heads=config.num_key_value_heads,
                        head_dim=config.head_dim,
                        page_size=page_size,
                        dtype=dtype,
                        bytes_budget=cpu_bytes,
                    )
                    logger.info(
                        "HiCache: CPU KV tier %d slots × %d tokens (%.2f GB)  "
                        "ratio cpu/gpu=%.1fx  pinned=%s  overlap=%s",
                        self.cpu_kv_pool.num_pages,
                        page_size,
                        cpu_bytes / 1e9,
                        self.cpu_kv_pool.num_pages / max(1, self.kv_pool.num_pages),
                        self.cpu_kv_pool.is_pinned,
                        self.hicache_overlap,
                    )
                self.radix_cache = RadixCache(self.kv_pool, cpu_pool=self.cpu_kv_pool)
                self.kv_pool.attach_cache(self.radix_cache)
                logger.info(
                    "Radix prefix cache enabled%s.",
                    " (HiCache GPU+CPU)" if self.cpu_kv_pool is not None else "",
                )
            else:
                if self.cpu_cache_size_gb > 0:
                    raise RuntimeError(
                        "--cpu-cache-size-gb requires the radix cache; "
                        "remove --disable-radix-cache."
                    )
                logger.info("Radix prefix cache DISABLED (--disable-radix-cache).")

            if self.torch_compile_enabled:
                # Compile the DECODE-path MLP only — stable shape per
                # token (batch, 1, hidden), batch is dynamic. The MLP is
                # the largest stable-shape compute chunk in decode;
                # attention has dynamic seq lengths and goes to
                # flash-attn anyway.
                #
                # Critical: we install the compiled module on
                # ``layer.mlp_decode``, NOT ``layer.mlp``. forward_prefill
                # keeps calling the eager ``layer.mlp`` because varlen
                # prefill has unstable input shape (total_packed_tokens
                # varies per batch composition), which would otherwise
                # trigger per-batch recompiles and regress throughput
                # (see report §3.3.3 for the old all-paths-compiled
                # results). forward_decode calls ``layer.mlp_decode``.
                #
                # When stacking with manual CUDA-graph capture we use
                # mode="default" to avoid nested cudagraph capture; the
                # outer manual graph then wraps the compiled kernels.
                compile_mode = "default" if self.cuda_graph_enabled else "reduce-overhead"
                logger.info(
                    "Compiling per-layer decode MLP with torch.compile "
                    "(mode=%s) — prefill stays eager.", compile_mode
                )
                for layer in self.model.model.layers:
                    if hasattr(layer, "mlp_decode"):
                        # Paged path: compile only the decode-side handle,
                        # leave the shared eager ``layer.mlp`` for prefill.
                        layer.mlp_decode = torch.compile(
                            layer.mlp, mode=compile_mode, dynamic=True
                        )
                    else:
                        # Batched (M1) path has only one MLP handle; the
                        # forward is shape-stable enough that compiling
                        # in place is fine.
                        layer.mlp = torch.compile(
                            layer.mlp, mode=compile_mode, dynamic=True
                        )

            if self.cuda_graph_enabled:
                # Reserve a scratch page so padded batch entries during
                # graph replay write garbage into a known-safe slot.
                self.scratch_page_idx = self.kv_pool.allocate(1)[0]
                logger.info(
                    "CUDA-graph mode: reserved scratch page %d, capturing buckets %s "
                    "with max_blocks=%d …",
                    self.scratch_page_idx,
                    self.cuda_graph_batch_sizes,
                    self.cuda_graph_max_blocks,
                )
                self.cuda_graph_runner = CudaGraphRunner(
                    decode_fn=self.model.decode,
                    kv_pool=self.kv_pool,
                    vocab_size=config.vocab_size,
                    dtype=dtype,
                    device=device,
                    bucket_batch_sizes=self.cuda_graph_batch_sizes,
                    max_blocks=self.cuda_graph_max_blocks,
                    scratch_page_idx=self.scratch_page_idx,
                )
                logger.info("CUDA-graph capture complete.")

        # ── Stop tokens ─────────────────────────────────────────────────
        self.stop_token_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok_name in ("eos_token", "pad_token"):
            tid = getattr(self.tokenizer, f"{tok_name}_id", None)
            if tid is not None:
                self.stop_token_ids.add(tid)
        for token_str in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tid)

        logger.info(
            "Engine ready  —  vocab=%d, stop_ids=%s, params=%dM",
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the model's chat template and tokenize into ids."""
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 models support enable_thinking; silently ignore if unsupported
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id back to a string."""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    # ── Forward passes ──────────────────────────────────────────────────

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """
        Run the prefill phase for one request.

        Processes the full prompt in a single forward pass, stores the
        resulting KV cache on the request, and samples the first output
        token.

        Returns:
            The first generated token id.
        """
        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)
        request.kv_cache = kv_caches

        # Sample from the last position
        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """
        Run one decode step for a request that has already been prefilled.

        Feeds the last generated token through the model together with the
        cached KV values, updates the cache, and samples the next token.

        Returns:
            The next generated token id.
        """
        input_ids = torch.tensor(
            [[request.output_ids[-1]]], dtype=torch.long, device=self.device
        )
        # Position = current KV cache length (= num tokens already processed)
        cache_len = request.kv_cache[0][0].shape[2]  # layer 0, key tensor, seq dim
        position_ids = torch.tensor([[cache_len]], device=self.device)

        logits, kv_caches = self.model(
            input_ids, position_ids, kv_caches=request.kv_cache
        )
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Run one decode step for multiple requests in a single batched forward pass.

        Pads KV caches to the max cache length, builds an attention mask to
        ignore padding, runs one forward pass, then extracts per-request KV
        caches (removing padding) and samples one token per request.

        Returns:
            List of next token ids, one per request.
        """
        batch_size = len(requests)

        # 1. Stack last tokens → (batch, 1)
        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long, device=self.device,
        )

        # 2. Cache lengths per request
        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        max_cache_len = max(cache_lens)

        # 3. Position IDs: each request's position = its own cache length
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens], device=self.device,
        )

        # 4. Pad KV caches per layer and stack into batched tensors
        num_layers = len(requests[0].kv_cache)
        padded_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            keys = []
            values = []
            for req_idx, req in enumerate(requests):
                k, v = req.kv_cache[layer_idx]
                # k, v: (1, kv_heads, cache_len_i, head_dim)
                pad_len = max_cache_len - cache_lens[req_idx]
                if pad_len > 0:
                    k = F.pad(k, (0, 0, 0, pad_len))  # pad seq dim on right
                    v = F.pad(v, (0, 0, 0, pad_len))
                keys.append(k)
                values.append(v)
            padded_kv_caches.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        # 5. Attention mask: (batch, 1, 1, max_cache_len + 1)
        #    After model concatenates new K/V, total KV length = max_cache_len + 1.
        #    Mask padding positions between each request's actual cache end and max_cache_len.
        total_kv_len = max_cache_len + 1
        attn_mask = torch.zeros(
            (batch_size, 1, 1, total_kv_len), dtype=self.dtype, device=self.device,
        )
        for i, cl in enumerate(cache_lens):
            if cl < max_cache_len:
                attn_mask[i, 0, 0, cl:max_cache_len] = float("-inf")

        # 6. Batched forward pass
        logits, new_kv_caches = self.model(
            input_ids, position_ids, kv_caches=padded_kv_caches, attention_mask=attn_mask,
        )

        # 7. Extract per-request KV caches (strip padding from the middle)
        for req_idx, req in enumerate(requests):
            L = cache_lens[req_idx]
            req_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer_idx in range(num_layers):
                k, v = new_kv_caches[layer_idx]
                # k: (batch, kv_heads, max_cache_len+1, head_dim)
                # Keep [:L] (old actual) + [-1:] (new token), skip padding
                old_k = k[req_idx : req_idx + 1, :, :L, :]
                new_k = k[req_idx : req_idx + 1, :, -1:, :]
                old_v = v[req_idx : req_idx + 1, :, :L, :]
                new_v = v[req_idx : req_idx + 1, :, -1:, :]
                req_kv.append((
                    torch.cat([old_k, new_k], dim=2),
                    torch.cat([old_v, new_v], dim=2),
                ))
            req.kv_cache = req_kv

        # 8. Sample per request
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            token_id = sample_token(
                logits[i : i + 1, -1, :], req.sampling_params, req.output_ids,
            )
            token_ids.append(token_id)

        return token_ids

    # ── Paged forward passes (milestone 2) ──────────────────────────────

    @property
    def pool(self) -> KVMemoryPool | None:
        """Alias for ``kv_pool`` — server.py and bench utilities use this name."""
        return self.kv_pool

    def _slot_for(self, page_idx: int, slot_in_page: int) -> int:
        """Compute the flat slot index in the pool's per-layer K/V tensor."""
        return page_idx * self.page_size + slot_in_page

    # ── Prefill setup (cache lookup + lazy prompt-page alloc) ──────────

    def _setup_paged_request(self, req: Request) -> int:
        """Per-request bookkeeping run once at prefill start (milestone 3).

        * If a radix cache is attached, look up the longest page-aligned
          prefix of ``req.input_ids``. The matched pages are pinned (lock-ref
          ++) and reused as the head of the request's page_table — no need
          to recompute or rewrite their K/V.
        * Allocate pool pages only for the *uncached* portion of the prompt.
          No reservation for ``max_new_tokens`` (lazy alloc — decode appends
          pages as ``cache_len`` crosses page boundaries).
        * Record ``cache_hit_tokens`` for the per-request usage block.

        On ``KVOutOfMemory`` the function unwinds (drops the lock ref it
        just took, clears ``matched_node``/``cache_hit_tokens``) so the
        caller sees the request in its pre-call state and can re-queue it.

        Returns ``matched_tokens`` (0 if no cache hit / cache disabled).
        """
        assert self.kv_pool is not None
        matched_pages: list[int] = []
        matched_tokens = 0
        if self.radix_cache is not None:
            match = self.radix_cache.match_prefix(req.input_ids)
            # HiCache: lift any CPU-tier nodes on the matched path back to
            # GPU. No-op when cpu_pool is None or every matched node is
            # already GPU-tier (so the m3 cold path is unchanged). May
            # raise KVOutOfMemory if even after demoting other cold nodes
            # we can't find GPU pages for the promotion — the existing
            # outer try/except handles unwind safely (matched_node not yet
            # set at this point, so no lock leaks).
            self.radix_cache.promote_match(match)
            matched_pages = list(match.matched_pages)
            matched_tokens = match.matched_tokens
            self.radix_cache.inc_lock_ref(match.last_node)
            req.matched_node = match.last_node

        req.cache_hit_tokens = matched_tokens
        remaining_prompt = req.num_input_tokens - matched_tokens
        n_new_pages = self.kv_pool.pages_needed(remaining_prompt)
        try:
            new_pages = (
                self.kv_pool.allocate(n_new_pages) if n_new_pages > 0 else []
            )
        except KVOutOfMemory:
            # Unwind the lock we just took before re-raising so caller can
            # safely re-queue this request without leaking a pin.
            if self.radix_cache is not None and req.matched_node is not None:
                self.radix_cache.dec_lock_ref(req.matched_node)
                req.matched_node = None
            req.cache_hit_tokens = 0
            raise

        req.page_table = matched_pages + new_pages
        req.cache_len = matched_tokens
        req.prefill_offset = matched_tokens
        return matched_tokens

    def _ensure_decode_page(self, req: Request) -> None:
        """Lazy-alloc one more page if the next decode-token slot is out
        of range. Raises ``KVOutOfMemory`` if the pool can't grow.
        """
        next_slot_page = req.cache_len // self.page_size
        while next_slot_page >= len(req.page_table):
            new_page = self.kv_pool.allocate(1)[0]
            req.page_table.append(new_page)

    @torch.inference_mode()
    def paged_prefill_batch(self, requests: list[Request]) -> list[int]:
        """
        Packed prefill of N requests in a single forward pass.

        Lazy-alloc: only allocates pages for the prompt (decode appends).
        Cache-aware: uses block_table + cu_seqlens_k so cache-hit prefixes
        are read straight from the pool — only the uncached suffix gets a
        forward pass. When no requests hit the cache, this degrades to a
        full-prompt packed varlen prefill.
        """
        assert self.kv_pool is not None, "paged_prefill_batch requires --mode paged"
        if not requests:
            return []

        page_size = self.page_size

        # ── 1. Cache lookup + lazy prompt-page allocation ─────────────
        # If any request's allocation fails, unwind the earlier ones so
        # the scheduler can re-queue the batch with consistent state.
        setups_completed: list[Request] = []
        try:
            for req in requests:
                self._setup_paged_request(req)
                setups_completed.append(req)
        except KVOutOfMemory:
            for r in setups_completed:
                self.retract_paged_request(r)
            raise

        # If every request hits the cache fully (matched_tokens == prompt_len),
        # there's nothing to prefill — we still need a logits pass over the
        # last token to sample the first generated token, so we fall through.

        # ── 2. Build packed tensors over the UNCACHED suffix of each req ──
        seq_lens_full = [req.num_input_tokens for req in requests]   # prompt length
        seq_lens_q = [
            max(1, req.num_input_tokens - req.prefill_offset)
            for req in requests
        ]
        # Edge case: full cache hit. We still need 1 query token at the
        # last prompt position to get the logits. We "redo" only that last
        # token's forward pass — its K/V will overwrite the cached page slot
        # which is fine because the K/V is identical (same input, same RoPE).
        offsets = [
            req.prefill_offset if req.prefill_offset < req.num_input_tokens
            else req.num_input_tokens - 1
            for req in requests
        ]

        packed_ids: list[int] = []
        packed_pos: list[int] = []
        slot_mapping_list: list[int] = []
        cu_seqlens_q_list = [0]
        cu_seqlens_k_list = [0]
        last_token_indices_list: list[int] = []
        cumulative_q = 0

        for req, full_len, off in zip(requests, seq_lens_full, offsets):
            # q tokens: input_ids[off : full_len]
            for t in range(off, full_len):
                packed_ids.append(req.input_ids[t])
                packed_pos.append(t)
                page_idx = req.page_table[t // page_size]
                slot_in_page = t % page_size
                slot_mapping_list.append(self._slot_for(page_idx, slot_in_page))
            q_len = full_len - off
            cumulative_q += q_len
            cu_seqlens_q_list.append(cumulative_q)
            # k spans the FULL prompt (0..full_len) for this request.
            cu_seqlens_k_list.append(cu_seqlens_k_list[-1] + full_len)
            last_token_indices_list.append(cumulative_q - 1)

        max_seqlen_q = max(seq_lens_q)
        max_seqlen_k = max(seq_lens_full)
        max_blocks = max(len(req.page_table) for req in requests)

        input_ids = torch.tensor(packed_ids, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(packed_pos, dtype=torch.long, device=self.device)
        cu_seqlens_q = torch.tensor(cu_seqlens_q_list, dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=self.device)
        last_token_indices = torch.tensor(
            last_token_indices_list, dtype=torch.long, device=self.device
        )

        block_table = torch.zeros(
            (len(requests), max_blocks), dtype=torch.int32, device=self.device
        )
        for i, req in enumerate(requests):
            pt = req.page_table
            block_table[i, : len(pt)] = torch.tensor(
                pt, dtype=torch.int32, device=self.device
            )

        # ── 3. Forward pass ──────────────────────────────────────────
        logits = self.model.prefill_chunked(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_table=block_table,
            last_token_indices=last_token_indices,
            kv_pool=self.kv_pool,
        )  # (batch, vocab)

        # ── 4. Sample first token; advance state; insert prefix into cache ─
        token_ids: list[int] = []
        for i, (req, L) in enumerate(zip(requests, seq_lens_full)):
            req.cache_len = L
            req.prefill_offset = L
            tok = sample_token(
                logits[i : i + 1, :], req.sampling_params, req.output_ids
            )
            token_ids.append(tok)

        # Insert prompt-aligned prefix into the radix cache so subsequent
        # requests can hit it. We do this AFTER all requests in the batch
        # have completed prefill so concurrent batchmates don't fight over
        # who owns the prefix.
        self._insert_prompt_into_cache(requests)

        return token_ids

    @torch.inference_mode()
    def paged_prefill_chunk(self, req: Request, chunk_size: int) -> int | None:
        """One chunked-prefill step for a single request.

        Returns the first generated token on the LAST chunk, ``None``
        otherwise. The caller is expected to have called
        ``_setup_paged_request(req)`` previously (e.g. via
        ``start_paged_prefill``), so ``req.page_table``,
        ``req.prefill_offset``, and ``req.cache_len`` are already wired up.
        """
        assert self.kv_pool is not None
        assert chunk_size > 0

        page_size = self.page_size
        off = req.prefill_offset
        full_len = req.num_input_tokens
        if off >= full_len:
            # Already done. Should not happen in normal scheduler flow.
            return None

        # Edge: cache-hit-only request. We must still run 1 forward pass
        # over the last prompt token to obtain logits for the first sample.
        if off == full_len - 0:  # never true; placeholder to keep symmetry
            pass

        end = min(off + chunk_size, full_len)
        is_final = end == full_len

        # Build packed tensors (single request, B=1).
        packed_ids: list[int] = list(req.input_ids[off:end])
        packed_pos: list[int] = list(range(off, end))
        slot_mapping_list: list[int] = []
        for t in range(off, end):
            page_idx = req.page_table[t // page_size]
            slot_in_page = t % page_size
            slot_mapping_list.append(self._slot_for(page_idx, slot_in_page))

        q_len = end - off
        cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, end], dtype=torch.int32, device=self.device)
        input_ids = torch.tensor(packed_ids, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(packed_pos, dtype=torch.long, device=self.device)
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=self.device)
        block_table = torch.tensor(
            [req.page_table], dtype=torch.int32, device=self.device
        )

        last_token_indices = (
            torch.tensor([q_len - 1], dtype=torch.long, device=self.device)
            if is_final
            else None
        )

        result = self.model.prefill_chunked(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_len,
            max_seqlen_k=end,
            slot_mapping=slot_mapping,
            block_table=block_table,
            last_token_indices=last_token_indices,
            kv_pool=self.kv_pool,
        )

        req.prefill_offset = end
        req.cache_len = end

        if not is_final:
            return None

        # Final chunk: result is (1, vocab).
        tok = sample_token(result[0:1, :], req.sampling_params, req.output_ids)
        # Now that the prompt is fully prefilled, insert into the cache.
        self._insert_prompt_into_cache([req])
        return tok

    def start_paged_prefill(self, req: Request) -> int:
        """Set up a chunked prefill: cache lookup, allocate prompt pages.

        Returns the number of cache-hit tokens (also stored on the request).
        After this call, the caller should drive
        ``paged_prefill_chunk(req, chunk_size)`` repeatedly until it returns
        a non-None token (the first generated token).
        """
        return self._setup_paged_request(req)

    def _insert_prompt_into_cache(self, requests: list[Request]) -> None:
        """Insert the page-aligned prefix of each request's prompt into the
        radix cache and free any redundant pages back to the pool."""
        if self.radix_cache is None:
            return
        ps = self.page_size
        for req in requests:
            aligned = req.num_input_tokens - (req.num_input_tokens % ps)
            if aligned == 0:
                continue
            n_pages = aligned // ps
            prefix_tokens = req.input_ids[:aligned]
            prefix_pages = req.page_table[:n_pages]
            _leaf, redundant = self.radix_cache.insert_and_return(
                prefix_tokens, prefix_pages
            )
            if redundant:
                # Another concurrent request beat us to caching this prefix.
                # The cache's pages win; ours go back to the pool. We do NOT
                # rewrite req.page_table — both copies contain the same K/V
                # (deterministic forward), so the request can keep decoding
                # against its own pages until it finishes.
                self.kv_pool.free(redundant)

    @torch.inference_mode()
    def paged_decode_step(self, requests: list[Request]) -> list[int]:
        """One decode step for a batch of running requests using paged KV.

        Lazy-alloc (milestone 3): each request grows its page_table by one
        page when ``cache_len`` is about to cross a page boundary. This
        avoids reserving worst-case ``max_new_tokens`` pages at prefill,
        making the cache effective, but means the call may raise
        ``KVOutOfMemory`` mid-step. The scheduler catches that and retracts.
        """
        assert self.kv_pool is not None, "paged_decode_step requires --mode paged"
        if not requests:
            return []

        page_size = self.page_size
        batch_size = len(requests)

        # Lazy allocation: grow the page_table of any request whose next
        # token slot lands on an unallocated page. This is the only place
        # decode-time pool pressure can manifest as OOM.
        for req in requests:
            self._ensure_decode_page(req)

        max_blocks = max(len(req.page_table) for req in requests)

        # ── Build per-step tensors ─────────────────────────────────────
        last_tokens = [[req.output_ids[-1]] for req in requests]
        cache_lens = [req.cache_len for req in requests]
        position_ids_list = [[cl] for cl in cache_lens]

        # Block table: (batch, max_blocks) int32. Padded with 0s; flash-attn
        # bounds reads by cache_seqlens so the padding is never indexed.
        block_table = torch.zeros(
            (batch_size, max_blocks), dtype=torch.int32, device=self.device
        )
        for i, req in enumerate(requests):
            pt = req.page_table
            block_table[i, : len(pt)] = torch.tensor(pt, dtype=torch.int32, device=self.device)

        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(position_ids_list, dtype=torch.long, device=self.device)
        cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device=self.device)

        if self.cuda_graph_runner is not None:
            logits = self.cuda_graph_runner.replay(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
            )  # (batch, vocab)
        else:
            logits = self.model.decode(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                kv_pool=self.kv_pool,
            )  # (batch, vocab)

        # ── Sample, advance cache_len ──────────────────────────────────
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            tok = sample_token(logits[i : i + 1, :], req.sampling_params, req.output_ids)
            req.cache_len += 1
            token_ids.append(tok)
        return token_ids

    def free_paged_request(self, req: Request) -> None:
        """Release a finished request's resources.

        Cache-aware (milestone 3):
          * Decrement the lock-ref on the matched node (the cached prefix
            this request was borrowing).
          * Insert the request's full sequence (prompt + output) into the
            radix cache at page-aligned granularity so future multi-turn
            requests can hit on the assistant response.
          * Free pages that turn out to be redundant (after insert) and the
            unaligned tail back to the pool.
        """
        if self.kv_pool is None or req.page_table is None:
            return

        if self.radix_cache is not None and req.matched_node is not None:
            self.radix_cache.dec_lock_ref(req.matched_node)
            req.matched_node = None

        ps = self.page_size
        if self.radix_cache is not None:
            full_tokens = req.input_ids + req.output_ids
            aligned = len(full_tokens) - (len(full_tokens) % ps)
            if aligned > 0:
                full_pages = req.page_table[: aligned // ps]
                _leaf, redundant = self.radix_cache.insert_and_return(
                    full_tokens[:aligned], full_pages
                )
                if redundant:
                    self.kv_pool.free(redundant)
                # Unaligned tail (partial last page) can't be cached.
                tail_pages = req.page_table[aligned // ps :]
                if tail_pages:
                    self.kv_pool.free(tail_pages)
            else:
                # No page-aligned content (shouldn't happen with non-empty
                # prompt, but defend) — just dump everything back.
                self.kv_pool.free(req.page_table)
        else:
            self.kv_pool.free(req.page_table)

        req.page_table = None
        req.cache_len = 0
        req.prefill_offset = 0

    def retract_paged_request(self, req: Request) -> None:
        """Free a victim's pages without caching its (partial, in-flight)
        output. Used by the scheduler's retraction loop *and* by
        ``paged_prefill_batch`` to unwind partially-set-up admissions on
        mid-batch ``KVOutOfMemory``.
        """
        # Always drop the lock-ref first — works even if page_table is
        # None (setup failed before alloc).
        if self.radix_cache is not None and req.matched_node is not None:
            self.radix_cache.dec_lock_ref(req.matched_node)
            req.matched_node = None
        if self.kv_pool is not None and req.page_table is not None:
            # Note: we do not insert anything into the cache — the request
            # didn't finish, so its output isn't authoritative. Pages just
            # return to the pool.
            self.kv_pool.free(req.page_table)
        req.page_table = None
        req.cache_len = 0
        req.prefill_offset = 0
        req.cache_hit_tokens = 0

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids

    # ── Batched decode ──────────────────────────────────────────────────

    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Decode one token for each request in a single forward pass.

        Pads per-request KV caches to the longest in the batch, builds a
        float attention mask that ignores padding, runs the model once,
        then extracts each request's actual KV (real prefix + new token)
        and samples its next token.
        """
        if not requests:
            return []

        batch_size = len(requests)
        num_layers = len(requests[0].kv_cache)

        # Stack last generated token from each request → (batch, 1)
        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=self.device,
        )

        # Each request's current KV length and the per-request RoPE position
        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        max_cache_len = max(cache_lens)
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens],
            dtype=torch.long,
            device=self.device,
        )

        # Pad and stack KV caches per layer to (batch, kv_heads, max_cache_len, head_dim)
        padded_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_list, v_list = [], []
            for req in requests:
                k, v = req.kv_cache[layer_idx]
                pad_len = max_cache_len - k.shape[2]
                if pad_len > 0:
                    k = F.pad(k, (0, 0, 0, pad_len))
                    v = F.pad(v, (0, 0, 0, pad_len))
                k_list.append(k)
                v_list.append(v)
            padded_kv_caches.append(
                (torch.cat(k_list, dim=0), torch.cat(v_list, dim=0))
            )

        # Mask shape (batch, 1, 1, max_cache_len + 1): the attention forward
        # appends the new token to the cache, so kv_len = max_cache_len + 1.
        # Mask only the padding window [cl, max_cache_len) per request.
        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            max_cache_len + 1,
            device=self.device,
            dtype=self.dtype,
        )
        for i, cl in enumerate(cache_lens):
            attention_mask[i, 0, 0, cl:max_cache_len] = float("-inf")

        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=padded_kv_caches,
            attention_mask=attention_mask,
        )

        # Extract each request's real KV (actual prefix + new token at -1).
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            cl = cache_lens[i]
            per_req_kv = []
            for layer_idx in range(num_layers):
                k_full = new_kv_caches[layer_idx][0][i : i + 1]
                v_full = new_kv_caches[layer_idx][1][i : i + 1]
                k_new = torch.cat([k_full[:, :, :cl, :], k_full[:, :, -1:, :]], dim=2)
                v_new = torch.cat([v_full[:, :, :cl, :], v_full[:, :, -1:, :]], dim=2)
                per_req_kv.append((k_new, v_new))
            req.kv_cache = per_req_kv
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids
