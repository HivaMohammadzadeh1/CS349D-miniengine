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
from miniengine.cuda_graph_runner import CudaGraphRunner
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import CausalLM, ModelConfig, load_weights
from miniengine.paged_model import PagedCausalLM
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
        self.kv_pool: KVMemoryPool | None = None
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

    def _slot_for(self, page_idx: int, slot_in_page: int) -> int:
        """Compute the flat slot index in the pool's per-layer K/V tensor."""
        return page_idx * self.page_size + slot_in_page

    @torch.inference_mode()
    def paged_prefill_batch(self, requests: list[Request]) -> list[int]:
        """
        Packed prefill of N requests in a single forward pass.

        For each request: allocates worst-case pages (prompt_len + max_new_tokens)
        from the pool, sets request.page_table, writes K/V into the pool, and
        samples the first generated token.

        Returns the first generated token id for each request.
        """
        assert self.kv_pool is not None, "paged_prefill_batch requires --mode paged"
        if not requests:
            return []

        page_size = self.page_size

        # ── Allocate pages (worst-case) and reserve page-table slots ───
        for req in requests:
            worst_case = req.num_input_tokens + req.sampling_params.max_new_tokens
            n_pages = self.kv_pool.pages_needed(worst_case)
            req.page_table = self.kv_pool.allocate(n_pages)
            req.cache_len = 0  # will become prompt_len after this prefill

        # ── Build packed tensors ───────────────────────────────────────
        seq_lens = [req.num_input_tokens for req in requests]
        total_tokens = sum(seq_lens)
        max_seqlen = max(seq_lens)

        packed_ids: list[int] = []
        packed_pos: list[int] = []
        slot_mapping_list: list[int] = []
        cu_seqlens_list = [0]
        last_token_indices_list: list[int] = []
        cumulative = 0

        for req, L in zip(requests, seq_lens):
            packed_ids.extend(req.input_ids)
            packed_pos.extend(range(L))
            for t in range(L):
                page_idx = req.page_table[t // page_size]
                slot_in_page = t % page_size
                slot_mapping_list.append(self._slot_for(page_idx, slot_in_page))
            cumulative += L
            cu_seqlens_list.append(cumulative)
            last_token_indices_list.append(cumulative - 1)

        input_ids = torch.tensor(packed_ids, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(packed_pos, dtype=torch.long, device=self.device)
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=self.device)
        last_token_indices = torch.tensor(
            last_token_indices_list, dtype=torch.long, device=self.device
        )

        logits = self.model.prefill(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            slot_mapping=slot_mapping,
            last_token_indices=last_token_indices,
            kv_pool=self.kv_pool,
        )  # (batch, vocab)

        # ── Sample first token, advance cache_len ──────────────────────
        token_ids: list[int] = []
        for i, (req, L) in enumerate(zip(requests, seq_lens)):
            req.cache_len = L
            tok = sample_token(logits[i : i + 1, :], req.sampling_params, req.output_ids)
            token_ids.append(tok)
        return token_ids

    @torch.inference_mode()
    def paged_decode_step(self, requests: list[Request]) -> list[int]:
        """One decode step for a batch of running requests using paged KV."""
        assert self.kv_pool is not None, "paged_decode_step requires --mode paged"
        if not requests:
            return []

        page_size = self.page_size
        batch_size = len(requests)

        # Page tables have already been allocated for the worst case at
        # prefill time, so no per-step allocation is needed here.
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
        """Release a finished request's pages back to the pool."""
        if self.kv_pool is None or req.page_table is None:
            return
        self.kv_pool.free(req.page_table)
        req.page_table = None
        req.cache_len = 0

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
