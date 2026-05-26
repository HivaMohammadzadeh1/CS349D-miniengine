"""
Paged transformer model — Milestone 2 attention path.

Mirrors `model.py`'s module hierarchy exactly (same parameter names) so a
single Qwen3 safetensors checkpoint loads into either `CausalLM` (M1) or
`PagedCausalLM` (M2). Only the attention path differs:

  * Prefill is packed: N prompts flatten into one (total_tokens, ...)
    sequence and run through `flash_attn_varlen_func`. The just-computed
    K/V are scattered into the KVMemoryPool at the slot positions
    given by `slot_mapping`.

  * Decode runs one token per request through `flash_attn_with_kvcache`,
    which both writes the new K/V into the pool (via `block_table` +
    `cache_seqlens`) and computes attention against the resulting
    cache in a single kernel.

flash-attn handles GQA internally: pass Q with `num_attention_heads` and
K/V with `num_key_value_heads`; the kernel broadcasts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import MLP, ModelConfig, RMSNorm, RotaryEmbedding, _rotate_half


# ── RoPE helpers (paged-friendly, shape-agnostic) ──────────────────────


def _apply_rope_flat(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE for packed tensors.

    x:   (total_tokens, num_heads, head_dim)
    cos: (total_tokens, head_dim)
    sin: (total_tokens, head_dim)
    """
    cos = cos[:, None, :].to(x.dtype)
    sin = sin[:, None, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


def _apply_rope_decode(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE for one-token-per-batch tensors.

    x:   (batch, 1, num_heads, head_dim)
    cos: (batch, 1, head_dim)
    sin: (batch, 1, head_dim)
    """
    cos = cos[:, :, None, :].to(x.dtype)
    sin = sin[:, :, None, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


def _extend_rope(rotary: RotaryEmbedding, length: int) -> None:
    """(Re)populate `rotary._cos` and `rotary._sin` to `length` positions.

    Shared by `_lookup_rope` (on-demand growth during eager execution) and
    `CudaGraphRunner._capture_all` (one-shot pre-warm before capture, so
    `_lookup_rope` doesn't have to grow the cache from inside the graph).
    """
    t = torch.arange(length, device=rotary.inv_freq.device, dtype=rotary.inv_freq.dtype)
    freqs = torch.outer(t, rotary.inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    rotary._cos = emb.cos()
    rotary._sin = emb.sin()
    rotary._cached_len = length


def _lookup_rope(
    rotary: RotaryEmbedding, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up cos/sin for arbitrary-shape `position_ids`.

    Returns cos, sin with shape `position_ids.shape + (head_dim,)`.

    Cuda-graph note: during graph capture we cannot grow the cache (that
    would need `.item()`, which is a banned CPU↔GPU sync inside capture).
    Callers that drive captured graphs are responsible for pre-warming the
    cache to a size ≥ the maximum `position_ids` value they will replay
    with — see `_extend_rope` and `CudaGraphRunner._capture_all`.
    """
    if torch.cuda.is_current_stream_capturing():
        # Cache must already be sized; skip the bounds check (no .item()).
        return rotary._cos[position_ids], rotary._sin[position_ids]
    max_pos = int(position_ids.max().item()) + 1
    if rotary._cos is None or max_pos > rotary._cached_len:
        _extend_rope(rotary, max(max_pos, rotary._cached_len * 2, 256))
    return rotary._cos[position_ids], rotary._sin[position_ids]


# ── Attention ──────────────────────────────────────────────────────────


class PagedAttention(nn.Module):
    """GQA + QK-Norm + RoPE attention with flash-attn paged kernels."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    # ── Prefill (packed varlen) ────────────────────────────────────────

    def forward_prefill(
        self,
        hidden: torch.Tensor,            # (T, hidden)
        cos: torch.Tensor,               # (T, head_dim)
        sin: torch.Tensor,               # (T, head_dim)
        cu_seqlens: torch.Tensor,        # (batch+1,) int32
        max_seqlen: int,
        slot_mapping: torch.Tensor,      # (T,) int64 — flat slot index in pool
        kv_pool_layer: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Returns (T, hidden)."""
        from flash_attn import flash_attn_varlen_func

        T = hidden.shape[0]
        q = self.q_proj(hidden).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope_flat(q, cos, sin)
        k = _apply_rope_flat(k, cos, sin)

        # Self-attention over packed prompts. cu_seqlens delimits each
        # request; causal=True applies a per-request causal mask.
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )  # (T, num_heads, head_dim)

        # Scatter K, V into the pool. Pool layout is
        # (num_pages, page_size, num_kv_heads, head_dim); flatten the
        # first two dims so slot_mapping can be a flat index.
        K_pool, V_pool = kv_pool_layer
        np_, ps, kh, hd = K_pool.shape
        K_flat = K_pool.view(np_ * ps, kh, hd)
        V_flat = V_pool.view(np_ * ps, kh, hd)
        K_flat[slot_mapping] = k
        V_flat[slot_mapping] = v

        out = out.reshape(T, self.num_heads * self.head_dim)
        return self.o_proj(out)

    # ── Chunked / prefix-attention prefill (milestone 3) ───────────────

    def forward_prefill_chunked(
        self,
        hidden: torch.Tensor,            # (T, hidden) — T = chunk's q tokens
        cos: torch.Tensor,               # (T, head_dim)
        sin: torch.Tensor,               # (T, head_dim)
        cu_seqlens_q: torch.Tensor,      # (B+1,) int32 — chunk lengths
        cu_seqlens_k: torch.Tensor,      # (B+1,) int32 — full lengths (cached prefix + so-far)
        max_seqlen_q: int,
        max_seqlen_k: int,
        slot_mapping: torch.Tensor,      # (T,) int64 — flat pool slots for new K/V
        block_table: torch.Tensor,       # (B, max_blocks) int32 — full page tables
        kv_pool_layer: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Paged-attention varlen prefill.

        Handles two cases the M2 packed prefill cannot:
          1. A request's prompt is being processed in chunks; the new chunk
             must attend back to already-prefilled chunks of the same request
             that already live in the pool.
          2. A request hit the radix cache; the new (uncached) suffix must
             attend back to the cached prefix pages in the pool.

        Mechanism: write the new chunk's K/V into the pool FIRST (via
        ``slot_mapping``), then call ``flash_attn_varlen_func`` with the
        full ``cu_seqlens_k`` and ``block_table`` so the kernel reads K/V
        for the full sequence directly from the pool.
        """
        from flash_attn import flash_attn_varlen_func

        T = hidden.shape[0]
        q = self.q_proj(hidden).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope_flat(q, cos, sin)
        k = _apply_rope_flat(k, cos, sin)

        # Scatter the new chunk's K/V into the pool BEFORE the attention
        # call — the kernel reads them back via block_table.
        K_pool, V_pool = kv_pool_layer
        np_, ps, kh, hd = K_pool.shape
        K_flat = K_pool.view(np_ * ps, kh, hd)
        V_flat = V_pool.view(np_ * ps, kh, hd)
        K_flat[slot_mapping] = k
        V_flat[slot_mapping] = v

        out = flash_attn_varlen_func(
            q,
            K_pool,
            V_pool,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            block_table=block_table,
        )  # (T, num_heads, head_dim)

        out = out.reshape(T, self.num_heads * self.head_dim)
        return self.o_proj(out)

    # ── Decode (paged kvcache) ─────────────────────────────────────────

    def forward_decode(
        self,
        hidden: torch.Tensor,            # (B, 1, hidden)
        cos: torch.Tensor,               # (B, 1, head_dim)
        sin: torch.Tensor,               # (B, 1, head_dim)
        cache_seqlens: torch.Tensor,     # (B,) int32 — cache len BEFORE this token
        block_table: torch.Tensor,       # (B, max_blocks) int32
        kv_pool_layer: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Returns (B, 1, hidden)."""
        from flash_attn import flash_attn_with_kvcache

        B = hidden.shape[0]
        q = self.q_proj(hidden).view(B, 1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(B, 1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(B, 1, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope_decode(q, cos, sin)
        k = _apply_rope_decode(k, cos, sin)

        K_pool, V_pool = kv_pool_layer
        # flash_attn_with_kvcache:
        #   - Writes (k, v) into K_pool/V_pool at slots determined by
        #     (block_table, cache_seqlens).
        #   - Computes attention of q against the resulting cache.
        out = flash_attn_with_kvcache(
            q,
            K_pool,
            V_pool,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
        )  # (B, 1, num_heads, head_dim)

        out = out.reshape(B, 1, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ── Block ──────────────────────────────────────────────────────────────


class PagedTransformerBlock(nn.Module):
    """Pre-norm: LN → PagedAttention → residual → LN → MLP → residual.

    The block keeps two references to the same MLP weights:
      * ``self.mlp`` — eager mode, used by ``forward_prefill``. Packed
        varlen prefill has *variable* input shape `(total_packed_tokens,
        hidden)` per batch composition; running compiled here triggers a
        dynamo recompile per novel shape and regresses both throughput
        and latency (see report §3.3.3 for the eager-vs-compiled-on-prefill
        comparison).
      * ``self.mlp_decode`` — what ``forward_decode`` calls. By default
        it's the same eager module; the engine swaps it for a compiled
        version when ``--torch-compile`` is set. Decode input shape is
        `(B, 1, hidden)` which is stable up to dynamic B, so dynamo
        caches one kernel.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = PagedAttention(config)
        self.mlp = MLP(config)
        # Sentinel: starts as None (a plain attribute, NOT a registered
        # submodule, since None is not an nn.Module). State-dict load is
        # therefore not affected — the checkpoint's keys still cover
        # only ``mlp.*``. The engine sets this to a torch.compile-
        # wrapped version of ``self.mlp`` after load when
        # ``--torch-compile`` is enabled; until then, forward_decode
        # falls back to the eager ``self.mlp`` (see getattr below).
        self.mlp_decode = None
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward_prefill(
        self,
        hidden,
        cos,
        sin,
        cu_seqlens,
        max_seqlen,
        slot_mapping,
        kv_pool_layer,
    ):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn.forward_prefill(
            hidden, cos, sin, cu_seqlens, max_seqlen, slot_mapping, kv_pool_layer,
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        # Eager MLP: prefill has unstable shape; compiling here recompiles
        # per prompt-length-combination and regresses throughput.
        hidden = self.mlp(hidden)
        return residual + hidden

    def forward_prefill_chunked(
        self,
        hidden,
        cos,
        sin,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        block_table,
        kv_pool_layer,
    ):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn.forward_prefill_chunked(
            hidden,
            cos,
            sin,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            block_table,
            kv_pool_layer,
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        # Same eager-MLP choice as forward_prefill: chunk shape varies
        # batch-to-batch so compiling here would recompile.
        hidden = self.mlp(hidden)
        return residual + hidden

    def forward_decode(self, hidden, cos, sin, cache_seqlens, block_table, kv_pool_layer):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn.forward_decode(
            hidden, cos, sin, cache_seqlens, block_table, kv_pool_layer,
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        # Compiled MLP when --torch-compile is set; eager otherwise. Decode
        # input shape (B, 1, hidden) is stable for dynamo's cache.
        # ``mlp_decode`` is None until the engine attaches a compiled
        # version post-load; fall back to eager ``self.mlp`` either way.
        mlp = self.mlp_decode if self.mlp_decode is not None else self.mlp
        hidden = mlp(hidden)
        return residual + hidden


# ── Full transformer ───────────────────────────────────────────────────


class PagedTransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [PagedTransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, theta=config.rope_theta)

    def forward_prefill(
        self, input_ids, position_ids, cu_seqlens, max_seqlen, slot_mapping, kv_pool: KVMemoryPool,
    ):
        hidden = self.embed_tokens(input_ids)  # (T, hidden)
        cos, sin = _lookup_rope(self.rotary_emb, position_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_prefill(
                hidden, cos, sin, cu_seqlens, max_seqlen, slot_mapping,
                kv_pool.kv_caches[i],
            )
        return self.norm(hidden)

    def forward_prefill_chunked(
        self,
        input_ids,
        position_ids,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        block_table,
        kv_pool: KVMemoryPool,
    ):
        hidden = self.embed_tokens(input_ids)  # (T, hidden)
        cos, sin = _lookup_rope(self.rotary_emb, position_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_prefill_chunked(
                hidden,
                cos,
                sin,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                slot_mapping,
                block_table,
                kv_pool.kv_caches[i],
            )
        return self.norm(hidden)

    def forward_decode(
        self, input_ids, position_ids, cache_seqlens, block_table, kv_pool: KVMemoryPool,
    ):
        hidden = self.embed_tokens(input_ids)  # (B, 1, hidden)
        cos, sin = _lookup_rope(self.rotary_emb, position_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_decode(
                hidden, cos, sin, cache_seqlens, block_table,
                kv_pool.kv_caches[i],
            )
        return self.norm(hidden)


class PagedCausalLM(nn.Module):
    """Causal LM with paged attention. Same checkpoint layout as `CausalLM`."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = PagedTransformerModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def prefill(
        self,
        input_ids: torch.Tensor,        # (T,)
        position_ids: torch.Tensor,     # (T,)
        cu_seqlens: torch.Tensor,       # (batch+1,) int32
        max_seqlen: int,
        slot_mapping: torch.Tensor,     # (T,) int64
        last_token_indices: torch.Tensor,  # (batch,) int64 — index in T of each request's last token
        kv_pool: KVMemoryPool,
    ) -> torch.Tensor:
        """Run packed prefill. Returns logits at each request's last token, shape (batch, vocab)."""
        hidden = self.model.forward_prefill(
            input_ids, position_ids, cu_seqlens, max_seqlen, slot_mapping, kv_pool,
        )  # (T, hidden)
        last_hidden = hidden[last_token_indices]  # (batch, hidden)
        if self.config.tie_word_embeddings:
            return F.linear(last_hidden, self.model.embed_tokens.weight)
        return self.lm_head(last_hidden)

    def prefill_chunked(
        self,
        input_ids: torch.Tensor,        # (T,) — packed q tokens for this chunk
        position_ids: torch.Tensor,     # (T,)
        cu_seqlens_q: torch.Tensor,     # (B+1,) int32 — chunk lengths
        cu_seqlens_k: torch.Tensor,     # (B+1,) int32 — full lengths (cached prefix + so-far)
        max_seqlen_q: int,
        max_seqlen_k: int,
        slot_mapping: torch.Tensor,     # (T,) int64
        block_table: torch.Tensor,      # (B, max_blocks) int32
        last_token_indices: torch.Tensor | None,
        kv_pool: KVMemoryPool,
    ) -> torch.Tensor:
        """Run a single chunk of paged-attention varlen prefill.

        Returns:
          - If ``last_token_indices`` is None: hidden states ``(T, hidden)``.
            Caller invokes this when the chunk isn't the final one and the
            logits aren't needed (we skip the LM head to save a vocab-size
            matmul on intermediate chunks).
          - Else: logits at each request's last token ``(batch, vocab)``.
            Used on the FINAL chunk to sample the first generated token.
        """
        hidden = self.model.forward_prefill_chunked(
            input_ids,
            position_ids,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            block_table,
            kv_pool,
        )  # (T, hidden)
        if last_token_indices is None:
            return hidden
        last_hidden = hidden[last_token_indices]  # (batch, hidden)
        if self.config.tie_word_embeddings:
            return F.linear(last_hidden, self.model.embed_tokens.weight)
        return self.lm_head(last_hidden)

    def decode(
        self,
        input_ids: torch.Tensor,        # (B, 1)
        position_ids: torch.Tensor,     # (B, 1)
        cache_seqlens: torch.Tensor,    # (B,) int32
        block_table: torch.Tensor,      # (B, max_blocks) int32
        kv_pool: KVMemoryPool,
    ) -> torch.Tensor:
        """One decode step. Returns logits, shape (B, vocab)."""
        hidden = self.model.forward_decode(
            input_ids, position_ids, cache_seqlens, block_table, kv_pool,
        )  # (B, 1, hidden)
        hidden = hidden.squeeze(1)  # (B, hidden)
        if self.config.tie_word_embeddings:
            return F.linear(hidden, self.model.embed_tokens.weight)
        return self.lm_head(hidden)
