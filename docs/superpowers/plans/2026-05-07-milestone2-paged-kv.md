# Milestone 2 — Paged KV + torch.compile Implementation Plan

**Goal:** Replace milestone-1 dynamic KV (`torch.cat`-grown per request) with a pre-allocated paged KV pool, integrate flash-attn paged-attention for prefill (packed via `flash_attn_varlen_func`) and decode (`flash_attn_with_kvcache` with block table), and apply `torch.compile` to a stable-shape sub-region (per-layer MLP).

**Architecture:**
- **Pool:** per-layer K and V tensors of shape `(num_pages, page_size, num_kv_heads, head_dim)` — exactly the layout `flash_attn_with_kvcache` expects for paged caches. Free list = a `deque[int]` of page indices.
- **Per-request state:** `page_table: list[int]` and `cache_len: int` on `Request`. The pool itself is owned by the `Engine`; requests just hold indices.
- **Paged model:** new `paged_model.py` mirrors `model.py` but uses flash-attn. Same parameter names so we can `load_state_dict` the same Qwen3 checkpoint into either.
- **Prefill path:** flatten N prompts → packed `(total_tokens, ...)` tensors → `flash_attn_varlen_func(causal=True)` for attention → scatter the just-computed K/V into pool slots via `slot_mapping`. No padding, no `torch.cat`.
- **Decode path:** stack one new token per running request → `flash_attn_with_kvcache(q, k_cache=pool_K, v_cache=pool_V, k=new_k, v=new_v, cache_seqlens, block_table, causal=True)` — flash-attn writes new tokens into the cache and computes attention in one call.
- **Page allocation:** at prefill, allocate `ceil(prompt_len / page_size)` pages. Each decode step, if `cache_len % page_size == 0` we grow the page table by 1.
- **Admission control:** scheduler asks the pool for free-page count; rejects/defers requests whose worst-case page demand exceeds availability. Bounded KV usage.
- **torch.compile:** wrap each layer's `MLP.forward` with `torch.compile(mode="reduce-overhead", dynamic=True)`. Stable shape per token (1 token in decode), batch dim dynamic. The MLP is the largest stable-shape compute chunk in decode.

**Tech Stack:** PyTorch ≥2.4, flash-attn ≥2.5 (paged-cache support), existing transformers/safetensors/HF tooling.

## File map

| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | modify | add `flash-attn>=2.5.0` |
| `miniengine/__main__.py` | modify | add `--mode paged`, `--mem-fraction-static`, `--page-size`, `--torch-compile`, `--cuda-graph` (stub), `--cuda-graph-batch-sizes` (stub) |
| `miniengine/core.py` | modify | add `page_table`, `cache_len` fields to `Request` |
| `miniengine/kv_memory_pool.py` | implement | the pool itself |
| `miniengine/paged_model.py` | create | `PagedCausalLM`, `PagedAttention`, etc. |
| `miniengine/engine.py` | modify | add `paged_prefill_batch`, `paged_decode_step`, build pool when `mode=="paged"` |
| `miniengine/scheduler.py` | modify | add `_step_paged` |
| `tests/test_kv_memory_pool.py` | create | CPU unit tests for the pool data structure |
| `milestone2_report.md` | create | report draft with placeholders for L4 benchmark output |

## Tasks

Tracked via TaskCreate (#1–#9). Order matches dependencies: deps/CLI → pool → request fields → paged model → engine paged paths → scheduler paged step → torch.compile → smoke test → report.

## Out of scope (intentional)

- **CUDA graphs (extra credit):** `--cuda-graph*` flags are accepted but not yet implemented. Would add `CudaGraphRunner` after the required path is verified.
- **Prefix caching:** every prefill freshly allocates pages; no shared prefix detection.
- **Eviction / preemption:** if pool is full at admission time the scheduler defers; no running-request preemption.

## Benchmarking (user-side)

The host running this plan has no GPU. After implementation:

```bash
# On the L4 host
pip install -e .
pip install flash-attn --no-build-isolation

# Accuracy
python -m benchmark.bench_accuracy --dataset mmlu --num-samples 100  # paged
# (also rerun with --torch-compile and against milestone-1 batched for comparison)

# Throughput (in another terminal: python -m miniengine --model Qwen/Qwen3-8B --mode <X>)
python -m benchmark.bench_serving --num-requests 64 --concurrencies 1,4,16,32

# Required modes to capture
#   batched                       (M1 baseline)
#   paged                         (≥2× over M1)
#   paged + torch-compile         (≥10% over paged)

# Page-size sweep
#   --page-size 16  vs  --page-size 128
```

Paste TTFT p50/p99, TPOT p50/p99, generation throughput screenshots into `milestone2_report.md`.
