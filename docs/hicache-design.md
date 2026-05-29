# HiCache — Hierarchical KV Cache (GPU + CPU)

**Milestone 4, Track 1.** Extends the milestone-3 radix prefix cache with a CPU-memory tier so evicted GPU prefixes are demoted to pinned host pages instead of dropped, and re-promoted on hit.

**Targets** (in priority order):
1. **Full credit.** On `bench_cache.py --workload multiturn`, show the GPU-only hit-rate cliff (≥70% turn 1, <20% last turn) and show HiCache restores per-turn hit rate. End-to-end must complete (no hangs/OOM, token-level sanity).
2. **Bonus A — overlap.** `--hicache-overlap`: dedicated CUDA stream, pinned host memory, async H2D/D2H. Demonstrate overlap is real (instrumented promote-time-hidden ratio + an `nsys` timeline).
3. **Bonus B — perf win.** ≥20% throughput **or** ≥20% TTFT improvement on at least one multiturn configuration vs the milestone-3 default.

**Hardware (locked).** Qwen3-8B on a single L4 (23 GB HBM, 60 GB host RAM, no swap). Target CPU tier ≈ **40 GB** (~10× the ≈4 GB GPU KV pool at `--mem-fraction-static 0.85`).

---

## Architecture

One radix tree, two tiers. Each `RadixNode` carries a `tier ∈ {"gpu","cpu"}` field; its `pages` list is interpreted in that tier's index space. **A node's pages live entirely in one tier** — no mixed-tier nodes, page granularity preserved.

**`cpu_pool is None` ⇒ behavior is byte-identical to milestone-3.** The baseline comparison is the same binary with `--cpu-cache-size-gb 0`.

### Module map

| File | Change |
|---|---|
| `miniengine/cpu_kv_pool.py` | **NEW.** Pinned-host mirror of `KVMemoryPool`. |
| `miniengine/radix_cache.py` | `RadixNode.tier` field; `RadixCache` takes optional `cpu_pool` + overlap flag; `evict` demotes instead of dropping; new `promote_path` helper; CPU-tier LRU; counters. |
| `miniengine/engine.py` | Build + attach `CpuKvPool` when flag set; call `promote_path` in `start_paged_prefill` before locking the matched node. |
| `miniengine/__main__.py` | `--cpu-cache-size-gb`, `--hicache-overlap` flags; validation. |
| `tests/test_cpu_kv_pool.py` | **NEW.** Sizing + alloc/free. |
| `tests/test_radix_cache.py` | New tiering tests (demote/promote round-trip, CPU LRU, locking). Existing tests stay green. |

---

## Components

### `CpuKvPool` (new)

Per-layer pinned host tensors, layout matched to `KVMemoryPool` for direct copy indexing.

```
shape = (num_cpu_pages, page_size, num_kv_heads, head_dim)  # per layer, K and V
device = "cpu", pin_memory = True, dtype = same as GPU pool
```

Capacity is derived from the flag:
```
per_page_bytes = 2 * num_layers * page_size * num_kv_heads * head_dim * dtype_bytes
num_cpu_pages  = floor(cpu_cache_size_gb * 1e9 / per_page_bytes)
```
Logged at startup so the report can quote it.

Public surface:
- `allocate(n: int) -> list[int]` — raises `CpuKvOutOfMemory` if full; caller (the cache) handles overflow by CPU-LRU eviction.
- `free(slots: list[int]) -> None`.
- `num_free: int`, `capacity: int`.
- `.k_buffers[layer]`, `.v_buffers[layer]` — exposed so the cache can index copies directly.

### `RadixNode.tier`

Single new field, default `"gpu"`. `node.pages` are GPU page indices when `tier=="gpu"`, CPU slot indices when `tier=="cpu"`. Match/insert/lock-ref logic is **unchanged** — it operates on token keys and is tier-agnostic.

### `RadixCache` changes

Optional dependencies: `cpu_pool: CpuKvPool | None`, `copy_stream: torch.cuda.Stream | None`, `overlap: bool`.

**`evict(n_pages_needed)` — demote-on-evict.** When `cpu_pool is None`, behavior is exactly as today. When set, for each LRU-selected *GPU-tier leaf* (`tier=="gpu"`, `ref_count==0`, `not children`):

1. Ensure CPU room. If `cpu_pool.num_free < len(node.pages)`, run `_cpu_evict(need)` (LRU walk over CPU-tier leaves, dropping them entirely). If still insufficient, **drop the GPU node entirely** as in m3 (guarantees progress).
2. `cpu_slots = cpu_pool.allocate(len(node.pages))`.
3. **D2H copy** — see "Copy mechanism" below.
4. `gpu_pool.free(node.pages)` (deferred in overlap mode; see pending-free queue).
5. `node.tier = "cpu"; node.pages = cpu_slots`. **Node stays in the tree** — the prefix is still cached, just colder.

The GPU LRU walk only considers GPU-tier leaves (CPU-tier nodes have no GPU pages to reclaim).

**`promote_path(match_result) -> MatchResult` (new).** Called from `start_paged_prefill` after `match_prefix`. For each CPU-tier node on the matched path (root → leaf direction, so parents promoted before children):

1. `gpu_pages = gpu_pool.allocate(len(node.pages))` — *this may itself demote colder nodes; that's fine.*
2. **H2D copy** — see below.
3. `cpu_pool.free(node.pages)`; `node.tier = "gpu"; node.pages = gpu_pages`.

Then proceed exactly like a normal GPU hit: `inc_lock_ref(deepest_node)` (pins ancestor chain), build the request's page table from now-all-GPU matched pages, prefill only the unmatched suffix.

### `_cpu_evict(n_slots_needed)` (new)

LRU walk over **CPU-tier** leaves (`tier=="cpu"`, `ref_count==0`, `not children`), dropping each: `cpu_pool.free(node.pages)`, remove node from tree. No lower tier. Loop until enough CPU slots free, or no more evictable CPU leaves (caller falls back to dropping the GPU node).

---

## Copy mechanism

### Blocking mode (default)

Synchronous per-layer copies on the default stream. For each layer, K and V:
```python
dst[dst_slots] = src[src_pages]   # torch indexing, blocking
```

This is correct by construction (the next CUDA op sees a fully copied tensor) and simple enough to land first.

### Overlap mode (`--hicache-overlap`)

A dedicated `self.copy_stream = torch.cuda.Stream()` and a **pending-free queue** on each pool: `list[tuple[pages, event]]`.

**Demote (D2H async).**
1. `with torch.cuda.stream(self.copy_stream):` issue `cpu.k[l][slots].copy_(gpu.k[l][pages], non_blocking=True)` per layer (and V).
2. `event = torch.cuda.Event(); event.record(self.copy_stream)`.
3. `gpu_pool.pending_free.append((pages, event))` — *don't* return pages to the free list yet.
4. `gpu_pool.allocate` drains the queue at its top: returns pages whose event has fired (`event.query()`). This prevents a fresh `allocate` from handing out pages that a D2H is still reading.

**Promote (H2D async).**
1. Same stream/event pattern, copying `cpu → gpu`.
2. After issuing, `torch.cuda.current_stream().wait_event(event)` — the compute stream that will run the request's forward waits on the copy event. Flash-attn never reads half-copied KV.
3. CPU slots can be freed immediately (we no longer need them); the H2D source data is in CPU pinned memory which we won't touch.

**Proving overlap is real (bonus deliverable).**
- Instrumented per-step: accumulate copy-stream wall time (event-pair `elapsed_time`) and total step wall time → report **promote-time-hidden ratio** = `1 - (copy_outside_compute_window / total_copy)`.
- `nsys profile` on a short multiturn run; screenshot timeline in the report showing copy-stream bars under compute bars.

---

## CLI & validation

Added to `miniengine/__main__.py`:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--cpu-cache-size-gb` | float | `0.0` | CPU KV tier size in GiB. `0` disables HiCache (radix cache stays GPU-only, exactly as m3). |
| `--hicache-overlap` | flag | off | Use a dedicated CUDA stream + pending-free queue for async D2H/H2D. Only meaningful when `cpu-cache-size-gb > 0`. |

Validation (fail-fast at startup):
- `cpu-cache-size-gb > 0` requires `--mode paged` and the radix cache enabled (not `--disable-radix-cache`).
- `hicache-overlap` requires `cpu-cache-size-gb > 0`.

Engine logs at startup: GPU pool capacity, CPU pool capacity, ratio, page size, dtype.

---

## Observability

New counters on `RadixCache`:
- `total_demoted_pages`, `total_promoted_pages`, `total_cpu_evicted_pages`
- `total_demote_time_ms`, `total_promote_time_ms` (event-measured in overlap mode, wall-clock in blocking mode)

Existing `bench_cache.py` per-turn `hit_rate` already counts `matched_tokens` regardless of tier, so a CPU-resident match is naturally a hit. That's the point: HiCache keeps the per-turn hit rate high across turns because the prefix node stays in the tree.

---

## Correctness

**Token identity (greedy).** A demote+promote round-trip is a memcpy of K and V, so KV is bitwise preserved. The smoke test: run a fixed multiturn workload greedy, with HiCache off and on; assert the output token streams are identical.

**MMLU within ±1pp.** Existing `bench_accuracy.py --dataset mmlu`, run on Qwen3-8B with HiCache on, compared against the m3 baseline.

**Concurrency.** The engine loop is single-threaded across scheduler steps. The hazard is a *later* step evicting pages still in use by a request, fixed the same way as m3: a request `inc_lock_ref`s the matched (now-all-GPU) node *before* yielding control; locked nodes (`ref_count>0`) are excluded from both GPU demotion and CPU eviction. In overlap mode the CUDA event additionally gates the compute on the H2D completion.

**No deadlock.** Eviction has a guaranteed-progress fallback: if both CPU eviction and GPU demotion can't free pages, drop the GPU node directly (m3 behavior). `allocate` therefore terminates or raises `KVOutOfMemory` for the scheduler to retract on.

---

## Testing

**CPU-side (no GPU, runs anywhere — primary correctness gate):**
- `tests/test_cpu_kv_pool.py`: sizing math, alloc/free, capacity exhaustion.
- `tests/test_radix_cache.py` (extended): demote moves node to cpu tier + frees gpu pages + values preserved; promote reverses; round-trip bit-identical; CPU overflow drops LRU; locked nodes never demoted/dropped; existing m3 tests stay green with `cpu_pool=None`.

**GPU-side (L4 — secondary correctness + perf):**
- Smoke: `python -m miniengine --model Qwen/Qwen3-8B --mode paged --cpu-cache-size-gb 40 ...` then a short multiturn; no OOM, no hang.
- Token-identity on greedy multiturn vs baseline (m3).
- The full-credit cliff demo + restore.
- Overlap measurement + nsys.
- MMLU ±1pp.
- ≥20% throughput/TTFT on at least one config.

---

## Bonus B — strategy for the ≥20% win

The win is wall-clock from **avoiding re-prefill on prefixes that would otherwise miss**. The arithmetic: H2D of a 2k-token prefix's KV at fp16 ≈ tens of MB → ~1–2 ms over PCIe gen4. Re-prefilling that prefix on Qwen3-8B ≈ 50–150 ms. So promote is ~50–100× cheaper than re-prefill for long prefixes — the win is structurally plausible.

To realize it, the demo config needs:
- Working set that *overflows* the GPU pool (otherwise nothing demotes; HiCache and m3 are identical).
- Working set that *fits* in the CPU pool (otherwise CPU LRU kicks the entries we'd hit on).
- Long enough shared prefixes per turn that promote cost ≪ re-prefill cost.

Concretely: `bench_cache.py --workload multiturn` with `--num-sessions` and `--turns-per-session` tuned so total working set ~3–5× the GPU pool (cliff territory for m3) and well under 40 GB. Throughput reported via `bench_serving` on the same prompts replayed.

---

## Risks & "what didn't work" candidates (for the report)

- **H2D not actually overlapping.** Event misuse or implicit sync stalling the copy stream. Measured by the overlap ratio.
- **GPU pages reused before D2H completes** — pending-free queue must be drained before every allocate.
- **Host RAM pressure.** No swap; if the CPU pool plus the Python process exceeds ~58 GB, OS will OOM-kill. Cap CPU pool at ~48 GB ceiling; default to 40.
- **Win shows only for long prefixes.** Short multiturn turns may not beat re-prefill — pick demo config accordingly and document the inflection point.
- **Subtle: a node split mid-demote.** Eviction operates on leaves; demote runs after the leaf is locked-out of further child insertion within the same `evict` call. The radix-tree split path (`insert_and_return`) only creates new nodes on insertion, not during eviction, so a node can't sprout children while being demoted.
