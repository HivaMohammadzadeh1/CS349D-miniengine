# Milestone 3 Design — Chunked Prefill + Radix Prefix Cache + Retraction

**Status:** Approved 2026-05-25
**Scope:** Required Parts A + B + Retraction Bonus
**Baseline:** Milestone-2 paged engine on the `main` branch (after merging upstream/main).
**Target HW:** L4 GPU on AWS, `Qwen/Qwen3-8B`.

## Overview

Three additions layered on the M2 paged engine:

1. **Chunked prefill (Part A)** — bounds per-step prefill activation memory by capping per-step Q-tokens.
2. **Radix prefix cache (Part B)** — token-prefix → KV-pages tree; matches at page granularity; LRU eviction.
3. **Retraction (Bonus)** — decode-time OOM recovery by evicting a running request back to the waiting queue.

These changes also force a one-time shift in the paged engine: **KV pages are allocated lazily during decode**, not eagerly worst-case at prefill time. Without that shift, the cache fights for space against worst-case reservations and the speedup target is unreachable. Lazy allocation is also what makes retraction meaningful: with worst-case alloc, decode never OOMs, so there's nothing to retract.

## CLI Surface

Added to `miniengine/__main__.py`:

| Flag | Default | Effect |
|------|---------|--------|
| `--prefill-chunk-size N` | `0` | Per-step prefill q-token budget. `0` = single-shot (M2 behavior). |
| `--disable-radix-cache` | off | Disable the radix cache. Cache is **on** by default in paged mode. |
| `--enable-retraction` | off | Enable decode-time retraction (paged-only). |

Constraints (validated in `__main__`):
- `--prefill-chunk-size` and `--disable-radix-cache` and `--enable-retraction` only meaningful for `--mode paged`. Warn-and-ignore otherwise.
- Radix cache is auto-disabled when `--mode != paged`.

## File-level Changes

```
new:        miniengine/radix_cache.py        — fill the skeleton (already in repo)
modified:   miniengine/kv_memory_pool.py     — eviction hook, num_evictable, KVOutOfMemory
modified:   miniengine/paged_model.py        — prefix-attention varlen prefill (cu_seqlens_k + block_table)
modified:   miniengine/engine.py             — lazy alloc, chunked-prefill, cache wiring, retraction helpers, .pool alias
modified:   miniengine/scheduler.py          — chunked-prefill state machine, retraction loop
modified:   miniengine/core.py               — Request.matched_node, prefill_offset (transient state)
modified:   miniengine/__main__.py           — 3 new flags + plumbing
new tests:  tests/test_radix_cache.py        — unit tests for the data structure
new tests:  tests/test_scheduler_m3.py       — chunked-prefill + retraction state-machine tests
```

## Detailed Design

### Part A — Chunked Prefill

#### Engine API

```python
def paged_prefill_request_chunked(
    self,
    req: Request,
    chunk_size: int,
) -> int | None:
    """Advance one prefill chunk for `req`. Allocates and writes the
    next chunk_size tokens. On the LAST chunk, samples and returns the
    first generated token. On non-last chunks, returns None.

    Precondition: req.page_table is set up by the scheduler via a prior
    call to `start_paged_prefill(req)`. Each call advances
    `req._prefill_offset` by chunk_size (clipped to prompt_len).

    Uses flash_attn_varlen_func with cu_seqlens_q over the chunk's q
    tokens, cu_seqlens_k over the full prefix-so-far (cached prefix +
    already-prefilled chunks + this chunk), and block_table = req.page_table.
    """
```

```python
def start_paged_prefill(self, req: Request) -> int:
    """One-time setup before chunked prefill begins:
       1. Cache lookup: match_prefix(req.input_ids).
       2. inc_lock_ref on matched node.
       3. Allocate pages for the *unmatched* portion of the prompt
          only (no max_new_tokens reservation).
       4. Set req.page_table = matched_pages + new_pages.
       5. Set req.cache_len = matched_tokens (the cached portion is
          already 'prefilled' in the pool).
       6. Set req._prefill_offset = matched_tokens.
       7. Set req.cache_hit_tokens = matched_tokens (server emits).

       Returns: number of prompt tokens still to prefill.
    """
```

#### Scheduler

State additions:

```python
class Scheduler:
    self._prefilling: Request | None = None  # in-flight chunked prefill
    self.chunk_size: int = 0                  # 0 disables chunking
```

`_step_paged` flow when `chunk_size > 0`:

```
1. If self._prefilling is None and waiting:
       req = waiting.popleft()
       remaining = engine.start_paged_prefill(req)
       self._prefilling = req

2. If self._prefilling:
       tok = engine.paged_prefill_request_chunked(self._prefilling, chunk_size)
       if tok is not None:                       # final chunk done
           self._prefilling.output_ids.append(tok)
           self._stream_token(self._prefilling, tok)
           if check_finished(...):
               engine.free_paged_request(self._prefilling)
               finish_request(...)
           else:
               self.running.append(self._prefilling)
           self._prefilling = None

3. Decode phase as today (engine.paged_decode_step on self.running).
```

When `chunk_size == 0`: fall through to milestone-2 packed-prefill path. Cache lookup still runs but the prefill happens in one varlen call instead of N.

#### Why single-request

Per-step Q-token bound is deterministic (`= chunk_size`), no mixed batching complexity, no need to interleave admissions during a chunked prefill, and the milestone target ("OOM at chunk_size=0, succeed with chunked") doesn't require multi-request chunking.

### Part B — Radix Prefix Cache

#### Data structure

```python
class RadixNode:
    parent: RadixNode | None
    children: dict[int, RadixNode]   # keyed by first token of edge
    key: list[int]                    # tokens on edge from parent (page-aligned)
    pages: list[int]                  # KV pages; len(pages) == len(key) // page_size
    ref_count: int                    # locked-leaf count in subtree
    last_access: float
```

Invariants:
- `len(key) % page_size == 0` for every node (page-aligned).
- `len(pages) == len(key) // page_size`.
- A child's `key[0]` is unique among its siblings (dict invariant).
- `ref_count > 0` ⇒ this node and all its ancestors are pinned against eviction.

#### Methods

`match_prefix(tokens: list[int]) -> MatchResult`:
1. Start at root, walk down. At each step, look up `children.get(tokens[off])`.
2. If found, match as many tokens as possible along that edge in page-aligned chunks. If the request's tokens diverge mid-edge (token-wise mismatch), stop at the last page boundary.
3. Stop when no child matches the next token, or when fewer than `page_size` tokens remain.
4. Update `metrics.total_lookups += 1`, `total_query_tokens += len(tokens)`, `total_hit_tokens += matched_tokens`.
5. Return `MatchResult(matched_pages, matched_tokens, last_node)`. `last_node` is `root` if zero match.

`insert_and_return(tokens, pages) -> (leaf, redundant_pages)`:
1. Walk from root, descending into existing children that match the next page-aligned chunk of `tokens`.
2. Where the input diverges from an existing edge at a page boundary, split the edge into two nodes.
3. Where the input goes deeper than any existing node, create new child nodes per page.
4. For every page-aligned chunk of `tokens` that maps to an *existing* page in the tree (same token sequence on the same path), the corresponding input page in `pages` is **redundant** — collect it; the cached page wins. Only genuinely new pages (no existing match at that depth/path) get attached to the tree.
5. Bump `metrics.total_inserted_pages` by *new* page count (not redundant).
6. Update `num_cached_pages` counter accordingly.

`evict(n_pages_needed) -> int`:
1. Walk the tree, collect all leaves with `ref_count == 0`. (A leaf is a node with no children.)
2. Use a min-heap keyed by `last_access` (oldest first).
3. Pop the oldest. Free its `pages` back to `pool._free`, detach from parent. If the parent now has no children AND `ref_count == 0`, push it onto the heap (it became a leaf).
4. Repeat until `freed >= n_pages_needed` or heap empty.
5. Bump `metrics.total_evicted_pages`. Decrement `num_cached_pages`.
6. Return actual freed count (may be < n if cache exhausted by locks).

`inc_lock_ref(node)` / `dec_lock_ref(node)`: walk parent chain, `++/--`. `dec_lock_ref` refreshes `last_access`.

`reset()`: walk tree, free every page back to pool, drop root's children, zero metrics.

#### Pool integration

`KVMemoryPool` additions:

```python
self.cache: RadixCache | None = None   # set by Engine post-construction

def attach_cache(self, cache: RadixCache) -> None:
    self.cache = cache

@property
def num_evictable(self) -> int:
    return self.cache.num_evictable_pages() if self.cache else 0

class KVOutOfMemory(RuntimeError): ...

def allocate(self, n: int) -> list[int]:
    if len(self._free) < n and self.cache is not None:
        self.cache.evict(n - len(self._free))
    if len(self._free) < n:
        raise KVOutOfMemory(
            f"KV pool out of pages: requested {n}, free {len(self._free)}, "
            f"evictable {self.num_evictable}"
        )
    return [self._free.popleft() for _ in range(n)]
```

#### Engine wiring

```python
class Engine:
    def __init__(self, ..., disable_radix_cache: bool = False):
        ...
        if self.mode == "paged" and not disable_radix_cache:
            self.radix_cache = RadixCache(self.kv_pool)
            self.kv_pool.attach_cache(self.radix_cache)
        else:
            self.radix_cache = None

    @property
    def pool(self) -> KVMemoryPool | None:
        return self.kv_pool    # alias for server.py compatibility
```

Prefill (cache-on path):
- `start_paged_prefill(req)` does the lookup, allocation, and bookkeeping above.
- `paged_prefill_request_chunked(req, chunk_size)` processes one chunk:
  - Build packed q/k/v for `input_ids[off : off+chunk_size]`.
  - Build `cu_seqlens_q = [0, chunk_len]`, `cu_seqlens_k = [0, off + chunk_len]` (full sequence so far).
  - `block_table = req.page_table` (tensorized to int32, (1, max_blocks)).
  - `slot_mapping` for the chunk only (writes to pages `[off // page_size : (off+chunk_len) // page_size]`).
  - Call `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=chunk_len, max_seqlen_k=off+chunk_len, causal=True, block_table=block_table)`. *Falls back to `flash_attn_with_kvcache` if installed version lacks varlen `block_table` support.*
  - Advance `req._prefill_offset`, `req.cache_len`.
  - On last chunk: sample first generated token from the last position's logits, return it.

After last chunk → insert into cache:

```python
def _insert_prompt_into_cache(self, req):
    if self.radix_cache is None: return
    page_aligned_len = req.num_input_tokens - (req.num_input_tokens % self.page_size)
    if page_aligned_len == 0: return
    prefix_tokens = req.input_ids[:page_aligned_len]
    prefix_pages = req.page_table[: page_aligned_len // self.page_size]
    _leaf, redundant = self.radix_cache.insert_and_return(prefix_tokens, prefix_pages)
    if redundant:
        # We were not the first to cache this prefix. The first insert wins;
        # our copies of those pages go back to the pool. We do NOT swap our
        # page_table — both copies contain identical K/V, so our request can
        # continue decoding against its own pages.
        self.kv_pool.free(redundant)
```

On finish (`free_paged_request`):

```python
def free_paged_request(self, req):
    if self.kv_pool is None or req.page_table is None: return
    if self.radix_cache is not None and getattr(req, "matched_node", None):
        self.radix_cache.dec_lock_ref(req.matched_node)
    # Insert (prompt + output) into cache for future multi-turn hits.
    full_tokens = req.input_ids + req.output_ids
    aligned = len(full_tokens) - (len(full_tokens) % self.page_size)
    if self.radix_cache is not None and aligned > 0:
        full_pages = req.page_table[: aligned // self.page_size]
        _, redundant = self.radix_cache.insert_and_return(full_tokens[:aligned], full_pages)
        self.kv_pool.free(redundant)
        # Remaining pages (the unaligned tail) go straight to the pool.
        tail_pages = req.page_table[aligned // self.page_size :]
        self.kv_pool.free(tail_pages)
    else:
        self.kv_pool.free(req.page_table)
    req.page_table = None
    req.cache_len = 0
```

### Lazy decode-time allocation

`paged_decode_step` change:

```python
for req in requests:
    if req.cache_len % page_size == 0 and req.cache_len > 0:
        # Need a new page for this step's token (cache_len is the
        # position the new token will be written to).
        try:
            new_page = self.kv_pool.allocate(1)[0]
        except KVOutOfMemory:
            raise   # propagated to scheduler for retraction
        req.page_table.append(new_page)
```

The current `start_paged_prefill` only allocates for prompt pages (no max_new_tokens reservation). Each decode step that crosses a page boundary appends one more page lazily.

### Retraction (Bonus)

`Scheduler._step_paged` wraps the decode phase:

```python
def _step_paged_decode_with_retraction(self):
    while True:
        try:
            token_ids = self.engine.paged_decode_step(self.running)
            return token_ids
        except KVOutOfMemory:
            if not self._retract_one_victim():
                raise  # no eligible victim — genuine OOM
            # loop and retry the decode step with one fewer request
```

`_retract_one_victim() -> bool`:

```
candidates = [r for r in self.running if r.cache_len > 0]
if not candidates: return False
# Youngest first (last admitted), tie-break by largest work-remaining.
victim = max(candidates, key=lambda r: (
    r.arrival_time, r.sampling_params.max_new_tokens - r.num_output_tokens
))
# Free its pages WITHOUT caching its partial output.
if self.engine.radix_cache is not None and victim.matched_node:
    self.engine.radix_cache.dec_lock_ref(victim.matched_node)
self.engine.kv_pool.free(victim.page_table)
victim.page_table = None
victim.cache_len = 0
victim.output_ids = []        # re-prefill from scratch on re-admission
victim.matched_node = None
victim.cache_hit_tokens = 0
victim.status = RequestStatus.WAITING
self.running.remove(victim)
self.waiting.appendleft(victim)   # head of queue
return True
```

Edge cases:
- **In-flight chunked-prefill request never a victim.** It's tracked separately in `self._prefilling`; not in `self.running`. So `candidates` excludes it automatically.
- **Pinned cache pages stay pinned.** `matched_node`'s `ref_count` was incremented at prefill start; we decrement only here, the cached pages remain available for future hits.
- **Genuine OOM (all requests already retracted, or only chunked-prefill in flight).** Re-raise `KVOutOfMemory` — caller (scheduler loop) logs and proceeds to next step. The chunked-prefill request will fail on its next allocation; that case can be improved later.

### `Request` additions

`miniengine/core.py`:

```python
# ── Milestone 3 transient state ─────────────────────────────────────
# Radix-cache node whose lock_ref we incremented at admission.
# Decremented on finish or retraction.
matched_node: Any = None  # RadixNode, but core.py doesn't import it
# Chunked-prefill cursor: how many tokens have been prefilled so far.
# Includes cached prefix length when cache hits.
_prefill_offset: int = 0
```

These are transient (scheduler/engine internal) and don't belong on the public-API surface, but for simplicity they live on `Request` rather than a sidecar dict.

## Testing

`tests/test_radix_cache.py` — pure data structure (no GPU):

```
- empty cache: match_prefix returns 0
- insert_and_return then match: full match, last_node has matching pages
- insert then match SHORTER prefix: page-aligned partial match
- insert(A), insert(A): second call returns A's pages as redundant
- insert(A_then_B), insert(A_then_C): tree has two divergent children at the split
- evict(n) frees only unlocked leaves, oldest first
- inc_lock_ref then evict: pinned subtree NOT touched
- dec_lock_ref then evict: previously pinned subtree evictable
- reset() drops every page back to a mock pool
```

`tests/test_scheduler_m3.py` — chunked-prefill state machine + retraction (engine mocked):

```
- chunk_size=0: behaves like M2 packed prefill
- chunk_size=256 with prompt_len=1024: 4 chunks, no admissions during
- decode advances other running requests in parallel during chunked prefill
- KVOutOfMemory: retraction kicks in, victim returns to waiting head
- KVOutOfMemory with no eligible victim: propagates
```

We don't test prefix-attention varlen prefill at the unit level — that's an integration concern; covered via end-to-end smoke run on L4.

## Out-of-Scope (explicit)

- Multi-request mixed-batch chunked prefill (sglang style).
- Mid-flight `page_table` swap on duplicate insert (de-fragmentation).
- Insertion during decode at every page boundary (only at prefill end + request finish).
- Speculative cache lookups for chunked-prefill chunks ≥ 1.
- Anything other than FCFS admission order.

## Acceptance Criteria

Implementation passes when:

1. Server starts in paged mode with the new flags; default behavior unchanged when none are set (modulo lazy alloc — see below).
2. `bench_serving --conc 1/4/16` on default workload: cache-on within noise of `--disable-radix-cache`.
3. `bench_cache --workload shared --shared-prefix-len 2000`: ≥ 2× throughput, ≥ 2× TTFT vs `--disable-radix-cache`.
4. `bench_cache --workload multiturn --turns-per-session 5+`: ≥ 50% throughput, ≥ 50% TTFT vs `--disable-radix-cache`; per-turn hit rate climbs from 0%.
5. `bench_accuracy` MMLU at `--prefill-chunk-size 0` vs e.g. 512: accuracy within noise.
6. A bench_serving config OOMs at `chunk_size=0` and succeeds at chunked. (User runs and selects exact config on L4.)
7. Retraction bonus: a constructed workload completes after retraction lands, fails without it. Victim-selection policy documented in report.

**Note on lazy alloc behavior change:** lazy KV allocation is **unconditional** in paged mode under this milestone. With `--prefill-chunk-size 0 --disable-radix-cache`, behavior matches M2's packed-prefill kernel-for-kernel, except decode no longer reserves `max_new_tokens` pages up front. Concurrency should be neutral-or-better (no worst-case reservation tax). The downside is that decode can now allocate a page mid-step and hit the pool empty — at which point the request raises `KVOutOfMemory`. Without `--enable-retraction`, that propagates and the request errors out; with retraction on, the scheduler evicts a victim back to the waiting queue and retries.
