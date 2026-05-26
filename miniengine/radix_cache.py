"""Radix-tree prefix cache — Milestone 3, Part B.

Stores already-computed KV pages keyed by token prefix so a new request whose
prompt starts with a cached prefix can reuse those pages instead of
recomputing them.

The data structure is a radix tree whose nodes own KV pages from the
``KVMemoryPool``.  Pages held by the cache are *not* in the pool's free list;
they return there only when the cache evicts them (LRU) or when an
``insert_and_return`` call discovers them as duplicates of pages already in
the tree.

Invariants
----------
* Every node's edge ``key`` is page-aligned: ``len(key) % page_size == 0``
  and ``len(pages) == len(key) // page_size``.
* Sibling edges (children of the same node) have unique **first pages** —
  the ``children`` dict is keyed by ``tuple(key[:page_size])``. Keying by
  the full first page (not just the first token) is necessary because two
  unrelated prompts can share the first token of the chat template but
  diverge within the first page; with page-aligned matching they must be
  stored as separate siblings.
* ``ref_count > 0`` on a node means the node *and* every ancestor up to root
  are pinned against eviction (an in-flight request is borrowing them).

Performance counters in ``CacheMetrics`` are surfaced via ``/cache_stats``.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Aggregate cache statistics — surfaced via ``/cache_stats``."""

    total_lookups: int = 0
    total_query_tokens: int = 0
    total_hit_tokens: int = 0
    total_inserted_pages: int = 0
    total_evicted_pages: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_query_tokens == 0:
            return 0.0
        return self.total_hit_tokens / self.total_query_tokens


class RadixNode:
    """A radix-tree node.

    Edge layout: ``key`` carries the tokens on the edge from the parent;
    ``pages`` carries the KV pages corresponding to ``key`` (one page per
    ``page_size`` tokens; both are page-aligned by construction).
    """

    __slots__ = ("parent", "children", "key", "pages", "ref_count", "last_access")

    def __init__(self) -> None:
        self.parent: RadixNode | None = None
        # Keyed by tuple(key[:page_size]) — see module docstring for why
        # first-token keying is not sufficient.
        self.children: dict[tuple[int, ...], "RadixNode"] = {}
        self.key: list[int] = []
        self.pages: list[int] = []
        self.ref_count: int = 0
        self.last_access: float = time.monotonic()


@dataclass
class MatchResult:
    """Result of a prefix lookup.

    ``matched_tokens`` is page-aligned (multiple of ``page_size``);
    ``matched_pages`` carries the KV pages for those tokens.
    ``last_node`` is the deepest node the walk reached — callers lock it
    (``inc_lock_ref``) for the lifetime of the borrowing request.
    """

    matched_pages: list[int] = field(default_factory=list)
    matched_tokens: int = 0
    last_node: "RadixNode | None" = None


class RadixCache:
    """Token-prefix → KV-pages cache backed by a radix tree.

    Page-aligned matching, LRU eviction of unlocked subtrees,
    eviction-on-allocate via ``KVMemoryPool.allocate``, and sglang-style
    ``inc_lock_ref`` / ``dec_lock_ref`` pinning for in-flight matches.
    """

    def __init__(self, pool: "KVMemoryPool") -> None:
        self.pool = pool
        self.page_size = pool.page_size
        self.root = RadixNode()
        self.metrics = CacheMetrics()
        self._num_cached_pages = 0
        # Tie-breaker for heap entries with equal last_access timestamps.
        self._heap_seq = itertools.count()

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def num_cached_pages(self) -> int:
        """Total pages currently held by the tree."""
        return self._num_cached_pages

    def num_evictable_pages(self) -> int:
        """Pages that an LRU sweep could free right now."""
        total = 0
        for node in self._walk():
            if node is self.root:
                continue
            if self._is_evictable(node):
                total += len(node.pages)
        return total

    def _walk(self):
        """Iterate every node in the tree (including root)."""
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    def _is_evictable(self, node: "RadixNode") -> bool:
        """A node is evictable when nothing in its subtree is locked."""
        return node.ref_count == 0 and not node.children

    # ── Lookup ─────────────────────────────────────────────────────────

    def match_prefix(self, tokens: list[int]) -> MatchResult:
        """Find the longest page-aligned prefix of ``tokens`` in the tree.

        Walks page-by-page: at each level the next child is looked up by
        ``tuple(tokens[offset:offset+ps])``. Along the matched edge we
        continue in ``page_size`` chunks until the edge ends or a page's
        tokens diverge.
        """
        ps = self.page_size
        self.metrics.total_lookups += 1
        self.metrics.total_query_tokens += len(tokens)

        matched_pages: list[int] = []
        matched_tokens = 0
        node = self.root
        offset = 0
        now = time.monotonic()

        while offset + ps <= len(tokens):
            page_key = tuple(tokens[offset : offset + ps])
            child = node.children.get(page_key)
            if child is None:
                break
            # First page guaranteed to match (dict key). Walk further pages
            # along this edge.
            edge_off = ps
            edge_len = len(child.key)
            matched_pages.append(child.pages[0])
            matched_tokens += ps
            offset += ps
            while (
                edge_off + ps <= edge_len
                and offset + ps <= len(tokens)
                and child.key[edge_off : edge_off + ps]
                == tokens[offset : offset + ps]
            ):
                matched_pages.append(child.pages[edge_off // ps])
                matched_tokens += ps
                edge_off += ps
                offset += ps

            if edge_off < edge_len:
                # Partial edge match — stop at the last full page boundary.
                # The child stays where it is; no split is necessary on a
                # pure lookup.
                node = child
                break

            # Full edge consumed; descend.
            child.last_access = now
            node = child

        self.metrics.total_hit_tokens += matched_tokens
        return MatchResult(
            matched_pages=matched_pages,
            matched_tokens=matched_tokens,
            last_node=node,
        )

    # ── Lock ref counting (sglang-style) ───────────────────────────────

    def inc_lock_ref(self, node: "RadixNode | None") -> None:
        """Lock ``node`` (and the path to root) against eviction."""
        if node is None:
            return
        cur = node
        while cur is not None and cur is not self.root:
            cur.ref_count += 1
            cur = cur.parent

    def dec_lock_ref(self, node: "RadixNode | None") -> None:
        """Release a lock.  Refresh ``last_access`` while walking."""
        if node is None:
            return
        now = time.monotonic()
        cur = node
        while cur is not None and cur is not self.root:
            if cur.ref_count > 0:
                cur.ref_count -= 1
            cur.last_access = now
            cur = cur.parent

    # ── Insertion ──────────────────────────────────────────────────────

    def insert_and_return(
        self, tokens: list[int], pages: list[int]
    ) -> tuple["RadixNode", list[int]]:
        """Insert (tokens, pages) into the tree.

        Both arguments must be page-aligned:
            len(tokens) % page_size == 0
            len(pages) == len(tokens) // page_size

        Returns ``(leaf_node, redundant_pages)``: ``redundant_pages`` are
        pages the caller handed in that turned out to be duplicates of pages
        already cached at the same prefix.  The caller should return them to
        the pool.
        """
        ps = self.page_size
        if len(tokens) % ps != 0:
            raise ValueError(
                f"insert_and_return requires page-aligned tokens; "
                f"got len={len(tokens)} ps={ps}"
            )
        if len(pages) != len(tokens) // ps:
            raise ValueError(
                f"insert_and_return: pages/tokens mismatch ({len(pages)} vs "
                f"{len(tokens) // ps})"
            )

        redundant: list[int] = []
        node = self.root
        offset = 0
        now = time.monotonic()

        while offset < len(tokens):
            page_key = tuple(tokens[offset : offset + ps])
            child = node.children.get(page_key)
            if child is None:
                # Fresh edge — append the remainder as a new leaf.
                new_node = RadixNode()
                new_node.parent = node
                new_node.key = list(tokens[offset:])
                new_node.pages = list(pages[offset // ps :])
                new_node.last_access = now
                node.children[page_key] = new_node
                self._num_cached_pages += len(new_node.pages)
                self.metrics.total_inserted_pages += len(new_node.pages)
                return new_node, redundant

            # First page guaranteed to match (dict key). Walk further pages.
            edge_off = ps
            edge_len = len(child.key)
            redundant.append(pages[offset // ps])
            offset += ps
            while (
                edge_off < edge_len
                and offset < len(tokens)
                and child.key[edge_off : edge_off + ps]
                == tokens[offset : offset + ps]
            ):
                # Page already cached — caller's copy is redundant.
                redundant.append(pages[offset // ps])
                edge_off += ps
                offset += ps

            if edge_off == edge_len:
                # Full edge consumed and matched; descend.
                child.last_access = now
                node = child
                continue

            # Mismatch mid-edge at a page boundary — split. edge_off >= ps
            # is guaranteed (first page matched via the dict lookup).
            split = RadixNode()
            split.parent = node
            split.key = child.key[:edge_off]
            split.pages = child.pages[: edge_off // ps]
            split.last_access = now
            # Re-parent the tail of the original child under the split.
            old_child_first_page = page_key   # before mutating child.key
            child.key = child.key[edge_off:]
            child.pages = child.pages[edge_off // ps :]
            child.parent = split
            split.children[tuple(child.key[:ps])] = child
            # Split inherits the SAME first-page key the old child had
            # under `node`, since split.key starts at child.key's old
            # first page. Overwrite the parent's dict entry.
            node.children[old_child_first_page] = split
            # ref_count on a node is "locked leaves in subtree"; after
            # split, all that lock-weight belongs to descendants, so split
            # inherits child's ref_count.
            split.ref_count = child.ref_count

            if offset == len(tokens):
                return split, redundant

            # Attach the remaining input under the split as a new branch.
            new_node = RadixNode()
            new_node.parent = split
            new_node.key = list(tokens[offset:])
            new_node.pages = list(pages[offset // ps :])
            new_node.last_access = now
            split.children[tuple(tokens[offset : offset + ps])] = new_node
            self._num_cached_pages += len(new_node.pages)
            self.metrics.total_inserted_pages += len(new_node.pages)
            return new_node, redundant

        # Exhausted tokens exactly at a node boundary — return the deepest
        # node we landed on. Everything was redundant.
        return node, redundant

    # ── Eviction ───────────────────────────────────────────────────────

    def evict(self, n_pages_needed: int) -> int:
        """LRU-evict at least ``n_pages_needed`` pages (best effort).

        Walks evictable leaves (no children, ref_count == 0) oldest-first.
        After freeing a leaf, its parent may become a leaf — push it back
        onto the heap if it is also unlocked.

        Returns the number actually freed.
        """
        if n_pages_needed <= 0:
            return 0

        # Min-heap of (last_access, seq, node) over current evictable leaves.
        heap: list[tuple[float, int, RadixNode]] = []
        for node in self._walk():
            if node is self.root:
                continue
            if self._is_evictable(node):
                heapq.heappush(
                    heap, (node.last_access, next(self._heap_seq), node)
                )

        freed = 0
        while heap and freed < n_pages_needed:
            _, _, node = heapq.heappop(heap)
            # Re-validate: a re-pushed parent may have re-acquired
            # children/locks in between heap pushes.
            if not self._is_evictable(node):
                continue
            # Free pages back to pool.
            if node.pages:
                self.pool.free(node.pages)
                freed += len(node.pages)
                self._num_cached_pages -= len(node.pages)
                self.metrics.total_evicted_pages += len(node.pages)
            parent = node.parent
            if parent is not None:
                # Detach from parent's children dict (keyed by first page).
                parent.children.pop(tuple(node.key[: self.page_size]), None)
                if self._is_evictable(parent) and parent is not self.root:
                    heapq.heappush(
                        heap, (parent.last_access, next(self._heap_seq), parent)
                    )

        return freed

    # ── Maintenance ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop the whole tree, return every page to the pool."""
        all_pages: list[int] = []
        for node in self._walk():
            if node is self.root:
                continue
            all_pages.extend(node.pages)
        if all_pages:
            self.pool.free(all_pages)
        self.root.children.clear()
        self._num_cached_pages = 0
        self.metrics = CacheMetrics()
