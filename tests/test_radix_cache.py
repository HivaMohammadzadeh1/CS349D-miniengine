"""Unit tests for RadixCache — pure data structure, no GPU."""

from __future__ import annotations

import pytest

from miniengine.radix_cache import RadixCache


class FakePool:
    """Minimal stand-in for KVMemoryPool. Records freed pages."""

    def __init__(self, page_size: int = 4):
        self.page_size = page_size
        self.freed: list[int] = []

    def free(self, pages: list[int]) -> None:
        self.freed.extend(pages)


def _cache(page_size: int = 4) -> tuple[RadixCache, FakePool]:
    pool = FakePool(page_size=page_size)
    return RadixCache(pool), pool


# ── match_prefix ────────────────────────────────────────────────────────


def test_empty_cache_no_match():
    cache, _ = _cache()
    result = cache.match_prefix([1, 2, 3, 4])
    assert result.matched_tokens == 0
    assert result.matched_pages == []
    assert result.last_node is cache.root


def test_insert_then_exact_match():
    cache, _ = _cache(page_size=4)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    pages = [100, 101]
    cache.insert_and_return(tokens, pages)

    result = cache.match_prefix(tokens)
    assert result.matched_tokens == 8
    assert result.matched_pages == [100, 101]
    assert result.last_node is not cache.root


def test_partial_match_at_page_boundary():
    cache, _ = _cache(page_size=4)
    cache.insert_and_return([1, 2, 3, 4, 5, 6, 7, 8], [100, 101])

    # Query with prefix that diverges at page 2.
    result = cache.match_prefix([1, 2, 3, 4, 9, 9, 9, 9])
    assert result.matched_tokens == 4
    assert result.matched_pages == [100]


def test_query_shorter_than_cached_match_partial():
    cache, _ = _cache(page_size=4)
    cache.insert_and_return([1, 2, 3, 4, 5, 6, 7, 8], [100, 101])

    # Query shorter than what's cached — still page-aligned match.
    result = cache.match_prefix([1, 2, 3, 4])
    assert result.matched_tokens == 4
    assert result.matched_pages == [100]


def test_unaligned_query_truncates_to_page():
    cache, _ = _cache(page_size=4)
    cache.insert_and_return([1, 2, 3, 4], [100])

    # Query has 6 tokens; matches the first 4 (one page).
    result = cache.match_prefix([1, 2, 3, 4, 5, 6])
    assert result.matched_tokens == 4


def test_query_shorter_than_page_returns_zero():
    cache, _ = _cache(page_size=4)
    cache.insert_and_return([1, 2, 3, 4], [100])

    # Three-token query can't match a 4-token page.
    result = cache.match_prefix([1, 2, 3])
    assert result.matched_tokens == 0


# ── insert_and_return ──────────────────────────────────────────────────


def test_insert_rejects_unaligned_tokens():
    cache, _ = _cache(page_size=4)
    with pytest.raises(ValueError):
        cache.insert_and_return([1, 2, 3], [100])


def test_insert_duplicate_returns_redundant_pages():
    cache, _ = _cache(page_size=4)
    leaf1, red1 = cache.insert_and_return([1, 2, 3, 4, 5, 6, 7, 8], [100, 101])
    assert red1 == []
    assert cache.num_cached_pages == 2

    # Second insert with same tokens, DIFFERENT pages.
    leaf2, red2 = cache.insert_and_return([1, 2, 3, 4, 5, 6, 7, 8], [200, 201])
    assert leaf2 is leaf1  # same node
    assert red2 == [200, 201]
    assert cache.num_cached_pages == 2  # no new pages added


def test_insert_branching_creates_split():
    cache, _ = _cache(page_size=4)
    cache.insert_and_return([1, 2, 3, 4, 5, 6, 7, 8], [100, 101])
    # New insert shares the first page, diverges at the second.
    leaf2, red = cache.insert_and_return([1, 2, 3, 4, 9, 9, 9, 9], [200, 201])
    assert red == [200]  # page 0 (shared prefix) is redundant
    # Tree now has a split node with two children at depth-1.
    children = cache.root.children
    assert len(children) == 1
    split = children[1]  # token 1 keyed it in
    assert len(split.children) == 2
    assert cache.num_cached_pages == 3  # page 100 + page 101 + page 201


def test_insert_longer_extends_existing():
    cache, _ = _cache(page_size=4)
    cache.insert_and_return([1, 2, 3, 4], [100])
    # Insert a longer sequence with the same prefix.
    leaf2, red = cache.insert_and_return(
        [1, 2, 3, 4, 5, 6, 7, 8], [100, 101]
    )
    # page 100 was already cached → redundant. page 101 is new.
    assert red == [100]
    assert cache.num_cached_pages == 2


# ── lock_ref + eviction ────────────────────────────────────────────────


def test_inc_dec_lock_ref():
    cache, _ = _cache(page_size=4)
    leaf, _ = cache.insert_and_return([1, 2, 3, 4], [100])
    cache.inc_lock_ref(leaf)
    assert leaf.ref_count == 1

    cache.dec_lock_ref(leaf)
    assert leaf.ref_count == 0


def test_evict_unlocked_leaf():
    cache, pool = _cache(page_size=4)
    cache.insert_and_return([1, 2, 3, 4], [100])
    cache.insert_and_return([5, 6, 7, 8], [200])

    freed = cache.evict(1)
    assert freed >= 1
    # At least one page was returned to pool.
    assert len(pool.freed) >= 1


def test_evict_respects_locked_subtree():
    cache, pool = _cache(page_size=4)
    leaf_locked, _ = cache.insert_and_return([1, 2, 3, 4], [100])
    leaf_free, _ = cache.insert_and_return([5, 6, 7, 8], [200])
    cache.inc_lock_ref(leaf_locked)

    freed = cache.evict(99)  # ask for more than we have
    # Locked page must not be freed.
    assert 100 not in pool.freed
    # Unlocked one should be.
    assert 200 in pool.freed
    assert freed == 1


def test_evict_oldest_first():
    cache, pool = _cache(page_size=4)
    # Insert in order: page 100 first, then page 200.
    leaf_old, _ = cache.insert_and_return([1, 2, 3, 4], [100])
    # Force the access timestamp ordering.
    import time

    time.sleep(0.001)
    leaf_new, _ = cache.insert_and_return([5, 6, 7, 8], [200])

    freed = cache.evict(1)
    assert freed == 1
    # The older one (100) goes first.
    assert pool.freed == [100]


def test_evict_cascades_to_parent_after_child_freed():
    cache, pool = _cache(page_size=4)
    # Build a chain: insert long, then short; the long version has
    # an intermediate split node.
    cache.insert_and_return([1, 2, 3, 4, 5, 6, 7, 8], [100, 101])
    cache.insert_and_return([1, 2, 3, 4, 9, 9, 9, 9], [100, 201])

    # Force eviction of everything.
    freed = cache.evict(99)
    # All non-redundant pages should be freed eventually:
    # 100 (split prefix) + 101 + 201
    assert sorted(pool.freed) == sorted([100, 101, 201])
    assert cache.num_cached_pages == 0


def test_reset_returns_all_pages():
    cache, pool = _cache(page_size=4)
    cache.insert_and_return([1, 2, 3, 4, 5, 6, 7, 8], [100, 101])
    cache.insert_and_return([9, 10, 11, 12], [200])

    cache.reset()
    assert cache.num_cached_pages == 0
    assert sorted(pool.freed) == [100, 101, 200]
    assert cache.root.children == {}


# ── metrics ────────────────────────────────────────────────────────────


def test_metrics_track_lookups():
    cache, _ = _cache(page_size=4)
    cache.insert_and_return([1, 2, 3, 4, 5, 6, 7, 8], [100, 101])

    cache.match_prefix([1, 2, 3, 4, 9, 9])
    cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])

    assert cache.metrics.total_lookups == 2
    assert cache.metrics.total_query_tokens == 6 + 8
    assert cache.metrics.total_hit_tokens == 4 + 8
    assert cache.metrics.hit_rate == (4 + 8) / (6 + 8)


def test_metrics_track_inserts_and_evicts():
    cache, _ = _cache(page_size=4)
    cache.insert_and_return([1, 2, 3, 4, 5, 6, 7, 8], [100, 101])
    assert cache.metrics.total_inserted_pages == 2

    cache.evict(99)
    assert cache.metrics.total_evicted_pages == 2
