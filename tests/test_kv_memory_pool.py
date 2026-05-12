"""Unit tests for KVMemoryPool — pure CPU, no GPU needed."""

from __future__ import annotations

import pytest
import torch

from miniengine.kv_memory_pool import KVMemoryPool


def _make_pool(num_pages=8, page_size=4, num_layers=2, num_kv_heads=2, head_dim=8):
    return KVMemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
    )


def test_initial_state_all_free():
    pool = _make_pool(num_pages=8)
    assert pool.num_free == 8


def test_kv_caches_shape_and_count():
    pool = _make_pool(num_pages=8, page_size=4, num_layers=3, num_kv_heads=2, head_dim=8)
    caches = pool.kv_caches
    assert len(caches) == 3
    for k, v in caches:
        assert k.shape == (8, 4, 2, 8)
        assert v.shape == (8, 4, 2, 8)


def test_kv_caches_identity_is_stable():
    """The pool must hand out the SAME tensors every time — flash-attn
    holds these references across the whole run."""
    pool = _make_pool()
    caches_a = pool.kv_caches
    caches_b = pool.kv_caches
    for (ka, va), (kb, vb) in zip(caches_a, caches_b):
        assert ka.data_ptr() == kb.data_ptr()
        assert va.data_ptr() == vb.data_ptr()


def test_allocate_returns_distinct_indices():
    pool = _make_pool(num_pages=8)
    pages = pool.allocate(3)
    assert len(pages) == 3
    assert len(set(pages)) == 3
    assert pool.num_free == 5


def test_allocate_then_free_round_trips():
    pool = _make_pool(num_pages=8)
    pages = pool.allocate(5)
    assert pool.num_free == 3
    pool.free(pages)
    assert pool.num_free == 8


def test_allocate_more_than_available_raises():
    pool = _make_pool(num_pages=4)
    with pytest.raises(RuntimeError, match="out of pages"):
        pool.allocate(5)


def test_allocate_does_not_leak_on_failure():
    pool = _make_pool(num_pages=4)
    with pytest.raises(RuntimeError):
        pool.allocate(5)
    # Failed allocation should leave the pool untouched.
    assert pool.num_free == 4


def test_pages_needed():
    pool = _make_pool(page_size=4)
    assert pool.pages_needed(0) == 0
    assert pool.pages_needed(1) == 1
    assert pool.pages_needed(4) == 1
    assert pool.pages_needed(5) == 2
    assert pool.pages_needed(16) == 4
    assert pool.pages_needed(17) == 5


def test_freed_pages_are_reusable():
    pool = _make_pool(num_pages=4)
    a = pool.allocate(4)
    pool.free(a[:2])
    b = pool.allocate(2)
    # The two freed pages should now be live again.
    assert set(b).issubset(set(a[:2]))
    assert pool.num_free == 0


def test_from_budget_derives_num_pages():
    # 4 layers, 2 KV heads, head_dim=8, page_size=4, fp32 (4 bytes)
    # bytes/page = 2 * 4 * 4 * 2 * 8 * 4 = 2048
    # budget = 8192 → 4 pages
    pool = KVMemoryPool.from_budget(
        num_layers=4,
        num_kv_heads=2,
        head_dim=8,
        page_size=4,
        dtype=torch.float32,
        device="cpu",
        bytes_budget=8192,
    )
    assert pool.num_pages == 4
    assert pool.num_free == 4


def test_from_budget_minimum_one_page():
    # Tiny budget → still produces a valid 1-page pool.
    pool = KVMemoryPool.from_budget(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        page_size=4,
        dtype=torch.float32,
        device="cpu",
        bytes_budget=1,
    )
    assert pool.num_pages == 1


def test_invalid_construction():
    with pytest.raises(ValueError):
        _make_pool(num_pages=0)
    with pytest.raises(ValueError):
        _make_pool(page_size=0)
