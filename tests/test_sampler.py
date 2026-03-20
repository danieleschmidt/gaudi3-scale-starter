"""Tests for LargeScaleBatchSampler."""
import pytest
from gaudi3_scale.sampler import LargeScaleBatchSampler


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

def test_init_basic():
    s = LargeScaleBatchSampler(dataset_size=1000, batch_size=32)
    assert s.dataset_size == 1000
    assert s.batch_size == 32
    assert s.accumulation_steps == 4  # default


def test_init_custom_accumulation():
    s = LargeScaleBatchSampler(1000, 32, accumulation_steps=8)
    assert s.accumulation_steps == 8


def test_init_invalid_dataset_size():
    with pytest.raises(ValueError, match="dataset_size"):
        LargeScaleBatchSampler(dataset_size=0, batch_size=32)


def test_init_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        LargeScaleBatchSampler(dataset_size=100, batch_size=0)


def test_init_invalid_accumulation_steps():
    with pytest.raises(ValueError, match="accumulation_steps"):
        LargeScaleBatchSampler(dataset_size=100, batch_size=8, accumulation_steps=0)


def test_repr():
    s = LargeScaleBatchSampler(500, 16, 2)
    r = repr(s)
    assert "LargeScaleBatchSampler" in r
    assert "dataset_size=500" in r


# ---------------------------------------------------------------------------
# effective_batch_size property
# ---------------------------------------------------------------------------

def test_effective_batch_size():
    s = LargeScaleBatchSampler(1000, 32, 4)
    assert s.effective_batch_size == 128


def test_effective_batch_size_accumulation_1():
    s = LargeScaleBatchSampler(1000, 64, 1)
    assert s.effective_batch_size == 64


# ---------------------------------------------------------------------------
# __len__ tests
# ---------------------------------------------------------------------------

def test_len_basic():
    # 1000 samples / 32 batch = 31 micro batches, 31 // 4 = 7 macro batches
    s = LargeScaleBatchSampler(1000, 32, 4)
    assert len(s) == 7


def test_len_exact_division():
    # 256 / 32 = 8 micro, 8 // 4 = 2 macro
    s = LargeScaleBatchSampler(256, 32, 4)
    assert len(s) == 2


def test_len_single_accumulation_step():
    # 100 / 10 = 10 micro, 10 // 1 = 10 macro
    s = LargeScaleBatchSampler(100, 10, 1)
    assert len(s) == 10


# ---------------------------------------------------------------------------
# __iter__ tests
# ---------------------------------------------------------------------------

def test_iter_yields_correct_count():
    s = LargeScaleBatchSampler(256, 32, 4)
    batches = list(s)
    assert len(batches) == len(s)


def test_iter_macro_batch_size():
    s = LargeScaleBatchSampler(256, 32, 4)
    for macro_batch in s:
        assert len(macro_batch) == s.effective_batch_size


def test_iter_indices_in_range():
    s = LargeScaleBatchSampler(100, 10, 1)
    for macro_batch in s:
        for idx in macro_batch:
            assert 0 <= idx < 100


def test_iter_no_duplicate_indices_within_epoch():
    s = LargeScaleBatchSampler(200, 10, 2)
    all_indices = []
    for macro_batch in s:
        all_indices.extend(macro_batch)
    # All indices should be unique within one epoch's worth of data
    assert len(all_indices) == len(set(all_indices))


def test_iter_is_repeatable():
    """Calling iter twice should give different orderings (shuffled) but same count."""
    s = LargeScaleBatchSampler(200, 10, 2)
    count1 = sum(1 for _ in s)
    count2 = sum(1 for _ in s)
    assert count1 == count2
