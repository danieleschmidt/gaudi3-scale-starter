"""Tests for LearningRateScaler."""
import pytest
from gaudi3_scale.lr_scaler import LearningRateScaler


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

def test_init_basic():
    scaler = LearningRateScaler(base_lr=0.1)
    assert scaler.base_lr == 0.1
    assert scaler.base_batch_size == 256


def test_init_custom_base_batch():
    scaler = LearningRateScaler(base_lr=0.01, base_batch_size=128)
    assert scaler.base_batch_size == 128


def test_init_invalid_lr():
    with pytest.raises(ValueError, match="base_lr"):
        LearningRateScaler(base_lr=0.0)


def test_init_negative_lr():
    with pytest.raises(ValueError, match="base_lr"):
        LearningRateScaler(base_lr=-0.1)


def test_init_invalid_base_batch():
    with pytest.raises(ValueError, match="base_batch_size"):
        LearningRateScaler(base_lr=0.1, base_batch_size=0)


def test_repr():
    s = LearningRateScaler(0.01, 128)
    r = repr(s)
    assert "LearningRateScaler" in r
    assert "base_lr=0.01" in r


# ---------------------------------------------------------------------------
# scale() tests
# ---------------------------------------------------------------------------

def test_scale_same_batch_returns_base_lr():
    scaler = LearningRateScaler(base_lr=0.1, base_batch_size=256)
    assert abs(scaler.scale(256) - 0.1) < 1e-12


def test_scale_double_batch():
    scaler = LearningRateScaler(base_lr=0.1, base_batch_size=256)
    assert abs(scaler.scale(512) - 0.2) < 1e-12


def test_scale_half_batch():
    scaler = LearningRateScaler(base_lr=0.1, base_batch_size=256)
    assert abs(scaler.scale(128) - 0.05) < 1e-12


def test_scale_large_batch():
    scaler = LearningRateScaler(base_lr=0.001, base_batch_size=32)
    # 4096 / 32 = 128x, so lr = 0.001 * 128 = 0.128
    assert abs(scaler.scale(4096) - 0.128) < 1e-12


def test_scale_single_sample():
    scaler = LearningRateScaler(base_lr=0.256, base_batch_size=256)
    assert abs(scaler.scale(1) - 0.001) < 1e-12


def test_scale_invalid_batch_size():
    scaler = LearningRateScaler(base_lr=0.1)
    with pytest.raises(ValueError, match="actual_batch_size"):
        scaler.scale(0)


# ---------------------------------------------------------------------------
# warmup_schedule() tests
# ---------------------------------------------------------------------------

def test_warmup_step_zero():
    scaler = LearningRateScaler(0.1)
    lr = scaler.warmup_schedule(step=0, warmup_steps=10, scaled_lr=0.5)
    assert abs(lr - 0.0) < 1e-12


def test_warmup_step_half():
    scaler = LearningRateScaler(0.1)
    lr = scaler.warmup_schedule(step=5, warmup_steps=10, scaled_lr=0.5)
    assert abs(lr - 0.25) < 1e-12


def test_warmup_step_at_warmup_end():
    scaler = LearningRateScaler(0.1)
    lr = scaler.warmup_schedule(step=10, warmup_steps=10, scaled_lr=0.5)
    assert abs(lr - 0.5) < 1e-12


def test_warmup_step_beyond_warmup():
    scaler = LearningRateScaler(0.1)
    lr = scaler.warmup_schedule(step=100, warmup_steps=10, scaled_lr=0.5)
    assert abs(lr - 0.5) < 1e-12


def test_warmup_zero_warmup_steps():
    scaler = LearningRateScaler(0.1)
    # With 0 warmup steps, should immediately return scaled_lr
    lr = scaler.warmup_schedule(step=0, warmup_steps=0, scaled_lr=0.5)
    assert abs(lr - 0.5) < 1e-12


def test_warmup_linearity():
    scaler = LearningRateScaler(0.1)
    scaled_lr = 1.0
    warmup_steps = 100
    lrs = [scaler.warmup_schedule(s, warmup_steps, scaled_lr) for s in range(warmup_steps)]
    # Should be strictly increasing
    for i in range(len(lrs) - 1):
        assert lrs[i] < lrs[i + 1]
    # Should be linear
    expected = [scaled_lr * s / warmup_steps for s in range(warmup_steps)]
    for actual, exp in zip(lrs, expected):
        assert abs(actual - exp) < 1e-12


def test_warmup_invalid_step():
    scaler = LearningRateScaler(0.1)
    with pytest.raises(ValueError, match="step"):
        scaler.warmup_schedule(step=-1, warmup_steps=10, scaled_lr=0.5)


def test_warmup_invalid_warmup_steps():
    scaler = LearningRateScaler(0.1)
    with pytest.raises(ValueError, match="warmup_steps"):
        scaler.warmup_schedule(step=0, warmup_steps=-1, scaled_lr=0.5)
