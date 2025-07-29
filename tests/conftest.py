"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_habana_torch():
    """Mock habana_frameworks.torch module."""
    mock_htorch = Mock()
    mock_htorch.hpu.is_available.return_value = True
    mock_htorch.hpu.device_count.return_value = 8
    mock_htorch.hpu.memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
    mock_htorch.hpu.memory_reserved.return_value = 2 * 1024 * 1024 * 1024  # 2GB
    return mock_htorch


@pytest.fixture
def mock_pytorch_lightning():
    """Mock PyTorch Lightning module."""
    mock_pl = Mock()
    mock_trainer = Mock()
    mock_pl.Trainer.return_value = mock_trainer
    return mock_pl, mock_trainer


@pytest.fixture
def mock_torch():
    """Mock PyTorch module."""
    mock_torch = Mock()
    mock_optimizer = Mock()
    mock_torch.optim.AdamW.return_value = mock_optimizer
    return mock_torch, mock_optimizer