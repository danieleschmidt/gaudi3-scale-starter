"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import os
import sys


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory that's cleaned up after test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_habana_torch():
    """Mock habana_frameworks.torch module."""
    mock_htorch = Mock()
    mock_htorch.hpu.is_available.return_value = True
    mock_htorch.hpu.device_count.return_value = 8
    mock_htorch.hpu.memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
    mock_htorch.hpu.memory_reserved.return_value = 2 * 1024 * 1024 * 1024  # 2GB
    mock_htorch.hpu.current_device.return_value = 0
    mock_htorch.hpu.get_device_name.return_value = "Gaudi3"
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
    mock_torch.cuda.is_available.return_value = False
    mock_torch.version.cuda = None
    mock_torch.backends.cudnn.enabled = False
    return mock_torch, mock_optimizer


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration for testing."""
    return {
        "model": {
            "name": "test_model",
            "batch_size": 32,
            "learning_rate": 0.001,
            "precision": "bf16-mixed"
        },
        "training": {
            "epochs": 10,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 4
        },
        "hardware": {
            "devices": 8,
            "accelerator": "hpu",
            "strategy": "habana_ddp"
        },
        "monitoring": {
            "wandb_project": "test_project",
            "log_every_n_steps": 50
        }
    }


@pytest.fixture
def mock_model():
    """Mock PyTorch Lightning model for testing."""
    mock_model = Mock()
    mock_model.training_step.return_value = Mock(loss=0.5)
    mock_model.validation_step.return_value = {"val_loss": 0.3}
    mock_model.test_step.return_value = {"test_loss": 0.25}
    mock_model.configure_optimizers.return_value = Mock()
    mock_model.parameters.return_value = []
    return mock_model


@pytest.fixture
def mock_dataloader():
    """Mock DataLoader for testing."""
    mock_dataloader = Mock()
    mock_dataloader.__len__.return_value = 100
    mock_dataloader.__iter__.return_value = iter([
        {"input_ids": Mock(), "labels": Mock()} for _ in range(10)
    ])
    return mock_dataloader


@pytest.fixture
def environment_vars():
    """Fixture for managing environment variables in tests."""
    original_env = os.environ.copy()
    
    def set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[key] = str(value)
    
    def reset_env():
        os.environ.clear()
        os.environ.update(original_env)
    
    yield set_env
    reset_env()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment with necessary mocks and configurations."""
    if 'habana_frameworks' not in sys.modules:
        sys.modules['habana_frameworks'] = Mock()
        sys.modules['habana_frameworks.torch'] = Mock()
        sys.modules['habana_frameworks.torch.hpu'] = Mock()
    
    os.environ.setdefault('TESTING', '1')
    os.environ.setdefault('PT_HPU_LAZY_MODE', '0')
    
    yield
    
    test_env_vars = ['TESTING', 'PT_HPU_LAZY_MODE']
    for var in test_env_vars:
        os.environ.pop(var, None)


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        'warmup_rounds': 3,
        'benchmark_rounds': 10,
        'max_time': 30.0,
        'min_rounds': 5,
        'disable_gc': True,
        'timer': 'time.perf_counter'
    }


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU/HPU hardware")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance)
        
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        if any(keyword in item.name.lower() for keyword in ['hpu', 'gpu', 'habana']):
            item.add_marker(pytest.mark.gpu)