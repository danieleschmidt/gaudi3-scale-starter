"""Pytest configuration and fixtures for Gaudi 3 Scale tests."""

import os
import tempfile
from typing import Dict, Generator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, patch

from src.gaudi3_scale.database.connection import DatabaseConnection
from src.gaudi3_scale.database.models import Base
from src.gaudi3_scale.cache.redis_cache import RedisCache
from src.gaudi3_scale.models.cluster import ClusterConfig, NodeConfig, CloudProvider, InstanceType
from src.gaudi3_scale.models.training import TrainingConfig, ModelConfig, DatasetConfig
from src.gaudi3_scale.services.cluster_service import ClusterService
from src.gaudi3_scale.services.cost_service import CostAnalyzer


@pytest.fixture(scope="session")
def test_db_url():
    """Create temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_url = f"sqlite:///{tmp.name}"
        yield db_url
        os.unlink(tmp.name)


@pytest.fixture(scope="session")
def test_engine(test_db_url):
    """Create test database engine."""
    engine = create_engine(test_db_url, echo=False)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_engine):
    """Create database session for testing."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    
    yield session
    
    session.rollback()
    session.close()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    with patch('redis.Redis') as mock:
        mock_client = Mock()
        mock.return_value = mock_client
        
        # Configure mock methods
        mock_client.set.return_value = True
        mock_client.get.return_value = None
        mock_client.delete.return_value = 1
        mock_client.exists.return_value = False
        mock_client.expire.return_value = True
        mock_client.ttl.return_value = -1
        mock_client.scan.return_value = (0, [])
        mock_client.ping.return_value = True
        mock_client.info.return_value = {
            'redis_version': '6.2.0',
            'used_memory': 1024000,
            'used_memory_human': '1.00M',
            'connected_clients': 1,
            'total_commands_processed': 100,
            'keyspace_hits': 80,
            'keyspace_misses': 20
        }
        
        yield mock_client


@pytest.fixture
def redis_cache(mock_redis):
    """Create Redis cache instance for testing."""
    with patch('src.gaudi3_scale.cache.redis_cache.RedisConnection') as mock_conn:
        mock_conn.return_value.get_client.return_value = mock_redis
        cache = RedisCache()
        yield cache


@pytest.fixture
def sample_node_config():
    """Create sample node configuration."""
    return NodeConfig(
        node_id="test-node-1",
        instance_type=InstanceType.AWS_DL2Q_24XLARGE,
        hpu_count=8,
        memory_gb=512,
        storage_gb=1000,
        network_bandwidth_gbps=200
    )


@pytest.fixture
def sample_cluster_config(sample_node_config):
    """Create sample cluster configuration."""
    return ClusterConfig(
        cluster_name="test-cluster",
        provider=CloudProvider.AWS,
        region="us-west-2",
        nodes=[sample_node_config],
        enable_monitoring=True,
        enable_spot_instances=False
    )


@pytest.fixture
def sample_model_config():
    """Create sample model configuration."""
    return ModelConfig(
        model_type="llama",
        model_name="meta-llama/Llama-2-7b-hf",
        model_size="7B",
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        sequence_length=2048
    )


@pytest.fixture
def sample_dataset_config():
    """Create sample dataset configuration."""
    return DatasetConfig(
        dataset_name="wikitext-103",
        dataset_type="huggingface",
        max_length=2048,
        train_split="train",
        validation_split="validation"
    )


@pytest.fixture
def sample_training_config():
    """Create sample training configuration."""
    return TrainingConfig(
        batch_size=32,
        gradient_accumulation_steps=4,
        max_epochs=3,
        learning_rate=6e-4,
        precision="bf16-mixed",
        devices=8
    )


@pytest.fixture
def cluster_service(sample_cluster_config):
    """Create cluster service instance."""
    return ClusterService(sample_cluster_config)


@pytest.fixture
def cost_analyzer():
    """Create cost analyzer instance."""
    return CostAnalyzer()


@pytest.fixture
def mock_habana():
    """Mock Habana frameworks for testing."""
    with patch('src.gaudi3_scale.accelerator.habana_frameworks') as mock:
        mock_htorch = Mock()
        mock_htorch.hpu.is_available.return_value = True
        mock_htorch.hpu.device_count.return_value = 8
        mock_htorch.hpu.memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        mock_htorch.hpu.memory_reserved.return_value = 2048 * 1024 * 1024   # 2GB
        
        mock.torch = mock_htorch
        yield mock


@pytest.fixture
def mock_pytorch_lightning():
    """Mock PyTorch Lightning for testing."""
    with patch('pytorch_lightning.Trainer') as mock_trainer:
        trainer_instance = Mock()
        trainer_instance.fit.return_value = None
        mock_trainer.return_value = trainer_instance
        yield trainer_instance


@pytest.fixture
def sample_metrics_data():
    """Create sample metrics data for testing."""
    return [
        {
            "metric_name": "hpu_utilization",
            "value": 85.5,
            "unit": "percent",
            "timestamp": "2025-01-15T10:00:00Z",
            "labels": {"node": "test-node-1", "hpu_id": "0"}
        },
        {
            "metric_name": "memory_usage",
            "value": 24.5,
            "unit": "GB",
            "timestamp": "2025-01-15T10:00:00Z",
            "labels": {"node": "test-node-1", "hpu_id": "0"}
        },
        {
            "metric_name": "throughput",
            "value": 1250.0,
            "unit": "tokens_per_second",
            "timestamp": "2025-01-15T10:00:00Z",
            "labels": {"node": "test-node-1", "job_id": "test-job-1"}
        }
    ]


@pytest.fixture
def sample_training_job_data():
    """Create sample training job data."""
    return {
        "name": "test-training-job",
        "model_name": "llama-7b",
        "model_size": "7B",
        "dataset_name": "wikitext-103",
        "batch_size": 32,
        "learning_rate": 6e-4,
        "max_epochs": 3,
        "precision": "bf16-mixed",
        "devices_used": 8,
        "status": "running"
    }


@pytest.fixture
def mock_terraform():
    """Mock Terraform operations."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Terraform operations completed successfully"
        mock_run.return_value.stderr = ""
        yield mock_run


@pytest.fixture
def mock_wandb():
    """Mock Weights & Biases integration."""
    with patch('wandb.init') as mock_init, \
         patch('wandb.log') as mock_log, \
         patch('wandb.finish') as mock_finish:
        
        mock_run = Mock()
        mock_run.id = "test-run-123"
        mock_init.return_value = mock_run
        
        yield {
            'init': mock_init,
            'log': mock_log,
            'finish': mock_finish,
            'run': mock_run
        }


@pytest.fixture
def environment_variables():
    """Set test environment variables."""
    original_env = os.environ.copy()
    
    test_env = {
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5432',
        'POSTGRES_DB': 'test_db',
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_password',
        'REDIS_URL': 'redis://localhost:6379/1',
        'DEBUG': 'true',
        'ENVIRONMENT': 'test',
        'SECRET_KEY': 'test-secret-key-32-chars-long!!',
        'AWS_ACCESS_KEY_ID': 'test-access-key',
        'AWS_SECRET_ACCESS_KEY': 'test-secret-key',
        'AWS_DEFAULT_REGION': 'us-west-2'
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def suppress_logging():
    """Suppress logging during tests."""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def api_client():
    """Create test API client."""
    from fastapi.testclient import TestClient
    try:
        from src.gaudi3_scale.api.main import app
        client = TestClient(app)
        yield client
    except ImportError:
        # API not implemented yet
        yield None


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_habana: mark test as requiring Habana hardware"
    )
    config.addinivalue_line(
        "markers", "requires_redis: mark test as requiring Redis"
    )
    config.addinivalue_line(
        "markers", "requires_postgres: mark test as requiring PostgreSQL"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location and name."""
    for item in items:
        # Add unit test marker to all tests by default
        if not any(marker.name in ['integration', 'slow'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add hardware requirement markers based on test content
        if "habana" in item.name.lower() or "hpu" in item.name.lower():
            item.add_marker(pytest.mark.requires_habana)
        
        if "redis" in item.name.lower() or "cache" in item.name.lower():
            item.add_marker(pytest.mark.requires_redis)
        
        if "postgres" in item.name.lower() or "database" in item.name.lower():
            item.add_marker(pytest.mark.requires_postgres)