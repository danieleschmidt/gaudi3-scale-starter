"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from src.gaudi3_scale.models.cluster import (
    ClusterConfig, NodeConfig, CloudProvider, InstanceType, NetworkConfig, StorageConfig
)
from src.gaudi3_scale.models.training import (
    TrainingConfig, ModelConfig, DatasetConfig, ModelType, PrecisionType, OptimizerType
)
from src.gaudi3_scale.models.monitoring import (
    MetricsConfig, HPUMetrics, SystemMetrics, AlertRule, HealthCheck, AlertSeverity, HealthStatus
)


class TestNodeConfig:
    """Test NodeConfig model."""
    
    def test_valid_node_config(self):
        """Test creating valid node configuration."""
        config = NodeConfig(
            node_id="test-node-1",
            instance_type=InstanceType.AWS_DL2Q_24XLARGE,
            hpu_count=8,
            memory_gb=512,
            storage_gb=1000
        )
        
        assert config.node_id == "test-node-1"
        assert config.instance_type == InstanceType.AWS_DL2Q_24XLARGE
        assert config.hpu_count == 8
        assert config.memory_gb == 512
        assert config.storage_gb == 1000
        assert config.network_bandwidth_gbps == 200  # Default value
    
    def test_invalid_hpu_count(self):
        """Test validation error for invalid HPU count."""
        with pytest.raises(ValidationError) as exc_info:
            NodeConfig(
                node_id="test-node-1",
                instance_type=InstanceType.AWS_DL2Q_24XLARGE,
                hpu_count=12,  # Invalid, must be 8 or 16
                memory_gb=512,
                storage_gb=1000
            )
        
        assert "HPU count must be 8 or 16" in str(exc_info.value)


class TestClusterConfig:
    """Test ClusterConfig model."""
    
    def test_valid_cluster_config(self, sample_node_config):
        """Test creating valid cluster configuration."""
        config = ClusterConfig(
            cluster_name="test-cluster",
            provider=CloudProvider.AWS,
            region="us-west-2",
            nodes=[sample_node_config]
        )
        
        assert config.cluster_name == "test-cluster"
        assert config.provider == CloudProvider.AWS
        assert config.region == "us-west-2"
        assert len(config.nodes) == 1
        assert config.total_hpus == 8
        assert config.estimated_cost_per_hour > 0
    
    def test_empty_nodes_validation(self):
        """Test validation error for empty nodes list."""
        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig(
                cluster_name="test-cluster",
                provider=CloudProvider.AWS,
                region="us-west-2",
                nodes=[]  # Empty nodes list
            )
        
        assert "At least one node must be configured" in str(exc_info.value)
    
    def test_too_many_nodes_validation(self, sample_node_config):
        """Test validation error for too many nodes."""
        with pytest.raises(ValidationError) as exc_info:
            ClusterConfig(
                cluster_name="test-cluster",
                provider=CloudProvider.AWS,
                region="us-west-2",
                nodes=[sample_node_config] * 65  # Too many nodes
            )
        
        assert "Maximum 64 nodes supported" in str(exc_info.value)
    
    def test_total_hpus_calculation(self, sample_node_config):
        """Test total HPU calculation."""
        # Create config with multiple nodes
        config = ClusterConfig(
            cluster_name="test-cluster",
            provider=CloudProvider.AWS,
            region="us-west-2",
            nodes=[sample_node_config, sample_node_config]  # 2 nodes
        )
        
        assert config.total_hpus == 16  # 2 nodes * 8 HPUs each
    
    def test_cost_estimation(self, sample_node_config):
        """Test cost estimation."""
        config = ClusterConfig(
            cluster_name="test-cluster",
            provider=CloudProvider.AWS,
            region="us-west-2",
            nodes=[sample_node_config],
            enable_spot_instances=False
        )
        
        # Should be AWS DL2Q.24xlarge cost
        assert config.estimated_cost_per_hour == 32.77
        
        # Test spot instance discount
        config.enable_spot_instances = True
        assert config.estimated_cost_per_hour == 32.77 * 0.3
    
    def test_terraform_vars_generation(self, sample_node_config):
        """Test Terraform variables generation."""
        config = ClusterConfig(
            cluster_name="test-cluster",
            provider=CloudProvider.AWS,
            region="us-west-2",
            nodes=[sample_node_config]
        )
        
        tf_vars = config.to_terraform_vars()
        
        assert tf_vars["cluster_name"] == "test-cluster"
        assert tf_vars["region"] == "us-west-2"
        assert tf_vars["node_count"] == 1
        assert tf_vars["hpu_count"] == 8
        assert tf_vars["enable_monitoring"] is True


class TestTrainingConfig:
    """Test TrainingConfig model."""
    
    def test_valid_training_config(self):
        """Test creating valid training configuration."""
        config = TrainingConfig(
            batch_size=32,
            gradient_accumulation_steps=4,
            max_epochs=3,
            learning_rate=6e-4,
            precision=PrecisionType.BF16_MIXED
        )
        
        assert config.batch_size == 32
        assert config.effective_batch_size == 128  # 32 * 4
        assert config.learning_rate == 6e-4
        assert config.precision == PrecisionType.BF16_MIXED
    
    def test_batch_size_validation(self):
        """Test batch size validation."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                batch_size=0,  # Invalid
                max_epochs=3,
                learning_rate=6e-4
            )
        
        assert "Batch size must be between 1 and 512" in str(exc_info.value)
    
    def test_learning_rate_validation(self):
        """Test learning rate validation."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                batch_size=32,
                max_epochs=3,
                learning_rate=2.0  # Invalid, > 1
            )
        
        assert "Learning rate must be between 0 and 1" in str(exc_info.value)
    
    def test_optimizer_config_generation(self):
        """Test optimizer configuration generation."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=6e-4,
            optimizer_type=OptimizerType.FUSED_ADAMW
        )
        
        opt_config = config.optimizer_config
        
        assert opt_config["lr"] == 6e-4
        assert opt_config["use_habana"] is True
        assert opt_config["betas"] == (0.9, 0.95)
    
    def test_lightning_config_generation(self):
        """Test PyTorch Lightning configuration generation."""
        config = TrainingConfig(
            batch_size=32,
            max_epochs=3,
            precision=PrecisionType.BF16_MIXED,
            gradient_clip_val=1.0
        )
        
        lightning_config = config.to_lightning_config()
        
        assert lightning_config["max_epochs"] == 3
        assert lightning_config["precision"] == "bf16-mixed"
        assert lightning_config["gradient_clip_val"] == 1.0


class TestModelConfig:
    """Test ModelConfig model."""
    
    def test_valid_model_config(self):
        """Test creating valid model configuration."""
        config = ModelConfig(
            model_type=ModelType.LLAMA,
            model_name="meta-llama/Llama-2-7b-hf",
            model_size="7B",
            hidden_size=4096,
            num_layers=32
        )
        
        assert config.model_type == ModelType.LLAMA
        assert config.model_name == "meta-llama/Llama-2-7b-hf"
        assert config.model_size == "7B"
        assert config.hidden_size == 4096
    
    def test_invalid_model_size(self):
        """Test validation error for invalid model size."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.LLAMA,
                model_name="test-model",
                model_size="5B"  # Invalid size
            )
        
        assert "Model size must be one of" in str(exc_info.value)


class TestDatasetConfig:
    """Test DatasetConfig model."""
    
    def test_valid_dataset_config(self):
        """Test creating valid dataset configuration."""
        config = DatasetConfig(
            dataset_name="wikitext-103",
            dataset_type="huggingface",
            max_length=2048,
            train_split="train"
        )
        
        assert config.dataset_name == "wikitext-103"
        assert config.dataset_type == "huggingface"
        assert config.max_length == 2048
        assert config.train_split == "train"
        assert config.padding == "max_length"  # Default value


class TestHPUMetrics:
    """Test HPUMetrics model."""
    
    def test_valid_hpu_metrics(self):
        """Test creating valid HPU metrics."""
        metrics = HPUMetrics(
            hpu_utilization=85.5,
            memory_usage=24.5,
            memory_utilization=75.2,
            throughput_tokens_per_sec=1250.0,
            effective_batch_size=32,
            steps_per_second=2.5,
            temperature_celsius=65.0,
            power_consumption_watts=350.0,
            graph_compilation_time_ms=125.5,
            lazy_mode_enabled=True,
            current_epoch=1,
            current_step=500
        )
        
        assert metrics.hpu_utilization == 85.5
        assert metrics.memory_usage == 24.5
        assert metrics.throughput_tokens_per_sec == 1250.0
        assert metrics.lazy_mode_enabled is True
    
    def test_utilization_validation(self):
        """Test validation for utilization percentages."""
        with pytest.raises(ValidationError) as exc_info:
            HPUMetrics(
                hpu_utilization=105.0,  # Invalid, > 100
                memory_usage=24.5,
                memory_utilization=75.2,
                throughput_tokens_per_sec=1250.0,
                effective_batch_size=32,
                steps_per_second=2.5,
                temperature_celsius=65.0,
                power_consumption_watts=350.0,
                graph_compilation_time_ms=125.5,
                lazy_mode_enabled=True,
                current_epoch=1,
                current_step=500
            )
        
        assert "Utilization must be between 0 and 100" in str(exc_info.value)


class TestAlertRule:
    """Test AlertRule model."""
    
    def test_valid_alert_rule(self):
        """Test creating valid alert rule."""
        rule = AlertRule(
            name="HighHPUUtilization",
            severity=AlertSeverity.WARNING,
            metric_name="hpu_utilization",
            condition="> 90",
            duration=300,
            notification_channels=["slack", "email"]
        )
        
        assert rule.name == "HighHPUUtilization"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.condition == "> 90"
        assert rule.duration == 300
        assert rule.enabled is True  # Default value


class TestHealthCheck:
    """Test HealthCheck model."""
    
    def test_valid_health_check(self):
        """Test creating valid health check."""
        check = HealthCheck(
            service_name="hpu-monitor",
            check_type="http",
            endpoint="/health",
            interval=30,
            timeout=10
        )
        
        assert check.service_name == "hpu-monitor"
        assert check.check_type == "http"
        assert check.endpoint == "/health"
        assert check.status == HealthStatus.UNKNOWN  # Default
    
    def test_timing_validation(self):
        """Test validation for timing values."""
        with pytest.raises(ValidationError) as exc_info:
            HealthCheck(
                service_name="test-service",
                check_type="http",
                interval=-5  # Invalid, must be positive
            )
        
        assert "Timing values must be positive" in str(exc_info.value)


class TestMetricsConfig:
    """Test MetricsConfig model."""
    
    def test_valid_metrics_config(self):
        """Test creating valid metrics configuration."""
        config = MetricsConfig(
            enabled=True,
            collection_interval=15,
            retention_days=30,
            hpu_metrics_enabled=True,
            alerting_enabled=True
        )
        
        assert config.enabled is True
        assert config.collection_interval == 15
        assert config.retention_days == 30
        assert config.hpu_metrics_enabled is True
        assert config.grafana_port == 3000  # Default value
    
    def test_default_alert_rules(self):
        """Test default alert rules generation."""
        config = MetricsConfig()
        default_rules = config.get_default_alert_rules()
        
        assert len(default_rules) > 0
        
        # Check for specific default rules
        rule_names = [rule.name for rule in default_rules]
        assert "HighHPUUtilization" in rule_names
        assert "HighMemoryUsage" in rule_names
        assert "LowTrainingThroughput" in rule_names
        assert "HighTemperature" in rule_names