"""Unit tests for service layer components."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.gaudi3_scale.services.cluster_service import ClusterService
from src.gaudi3_scale.services.cost_service import CostAnalyzer, BaselineHardware
from src.gaudi3_scale.models.cluster import ClusterConfig, CloudProvider
from src.gaudi3_scale.models.monitoring import HealthStatus


class TestClusterService:
    """Test ClusterService functionality."""
    
    def test_validate_cluster_config_valid(self, cluster_service):
        """Test validation of valid cluster configuration."""
        is_valid, errors = cluster_service.validate_cluster_config()
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_cluster_config_invalid_hpu_count(self, sample_node_config):
        """Test validation with too many HPUs."""
        # Create cluster with too many nodes
        nodes = [sample_node_config] * 65  # 65 * 8 = 520 HPUs (exceeds 512 limit)
        
        config = ClusterConfig(
            cluster_name="test-cluster",
            provider=CloudProvider.AWS,
            region="us-west-2",
            nodes=nodes
        )
        
        service = ClusterService(config)
        is_valid, errors = service.validate_cluster_config()
        
        assert is_valid is False
        assert any("exceeds maximum of 512" in error for error in errors)
    
    def test_estimate_deployment_time(self, cluster_service):
        """Test deployment time estimation."""
        estimated_time = cluster_service.estimate_deployment_time()
        
        # Should include base time + node time
        expected_minutes = 10 + (3 * 1)  # 1 node
        assert estimated_time.total_seconds() == expected_minutes * 60
    
    def test_estimate_deployment_time_with_monitoring(self, sample_cluster_config):
        """Test deployment time estimation with monitoring enabled."""
        sample_cluster_config.enable_monitoring = True
        service = ClusterService(sample_cluster_config)
        
        estimated_time = service.estimate_deployment_time()
        
        # Should include monitoring setup time
        expected_minutes = 10 + (3 * 1) + 5  # Base + node + monitoring
        assert estimated_time.total_seconds() == expected_minutes * 60
    
    def test_get_resource_requirements(self, cluster_service):
        """Test resource requirements calculation."""
        requirements = cluster_service.get_resource_requirements()
        
        assert requirements["total_nodes"] == 1
        assert requirements["total_hpus"] == 8
        assert requirements["total_memory_gb"] == 512
        assert requirements["estimated_cost_per_hour"] > 0
        assert requirements["estimated_cost_per_month"] > 0
    
    def test_generate_terraform_config_aws(self, cluster_service):
        """Test Terraform configuration generation for AWS."""
        tf_config = cluster_service.generate_terraform_config()
        
        assert tf_config["cluster_name"] == "test-cluster"
        assert tf_config["region"] == "us-west-2"
        assert tf_config["provider"] == "aws"
        assert "availability_zones" in tf_config
        assert tf_config["enable_efa"] is True
    
    def test_generate_terraform_config_azure(self, sample_node_config):
        """Test Terraform configuration generation for Azure."""
        config = ClusterConfig(
            cluster_name="test-cluster",
            provider=CloudProvider.AZURE,
            region="westus3",
            nodes=[sample_node_config]
        )
        
        service = ClusterService(config)
        tf_config = service.generate_terraform_config()
        
        assert tf_config["provider"] == "azure"
        assert "resource_group_name" in tf_config
        assert tf_config["enable_accelerated_networking"] is True
    
    def test_check_cluster_health(self, cluster_service):
        """Test cluster health checking."""
        health_status = cluster_service.check_cluster_health()
        
        assert "node-1" in health_status
        assert "networking" in health_status
        assert "storage" in health_status
        
        # All should be healthy in mock
        for service_name, status in health_status.items():
            assert status == HealthStatus.HEALTHY
    
    def test_scale_cluster_valid(self, cluster_service):
        """Test valid cluster scaling."""
        cluster_service.config.max_nodes = 5
        cluster_service.config.min_nodes = 1
        
        result = cluster_service.scale_cluster(target_nodes=3)
        
        assert result["status"] == "initiated"
        assert result["scaling_direction"] == "up"
        assert result["nodes_to_change"] == 2
        assert result["current_nodes"] == 1
        assert result["target_nodes"] == 3
    
    def test_scale_cluster_no_change(self, cluster_service):
        """Test cluster scaling with no change needed."""
        result = cluster_service.scale_cluster(target_nodes=1)
        
        assert result["status"] == "no_change"
    
    def test_scale_cluster_exceeds_max(self, cluster_service):
        """Test cluster scaling exceeding maximum."""
        cluster_service.config.max_nodes = 5
        
        result = cluster_service.scale_cluster(target_nodes=10)
        
        assert result["status"] == "error"
        assert "exceeds maximum" in result["message"]
    
    def test_get_cluster_metrics(self, cluster_service):
        """Test cluster metrics retrieval."""
        metrics = cluster_service.get_cluster_metrics()
        
        assert "cluster_name" in metrics
        assert "total_nodes" in metrics
        assert "total_hpus" in metrics
        assert "cluster_utilization" in metrics
        assert "estimated_monthly_cost" in metrics
        
        assert metrics["cluster_name"] == "test-cluster"
        assert metrics["total_nodes"] == 1
        assert metrics["total_hpus"] == 8


class TestCostAnalyzer:
    """Test CostAnalyzer functionality."""
    
    def test_analyze_cluster_cost_aws(self, cost_analyzer, sample_cluster_config):
        """Test cost analysis for AWS cluster."""
        analysis = cost_analyzer.analyze_cluster_cost(sample_cluster_config, duration_hours=720)
        
        assert "total_cost" in analysis
        assert "cost_per_hour" in analysis
        assert "cost_breakdown" in analysis
        assert "cost_per_hpu" in analysis
        
        assert analysis["total_cost"] > 0
        assert analysis["cost_per_hour"] > 0
        assert analysis["currency"] == "USD"
    
    def test_analyze_cluster_cost_onprem(self, cost_analyzer, sample_node_config):
        """Test cost analysis for on-premises cluster."""
        config = ClusterConfig(
            cluster_name="test-cluster",
            provider=CloudProvider.ONPREM,
            region="datacenter-1",
            nodes=[sample_node_config]
        )
        
        analysis = cost_analyzer.analyze_cluster_cost(config, duration_hours=720)
        
        # On-premises should have zero compute cost
        assert analysis["cost_breakdown"]["compute"] == 0.0
    
    def test_analyze_cluster_cost_spot_instances(self, cost_analyzer, sample_cluster_config):
        """Test cost analysis with spot instances enabled."""
        sample_cluster_config.enable_spot_instances = True
        
        analysis = cost_analyzer.analyze_cluster_cost(sample_cluster_config, duration_hours=720)
        
        # Spot instances should be cheaper
        spot_cost = analysis["cost_breakdown"]["compute"]
        
        # Disable spot instances and compare
        sample_cluster_config.enable_spot_instances = False
        regular_analysis = cost_analyzer.analyze_cluster_cost(sample_cluster_config, duration_hours=720)
        regular_cost = regular_analysis["cost_breakdown"]["compute"]
        
        assert spot_cost < regular_cost
    
    def test_compare_with_baseline_h100(self, cost_analyzer, sample_cluster_config):
        """Test cost comparison with H100 baseline."""
        comparison = cost_analyzer.compare_with_baseline(
            sample_cluster_config,
            BaselineHardware.H100,
            duration_hours=720
        )
        
        assert "gaudi3_cost" in comparison
        assert "baseline_cost" in comparison
        assert "cost_savings" in comparison
        assert "savings_percentage" in comparison
        assert "cost_ratio" in comparison
        assert "recommendation" in comparison
        
        assert comparison["baseline_hardware"] == "h100"
        assert comparison["cost_savings"] > 0  # Gaudi 3 should be cheaper
        assert comparison["savings_percentage"] > 0
    
    def test_compare_with_baseline_a100(self, cost_analyzer, sample_cluster_config):
        """Test cost comparison with A100 baseline."""
        comparison = cost_analyzer.compare_with_baseline(
            sample_cluster_config,
            BaselineHardware.A100,
            duration_hours=720
        )
        
        assert comparison["baseline_hardware"] == "a100"
        assert comparison["cost_savings"] > 0
    
    def test_optimize_cluster_cost(self, cost_analyzer, sample_cluster_config):
        """Test cost optimization suggestions."""
        optimizations = cost_analyzer.optimize_cluster_cost(sample_cluster_config)
        
        assert "current_monthly_cost" in optimizations
        assert "potential_monthly_savings" in optimizations
        assert "optimizations" in optimizations
        assert "optimization_score" in optimizations
        
        assert len(optimizations["optimizations"]) > 0
        
        # Check for specific optimization suggestions
        opt_types = [opt["optimization"] for opt in optimizations["optimizations"]]
        assert "Enable spot instances" in opt_types
        assert "Enable auto-scaling" in opt_types
    
    def test_optimize_cluster_cost_already_optimized(self, cost_analyzer, sample_cluster_config):
        """Test cost optimization for already optimized cluster."""
        # Enable optimizations
        sample_cluster_config.enable_spot_instances = True
        sample_cluster_config.auto_scaling_enabled = True
        sample_cluster_config.storage.data_volume_size_gb = 500  # Smaller storage
        
        optimizations = cost_analyzer.optimize_cluster_cost(sample_cluster_config)
        
        # Should have fewer optimization suggestions
        assert len(optimizations["optimizations"]) < 3
    
    def test_forecast_cost_trends(self, cost_analyzer, sample_cluster_config):
        """Test cost trend forecasting."""
        forecast = cost_analyzer.forecast_cost_trends(sample_cluster_config, months=12)
        
        assert "forecast_months" in forecast
        assert "monthly_costs" in forecast
        assert "total_forecast_cost" in forecast
        assert "average_monthly_cost" in forecast
        assert "cost_trend" in forecast
        assert "projected_annual_savings" in forecast
        
        assert len(forecast["monthly_costs"]) == 12
        assert forecast["cost_trend"] == "decreasing"
        
        # Check that costs decrease over time due to optimizations
        first_month_cost = forecast["monthly_costs"][0]["cost"]
        last_month_cost = forecast["monthly_costs"][-1]["cost"]
        assert last_month_cost < first_month_cost
    
    def test_cost_calculation_methods(self, cost_analyzer, sample_cluster_config):
        """Test individual cost calculation methods."""
        # Test compute cost calculation
        compute_cost = cost_analyzer._calculate_compute_cost(sample_cluster_config, 720)
        assert compute_cost > 0
        
        # Test storage cost calculation
        storage_cost = cost_analyzer._calculate_storage_cost(sample_cluster_config, 720)
        assert storage_cost >= 0
        
        # Test network cost calculation
        network_cost = cost_analyzer._calculate_network_cost(sample_cluster_config, 720)
        assert network_cost >= 0
        
        # Test monitoring cost calculation
        monitoring_cost = cost_analyzer._calculate_monitoring_cost(sample_cluster_config, 720)
        assert monitoring_cost >= 0
        
        # Test support cost calculation
        support_cost = cost_analyzer._calculate_support_cost(sample_cluster_config, 720)
        assert support_cost >= 0
    
    def test_cost_recommendation_generation(self, cost_analyzer):
        """Test cost recommendation generation."""
        # Test excellent savings
        recommendation = cost_analyzer._generate_cost_recommendation(65.0, 2.8)
        assert "Excellent cost savings" in recommendation
        
        # Test good savings
        recommendation = cost_analyzer._generate_cost_recommendation(45.0, 1.8)
        assert "Good cost savings" in recommendation
        
        # Test moderate savings
        recommendation = cost_analyzer._generate_cost_recommendation(25.0, 1.3)
        assert "Moderate savings" in recommendation
        
        # Test minimal savings
        recommendation = cost_analyzer._generate_cost_recommendation(10.0, 1.1)
        assert "Minimal savings" in recommendation
        
        # Test higher cost
        recommendation = cost_analyzer._generate_cost_recommendation(-10.0, 0.9)
        assert "Higher cost than baseline" in recommendation