"""Cluster management service for Gaudi 3 infrastructure."""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from ..models.cluster import ClusterConfig, NodeConfig, CloudProvider
from ..models.monitoring import HealthStatus, HealthCheck

logger = logging.getLogger(__name__)


class ClusterService:
    """Service for managing Gaudi 3 clusters."""
    
    def __init__(self, config: ClusterConfig):
        """Initialize cluster service with configuration."""
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self._health_checks: Dict[str, HealthCheck] = {}
        
    def validate_cluster_config(self) -> Tuple[bool, List[str]]:
        """Validate cluster configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check node count limits
        if len(self.config.nodes) == 0:
            errors.append("At least one node must be configured")
        elif len(self.config.nodes) > 64:
            errors.append("Maximum 64 nodes supported")
            
        # Check HPU limits
        total_hpus = self.config.total_hpus
        if total_hpus > 512:
            errors.append(f"Total HPUs ({total_hpus}) exceeds maximum of 512")
            
        # Check provider-specific requirements
        if self.config.provider == CloudProvider.AWS:
            if not self.config.region.startswith(('us-', 'eu-', 'ap-')):
                errors.append("Invalid AWS region format")
                
        elif self.config.provider == CloudProvider.AZURE:
            if not any(rg in self.config.region for rg in ['east', 'west', 'north', 'south']):
                errors.append("Invalid Azure region format")
                
        # Check network configuration
        if self.config.network.enable_efa and self.config.provider != CloudProvider.AWS:
            errors.append("EFA is only supported on AWS")
            
        # Check storage configuration
        if self.config.storage.data_volume_size_gb < 100:
            errors.append("Data volume must be at least 100GB")
            
        return len(errors) == 0, errors
    
    def estimate_deployment_time(self) -> timedelta:
        """Estimate cluster deployment time.
        
        Returns:
            Estimated deployment duration
        """
        base_time = timedelta(minutes=10)  # Base infrastructure setup
        
        # Add time per node
        node_time = timedelta(minutes=3) * len(self.config.nodes)
        
        # Add time for monitoring setup
        if self.config.enable_monitoring:
            node_time += timedelta(minutes=5)
            
        # Add time for spot instances (additional validation)
        if self.config.enable_spot_instances:
            node_time += timedelta(minutes=2)
            
        return base_time + node_time
    
    def get_resource_requirements(self) -> Dict[str, any]:
        """Calculate total resource requirements.
        
        Returns:
            Dictionary with resource totals
        """
        total_memory = sum(node.memory_gb for node in self.config.nodes)
        total_storage = (
            sum(node.storage_gb for node in self.config.nodes) + 
            self.config.storage.data_volume_size_gb * len(self.config.nodes)
        )
        total_network_bandwidth = sum(
            node.network_bandwidth_gbps for node in self.config.nodes
        )
        
        return {
            "total_nodes": len(self.config.nodes),
            "total_hpus": self.config.total_hpus,
            "total_memory_gb": total_memory,
            "total_storage_gb": total_storage,
            "total_network_bandwidth_gbps": total_network_bandwidth,
            "estimated_cost_per_hour": self.config.estimated_cost_per_hour,
            "estimated_cost_per_month": self.config.estimated_cost_per_hour * 24 * 30
        }
    
    def generate_terraform_config(self) -> Dict[str, any]:
        """Generate Terraform configuration for the cluster.
        
        Returns:
            Terraform configuration dictionary
        """
        base_config = self.config.to_terraform_vars()
        
        # Add provider-specific configurations
        if self.config.provider == CloudProvider.AWS:
            base_config.update({
                "availability_zones": self._get_aws_azs(),
                "enable_efa": self.config.network.enable_efa,
                "instance_storage_optimized": True
            })
        elif self.config.provider == CloudProvider.AZURE:
            base_config.update({
                "resource_group_name": f"{self.config.cluster_name}-rg",
                "enable_accelerated_networking": True
            })
        elif self.config.provider == CloudProvider.GCP:
            base_config.update({
                "project_id": "gaudi3-scale-project",
                "enable_gpu_sharing": False
            })
            
        return base_config
    
    def _get_aws_azs(self) -> List[str]:
        """Get AWS availability zones for the region."""
        az_map = {
            "us-west-2": ["us-west-2a", "us-west-2b", "us-west-2c"],
            "us-east-1": ["us-east-1a", "us-east-1b", "us-east-1c"],
            "eu-west-1": ["eu-west-1a", "eu-west-1b", "eu-west-1c"]
        }
        return az_map.get(self.config.region, [f"{self.config.region}a"])
    
    def check_cluster_health(self) -> Dict[str, HealthStatus]:
        """Check overall cluster health.
        
        Returns:
            Dictionary mapping service names to health status
        """
        health_status = {}
        
        # Check node health
        for i, node in enumerate(self.config.nodes):
            node_name = f"node-{i+1}"
            health_status[node_name] = self._check_node_health(node)
            
        # Check cluster services
        health_status["networking"] = self._check_network_health()
        health_status["storage"] = self._check_storage_health()
        
        if self.config.enable_monitoring:
            health_status["monitoring"] = self._check_monitoring_health()
            
        return health_status
    
    def _check_node_health(self, node: NodeConfig) -> HealthStatus:
        """Check individual node health."""
        # Simulate node health check
        # In real implementation, this would check:
        # - HPU availability and status
        # - System resources
        # - Network connectivity
        # - Storage accessibility
        return HealthStatus.HEALTHY
    
    def _check_network_health(self) -> HealthStatus:
        """Check cluster network health."""
        # Simulate network health check
        # In real implementation, this would check:
        # - Inter-node connectivity
        # - Bandwidth availability
        # - Network latency
        return HealthStatus.HEALTHY
    
    def _check_storage_health(self) -> HealthStatus:
        """Check cluster storage health."""
        # Simulate storage health check
        # In real implementation, this would check:
        # - Volume availability
        # - Disk space
        # - I/O performance
        return HealthStatus.HEALTHY
    
    def _check_monitoring_health(self) -> HealthStatus:
        """Check monitoring stack health."""
        # Simulate monitoring health check
        # In real implementation, this would check:
        # - Prometheus status
        # - Grafana status  
        # - Alert manager status
        return HealthStatus.HEALTHY
    
    def scale_cluster(self, target_nodes: int) -> Dict[str, any]:
        """Scale cluster to target number of nodes.
        
        Args:
            target_nodes: Target number of nodes
            
        Returns:
            Scaling operation details
        """
        current_nodes = len(self.config.nodes)
        
        if target_nodes == current_nodes:
            return {"status": "no_change", "message": "Cluster already at target size"}
            
        if target_nodes > self.config.max_nodes:
            return {
                "status": "error", 
                "message": f"Target exceeds maximum nodes ({self.config.max_nodes})"
            }
            
        if target_nodes < self.config.min_nodes:
            return {
                "status": "error",
                "message": f"Target below minimum nodes ({self.config.min_nodes})"
            }
        
        scaling_direction = "up" if target_nodes > current_nodes else "down"
        nodes_to_change = abs(target_nodes - current_nodes)
        
        estimated_time = timedelta(minutes=3 * nodes_to_change)
        
        return {
            "status": "initiated",
            "scaling_direction": scaling_direction,
            "nodes_to_change": nodes_to_change,
            "current_nodes": current_nodes,
            "target_nodes": target_nodes,
            "estimated_time": estimated_time,
            "estimated_cost_change": self._calculate_cost_change(nodes_to_change, scaling_direction)
        }
    
    def _calculate_cost_change(self, node_count: int, direction: str) -> float:
        """Calculate cost change for scaling operation."""
        if not self.config.nodes:
            return 0.0
            
        cost_per_node = self.config.estimated_cost_per_hour / len(self.config.nodes)
        change = cost_per_node * node_count
        
        return change if direction == "up" else -change
    
    def get_cluster_metrics(self) -> Dict[str, any]:
        """Get current cluster metrics summary.
        
        Returns:
            Dictionary with cluster metrics
        """
        return {
            "cluster_name": self.config.cluster_name,
            "provider": self.config.provider.value,
            "region": self.config.region,
            "total_nodes": len(self.config.nodes),
            "total_hpus": self.config.total_hpus,
            "cluster_utilization": 0.85,  # Simulated
            "average_hpu_utilization": 0.87,  # Simulated
            "total_memory_usage_gb": sum(node.memory_gb for node in self.config.nodes) * 0.75,
            "network_throughput_gbps": 180.5,  # Simulated
            "uptime_hours": 72.5,  # Simulated
            "cost_to_date": self.config.estimated_cost_per_hour * 72.5,
            "estimated_monthly_cost": self.config.estimated_cost_per_hour * 24 * 30
        }