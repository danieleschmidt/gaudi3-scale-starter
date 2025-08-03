"""Cost analysis and optimization service for Gaudi 3 infrastructure."""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from ..models.cluster import ClusterConfig, CloudProvider

logger = logging.getLogger(__name__)


class CostCategory(str, Enum):
    """Cost categories for analysis."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    MONITORING = "monitoring"
    SUPPORT = "support"


class BaselineHardware(str, Enum):
    """Baseline hardware for cost comparison."""
    H100 = "h100"
    A100 = "a100"
    V100 = "v100"
    TPU_V4 = "tpu_v4"


class CostAnalyzer:
    """Service for analyzing and optimizing Gaudi 3 infrastructure costs."""
    
    def __init__(self):
        """Initialize cost analyzer."""
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Cost data per hour (USD)
        self.compute_costs = {
            CloudProvider.AWS: {
                "dl2q.24xlarge": 32.77,  # 8 Gaudi 3 HPUs
                "dl2q.48xlarge": 65.54,  # 16 Gaudi 3 HPUs
                "spot_discount": 0.3     # 70% discount
            },
            CloudProvider.AZURE: {
                "Standard_HX176rs": 45.50,  # 8 Gaudi 3 HPUs
                "Standard_HX352rs": 91.00,  # 16 Gaudi 3 HPUs
                "spot_discount": 0.2        # 80% discount
            },
            CloudProvider.GCP: {
                "a2-ultragpu-8g": 38.20,   # Estimated
                "spot_discount": 0.4       # 60% discount
            }
        }
        
        # Baseline hardware costs (per hour)
        self.baseline_costs = {
            BaselineHardware.H100: 98.32,     # 8x H100 80GB
            BaselineHardware.A100: 52.88,     # 8x A100 80GB
            BaselineHardware.V100: 24.48,     # 8x V100 32GB
            BaselineHardware.TPU_V4: 84.00    # TPU v4 Pod
        }
        
        # Storage costs per GB per month
        self.storage_costs = {
            CloudProvider.AWS: {
                "gp3": 0.08,
                "io2": 0.125,
                "s3_standard": 0.023
            },
            CloudProvider.AZURE: {
                "premium_ssd": 0.12,
                "standard_ssd": 0.06,
                "blob_hot": 0.0184
            },
            CloudProvider.GCP: {
                "ssd": 0.10,
                "standard": 0.04,
                "cloud_storage": 0.020
            }
        }
        
        # Network costs per GB
        self.network_costs = {
            CloudProvider.AWS: 0.09,    # Data transfer out
            CloudProvider.AZURE: 0.08,  # Bandwidth
            CloudProvider.GCP: 0.12     # Egress
        }
    
    def analyze_cluster_cost(self, config: ClusterConfig, 
                           duration_hours: float = 720) -> Dict[str, any]:
        """Analyze total cost for a cluster configuration.
        
        Args:
            config: Cluster configuration
            duration_hours: Analysis duration in hours (default: 1 month)
            
        Returns:
            Detailed cost breakdown
        """
        cost_breakdown = {
            CostCategory.COMPUTE: self._calculate_compute_cost(config, duration_hours),
            CostCategory.STORAGE: self._calculate_storage_cost(config, duration_hours),
            CostCategory.NETWORK: self._calculate_network_cost(config, duration_hours),
            CostCategory.MONITORING: self._calculate_monitoring_cost(config, duration_hours),
            CostCategory.SUPPORT: self._calculate_support_cost(config, duration_hours)
        }
        
        total_cost = sum(cost_breakdown.values())
        
        return {
            "total_cost": total_cost,
            "cost_per_hour": total_cost / duration_hours,
            "cost_breakdown": cost_breakdown,
            "cost_per_hpu": total_cost / max(config.total_hpus, 1),
            "duration_hours": duration_hours,
            "currency": "USD"
        }
    
    def _calculate_compute_cost(self, config: ClusterConfig, hours: float) -> float:
        """Calculate compute costs."""
        if config.provider == CloudProvider.ONPREM:
            return 0.0  # On-premises has no hourly compute cost
            
        if not config.nodes:
            return 0.0
            
        instance_type = config.nodes[0].instance_type.value
        provider_costs = self.compute_costs.get(config.provider, {})
        base_cost_per_hour = provider_costs.get(instance_type, 30.0)
        
        # Apply spot discount if enabled
        if config.enable_spot_instances:
            spot_discount = provider_costs.get("spot_discount", 0.3)
            base_cost_per_hour *= spot_discount
            
        total_compute_cost = base_cost_per_hour * len(config.nodes) * hours
        
        return total_compute_cost
    
    def _calculate_storage_cost(self, config: ClusterConfig, hours: float) -> float:
        """Calculate storage costs."""
        if config.provider == CloudProvider.ONPREM:
            return 0.0
            
        provider_storage = self.storage_costs.get(config.provider, {})
        volume_cost_per_gb_month = provider_storage.get("gp3", 0.08)
        
        # Calculate total storage per node
        storage_per_node = (
            config.storage.root_volume_size_gb + 
            config.storage.data_volume_size_gb
        )
        
        total_storage_gb = storage_per_node * len(config.nodes)
        
        # Convert hourly to monthly cost
        monthly_factor = hours / 720  # 720 hours in a month
        total_storage_cost = total_storage_gb * volume_cost_per_gb_month * monthly_factor
        
        return total_storage_cost
    
    def _calculate_network_cost(self, config: ClusterConfig, hours: float) -> float:
        """Calculate network costs."""
        if config.provider == CloudProvider.ONPREM:
            return 0.0
            
        # Estimate data transfer (GB) based on training workload
        estimated_gb_per_hour = config.total_hpus * 0.5  # Conservative estimate
        total_data_transfer_gb = estimated_gb_per_hour * hours
        
        cost_per_gb = self.network_costs.get(config.provider, 0.09)
        
        return total_data_transfer_gb * cost_per_gb
    
    def _calculate_monitoring_cost(self, config: ClusterConfig, hours: float) -> float:
        """Calculate monitoring stack costs."""
        if not config.enable_monitoring:
            return 0.0
            
        # Monitoring typically adds 5-10% to compute costs
        compute_cost = self._calculate_compute_cost(config, hours)
        monitoring_overhead = 0.07  # 7%
        
        return compute_cost * monitoring_overhead
    
    def _calculate_support_cost(self, config: ClusterConfig, hours: float) -> float:
        """Calculate support and maintenance costs."""
        # Support typically 10-15% for enterprise deployments
        if config.provider == CloudProvider.ONPREM:
            # On-premises support is higher
            support_rate = 0.15
        else:
            support_rate = 0.10
            
        compute_cost = self._calculate_compute_cost(config, hours)
        return compute_cost * support_rate
    
    def compare_with_baseline(self, config: ClusterConfig, 
                            baseline: BaselineHardware,
                            duration_hours: float = 720) -> Dict[str, any]:
        """Compare Gaudi 3 costs with baseline hardware.
        
        Args:
            config: Gaudi 3 cluster configuration
            baseline: Baseline hardware for comparison
            duration_hours: Analysis duration
            
        Returns:
            Cost comparison analysis
        """
        gaudi_analysis = self.analyze_cluster_cost(config, duration_hours)
        gaudi_total = gaudi_analysis["total_cost"]
        
        # Calculate equivalent baseline cost
        baseline_cost_per_hour = self.baseline_costs[baseline]
        baseline_nodes_needed = len(config.nodes)  # Assume 1:1 node mapping
        baseline_total = baseline_cost_per_hour * baseline_nodes_needed * duration_hours
        
        # Calculate savings
        cost_savings = baseline_total - gaudi_total
        savings_percentage = (cost_savings / baseline_total) * 100 if baseline_total > 0 else 0
        cost_ratio = baseline_total / gaudi_total if gaudi_total > 0 else 0
        
        return {
            "gaudi3_cost": gaudi_total,
            "baseline_cost": baseline_total,
            "cost_savings": cost_savings,
            "savings_percentage": savings_percentage,
            "cost_ratio": cost_ratio,
            "baseline_hardware": baseline.value,
            "recommendation": self._generate_cost_recommendation(
                savings_percentage, cost_ratio
            )
        }
    
    def _generate_cost_recommendation(self, savings_pct: float, 
                                    cost_ratio: float) -> str:
        """Generate cost optimization recommendation."""
        if savings_pct >= 60:
            return "Excellent cost savings! Gaudi 3 is highly cost-effective for this workload."
        elif savings_pct >= 40:
            return "Good cost savings. Gaudi 3 provides significant value."
        elif savings_pct >= 20:
            return "Moderate savings. Consider workload optimization for better ROI."
        elif savings_pct >= 0:
            return "Minimal savings. Evaluate performance vs. cost trade-offs."
        else:
            return "Higher cost than baseline. Consider alternative configurations."
    
    def optimize_cluster_cost(self, config: ClusterConfig) -> Dict[str, any]:
        """Suggest cost optimizations for cluster configuration.
        
        Args:
            config: Current cluster configuration
            
        Returns:
            Cost optimization suggestions
        """
        optimizations = []
        potential_savings = 0.0
        
        # Check spot instance opportunity
        if not config.enable_spot_instances and config.provider != CloudProvider.ONPREM:
            spot_savings = self._calculate_compute_cost(config, 720) * 0.7
            optimizations.append({
                "optimization": "Enable spot instances",
                "potential_savings": spot_savings,
                "risk": "Potential interruptions",
                "recommendation": "Use for fault-tolerant training workloads"
            })
            potential_savings += spot_savings
        
        # Check auto-scaling opportunity
        if not config.auto_scaling_enabled:
            autoscale_savings = self._calculate_compute_cost(config, 720) * 0.2
            optimizations.append({
                "optimization": "Enable auto-scaling",
                "potential_savings": autoscale_savings,
                "risk": "Scaling complexity",
                "recommendation": "Scale down during low utilization periods"
            })
            potential_savings += autoscale_savings
        
        # Check storage optimization
        if config.storage.data_volume_size_gb > 2000:
            storage_savings = self._calculate_storage_cost(config, 720) * 0.3
            optimizations.append({
                "optimization": "Optimize storage tiers",
                "potential_savings": storage_savings,
                "risk": "Reduced I/O performance",
                "recommendation": "Use cheaper storage for infrequently accessed data"
            })
            potential_savings += storage_savings
        
        # Check monitoring overhead
        if config.enable_monitoring:
            monitoring_cost = self._calculate_monitoring_cost(config, 720)
            if monitoring_cost > 1000:  # Arbitrary threshold
                monitoring_savings = monitoring_cost * 0.4
                optimizations.append({
                    "optimization": "Optimize monitoring stack",
                    "potential_savings": monitoring_savings,
                    "risk": "Reduced observability",
                    "recommendation": "Use selective metrics collection"
                })
                potential_savings += monitoring_savings
        
        return {
            "current_monthly_cost": self.analyze_cluster_cost(config, 720)["total_cost"],
            "potential_monthly_savings": potential_savings,
            "optimizations": optimizations,
            "optimization_score": min(len(optimizations) * 20, 100)  # Max 100
        }
    
    def forecast_cost_trends(self, config: ClusterConfig, 
                           months: int = 12) -> Dict[str, any]:
        """Forecast cost trends over time.
        
        Args:
            config: Cluster configuration
            months: Number of months to forecast
            
        Returns:
            Cost trend forecast
        """
        monthly_costs = []
        base_monthly_cost = self.analyze_cluster_cost(config, 720)["total_cost"]
        
        for month in range(months):
            # Apply expected cost reductions over time
            cost_reduction_factor = 1 - (month * 0.02)  # 2% reduction per month
            cost_reduction_factor = max(cost_reduction_factor, 0.8)  # Cap at 20% total reduction
            
            monthly_cost = base_monthly_cost * cost_reduction_factor
            monthly_costs.append({
                "month": month + 1,
                "cost": monthly_cost,
                "savings_vs_baseline": base_monthly_cost - monthly_cost
            })
        
        total_forecast_cost = sum(m["cost"] for m in monthly_costs)
        
        return {
            "forecast_months": months,
            "monthly_costs": monthly_costs,
            "total_forecast_cost": total_forecast_cost,
            "average_monthly_cost": total_forecast_cost / months,
            "cost_trend": "decreasing",  # Due to optimizations
            "projected_annual_savings": (base_monthly_cost - monthly_costs[-1]["cost"]) * 12
        }