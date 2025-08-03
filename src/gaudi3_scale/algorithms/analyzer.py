"""Performance and cost analysis algorithms."""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """Model size categories."""
    SMALL = "small"     # < 1B parameters
    MEDIUM = "medium"   # 1B - 10B parameters
    LARGE = "large"     # 10B - 100B parameters
    XLARGE = "xlarge"   # > 100B parameters


@dataclass
class PerformanceMetrics:
    """Performance analysis metrics."""
    throughput: float  # tokens/samples per second
    latency: float     # seconds per batch
    hpu_utilization: float  # percentage
    memory_efficiency: float  # percentage
    power_efficiency: float  # tokens per watt
    cost_per_token: float   # USD per million tokens
    scaling_efficiency: float  # efficiency when scaling across nodes


@dataclass
class CostBreakdown:
    """Cost analysis breakdown."""
    compute_cost: float     # USD for compute resources
    storage_cost: float     # USD for storage
    network_cost: float     # USD for data transfer
    overhead_cost: float    # USD for management overhead
    total_cost: float       # Total USD cost
    cost_per_hour: float    # USD per hour
    cost_per_token: float   # USD per million tokens


class PerformanceAnalyzer:
    """Analyzes training performance and identifies optimization opportunities.
    
    Provides comprehensive analysis of training performance metrics,
    identifies bottlenecks, and suggests optimizations.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        # Performance baselines for different model sizes on Gaudi 3
        self.baselines = {
            ModelSize.SMALL: {
                "throughput": 15000,      # tokens/sec
                "hpu_utilization": 85,    # %
                "memory_efficiency": 80,  # %
                "power_per_token": 0.025  # watts per token/sec
            },
            ModelSize.MEDIUM: {
                "throughput": 8000,
                "hpu_utilization": 88,
                "memory_efficiency": 85,
                "power_per_token": 0.035
            },
            ModelSize.LARGE: {
                "throughput": 2000,
                "hpu_utilization": 92,
                "memory_efficiency": 90,
                "power_per_token": 0.045
            },
            ModelSize.XLARGE: {
                "throughput": 500,
                "hpu_utilization": 95,
                "memory_efficiency": 95,
                "power_per_token": 0.055
            }
        }
    
    def analyze_performance(self, training_data: Dict[str, Any]) -> PerformanceMetrics:
        """Analyze training performance.
        
        Args:
            training_data: Training metrics and configuration data
            
        Returns:
            Performance analysis metrics
        """
        model_size = self._classify_model_size(training_data.get("parameters", 0))
        baseline = self.baselines[model_size]
        
        # Extract metrics
        throughput = training_data.get("throughput", 0)
        latency = training_data.get("step_time", 0)
        hpu_util = training_data.get("hpu_utilization", 0)
        memory_used = training_data.get("memory_usage", 0)
        memory_total = training_data.get("memory_total", 32)  # GB
        power_consumption = training_data.get("power_consumption", 350)
        
        # Calculate efficiency metrics
        memory_efficiency = (memory_used / memory_total * 100) if memory_total > 0 else 0
        power_efficiency = throughput / power_consumption if power_consumption > 0 else 0
        
        # Calculate relative performance vs baseline
        throughput_ratio = throughput / baseline["throughput"] if baseline["throughput"] > 0 else 0
        
        # Estimate cost per token (simplified)
        cost_per_hour = self._calculate_hourly_cost(training_data)
        cost_per_token = (cost_per_hour / (throughput * 3600)) * 1_000_000 if throughput > 0 else 0
        
        # Calculate scaling efficiency (if multi-node)
        scaling_efficiency = self._calculate_scaling_efficiency(training_data)
        
        return PerformanceMetrics(
            throughput=throughput,
            latency=latency,
            hpu_utilization=hpu_util,
            memory_efficiency=memory_efficiency,
            power_efficiency=power_efficiency,
            cost_per_token=cost_per_token,
            scaling_efficiency=scaling_efficiency
        )
    
    def identify_bottlenecks(self, performance: PerformanceMetrics, 
                           training_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks.
        
        Args:
            performance: Performance metrics
            training_data: Training configuration and metrics
            
        Returns:
            List of identified bottlenecks with recommendations
        """
        bottlenecks = []
        model_size = self._classify_model_size(training_data.get("parameters", 0))
        baseline = self.baselines[model_size]
        
        # Check HPU utilization
        if performance.hpu_utilization < baseline["hpu_utilization"] * 0.9:
            bottlenecks.append({
                "type": "hpu_underutilization",
                "severity": "high" if performance.hpu_utilization < 70 else "medium",
                "current": performance.hpu_utilization,
                "target": baseline["hpu_utilization"],
                "description": f"HPU utilization is {performance.hpu_utilization:.1f}%, below target of {baseline['hpu_utilization']}%",
                "recommendations": [
                    "Increase batch size to improve HPU utilization",
                    "Enable mixed precision training (BF16)",
                    "Optimize data loading pipeline",
                    "Check for CPU bottlenecks in preprocessing"
                ]
            })
        
        # Check memory efficiency
        if performance.memory_efficiency < baseline["memory_efficiency"] * 0.9:
            bottlenecks.append({
                "type": "memory_inefficiency",
                "severity": "medium",
                "current": performance.memory_efficiency,
                "target": baseline["memory_efficiency"],
                "description": f"Memory efficiency is {performance.memory_efficiency:.1f}%, below target of {baseline['memory_efficiency']}%",
                "recommendations": [
                    "Enable gradient accumulation to use more memory",
                    "Increase sequence length or batch size",
                    "Use activation checkpointing if needed",
                    "Consider model parallelism for larger models"
                ]
            })
        
        # Check throughput
        throughput_target = baseline["throughput"]
        if performance.throughput < throughput_target * 0.8:
            bottlenecks.append({
                "type": "low_throughput",
                "severity": "high",
                "current": performance.throughput,
                "target": throughput_target,
                "description": f"Throughput is {performance.throughput:.0f} tokens/sec, below target of {throughput_target:.0f}",
                "recommendations": [
                    "Optimize batch size and gradient accumulation",
                    "Enable graph compilation optimizations",
                    "Check data loading bottlenecks",
                    "Verify model implementation efficiency"
                ]
            })
        
        # Check scaling efficiency for multi-node
        if training_data.get("num_nodes", 1) > 1 and performance.scaling_efficiency < 0.8:
            bottlenecks.append({
                "type": "scaling_inefficiency",
                "severity": "high",
                "current": performance.scaling_efficiency,
                "target": 0.9,
                "description": f"Multi-node scaling efficiency is {performance.scaling_efficiency:.2f}, indicating communication overhead",
                "recommendations": [
                    "Increase batch size per node",
                    "Optimize gradient synchronization",
                    "Check network bandwidth and latency",
                    "Consider gradient compression techniques"
                ]
            })
        
        # Check power efficiency
        target_power_efficiency = 1.0 / baseline["power_per_token"]
        if performance.power_efficiency < target_power_efficiency * 0.8:
            bottlenecks.append({
                "type": "power_inefficiency",
                "severity": "low",
                "current": performance.power_efficiency,
                "target": target_power_efficiency,
                "description": f"Power efficiency is {performance.power_efficiency:.3f} tokens/watt, below target",
                "recommendations": [
                    "Optimize model utilization to reduce idle time",
                    "Use dynamic frequency scaling",
                    "Check for thermal throttling",
                    "Optimize batch size for power efficiency"
                ]
            })
        
        return sorted(bottlenecks, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["severity"]], reverse=True)
    
    def suggest_optimizations(self, bottlenecks: List[Dict[str, Any]], 
                            training_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest specific optimizations based on bottlenecks.
        
        Args:
            bottlenecks: Identified bottlenecks
            training_data: Training configuration
            
        Returns:
            List of optimization suggestions
        """
        optimizations = []
        current_batch_size = training_data.get("batch_size", 32)
        current_grad_accum = training_data.get("gradient_accumulation_steps", 1)
        
        # Batch size optimization
        if any(b["type"] in ["hpu_underutilization", "low_throughput"] for b in bottlenecks):
            suggested_batch_size = self._suggest_batch_size(training_data)
            if suggested_batch_size > current_batch_size:
                optimizations.append({
                    "type": "batch_size",
                    "priority": "high",
                    "current_value": current_batch_size,
                    "suggested_value": suggested_batch_size,
                    "description": f"Increase batch size from {current_batch_size} to {suggested_batch_size}",
                    "expected_improvement": "10-25% throughput increase",
                    "implementation": f"--batch-size {suggested_batch_size}"
                })
        
        # Gradient accumulation optimization
        if any(b["type"] == "memory_inefficiency" for b in bottlenecks):
            suggested_grad_accum = min(current_grad_accum * 2, 8)
            optimizations.append({
                "type": "gradient_accumulation",
                "priority": "medium",
                "current_value": current_grad_accum,
                "suggested_value": suggested_grad_accum,
                "description": f"Increase gradient accumulation from {current_grad_accum} to {suggested_grad_accum}",
                "expected_improvement": "Better memory utilization",
                "implementation": f"--gradient-accumulation-steps {suggested_grad_accum}"
            })
        
        # Mixed precision optimization
        if not training_data.get("use_mixed_precision", False):
            optimizations.append({
                "type": "mixed_precision",
                "priority": "high",
                "current_value": False,
                "suggested_value": True,
                "description": "Enable BF16 mixed precision training",
                "expected_improvement": "30-50% speedup with minimal accuracy loss",
                "implementation": "--precision bf16-mixed"
            })
        
        # Graph compilation optimization
        if not training_data.get("use_graph_compilation", False):
            optimizations.append({
                "type": "graph_compilation",
                "priority": "medium",
                "current_value": False,
                "suggested_value": True,
                "description": "Enable Habana graph compiler optimizations",
                "expected_improvement": "10-20% speedup after warmup",
                "implementation": "Set PT_HPU_LAZY_MODE=1 and PT_HPU_ENABLE_LAZY_COMPILATION=1"
            })
        
        return sorted(optimizations, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True)
    
    def _classify_model_size(self, parameters: int) -> ModelSize:
        """Classify model size based on parameter count.
        
        Args:
            parameters: Number of model parameters
            
        Returns:
            Model size category
        """
        if parameters < 1_000_000_000:  # < 1B
            return ModelSize.SMALL
        elif parameters < 10_000_000_000:  # 1B - 10B
            return ModelSize.MEDIUM
        elif parameters < 100_000_000_000:  # 10B - 100B
            return ModelSize.LARGE
        else:  # > 100B
            return ModelSize.XLARGE
    
    def _calculate_hourly_cost(self, training_data: Dict[str, Any]) -> float:
        """Calculate hourly training cost.
        
        Args:
            training_data: Training configuration
            
        Returns:
            Cost per hour in USD
        """
        # AWS dl2q.24xlarge (8 Gaudi 3 HPUs) approximate cost
        base_cost_per_hour = 32.77
        
        num_nodes = training_data.get("num_nodes", 1)
        return base_cost_per_hour * num_nodes
    
    def _calculate_scaling_efficiency(self, training_data: Dict[str, Any]) -> float:
        """Calculate multi-node scaling efficiency.
        
        Args:
            training_data: Training configuration
            
        Returns:
            Scaling efficiency (0-1)
        """
        num_nodes = training_data.get("num_nodes", 1)
        
        if num_nodes == 1:
            return 1.0
        
        # Simplified scaling efficiency model
        # Real implementation would use actual measurements
        theoretical_speedup = num_nodes
        communication_overhead = 0.05 * (num_nodes - 1)  # 5% overhead per additional node
        actual_speedup = theoretical_speedup * (1 - communication_overhead)
        
        return actual_speedup / theoretical_speedup
    
    def _suggest_batch_size(self, training_data: Dict[str, Any]) -> int:
        """Suggest optimal batch size.
        
        Args:
            training_data: Training configuration
            
        Returns:
            Suggested batch size
        """
        current_batch_size = training_data.get("batch_size", 32)
        model_size = self._classify_model_size(training_data.get("parameters", 0))
        sequence_length = training_data.get("sequence_length", 512)
        
        # Batch size suggestions based on model size and memory
        if model_size == ModelSize.SMALL:
            target_batch_size = max(current_batch_size, 128)
        elif model_size == ModelSize.MEDIUM:
            target_batch_size = max(current_batch_size, 64)
        elif model_size == ModelSize.LARGE:
            target_batch_size = max(current_batch_size, 32)
        else:  # XLARGE
            target_batch_size = max(current_batch_size, 16)
        
        # Adjust for sequence length
        if sequence_length > 1024:
            target_batch_size = target_batch_size // 2
        elif sequence_length > 2048:
            target_batch_size = target_batch_size // 4
        
        return min(target_batch_size, current_batch_size * 4)  # Don't increase too aggressively


class CostAnalyzer:
    """Analyzes training costs and provides cost optimization recommendations."""
    
    def __init__(self):
        """Initialize cost analyzer."""
        # Cost rates for different cloud providers and instance types
        self.instance_costs = {
            "aws": {
                "dl2q.24xlarge": 32.77,  # 8 Gaudi 3 HPUs
                "p4d.24xlarge": 98.32,   # 8 A100 GPUs (comparison)
                "p3.16xlarge": 52.88     # 8 V100 GPUs (comparison)
            },
            "azure": {
                "ND96isr_H100_v5": 98.50,  # 8 H100 GPUs
                "ND96amsr_A100_v4": 76.20  # 8 A100 GPUs
            }
        }
        
        # Storage costs (per GB per month)
        self.storage_costs = {
            "aws_s3": 0.023,
            "azure_blob": 0.021,
            "gcp_storage": 0.020
        }
        
        # Network costs (per GB)
        self.network_costs = {
            "aws_transfer": 0.09,
            "azure_transfer": 0.087,
            "gcp_transfer": 0.12
        }
    
    def analyze_training_cost(self, training_config: Dict[str, Any], 
                            training_metrics: Dict[str, Any]) -> CostBreakdown:
        """Analyze total training cost.
        
        Args:
            training_config: Training configuration
            training_metrics: Training performance metrics
            
        Returns:
            Detailed cost breakdown
        """
        # Extract configuration
        provider = training_config.get("cloud_provider", "aws")
        instance_type = training_config.get("instance_type", "dl2q.24xlarge")
        num_nodes = training_config.get("num_nodes", 1)
        training_hours = training_metrics.get("training_hours", 0)
        
        # Calculate compute cost
        hourly_rate = self.instance_costs.get(provider, {}).get(instance_type, 32.77)
        compute_cost = hourly_rate * num_nodes * training_hours
        
        # Calculate storage cost
        dataset_size_gb = training_config.get("dataset_size_gb", 100)
        model_size_gb = training_config.get("model_size_gb", 50)
        storage_key = f"{provider}_s3" if provider == "aws" else f"{provider}_blob"
        monthly_storage_rate = self.storage_costs.get(storage_key, 0.023)
        storage_cost = (dataset_size_gb + model_size_gb) * monthly_storage_rate * (training_hours / 720)  # 720 hours per month
        
        # Calculate network cost
        data_transfer_gb = training_config.get("data_transfer_gb", 10)
        network_key = f"{provider}_transfer"
        transfer_rate = self.network_costs.get(network_key, 0.09)
        network_cost = data_transfer_gb * transfer_rate
        
        # Calculate overhead (management, monitoring, etc.)
        overhead_cost = compute_cost * 0.1  # 10% overhead
        
        total_cost = compute_cost + storage_cost + network_cost + overhead_cost
        cost_per_hour = total_cost / max(training_hours, 1)
        
        # Calculate cost per token
        total_tokens = training_metrics.get("total_tokens_processed", 0)
        cost_per_token = (total_cost / total_tokens) * 1_000_000 if total_tokens > 0 else 0
        
        return CostBreakdown(
            compute_cost=compute_cost,
            storage_cost=storage_cost,
            network_cost=network_cost,
            overhead_cost=overhead_cost,
            total_cost=total_cost,
            cost_per_hour=cost_per_hour,
            cost_per_token=cost_per_token
        )
    
    def compare_providers(self, training_config: Dict[str, Any], 
                         training_metrics: Dict[str, Any]) -> Dict[str, CostBreakdown]:
        """Compare costs across different cloud providers.
        
        Args:
            training_config: Training configuration
            training_metrics: Training performance metrics
            
        Returns:
            Cost breakdown for each provider
        """
        comparisons = {}
        original_provider = training_config.get("cloud_provider", "aws")
        
        for provider in ["aws", "azure"]:
            config_copy = training_config.copy()
            config_copy["cloud_provider"] = provider
            
            # Map to equivalent instance types
            if provider == "aws":
                config_copy["instance_type"] = "dl2q.24xlarge"
            elif provider == "azure":
                config_copy["instance_type"] = "ND96amsr_A100_v4"  # Closest equivalent
            
            cost_breakdown = self.analyze_training_cost(config_copy, training_metrics)
            comparisons[provider] = cost_breakdown
        
        return comparisons
    
    def calculate_tco(self, training_config: Dict[str, Any], 
                     usage_pattern: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Total Cost of Ownership (TCO) for different scenarios.
        
        Args:
            training_config: Training configuration
            usage_pattern: Usage pattern over time
            
        Returns:
            TCO analysis for different time periods
        """
        monthly_training_hours = usage_pattern.get("monthly_training_hours", 200)
        months = usage_pattern.get("months", 12)
        
        # Base costs
        provider = training_config.get("cloud_provider", "aws")
        instance_type = training_config.get("instance_type", "dl2q.24xlarge")
        num_nodes = training_config.get("num_nodes", 1)
        
        hourly_rate = self.instance_costs.get(provider, {}).get(instance_type, 32.77)
        monthly_compute_cost = hourly_rate * num_nodes * monthly_training_hours
        
        # Storage costs (persistent)
        dataset_size_gb = training_config.get("dataset_size_gb", 100)
        storage_key = f"{provider}_s3" if provider == "aws" else f"{provider}_blob"
        monthly_storage_rate = self.storage_costs.get(storage_key, 0.023)
        monthly_storage_cost = dataset_size_gb * monthly_storage_rate
        
        # Management overhead
        monthly_overhead = monthly_compute_cost * 0.15  # 15% for long-term overhead
        
        monthly_total = monthly_compute_cost + monthly_storage_cost + monthly_overhead
        
        return {
            "monthly_cost": monthly_total,
            "annual_cost": monthly_total * 12,
            "custom_period_cost": monthly_total * months,
            "cost_breakdown": {
                "compute": monthly_compute_cost,
                "storage": monthly_storage_cost,
                "overhead": monthly_overhead
            }
        }
    
    def suggest_cost_optimizations(self, cost_breakdown: CostBreakdown, 
                                 training_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest cost optimization strategies.
        
        Args:
            cost_breakdown: Current cost breakdown
            training_config: Training configuration
            
        Returns:
            List of cost optimization suggestions
        """
        optimizations = []
        
        # Spot instances
        if training_config.get("use_spot_instances", False) is False:
            potential_savings = cost_breakdown.compute_cost * 0.6  # 60% savings typical
            optimizations.append({
                "type": "spot_instances",
                "priority": "high",
                "description": "Use spot instances for fault-tolerant training",
                "potential_savings_usd": potential_savings,
                "potential_savings_percent": 60,
                "implementation": "Enable spot instance usage with checkpointing",
                "trade_offs": "May require restart handling and longer training times"
            })
        
        # Reserved instances for long-term usage
        training_hours_per_month = training_config.get("monthly_training_hours", 200)
        if training_hours_per_month > 100:
            potential_savings = cost_breakdown.compute_cost * 0.3  # 30% savings
            optimizations.append({
                "type": "reserved_instances",
                "priority": "medium",
                "description": "Use reserved instances for predictable workloads",
                "potential_savings_usd": potential_savings,
                "potential_savings_percent": 30,
                "implementation": "Purchase 1-year reserved instances",
                "trade_offs": "Requires upfront commitment"
            })
        
        # Multi-region optimization
        if cost_breakdown.network_cost > cost_breakdown.total_cost * 0.1:
            optimizations.append({
                "type": "data_locality",
                "priority": "medium",
                "description": "Move data closer to compute resources",
                "potential_savings_usd": cost_breakdown.network_cost * 0.7,
                "potential_savings_percent": 70,
                "implementation": "Replicate data to training region",
                "trade_offs": "Additional storage costs"
            })
        
        # Storage optimization
        if cost_breakdown.storage_cost > cost_breakdown.total_cost * 0.05:
            optimizations.append({
                "type": "storage_optimization",
                "priority": "low",
                "description": "Use cheaper storage tiers for archival data",
                "potential_savings_usd": cost_breakdown.storage_cost * 0.4,
                "potential_savings_percent": 40,
                "implementation": "Move old datasets to cold storage",
                "trade_offs": "Slower access for archived data"
            })
        
        # Batch size optimization for cost efficiency
        current_batch_size = training_config.get("batch_size", 32)
        if current_batch_size < 64:
            optimizations.append({
                "type": "batch_size_efficiency",
                "priority": "high",
                "description": "Increase batch size for better cost efficiency",
                "potential_savings_usd": cost_breakdown.total_cost * 0.15,
                "potential_savings_percent": 15,
                "implementation": f"Increase batch size from {current_batch_size} to {current_batch_size * 2}",
                "trade_offs": "May require more memory or gradient accumulation"
            })
        
        return sorted(optimizations, key=lambda x: x["potential_savings_usd"], reverse=True)