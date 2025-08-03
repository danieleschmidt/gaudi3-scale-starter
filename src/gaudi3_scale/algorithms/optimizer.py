"""Training and resource optimization algorithms."""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    THROUGHPUT = "throughput"        # Maximize tokens/sec
    COST = "cost"                   # Minimize cost per token
    QUALITY = "quality"             # Maximize model quality
    BALANCED = "balanced"           # Balance all factors


@dataclass
class OptimizationResult:
    """Optimization result with recommended configuration."""
    strategy: OptimizationStrategy
    config: Dict[str, Any]
    expected_improvement: Dict[str, float]
    confidence: float
    explanation: str


class TrainingOptimizer:
    """Optimizes training configuration for performance and cost.
    
    Uses algorithms to automatically tune hyperparameters, batch sizes,
    and other training configuration parameters.
    """
    
    def __init__(self):
        """Initialize training optimizer."""
        # Memory constraints for different model sizes (in GB)
        self.memory_constraints = {
            "small": 8,      # < 1B params
            "medium": 16,    # 1B-10B params  
            "large": 24,     # 10B-100B params
            "xlarge": 32     # > 100B params
        }
        
        # Performance baselines for optimization
        self.performance_targets = {
            "throughput": {
                "small": 15000,   # tokens/sec
                "medium": 8000,
                "large": 2000,
                "xlarge": 500
            },
            "hpu_utilization": 88,  # target %
            "memory_utilization": 85  # target %
        }
    
    def optimize_batch_size(self, model_config: Dict[str, Any], 
                          hardware_config: Dict[str, Any],
                          strategy: OptimizationStrategy = OptimizationStrategy.THROUGHPUT) -> OptimizationResult:
        """Optimize batch size for given model and hardware.
        
        Args:
            model_config: Model configuration (params, seq_length, etc.)
            hardware_config: Hardware configuration (num_hpus, memory, etc.)
            strategy: Optimization strategy
            
        Returns:
            Optimization result with recommended batch size
        """
        num_params = model_config.get("parameters", 1_000_000_000)
        seq_length = model_config.get("sequence_length", 512)
        num_hpus = hardware_config.get("num_hpus", 8)
        memory_per_hpu = hardware_config.get("memory_per_hpu_gb", 32)
        
        model_size_category = self._classify_model_size(num_params)
        
        # Calculate memory requirements
        memory_per_param = 4  # bytes for FP32, adjust for mixed precision
        model_memory = (num_params * memory_per_param) / (1024**3)  # GB
        
        # Account for activations and gradients
        activation_memory_factor = seq_length / 512  # Scale with sequence length
        total_memory_per_hpu = model_memory / num_hpus * 2.5 * activation_memory_factor  # 2.5x for gradients + activations
        
        # Calculate maximum batch size based on memory
        available_memory = memory_per_hpu * 0.8  # Use 80% of available memory
        max_batch_size_memory = int((available_memory - total_memory_per_hpu) / 
                                  (seq_length * 4 * 1e-9 * 50))  # Rough activation size estimate
        
        # Strategy-specific optimization
        if strategy == OptimizationStrategy.THROUGHPUT:
            # Maximize throughput
            optimal_batch_size = self._optimize_for_throughput(
                max_batch_size_memory, model_size_category, num_hpus
            )
            expected_improvement = {"throughput": 25, "cost_per_token": -5}
            
        elif strategy == OptimizationStrategy.COST:
            # Maximize cost efficiency
            optimal_batch_size = self._optimize_for_cost(
                max_batch_size_memory, model_size_category, num_hpus
            )
            expected_improvement = {"cost_per_token": -20, "throughput": 10}
            
        else:  # BALANCED or QUALITY
            # Balance between throughput and stability
            optimal_batch_size = self._optimize_balanced(
                max_batch_size_memory, model_size_category, num_hpus
            )
            expected_improvement = {"throughput": 15, "cost_per_token": -10, "stability": 5}
        
        # Ensure batch size is within reasonable bounds
        optimal_batch_size = max(1, min(optimal_batch_size, max_batch_size_memory, 512))
        
        # Calculate gradient accumulation if needed
        target_effective_batch = self._get_target_effective_batch(model_size_category)
        grad_accumulation = max(1, target_effective_batch // optimal_batch_size)
        
        config = {
            "batch_size": optimal_batch_size,
            "gradient_accumulation_steps": grad_accumulation,
            "effective_batch_size": optimal_batch_size * grad_accumulation
        }
        
        explanation = (f"Optimized for {strategy.value}: batch_size={optimal_batch_size}, "
                      f"grad_accumulation={grad_accumulation}, effective_batch={config['effective_batch_size']}")
        
        return OptimizationResult(
            strategy=strategy,
            config=config,
            expected_improvement=expected_improvement,
            confidence=0.85,
            explanation=explanation
        )
    
    def optimize_learning_rate(self, model_config: Dict[str, Any],
                             batch_config: Dict[str, Any],
                             strategy: OptimizationStrategy = OptimizationStrategy.QUALITY) -> OptimizationResult:
        """Optimize learning rate based on model size and batch configuration.
        
        Args:
            model_config: Model configuration
            batch_config: Batch size configuration
            strategy: Optimization strategy
            
        Returns:
            Optimization result with recommended learning rate schedule
        """
        num_params = model_config.get("parameters", 1_000_000_000)
        effective_batch_size = batch_config.get("effective_batch_size", 32)
        model_type = model_config.get("model_type", "transformer")
        
        model_size_category = self._classify_model_size(num_params)
        
        # Base learning rates for different model sizes
        base_lrs = {
            "small": 3e-4,
            "medium": 1.5e-4,
            "large": 6e-5,
            "xlarge": 3e-5
        }
        
        base_lr = base_lrs[model_size_category]
        
        # Scale learning rate with effective batch size (sqrt scaling)
        batch_scale_factor = math.sqrt(effective_batch_size / 32)
        scaled_lr = base_lr * batch_scale_factor
        
        # Strategy-specific adjustments
        if strategy == OptimizationStrategy.THROUGHPUT:
            # Slightly higher LR for faster convergence
            final_lr = scaled_lr * 1.2
            schedule = "cosine_with_warmup"
            warmup_steps = 1000
            expected_improvement = {"training_speed": 15, "convergence": 10}
            
        elif strategy == OptimizationStrategy.QUALITY:
            # Conservative LR for better final quality
            final_lr = scaled_lr * 0.8
            schedule = "cosine_with_warmup"
            warmup_steps = 2000
            expected_improvement = {"final_accuracy": 5, "stability": 10}
            
        else:  # COST or BALANCED
            # Standard scaling
            final_lr = scaled_lr
            schedule = "cosine_with_warmup"
            warmup_steps = 1500
            expected_improvement = {"training_speed": 8, "final_accuracy": 3}
        
        config = {
            "learning_rate": final_lr,
            "lr_scheduler": schedule,
            "warmup_steps": warmup_steps,
            "min_lr": final_lr * 0.01
        }
        
        explanation = (f"Optimized LR for {model_size_category} model: "
                      f"base_lr={base_lr:.2e}, scaled_lr={final_lr:.2e}, "
                      f"batch_scale={batch_scale_factor:.2f}")
        
        return OptimizationResult(
            strategy=strategy,
            config=config,
            expected_improvement=expected_improvement,
            confidence=0.8,
            explanation=explanation
        )
    
    def optimize_mixed_precision(self, model_config: Dict[str, Any],
                               hardware_config: Dict[str, Any]) -> OptimizationResult:
        """Optimize mixed precision configuration.
        
        Args:
            model_config: Model configuration
            hardware_config: Hardware configuration
            
        Returns:
            Optimization result with mixed precision settings
        """
        num_params = model_config.get("parameters", 1_000_000_000)
        model_type = model_config.get("model_type", "transformer")
        precision_support = hardware_config.get("bf16_support", True)
        
        if not precision_support:
            config = {"use_mixed_precision": False}
            return OptimizationResult(
                strategy=OptimizationStrategy.BALANCED,
                config=config,
                expected_improvement={},
                confidence=1.0,
                explanation="BF16 not supported on this hardware"
            )
        
        # Mixed precision is generally beneficial for large models
        model_size_category = self._classify_model_size(num_params)
        
        if model_size_category in ["large", "xlarge"]:
            # Aggressive mixed precision for large models
            config = {
                "use_mixed_precision": True,
                "precision": "bf16-mixed",
                "loss_scaling": "dynamic",
                "fp32_operations": ["layer_norm", "softmax"]
            }
            expected_improvement = {"throughput": 40, "memory_usage": -25}
            confidence = 0.9
            
        elif model_size_category == "medium":
            # Conservative mixed precision
            config = {
                "use_mixed_precision": True,
                "precision": "bf16-mixed", 
                "loss_scaling": "static",
                "fp32_operations": ["layer_norm", "softmax", "embedding"]
            }
            expected_improvement = {"throughput": 30, "memory_usage": -20}
            confidence = 0.85
            
        else:  # small models
            # May not benefit as much from mixed precision
            config = {
                "use_mixed_precision": True,
                "precision": "bf16-mixed",
                "loss_scaling": "static"
            }
            expected_improvement = {"throughput": 20, "memory_usage": -15}
            confidence = 0.7
        
        explanation = f"Mixed precision recommended for {model_size_category} model with {expected_improvement['throughput']}% speedup"
        
        return OptimizationResult(
            strategy=OptimizationStrategy.THROUGHPUT,
            config=config,
            expected_improvement=expected_improvement,
            confidence=confidence,
            explanation=explanation
        )
    
    def optimize_parallelism(self, model_config: Dict[str, Any],
                           hardware_config: Dict[str, Any]) -> OptimizationResult:
        """Optimize parallelism strategy (data, model, pipeline).
        
        Args:
            model_config: Model configuration
            hardware_config: Hardware configuration
            
        Returns:
            Optimization result with parallelism configuration
        """
        num_params = model_config.get("parameters", 1_000_000_000)
        num_hpus = hardware_config.get("num_hpus", 8)
        num_nodes = hardware_config.get("num_nodes", 1)
        memory_per_hpu = hardware_config.get("memory_per_hpu_gb", 32)
        
        model_size_category = self._classify_model_size(num_params)
        
        # Calculate if model fits in single HPU memory
        model_memory_gb = (num_params * 4) / (1024**3) * 1.5  # 1.5x for gradients
        fits_single_hpu = model_memory_gb < memory_per_hpu * 0.8
        
        if fits_single_hpu and num_hpus <= 8:
            # Use data parallelism
            config = {
                "strategy": "data_parallel",
                "data_parallel_size": num_hpus,
                "model_parallel_size": 1,
                "pipeline_parallel_size": 1
            }
            expected_improvement = {"throughput": num_hpus * 0.9}  # 90% scaling efficiency
            explanation = "Data parallelism for model that fits in single HPU"
            
        elif not fits_single_hpu and model_size_category in ["large", "xlarge"]:
            # Use model parallelism + data parallelism
            if num_hpus <= 8:
                model_parallel = min(4, num_hpus)
                data_parallel = num_hpus // model_parallel
            else:
                model_parallel = 8
                data_parallel = num_hpus // 8
            
            config = {
                "strategy": "hybrid_parallel",
                "data_parallel_size": data_parallel,
                "model_parallel_size": model_parallel,
                "pipeline_parallel_size": 1
            }
            expected_improvement = {"throughput": num_hpus * 0.7}  # 70% scaling efficiency
            explanation = f"Hybrid parallelism: {model_parallel}x model, {data_parallel}x data"
            
        else:
            # Large scale: use 3D parallelism
            pipeline_parallel = min(4, num_nodes)
            model_parallel = min(8, num_hpus // pipeline_parallel)
            data_parallel = num_hpus // (model_parallel * pipeline_parallel)
            
            config = {
                "strategy": "3d_parallel",
                "data_parallel_size": data_parallel,
                "model_parallel_size": model_parallel,
                "pipeline_parallel_size": pipeline_parallel
            }
            expected_improvement = {"throughput": num_hpus * 0.6}  # 60% scaling efficiency
            explanation = f"3D parallelism: {pipeline_parallel}x pipeline, {model_parallel}x model, {data_parallel}x data"
        
        return OptimizationResult(
            strategy=OptimizationStrategy.THROUGHPUT,
            config=config,
            expected_improvement={"scaling_efficiency": expected_improvement["throughput"] / num_hpus},
            confidence=0.8,
            explanation=explanation
        )
    
    def _classify_model_size(self, num_params: int) -> str:
        """Classify model size."""
        if num_params < 1_000_000_000:
            return "small"
        elif num_params < 10_000_000_000:
            return "medium"
        elif num_params < 100_000_000_000:
            return "large"
        else:
            return "xlarge"
    
    def _optimize_for_throughput(self, max_batch_size: int, model_size: str, num_hpus: int) -> int:
        """Optimize batch size for maximum throughput."""
        # Larger batch sizes generally better for throughput
        target_batch_sizes = {
            "small": min(max_batch_size, 128),
            "medium": min(max_batch_size, 64),
            "large": min(max_batch_size, 32),
            "xlarge": min(max_batch_size, 16)
        }
        return target_batch_sizes[model_size]
    
    def _optimize_for_cost(self, max_batch_size: int, model_size: str, num_hpus: int) -> int:
        """Optimize batch size for minimum cost per token."""
        # Balance between throughput and convergence
        target_batch_sizes = {
            "small": min(max_batch_size, 96),
            "medium": min(max_batch_size, 48),
            "large": min(max_batch_size, 24),
            "xlarge": min(max_batch_size, 12)
        }
        return target_batch_sizes[model_size]
    
    def _optimize_balanced(self, max_batch_size: int, model_size: str, num_hpus: int) -> int:
        """Optimize batch size for balanced performance."""
        # Conservative but reliable settings
        target_batch_sizes = {
            "small": min(max_batch_size, 64),
            "medium": min(max_batch_size, 32),
            "large": min(max_batch_size, 16),
            "xlarge": min(max_batch_size, 8)
        }
        return target_batch_sizes[model_size]
    
    def _get_target_effective_batch(self, model_size: str) -> int:
        """Get target effective batch size for good convergence."""
        target_batches = {
            "small": 256,
            "medium": 512,
            "large": 1024,
            "xlarge": 2048
        }
        return target_batches[model_size]


class ResourceOptimizer:
    """Optimizes resource allocation and scaling decisions."""
    
    def __init__(self):
        """Initialize resource optimizer."""
        self.scaling_efficiency = {
            1: 1.0,     # Single node baseline
            2: 0.95,    # 95% efficiency with 2 nodes
            4: 0.88,    # 88% efficiency with 4 nodes
            8: 0.80,    # 80% efficiency with 8 nodes
            16: 0.70    # 70% efficiency with 16 nodes
        }
    
    def optimize_cluster_size(self, workload_config: Dict[str, Any],
                            cost_constraints: Dict[str, Any] = None) -> OptimizationResult:
        """Optimize cluster size for given workload.
        
        Args:
            workload_config: Workload configuration and requirements
            cost_constraints: Optional cost constraints
            
        Returns:
            Optimization result with recommended cluster configuration
        """
        target_completion_time = workload_config.get("target_completion_hours", 24)
        model_params = workload_config.get("parameters", 1_000_000_000)
        dataset_size = workload_config.get("dataset_size_tokens", 1_000_000_000)
        
        # Estimate single-node training time
        model_size_category = self._classify_model_size(model_params)
        base_throughput = self._get_base_throughput(model_size_category)
        single_node_hours = dataset_size / (base_throughput * 3600)
        
        # Find optimal number of nodes
        optimal_nodes = 1
        best_cost_efficiency = float('inf')
        
        max_budget = cost_constraints.get("max_cost_usd", 10000) if cost_constraints else 10000
        hourly_cost_per_node = 32.77  # AWS dl2q.24xlarge
        
        for num_nodes in [1, 2, 4, 8, 16]:
            if num_nodes not in self.scaling_efficiency:
                continue
            
            # Calculate training time with scaling efficiency
            efficiency = self.scaling_efficiency[num_nodes]
            training_hours = single_node_hours / (num_nodes * efficiency)
            
            # Calculate total cost
            total_cost = training_hours * hourly_cost_per_node * num_nodes
            
            # Check constraints
            if training_hours > target_completion_time:
                continue
            if total_cost > max_budget:
                continue
            
            # Calculate cost efficiency (cost per speedup)
            speedup = single_node_hours / training_hours
            cost_efficiency = total_cost / speedup
            
            if cost_efficiency < best_cost_efficiency:
                best_cost_efficiency = cost_efficiency
                optimal_nodes = num_nodes
        
        # Calculate expected performance
        efficiency = self.scaling_efficiency[optimal_nodes]
        expected_training_hours = single_node_hours / (optimal_nodes * efficiency)
        expected_cost = expected_training_hours * hourly_cost_per_node * optimal_nodes
        
        config = {
            "num_nodes": optimal_nodes,
            "hpus_per_node": 8,
            "total_hpus": optimal_nodes * 8,
            "estimated_training_hours": expected_training_hours,
            "estimated_cost_usd": expected_cost
        }
        
        expected_improvement = {
            "speedup": optimal_nodes * efficiency,
            "cost_efficiency": best_cost_efficiency
        }
        
        explanation = (f"Optimal cluster: {optimal_nodes} nodes, "
                      f"{expected_training_hours:.1f}h training time, "
                      f"${expected_cost:.0f} total cost")
        
        return OptimizationResult(
            strategy=OptimizationStrategy.BALANCED,
            config=config,
            expected_improvement=expected_improvement,
            confidence=0.8,
            explanation=explanation
        )
    
    def optimize_auto_scaling(self, workload_pattern: Dict[str, Any]) -> OptimizationResult:
        """Optimize auto-scaling configuration.
        
        Args:
            workload_pattern: Historical workload patterns
            
        Returns:
            Optimization result with auto-scaling configuration
        """
        peak_utilization = workload_pattern.get("peak_utilization_percent", 90)
        average_utilization = workload_pattern.get("average_utilization_percent", 60)
        workload_variance = workload_pattern.get("variance", 0.3)
        
        # Configure scaling thresholds based on workload patterns
        if workload_variance < 0.2:  # Stable workload
            scale_up_threshold = 85
            scale_down_threshold = 40
            scale_up_cooldown = 300   # 5 minutes
            scale_down_cooldown = 600  # 10 minutes
        elif workload_variance < 0.5:  # Moderate variance
            scale_up_threshold = 80
            scale_down_threshold = 35
            scale_up_cooldown = 180   # 3 minutes
            scale_down_cooldown = 900  # 15 minutes
        else:  # High variance
            scale_up_threshold = 75
            scale_down_threshold = 30
            scale_up_cooldown = 120   # 2 minutes
            scale_down_cooldown = 1200  # 20 minutes
        
        config = {
            "auto_scaling_enabled": True,
            "min_nodes": 1,
            "max_nodes": 8,
            "scale_up_threshold": scale_up_threshold,
            "scale_down_threshold": scale_down_threshold,
            "scale_up_cooldown_seconds": scale_up_cooldown,
            "scale_down_cooldown_seconds": scale_down_cooldown,
            "target_utilization": 70
        }
        
        expected_improvement = {
            "cost_savings": 20,  # 20% cost savings from right-sizing
            "availability": 95   # 95% availability during traffic spikes
        }
        
        explanation = (f"Auto-scaling config for {workload_variance:.1f} variance workload: "
                      f"scale up at {scale_up_threshold}%, down at {scale_down_threshold}%")
        
        return OptimizationResult(
            strategy=OptimizationStrategy.COST,
            config=config,
            expected_improvement=expected_improvement,
            confidence=0.75,
            explanation=explanation
        )
    
    def _classify_model_size(self, num_params: int) -> str:
        """Classify model size."""
        if num_params < 1_000_000_000:
            return "small"
        elif num_params < 10_000_000_000:
            return "medium"
        elif num_params < 100_000_000_000:
            return "large"
        else:
            return "xlarge"
    
    def _get_base_throughput(self, model_size: str) -> float:
        """Get base throughput for model size."""
        base_throughputs = {
            "small": 15000,   # tokens/sec
            "medium": 8000,
            "large": 2000,
            "xlarge": 500
        }
        return base_throughputs[model_size]