"""Performance and cost prediction algorithms."""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Prediction result with confidence intervals."""
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    model_accuracy: float
    prediction_horizon: str


class PerformancePredictor:
    """Predicts training performance based on historical data and model characteristics.
    
    Uses statistical models and machine learning principles to predict
    training throughput, convergence time, and resource utilization.
    """
    
    def __init__(self):
        """Initialize performance predictor."""
        # Historical performance data for different model types
        self.performance_history: List[Dict[str, Any]] = []
        
        # Model coefficients for prediction (simplified linear models)
        self.throughput_model = {
            "base_throughput": {
                "small": 15000,   # tokens/sec for 1B param model
                "medium": 8000,   # tokens/sec for 7B param model
                "large": 2000,    # tokens/sec for 70B param model
                "xlarge": 500     # tokens/sec for 175B+ param model
            },
            "batch_size_factor": 0.8,      # Throughput scales with batch size^0.8
            "sequence_length_factor": -0.3, # Throughput scales with seq_len^-0.3
            "precision_factor": {
                "fp32": 1.0,
                "bf16": 1.4,      # 40% speedup with BF16
                "fp16": 1.5       # 50% speedup with FP16
            },
            "node_scaling": {
                1: 1.0,
                2: 1.9,    # 95% efficiency
                4: 3.6,    # 90% efficiency
                8: 6.8,    # 85% efficiency
                16: 12.8   # 80% efficiency
            }
        }
    
    def predict_training_throughput(self, model_config: Dict[str, Any], 
                                  hardware_config: Dict[str, Any]) -> PredictionResult:
        """Predict training throughput.
        
        Args:
            model_config: Model configuration (parameters, sequence length, etc.)
            hardware_config: Hardware configuration (nodes, batch size, etc.)
            
        Returns:
            Predicted throughput with confidence interval
        """
        # Extract parameters
        num_params = model_config.get("parameters", 7_000_000_000)
        seq_length = model_config.get("sequence_length", 2048)
        batch_size = hardware_config.get("batch_size", 32)
        num_nodes = hardware_config.get("num_nodes", 1)
        precision = model_config.get("precision", "bf16")
        
        # Classify model size
        model_size = self._classify_model_size(num_params)
        
        # Base throughput for model size
        base_throughput = self.throughput_model["base_throughput"][model_size]
        
        # Apply scaling factors
        batch_factor = (batch_size / 32) ** self.throughput_model["batch_size_factor"]
        seq_factor = (seq_length / 2048) ** self.throughput_model["sequence_length_factor"]
        precision_factor = self.throughput_model["precision_factor"].get(precision, 1.0)
        node_factor = self._interpolate_node_scaling(num_nodes)
        
        # Calculate predicted throughput
        predicted_throughput = (base_throughput * batch_factor * seq_factor * 
                              precision_factor * node_factor)
        
        # Calculate confidence interval based on historical variance
        variance_factor = 0.15  # 15% typical variance
        confidence_interval = (
            predicted_throughput * (1 - variance_factor),
            predicted_throughput * (1 + variance_factor)
        )
        
        # Model accuracy based on historical predictions
        model_accuracy = self._calculate_model_accuracy("throughput")
        
        return PredictionResult(
            predicted_value=predicted_throughput,
            confidence_interval=confidence_interval,
            confidence_level=0.85,
            model_accuracy=model_accuracy,
            prediction_horizon="current_configuration"
        )
    
    def predict_convergence_time(self, model_config: Dict[str, Any],
                               training_config: Dict[str, Any]) -> PredictionResult:
        """Predict time to convergence.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration (dataset size, target loss, etc.)
            
        Returns:
            Predicted convergence time in hours
        """
        # Extract parameters
        num_params = model_config.get("parameters", 7_000_000_000)
        dataset_size = training_config.get("dataset_size_tokens", 1_000_000_000)
        target_loss = training_config.get("target_loss", None)
        learning_rate = training_config.get("learning_rate", 1e-4)
        batch_size = training_config.get("effective_batch_size", 512)
        
        # Estimate tokens needed for convergence
        model_size = self._classify_model_size(num_params)
        
        # Scaling laws for tokens needed (simplified Chinchilla scaling)
        tokens_for_convergence = {
            "small": num_params * 20,      # 20 tokens per parameter
            "medium": num_params * 25,     # 25 tokens per parameter
            "large": num_params * 30,      # 30 tokens per parameter
            "xlarge": num_params * 35      # 35 tokens per parameter
        }
        
        required_tokens = tokens_for_convergence[model_size]
        
        # Limit by dataset size
        actual_tokens = min(required_tokens, dataset_size)
        
        # Calculate number of steps
        steps_needed = actual_tokens / batch_size
        
        # Estimate throughput
        throughput_prediction = self.predict_training_throughput(
            model_config, training_config
        )
        
        # Calculate time to convergence
        convergence_hours = steps_needed / (throughput_prediction.predicted_value / batch_size * 3600)
        
        # Adjust for learning rate (higher LR = faster convergence, but less stable)
        lr_factor = math.sqrt(learning_rate / 1e-4)  # Relative to baseline LR
        convergence_hours = convergence_hours / lr_factor
        
        # Add variance based on model size (larger models more unpredictable)
        variance_factors = {
            "small": 0.2,
            "medium": 0.25,
            "large": 0.3,
            "xlarge": 0.4
        }
        
        variance = variance_factors[model_size]
        confidence_interval = (
            convergence_hours * (1 - variance),
            convergence_hours * (1 + variance)
        )
        
        return PredictionResult(
            predicted_value=convergence_hours,
            confidence_interval=confidence_interval,
            confidence_level=0.75,  # Lower confidence for convergence predictions
            model_accuracy=0.65,    # Convergence is harder to predict
            prediction_horizon="training_completion"
        )
    
    def predict_resource_utilization(self, training_config: Dict[str, Any]) -> Dict[str, PredictionResult]:
        """Predict resource utilization metrics.
        
        Args:
            training_config: Training configuration
            
        Returns:
            Dictionary of predicted utilization metrics
        """
        batch_size = training_config.get("batch_size", 32)
        sequence_length = training_config.get("sequence_length", 2048)
        num_params = training_config.get("parameters", 7_000_000_000)
        precision = training_config.get("precision", "bf16")
        
        model_size = self._classify_model_size(num_params)
        
        # Predict HPU utilization
        base_hpu_util = {
            "small": 85,
            "medium": 88,
            "large": 92,
            "xlarge": 95
        }[model_size]
        
        # Adjust for batch size (larger batches = better utilization)
        batch_factor = min(1.2, (batch_size / 16) ** 0.3)
        predicted_hpu_util = min(98, base_hpu_util * batch_factor)
        
        # Predict memory utilization
        memory_per_param = 4 if precision == "fp32" else 2  # bytes
        model_memory = num_params * memory_per_param / (1024**3)  # GB
        
        # Account for activations (scales with batch size and sequence length)
        activation_memory = (batch_size * sequence_length * 4 * 
                           math.sqrt(num_params)) / (1024**3)
        
        total_memory = (model_memory + activation_memory) * 1.5  # 1.5x for gradients
        available_memory = 32  # GB per HPU
        predicted_memory_util = min(95, (total_memory / available_memory) * 100)
        
        # Predict power utilization
        base_power_util = {
            "small": 70,
            "medium": 80,
            "large": 88,
            "xlarge": 93
        }[model_size]
        
        predictions = {}
        
        for metric, value in [("hpu_utilization", predicted_hpu_util),
                             ("memory_utilization", predicted_memory_util),
                             ("power_utilization", base_power_util)]:
            
            variance = 0.1  # 10% variance
            predictions[metric] = PredictionResult(
                predicted_value=value,
                confidence_interval=(value * (1 - variance), value * (1 + variance)),
                confidence_level=0.8,
                model_accuracy=0.75,
                prediction_horizon="steady_state"
            )
        
        return predictions
    
    def add_training_result(self, config: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Add training result to historical data for model improvement.
        
        Args:
            config: Training configuration used
            results: Actual training results
        """
        data_point = {
            "timestamp": datetime.now(),
            "config": config.copy(),
            "results": results.copy()
        }
        
        self.performance_history.append(data_point)
        
        # Keep only recent history (last 1000 runs)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        logger.info(f"Added training result to prediction model (history size: {len(self.performance_history)})")
    
    def _classify_model_size(self, num_params: int) -> str:
        """Classify model size category."""
        if num_params < 1_000_000_000:
            return "small"
        elif num_params < 10_000_000_000:
            return "medium"
        elif num_params < 100_000_000_000:
            return "large"
        else:
            return "xlarge"
    
    def _interpolate_node_scaling(self, num_nodes: int) -> float:
        """Interpolate node scaling factor."""
        scaling = self.throughput_model["node_scaling"]
        
        if num_nodes in scaling:
            return scaling[num_nodes]
        
        # Linear interpolation for intermediate values
        keys = sorted(scaling.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= num_nodes <= keys[i + 1]:
                # Linear interpolation
                x1, y1 = keys[i], scaling[keys[i]]
                x2, y2 = keys[i + 1], scaling[keys[i + 1]]
                return y1 + (y2 - y1) * (num_nodes - x1) / (x2 - x1)
        
        # Extrapolation for values outside range
        if num_nodes < keys[0]:
            return scaling[keys[0]]
        else:
            # Assume diminishing returns for very large clusters
            return scaling[keys[-1]] * (keys[-1] / num_nodes) ** 0.2
    
    def _calculate_model_accuracy(self, metric: str) -> float:
        """Calculate model accuracy based on historical predictions."""
        if not self.performance_history:
            return 0.7  # Default accuracy
        
        # This would implement actual accuracy calculation
        # For now, return estimated accuracy based on data size
        data_size = len(self.performance_history)
        if data_size < 10:
            return 0.6
        elif data_size < 50:
            return 0.75
        elif data_size < 200:
            return 0.85
        else:
            return 0.9


class CostPredictor:
    """Predicts training costs based on configuration and market data."""
    
    def __init__(self):
        """Initialize cost predictor."""
        # Instance pricing data (per hour)
        self.instance_pricing = {
            "aws": {
                "dl2q.24xlarge": 32.77,    # 8 Gaudi 3 HPUs
                "p4d.24xlarge": 98.32,     # 8 A100 GPUs
                "p3.16xlarge": 52.88       # 8 V100 GPUs
            },
            "azure": {
                "ND96isr_H100_v5": 98.50,  # 8 H100 GPUs
                "ND96amsr_A100_v4": 76.20  # 8 A100 GPUs
            }
        }
        
        # Spot instance discounts
        self.spot_discounts = {
            "aws": 0.6,    # 60% discount typical
            "azure": 0.7   # 70% discount typical
        }
        
        # Storage and network costs
        self.storage_cost_per_gb_month = 0.023  # USD
        self.network_cost_per_gb = 0.09         # USD
    
    def predict_training_cost(self, training_config: Dict[str, Any],
                            performance_prediction: PredictionResult) -> PredictionResult:
        """Predict total training cost.
        
        Args:
            training_config: Training configuration
            performance_prediction: Predicted performance metrics
            
        Returns:
            Predicted cost with confidence interval
        """
        # Extract configuration
        provider = training_config.get("cloud_provider", "aws")
        instance_type = training_config.get("instance_type", "dl2q.24xlarge")
        num_nodes = training_config.get("num_nodes", 1)
        use_spot = training_config.get("use_spot_instances", False)
        dataset_size_gb = training_config.get("dataset_size_gb", 100)
        
        # Get instance pricing
        hourly_cost = self.instance_pricing.get(provider, {}).get(instance_type, 32.77)
        
        # Apply spot discount if applicable
        if use_spot:
            spot_discount = self.spot_discounts.get(provider, 0.6)
            hourly_cost *= (1 - spot_discount)
        
        # Calculate compute cost from performance prediction
        training_hours = performance_prediction.predicted_value
        compute_cost = hourly_cost * num_nodes * training_hours
        
        # Calculate storage cost (assume data stored for training duration + 1 month)
        storage_months = max(0.1, training_hours / 720 + 1)  # 720 hours per month
        storage_cost = dataset_size_gb * self.storage_cost_per_gb_month * storage_months
        
        # Calculate network cost (assume 10% of dataset transferred)
        network_transfer_gb = dataset_size_gb * 0.1
        network_cost = network_transfer_gb * self.network_cost_per_gb
        
        # Add overhead (monitoring, management, etc.)
        overhead_cost = compute_cost * 0.1
        
        # Total cost
        total_cost = compute_cost + storage_cost + network_cost + overhead_cost
        
        # Calculate confidence interval
        # Cost prediction is generally more accurate than performance
        cost_variance = 0.1  # 10% variance
        confidence_interval = (
            total_cost * (1 - cost_variance),
            total_cost * (1 + cost_variance)
        )
        
        return PredictionResult(
            predicted_value=total_cost,
            confidence_interval=confidence_interval,
            confidence_level=0.9,
            model_accuracy=0.85,
            prediction_horizon="training_completion"
        )
    
    def predict_cost_per_token(self, training_config: Dict[str, Any],
                             cost_prediction: PredictionResult,
                             throughput_prediction: PredictionResult) -> PredictionResult:
        """Predict cost per million tokens.
        
        Args:
            training_config: Training configuration
            cost_prediction: Predicted total cost
            throughput_prediction: Predicted throughput
            
        Returns:
            Predicted cost per million tokens
        """
        # Calculate total tokens processed
        training_hours = training_config.get("training_hours", 
                                           throughput_prediction.predicted_value)
        avg_throughput = throughput_prediction.predicted_value
        
        total_tokens = avg_throughput * training_hours * 3600  # tokens per second * seconds
        
        # Cost per million tokens
        cost_per_million_tokens = (cost_prediction.predicted_value / total_tokens) * 1_000_000
        
        # Propagate uncertainty from both cost and throughput predictions
        cost_variance = ((cost_prediction.confidence_interval[1] - 
                         cost_prediction.confidence_interval[0]) / 
                        (2 * cost_prediction.predicted_value))
        
        throughput_variance = ((throughput_prediction.confidence_interval[1] - 
                               throughput_prediction.confidence_interval[0]) / 
                              (2 * throughput_prediction.predicted_value))
        
        # Combined variance (assuming independence)
        combined_variance = math.sqrt(cost_variance**2 + throughput_variance**2)
        
        confidence_interval = (
            cost_per_million_tokens * (1 - combined_variance),
            cost_per_million_tokens * (1 + combined_variance)
        )
        
        return PredictionResult(
            predicted_value=cost_per_million_tokens,
            confidence_interval=confidence_interval,
            confidence_level=0.8,
            model_accuracy=0.75,
            prediction_horizon="training_completion"
        )
    
    def compare_provider_costs(self, training_config: Dict[str, Any],
                             performance_prediction: PredictionResult) -> Dict[str, PredictionResult]:
        """Compare costs across different providers.
        
        Args:
            training_config: Training configuration
            performance_prediction: Performance prediction
            
        Returns:
            Cost predictions for each provider
        """
        comparisons = {}
        original_provider = training_config.get("cloud_provider", "aws")
        
        for provider in self.instance_pricing.keys():
            # Create config for this provider
            provider_config = training_config.copy()
            provider_config["cloud_provider"] = provider
            
            # Map to equivalent instance type
            if provider == "aws":
                provider_config["instance_type"] = "dl2q.24xlarge"
            elif provider == "azure":
                provider_config["instance_type"] = "ND96amsr_A100_v4"
            
            # Predict cost for this provider
            cost_prediction = self.predict_training_cost(provider_config, performance_prediction)
            comparisons[provider] = cost_prediction
        
        return comparisons
    
    def predict_monthly_cost(self, usage_pattern: Dict[str, Any],
                           instance_config: Dict[str, Any]) -> PredictionResult:
        """Predict monthly costs based on usage patterns.
        
        Args:
            usage_pattern: Monthly usage pattern
            instance_config: Instance configuration
            
        Returns:
            Predicted monthly cost
        """
        # Extract usage parameters
        monthly_training_hours = usage_pattern.get("training_hours_per_month", 200)
        development_hours = usage_pattern.get("development_hours_per_month", 50)
        num_users = usage_pattern.get("number_of_users", 5)
        
        # Extract instance configuration
        provider = instance_config.get("cloud_provider", "aws")
        instance_type = instance_config.get("instance_type", "dl2q.24xlarge")
        use_spot = instance_config.get("use_spot_instances", False)
        
        # Calculate base cost
        hourly_cost = self.instance_pricing.get(provider, {}).get(instance_type, 32.77)
        
        if use_spot:
            spot_discount = self.spot_discounts.get(provider, 0.6)
            hourly_cost *= (1 - spot_discount)
        
        # Calculate monthly costs
        training_cost = monthly_training_hours * hourly_cost
        development_cost = development_hours * hourly_cost * 0.5  # Smaller instances for dev
        
        # Storage cost (datasets and models)
        avg_storage_gb = 500 * num_users  # 500GB per user
        storage_cost = avg_storage_gb * self.storage_cost_per_gb_month
        
        # Network cost (data transfers)
        network_cost = 50 * num_users  # $50 per user per month
        
        total_monthly_cost = training_cost + development_cost + storage_cost + network_cost
        
        # Add variance for monthly predictions (higher due to usage uncertainty)
        variance = 0.25  # 25% variance
        confidence_interval = (
            total_monthly_cost * (1 - variance),
            total_monthly_cost * (1 + variance)
        )
        
        return PredictionResult(
            predicted_value=total_monthly_cost,
            confidence_interval=confidence_interval,
            confidence_level=0.75,
            model_accuracy=0.7,
            prediction_horizon="monthly_usage"
        )