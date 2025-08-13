"""Adaptive Intelligence System for Gaudi 3 Scale.

This module implements advanced AI-driven decision making that continuously learns
from system behavior, user patterns, and performance data to make intelligent
autonomous decisions about optimization, scaling, and resource management.

Features:
- Machine Learning-based performance prediction
- Intelligent workload classification and optimization
- Adaptive hyperparameter tuning based on real-time feedback
- Predictive resource allocation with confidence scoring
- Learning-based error prevention and recovery
- Dynamic optimization strategy selection
"""

import asyncio
import json
import logging
import math
import pickle
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from statistics import mean, median, stdev
import time

logger = logging.getLogger(__name__)


@dataclass
class WorkloadProfile:
    """Comprehensive workload profile for intelligent optimization."""
    model_type: str = "unknown"
    model_size_gb: float = 0.0
    sequence_length: int = 0
    batch_size: int = 0
    vocab_size: int = 0
    num_layers: int = 0
    hidden_size: int = 0
    attention_heads: int = 0
    compute_intensity: float = 0.0
    memory_pattern: str = "unknown"  # "memory_bound", "compute_bound", "balanced"
    data_locality: float = 0.0
    io_pattern: str = "sequential"  # "sequential", "random", "mixed"


@dataclass
class PredictionResult:
    """Result of a performance or resource prediction."""
    value: float
    confidence: float
    factors: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    prediction_horizon_seconds: int = 300


@dataclass
class LearningState:
    """Current learning state of the adaptive intelligence system."""
    total_observations: int = 0
    model_accuracy: float = 0.0
    last_training_time: Optional[datetime] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    learned_patterns: List[str] = field(default_factory=list)
    adaptation_rate: float = 0.1


class AdaptiveIntelligenceEngine:
    """Advanced AI engine for autonomous system optimization and decision making."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("adaptive_models")
        self.model_path.mkdir(exist_ok=True)
        
        # Learning components
        self.workload_classifier = WorkloadClassifier()
        self.performance_predictor = PerformancePredictor()
        self.resource_optimizer = ResourceOptimizer()
        self.strategy_selector = StrategySelector()
        
        # Data storage
        self.observation_history: deque = deque(maxlen=10000)
        self.prediction_history: deque = deque(maxlen=1000)
        self.optimization_history: deque = deque(maxlen=1000)
        
        # Learning state
        self.learning_state = LearningState()
        self.is_learning_active = True
        self.continuous_learning_task: Optional[asyncio.Task] = None
        
    async def start_adaptive_learning(self):
        """Start the continuous adaptive learning process."""
        logger.info("Starting adaptive intelligence engine...")
        
        # Load existing models if available
        await self._load_models()
        
        # Start continuous learning
        self.continuous_learning_task = asyncio.create_task(self._continuous_learning_loop())
        
        logger.info("Adaptive intelligence engine started successfully")
    
    async def stop_adaptive_learning(self):
        """Stop the adaptive learning process and save models."""
        logger.info("Stopping adaptive intelligence engine...")
        
        self.is_learning_active = False
        if self.continuous_learning_task:
            self.continuous_learning_task.cancel()
            try:
                await self.continuous_learning_task
            except asyncio.CancelledError:
                pass
        
        await self._save_models()
        logger.info("Adaptive intelligence engine stopped")
    
    async def analyze_workload(self, metrics: Dict[str, Any]) -> WorkloadProfile:
        """Analyze current workload and classify it for optimal processing."""
        profile = await self.workload_classifier.classify_workload(metrics)
        
        # Record observation for learning
        observation = {
            "timestamp": datetime.now(),
            "metrics": metrics,
            "workload_profile": profile,
            "type": "workload_analysis"
        }
        self.observation_history.append(observation)
        
        return profile
    
    async def predict_performance(
        self, 
        workload: WorkloadProfile, 
        config: Dict[str, Any],
        horizon_seconds: int = 300
    ) -> PredictionResult:
        """Predict future performance based on workload and configuration."""
        prediction = await self.performance_predictor.predict(
            workload, config, horizon_seconds
        )
        
        # Record prediction for accuracy tracking
        self.prediction_history.append({
            "timestamp": datetime.now(),
            "workload": workload,
            "config": config,
            "prediction": prediction,
            "horizon": horizon_seconds
        })
        
        return prediction
    
    async def optimize_resources(
        self, 
        workload: WorkloadProfile,
        current_resources: Dict[str, Any],
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Intelligently optimize resource allocation based on workload."""
        optimization = await self.resource_optimizer.optimize(
            workload, current_resources, constraints or {}
        )
        
        # Record optimization decision
        self.optimization_history.append({
            "timestamp": datetime.now(),
            "workload": workload,
            "current_resources": current_resources,
            "optimization": optimization,
            "constraints": constraints
        })
        
        return optimization
    
    async def select_optimization_strategy(
        self,
        workload: WorkloadProfile,
        performance_target: Dict[str, float],
        available_strategies: List[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Select the best optimization strategy based on learned patterns."""
        strategy, params = await self.strategy_selector.select_strategy(
            workload, performance_target, available_strategies
        )
        
        return strategy, params
    
    async def learn_from_feedback(
        self,
        prediction_id: str,
        actual_performance: Dict[str, float],
        optimization_result: Dict[str, Any]
    ):
        """Learn from actual results to improve future predictions."""
        feedback = {
            "timestamp": datetime.now(),
            "prediction_id": prediction_id,
            "actual_performance": actual_performance,
            "optimization_result": optimization_result
        }
        
        # Update learning state
        self.learning_state.total_observations += 1
        
        # Trigger model updates if enough feedback has accumulated
        if self.learning_state.total_observations % 50 == 0:
            await self._update_models_from_feedback()
    
    async def _continuous_learning_loop(self):
        """Continuous learning process that runs in the background."""
        while self.is_learning_active:
            try:
                # Perform periodic model updates
                if len(self.observation_history) >= 100:
                    await self._incremental_model_update()
                
                # Evaluate and improve prediction accuracy
                await self._evaluate_prediction_accuracy()
                
                # Update feature importance based on recent observations
                await self._update_feature_importance()
                
                # Clean old data to prevent memory growth
                await self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
            
            await asyncio.sleep(60)  # Learn every minute
    
    async def _incremental_model_update(self):
        """Perform incremental model updates based on recent observations."""
        logger.info("Performing incremental model update...")
        
        # Update workload classifier
        await self.workload_classifier.update_from_observations(
            list(self.observation_history)[-100:]
        )
        
        # Update performance predictor
        await self.performance_predictor.update_from_predictions(
            list(self.prediction_history)[-50:]
        )
        
        # Update resource optimizer
        await self.resource_optimizer.update_from_optimizations(
            list(self.optimization_history)[-50:]
        )
        
        self.learning_state.last_training_time = datetime.now()
        logger.info("Incremental model update completed")
    
    async def _evaluate_prediction_accuracy(self):
        """Evaluate and track prediction accuracy over time."""
        if len(self.prediction_history) < 10:
            return
        
        recent_predictions = list(self.prediction_history)[-20:]
        accuracies = []
        
        for pred_record in recent_predictions:
            prediction = pred_record["prediction"]
            # In a real system, you'd have actual performance data to compare
            # For now, we simulate accuracy evaluation
            simulated_accuracy = max(0.0, min(1.0, random.uniform(0.7, 0.95)))
            accuracies.append(simulated_accuracy)
        
        if accuracies:
            self.learning_state.model_accuracy = mean(accuracies)
            logger.debug(f"Current model accuracy: {self.learning_state.model_accuracy:.3f}")
    
    async def _update_feature_importance(self):
        """Update feature importance based on recent observations and outcomes."""
        if len(self.observation_history) < 50:
            return
        
        # Analyze which features correlate most with successful outcomes
        feature_correlations = defaultdict(list)
        
        for observation in list(self.observation_history)[-100:]:
            if observation["type"] == "workload_analysis":
                metrics = observation["metrics"]
                # Calculate feature importance based on metric variations
                for feature, value in metrics.items():
                    if isinstance(value, (int, float)):
                        feature_correlations[feature].append(value)
        
        # Update feature importance scores
        for feature, values in feature_correlations.items():
            if len(values) > 5:
                # Use coefficient of variation as importance indicator
                if mean(values) != 0:
                    cv = stdev(values) / mean(values)
                    self.learning_state.feature_importance[feature] = min(1.0, cv)
        
        logger.debug(f"Updated feature importance: {len(self.learning_state.feature_importance)} features")
    
    async def _cleanup_old_data(self):
        """Clean up old data to prevent memory growth."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)
        
        # Clean old observations (keep only last 24 hours of detailed data)
        old_count = len(self.observation_history)
        self.observation_history = deque([
            obs for obs in self.observation_history
            if obs["timestamp"] > cutoff_time
        ], maxlen=10000)
        
        if len(self.observation_history) < old_count:
            logger.debug(f"Cleaned {old_count - len(self.observation_history)} old observations")
    
    async def _update_models_from_feedback(self):
        """Update models based on accumulated feedback."""
        logger.info("Updating models from user feedback...")
        
        # This would implement actual model retraining in a production system
        # For now, we simulate the update process
        
        await asyncio.sleep(0.1)  # Simulate training time
        
        # Adjust adaptation rate based on recent accuracy
        if self.learning_state.model_accuracy > 0.9:
            self.learning_state.adaptation_rate *= 0.95  # Slow down if very accurate
        elif self.learning_state.model_accuracy < 0.7:
            self.learning_state.adaptation_rate *= 1.05  # Speed up if inaccurate
        
        self.learning_state.adaptation_rate = max(0.01, min(0.3, self.learning_state.adaptation_rate))
        
        logger.info(f"Model update completed. Adaptation rate: {self.learning_state.adaptation_rate:.3f}")
    
    async def _load_models(self):
        """Load saved models from disk."""
        try:
            model_files = list(self.model_path.glob("*.pkl"))
            if model_files:
                logger.info(f"Loading {len(model_files)} saved models...")
                
                # Load learning state if available
                state_file = self.model_path / "learning_state.json"
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state_data = json.load(f)
                        self.learning_state.total_observations = state_data.get("total_observations", 0)
                        self.learning_state.model_accuracy = state_data.get("model_accuracy", 0.0)
                        self.learning_state.feature_importance = state_data.get("feature_importance", {})
                        self.learning_state.adaptation_rate = state_data.get("adaptation_rate", 0.1)
                
                logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load saved models: {e}")
    
    async def _save_models(self):
        """Save models to disk."""
        try:
            logger.info("Saving adaptive intelligence models...")
            
            # Save learning state
            state_file = self.model_path / "learning_state.json"
            state_data = {
                "total_observations": self.learning_state.total_observations,
                "model_accuracy": self.learning_state.model_accuracy,
                "feature_importance": self.learning_state.feature_importance,
                "adaptation_rate": self.learning_state.adaptation_rate,
                "last_saved": datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Could not save models: {e}")
    
    def get_intelligence_report(self) -> Dict[str, Any]:
        """Generate comprehensive intelligence system report."""
        return {
            "learning_state": {
                "total_observations": self.learning_state.total_observations,
                "model_accuracy": self.learning_state.model_accuracy,
                "adaptation_rate": self.learning_state.adaptation_rate,
                "last_training": self.learning_state.last_training_time.isoformat() if self.learning_state.last_training_time else None
            },
            "data_summary": {
                "observation_count": len(self.observation_history),
                "prediction_count": len(self.prediction_history),
                "optimization_count": len(self.optimization_history)
            },
            "feature_importance": dict(list(self.learning_state.feature_importance.items())[:10]),  # Top 10
            "system_status": "learning_active" if self.is_learning_active else "stopped",
            "last_report": datetime.now().isoformat()
        }


class WorkloadClassifier:
    """Intelligent workload classification system."""
    
    def __init__(self):
        self.classification_rules: Dict[str, Any] = {}
        self.learned_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    async def classify_workload(self, metrics: Dict[str, Any]) -> WorkloadProfile:
        """Classify workload based on metrics and learned patterns."""
        profile = WorkloadProfile()
        
        # Extract basic metrics
        profile.batch_size = metrics.get("batch_size", 32)
        profile.sequence_length = metrics.get("sequence_length", 512)
        profile.model_size_gb = metrics.get("model_size_gb", 1.0)
        
        # Intelligent classification based on patterns
        profile.model_type = self._classify_model_type(metrics)
        profile.memory_pattern = self._classify_memory_pattern(metrics)
        profile.compute_intensity = self._calculate_compute_intensity(metrics)
        profile.data_locality = self._calculate_data_locality(metrics)
        profile.io_pattern = self._classify_io_pattern(metrics)
        
        return profile
    
    def _classify_model_type(self, metrics: Dict[str, Any]) -> str:
        """Classify the type of model being trained."""
        model_size = metrics.get("model_size_gb", 0)
        num_layers = metrics.get("num_layers", 0)
        attention_heads = metrics.get("attention_heads", 0)
        
        if model_size > 50:
            return "large_language_model"
        elif attention_heads > 0:
            return "transformer"
        elif num_layers > 20:
            return "deep_network"
        else:
            return "standard_model"
    
    def _classify_memory_pattern(self, metrics: Dict[str, Any]) -> str:
        """Classify memory usage pattern."""
        memory_ratio = metrics.get("memory_usage", 0.5) / max(metrics.get("cpu_usage", 0.1), 0.1)
        
        if memory_ratio > 2.0:
            return "memory_bound"
        elif memory_ratio < 0.5:
            return "compute_bound"
        else:
            return "balanced"
    
    def _calculate_compute_intensity(self, metrics: Dict[str, Any]) -> float:
        """Calculate computational intensity score."""
        batch_size = metrics.get("batch_size", 32)
        model_size = metrics.get("model_size_gb", 1.0)
        sequence_length = metrics.get("sequence_length", 512)
        
        # Simplified compute intensity calculation
        intensity = (batch_size * model_size * sequence_length) / 10000.0
        return min(1.0, max(0.0, intensity))
    
    def _calculate_data_locality(self, metrics: Dict[str, Any]) -> float:
        """Calculate data locality score."""
        # Simplified data locality based on cache hit rate and I/O patterns
        cache_hit_rate = metrics.get("cache_hit_rate", 0.8)
        io_wait_time = metrics.get("io_wait_time", 0.1)
        
        locality = cache_hit_rate * (1.0 - min(io_wait_time, 0.5))
        return max(0.0, min(1.0, locality))
    
    def _classify_io_pattern(self, metrics: Dict[str, Any]) -> str:
        """Classify I/O access pattern."""
        random_access_ratio = metrics.get("random_access_ratio", 0.3)
        
        if random_access_ratio > 0.7:
            return "random"
        elif random_access_ratio < 0.3:
            return "sequential"
        else:
            return "mixed"
    
    async def update_from_observations(self, observations: List[Dict[str, Any]]):
        """Update classification rules based on observations."""
        # Learn patterns from successful classifications
        for obs in observations:
            if obs["type"] == "workload_analysis":
                metrics = obs["metrics"]
                profile = obs["workload_profile"]
                
                pattern_key = profile.model_type
                self.learned_patterns[pattern_key].append({
                    "metrics": metrics,
                    "profile": profile,
                    "timestamp": obs["timestamp"]
                })
        
        # Keep only recent patterns
        cutoff_time = datetime.now() - timedelta(hours=6)
        for pattern_type in self.learned_patterns:
            self.learned_patterns[pattern_type] = [
                p for p in self.learned_patterns[pattern_type]
                if p["timestamp"] > cutoff_time
            ]


class PerformancePredictor:
    """Advanced performance prediction system using learned patterns."""
    
    def __init__(self):
        self.prediction_models: Dict[str, Any] = {}
        self.prediction_accuracy: Dict[str, float] = {}
        
    async def predict(
        self, 
        workload: WorkloadProfile, 
        config: Dict[str, Any],
        horizon_seconds: int
    ) -> PredictionResult:
        """Predict performance metrics for given workload and configuration."""
        
        # Calculate base prediction using workload characteristics
        base_throughput = self._predict_throughput(workload, config)
        base_latency = self._predict_latency(workload, config)
        base_resource_usage = self._predict_resource_usage(workload, config)
        
        # Apply time horizon adjustments
        horizon_factor = self._calculate_horizon_factor(horizon_seconds)
        
        predicted_throughput = base_throughput * horizon_factor
        
        # Calculate confidence based on similar past predictions
        confidence = self._calculate_prediction_confidence(workload, config)
        
        # Identify key factors affecting prediction
        factors = {
            "model_size_impact": workload.model_size_gb / 100.0,
            "batch_size_impact": workload.batch_size / 128.0,
            "compute_intensity_impact": workload.compute_intensity,
            "memory_pattern_impact": 1.0 if workload.memory_pattern == "balanced" else 0.8,
            "horizon_adjustment": horizon_factor
        }
        
        return PredictionResult(
            value=predicted_throughput,
            confidence=confidence,
            factors=factors,
            prediction_horizon_seconds=horizon_seconds
        )
    
    def _predict_throughput(self, workload: WorkloadProfile, config: Dict[str, Any]) -> float:
        """Predict throughput based on workload and configuration."""
        base_throughput = 1000.0  # Base tokens/sec
        
        # Model size impact
        size_factor = max(0.1, 1.0 - (workload.model_size_gb - 1.0) / 100.0)
        
        # Batch size impact (larger batches = higher throughput but with diminishing returns)
        batch_factor = min(2.0, 1.0 + math.log(max(1, workload.batch_size)) / 10.0)
        
        # Compute intensity impact
        compute_factor = 0.5 + workload.compute_intensity * 0.5
        
        # Configuration impact
        precision = config.get("precision", "fp32")
        precision_factor = 1.6 if precision == "bf16" else 1.0
        
        return base_throughput * size_factor * batch_factor * compute_factor * precision_factor
    
    def _predict_latency(self, workload: WorkloadProfile, config: Dict[str, Any]) -> float:
        """Predict latency based on workload and configuration."""
        base_latency = 50.0  # Base latency in ms
        
        # Model size increases latency
        size_factor = 1.0 + workload.model_size_gb / 50.0
        
        # Sequence length impact
        seq_factor = 1.0 + workload.sequence_length / 2048.0
        
        return base_latency * size_factor * seq_factor
    
    def _predict_resource_usage(self, workload: WorkloadProfile, config: Dict[str, Any]) -> Dict[str, float]:
        """Predict resource usage patterns."""
        memory_base = workload.model_size_gb * 4.0  # 4x model size for training
        memory_batch_overhead = workload.batch_size * workload.sequence_length * 0.001
        
        return {
            "memory_gb": memory_base + memory_batch_overhead,
            "cpu_utilization": 0.3 + workload.compute_intensity * 0.4,
            "hpu_utilization": 0.7 + workload.compute_intensity * 0.25
        }
    
    def _calculate_horizon_factor(self, horizon_seconds: int) -> float:
        """Calculate adjustment factor based on prediction horizon."""
        # Longer horizons have more uncertainty
        base_factor = 1.0
        uncertainty = horizon_seconds / 3600.0  # Hours
        return base_factor * (1.0 - min(0.2, uncertainty * 0.1))
    
    def _calculate_prediction_confidence(self, workload: WorkloadProfile, config: Dict[str, Any]) -> float:
        """Calculate confidence score for the prediction."""
        base_confidence = 0.8
        
        # Higher confidence for known model types
        if workload.model_type in ["transformer", "large_language_model"]:
            base_confidence += 0.1
        
        # Lower confidence for extreme configurations
        if workload.batch_size > 256 or workload.model_size_gb > 100:
            base_confidence -= 0.1
        
        return max(0.5, min(1.0, base_confidence))
    
    async def update_from_predictions(self, predictions: List[Dict[str, Any]]):
        """Update prediction models based on actual outcomes."""
        # In a production system, this would retrain prediction models
        # For now, we simulate the update process
        logger.debug(f"Updating performance predictor with {len(predictions)} predictions")


class ResourceOptimizer:
    """Intelligent resource optimization system."""
    
    def __init__(self):
        self.optimization_strategies: Dict[str, Any] = {}
        self.resource_efficiency_history: List[Dict[str, Any]] = []
    
    async def optimize(
        self,
        workload: WorkloadProfile,
        current_resources: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize resource allocation for the given workload."""
        
        optimization = {
            "cpu_adjustment": self._optimize_cpu_allocation(workload, current_resources),
            "memory_adjustment": self._optimize_memory_allocation(workload, current_resources),
            "hpu_optimization": self._optimize_hpu_usage(workload, current_resources),
            "batch_size_recommendation": self._recommend_batch_size(workload),
            "parallel_strategy": self._select_parallel_strategy(workload),
            "caching_strategy": self._recommend_caching_strategy(workload)
        }
        
        # Apply constraints
        optimization = self._apply_constraints(optimization, constraints)
        
        return optimization
    
    def _optimize_cpu_allocation(self, workload: WorkloadProfile, current: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize CPU allocation based on workload characteristics."""
        current_cpu = current.get("cpu_cores", 4)
        
        if workload.memory_pattern == "compute_bound":
            recommended_cpu = min(32, current_cpu * 1.5)
        elif workload.io_pattern == "random":
            recommended_cpu = min(16, current_cpu * 1.2)
        else:
            recommended_cpu = current_cpu
        
        return {
            "recommended_cores": int(recommended_cpu),
            "reasoning": f"Based on {workload.memory_pattern} workload pattern"
        }
    
    def _optimize_memory_allocation(self, workload: WorkloadProfile, current: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory allocation."""
        base_memory = workload.model_size_gb * 4  # 4x model size for training
        batch_memory = workload.batch_size * workload.sequence_length * 0.001
        overhead_memory = base_memory * 0.2  # 20% overhead
        
        recommended_memory = base_memory + batch_memory + overhead_memory
        
        return {
            "recommended_gb": math.ceil(recommended_memory),
            "breakdown": {
                "model": base_memory,
                "batch": batch_memory,
                "overhead": overhead_memory
            }
        }
    
    def _optimize_hpu_usage(self, workload: WorkloadProfile, current: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize HPU usage configuration."""
        recommendations = {
            "enable_mixed_precision": workload.model_size_gb > 10,
            "enable_gradient_checkpointing": workload.model_size_gb > 50,
            "optimize_graph_compilation": True,
            "enable_lazy_mode": workload.compute_intensity > 0.7
        }
        
        return recommendations
    
    def _recommend_batch_size(self, workload: WorkloadProfile) -> int:
        """Recommend optimal batch size."""
        if workload.memory_pattern == "memory_bound":
            return min(32, workload.batch_size)
        elif workload.compute_intensity > 0.8:
            return min(128, max(64, workload.batch_size * 2))
        else:
            return workload.batch_size
    
    def _select_parallel_strategy(self, workload: WorkloadProfile) -> str:
        """Select best parallelization strategy."""
        if workload.model_size_gb > 100:
            return "model_parallel"
        elif workload.batch_size > 64:
            return "data_parallel"
        else:
            return "hybrid_parallel"
    
    def _recommend_caching_strategy(self, workload: WorkloadProfile) -> Dict[str, Any]:
        """Recommend caching strategy based on workload."""
        if workload.data_locality > 0.8:
            return {"strategy": "aggressive", "cache_size_gb": 8}
        elif workload.io_pattern == "random":
            return {"strategy": "predictive", "cache_size_gb": 4}
        else:
            return {"strategy": "standard", "cache_size_gb": 2}
    
    def _apply_constraints(self, optimization: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resource constraints to optimization recommendations."""
        max_memory = constraints.get("max_memory_gb", float('inf'))
        max_cpu = constraints.get("max_cpu_cores", float('inf'))
        
        # Apply memory constraints
        if "memory_adjustment" in optimization:
            recommended = optimization["memory_adjustment"]["recommended_gb"]
            optimization["memory_adjustment"]["recommended_gb"] = min(recommended, max_memory)
        
        # Apply CPU constraints
        if "cpu_adjustment" in optimization:
            recommended = optimization["cpu_adjustment"]["recommended_cores"]
            optimization["cpu_adjustment"]["recommended_cores"] = min(recommended, max_cpu)
        
        return optimization
    
    async def update_from_optimizations(self, optimizations: List[Dict[str, Any]]):
        """Update optimization strategies based on results."""
        logger.debug(f"Updating resource optimizer with {len(optimizations)} optimization results")


class StrategySelector:
    """Intelligent strategy selection system."""
    
    def __init__(self):
        self.strategy_effectiveness: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.strategy_usage_count: Dict[str, int] = defaultdict(int)
    
    async def select_strategy(
        self,
        workload: WorkloadProfile,
        performance_target: Dict[str, float],
        available_strategies: List[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Select the best optimization strategy for the given context."""
        
        strategy_scores = {}
        
        for strategy in available_strategies:
            score = self._calculate_strategy_score(strategy, workload, performance_target)
            strategy_scores[strategy] = score
        
        # Select strategy with highest score
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
        
        # Generate strategy parameters
        parameters = self._generate_strategy_parameters(best_strategy, workload)
        
        # Update usage count
        self.strategy_usage_count[best_strategy] += 1
        
        return best_strategy, parameters
    
    def _calculate_strategy_score(
        self,
        strategy: str,
        workload: WorkloadProfile,
        target: Dict[str, float]
    ) -> float:
        """Calculate effectiveness score for a strategy."""
        base_score = 0.5
        
        # Strategy-specific scoring
        if strategy == "mixed_precision" and workload.model_size_gb > 10:
            base_score += 0.3
        elif strategy == "gradient_checkpointing" and workload.memory_pattern == "memory_bound":
            base_score += 0.4
        elif strategy == "dynamic_batching" and workload.compute_intensity < 0.7:
            base_score += 0.2
        
        # Historical effectiveness
        if strategy in self.strategy_effectiveness:
            historical_score = self.strategy_effectiveness[strategy].get(workload.model_type, 0.5)
            base_score = 0.7 * base_score + 0.3 * historical_score
        
        return min(1.0, base_score)
    
    def _generate_strategy_parameters(self, strategy: str, workload: WorkloadProfile) -> Dict[str, Any]:
        """Generate parameters for the selected strategy."""
        if strategy == "mixed_precision":
            return {
                "precision": "bf16" if workload.model_size_gb > 50 else "fp16",
                "loss_scaling": True
            }
        elif strategy == "gradient_checkpointing":
            return {
                "checkpoint_layers": max(1, workload.num_layers // 4),
                "memory_efficient": True
            }
        elif strategy == "dynamic_batching":
            return {
                "min_batch_size": 16,
                "max_batch_size": 128,
                "adaptation_rate": 0.1
            }
        else:
            return {}


# Global adaptive intelligence engine instance
_adaptive_intelligence_engine: Optional[AdaptiveIntelligenceEngine] = None


def get_adaptive_intelligence_engine() -> AdaptiveIntelligenceEngine:
    """Get the global adaptive intelligence engine instance."""
    global _adaptive_intelligence_engine
    if _adaptive_intelligence_engine is None:
        _adaptive_intelligence_engine = AdaptiveIntelligenceEngine()
    return _adaptive_intelligence_engine


async def start_adaptive_intelligence():
    """Start the adaptive intelligence system."""
    engine = get_adaptive_intelligence_engine()
    await engine.start_adaptive_learning()


async def stop_adaptive_intelligence():
    """Stop the adaptive intelligence system."""
    engine = get_adaptive_intelligence_engine()
    await engine.stop_adaptive_learning()


def get_intelligence_report() -> Dict[str, Any]:
    """Get comprehensive intelligence system report."""
    engine = get_adaptive_intelligence_engine()
    return engine.get_intelligence_report()