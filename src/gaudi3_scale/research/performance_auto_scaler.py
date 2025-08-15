"""Performance-Aware Auto-Scaler for Gaudi 3 Clusters.

This module implements an intelligent auto-scaling system that dynamically
adjusts cluster resources based on real-time performance metrics, workload
patterns, and cost optimization objectives.
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from enum import Enum
from collections import defaultdict, deque
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import math

try:
    import aiohttp
    import aiofiles
    _async_available = True
except ImportError:
    aiohttp = None
    aiofiles = None
    _async_available = False

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ResourceTier(Enum):
    """Resource tier classifications."""
    MICRO = "micro"          # 1-2 HPUs
    SMALL = "small"          # 3-8 HPUs
    MEDIUM = "medium"        # 9-32 HPUs  
    LARGE = "large"          # 33-128 HPUs
    XLARGE = "xlarge"        # 129-512 HPUs
    XXLARGE = "xxlarge"      # 513+ HPUs


class WorkloadPattern(Enum):
    """Workload pattern types."""
    STEADY = "steady"
    BURSTY = "bursty"
    CYCLICAL = "cyclical"
    GROWING = "growing"
    DECLINING = "declining"
    UNKNOWN = "unknown"


@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions."""
    timestamp: float
    hpu_utilization: float
    memory_utilization: float
    throughput_samples_per_sec: float
    queue_length: int
    response_time_ms: float
    error_rate: float
    cost_per_sample: float
    energy_efficiency: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'hpu_utilization': self.hpu_utilization,
            'memory_utilization': self.memory_utilization,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'queue_length': self.queue_length,
            'response_time_ms': self.response_time_ms,
            'error_rate': self.error_rate,
            'cost_per_sample': self.cost_per_sample,
            'energy_efficiency': self.energy_efficiency
        }


@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    event_id: str
    timestamp: float
    direction: ScalingDirection
    current_nodes: int
    target_nodes: int
    trigger_reason: str
    estimated_cost_impact: float
    confidence_score: float
    metrics_snapshot: PerformanceMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'direction': self.direction.value,
            'current_nodes': self.current_nodes,
            'target_nodes': self.target_nodes,
            'trigger_reason': self.trigger_reason,
            'estimated_cost_impact': self.estimated_cost_impact,
            'confidence_score': self.confidence_score,
            'metrics_snapshot': self.metrics_snapshot.to_dict()
        }


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    name: str
    min_nodes: int = 1
    max_nodes: int = 100
    target_hpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 40.0
    scale_up_cooldown_seconds: float = 300.0
    scale_down_cooldown_seconds: float = 600.0
    aggressive_scaling: bool = False
    cost_optimization_enabled: bool = True
    max_cost_per_hour: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'min_nodes': self.min_nodes,
            'max_nodes': self.max_nodes,
            'target_hpu_utilization': self.target_hpu_utilization,
            'target_memory_utilization': self.target_memory_utilization,
            'scale_up_threshold': self.scale_up_threshold,
            'scale_down_threshold': self.scale_down_threshold,
            'scale_up_cooldown_seconds': self.scale_up_cooldown_seconds,
            'scale_down_cooldown_seconds': self.scale_down_cooldown_seconds,
            'aggressive_scaling': self.aggressive_scaling,
            'cost_optimization_enabled': self.cost_optimization_enabled,
            'max_cost_per_hour': self.max_cost_per_hour
        }


class WorkloadPredictor:
    """Predicts workload patterns for proactive scaling."""
    
    def __init__(self, history_window: int = 100):
        """Initialize workload predictor.
        
        Args:
            history_window: Number of historical data points to consider
        """
        self.history_window = history_window
        self.metrics_history: deque = deque(maxlen=history_window)
        self.pattern_cache: Dict[str, Any] = {}
        self.prediction_accuracy: float = 0.0
        
    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add new metrics data point.
        
        Args:
            metrics: Performance metrics to add
        """
        self.metrics_history.append(metrics)
        
        # Clear pattern cache when new data arrives
        self.pattern_cache.clear()
    
    def detect_pattern(self) -> WorkloadPattern:
        """Detect current workload pattern.
        
        Returns:
            Detected workload pattern
        """
        if len(self.metrics_history) < 10:
            return WorkloadPattern.UNKNOWN
        
        # Extract utilization time series
        utilizations = [m.hpu_utilization for m in self.metrics_history]
        
        # Calculate statistical measures
        mean_util = np.mean(utilizations)
        std_util = np.std(utilizations)
        trend = self._calculate_trend(utilizations)
        
        # Pattern detection logic
        if std_util / mean_util < 0.1:  # Low variance
            return WorkloadPattern.STEADY
        elif std_util / mean_util > 0.5:  # High variance
            if self._is_cyclical(utilizations):
                return WorkloadPattern.CYCLICAL
            else:
                return WorkloadPattern.BURSTY
        elif trend > 0.1:  # Positive trend
            return WorkloadPattern.GROWING
        elif trend < -0.1:  # Negative trend
            return WorkloadPattern.DECLINING
        else:
            return WorkloadPattern.STEADY
    
    def predict_utilization(self, horizon_minutes: float = 30) -> Tuple[float, float]:
        """Predict future utilization.
        
        Args:
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            Tuple of (predicted_utilization, confidence)
        """
        if len(self.metrics_history) < 5:
            # Not enough data, return current utilization
            current = self.metrics_history[-1].hpu_utilization if self.metrics_history else 50.0
            return current, 0.1
        
        utilizations = [m.hpu_utilization for m in self.metrics_history]
        pattern = self.detect_pattern()
        
        if pattern == WorkloadPattern.STEADY:
            predicted = np.mean(utilizations[-10:])
            confidence = 0.8
        elif pattern == WorkloadPattern.GROWING:
            trend = self._calculate_trend(utilizations)
            predicted = utilizations[-1] + trend * horizon_minutes
            confidence = 0.6
        elif pattern == WorkloadPattern.DECLINING:
            trend = self._calculate_trend(utilizations)
            predicted = utilizations[-1] + trend * horizon_minutes
            confidence = 0.6
        elif pattern == WorkloadPattern.CYCLICAL:
            predicted = self._predict_cyclical(utilizations, horizon_minutes)
            confidence = 0.5
        else:  # BURSTY or UNKNOWN
            predicted = np.mean(utilizations[-5:])
            confidence = 0.3
        
        # Clamp prediction to reasonable bounds
        predicted = max(0, min(100, predicted))
        
        return predicted, confidence
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values.
        
        Args:
            values: List of values to analyze
            
        Returns:
            Trend value (positive for increasing, negative for decreasing)
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope of linear fit
    
    def _is_cyclical(self, values: List[float]) -> bool:
        """Check if values show cyclical pattern.
        
        Args:
            values: List of values to analyze
            
        Returns:
            True if cyclical pattern detected
        """
        if len(values) < 20:
            return False
        
        # Simple autocorrelation check
        try:
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for peaks indicating periodicity
            for lag in range(5, min(len(autocorr) // 2, 50)):
                if autocorr[lag] > 0.7 * autocorr[0]:
                    return True
            
            return False
        except Exception:
            return False
    
    def _predict_cyclical(self, values: List[float], horizon_minutes: float) -> float:
        """Predict value for cyclical pattern.
        
        Args:
            values: Historical values
            horizon_minutes: Prediction horizon
            
        Returns:
            Predicted value
        """
        # Simple cyclical prediction - look back at similar time
        period_length = 24  # Assume 24-point cycle for simplicity
        if len(values) >= period_length:
            cycle_position = len(values) % period_length
            target_position = (cycle_position + int(horizon_minutes)) % period_length
            return values[-period_length + target_position]
        else:
            return np.mean(values[-5:])


class CostOptimizer:
    """Optimizes scaling decisions for cost efficiency."""
    
    def __init__(self):
        """Initialize cost optimizer."""
        # Cost per hour by resource tier (mock values)
        self.tier_costs = {
            ResourceTier.MICRO: 2.5,
            ResourceTier.SMALL: 18.0,
            ResourceTier.MEDIUM: 65.0,
            ResourceTier.LARGE: 250.0,
            ResourceTier.XLARGE: 950.0,
            ResourceTier.XXLARGE: 3500.0
        }
        
        self.cost_history: deque = deque(maxlen=100)
        
    def calculate_cost_impact(
        self,
        current_nodes: int,
        target_nodes: int,
        tier: ResourceTier = ResourceTier.SMALL,
        duration_hours: float = 1.0
    ) -> float:
        """Calculate cost impact of scaling decision.
        
        Args:
            current_nodes: Current number of nodes
            target_nodes: Target number of nodes
            tier: Resource tier
            duration_hours: Expected duration in hours
            
        Returns:
            Cost impact in dollars (positive for cost increase)
        """
        cost_per_node_hour = self.tier_costs.get(tier, 18.0)
        
        current_cost = current_nodes * cost_per_node_hour * duration_hours
        target_cost = target_nodes * cost_per_node_hour * duration_hours
        
        return target_cost - current_cost
    
    def optimize_node_count(
        self,
        required_capacity: float,
        tier: ResourceTier = ResourceTier.SMALL,
        max_cost_per_hour: Optional[float] = None
    ) -> int:
        """Optimize node count for required capacity and cost constraints.
        
        Args:
            required_capacity: Required processing capacity (arbitrary units)
            tier: Resource tier
            max_cost_per_hour: Maximum cost per hour constraint
            
        Returns:
            Optimal number of nodes
        """
        # Capacity per node by tier (mock values)
        tier_capacity = {
            ResourceTier.MICRO: 10,
            ResourceTier.SMALL: 80,
            ResourceTier.MEDIUM: 320,
            ResourceTier.LARGE: 1280,
            ResourceTier.XLARGE: 5120,
            ResourceTier.XXLARGE: 20480
        }
        
        capacity_per_node = tier_capacity.get(tier, 80)
        cost_per_node_hour = self.tier_costs.get(tier, 18.0)
        
        # Calculate minimum nodes needed
        min_nodes = max(1, math.ceil(required_capacity / capacity_per_node))
        
        # Apply cost constraint if specified
        if max_cost_per_hour:
            max_nodes_by_cost = int(max_cost_per_hour / cost_per_node_hour)
            min_nodes = min(min_nodes, max_nodes_by_cost)
        
        return min_nodes
    
    def get_cost_efficiency_score(
        self,
        nodes: int,
        utilization: float,
        tier: ResourceTier = ResourceTier.SMALL
    ) -> float:
        """Calculate cost efficiency score.
        
        Args:
            nodes: Number of nodes
            utilization: Current utilization percentage
            tier: Resource tier
            
        Returns:
            Cost efficiency score (higher is better)
        """
        cost_per_hour = nodes * self.tier_costs.get(tier, 18.0)
        
        # Efficiency is utilization / cost, normalized
        if cost_per_hour == 0:
            return 0.0
        
        raw_efficiency = utilization / cost_per_hour
        
        # Normalize to 0-1 scale (assuming max efficiency around 100/18 ≈ 5.5)
        normalized_efficiency = min(1.0, raw_efficiency / 5.5)
        
        return normalized_efficiency


class PerformanceAutoScaler:
    """Advanced auto-scaler with performance optimization and cost awareness."""
    
    def __init__(
        self,
        scaling_policy: ScalingPolicy,
        cluster_name: str = "gaudi3-cluster",
        monitoring_interval: float = 30.0,
        enable_predictive_scaling: bool = True,
        enable_cost_optimization: bool = True
    ):
        """Initialize performance auto-scaler.
        
        Args:
            scaling_policy: Scaling policy configuration
            cluster_name: Name of the cluster
            monitoring_interval: Monitoring interval in seconds
            enable_predictive_scaling: Enable predictive scaling
            enable_cost_optimization: Enable cost optimization
        """
        self.scaling_policy = scaling_policy
        self.cluster_name = cluster_name
        self.monitoring_interval = monitoring_interval
        self.enable_predictive_scaling = enable_predictive_scaling
        self.enable_cost_optimization = enable_cost_optimization
        
        # Components
        self.workload_predictor = WorkloadPredictor()
        self.cost_optimizer = CostOptimizer()
        
        # State tracking
        self.current_nodes = scaling_policy.min_nodes
        self.target_nodes = scaling_policy.min_nodes
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
        
        # History tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_events: List[ScalingEvent] = []
        self.performance_cache: Dict[str, Any] = {}
        
        # Threading
        self.monitor_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.lock = threading.RLock()
        
        # Async components
        self.session: Optional[Any] = None
        
        logger.info(f"Initialized PerformanceAutoScaler for cluster '{cluster_name}'")
    
    async def start_async(self) -> None:
        """Start async components."""
        if _async_available:
            self.session = aiohttp.ClientSession()
    
    async def stop_async(self) -> None:
        """Stop async components."""
        if self.session:
            await self.session.close()
    
    def start_monitoring(self) -> None:
        """Start the monitoring and scaling loop."""
        if self.running:
            logger.warning("Auto-scaler is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started performance auto-scaler monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring and scaling loop."""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.executor.shutdown(wait=True)
        logger.info("Stopped performance auto-scaler monitoring")
    
    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add new performance metrics.
        
        Args:
            metrics: Performance metrics to add
        """
        with self.lock:
            self.metrics_history.append(metrics)
            self.workload_predictor.add_metrics(metrics)
            
            # Clear performance cache when new metrics arrive
            self.performance_cache.clear()
    
    def _monitoring_loop(self) -> None:
        """Main monitoring and scaling loop."""
        while self.running:
            try:
                # Collect current metrics (mock for now)
                current_metrics = self._collect_current_metrics()
                self.add_metrics(current_metrics)
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(current_metrics)
                
                if scaling_decision.direction != ScalingDirection.MAINTAIN:
                    # Execute scaling action
                    self._execute_scaling(scaling_decision)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Brief pause on error
    
    def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics.
        
        Returns:
            Current performance metrics
        """
        # Mock metrics collection - in real implementation, this would
        # collect from monitoring systems, HPU drivers, etc.
        
        base_utilization = 60.0 + 20.0 * np.sin(time.time() / 300.0)  # Cyclical pattern
        noise = np.random.normal(0, 5)
        
        return PerformanceMetrics(
            timestamp=time.time(),
            hpu_utilization=max(0, min(100, base_utilization + noise)),
            memory_utilization=max(0, min(100, base_utilization * 0.8 + noise)),
            throughput_samples_per_sec=800 + 200 * np.random.normal(),
            queue_length=max(0, int(10 + 5 * np.random.normal())),
            response_time_ms=max(1, 50 + 20 * np.random.normal()),
            error_rate=max(0, min(1, 0.01 + 0.005 * np.random.normal())),
            cost_per_sample=0.001 + 0.0002 * np.random.normal(),
            energy_efficiency=max(0, min(1, 0.8 + 0.1 * np.random.normal()))
        )
    
    def _make_scaling_decision(self, current_metrics: PerformanceMetrics) -> ScalingEvent:
        """Make scaling decision based on current metrics and policy.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Scaling event with decision
        """
        current_time = time.time()
        direction = ScalingDirection.MAINTAIN
        target_nodes = self.current_nodes
        trigger_reason = "No scaling needed"
        confidence_score = 1.0
        
        # Check cooldown periods
        scale_up_ready = (current_time - self.last_scale_up_time) > self.scaling_policy.scale_up_cooldown_seconds
        scale_down_ready = (current_time - self.last_scale_down_time) > self.scaling_policy.scale_down_cooldown_seconds
        
        # Basic reactive scaling
        if current_metrics.hpu_utilization > self.scaling_policy.scale_up_threshold and scale_up_ready:
            if self.current_nodes < self.scaling_policy.max_nodes:
                direction = ScalingDirection.SCALE_UP
                target_nodes = min(self.scaling_policy.max_nodes, self.current_nodes + 1)
                trigger_reason = f"HPU utilization {current_metrics.hpu_utilization:.1f}% > threshold {self.scaling_policy.scale_up_threshold}%"
                confidence_score = 0.8
        
        elif current_metrics.hpu_utilization < self.scaling_policy.scale_down_threshold and scale_down_ready:
            if self.current_nodes > self.scaling_policy.min_nodes:
                direction = ScalingDirection.SCALE_DOWN
                target_nodes = max(self.scaling_policy.min_nodes, self.current_nodes - 1)
                trigger_reason = f"HPU utilization {current_metrics.hpu_utilization:.1f}% < threshold {self.scaling_policy.scale_down_threshold}%"
                confidence_score = 0.7
        
        # Predictive scaling enhancement
        if self.enable_predictive_scaling and direction == ScalingDirection.MAINTAIN:
            predicted_util, prediction_confidence = self.workload_predictor.predict_utilization(
                horizon_minutes=self.monitoring_interval / 60.0 * 3  # 3 intervals ahead
            )
            
            if prediction_confidence > 0.6:
                if predicted_util > self.scaling_policy.scale_up_threshold and scale_up_ready:
                    if self.current_nodes < self.scaling_policy.max_nodes:
                        direction = ScalingDirection.SCALE_UP
                        target_nodes = min(self.scaling_policy.max_nodes, self.current_nodes + 1)
                        trigger_reason = f"Predicted HPU utilization {predicted_util:.1f}% > threshold"
                        confidence_score = prediction_confidence * 0.9
                
                elif predicted_util < self.scaling_policy.scale_down_threshold and scale_down_ready:
                    if self.current_nodes > self.scaling_policy.min_nodes:
                        direction = ScalingDirection.SCALE_DOWN
                        target_nodes = max(self.scaling_policy.min_nodes, self.current_nodes - 1)
                        trigger_reason = f"Predicted HPU utilization {predicted_util:.1f}% < threshold"
                        confidence_score = prediction_confidence * 0.8
        
        # Cost optimization check
        if self.enable_cost_optimization and direction != ScalingDirection.MAINTAIN:
            cost_impact = self.cost_optimizer.calculate_cost_impact(
                current_nodes=self.current_nodes,
                target_nodes=target_nodes,
                duration_hours=1.0
            )
            
            # Check cost constraints
            if self.scaling_policy.max_cost_per_hour:
                estimated_hourly_cost = target_nodes * 18.0  # Default cost per node
                if estimated_hourly_cost > self.scaling_policy.max_cost_per_hour:
                    direction = ScalingDirection.MAINTAIN
                    target_nodes = self.current_nodes
                    trigger_reason = "Scaling blocked by cost constraint"
                    confidence_score = 1.0
        else:
            cost_impact = 0.0
        
        # Create scaling event
        event_id = f"scale_{direction.value}_{int(current_time)}"
        
        return ScalingEvent(
            event_id=event_id,
            timestamp=current_time,
            direction=direction,
            current_nodes=self.current_nodes,
            target_nodes=target_nodes,
            trigger_reason=trigger_reason,
            estimated_cost_impact=cost_impact,
            confidence_score=confidence_score,
            metrics_snapshot=current_metrics
        )
    
    def _execute_scaling(self, scaling_event: ScalingEvent) -> None:
        """Execute scaling action.
        
        Args:
            scaling_event: Scaling event to execute
        """
        with self.lock:
            logger.info(f"Executing scaling: {scaling_event.direction.value} "
                       f"from {scaling_event.current_nodes} to {scaling_event.target_nodes} nodes. "
                       f"Reason: {scaling_event.trigger_reason}")
            
            # Update state
            self.current_nodes = scaling_event.target_nodes
            self.target_nodes = scaling_event.target_nodes
            
            # Update cooldown timers
            if scaling_event.direction == ScalingDirection.SCALE_UP:
                self.last_scale_up_time = scaling_event.timestamp
            elif scaling_event.direction == ScalingDirection.SCALE_DOWN:
                self.last_scale_down_time = scaling_event.timestamp
            
            # Store scaling event
            self.scaling_events.append(scaling_event)
            
            # Execute actual scaling (mock for now)
            self._submit_scaling_task(scaling_event)
    
    def _submit_scaling_task(self, scaling_event: ScalingEvent) -> None:
        """Submit scaling task to executor.
        
        Args:
            scaling_event: Scaling event to execute
        """
        def scaling_task():
            """Actual scaling task execution."""
            try:
                if scaling_event.direction == ScalingDirection.SCALE_UP:
                    # Mock scale up - add nodes
                    logger.info(f"Adding nodes to reach {scaling_event.target_nodes} total nodes")
                    # In real implementation: call cloud provider APIs, Kubernetes, etc.
                    
                elif scaling_event.direction == ScalingDirection.SCALE_DOWN:
                    # Mock scale down - remove nodes
                    logger.info(f"Removing nodes to reach {scaling_event.target_nodes} total nodes")
                    # In real implementation: drain and terminate nodes
                
                # Simulate scaling time
                time.sleep(2.0)
                
                logger.info(f"Scaling completed: {scaling_event.event_id}")
                
            except Exception as e:
                logger.error(f"Scaling failed for event {scaling_event.event_id}: {e}")
        
        self.executor.submit(scaling_task)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status.
        
        Returns:
            Dictionary containing cluster status
        """
        with self.lock:
            current_metrics = self.metrics_history[-1] if self.metrics_history else None
            predicted_util, prediction_confidence = self.workload_predictor.predict_utilization()
            workload_pattern = self.workload_predictor.detect_pattern()
            
            status = {
                "cluster_name": self.cluster_name,
                "timestamp": time.time(),
                "current_nodes": self.current_nodes,
                "target_nodes": self.target_nodes,
                "scaling_policy": self.scaling_policy.to_dict(),
                "current_metrics": current_metrics.to_dict() if current_metrics else None,
                "workload_pattern": workload_pattern.value,
                "predicted_utilization": predicted_util,
                "prediction_confidence": prediction_confidence,
                "recent_scaling_events": [event.to_dict() for event in self.scaling_events[-5:]],
                "cost_efficiency": self._calculate_current_cost_efficiency(),
                "monitoring_enabled": self.running
            }
            
            return status
    
    def _calculate_current_cost_efficiency(self) -> float:
        """Calculate current cost efficiency score."""
        if not self.metrics_history:
            return 0.0
        
        latest_metrics = self.metrics_history[-1]
        return self.cost_optimizer.get_cost_efficiency_score(
            nodes=self.current_nodes,
            utilization=latest_metrics.hpu_utilization
        )
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations based on current state.
        
        Returns:
            Dictionary containing scaling recommendations
        """
        if not self.metrics_history:
            return {"recommendations": [], "confidence": 0.0}
        
        current_metrics = self.metrics_history[-1]
        predicted_util, prediction_confidence = self.workload_predictor.predict_utilization(60)  # 1 hour ahead
        workload_pattern = self.workload_predictor.detect_pattern()
        
        recommendations = []
        
        # Cost optimization recommendations
        if self.enable_cost_optimization:
            cost_efficiency = self._calculate_current_cost_efficiency()
            if cost_efficiency < 0.5:
                recommendations.append({
                    "type": "cost_optimization",
                    "description": f"Current cost efficiency is low ({cost_efficiency:.2f}). Consider scaling down during low usage periods.",
                    "priority": "medium"
                })
        
        # Workload pattern recommendations
        if workload_pattern == WorkloadPattern.BURSTY:
            recommendations.append({
                "type": "policy_adjustment",
                "description": "Bursty workload detected. Consider enabling aggressive scaling for faster response to spikes.",
                "priority": "low"
            })
        elif workload_pattern == WorkloadPattern.CYCLICAL:
            recommendations.append({
                "type": "predictive_scaling",
                "description": "Cyclical pattern detected. Predictive scaling can help anticipate capacity needs.",
                "priority": "medium"
            })
        
        # Resource utilization recommendations
        if current_metrics.hpu_utilization > 90:
            recommendations.append({
                "type": "capacity_planning",
                "description": f"High HPU utilization ({current_metrics.hpu_utilization:.1f}%). Consider increasing max_nodes limit.",
                "priority": "high"
            })
        elif current_metrics.hpu_utilization < 20:
            recommendations.append({
                "type": "resource_optimization",
                "description": f"Low HPU utilization ({current_metrics.hpu_utilization:.1f}%). Consider scaling down or adjusting thresholds.",
                "priority": "medium"
            })
        
        return {
            "recommendations": recommendations,
            "confidence": prediction_confidence,
            "workload_pattern": workload_pattern.value,
            "predicted_utilization": predicted_util
        }
    
    def update_scaling_policy(self, new_policy: ScalingPolicy) -> None:
        """Update scaling policy.
        
        Args:
            new_policy: New scaling policy to apply
        """
        with self.lock:
            old_policy_name = self.scaling_policy.name
            self.scaling_policy = new_policy
            
            # Validate current node count against new policy
            if self.current_nodes < new_policy.min_nodes:
                self.target_nodes = new_policy.min_nodes
                logger.info(f"Scaling up to meet new minimum nodes: {new_policy.min_nodes}")
            elif self.current_nodes > new_policy.max_nodes:
                self.target_nodes = new_policy.max_nodes
                logger.info(f"Scaling down to meet new maximum nodes: {new_policy.max_nodes}")
            
            logger.info(f"Updated scaling policy from '{old_policy_name}' to '{new_policy.name}'")
    
    def save_state(self, filepath: str) -> None:
        """Save auto-scaler state to file.
        
        Args:
            filepath: File path to save state
        """
        with self.lock:
            state = {
                "cluster_name": self.cluster_name,
                "timestamp": time.time(),
                "current_nodes": self.current_nodes,
                "target_nodes": self.target_nodes,
                "scaling_policy": self.scaling_policy.to_dict(),
                "scaling_events": [event.to_dict() for event in self.scaling_events],
                "metrics_history": [metrics.to_dict() for metrics in list(self.metrics_history)[-50:]],  # Last 50 metrics
                "last_scale_up_time": self.last_scale_up_time,
                "last_scale_down_time": self.last_scale_down_time
            }
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(state, f, indent=2)
                logger.info(f"Auto-scaler state saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save auto-scaler state: {e}")


# Factory functions

def create_default_scaling_policy(
    cluster_size: str = "medium",
    cost_conscious: bool = True
) -> ScalingPolicy:
    """Create a default scaling policy for common scenarios.
    
    Args:
        cluster_size: Target cluster size (small, medium, large)
        cost_conscious: Whether to prioritize cost optimization
        
    Returns:
        Configured scaling policy
    """
    if cluster_size == "small":
        min_nodes, max_nodes = 1, 8
        target_util = 75.0 if cost_conscious else 70.0
    elif cluster_size == "large":
        min_nodes, max_nodes = 4, 64
        target_util = 70.0 if cost_conscious else 65.0
    else:  # medium
        min_nodes, max_nodes = 2, 32
        target_util = 70.0 if cost_conscious else 65.0
    
    return ScalingPolicy(
        name=f"{cluster_size}_cluster_policy",
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        target_hpu_utilization=target_util,
        scale_up_threshold=target_util + 15.0,
        scale_down_threshold=target_util - 25.0,
        scale_up_cooldown_seconds=300.0 if cost_conscious else 180.0,
        scale_down_cooldown_seconds=600.0 if cost_conscious else 300.0,
        cost_optimization_enabled=cost_conscious,
        max_cost_per_hour=1000.0 if cost_conscious else None
    )


def create_performance_auto_scaler(
    cluster_name: str = "gaudi3-cluster",
    cluster_size: str = "medium",
    cost_conscious: bool = True
) -> PerformanceAutoScaler:
    """Create a configured performance auto-scaler.
    
    Args:
        cluster_name: Name of the cluster
        cluster_size: Target cluster size
        cost_conscious: Whether to prioritize cost optimization
        
    Returns:
        Configured performance auto-scaler
    """
    policy = create_default_scaling_policy(cluster_size, cost_conscious)
    
    return PerformanceAutoScaler(
        scaling_policy=policy,
        cluster_name=cluster_name,
        monitoring_interval=30.0,
        enable_predictive_scaling=True,
        enable_cost_optimization=cost_conscious
    )


# Export main classes and functions
__all__ = [
    'PerformanceAutoScaler',
    'ScalingPolicy',
    'ScalingDirection',
    'WorkloadPattern',
    'PerformanceMetrics',
    'ScalingEvent',
    'WorkloadPredictor',
    'CostOptimizer',
    'create_default_scaling_policy',
    'create_performance_auto_scaler'
]