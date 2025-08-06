"""Advanced auto-scaling and load balancing system for cluster management."""

import asyncio
import time
import logging
import math
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import weakref

from ..monitoring.performance import get_performance_monitor, MetricPoint
from ..cache.distributed_cache import get_distributed_cache

logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    CUSTOM = "custom"

class AutoScalingPolicy(Enum):
    """Auto-scaling policies."""
    TARGET_TRACKING = "target_tracking"
    STEP_SCALING = "step_scaling"
    PREDICTIVE = "predictive"
    REACTIVE = "reactive"

@dataclass
class Node:
    """Cluster node representation."""
    id: str
    address: str
    port: int
    capacity: float = 1.0
    weight: float = 1.0
    current_load: float = 0.0
    connections: int = 0
    response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    healthy: bool = True
    last_health_check: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_score(self, strategy: LoadBalancingStrategy) -> float:
        """Calculate node score for load balancing."""
        if not self.healthy:
            return float('inf')
        
        if strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self.connections / self.capacity
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self.response_time
        elif strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return (self.cpu_usage * 0.6 + self.memory_usage * 0.4) / self.capacity
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self.current_load / (self.weight * self.capacity)
        else:
            return self.current_load / self.capacity

@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    response_time: float
    queue_length: int
    error_rate: float
    active_connections: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""
    # Target metrics
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_response_time: float = 100.0  # milliseconds
    target_error_rate: float = 1.0  # percentage
    
    # Scaling parameters
    min_nodes: int = 2
    max_nodes: int = 20
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: float = 300.0  # seconds
    scale_down_cooldown: float = 600.0  # seconds
    
    # Evaluation parameters
    evaluation_period: float = 60.0  # seconds
    datapoints_to_alarm: int = 2
    metric_window_size: int = 10
    
    # Load balancing
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    connection_drain_timeout: float = 30.0
    
    # Predictive scaling
    enable_predictive_scaling: bool = False
    prediction_window: int = 300  # seconds
    prediction_accuracy_threshold: float = 0.8

@dataclass
class ScalingAction:
    """Scaling action to be executed."""
    direction: ScalingDirection
    target_count: int
    reason: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    executed: bool = False

class MetricsCollector:
    """Collects and aggregates scaling metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._metrics_history = deque(maxlen=window_size)
        self._performance_monitor = get_performance_monitor()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def collect_current_metrics(self, nodes: List[Node]) -> ScalingMetrics:
        """Collect current cluster metrics."""
        now = time.time()
        
        # Aggregate node metrics
        total_cpu = sum(node.cpu_usage for node in nodes if node.healthy)
        total_memory = sum(node.memory_usage for node in nodes if node.healthy)
        total_connections = sum(node.connections for node in nodes if node.healthy)
        avg_response_time = statistics.mean([node.response_time for node in nodes if node.healthy]) if nodes else 0.0
        
        healthy_nodes = len([node for node in nodes if node.healthy])
        
        # Get system metrics from performance monitor
        system_metrics = self._performance_monitor.get_all_metrics().get('system_metrics', {})
        
        metrics = ScalingMetrics(
            timestamp=now,
            cpu_utilization=total_cpu / healthy_nodes if healthy_nodes > 0 else 0.0,
            memory_utilization=total_memory / healthy_nodes if healthy_nodes > 0 else 0.0,
            request_rate=system_metrics.get('network_packets_recv', 0) / 60.0,  # requests per second estimate
            response_time=avg_response_time * 1000,  # convert to milliseconds
            queue_length=0,  # This would come from application metrics
            error_rate=0.0,  # This would come from application metrics
            active_connections=total_connections,
            custom_metrics={}
        )
        
        self._metrics_history.append(metrics)
        return metrics
    
    def get_metrics_trend(self, metric_name: str, window: int = 5) -> float:
        """Get trend for specific metric."""
        if len(self._metrics_history) < window:
            return 0.0
        
        recent_metrics = list(self._metrics_history)[-window:]
        values = [getattr(metric, metric_name) for metric in recent_metrics]
        
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x_values = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except ZeroDivisionError:
            return 0.0
    
    def get_average_metric(self, metric_name: str, window: int = 5) -> float:
        """Get average value for specific metric."""
        if len(self._metrics_history) < window:
            return 0.0
        
        recent_metrics = list(self._metrics_history)[-window:]
        values = [getattr(metric, metric_name) for metric in recent_metrics]
        return statistics.mean(values) if values else 0.0

class PredictiveScaler:
    """Predictive scaling using historical patterns."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self._historical_patterns = defaultdict(list)
        self._prediction_accuracy = deque(maxlen=100)
        self.logger = logger.getChild(self.__class__.__name__)
    
    def learn_pattern(self, metrics: ScalingMetrics) -> None:
        """Learn from historical metrics patterns."""
        # Simple time-based pattern learning
        hour_of_day = int(time.gmtime(metrics.timestamp).tm_hour)
        day_of_week = int(time.gmtime(metrics.timestamp).tm_wday)
        
        pattern_key = f"{day_of_week}_{hour_of_day}"
        self._historical_patterns[pattern_key].append({
            'cpu': metrics.cpu_utilization,
            'memory': metrics.memory_utilization,
            'request_rate': metrics.request_rate,
            'timestamp': metrics.timestamp
        })
        
        # Keep only recent patterns
        cutoff_time = metrics.timestamp - (7 * 24 * 3600)  # Last 7 days
        for key in self._historical_patterns:
            self._historical_patterns[key] = [
                p for p in self._historical_patterns[key] 
                if p['timestamp'] > cutoff_time
            ]
    
    def predict_demand(self, current_time: float) -> Dict[str, float]:
        """Predict future resource demand."""
        future_time = current_time + self.config.prediction_window
        future_hour = int(time.gmtime(future_time).tm_hour)
        future_day = int(time.gmtime(future_time).tm_wday)
        
        pattern_key = f"{future_day}_{future_hour}"
        
        if pattern_key not in self._historical_patterns or not self._historical_patterns[pattern_key]:
            return {'cpu': 50.0, 'memory': 60.0, 'request_rate': 100.0}
        
        patterns = self._historical_patterns[pattern_key]
        
        return {
            'cpu': statistics.mean([p['cpu'] for p in patterns]),
            'memory': statistics.mean([p['memory'] for p in patterns]),
            'request_rate': statistics.mean([p['request_rate'] for p in patterns])
        }
    
    def validate_prediction(self, predicted: Dict[str, float], actual: ScalingMetrics) -> float:
        """Validate prediction accuracy."""
        cpu_error = abs(predicted['cpu'] - actual.cpu_utilization) / max(actual.cpu_utilization, 1.0)
        memory_error = abs(predicted['memory'] - actual.memory_utilization) / max(actual.memory_utilization, 1.0)
        
        accuracy = 1.0 - (cpu_error + memory_error) / 2.0
        self._prediction_accuracy.append(max(0.0, accuracy))
        
        return accuracy
    
    def get_prediction_accuracy(self) -> float:
        """Get current prediction accuracy."""
        if not self._prediction_accuracy:
            return 0.0
        return statistics.mean(self._prediction_accuracy)

class AutoScaler:
    """Main auto-scaling controller."""
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self.nodes: List[Node] = []
        self._metrics_collector = MetricsCollector(self.config.metric_window_size)
        self._predictive_scaler = PredictiveScaler(self.config) if self.config.enable_predictive_scaling else None
        
        # Scaling state
        self._last_scale_up = 0.0
        self._last_scale_down = 0.0
        self._scaling_actions_history = deque(maxlen=100)
        
        # Callbacks
        self._scale_up_callback: Optional[Callable] = None
        self._scale_down_callback: Optional[Callable] = None
        self._health_check_callback: Optional[Callable] = None
        
        # Cache
        self._cache = get_distributed_cache()
        
        self.logger = logger.getChild(self.__class__.__name__)
    
    def register_callbacks(self, 
                          scale_up: Optional[Callable] = None,
                          scale_down: Optional[Callable] = None,
                          health_check: Optional[Callable] = None) -> None:
        """Register scaling callbacks."""
        self._scale_up_callback = scale_up
        self._scale_down_callback = scale_down
        self._health_check_callback = health_check
    
    def add_node(self, node: Node) -> None:
        """Add node to cluster."""
        self.nodes.append(node)
        self.logger.info(f"Added node {node.id} to cluster")
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from cluster."""
        for i, node in enumerate(self.nodes):
            if node.id == node_id:
                removed_node = self.nodes.pop(i)
                self.logger.info(f"Removed node {node_id} from cluster")
                return True
        return False
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        return next((node for node in self.nodes if node.id == node_id), None)
    
    def update_node_metrics(self, node_id: str, **metrics) -> None:
        """Update node metrics."""
        node = self.get_node(node_id)
        if node:
            for key, value in metrics.items():
                if hasattr(node, key):
                    setattr(node, key, value)
            node.last_health_check = time.time()
    
    async def evaluate_scaling(self) -> Optional[ScalingAction]:
        """Evaluate if scaling action is needed."""
        current_metrics = self._metrics_collector.collect_current_metrics(self.nodes)
        
        # Learn patterns if predictive scaling is enabled
        if self._predictive_scaler:
            self._predictive_scaler.learn_pattern(current_metrics)
        
        # Check cooldown periods
        now = time.time()
        if (now - self._last_scale_up < self.config.scale_up_cooldown and 
            now - self._last_scale_down < self.config.scale_down_cooldown):
            return None
        
        # Determine scaling direction
        scaling_action = None
        
        if self.config.enable_predictive_scaling and self._predictive_scaler:
            scaling_action = await self._evaluate_predictive_scaling(current_metrics)
        
        if not scaling_action:
            scaling_action = await self._evaluate_reactive_scaling(current_metrics)
        
        if scaling_action:
            self._scaling_actions_history.append(scaling_action)
        
        return scaling_action
    
    async def _evaluate_reactive_scaling(self, metrics: ScalingMetrics) -> Optional[ScalingAction]:
        """Evaluate reactive scaling based on current metrics."""
        healthy_nodes = len([node for node in self.nodes if node.healthy])
        
        # Check if we need to scale up
        scale_up_reasons = []
        if metrics.cpu_utilization > self.config.target_cpu_utilization:
            scale_up_reasons.append(f"CPU: {metrics.cpu_utilization:.1f}% > {self.config.target_cpu_utilization}%")
        
        if metrics.memory_utilization > self.config.target_memory_utilization:
            scale_up_reasons.append(f"Memory: {metrics.memory_utilization:.1f}% > {self.config.target_memory_utilization}%")
        
        if metrics.response_time > self.config.target_response_time:
            scale_up_reasons.append(f"Response time: {metrics.response_time:.1f}ms > {self.config.target_response_time}ms")
        
        if metrics.error_rate > self.config.target_error_rate:
            scale_up_reasons.append(f"Error rate: {metrics.error_rate:.1f}% > {self.config.target_error_rate}%")
        
        # Scale up if conditions are met
        if (len(scale_up_reasons) >= self.config.datapoints_to_alarm and 
            healthy_nodes < self.config.max_nodes and
            time.time() - self._last_scale_up >= self.config.scale_up_cooldown):
            
            target_count = min(healthy_nodes + 1, self.config.max_nodes)
            return ScalingAction(
                direction=ScalingDirection.UP,
                target_count=target_count,
                reason="; ".join(scale_up_reasons),
                confidence=0.8
            )
        
        # Check if we need to scale down
        scale_down_reasons = []
        if (metrics.cpu_utilization < self.config.scale_down_threshold and 
            metrics.memory_utilization < self.config.scale_down_threshold and
            metrics.response_time < self.config.target_response_time * 0.5):
            scale_down_reasons.append(f"Low utilization: CPU {metrics.cpu_utilization:.1f}%, Memory {metrics.memory_utilization:.1f}%")
        
        # Scale down if conditions are met
        if (scale_down_reasons and 
            healthy_nodes > self.config.min_nodes and
            time.time() - self._last_scale_down >= self.config.scale_down_cooldown):
            
            target_count = max(healthy_nodes - 1, self.config.min_nodes)
            return ScalingAction(
                direction=ScalingDirection.DOWN,
                target_count=target_count,
                reason="; ".join(scale_down_reasons),
                confidence=0.7
            )
        
        return None
    
    async def _evaluate_predictive_scaling(self, current_metrics: ScalingMetrics) -> Optional[ScalingAction]:
        """Evaluate predictive scaling based on forecasted demand."""
        if not self._predictive_scaler:
            return None
        
        # Get prediction accuracy
        accuracy = self._predictive_scaler.get_prediction_accuracy()
        if accuracy < self.config.prediction_accuracy_threshold:
            return None  # Fall back to reactive scaling
        
        # Get prediction
        prediction = self._predictive_scaler.predict_demand(time.time())
        
        healthy_nodes = len([node for node in self.nodes if node.healthy])
        
        # Calculate required capacity based on prediction
        predicted_cpu_load = prediction['cpu']
        predicted_memory_load = prediction['memory']
        
        # Simple capacity planning: if predicted load exceeds target, scale proactively
        if (predicted_cpu_load > self.config.target_cpu_utilization * 0.9 or 
            predicted_memory_load > self.config.target_memory_utilization * 0.9):
            
            if healthy_nodes < self.config.max_nodes:
                return ScalingAction(
                    direction=ScalingDirection.UP,
                    target_count=min(healthy_nodes + 1, self.config.max_nodes),
                    reason=f"Predictive: CPU {predicted_cpu_load:.1f}%, Memory {predicted_memory_load:.1f}%",
                    confidence=accuracy
                )
        
        return None
    
    async def execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute scaling action."""
        if action.executed:
            return True
        
        try:
            if action.direction == ScalingDirection.UP:
                if self._scale_up_callback:
                    await self._call_callback(self._scale_up_callback, action)
                    self._last_scale_up = time.time()
                    action.executed = True
                    self.logger.info(f"Scaled up to {action.target_count} nodes: {action.reason}")
                    return True
            
            elif action.direction == ScalingDirection.DOWN:
                if self._scale_down_callback:
                    await self._call_callback(self._scale_down_callback, action)
                    self._last_scale_down = time.time()
                    action.executed = True
                    self.logger.info(f"Scaled down to {action.target_count} nodes: {action.reason}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing scaling action: {e}")
            return False
    
    async def _call_callback(self, callback: Callable, *args) -> Any:
        """Call callback function, handling both sync and async."""
        if asyncio.iscoroutinefunction(callback):
            return await callback(*args)
        else:
            return callback(*args)
    
    async def perform_health_checks(self) -> None:
        """Perform health checks on all nodes."""
        if not self._health_check_callback:
            return
        
        for node in self.nodes:
            try:
                result = await self._call_callback(self._health_check_callback, node)
                node.healthy = bool(result)
                node.last_health_check = time.time()
            except Exception as e:
                self.logger.warning(f"Health check failed for node {node.id}: {e}")
                node.healthy = False
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        healthy_nodes = [node for node in self.nodes if node.healthy]
        
        return {
            'total_nodes': len(self.nodes),
            'healthy_nodes': len(healthy_nodes),
            'average_cpu': statistics.mean([node.cpu_usage for node in healthy_nodes]) if healthy_nodes else 0.0,
            'average_memory': statistics.mean([node.memory_usage for node in healthy_nodes]) if healthy_nodes else 0.0,
            'total_connections': sum(node.connections for node in healthy_nodes),
            'average_response_time': statistics.mean([node.response_time for node in healthy_nodes]) if healthy_nodes else 0.0,
            'last_scale_up': self._last_scale_up,
            'last_scale_down': self._last_scale_down,
            'scaling_actions': len(self._scaling_actions_history),
            'prediction_accuracy': self._predictive_scaler.get_prediction_accuracy() if self._predictive_scaler else None
        }

class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self._round_robin_index = 0
        self._custom_selector: Optional[Callable] = None
        self.logger = logger.getChild(self.__class__.__name__)
    
    def set_custom_selector(self, selector: Callable[[List[Node]], Node]) -> None:
        """Set custom node selection function."""
        self._custom_selector = selector
        self.strategy = LoadBalancingStrategy.CUSTOM
    
    def select_node(self, nodes: List[Node], request_info: Optional[Dict[str, Any]] = None) -> Optional[Node]:
        """Select best node for handling request."""
        healthy_nodes = [node for node in nodes if node.healthy]
        
        if not healthy_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_nodes)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_nodes)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_nodes, key=lambda n: n.get_score(self.strategy))
        
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return min(healthy_nodes, key=lambda n: n.get_score(self.strategy))
        
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return min(healthy_nodes, key=lambda n: n.get_score(self.strategy))
        
        elif self.strategy == LoadBalancingStrategy.CUSTOM and self._custom_selector:
            return self._custom_selector(healthy_nodes)
        
        else:
            # Default to least connections
            return min(healthy_nodes, key=lambda n: n.connections)
    
    def _round_robin_select(self, nodes: List[Node]) -> Node:
        """Round-robin node selection."""
        node = nodes[self._round_robin_index % len(nodes)]
        self._round_robin_index += 1
        return node
    
    def _weighted_round_robin_select(self, nodes: List[Node]) -> Node:
        """Weighted round-robin node selection."""
        total_weight = sum(node.weight for node in nodes)
        
        if total_weight == 0:
            return self._round_robin_select(nodes)
        
        # Create weighted list
        weighted_nodes = []
        for node in nodes:
            weight_ratio = node.weight / total_weight
            count = max(1, int(weight_ratio * 100))  # Scale to reasonable numbers
            weighted_nodes.extend([node] * count)
        
        if not weighted_nodes:
            return nodes[0]
        
        selected = weighted_nodes[self._round_robin_index % len(weighted_nodes)]
        self._round_robin_index += 1
        return selected
    
    def update_node_stats(self, node: Node, response_time: float, success: bool) -> None:
        """Update node statistics after request completion."""
        node.response_time = (node.response_time * 0.9) + (response_time * 0.1)  # EMA
        
        if success:
            node.current_load = max(0, node.current_load - 0.1)
        else:
            node.current_load = min(1.0, node.current_load + 0.1)


# Global instances
_autoscaler: Optional[AutoScaler] = None
_load_balancer: Optional[LoadBalancer] = None

def get_autoscaler() -> AutoScaler:
    """Get global autoscaler instance."""
    global _autoscaler
    if _autoscaler is None:
        _autoscaler = AutoScaler()
    return _autoscaler

def get_load_balancer() -> LoadBalancer:
    """Get global load balancer instance."""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    return _load_balancer