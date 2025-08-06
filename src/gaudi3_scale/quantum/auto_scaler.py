"""Quantum-Inspired Auto-Scaling for Dynamic HPU Cluster Management.

Implements quantum algorithms for intelligent auto-scaling decisions:
- Quantum superposition for exploring multiple scaling paths
- Quantum annealing for optimal resource allocation
- Quantum entanglement for coordinated scaling across nodes
- Quantum interference for load balancing optimization
"""

import asyncio
import logging
import math
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from collections import deque
import uuid

from ..exceptions import ResourceAllocationError, QuantumOptimizationError
from ..validation import DataValidator
from ..monitoring.performance import PerformanceMonitor
from .resource_allocator import QuantumResourceAllocator, ResourceType

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Quantum scaling directions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"      # Add more nodes
    SCALE_IN = "scale_in"        # Remove nodes
    MAINTAIN = "maintain"
    QUANTUM_SUPERPOSITION = "quantum_superposition"  # Multiple states simultaneously


class QuantumScalingState(Enum):
    """Quantum states for scaling operations."""
    COHERENT = "coherent"        # Normal scaling state
    SUPERPOSITION = "superposition"  # Exploring multiple scaling options
    ENTANGLED = "entangled"      # Coordinated scaling across resources
    COLLAPSED = "collapsed"      # Scaling decision made
    DECOHERENT = "decoherent"    # Scaling failure or timeout


@dataclass
class QuantumScalingDecision:
    """Quantum representation of scaling decision."""
    decision_id: str
    resource_type: ResourceType
    scaling_direction: ScalingDirection
    scaling_factor: float = 1.0  # Multiplier for scaling
    quantum_amplitude: complex = field(default_factory=lambda: complex(1.0, 0.0))
    quantum_phase: float = 0.0
    confidence_level: float = 1.0
    expected_improvement: float = 0.0
    quantum_state: QuantumScalingState = QuantumScalingState.COHERENT
    entangled_decisions: Set[str] = field(default_factory=set)
    timestamp: float = field(default_factory=time.time)
    
    @property
    def probability_amplitude(self) -> float:
        """Calculate probability amplitude |ψ|²."""
        return abs(self.quantum_amplitude) ** 2


@dataclass
class ClusterMetrics:
    """Real-time cluster performance metrics."""
    timestamp: float
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    hpu_utilization: float = 0.0
    network_utilization: float = 0.0
    storage_utilization: float = 0.0
    power_consumption: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    queue_length: int = 0
    active_tasks: int = 0


class QuantumAutoScaler:
    """Quantum-inspired auto-scaler for HPU clusters."""
    
    def __init__(self,
                 min_nodes: int = 1,
                 max_nodes: int = 64,
                 target_utilization: float = 0.75,
                 scaling_cooldown: float = 300.0,  # 5 minutes
                 quantum_coherence_time: float = 60.0,
                 enable_predictive_scaling: bool = True,
                 prediction_horizon: float = 1800.0):  # 30 minutes
        """Initialize quantum auto-scaler.
        
        Args:
            min_nodes: Minimum cluster nodes
            max_nodes: Maximum cluster nodes  
            target_utilization: Target resource utilization (0-1)
            scaling_cooldown: Cooldown between scaling operations
            quantum_coherence_time: Quantum coherence time for decisions
            enable_predictive_scaling: Enable predictive scaling algorithms
            prediction_horizon: Time horizon for predictions (seconds)
        """
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.target_utilization = target_utilization
        self.scaling_cooldown = scaling_cooldown
        self.quantum_coherence_time = quantum_coherence_time
        self.enable_predictive_scaling = enable_predictive_scaling
        self.prediction_horizon = prediction_horizon
        
        # Current cluster state
        self.current_nodes = min_nodes
        self.current_resources: Dict[ResourceType, float] = {}
        
        # Quantum scaling system
        self.quantum_decisions: Dict[str, QuantumScalingDecision] = {}
        self.entanglement_graph: Dict[str, Set[str]] = {}
        
        # Metrics collection
        self.metrics_history: deque = deque(maxlen=1000)  # Last 1000 metrics
        self.performance_monitor = PerformanceMonitor()
        
        # Resource allocator integration
        self.resource_allocator = QuantumResourceAllocator()
        
        # Scaling state
        self.last_scaling_time = 0.0
        self.scaling_in_progress = False
        self._monitoring_task = None
        self._running = False
        
        # Quantum prediction model
        self.prediction_weights = np.random.random(8)  # 8 features
        self.prediction_learning_rate = 0.01
        
        logger.info(f"Initialized QuantumAutoScaler: {min_nodes}-{max_nodes} nodes, target {target_utilization:.2f}")
    
    async def start(self):
        """Start the quantum auto-scaling system."""
        if self._running:
            return
        
        self._running = True
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Initialize resource allocator
        await self.resource_allocator.__init__(self.current_nodes)
        
        logger.info("Started quantum auto-scaling system")
    
    async def stop(self):
        """Stop the quantum auto-scaling system."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped quantum auto-scaling system")
    
    async def update_metrics(self, metrics: ClusterMetrics):
        """Update cluster metrics for scaling decisions."""
        self.metrics_history.append(metrics)
        
        # Check if scaling evaluation is needed
        if await self._should_evaluate_scaling():
            await self._evaluate_quantum_scaling()
    
    async def _monitoring_loop(self):
        """Background monitoring and scaling loop."""
        
        while self._running:
            try:
                # Collect current metrics (simulated for now)
                current_metrics = await self._collect_cluster_metrics()
                await self.update_metrics(current_metrics)
                
                # Evolve quantum scaling decisions
                await self._evolve_quantum_decisions()
                
                # Execute any ready scaling decisions
                await self._execute_scaling_decisions()
                
                # Clean up old decisions
                await self._cleanup_old_decisions()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(30.0)  # 30-second monitoring cycle
                
            except Exception as e:
                logger.error(f"Error in quantum auto-scaler monitoring loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _collect_cluster_metrics(self) -> ClusterMetrics:
        """Collect current cluster performance metrics."""
        # Simulate metrics collection (in real implementation, collect from monitoring systems)
        base_utilization = 0.6 + 0.3 * np.sin(time.time() / 3600)  # Hourly pattern
        noise = np.random.normal(0, 0.05)
        
        return ClusterMetrics(
            timestamp=time.time(),
            cpu_utilization=max(0.0, min(1.0, base_utilization + noise)),
            memory_utilization=max(0.0, min(1.0, base_utilization + noise * 0.8)),
            hpu_utilization=max(0.0, min(1.0, base_utilization + noise * 1.2)),
            network_utilization=max(0.0, min(1.0, base_utilization * 0.7 + noise)),
            storage_utilization=0.3 + 0.1 * np.sin(time.time() / 7200),  # Slower change
            power_consumption=self.current_nodes * 1500 * (0.5 + base_utilization * 0.5),
            throughput=1000 * self.current_nodes * base_utilization,
            latency=50 + 100 * (base_utilization ** 2),
            error_rate=max(0.0, min(0.1, 0.001 + noise * 0.002)),
            queue_length=max(0, int(20 * max(0, base_utilization - 0.7))),
            active_tasks=int(self.current_nodes * 8 * base_utilization)
        )
    
    async def _should_evaluate_scaling(self) -> bool:
        """Determine if scaling evaluation is needed."""
        if len(self.metrics_history) < 5:
            return False
        
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return False
        
        # Check for resource pressure
        recent_metrics = list(self.metrics_history)[-5:]
        avg_hpu_util = np.mean([m.hpu_utilization for m in recent_metrics])
        
        # Evaluate if utilization is outside target range
        if avg_hpu_util > self.target_utilization + 0.1:
            return True  # High utilization - consider scaling up
        elif avg_hpu_util < self.target_utilization - 0.2:
            return True  # Low utilization - consider scaling down
        
        # Check queue buildup
        current_queue = recent_metrics[-1].queue_length
        if current_queue > 50:
            return True
        
        return False
    
    async def _evaluate_quantum_scaling(self):
        """Evaluate quantum scaling options using superposition."""
        
        # Create quantum scaling decision in superposition
        decision_id = f"scale_decision_{uuid.uuid4().hex[:8]}"
        
        # Analyze current state and predict future needs
        scaling_analysis = await self._analyze_scaling_needs()
        
        # Create quantum decision in superposition of multiple scaling options
        quantum_decision = QuantumScalingDecision(
            decision_id=decision_id,
            resource_type=ResourceType.HPU_CORES,  # Primary scaling resource
            scaling_direction=ScalingDirection.QUANTUM_SUPERPOSITION,
            quantum_state=QuantumScalingState.SUPERPOSITION
        )
        
        # Apply quantum superposition - explore multiple scaling paths simultaneously
        await self._create_scaling_superposition(quantum_decision, scaling_analysis)
        
        # Store decision
        self.quantum_decisions[decision_id] = quantum_decision
        
        logger.info(f"Created quantum scaling decision {decision_id} in superposition")
    
    async def _analyze_scaling_needs(self) -> Dict[str, Any]:
        """Analyze current scaling needs and predict future requirements."""
        
        if len(self.metrics_history) < 10:
            return {"recommendation": "maintain", "confidence": 0.5}
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate trends
        timestamps = [m.timestamp for m in recent_metrics]
        hpu_utils = [m.hpu_utilization for m in recent_metrics]
        memory_utils = [m.memory_utilization for m in recent_metrics]
        queue_lengths = [m.queue_length for m in recent_metrics]
        
        # Linear regression for trend analysis
        time_diffs = [(t - timestamps[0]) / 3600 for t in timestamps]  # Hours
        
        hpu_trend = np.polyfit(time_diffs, hpu_utils, 1)[0]  # Slope
        memory_trend = np.polyfit(time_diffs, memory_utils, 1)[0]
        queue_trend = np.polyfit(time_diffs, queue_lengths, 1)[0]
        
        current_hpu_util = recent_metrics[-1].hpu_utilization
        current_queue = recent_metrics[-1].queue_length
        
        # Scaling analysis
        analysis = {
            "current_utilization": current_hpu_util,
            "utilization_trend": hpu_trend,
            "memory_trend": memory_trend,
            "queue_length": current_queue,
            "queue_trend": queue_trend,
            "target_utilization": self.target_utilization,
            "current_nodes": self.current_nodes,
            "max_nodes": self.max_nodes,
            "min_nodes": self.min_nodes
        }
        
        # Decision logic
        if current_hpu_util > self.target_utilization + 0.1 or current_queue > 30:
            if self.current_nodes < self.max_nodes:
                analysis["recommendation"] = "scale_up"
                analysis["confidence"] = min(0.9, 0.6 + 2 * (current_hpu_util - self.target_utilization))
            else:
                analysis["recommendation"] = "maintain"
                analysis["confidence"] = 0.7
        elif current_hpu_util < self.target_utilization - 0.2 and current_queue < 5:
            if self.current_nodes > self.min_nodes:
                analysis["recommendation"] = "scale_down"
                analysis["confidence"] = min(0.9, 0.6 + 2 * (self.target_utilization - current_hpu_util))
            else:
                analysis["recommendation"] = "maintain"
                analysis["confidence"] = 0.7
        else:
            analysis["recommendation"] = "maintain"
            analysis["confidence"] = 0.8
        
        # Add predictive analysis if enabled
        if self.enable_predictive_scaling:
            prediction = await self._predict_future_demand()
            analysis["predicted_demand"] = prediction
        
        return analysis
    
    async def _create_scaling_superposition(self, 
                                          quantum_decision: QuantumScalingDecision,
                                          analysis: Dict[str, Any]):
        """Create quantum superposition of scaling options."""
        
        recommendation = analysis["recommendation"]
        confidence = analysis["confidence"]
        
        # Create superposition amplitudes for different scaling options
        if recommendation == "scale_up":
            # Superposition favoring scale-up
            scale_up_amplitude = complex(np.sqrt(confidence), 0)
            maintain_amplitude = complex(np.sqrt(1 - confidence), 0)
            scale_down_amplitude = complex(0, 0)
        elif recommendation == "scale_down":
            # Superposition favoring scale-down  
            scale_up_amplitude = complex(0, 0)
            maintain_amplitude = complex(np.sqrt(1 - confidence), 0)
            scale_down_amplitude = complex(np.sqrt(confidence), 0)
        else:
            # Balanced superposition
            scale_up_amplitude = complex(np.sqrt(0.3), 0)
            maintain_amplitude = complex(np.sqrt(0.4), 0)
            scale_down_amplitude = complex(np.sqrt(0.3), 0)
        
        # Store superposition state (simplified representation)
        quantum_decision.quantum_amplitude = scale_up_amplitude  # Primary amplitude
        quantum_decision.confidence_level = confidence
        
        # Add quantum phase based on urgency
        current_util = analysis["current_utilization"]
        util_deviation = abs(current_util - self.target_utilization)
        quantum_decision.quantum_phase = util_deviation * math.pi  # Phase proportional to urgency
        
        # Calculate expected improvement
        if recommendation == "scale_up":
            expected_nodes = min(self.max_nodes, self.current_nodes + 1)
            improvement = (expected_nodes - self.current_nodes) / self.current_nodes
        elif recommendation == "scale_down":
            expected_nodes = max(self.min_nodes, self.current_nodes - 1)
            improvement = (self.current_nodes - expected_nodes) / self.current_nodes * 0.5  # Cost savings
        else:
            improvement = 0.0
        
        quantum_decision.expected_improvement = improvement
    
    async def _predict_future_demand(self) -> Dict[str, float]:
        """Predict future resource demand using quantum-inspired algorithms."""
        
        if len(self.metrics_history) < 20:
            return {"predicted_utilization": self.metrics_history[-1].hpu_utilization}
        
        # Extract features for prediction
        recent_metrics = list(self.metrics_history)[-20:]
        
        features = []
        for i, metrics in enumerate(recent_metrics):
            feature_vector = [
                metrics.hpu_utilization,
                metrics.memory_utilization,
                metrics.cpu_utilization,
                metrics.network_utilization,
                metrics.throughput / 1000,  # Normalized
                metrics.latency / 100,      # Normalized
                metrics.queue_length / 50,  # Normalized
                math.sin(2 * math.pi * (metrics.timestamp % 86400) / 86400)  # Time of day
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        # Simple quantum-inspired prediction using weighted average
        # In practice, this could be a more sophisticated quantum ML model
        
        # Apply quantum interference patterns to features
        interference_weights = np.array([
            1.0,  # HPU utilization - most important
            0.8,  # Memory utilization
            0.6,  # CPU utilization
            0.4,  # Network utilization
            0.7,  # Throughput
            0.5,  # Latency
            0.9,  # Queue length - very important for scaling
            0.3   # Time pattern
        ])
        
        # Calculate quantum prediction using superposition principle
        weighted_features = features_array * interference_weights
        recent_trend = np.mean(weighted_features[-5:], axis=0)
        historical_avg = np.mean(weighted_features[:-5], axis=0)
        
        # Quantum superposition of recent trend and historical pattern
        prediction_vector = 0.7 * recent_trend + 0.3 * historical_avg
        
        # Extract primary prediction (HPU utilization)
        predicted_utilization = prediction_vector[0]
        
        # Apply bounds
        predicted_utilization = max(0.0, min(1.0, predicted_utilization))
        
        return {
            "predicted_utilization": predicted_utilization,
            "prediction_confidence": 0.7,
            "time_horizon": self.prediction_horizon
        }
    
    async def _evolve_quantum_decisions(self):
        """Apply quantum evolution to scaling decisions."""
        
        current_time = time.time()
        
        for decision in self.quantum_decisions.values():
            if decision.quantum_state == QuantumScalingState.SUPERPOSITION:
                
                # Check if decision should collapse
                age = current_time - decision.timestamp
                
                if age > self.quantum_coherence_time:
                    # Quantum decoherence - collapse to classical decision
                    await self._collapse_quantum_decision(decision)
                else:
                    # Apply quantum evolution
                    time_factor = age / self.quantum_coherence_time
                    
                    # Phase evolution
                    decision.quantum_phase += 0.1 * time_factor
                    
                    # Amplitude evolution (gradual decoherence)
                    decoherence_factor = np.exp(-0.5 * time_factor)
                    decision.quantum_amplitude *= decoherence_factor
    
    async def _collapse_quantum_decision(self, decision: QuantumScalingDecision):
        """Collapse quantum decision to classical scaling action."""
        
        # Measure quantum state - probabilistic collapse
        measurement_probability = decision.probability_amplitude
        
        # Quantum measurement outcome
        measurement_outcome = np.random.random() < measurement_probability
        
        if measurement_outcome and decision.confidence_level > 0.6:
            # Collapse to scaling action
            if decision.expected_improvement > 0:
                if decision.quantum_amplitude.real > 0.5:
                    decision.scaling_direction = ScalingDirection.SCALE_UP
                else:
                    decision.scaling_direction = ScalingDirection.SCALE_DOWN
            else:
                decision.scaling_direction = ScalingDirection.MAINTAIN
        else:
            # Collapse to maintain current state
            decision.scaling_direction = ScalingDirection.MAINTAIN
        
        decision.quantum_state = QuantumScalingState.COLLAPSED
        
        logger.info(f"Quantum decision {decision.decision_id} collapsed to {decision.scaling_direction.value}")
    
    async def _execute_scaling_decisions(self):
        """Execute ready quantum scaling decisions."""
        
        if self.scaling_in_progress:
            return
        
        # Find collapsed decisions ready for execution
        ready_decisions = [
            d for d in self.quantum_decisions.values() 
            if d.quantum_state == QuantumScalingState.COLLAPSED and 
               d.scaling_direction != ScalingDirection.MAINTAIN
        ]
        
        if not ready_decisions:
            return
        
        # Execute highest confidence decision
        best_decision = max(ready_decisions, key=lambda d: d.confidence_level)
        
        if best_decision.confidence_level > 0.7:  # Only execute high confidence decisions
            await self._perform_scaling_action(best_decision)
    
    async def _perform_scaling_action(self, decision: QuantumScalingDecision):
        """Perform actual scaling action."""
        
        self.scaling_in_progress = True
        
        try:
            if decision.scaling_direction == ScalingDirection.SCALE_UP:
                if self.current_nodes < self.max_nodes:
                    new_nodes = min(self.max_nodes, self.current_nodes + 1)
                    await self._scale_cluster_to_nodes(new_nodes)
                    logger.info(f"Scaled up cluster from {self.current_nodes} to {new_nodes} nodes")
                    
            elif decision.scaling_direction == ScalingDirection.SCALE_DOWN:
                if self.current_nodes > self.min_nodes:
                    new_nodes = max(self.min_nodes, self.current_nodes - 1)
                    await self._scale_cluster_to_nodes(new_nodes)
                    logger.info(f"Scaled down cluster from {self.current_nodes} to {new_nodes} nodes")
            
            self.last_scaling_time = time.time()
            decision.quantum_state = QuantumScalingState.COHERENT  # Mark as completed
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
            decision.quantum_state = QuantumScalingState.DECOHERENT
            
            # Create quantum error for tracking
            error = ResourceAllocationError(
                f"Scaling action failed: {e}",
                resource_type="cluster_nodes"
            )
            
        finally:
            self.scaling_in_progress = False
    
    async def _scale_cluster_to_nodes(self, target_nodes: int):
        """Scale cluster to target number of nodes."""
        
        if target_nodes == self.current_nodes:
            return
        
        # Simulate scaling operation (in real implementation, this would call cloud APIs)
        await asyncio.sleep(2.0)  # Simulate scaling delay
        
        # Update current state
        old_nodes = self.current_nodes
        self.current_nodes = target_nodes
        
        # Update resource allocator
        self.resource_allocator.cluster_nodes = target_nodes
        self.resource_allocator.total_hpus = target_nodes * self.resource_allocator.hpu_per_node
        
        # Recalculate available resources
        hpu_change = (target_nodes - old_nodes) * self.resource_allocator.hpu_per_node
        memory_change = hpu_change * 96  # GB per HPU
        network_change = (target_nodes - old_nodes) * 200  # Gbps per node
        
        self.resource_allocator.available_resources["hpu_cores"] += hpu_change
        self.resource_allocator.available_resources["memory_gb"] += memory_change
        self.resource_allocator.available_resources["network_bandwidth"] += network_change
    
    async def _cleanup_old_decisions(self):
        """Clean up old quantum decisions."""
        
        current_time = time.time()
        cleanup_age = 3600.0  # 1 hour
        
        old_decision_ids = [
            decision_id for decision_id, decision in self.quantum_decisions.items()
            if (current_time - decision.timestamp) > cleanup_age
        ]
        
        for decision_id in old_decision_ids:
            del self.quantum_decisions[decision_id]
            
        if old_decision_ids:
            logger.info(f"Cleaned up {len(old_decision_ids)} old quantum scaling decisions")
    
    async def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling metrics."""
        
        active_decisions = len([d for d in self.quantum_decisions.values() 
                               if d.quantum_state != QuantumScalingState.DECOHERENT])
        
        superposition_decisions = len([d for d in self.quantum_decisions.values() 
                                     if d.quantum_state == QuantumScalingState.SUPERPOSITION])
        
        # Calculate quantum coherence of scaling system
        total_amplitude = sum(abs(d.quantum_amplitude) for d in self.quantum_decisions.values())
        quantum_coherence = total_amplitude / max(1, len(self.quantum_decisions))
        
        # Recent performance
        recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
        avg_utilization = np.mean([m.hpu_utilization for m in recent_metrics]) if recent_metrics else 0.0
        
        return {
            "current_nodes": self.current_nodes,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "target_utilization": self.target_utilization,
            "current_utilization": avg_utilization,
            "active_decisions": active_decisions,
            "superposition_decisions": superposition_decisions,
            "quantum_coherence": quantum_coherence,
            "scaling_in_progress": self.scaling_in_progress,
            "last_scaling_time": self.last_scaling_time,
            "time_since_last_scaling": time.time() - self.last_scaling_time,
            "metrics_history_size": len(self.metrics_history),
            "predictive_scaling_enabled": self.enable_predictive_scaling,
            "quantum_coherence_time": self.quantum_coherence_time
        }
    
    def __del__(self):
        """Clean up resources."""
        if self._running:
            asyncio.create_task(self.stop())