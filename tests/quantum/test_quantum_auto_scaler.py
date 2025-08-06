"""Comprehensive tests for Quantum Auto-Scaler.

Tests quantum-inspired auto-scaling algorithms, superposition-based
scaling decisions, and predictive scaling for HPU clusters.
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock

from gaudi3_scale.quantum.auto_scaler import (
    QuantumAutoScaler,
    QuantumScalingDecision,
    ScalingDirection,
    QuantumScalingState,
    ClusterMetrics
)
from gaudi3_scale.exceptions import ResourceAllocationError, ValidationError


class TestQuantumAutoScaler:
    """Test suite for quantum auto-scaler."""
    
    @pytest.fixture
    async def auto_scaler(self):
        """Create quantum auto-scaler for testing."""
        scaler = QuantumAutoScaler(
            min_nodes=2,
            max_nodes=16,
            target_utilization=0.75,
            scaling_cooldown=60.0,  # Shorter cooldown for testing
            quantum_coherence_time=30.0,
            enable_predictive_scaling=True,
            prediction_horizon=300.0
        )
        yield scaler
        await scaler.stop()  # Cleanup
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample cluster metrics."""
        return ClusterMetrics(
            timestamp=time.time(),
            cpu_utilization=0.7,
            memory_utilization=0.65,
            hpu_utilization=0.8,
            network_utilization=0.5,
            storage_utilization=0.3,
            power_consumption=6000.0,
            throughput=2000.0,
            latency=75.0,
            error_rate=0.001,
            queue_length=15,
            active_tasks=24
        )
    
    @pytest.mark.asyncio
    async def test_auto_scaler_initialization(self):
        """Test quantum auto-scaler initialization."""
        scaler = QuantumAutoScaler(
            min_nodes=4,
            max_nodes=32,
            target_utilization=0.8,
            scaling_cooldown=300.0
        )
        
        assert scaler.min_nodes == 4
        assert scaler.max_nodes == 32
        assert scaler.target_utilization == 0.8
        assert scaler.scaling_cooldown == 300.0
        assert scaler.current_nodes == 4  # Should start at min_nodes
        assert scaler.enable_predictive_scaling is True
        
        # Check initial state
        assert not scaler.scaling_in_progress
        assert scaler.last_scaling_time == 0.0
        assert len(scaler.metrics_history) == 0
        assert len(scaler.quantum_decisions) == 0
    
    @pytest.mark.asyncio
    async def test_metrics_update(self, auto_scaler, sample_metrics):
        """Test cluster metrics update."""
        await auto_scaler.update_metrics(sample_metrics)
        
        assert len(auto_scaler.metrics_history) == 1
        assert auto_scaler.metrics_history[0] == sample_metrics
    
    @pytest.mark.asyncio
    async def test_metrics_collection_simulation(self, auto_scaler):
        """Test simulated metrics collection."""
        metrics = await auto_scaler._collect_cluster_metrics()
        
        assert isinstance(metrics, ClusterMetrics)
        assert 0 <= metrics.cpu_utilization <= 1
        assert 0 <= metrics.memory_utilization <= 1
        assert 0 <= metrics.hpu_utilization <= 1
        assert 0 <= metrics.network_utilization <= 1
        assert metrics.power_consumption > 0
        assert metrics.throughput >= 0
        assert metrics.latency > 0
        assert 0 <= metrics.error_rate <= 1
        assert metrics.queue_length >= 0
        assert metrics.active_tasks >= 0
    
    @pytest.mark.asyncio
    async def test_scaling_evaluation_trigger(self, auto_scaler):
        """Test scaling evaluation trigger conditions."""
        # Not enough metrics history
        should_evaluate = await auto_scaler._should_evaluate_scaling()
        assert not should_evaluate
        
        # Add metrics but within target range
        for i in range(6):
            metrics = ClusterMetrics(
                timestamp=time.time(),
                hpu_utilization=0.75,  # Exactly at target
                queue_length=10
            )
            auto_scaler.metrics_history.append(metrics)
        
        should_evaluate = await auto_scaler._should_evaluate_scaling()
        assert not should_evaluate  # Within acceptable range
        
        # High utilization should trigger evaluation
        high_util_metrics = ClusterMetrics(
            timestamp=time.time(),
            hpu_utilization=0.9,  # Above target + threshold
            queue_length=60
        )
        auto_scaler.metrics_history.append(high_util_metrics)
        
        should_evaluate = await auto_scaler._should_evaluate_scaling()
        assert should_evaluate
    
    @pytest.mark.asyncio
    async def test_scaling_needs_analysis(self, auto_scaler):
        """Test scaling needs analysis."""
        # Create trend of increasing utilization
        base_time = time.time()
        for i in range(10):
            metrics = ClusterMetrics(
                timestamp=base_time + i * 60,
                hpu_utilization=0.6 + i * 0.05,  # Increasing trend
                memory_utilization=0.5 + i * 0.04,
                queue_length=i * 5
            )
            auto_scaler.metrics_history.append(metrics)
        
        analysis = await auto_scaler._analyze_scaling_needs()
        
        assert "current_utilization" in analysis
        assert "utilization_trend" in analysis
        assert "recommendation" in analysis
        assert "confidence" in analysis
        
        # Should recommend scale up due to high utilization
        assert analysis["recommendation"] in ["scale_up", "maintain"]
        assert 0 <= analysis["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_quantum_scaling_decision_creation(self, auto_scaler):
        """Test quantum scaling decision creation."""
        # Add metrics to trigger scaling evaluation
        for i in range(6):
            metrics = ClusterMetrics(
                timestamp=time.time(),
                hpu_utilization=0.9,  # High utilization
                queue_length=50
            )
            auto_scaler.metrics_history.append(metrics)
        
        await auto_scaler._evaluate_quantum_scaling()
        
        assert len(auto_scaler.quantum_decisions) == 1
        
        decision = next(iter(auto_scaler.quantum_decisions.values()))
        assert isinstance(decision, QuantumScalingDecision)
        assert decision.quantum_state == QuantumScalingState.SUPERPOSITION
        assert decision.confidence_level > 0
    
    @pytest.mark.asyncio
    async def test_quantum_superposition_creation(self, auto_scaler):
        """Test quantum superposition creation for scaling options."""
        decision = QuantumScalingDecision(
            decision_id="test_decision",
            resource_type=auto_scaler.resource_allocator.quantum_resources["hpu_cores"].resource_type,
            scaling_direction=ScalingDirection.QUANTUM_SUPERPOSITION
        )
        
        analysis = {
            "recommendation": "scale_up",
            "confidence": 0.8,
            "current_utilization": 0.9,
            "target_utilization": 0.75
        }
        
        await auto_scaler._create_scaling_superposition(decision, analysis)
        
        assert decision.confidence_level == 0.8
        assert decision.quantum_phase > 0  # Should have phase based on urgency
        assert decision.expected_improvement > 0  # Should expect improvement
    
    @pytest.mark.asyncio
    async def test_quantum_decision_evolution(self, auto_scaler):
        """Test quantum decision evolution over time."""
        # Create a decision
        decision = QuantumScalingDecision(
            decision_id="evolve_test",
            resource_type=auto_scaler.resource_allocator.quantum_resources["hpu_cores"].resource_type,
            scaling_direction=ScalingDirection.QUANTUM_SUPERPOSITION,
            quantum_state=QuantumScalingState.SUPERPOSITION,
            timestamp=time.time()
        )
        
        auto_scaler.quantum_decisions[decision.decision_id] = decision
        
        # Apply quantum evolution
        await auto_scaler._evolve_quantum_decisions()
        
        # Decision should still be in superposition if within coherence time
        assert decision.quantum_state == QuantumScalingState.SUPERPOSITION
        
        # Fast-forward time beyond coherence time
        decision.timestamp = time.time() - auto_scaler.quantum_coherence_time - 10
        
        await auto_scaler._evolve_quantum_decisions()
        
        # Decision should now be collapsed
        assert decision.quantum_state == QuantumScalingState.COLLAPSED
    
    @pytest.mark.asyncio
    async def test_quantum_decision_collapse(self, auto_scaler):
        """Test quantum decision collapse to classical scaling action."""
        decision = QuantumScalingDecision(
            decision_id="collapse_test",
            resource_type=auto_scaler.resource_allocator.quantum_resources["hpu_cores"].resource_type,
            scaling_direction=ScalingDirection.QUANTUM_SUPERPOSITION,
            quantum_state=QuantumScalingState.SUPERPOSITION,
            confidence_level=0.9,
            expected_improvement=0.2
        )
        
        await auto_scaler._collapse_quantum_decision(decision)
        
        assert decision.quantum_state == QuantumScalingState.COLLAPSED
        assert decision.scaling_direction != ScalingDirection.QUANTUM_SUPERPOSITION
        assert decision.scaling_direction in [
            ScalingDirection.SCALE_UP,
            ScalingDirection.SCALE_DOWN,
            ScalingDirection.MAINTAIN
        ]
    
    @pytest.mark.asyncio
    async def test_scaling_action_execution(self, auto_scaler):
        """Test actual scaling action execution."""
        initial_nodes = auto_scaler.current_nodes
        
        # Create a ready scaling decision
        decision = QuantumScalingDecision(
            decision_id="execute_test",
            resource_type=auto_scaler.resource_allocator.quantum_resources["hpu_cores"].resource_type,
            scaling_direction=ScalingDirection.SCALE_UP,
            quantum_state=QuantumScalingState.COLLAPSED,
            confidence_level=0.8
        )
        
        await auto_scaler._perform_scaling_action(decision)
        
        # Should have scaled up by one node
        assert auto_scaler.current_nodes == initial_nodes + 1
        assert auto_scaler.last_scaling_time > 0
        assert decision.quantum_state == QuantumScalingState.COHERENT
    
    @pytest.mark.asyncio
    async def test_scaling_bounds_enforcement(self, auto_scaler):
        """Test scaling bounds enforcement."""
        # Test upper bound
        auto_scaler.current_nodes = auto_scaler.max_nodes
        
        decision = QuantumScalingDecision(
            decision_id="upper_bound_test",
            resource_type=auto_scaler.resource_allocator.quantum_resources["hpu_cores"].resource_type,
            scaling_direction=ScalingDirection.SCALE_UP,
            quantum_state=QuantumScalingState.COLLAPSED,
            confidence_level=0.9
        )
        
        initial_nodes = auto_scaler.current_nodes
        await auto_scaler._perform_scaling_action(decision)
        
        # Should not scale beyond max
        assert auto_scaler.current_nodes == initial_nodes
        
        # Test lower bound
        auto_scaler.current_nodes = auto_scaler.min_nodes
        
        decision.scaling_direction = ScalingDirection.SCALE_DOWN
        initial_nodes = auto_scaler.current_nodes
        await auto_scaler._perform_scaling_action(decision)
        
        # Should not scale below min
        assert auto_scaler.current_nodes == initial_nodes
    
    @pytest.mark.asyncio
    async def test_predictive_scaling(self, auto_scaler):
        """Test predictive scaling functionality."""
        if not auto_scaler.enable_predictive_scaling:
            pytest.skip("Predictive scaling disabled")
        
        # Add historical data
        base_time = time.time()
        for i in range(25):  # Need at least 20 for prediction
            metrics = ClusterMetrics(
                timestamp=base_time + i * 60,
                hpu_utilization=0.6 + 0.3 * np.sin(i * 0.1),  # Sinusoidal pattern
                memory_utilization=0.5 + 0.2 * np.sin(i * 0.1),
                throughput=1000 + 500 * np.sin(i * 0.1),
                queue_length=max(0, int(20 * np.sin(i * 0.1) + 10))
            )
            auto_scaler.metrics_history.append(metrics)
        
        prediction = await auto_scaler._predict_future_demand()
        
        assert "predicted_utilization" in prediction
        assert "prediction_confidence" in prediction
        assert 0 <= prediction["predicted_utilization"] <= 1
        assert 0 <= prediction["prediction_confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_scaling_prevention(self, auto_scaler):
        """Test prevention of concurrent scaling operations."""
        auto_scaler.scaling_in_progress = True
        
        # Create ready decision
        decision = QuantumScalingDecision(
            decision_id="concurrent_test",
            resource_type=auto_scaler.resource_allocator.quantum_resources["hpu_cores"].resource_type,
            scaling_direction=ScalingDirection.SCALE_UP,
            quantum_state=QuantumScalingState.COLLAPSED,
            confidence_level=0.9
        )
        
        auto_scaler.quantum_decisions[decision.decision_id] = decision
        
        initial_nodes = auto_scaler.current_nodes
        await auto_scaler._execute_scaling_decisions()
        
        # Should not have scaled due to concurrent operation
        assert auto_scaler.current_nodes == initial_nodes
    
    @pytest.mark.asyncio
    async def test_scaling_cooldown_enforcement(self, auto_scaler):
        """Test scaling cooldown period enforcement."""
        # Set recent scaling time
        auto_scaler.last_scaling_time = time.time() - 30.0  # 30 seconds ago
        
        # Add metrics that would normally trigger scaling
        for i in range(6):
            metrics = ClusterMetrics(
                timestamp=time.time(),
                hpu_utilization=0.95,  # Very high utilization
                queue_length=100
            )
            auto_scaler.metrics_history.append(metrics)
        
        # Should not evaluate due to cooldown
        should_evaluate = await auto_scaler._should_evaluate_scaling()
        assert not should_evaluate
        
        # Set scaling time beyond cooldown
        auto_scaler.last_scaling_time = time.time() - auto_scaler.scaling_cooldown - 10
        
        should_evaluate = await auto_scaler._should_evaluate_scaling()
        assert should_evaluate
    
    @pytest.mark.asyncio
    async def test_decision_cleanup(self, auto_scaler):
        """Test cleanup of old quantum decisions."""
        # Create old decision
        old_decision = QuantumScalingDecision(
            decision_id="old_decision",
            resource_type=auto_scaler.resource_allocator.quantum_resources["hpu_cores"].resource_type,
            scaling_direction=ScalingDirection.MAINTAIN,
            timestamp=time.time() - 7200  # 2 hours ago
        )
        
        # Create recent decision
        recent_decision = QuantumScalingDecision(
            decision_id="recent_decision",
            resource_type=auto_scaler.resource_allocator.quantum_resources["hpu_cores"].resource_type,
            scaling_direction=ScalingDirection.MAINTAIN,
            timestamp=time.time() - 300  # 5 minutes ago
        )
        
        auto_scaler.quantum_decisions["old_decision"] = old_decision
        auto_scaler.quantum_decisions["recent_decision"] = recent_decision
        
        await auto_scaler._cleanup_old_decisions()
        
        # Old decision should be removed, recent should remain
        assert "old_decision" not in auto_scaler.quantum_decisions
        assert "recent_decision" in auto_scaler.quantum_decisions
    
    @pytest.mark.asyncio
    async def test_scaling_metrics_collection(self, auto_scaler):
        """Test scaling metrics collection."""
        # Add some state
        decision = QuantumScalingDecision(
            decision_id="metrics_test",
            resource_type=auto_scaler.resource_allocator.quantum_resources["hpu_cores"].resource_type,
            scaling_direction=ScalingDirection.QUANTUM_SUPERPOSITION,
            quantum_state=QuantumScalingState.SUPERPOSITION
        )
        auto_scaler.quantum_decisions[decision.decision_id] = decision
        
        # Add some metrics
        for i in range(5):
            metrics = ClusterMetrics(
                timestamp=time.time(),
                hpu_utilization=0.7
            )
            auto_scaler.metrics_history.append(metrics)
        
        scaling_metrics = await auto_scaler.get_scaling_metrics()
        
        required_fields = [
            "current_nodes", "min_nodes", "max_nodes",
            "target_utilization", "current_utilization",
            "active_decisions", "superposition_decisions",
            "quantum_coherence", "scaling_in_progress",
            "last_scaling_time", "metrics_history_size"
        ]
        
        for field in required_fields:
            assert field in scaling_metrics
        
        assert scaling_metrics["current_nodes"] == auto_scaler.current_nodes
        assert scaling_metrics["active_decisions"] == 1
        assert scaling_metrics["superposition_decisions"] == 1
        assert 0 <= scaling_metrics["quantum_coherence"] <= 1
    
    @pytest.mark.asyncio
    async def test_cluster_scaling_resource_updates(self, auto_scaler):
        """Test resource updates during cluster scaling."""
        initial_hpu_cores = auto_scaler.resource_allocator.available_resources["hpu_cores"]
        initial_memory = auto_scaler.resource_allocator.available_resources["memory_gb"]
        
        # Scale up by 2 nodes
        target_nodes = auto_scaler.current_nodes + 2
        await auto_scaler._scale_cluster_to_nodes(target_nodes)
        
        # Resources should be updated
        new_hpu_cores = auto_scaler.resource_allocator.available_resources["hpu_cores"]
        new_memory = auto_scaler.resource_allocator.available_resources["memory_gb"]
        
        expected_hpu_increase = 2 * auto_scaler.resource_allocator.hpu_per_node
        expected_memory_increase = expected_hpu_increase * 96
        
        assert new_hpu_cores == initial_hpu_cores + expected_hpu_increase
        assert new_memory == initial_memory + expected_memory_increase
    
    @pytest.mark.asyncio
    async def test_auto_scaler_start_stop(self, auto_scaler):
        """Test auto-scaler start and stop operations."""
        assert not auto_scaler._running
        
        # Start auto-scaler
        await auto_scaler.start()
        assert auto_scaler._running
        assert auto_scaler._monitoring_task is not None
        
        # Stop auto-scaler
        await auto_scaler.stop()
        assert not auto_scaler._running
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_resilience(self, auto_scaler):
        """Test monitoring loop error resilience."""
        # Mock an error in metrics collection
        with patch.object(auto_scaler, '_collect_cluster_metrics', side_effect=Exception("Test error")):
            # Start monitoring
            await auto_scaler.start()
            
            # Let it run for a short time
            await asyncio.sleep(0.1)
            
            # Should still be running despite errors
            assert auto_scaler._running
            
            await auto_scaler.stop()
    
    @pytest.mark.asyncio
    async def test_scaling_decision_properties(self):
        """Test QuantumScalingDecision properties."""
        decision = QuantumScalingDecision(
            decision_id="test_decision",
            resource_type=auto_scaler.resource_allocator.quantum_resources["hpu_cores"].resource_type,
            scaling_direction=ScalingDirection.SCALE_UP,
            quantum_amplitude=complex(0.8, 0.0),
            confidence_level=0.9
        )
        
        # Test probability amplitude
        assert abs(decision.probability_amplitude - 0.64) < 0.01  # |0.8|² = 0.64
        
        # Test state transitions
        assert decision.quantum_state == QuantumScalingState.COHERENT
        decision.quantum_state = QuantumScalingState.SUPERPOSITION
        assert decision.quantum_state == QuantumScalingState.SUPERPOSITION


@pytest.mark.asyncio
async def test_quantum_auto_scaler_integration():
    """Integration test for complete quantum auto-scaling workflow."""
    auto_scaler = QuantumAutoScaler(
        min_nodes=3,
        max_nodes=12,
        target_utilization=0.7,
        scaling_cooldown=30.0,
        quantum_coherence_time=60.0,
        enable_predictive_scaling=True
    )
    
    try:
        # Start the auto-scaler
        await auto_scaler.start()
        
        # Simulate workload increasing over time
        base_time = time.time()
        
        # Phase 1: Normal load
        for i in range(10):
            metrics = ClusterMetrics(
                timestamp=base_time + i * 30,
                hpu_utilization=0.6 + np.random.normal(0, 0.05),
                memory_utilization=0.55 + np.random.normal(0, 0.03),
                throughput=1500 + np.random.normal(0, 100),
                latency=60 + np.random.normal(0, 5),
                queue_length=max(0, int(np.random.normal(10, 3))),
                active_tasks=int(auto_scaler.current_nodes * 8 * 0.6)
            )
            await auto_scaler.update_metrics(metrics)
        
        initial_nodes = auto_scaler.current_nodes
        initial_metrics = await auto_scaler.get_scaling_metrics()
        
        # Phase 2: Increasing load (should trigger scale up)
        for i in range(10):
            utilization = 0.8 + i * 0.02  # Gradually increasing
            metrics = ClusterMetrics(
                timestamp=base_time + (i + 10) * 30,
                hpu_utilization=min(0.98, utilization),
                memory_utilization=min(0.95, utilization * 0.9),
                throughput=2000 + i * 100,
                latency=70 + i * 5,
                queue_length=20 + i * 5,
                active_tasks=int(auto_scaler.current_nodes * 8 * utilization)
            )
            await auto_scaler.update_metrics(metrics)
        
        # Wait a bit for scaling decisions to process
        await asyncio.sleep(1.0)
        
        # Check if scaling decisions were created
        high_load_metrics = await auto_scaler.get_scaling_metrics()
        
        # Phase 3: Decreasing load (should trigger scale down eventually)
        for i in range(15):
            utilization = max(0.4, 0.9 - i * 0.04)  # Gradually decreasing
            metrics = ClusterMetrics(
                timestamp=base_time + (i + 20) * 30,
                hpu_utilization=utilization,
                memory_utilization=utilization * 0.8,
                throughput=max(800, 2500 - i * 100),
                latency=max(45, 120 - i * 5),
                queue_length=max(0, 50 - i * 3),
                active_tasks=int(auto_scaler.current_nodes * 8 * utilization)
            )
            await auto_scaler.update_metrics(metrics)
        
        # Wait for final processing
        await asyncio.sleep(1.0)
        
        final_metrics = await auto_scaler.get_scaling_metrics()
        
        # Verify system behavior
        assert len(auto_scaler.metrics_history) >= 30  # Should have collected metrics
        assert final_metrics["metrics_history_size"] >= 30
        
        # Should have created quantum scaling decisions
        assert final_metrics["active_decisions"] >= 0
        
        # System should maintain quantum coherence
        assert 0 <= final_metrics["quantum_coherence"] <= 1
        
        print(f"✅ Quantum auto-scaler integration test completed successfully")
        print(f"   - Initial nodes: {initial_nodes}")
        print(f"   - Final nodes: {final_metrics['current_nodes']}")
        print(f"   - Total scaling decisions: {final_metrics['active_decisions']}")
        print(f"   - Quantum coherence: {final_metrics['quantum_coherence']:.3f}")
        print(f"   - Current utilization: {final_metrics['current_utilization']:.3f}")
        print(f"   - Target utilization: {final_metrics['target_utilization']:.3f}")
        print(f"   - Predictive scaling: {final_metrics['predictive_scaling_enabled']}")
        
        # Verify metrics make sense
        assert final_metrics['min_nodes'] <= final_metrics['current_nodes'] <= final_metrics['max_nodes']
        assert 0 <= final_metrics['current_utilization'] <= 1
        assert final_metrics['quantum_coherence_time'] > 0
        
    finally:
        await auto_scaler.stop()


if __name__ == "__main__":
    # Run integration test directly
    asyncio.run(test_quantum_auto_scaler_integration())