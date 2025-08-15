"""Comprehensive Test Suite for Research Components.

This module provides extensive testing for the research framework including:
- Adaptive batch optimization algorithms
- Quantum-hybrid scheduling systems  
- Error recovery mechanisms
- Performance benchmarking
- Statistical validation
"""

import pytest
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import logging

# Test imports
from .adaptive_batch_optimizer import (
    AdaptiveBatchOptimizer, OptimizationStrategy, BatchMetrics, 
    QuantumInspiredScheduler, ReinforcementLearningOptimizer
)
from .quantum_hybrid_scheduler import (
    QuantumHybridScheduler, Task, Node, TaskPriority, ResourceType,
    create_gaudi3_node, create_training_task
)
from .error_recovery import (
    SmartCheckpointManager, AdvancedErrorRecovery, ErrorType,
    create_checkpoint_manager, create_error_recovery_system
)

logger = logging.getLogger(__name__)


class TestAdaptiveBatchOptimizer:
    """Test suite for adaptive batch optimization."""
    
    def test_batch_metrics_creation(self):
        """Test BatchMetrics dataclass creation and validation."""
        metrics = BatchMetrics(
            batch_size=64,
            throughput_samples_per_sec=850.5,
            memory_utilization_percent=78.2,
            convergence_rate=0.89,
            training_loss=0.245,
            gradient_norm=1.23,
            hpu_utilization_percent=92.1,
            latency_ms=45.6,
            energy_efficiency_score=0.87
        )
        
        assert metrics.batch_size == 64
        assert metrics.throughput_samples_per_sec == 850.5
        assert metrics.timestamp is not None
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        assert 'batch_size' in metrics_dict
        assert 'timestamp' in metrics_dict
        
        # Test deserialization
        reconstructed = BatchMetrics.from_dict(metrics_dict)
        assert reconstructed.batch_size == metrics.batch_size
        assert reconstructed.throughput_samples_per_sec == metrics.throughput_samples_per_sec
    
    def test_quantum_scheduler_initialization(self):
        """Test quantum-inspired scheduler initialization."""
        scheduler = QuantumInspiredScheduler(
            initial_temperature=100.0,
            cooling_rate=0.95,
            min_temperature=0.01,
            max_iterations=100
        )
        
        assert scheduler.initial_temperature == 100.0
        assert scheduler.cooling_rate == 0.95
        assert scheduler.current_temperature == 100.0
    
    def test_quantum_scheduler_batch_generation(self):
        """Test quantum scheduler batch size generation."""
        scheduler = QuantumInspiredScheduler()
        
        current_batch = 64
        current_score = 0.8
        search_space = (8, 256)
        iteration = 5
        
        next_batch = scheduler.next_batch_size(
            current_batch, current_score, search_space, iteration
        )
        
        # Should be within search space
        assert search_space[0] <= next_batch <= search_space[1]
        
        # Should be power of 2 aligned
        assert next_batch & (next_batch - 1) == 0  # Check if power of 2
    
    def test_rl_optimizer_initialization(self):
        """Test reinforcement learning optimizer initialization."""
        rl_optimizer = ReinforcementLearningOptimizer(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1
        )
        
        assert rl_optimizer.learning_rate == 0.1
        assert rl_optimizer.discount_factor == 0.95
        assert len(rl_optimizer.actions) > 0
    
    def test_rl_optimizer_state_generation(self):
        """Test RL optimizer state representation."""
        rl_optimizer = ReinforcementLearningOptimizer()
        
        metrics = BatchMetrics(
            batch_size=64,
            throughput_samples_per_sec=800.0,
            memory_utilization_percent=75.0,
            convergence_rate=0.85,
            training_loss=0.3,
            gradient_norm=1.0,
            hpu_utilization_percent=90.0,
            latency_ms=50.0,
            energy_efficiency_score=0.8
        )
        
        state = rl_optimizer.get_state(metrics)
        assert isinstance(state, str)
        assert len(state) > 0
    
    def test_rl_optimizer_reward_calculation(self):
        """Test RL reward calculation."""
        rl_optimizer = ReinforcementLearningOptimizer()
        
        good_metrics = BatchMetrics(
            batch_size=64,
            throughput_samples_per_sec=1000.0,
            memory_utilization_percent=85.0,
            convergence_rate=0.95,
            training_loss=0.1,
            gradient_norm=0.8,
            hpu_utilization_percent=95.0,
            latency_ms=30.0,
            energy_efficiency_score=0.95
        )
        
        poor_metrics = BatchMetrics(
            batch_size=32,
            throughput_samples_per_sec=200.0,
            memory_utilization_percent=40.0,
            convergence_rate=0.4,
            training_loss=1.5,
            gradient_norm=5.0,
            hpu_utilization_percent=50.0,
            latency_ms=100.0,
            energy_efficiency_score=0.3
        )
        
        good_reward = rl_optimizer.get_reward(good_metrics)
        poor_reward = rl_optimizer.get_reward(poor_metrics)
        
        assert good_reward > poor_reward
        assert 0 <= good_reward <= 1.2  # Allow for improvement bonus
        assert 0 <= poor_reward <= 1.2
    
    def test_adaptive_optimizer_initialization(self):
        """Test adaptive batch optimizer initialization."""
        optimizer = AdaptiveBatchOptimizer(
            initial_batch_size=64,
            min_batch_size=8,
            max_batch_size=512,
            strategy=OptimizationStrategy.QUANTUM_ANNEALING
        )
        
        assert optimizer.initial_batch_size == 64
        assert optimizer.min_batch_size == 8
        assert optimizer.max_batch_size == 512
        assert optimizer.strategy == OptimizationStrategy.QUANTUM_ANNEALING
    
    def test_adaptive_optimizer_mock_evaluation(self):
        """Test mock evaluation functionality."""
        optimizer = AdaptiveBatchOptimizer()
        
        # Test various batch sizes
        for batch_size in [16, 32, 64, 128, 256]:
            metrics = optimizer._mock_evaluation(batch_size)
            
            assert metrics.batch_size == batch_size
            assert metrics.throughput_samples_per_sec >= 0
            assert 0 <= metrics.memory_utilization_percent <= 100
            assert 0 <= metrics.convergence_rate <= 1
            assert metrics.training_loss >= 0
            assert metrics.gradient_norm >= 0
            assert 0 <= metrics.hpu_utilization_percent <= 100
            assert metrics.latency_ms >= 0
            assert 0 <= metrics.energy_efficiency_score <= 1
    
    def test_adaptive_optimizer_scoring(self):
        """Test scoring function consistency."""
        optimizer = AdaptiveBatchOptimizer()
        
        # Create metrics with known good values
        good_metrics = BatchMetrics(
            batch_size=128,
            throughput_samples_per_sec=1000.0,
            memory_utilization_percent=85.0,
            convergence_rate=0.95,
            training_loss=0.1,
            gradient_norm=1.0,
            hpu_utilization_percent=90.0,
            latency_ms=30.0,
            energy_efficiency_score=0.9
        )
        
        # Create metrics with poor values
        poor_metrics = BatchMetrics(
            batch_size=32,
            throughput_samples_per_sec=100.0,
            memory_utilization_percent=20.0,
            convergence_rate=0.3,
            training_loss=2.0,
            gradient_norm=10.0,
            hpu_utilization_percent=30.0,
            latency_ms=200.0,
            energy_efficiency_score=0.2
        )
        
        good_score = optimizer._get_score(good_metrics)
        poor_score = optimizer._get_score(poor_metrics)
        
        assert good_score > poor_score
        assert 0 <= good_score <= 1
        assert 0 <= poor_score <= 1
    
    @pytest.mark.parametrize("strategy", [
        OptimizationStrategy.QUANTUM_ANNEALING,
        OptimizationStrategy.REINFORCEMENT_LEARNING,
        OptimizationStrategy.BINARY_SEARCH,
        OptimizationStrategy.GOLDEN_RATIO
    ])
    def test_optimization_strategies(self, strategy):
        """Test different optimization strategies."""
        optimizer = AdaptiveBatchOptimizer(
            strategy=strategy,
            max_iterations=10  # Keep test fast
        )
        
        result = optimizer.optimize()
        
        assert result.optimal_batch_size >= optimizer.min_batch_size
        assert result.optimal_batch_size <= optimizer.max_batch_size
        assert 0 <= result.confidence_score <= 1
        assert result.iterations_count <= optimizer.max_iterations
        assert result.strategy_used == strategy
        assert len(result.metrics_history) > 0
    
    def test_optimization_result_serialization(self):
        """Test optimization result serialization."""
        optimizer = AdaptiveBatchOptimizer(max_iterations=5)
        result = optimizer.optimize()
        
        # Test serialization
        result_dict = result.to_dict()
        assert 'optimal_batch_size' in result_dict
        assert 'strategy_used' in result_dict
        assert 'metrics_history' in result_dict
        
        # Verify metrics history serialization
        assert len(result_dict['metrics_history']) == len(result.metrics_history)
        for metrics_dict in result_dict['metrics_history']:
            assert 'batch_size' in metrics_dict
            assert 'throughput_samples_per_sec' in metrics_dict


class TestQuantumHybridScheduler:
    """Test suite for quantum-hybrid scheduling."""
    
    def test_node_creation(self):
        """Test node creation and resource management."""
        node = create_gaudi3_node("node-1", hpu_count=8, memory_gb=96)
        
        assert node.node_id == "node-1"
        assert node.node_type == "gaudi3"
        assert node.total_resources[ResourceType.HPU] == 8.0
        assert node.total_resources[ResourceType.MEMORY] == 96.0
        assert node.available_resources[ResourceType.HPU] == 8.0
    
    def test_node_resource_allocation(self):
        """Test node resource allocation and release."""
        node = create_gaudi3_node("node-1")
        
        # Test resource allocation
        required_resources = {
            ResourceType.HPU: 2.0,
            ResourceType.MEMORY: 16.0
        }
        
        # Should be able to allocate
        assert node.can_accommodate(required_resources)
        assert node.allocate_resources(required_resources)
        
        # Check remaining resources
        assert node.available_resources[ResourceType.HPU] == 6.0
        assert node.available_resources[ResourceType.MEMORY] == 80.0
        
        # Test resource release
        node.release_resources(required_resources)
        assert node.available_resources[ResourceType.HPU] == 8.0
        assert node.available_resources[ResourceType.MEMORY] == 96.0
    
    def test_node_resource_overallocation(self):
        """Test node behavior when resources are overallocated."""
        node = create_gaudi3_node("node-1", hpu_count=2, memory_gb=16)
        
        # Try to allocate more than available
        excessive_resources = {
            ResourceType.HPU: 5.0,
            ResourceType.MEMORY: 32.0
        }
        
        assert not node.can_accommodate(excessive_resources)
        assert not node.allocate_resources(excessive_resources)
        
        # Resources should remain unchanged
        assert node.available_resources[ResourceType.HPU] == 2.0
        assert node.available_resources[ResourceType.MEMORY] == 16.0
    
    def test_task_creation(self):
        """Test task creation and dependency handling."""
        task = create_training_task(
            task_id="task-1",
            name="Test Training Task",
            hpu_requirement=2,
            memory_gb=16,
            estimated_hours=2.0,
            priority=TaskPriority.HIGH
        )
        
        assert task.task_id == "task-1"
        assert task.name == "Test Training Task"
        assert task.priority == TaskPriority.HIGH
        assert task.required_resources[ResourceType.HPU] == 2.0
        assert task.required_resources[ResourceType.MEMORY] == 16.0
        assert task.estimated_duration == 2.0 * 3600  # 2 hours in seconds
    
    def test_task_dependency_checking(self):
        """Test task dependency satisfaction."""
        task = create_training_task("task-1", "Test Task")
        task.dependencies = ["dep-1", "dep-2"]
        
        # No dependencies satisfied
        assert not task.is_ready_to_schedule(set())
        
        # Partial dependencies satisfied
        assert not task.is_ready_to_schedule({"dep-1"})
        
        # All dependencies satisfied
        assert task.is_ready_to_schedule({"dep-1", "dep-2"})
        
        # Extra dependencies satisfied (should still work)
        assert task.is_ready_to_schedule({"dep-1", "dep-2", "dep-3"})
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = QuantumHybridScheduler(
            cluster_name="test-cluster",
            scheduling_interval=1.0,
            enable_quantum_optimization=True
        )
        
        assert scheduler.cluster_name == "test-cluster"
        assert scheduler.scheduling_interval == 1.0
        assert scheduler.enable_quantum_optimization
        assert not scheduler.running
        assert len(scheduler.nodes) == 0
        assert len(scheduler.tasks) == 0
    
    def test_scheduler_node_management(self):
        """Test scheduler node addition and removal."""
        scheduler = QuantumHybridScheduler()
        
        # Add nodes
        node1 = create_gaudi3_node("node-1")
        node2 = create_gaudi3_node("node-2")
        
        scheduler.add_node(node1)
        scheduler.add_node(node2)
        
        assert len(scheduler.nodes) == 2
        assert "node-1" in scheduler.nodes
        assert "node-2" in scheduler.nodes
        
        # Remove node
        scheduler.remove_node("node-1")
        assert len(scheduler.nodes) == 1
        assert "node-1" not in scheduler.nodes
        assert "node-2" in scheduler.nodes
    
    def test_scheduler_task_submission(self):
        """Test task submission to scheduler."""
        scheduler = QuantumHybridScheduler()
        
        task1 = create_training_task("task-1", "Task 1")
        task2 = create_training_task("task-2", "Task 2")
        
        # Submit tasks
        assert scheduler.submit_task(task1)
        assert scheduler.submit_task(task2)
        
        assert len(scheduler.tasks) == 2
        assert len(scheduler.pending_tasks) == 2
        assert "task-1" in scheduler.tasks
        assert "task-2" in scheduler.tasks
    
    def test_scheduler_task_cancellation(self):
        """Test task cancellation."""
        scheduler = QuantumHybridScheduler()
        
        task = create_training_task("task-1", "Test Task")
        scheduler.submit_task(task)
        
        # Cancel task
        assert scheduler.cancel_task("task-1")
        assert scheduler.tasks["task-1"].state.value == "cancelled"
        
        # Try to cancel non-existent task
        assert not scheduler.cancel_task("non-existent")
    
    def test_scheduler_node_suitability(self):
        """Test node suitability calculation."""
        scheduler = QuantumHybridScheduler()
        
        # Create node and task
        node = create_gaudi3_node("node-1")
        task = create_training_task("task-1", "Test Task", hpu_requirement=2, memory_gb=16)
        
        # Calculate suitability
        suitability = scheduler._calculate_node_suitability(task, node)
        
        assert isinstance(suitability, float)
        assert suitability >= 0.0
        
        # Node with insufficient resources should have lower suitability
        small_node = create_gaudi3_node("small-node", hpu_count=1, memory_gb=8)
        large_task = create_training_task("large-task", "Large Task", hpu_requirement=4, memory_gb=32)
        
        small_suitability = scheduler._calculate_node_suitability(large_task, small_node)
        normal_suitability = scheduler._calculate_node_suitability(task, node)
        
        # Normal task on normal node should be more suitable
        assert normal_suitability >= small_suitability
    
    def test_scheduler_cluster_status(self):
        """Test cluster status reporting."""
        scheduler = QuantumHybridScheduler()
        
        # Add some nodes and tasks
        node = create_gaudi3_node("node-1")
        scheduler.add_node(node)
        
        task = create_training_task("task-1", "Test Task")
        scheduler.submit_task(task)
        
        status = scheduler.get_cluster_status()
        
        assert 'cluster_name' in status
        assert 'timestamp' in status
        assert 'nodes' in status
        assert 'tasks' in status
        assert len(status['nodes']) == 1
        assert status['tasks']['total'] == 1
    
    def test_scheduler_task_status(self):
        """Test individual task status reporting."""
        scheduler = QuantumHybridScheduler()
        
        task = create_training_task("task-1", "Test Task")
        scheduler.submit_task(task)
        
        status = scheduler.get_task_status("task-1")
        
        assert status is not None
        assert status['task_id'] == "task-1"
        assert status['name'] == "Test Task"
        assert status['state'] == "pending"
        
        # Non-existent task
        assert scheduler.get_task_status("non-existent") is None


class TestErrorRecovery:
    """Test suite for error recovery system."""
    
    def test_checkpoint_manager_initialization(self):
        """Test checkpoint manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SmartCheckpointManager(
                checkpoint_dir=tmpdir,
                max_checkpoints=5,
                verification_enabled=True
            )
            
            assert manager.checkpoint_dir == Path(tmpdir)
            assert manager.max_checkpoints == 5
            assert manager.verification_enabled
            assert len(manager.checkpoints) == 0
    
    def test_checkpoint_saving_without_torch(self):
        """Test checkpoint saving in environments without PyTorch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SmartCheckpointManager(tmpdir)
            
            # Mock model and optimizer
            mock_model = Mock()
            mock_model.state_dict.return_value = {"layer1.weight": "mock_weights"}
            
            mock_optimizer = Mock()
            mock_optimizer.state_dict.return_value = {"lr": 0.01, "momentum": 0.9}
            
            # Save checkpoint
            checkpoint_info = manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                epoch=1,
                step=100,
                metrics={"loss": 0.5, "accuracy": 0.8}
            )
            
            assert checkpoint_info.epoch == 1
            assert checkpoint_info.step == 100
            assert checkpoint_info.metrics["loss"] == 0.5
            assert checkpoint_info.filepath.exists()
    
    def test_checkpoint_loading_without_torch(self):
        """Test checkpoint loading in environments without PyTorch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SmartCheckpointManager(tmpdir, verification_enabled=False)
            
            # Create and save checkpoint
            mock_model = Mock()
            mock_model.state_dict.return_value = {"layer1.weight": "mock_weights"}
            mock_optimizer = Mock()
            mock_optimizer.state_dict.return_value = {"lr": 0.01}
            
            checkpoint_info = manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                epoch=1,
                step=100
            )
            
            # Load checkpoint
            loaded_data = manager.load_checkpoint(
                checkpoint_info.checkpoint_id,
                model=mock_model,
                optimizer=mock_optimizer,
                strict=False
            )
            
            assert loaded_data['epoch'] == 1
            assert loaded_data['step'] == 100
            assert 'model_state_dict' in loaded_data
            assert 'optimizer_state_dict' in loaded_data
    
    def test_checkpoint_verification(self):
        """Test checkpoint verification functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SmartCheckpointManager(tmpdir, verification_enabled=True)
            
            mock_model = Mock()
            mock_model.state_dict.return_value = {"param": "value"}
            
            checkpoint_info = manager.save_checkpoint(
                model=mock_model,
                optimizer=None,
                epoch=1,
                step=100
            )
            
            # Checkpoint should be verified
            assert checkpoint_info.is_verified
            assert checkpoint_info.checkpoint_id in manager.verified_checkpoints
    
    def test_checkpoint_cleanup(self):
        """Test automatic checkpoint cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SmartCheckpointManager(tmpdir, max_checkpoints=2)
            
            mock_model = Mock()
            mock_model.state_dict.return_value = {"param": "value"}
            
            # Save multiple checkpoints
            for i in range(5):
                manager.save_checkpoint(
                    model=mock_model,
                    optimizer=None,
                    epoch=i,
                    step=i * 100
                )
                time.sleep(0.01)  # Ensure different timestamps
            
            # Should only keep max_checkpoints
            assert len(manager.checkpoints) == 2
            
            # Should keep the most recent ones
            epochs = [info.epoch for info in manager.checkpoints.values()]
            assert 3 in epochs
            assert 4 in epochs
    
    def test_error_recovery_initialization(self):
        """Test error recovery system initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = SmartCheckpointManager(tmpdir)
            recovery = AdvancedErrorRecovery(
                checkpoint_manager=checkpoint_manager,
                max_recovery_attempts=3
            )
            
            assert recovery.checkpoint_manager == checkpoint_manager
            assert recovery.max_recovery_attempts == 3
            assert len(recovery.error_history) == 0
            assert len(recovery.current_errors) == 0
    
    def test_error_type_detection(self):
        """Test automatic error type detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = create_error_recovery_system(tmpdir)
            
            # Test memory error detection
            memory_error = MemoryError("Out of memory")
            error_context = recovery._create_error_context(memory_error, None, None)
            assert error_context.error_type == ErrorType.MEMORY_ERROR
            
            # Test general error detection
            runtime_error = RuntimeError("HPU driver failed")
            error_context = recovery._create_error_context(runtime_error, None, None)
            assert error_context.error_type == ErrorType.HPU_DRIVER_ERROR
            
            # Test unknown error
            value_error = ValueError("Invalid parameter")
            error_context = recovery._create_error_context(value_error, None, None)
            assert error_context.error_type == ErrorType.UNKNOWN_ERROR
    
    def test_error_severity_determination(self):
        """Test error severity determination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = create_error_recovery_system(tmpdir)
            
            # Critical error
            severity = recovery._determine_severity(ErrorType.MEMORY_ERROR, "Out of memory")
            assert severity.value == "critical"
            
            # High severity error
            severity = recovery._determine_severity(ErrorType.DISTRIBUTED_ERROR, "Network failed")
            assert severity.value == "high"
            
            # Medium severity error
            severity = recovery._determine_severity(ErrorType.DATA_LOADING_ERROR, "Data loading failed")
            assert severity.value == "medium"
    
    def test_error_handling_workflow(self):
        """Test complete error handling workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = create_error_recovery_system(tmpdir)
            
            # Create a mock error
            test_error = RuntimeError("Test error for recovery")
            
            # Handle the error
            recovery_success = recovery.handle_error(
                error=test_error,
                context={"component": "trainer"},
                error_type=ErrorType.MODEL_ERROR
            )
            
            # Should attempt recovery
            assert len(recovery.error_history) == 1
            assert recovery.error_history[0].error_type == ErrorType.MODEL_ERROR
    
    def test_error_summary_generation(self):
        """Test error summary generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = create_error_recovery_system(tmpdir)
            
            # Initially no errors
            summary = recovery.get_error_summary()
            assert summary["total_errors"] == 0
            assert summary["recovery_rate"] == 1.0
            
            # Add some errors
            for i in range(3):
                recovery.handle_error(RuntimeError(f"Test error {i}"))
            
            summary = recovery.get_error_summary()
            assert summary["total_errors"] == 3
            assert "error_types" in summary
            assert "recent_errors" in summary


class TestIntegration:
    """Integration tests for research components."""
    
    def test_optimizer_with_scheduler_integration(self):
        """Test integration between batch optimizer and scheduler."""
        # Create scheduler
        scheduler = QuantumHybridScheduler()
        
        # Add nodes
        for i in range(3):
            node = create_gaudi3_node(f"node-{i}")
            scheduler.add_node(node)
        
        # Create batch optimizer
        optimizer = AdaptiveBatchOptimizer(
            strategy=OptimizationStrategy.QUANTUM_ANNEALING,
            max_iterations=5
        )
        
        # Run optimization
        result = optimizer.optimize()
        
        # Create training task with optimal batch size
        task = create_training_task(
            task_id="optimized-task",
            name="Optimized Training Task",
            hpu_requirement=min(2, result.optimal_batch_size // 32)  # Reasonable HPU count
        )
        
        # Submit to scheduler
        assert scheduler.submit_task(task)
        
        # Verify task is scheduled
        assert len(scheduler.tasks) == 1
        assert len(scheduler.pending_tasks) == 1
    
    def test_error_recovery_with_checkpoints(self):
        """Test error recovery integration with checkpoint management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create error recovery system
            recovery = create_error_recovery_system(tmpdir)
            
            # Create mock checkpoint
            mock_model = Mock()
            mock_model.state_dict.return_value = {"param": "value"}
            
            recovery.checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=None,
                epoch=1,
                step=100,
                metrics={"loss": 0.5}
            )
            
            # Simulate checkpoint error and recovery
            checkpoint_error = RuntimeError("Checkpoint corrupted")
            recovery_success = recovery.handle_error(
                error=checkpoint_error,
                error_type=ErrorType.CHECKPOINT_ERROR
            )
            
            # Should have attempted recovery
            assert len(recovery.error_history) == 1
            assert recovery.error_history[0].error_type == ErrorType.CHECKPOINT_ERROR
    
    def test_full_system_workflow(self):
        """Test complete system workflow with all components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create scheduler
            scheduler = QuantumHybridScheduler(cluster_name="test-cluster")
            
            # 2. Add nodes
            for i in range(2):
                node = create_gaudi3_node(f"node-{i}")
                scheduler.add_node(node)
            
            # 3. Create error recovery
            recovery = create_error_recovery_system(tmpdir)
            
            # 4. Create batch optimizer
            optimizer = AdaptiveBatchOptimizer(max_iterations=3)
            
            # 5. Run optimization
            optimization_result = optimizer.optimize()
            
            # 6. Create tasks based on optimization
            for i in range(2):
                task = create_training_task(
                    task_id=f"task-{i}",
                    name=f"Training Task {i}",
                    hpu_requirement=1
                )
                scheduler.submit_task(task)
            
            # 7. Simulate error and recovery
            test_error = RuntimeError("System error")
            recovery.handle_error(test_error)
            
            # 8. Verify system state
            cluster_status = scheduler.get_cluster_status()
            error_summary = recovery.get_error_summary()
            
            assert cluster_status["tasks"]["total"] == 2
            assert error_summary["total_errors"] == 1
            assert optimization_result.optimal_batch_size > 0


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for research components."""
    
    def test_batch_optimizer_performance(self):
        """Benchmark batch optimizer performance."""
        start_time = time.time()
        
        optimizer = AdaptiveBatchOptimizer(
            strategy=OptimizationStrategy.QUANTUM_ANNEALING,
            max_iterations=20
        )
        
        result = optimizer.optimize()
        
        duration = time.time() - start_time
        
        # Should complete within reasonable time
        assert duration < 10.0  # 10 seconds max
        assert result.optimization_time_seconds > 0
        assert len(result.metrics_history) <= 20
    
    def test_scheduler_scalability(self):
        """Test scheduler performance with many nodes and tasks."""
        scheduler = QuantumHybridScheduler()
        
        # Add many nodes
        start_time = time.time()
        for i in range(50):
            node = create_gaudi3_node(f"node-{i}")
            scheduler.add_node(node)
        
        node_addition_time = time.time() - start_time
        
        # Add many tasks
        start_time = time.time()
        for i in range(100):
            task = create_training_task(f"task-{i}", f"Task {i}")
            scheduler.submit_task(task)
        
        task_addition_time = time.time() - start_time
        
        # Get status (should be fast)
        start_time = time.time()
        status = scheduler.get_cluster_status()
        status_time = time.time() - start_time
        
        # Performance assertions
        assert node_addition_time < 5.0  # 5 seconds for 50 nodes
        assert task_addition_time < 5.0  # 5 seconds for 100 tasks
        assert status_time < 1.0  # 1 second for status
        assert len(scheduler.nodes) == 50
        assert len(scheduler.tasks) == 100


# Statistical validation tests
class TestStatisticalValidation:
    """Statistical validation of research algorithms."""
    
    def test_quantum_scheduler_randomness(self):
        """Test randomness in quantum scheduler decisions."""
        scheduler = QuantumInspiredScheduler()
        
        # Generate multiple batch sizes with same parameters
        batch_sizes = []
        for _ in range(50):
            batch_size = scheduler.next_batch_size(64, 0.8, (16, 256), 5)
            batch_sizes.append(batch_size)
        
        # Should have some variation
        unique_sizes = set(batch_sizes)
        assert len(unique_sizes) > 1  # Should generate different values
        
        # Should all be in valid range
        assert all(16 <= bs <= 256 for bs in batch_sizes)
    
    def test_rl_optimizer_learning(self):
        """Test that RL optimizer shows learning behavior."""
        rl_optimizer = ReinforcementLearningOptimizer(epsilon=0.1)
        
        # Simulate learning episodes
        state = "test_state"
        
        # Initially Q-table should be empty
        assert state not in rl_optimizer.q_table
        
        # Perform some updates
        for i in range(10):
            action = rl_optimizer.choose_action(state)
            reward = 0.8 + np.random.normal(0, 0.1)  # Noisy positive reward
            next_state = f"next_state_{i}"
            
            rl_optimizer.update_q_table(state, action, reward, next_state, done=(i==9))
            state = next_state
        
        # Q-table should have been updated
        assert len(rl_optimizer.q_table) > 0
    
    def test_optimization_convergence(self):
        """Test that optimization algorithms show convergence."""
        optimizer = AdaptiveBatchOptimizer(
            strategy=OptimizationStrategy.QUANTUM_ANNEALING,
            max_iterations=30,
            convergence_threshold=0.01
        )
        
        result = optimizer.optimize()
        
        # Should show improvement over time
        if len(result.metrics_history) > 5:
            early_scores = [optimizer._get_score(m) for m in result.metrics_history[:5]]
            late_scores = [optimizer._get_score(m) for m in result.metrics_history[-5:]]
            
            early_avg = np.mean(early_scores)
            late_avg = np.mean(late_scores)
            
            # Later scores should generally be better (or at least not much worse)
            assert late_avg >= early_avg - 0.1  # Allow for some variation


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])