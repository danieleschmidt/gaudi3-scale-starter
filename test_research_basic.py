#!/usr/bin/env python3
"""Basic test runner for research components without external dependencies."""

import sys
import os
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def mock_numpy():
    """Mock numpy functionality for testing without numpy."""
    class MockNp:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0
        
        @staticmethod
        def std(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        @staticmethod
        def polyfit(x, y, degree):
            # Simple linear fit for degree 1
            if degree == 1 and len(x) >= 2:
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(xi * yi for xi, yi in zip(x, y))
                sum_x2 = sum(xi * xi for xi in x)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                intercept = (sum_y - slope * sum_x) / n
                return [slope, intercept]
            return [0.0, 0.0]
        
        @staticmethod
        def correlate(a, b, mode='full'):
            # Simple correlation mock
            if len(a) != len(b):
                return [0.0]
            return [sum(ai * bi for ai, bi in zip(a, b))]
        
        @staticmethod
        def arange(n):
            return list(range(n))
        
        @staticmethod
        def var(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            return sum((x - mean_val) ** 2 for x in values) / len(values)
        
        @staticmethod
        def max(values):
            return max(values) if values else 0.0
        
        @staticmethod
        def min(values):
            return min(values) if values else 0.0
        
        @staticmethod
        def argmax(values):
            if not values:
                return 0
            max_val = max(values)
            return values.index(max_val)
        
        class random:
            @staticmethod
            def random():
                import random
                return random.random()
            
            @staticmethod
            def normal(mean=0, std=1):
                import random
                return random.gauss(mean, std)
            
            @staticmethod
            def uniform(low, high):
                import random
                return random.uniform(low, high)
            
            @staticmethod
            def choice(choices):
                import random
                return random.choice(choices)
            
            @staticmethod
            def randint(low, high):
                import random
                return random.randint(low, high)
    
    return MockNp()

# Mock numpy before importing research components
import random
import math
sys.modules['numpy'] = mock_numpy()

def test_adaptive_batch_optimizer():
    """Test adaptive batch optimizer functionality."""
    print("Testing Adaptive Batch Optimizer...")
    
    try:
        from gaudi3_scale.research.adaptive_batch_optimizer import (
            AdaptiveBatchOptimizer, OptimizationStrategy, BatchMetrics
        )
        
        # Test BatchMetrics creation
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
        print("  ✓ BatchMetrics creation and validation")
        
        # Test optimizer initialization
        optimizer = AdaptiveBatchOptimizer(
            initial_batch_size=32,
            min_batch_size=8,
            max_batch_size=256,
            strategy=OptimizationStrategy.QUANTUM_ANNEALING,
            max_iterations=5  # Keep test fast
        )
        
        assert optimizer.initial_batch_size == 32
        assert optimizer.strategy == OptimizationStrategy.QUANTUM_ANNEALING
        print("  ✓ AdaptiveBatchOptimizer initialization")
        
        # Test optimization
        result = optimizer.optimize()
        
        assert result.optimal_batch_size >= optimizer.min_batch_size
        assert result.optimal_batch_size <= optimizer.max_batch_size
        assert len(result.metrics_history) > 0
        print("  ✓ Optimization execution")
        
        print("✓ Adaptive Batch Optimizer tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Adaptive Batch Optimizer test failed: {e}")
        return False

def test_quantum_hybrid_scheduler():
    """Test quantum hybrid scheduler functionality."""
    print("Testing Quantum Hybrid Scheduler...")
    
    try:
        from gaudi3_scale.research.quantum_hybrid_scheduler import (
            QuantumHybridScheduler, create_gaudi3_node, create_training_task,
            TaskPriority, ResourceType
        )
        
        # Test node creation
        node = create_gaudi3_node("test-node", hpu_count=8, memory_gb=96)
        assert node.node_id == "test-node"
        assert node.total_resources[ResourceType.HPU] == 8.0
        print("  ✓ Node creation")
        
        # Test task creation
        task = create_training_task(
            "test-task", "Test Task", 
            hpu_requirement=2, memory_gb=16,
            priority=TaskPriority.HIGH
        )
        assert task.task_id == "test-task"
        assert task.priority == TaskPriority.HIGH
        print("  ✓ Task creation")
        
        # Test scheduler
        scheduler = QuantumHybridScheduler("test-cluster")
        scheduler.add_node(node)
        assert len(scheduler.nodes) == 1
        
        scheduler.submit_task(task)
        assert len(scheduler.tasks) == 1
        print("  ✓ Scheduler operations")
        
        # Test status reporting
        status = scheduler.get_cluster_status()
        assert 'cluster_name' in status
        assert status['tasks']['total'] == 1
        print("  ✓ Status reporting")
        
        print("✓ Quantum Hybrid Scheduler tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Quantum Hybrid Scheduler test failed: {e}")
        return False

def test_error_recovery():
    """Test error recovery system functionality."""
    print("Testing Error Recovery System...")
    
    try:
        from gaudi3_scale.research.error_recovery import (
            SmartCheckpointManager, create_error_recovery_system, ErrorType
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test checkpoint manager
            manager = SmartCheckpointManager(
                checkpoint_dir=tmpdir,
                max_checkpoints=3,
                verification_enabled=False  # Disable for simple test
            )
            
            # Mock model and optimizer
            class MockModel:
                def state_dict(self):
                    return {"layer1.weight": "mock_weights", "layer1.bias": "mock_bias"}
            
            class MockOptimizer:
                def state_dict(self):
                    return {"lr": 0.01, "momentum": 0.9}
                
                def load_state_dict(self, state_dict):
                    pass
            
            mock_model = MockModel()
            mock_optimizer = MockOptimizer()
            
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
            assert checkpoint_info.filepath.exists()
            print("  ✓ Checkpoint saving")
            
            # Load checkpoint
            loaded_data = manager.load_checkpoint(
                checkpoint_info.checkpoint_id,
                model=mock_model,
                optimizer=mock_optimizer,
                strict=False
            )
            
            assert loaded_data['epoch'] == 1
            assert 'model_state_dict' in loaded_data
            print("  ✓ Checkpoint loading")
            
            # Test error recovery
            recovery = create_error_recovery_system(tmpdir)
            
            # Handle an error
            test_error = RuntimeError("Test error")
            recovery.handle_error(test_error, error_type=ErrorType.MODEL_ERROR)
            
            assert len(recovery.error_history) == 1
            assert recovery.error_history[0].error_type == ErrorType.MODEL_ERROR
            print("  ✓ Error handling")
        
        print("✓ Error Recovery System tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Error Recovery System test failed: {e}")
        return False

def test_performance_auto_scaler():
    """Test performance auto-scaler functionality."""
    print("Testing Performance Auto-Scaler...")
    
    try:
        from gaudi3_scale.research.performance_auto_scaler import (
            PerformanceAutoScaler, create_performance_auto_scaler,
            ScalingPolicy, PerformanceMetrics, ScalingDirection
        )
        
        # Test scaling policy
        policy = ScalingPolicy(
            name="test_policy",
            min_nodes=1,
            max_nodes=10,
            target_hpu_utilization=70.0
        )
        
        assert policy.name == "test_policy"
        assert policy.min_nodes == 1
        print("  ✓ Scaling policy creation")
        
        # Test auto-scaler
        scaler = PerformanceAutoScaler(
            scaling_policy=policy,
            cluster_name="test-cluster",
            monitoring_interval=1.0  # Fast for testing
        )
        
        assert scaler.cluster_name == "test-cluster"
        assert scaler.current_nodes == policy.min_nodes
        print("  ✓ Auto-scaler initialization")
        
        # Test metrics
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            hpu_utilization=85.0,
            memory_utilization=75.0,
            throughput_samples_per_sec=800.0,
            queue_length=5,
            response_time_ms=50.0,
            error_rate=0.01,
            cost_per_sample=0.001,
            energy_efficiency=0.8
        )
        
        scaler.add_metrics(metrics)
        assert len(scaler.metrics_history) == 1
        print("  ✓ Metrics handling")
        
        # Test scaling decision
        scaling_event = scaler._make_scaling_decision(metrics)
        assert scaling_event.direction in [ScalingDirection.SCALE_UP, ScalingDirection.MAINTAIN, ScalingDirection.SCALE_DOWN]
        print("  ✓ Scaling decision")
        
        # Test status
        status = scaler.get_cluster_status()
        assert 'cluster_name' in status
        assert 'current_nodes' in status
        print("  ✓ Status reporting")
        
        print("✓ Performance Auto-Scaler tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Performance Auto-Scaler test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("Testing Component Integration...")
    
    try:
        # Import all components
        from gaudi3_scale.research.adaptive_batch_optimizer import AdaptiveBatchOptimizer
        from gaudi3_scale.research.quantum_hybrid_scheduler import QuantumHybridScheduler, create_gaudi3_node, create_training_task
        from gaudi3_scale.research.performance_auto_scaler import create_performance_auto_scaler
        
        # Create components
        optimizer = AdaptiveBatchOptimizer(max_iterations=3)
        scheduler = QuantumHybridScheduler()
        auto_scaler = create_performance_auto_scaler()
        
        # Add node to scheduler
        node = create_gaudi3_node("integration-node")
        scheduler.add_node(node)
        
        # Run optimization
        result = optimizer.optimize()
        
        # Create task based on optimization
        task = create_training_task(
            "integration-task",
            "Integration Test Task",
            hpu_requirement=1
        )
        scheduler.submit_task(task)
        
        # Verify integration
        assert len(scheduler.nodes) == 1
        assert len(scheduler.tasks) == 1
        assert result.optimal_batch_size > 0
        assert auto_scaler.current_nodes >= 1
        
        print("  ✓ Component integration successful")
        print("✓ Integration tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TERRAGON RESEARCH FRAMEWORK - BASIC TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_adaptive_batch_optimizer,
        test_quantum_hybrid_scheduler,
        test_error_recovery,
        test_performance_auto_scaler,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print()
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())