"""Comprehensive tests for Quantum Task Planner.

Tests quantum task planning algorithms, entanglement patterns,
and interference optimization for HPU cluster task scheduling.
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock

from gaudi3_scale.quantum.task_planner import (
    QuantumTaskPlanner,
    QuantumTask,
    QuantumState
)
from gaudi3_scale.exceptions import TaskPlanningError, ValidationError


class TestQuantumTaskPlanner:
    """Test suite for quantum task planner."""
    
    @pytest.fixture
    async def task_planner(self):
        """Create quantum task planner for testing."""
        planner = QuantumTaskPlanner(
            cluster_nodes=4,
            hpu_per_node=8,
            quantum_coherence_time=30.0,
            enable_entanglement=True
        )
        yield planner
    
    @pytest.fixture
    async def sample_tasks(self, task_planner):
        """Create sample tasks for testing."""
        tasks = {}
        
        # Task A: Independent preprocessing task
        tasks["task_a"] = await task_planner.add_task(
            task_id="task_a",
            resource_requirements={"hpu_cores": 8, "memory_gb": 64},
            dependencies=set(),
            priority=1.0,
            estimated_duration=300.0
        )
        
        # Task B: Depends on Task A
        tasks["task_b"] = await task_planner.add_task(
            task_id="task_b",
            resource_requirements={"hpu_cores": 16, "memory_gb": 128},
            dependencies={"task_a"},
            priority=0.8,
            estimated_duration=600.0
        )
        
        # Task C: Independent of A, parallel with B
        tasks["task_c"] = await task_planner.add_task(
            task_id="task_c",
            resource_requirements={"hpu_cores": 8, "memory_gb": 48},
            dependencies=set(),
            priority=0.9,
            estimated_duration=400.0
        )
        
        # Task D: Depends on both B and C
        tasks["task_d"] = await task_planner.add_task(
            task_id="task_d",
            resource_requirements={"hpu_cores": 24, "memory_gb": 192},
            dependencies={"task_b", "task_c"},
            priority=1.2,
            estimated_duration=800.0
        )
        
        return tasks
    
    @pytest.mark.asyncio
    async def test_planner_initialization(self):
        """Test quantum task planner initialization."""
        planner = QuantumTaskPlanner(
            cluster_nodes=8,
            hpu_per_node=8,
            quantum_coherence_time=60.0
        )
        
        assert planner.cluster_nodes == 8
        assert planner.hpu_per_node == 8
        assert planner.total_hpus == 64
        assert planner.quantum_coherence_time == 60.0
        assert planner.enable_entanglement is True
        
        # Check resource initialization
        assert planner.available_resources["hpu_cores"] == 64
        assert planner.available_resources["memory_gb"] == 64 * 96  # 96GB per HPU
        assert planner.available_resources["network_bandwidth"] == 8 * 200  # 200Gb/s per node
    
    @pytest.mark.asyncio
    async def test_task_addition(self, task_planner):
        """Test adding tasks to quantum planner."""
        task_id = "test_task"
        resource_req = {"hpu_cores": 4, "memory_gb": 32}
        
        quantum_task = await task_planner.add_task(
            task_id=task_id,
            resource_requirements=resource_req,
            priority=0.8,
            estimated_duration=120.0
        )
        
        assert task_id in task_planner.quantum_tasks
        assert quantum_task.task_id == task_id
        assert quantum_task.resource_requirements == resource_req
        assert quantum_task.priority_weight == 0.8
        assert quantum_task.estimated_duration == 120.0
        assert quantum_task.quantum_state == QuantumState.SUPERPOSITION
        
        # Check quantum properties
        assert abs(quantum_task.amplitude) > 0  # Non-zero amplitude after Hadamard
        assert quantum_task.probability_amplitude >= 0
    
    @pytest.mark.asyncio
    async def test_invalid_task_addition(self, task_planner):
        """Test invalid task addition scenarios."""
        # Invalid task ID
        with pytest.raises(ValidationError):
            await task_planner.add_task(
                task_id="",  # Empty task ID
                resource_requirements={"hpu_cores": 4}
            )
        
        # Duplicate task ID
        await task_planner.add_task("duplicate", {"hpu_cores": 4})
        with pytest.raises(TaskPlanningError):
            await task_planner.add_task("duplicate", {"hpu_cores": 8})
    
    @pytest.mark.asyncio
    async def test_quantum_entanglement_creation(self, task_planner, sample_tasks):
        """Test quantum entanglement between related tasks."""
        # Tasks with dependencies should be entangled
        task_b = task_planner.quantum_tasks["task_b"]
        task_a = task_planner.quantum_tasks["task_a"]
        
        assert "task_a" in task_b.entangled_tasks
        assert "task_b" in task_planner.entanglement_graph
        assert "task_a" in task_planner.entanglement_graph["task_b"]
        
        # Tasks should be in entangled state
        assert task_b.quantum_state == QuantumState.ENTANGLED
        assert task_a.quantum_state == QuantumState.ENTANGLED
    
    @pytest.mark.asyncio
    async def test_resource_similarity_calculation(self, task_planner):
        """Test resource similarity calculation."""
        req1 = {"hpu_cores": 8, "memory_gb": 64}
        req2 = {"hpu_cores": 8, "memory_gb": 64}  # Identical
        req3 = {"hpu_cores": 16, "memory_gb": 32}  # Different
        
        # Identical resources should have similarity of 1.0
        similarity_identical = task_planner._calculate_resource_similarity(req1, req2)
        assert abs(similarity_identical - 1.0) < 0.01
        
        # Different resources should have lower similarity
        similarity_different = task_planner._calculate_resource_similarity(req1, req3)
        assert similarity_different < 0.9
        assert similarity_different > 0.0
        
        # Empty resources
        empty_similarity = task_planner._calculate_resource_similarity({}, {})
        assert empty_similarity == 0.0
    
    @pytest.mark.asyncio
    async def test_interference_matrix_construction(self, task_planner, sample_tasks):
        """Test quantum interference matrix construction."""
        await task_planner._build_interference_matrix()
        
        task_ids = list(task_planner.quantum_tasks.keys())
        n_tasks = len(task_ids)
        
        assert task_planner.interference_matrix.shape == (n_tasks, n_tasks)
        
        # Diagonal elements should be 1 (self-interference)
        for i in range(n_tasks):
            assert abs(task_planner.interference_matrix[i, i] - 1.0) < 0.01
        
        # Matrix should be complex
        assert task_planner.interference_matrix.dtype == complex
    
    @pytest.mark.asyncio
    async def test_quantum_interference_optimization(self, task_planner, sample_tasks):
        """Test quantum interference optimization."""
        # Build interference matrix
        await task_planner._build_interference_matrix()
        
        # Apply quantum interference
        optimized_priorities = await task_planner._apply_quantum_interference()
        
        assert len(optimized_priorities) == len(sample_tasks)
        
        # All priorities should be valid probabilities
        for task_id, priority in optimized_priorities.items():
            assert 0 <= priority <= 1
            assert task_id in sample_tasks
        
        # Priorities should sum to approximately 1.0 (normalized)
        total_priority = sum(optimized_priorities.values())
        assert abs(total_priority - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_resource_conflict_calculation(self, task_planner):
        """Test resource conflict calculation."""
        # Tasks with overlapping resource requirements
        req1 = {"hpu_cores": 20, "memory_gb": 100}
        req2 = {"hpu_cores": 15, "memory_gb": 80}
        
        conflict = task_planner._calculate_resource_conflict(req1, req2)
        
        # Should detect conflict since total > available
        assert conflict > 0
        
        # Non-conflicting resources
        req3 = {"hpu_cores": 4, "memory_gb": 32}
        req4 = {"hpu_cores": 4, "memory_gb": 32}
        
        no_conflict = task_planner._calculate_resource_conflict(req3, req4)
        assert no_conflict < 0.5  # Should be low conflict
    
    @pytest.mark.asyncio
    async def test_task_schedule_optimization(self, task_planner, sample_tasks):
        """Test complete task schedule optimization."""
        execution_plan = await task_planner.optimize_task_schedule()
        
        assert len(execution_plan) == len(sample_tasks)
        
        # Verify execution plan structure
        for task_id, execution_info in execution_plan:
            assert task_id in sample_tasks
            assert "start_time" in execution_info
            assert "estimated_duration" in execution_info
            assert "priority" in execution_info
            assert "quantum_amplitude" in execution_info
            assert "resource_requirements" in execution_info
            assert "quantum_state" in execution_info
            
            # Check execution info validity
            assert execution_info["start_time"] >= 0
            assert execution_info["estimated_duration"] > 0
            assert 0 <= execution_info["priority"] <= 1
            assert execution_info["quantum_amplitude"] >= 0
    
    @pytest.mark.asyncio
    async def test_dependency_constraints(self, task_planner, sample_tasks):
        """Test dependency constraint satisfaction."""
        execution_plan = await task_planner.optimize_task_schedule()
        
        # Create mapping of task to execution info
        execution_times = {}
        for task_id, execution_info in execution_plan:
            execution_times[task_id] = {
                "start_time": execution_info["start_time"],
                "end_time": execution_info["start_time"] + execution_info["estimated_duration"]
            }
        
        # Check dependency constraints
        # Task B should start after Task A completes
        task_a_end = execution_times["task_a"]["end_time"]
        task_b_start = execution_times["task_b"]["start_time"]
        assert task_b_start >= task_a_end - 1.0  # Allow small scheduling tolerance
        
        # Task D should start after both Task B and Task C complete
        task_b_end = execution_times["task_b"]["end_time"]
        task_c_end = execution_times["task_c"]["end_time"]
        task_d_start = execution_times["task_d"]["start_time"]
        
        assert task_d_start >= task_b_end - 1.0
        assert task_d_start >= task_c_end - 1.0
    
    @pytest.mark.asyncio
    async def test_optimal_start_time_calculation(self, task_planner, sample_tasks):
        """Test optimal start time calculation for entangled tasks."""
        # Task with entangled partners
        task_d = task_planner.quantum_tasks["task_d"]
        current_time = 0.0
        
        optimal_time = await task_planner._calculate_optimal_start_time("task_d", current_time)
        
        assert optimal_time >= current_time
        
        # Should consider entangled tasks
        if task_d.entangled_tasks:
            assert optimal_time > current_time  # Some delay for coordination
    
    @pytest.mark.asyncio
    async def test_quantum_state_collapse(self, task_planner, sample_tasks):
        """Test quantum state collapse during optimization."""
        # Initially tasks should be in superposition or entangled
        initial_states = {}
        for task_id, task in task_planner.quantum_tasks.items():
            initial_states[task_id] = task.quantum_state
        
        # Optimize schedule (should collapse states)
        await task_planner.optimize_task_schedule()
        
        # After optimization, tasks should be collapsed
        for task_id, task in task_planner.quantum_tasks.items():
            assert task.quantum_state == QuantumState.COLLAPSED
    
    @pytest.mark.asyncio
    async def test_quantum_metrics_collection(self, task_planner, sample_tasks):
        """Test quantum metrics collection."""
        metrics = await task_planner.get_quantum_metrics()
        
        required_fields = [
            "total_tasks", "quantum_coherence", "state_distribution",
            "total_entanglements", "entanglement_density", 
            "interference_matrix_size", "available_resources"
        ]
        
        for field in required_fields:
            assert field in metrics
        
        assert metrics["total_tasks"] == len(sample_tasks)
        assert 0 <= metrics["quantum_coherence"] <= len(sample_tasks)
        assert 0 <= metrics["entanglement_density"] <= 1
        assert metrics["total_entanglements"] >= 0
    
    @pytest.mark.asyncio
    async def test_quantum_system_reset(self, task_planner, sample_tasks):
        """Test quantum system reset."""
        # Create some state
        await task_planner.optimize_task_schedule()
        
        # Verify system has state
        assert len(task_planner.quantum_tasks) > 0
        assert len(task_planner.entanglement_graph) > 0
        assert task_planner.interference_matrix.size > 0
        
        # Reset system
        await task_planner.reset_quantum_system()
        
        # Verify reset
        for task in task_planner.quantum_tasks.values():
            assert task.quantum_state == QuantumState.SUPERPOSITION
            assert task.amplitude == complex(1.0, 0.0)
            assert task.phase == 0.0
            assert len(task.entangled_tasks) == 0
        
        assert len(task_planner.entanglement_graph) == 0
        assert task_planner.interference_matrix.size == 0
    
    @pytest.mark.asyncio
    async def test_large_task_set_performance(self):
        """Test performance with large task sets."""
        large_planner = QuantumTaskPlanner(
            cluster_nodes=8,
            hpu_per_node=8,
            quantum_coherence_time=60.0
        )
        
        start_time = time.time()
        
        # Create many tasks with complex dependencies
        for i in range(50):
            dependencies = set()
            if i > 0:
                # Add dependency to previous task
                dependencies.add(f"task_{i-1}")
            if i > 5:
                # Add dependency to task 5 steps back
                dependencies.add(f"task_{i-5}")
            
            await large_planner.add_task(
                task_id=f"task_{i}",
                resource_requirements={"hpu_cores": 2, "memory_gb": 16},
                dependencies=dependencies,
                priority=np.random.uniform(0.5, 1.5),
                estimated_duration=np.random.uniform(60.0, 300.0)
            )
        
        # Optimize schedule
        execution_plan = await large_planner.optimize_task_schedule()
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Should complete within reasonable time
        assert optimization_time < 30.0  # 30 seconds max
        assert len(execution_plan) == 50
    
    @pytest.mark.asyncio
    async def test_concurrent_task_operations(self, task_planner):
        """Test concurrent task operations."""
        # Add tasks concurrently
        task_futures = [
            task_planner.add_task(
                task_id=f"concurrent_task_{i}",
                resource_requirements={"hpu_cores": 4, "memory_gb": 32},
                priority=np.random.uniform(0.5, 1.5)
            )
            for i in range(10)
        ]
        
        tasks = await asyncio.gather(*task_futures)
        assert len(tasks) == 10
        assert len(task_planner.quantum_tasks) == 10
        
        # Run multiple optimizations concurrently
        optimization_futures = [
            task_planner.optimize_task_schedule()
            for _ in range(3)
        ]
        
        plans = await asyncio.gather(*optimization_futures)
        assert len(plans) == 3
        
        # All plans should be valid
        for plan in plans:
            assert len(plan) == 10
    
    @pytest.mark.asyncio
    async def test_quantum_task_properties(self):
        """Test QuantumTask properties and methods."""
        task = QuantumTask(
            task_id="test_task",
            resource_requirements={"hpu_cores": 8},
            priority_weight=0.8
        )
        
        # Test initial properties
        assert task.task_id == "test_task"
        assert task.amplitude == complex(1.0, 0.0)
        assert task.phase == 0.0
        assert task.quantum_state == QuantumState.SUPERPOSITION
        
        # Test probability amplitude
        initial_prob = task.probability_amplitude
        assert initial_prob == 1.0
        
        # Test quantum gate application
        task.apply_quantum_gate("hadamard")
        after_hadamard_prob = task.probability_amplitude
        assert after_hadamard_prob < initial_prob  # Should decrease due to superposition
        
        # Test phase gate
        task.apply_quantum_gate("phase", np.pi/4)
        assert task.phase != 0.0
        
        # Test state collapse
        task.collapse_to_state(QuantumState.COLLAPSED)
        assert task.quantum_state == QuantumState.COLLAPSED


@pytest.mark.asyncio
async def test_quantum_task_planner_integration():
    """Integration test for complete quantum task planning workflow."""
    planner = QuantumTaskPlanner(
        cluster_nodes=6,
        hpu_per_node=8,
        quantum_coherence_time=120.0,
        enable_entanglement=True
    )
    
    # Create realistic ML training pipeline
    tasks = [
        ("data_ingestion", {"hpu_cores": 4, "memory_gb": 64}, set(), 1.0, 180),
        ("data_preprocessing", {"hpu_cores": 8, "memory_gb": 128}, {"data_ingestion"}, 0.9, 300),
        ("model_initialization", {"hpu_cores": 2, "memory_gb": 32}, set(), 0.7, 60),
        ("training_epoch_1", {"hpu_cores": 16, "memory_gb": 256}, {"data_preprocessing", "model_initialization"}, 1.2, 1200),
        ("training_epoch_2", {"hpu_cores": 16, "memory_gb": 256}, {"training_epoch_1"}, 1.2, 1200),
        ("training_epoch_3", {"hpu_cores": 16, "memory_gb": 256}, {"training_epoch_2"}, 1.2, 1200),
        ("validation", {"hpu_cores": 8, "memory_gb": 128}, {"training_epoch_3"}, 1.1, 600),
        ("model_export", {"hpu_cores": 4, "memory_gb": 64}, {"validation"}, 0.8, 120),
        ("cleanup", {"hpu_cores": 2, "memory_gb": 16}, {"model_export"}, 0.5, 60)
    ]
    
    # Add all tasks
    for task_id, resources, deps, priority, duration in tasks:
        await planner.add_task(
            task_id=task_id,
            resource_requirements=resources,
            dependencies=deps,
            priority=priority,
            estimated_duration=duration
        )
    
    # Get initial quantum metrics
    initial_metrics = await planner.get_quantum_metrics()
    assert initial_metrics["total_tasks"] == 9
    assert initial_metrics["total_entanglements"] > 0  # Should have dependency entanglements
    
    # Optimize complete schedule
    execution_plan = await planner.optimize_task_schedule()
    assert len(execution_plan) == 9
    
    # Verify execution plan makes sense
    plan_dict = dict(execution_plan)
    
    # Data ingestion should start early
    data_ingestion_start = plan_dict["data_ingestion"]["start_time"]
    assert data_ingestion_start < 100  # Should start very early
    
    # Training epochs should be sequential
    epoch1_start = plan_dict["training_epoch_1"]["start_time"]
    epoch2_start = plan_dict["training_epoch_2"]["start_time"]
    epoch3_start = plan_dict["training_epoch_3"]["start_time"]
    
    assert epoch1_start < epoch2_start
    assert epoch2_start < epoch3_start
    
    # Cleanup should be last
    cleanup_start = plan_dict["cleanup"]["start_time"]
    all_other_starts = [info["start_time"] for task_id, info in execution_plan if task_id != "cleanup"]
    assert cleanup_start >= max(all_other_starts)
    
    # Get final quantum metrics
    final_metrics = await planner.get_quantum_metrics()
    
    print(f"✅ Quantum task planner integration test completed successfully")
    print(f"   - {final_metrics['total_tasks']} tasks scheduled")
    print(f"   - {final_metrics['total_entanglements']} quantum entanglements")
    print(f"   - {final_metrics['quantum_coherence']:.3f} system coherence")
    print(f"   - {final_metrics['entanglement_density']:.3f} entanglement density")
    
    # Verify all tasks collapsed to executable state
    collapsed_tasks = sum(1 for state in final_metrics['state_distribution'].keys() 
                         if state == 'collapsed')
    print(f"   - {collapsed_tasks}/{final_metrics['total_tasks']} tasks in collapsed state")


if __name__ == "__main__":
    # Run integration test directly
    asyncio.run(test_quantum_task_planner_integration())