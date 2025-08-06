"""Comprehensive tests for Quantum Circuit Simulator.

Tests quantum circuit operations, task dependency modeling,
and quantum state evolution for HPU cluster task management.
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch

from gaudi3_scale.quantum.circuit_simulator import (
    QuantumCircuitSimulator,
    QuantumGate,
    QuantumGateOperation,
    TaskQuantumState
)
from gaudi3_scale.exceptions import QuantumCircuitError, ValidationError


class TestQuantumCircuitSimulator:
    """Test suite for quantum circuit simulator."""
    
    @pytest.fixture
    async def quantum_circuit(self):
        """Create quantum circuit for testing."""
        circuit = QuantumCircuitSimulator(num_qubits=8)
        yield circuit
    
    @pytest.fixture
    async def sample_tasks(self, quantum_circuit):
        """Create sample tasks for testing."""
        tasks = {}
        
        # Add sample tasks with dependencies
        tasks["task_a"] = await quantum_circuit.add_task("task_a", [])
        tasks["task_b"] = await quantum_circuit.add_task("task_b", ["task_a"])
        tasks["task_c"] = await quantum_circuit.add_task("task_c", ["task_a"])
        tasks["task_d"] = await quantum_circuit.add_task("task_d", ["task_b", "task_c"])
        
        return tasks
    
    @pytest.mark.asyncio
    async def test_circuit_initialization(self):
        """Test quantum circuit initialization."""
        circuit = QuantumCircuitSimulator(num_qubits=4)
        
        assert circuit.num_qubits == 4
        assert circuit.state_dim == 16  # 2^4
        assert len(circuit.state_vector) == 16
        assert abs(circuit.state_vector[0] - 1.0) < 1e-10  # Initial state |0000⟩
        assert all(abs(amp) < 1e-10 for amp in circuit.state_vector[1:])
    
    @pytest.mark.asyncio 
    async def test_invalid_circuit_size(self):
        """Test circuit with invalid size."""
        with pytest.raises(ValueError):
            QuantumCircuitSimulator(num_qubits=0)
        
        with pytest.raises(ValueError):
            QuantumCircuitSimulator(num_qubits=33)  # Too large
    
    @pytest.mark.asyncio
    async def test_task_registration(self, quantum_circuit):
        """Test task registration in quantum circuit."""
        task_id = "test_task"
        qubit_index = await quantum_circuit.add_task(task_id)
        
        assert task_id in quantum_circuit.task_qubit_mapping
        assert qubit_index in quantum_circuit.qubit_task_mapping
        assert quantum_circuit.task_qubit_mapping[task_id] == qubit_index
        assert quantum_circuit.qubit_task_mapping[qubit_index] == task_id
        assert task_id in quantum_circuit.task_states
    
    @pytest.mark.asyncio
    async def test_duplicate_task_registration(self, quantum_circuit):
        """Test duplicate task registration error."""
        task_id = "duplicate_task"
        await quantum_circuit.add_task(task_id)
        
        with pytest.raises(ValidationError):
            await quantum_circuit.add_task(task_id)
    
    @pytest.mark.asyncio
    async def test_task_with_dependencies(self, quantum_circuit):
        """Test task registration with dependencies."""
        # Add parent task first
        await quantum_circuit.add_task("parent_task")
        
        # Add child task with dependency
        qubit_index = await quantum_circuit.add_task("child_task", ["parent_task"])
        
        task_state = quantum_circuit.task_states["child_task"]
        assert "parent_task" in task_state.dependencies
        assert task_state.qubit_index == qubit_index
    
    @pytest.mark.asyncio
    async def test_quantum_gate_operations(self, quantum_circuit):
        """Test basic quantum gate operations."""
        await quantum_circuit.add_task("test_task")
        
        # Test Hadamard gate
        hadamard_op = QuantumGateOperation(
            gate=QuantumGate.HADAMARD,
            target_qubits=[0]
        )
        
        await quantum_circuit._apply_quantum_operation(hadamard_op)
        
        # After Hadamard, qubit should be in superposition
        # State vector should have non-zero amplitudes for |00000000⟩ and |00000001⟩
        assert abs(quantum_circuit.state_vector[0]) > 0.5  # |00000000⟩
        assert abs(quantum_circuit.state_vector[1]) > 0.5  # |00000001⟩
    
    @pytest.mark.asyncio
    async def test_cnot_gate_operation(self, quantum_circuit):
        """Test CNOT gate operation."""
        await quantum_circuit.add_task("control_task")
        await quantum_circuit.add_task("target_task")
        
        # First put control qubit in |1⟩ state with Pauli-X
        pauli_x_op = QuantumGateOperation(
            gate=QuantumGate.PAULI_X,
            target_qubits=[0]
        )
        await quantum_circuit._apply_quantum_operation(pauli_x_op)
        
        # Apply CNOT
        cnot_op = QuantumGateOperation(
            gate=QuantumGate.CNOT,
            control_qubits=[0],
            target_qubits=[1]
        )
        await quantum_circuit._apply_quantum_operation(cnot_op)
        
        # Target qubit should now also be |1⟩ (state |00000011⟩)
        assert abs(quantum_circuit.state_vector[3]) > 0.9  # |00000011⟩ = index 3
    
    @pytest.mark.asyncio
    async def test_rotation_gates(self, quantum_circuit):
        """Test parameterized rotation gates."""
        await quantum_circuit.add_task("rotation_test")
        
        # Test X-rotation
        rx_op = QuantumGateOperation(
            gate=QuantumGate.ROTATION_X,
            target_qubits=[0],
            parameter=np.pi/2
        )
        
        await quantum_circuit._apply_quantum_operation(rx_op)
        
        # After π/2 X-rotation, qubit should be in equal superposition
        assert abs(abs(quantum_circuit.state_vector[0]) - abs(quantum_circuit.state_vector[1])) < 0.1
    
    @pytest.mark.asyncio
    async def test_dependency_circuit_creation(self, quantum_circuit, sample_tasks):
        """Test dependency circuit creation."""
        operations = await quantum_circuit.create_dependency_circuit()
        
        # Should have operations for each task
        assert len(operations) > 0
        
        # Should include Hadamard gates for superposition
        hadamard_ops = [op for op in operations if op.gate == QuantumGate.HADAMARD]
        assert len(hadamard_ops) == len(sample_tasks)  # One per task
        
        # Should include CNOT gates for dependencies
        cnot_ops = [op for op in operations if op.gate == QuantumGate.CNOT]
        expected_dependencies = 3  # task_b->task_a, task_c->task_a, task_d->task_b, task_d->task_c
        assert len(cnot_ops) >= expected_dependencies
    
    @pytest.mark.asyncio
    async def test_circuit_simulation(self, quantum_circuit, sample_tasks):
        """Test complete circuit simulation."""
        simulation_result = await quantum_circuit.simulate_circuit()
        
        assert "simulation_time" in simulation_result
        assert "task_probabilities" in simulation_result
        assert "total_operations" in simulation_result
        assert "quantum_state_entropy" in simulation_result
        
        # Check task probabilities
        probabilities = simulation_result["task_probabilities"]
        assert len(probabilities) == len(sample_tasks)
        
        for task_id, prob in probabilities.items():
            assert 0 <= prob <= 1
            assert task_id in sample_tasks
    
    @pytest.mark.asyncio
    async def test_task_probability_calculation(self, quantum_circuit, sample_tasks):
        """Test task execution probability calculation."""
        # Simulate circuit
        await quantum_circuit.simulate_circuit()
        
        # Calculate probabilities
        probabilities = await quantum_circuit._calculate_task_probabilities()
        
        # All tasks should have valid probabilities
        for task_id in sample_tasks:
            assert task_id in probabilities
            prob = probabilities[task_id]
            assert 0 <= prob <= 1
            
            # Update task states
            quantum_circuit.task_states[task_id].execution_probability = prob
    
    @pytest.mark.asyncio
    async def test_quantum_measurement(self, quantum_circuit, sample_tasks):
        """Test quantum measurement and state collapse."""
        # Simulate circuit to create superposition
        await quantum_circuit.simulate_circuit()
        
        # Measure all tasks
        measurement_results = await quantum_circuit.measure_tasks()
        
        assert len(measurement_results) == len(sample_tasks)
        
        for task_id, measured_state in measurement_results.items():
            assert isinstance(measured_state, bool)
            assert task_id in sample_tasks
        
        # Check measurement history
        assert len(quantum_circuit.measurement_history) == 1
        assert "measurements" in quantum_circuit.measurement_history[0]
        assert "state_entropy" in quantum_circuit.measurement_history[0]
    
    @pytest.mark.asyncio
    async def test_quantum_state_entropy(self, quantum_circuit, sample_tasks):
        """Test quantum state entropy calculation."""
        # Initial state should have zero entropy
        entropy = await quantum_circuit._calculate_state_entropy()
        assert entropy < 0.1  # Very low entropy for |000...0⟩ state
        
        # After applying Hadamard gates, entropy should increase
        operations = await quantum_circuit.create_dependency_circuit()
        await quantum_circuit.simulate_circuit(operations)
        
        entropy_after = await quantum_circuit._calculate_state_entropy()
        assert entropy_after > entropy  # Increased entropy due to superposition
    
    @pytest.mark.asyncio
    async def test_circuit_depth_calculation(self, quantum_circuit, sample_tasks):
        """Test circuit depth calculation."""
        operations = await quantum_circuit.create_dependency_circuit()
        depth = await quantum_circuit._calculate_circuit_depth(operations)
        
        assert depth > 0
        assert isinstance(depth, int)
        
        # Depth should be reasonable for the number of operations
        assert depth <= len(operations)
    
    @pytest.mark.asyncio
    async def test_circuit_metrics(self, quantum_circuit, sample_tasks):
        """Test circuit metrics collection."""
        await quantum_circuit.simulate_circuit()
        
        metrics = await quantum_circuit.get_circuit_metrics()
        
        required_fields = [
            "num_qubits", "num_tasks", "circuit_operations", 
            "state_vector_norm", "state_entropy", "task_probabilities",
            "measurement_history_size", "available_qubits"
        ]
        
        for field in required_fields:
            assert field in metrics
        
        assert metrics["num_qubits"] == quantum_circuit.num_qubits
        assert metrics["num_tasks"] == len(sample_tasks)
        assert abs(metrics["state_vector_norm"] - 1.0) < 1e-10  # Normalized state
    
    @pytest.mark.asyncio
    async def test_circuit_reset(self, quantum_circuit, sample_tasks):
        """Test circuit reset functionality."""
        # Simulate and measure
        await quantum_circuit.simulate_circuit()
        await quantum_circuit.measure_tasks()
        
        # Verify circuit has state
        assert len(quantum_circuit.circuit_operations) > 0
        assert len(quantum_circuit.task_qubit_mapping) > 0
        assert len(quantum_circuit.measurement_history) > 0
        
        # Reset circuit
        await quantum_circuit.reset_circuit()
        
        # Verify reset
        assert len(quantum_circuit.circuit_operations) == 0
        assert len(quantum_circuit.task_qubit_mapping) == 0
        assert len(quantum_circuit.qubit_task_mapping) == 0
        assert len(quantum_circuit.task_states) == 0
        assert len(quantum_circuit.measurement_history) == 0
        
        # State vector should be back to |000...0⟩
        assert abs(quantum_circuit.state_vector[0] - 1.0) < 1e-10
        assert all(abs(amp) < 1e-10 for amp in quantum_circuit.state_vector[1:])
    
    @pytest.mark.asyncio
    async def test_invalid_gate_parameters(self, quantum_circuit):
        """Test invalid gate parameters."""
        await quantum_circuit.add_task("test_task")
        
        # Invalid CNOT (missing control qubit)
        invalid_cnot = QuantumGateOperation(
            gate=QuantumGate.CNOT,
            target_qubits=[0]
        )
        
        with pytest.raises(QuantumCircuitError):
            await quantum_circuit._apply_quantum_operation(invalid_cnot)
        
        # Invalid rotation (missing parameter)
        invalid_rotation = QuantumGateOperation(
            gate=QuantumGate.ROTATION_X,
            target_qubits=[0]
        )
        
        with pytest.raises(QuantumCircuitError):
            await quantum_circuit._apply_quantum_operation(invalid_rotation)
    
    @pytest.mark.asyncio
    async def test_quantum_gate_operation_validation(self):
        """Test quantum gate operation validation."""
        # Valid operation
        valid_op = QuantumGateOperation(
            gate=QuantumGate.HADAMARD,
            target_qubits=[0]
        )
        assert valid_op.target_qubits == [0]
        assert valid_op.control_qubits == []
        
        # Invalid - empty target qubits
        with pytest.raises(ValueError):
            QuantumGateOperation(
                gate=QuantumGate.HADAMARD,
                target_qubits=[]
            )
        
        # Invalid - qubit used as both target and control
        with pytest.raises(ValueError):
            QuantumGateOperation(
                gate=QuantumGate.CNOT,
                target_qubits=[0],
                control_qubits=[0]
            )
    
    @pytest.mark.asyncio
    async def test_task_quantum_state_properties(self):
        """Test TaskQuantumState properties."""
        task_state = TaskQuantumState(
            task_id="test_task",
            qubit_index=0,
            dependencies=["dep1", "dep2"]
        )
        
        assert task_state.task_id == "test_task"
        assert task_state.qubit_index == 0
        assert task_state.dependencies == ["dep1", "dep2"]
        
        # State vector should be normalized
        norm = np.linalg.norm(task_state.state_vector)
        assert abs(norm - 1.0) < 1e-10
        
        # Execution probability should be calculated
        expected_prob = abs(task_state.state_vector[1]) ** 2
        assert abs(task_state.execution_probability - expected_prob) < 1e-10
    
    @pytest.mark.asyncio
    async def test_large_circuit_performance(self):
        """Test performance with larger quantum circuits."""
        # Test with maximum practical size
        large_circuit = QuantumCircuitSimulator(num_qubits=16)
        
        start_time = time.time()
        
        # Add many tasks
        for i in range(10):
            await large_circuit.add_task(f"task_{i}")
        
        # Create and simulate circuit
        operations = await large_circuit.create_dependency_circuit()
        simulation_result = await large_circuit.simulate_circuit(operations)
        
        end_time = time.time()
        simulation_time = end_time - start_time
        
        # Should complete within reasonable time
        assert simulation_time < 10.0  # 10 seconds max
        assert simulation_result["total_operations"] == len(operations)
        assert len(simulation_result["task_probabilities"]) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_circuit_operations(self, quantum_circuit):
        """Test concurrent circuit operations."""
        # Add tasks concurrently
        tasks = await asyncio.gather(*[
            quantum_circuit.add_task(f"concurrent_task_{i}")
            for i in range(5)
        ])
        
        assert len(tasks) == 5
        assert len(quantum_circuit.task_qubit_mapping) == 5
        
        # Run multiple simulations concurrently
        simulation_tasks = [
            quantum_circuit.simulate_circuit() 
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*simulation_tasks)
        assert len(results) == 3
        
        # All simulations should produce valid results
        for result in results:
            assert "task_probabilities" in result
            assert len(result["task_probabilities"]) == 5


@pytest.mark.asyncio
async def test_quantum_circuit_integration():
    """Integration test for complete quantum circuit workflow."""
    circuit = QuantumCircuitSimulator(num_qubits=12)
    
    # Create complex task dependency graph
    await circuit.add_task("data_preprocessing", [])
    await circuit.add_task("model_loading", [])
    await circuit.add_task("training_step_1", ["data_preprocessing", "model_loading"])
    await circuit.add_task("training_step_2", ["training_step_1"])
    await circuit.add_task("validation", ["training_step_2"])
    await circuit.add_task("checkpointing", ["validation"])
    await circuit.add_task("metrics_logging", ["validation"])
    await circuit.add_task("cleanup", ["checkpointing", "metrics_logging"])
    
    # Create dependency circuit
    operations = await circuit.create_dependency_circuit()
    assert len(operations) > 8  # At least one operation per task
    
    # Simulate circuit
    simulation_result = await circuit.simulate_circuit(operations)
    
    # Verify results
    assert len(simulation_result["task_probabilities"]) == 8
    assert simulation_result["quantum_state_entropy"] > 0
    
    # Test measurements
    measurements = await circuit.measure_tasks()
    assert len(measurements) == 8
    
    # Verify dependency constraints in probabilities
    probabilities = simulation_result["task_probabilities"]
    
    # Root tasks should have higher probability of being ready
    assert probabilities["data_preprocessing"] >= probabilities["training_step_1"]
    assert probabilities["model_loading"] >= probabilities["training_step_1"]
    
    # Later tasks should have lower probability
    assert probabilities["cleanup"] <= probabilities["checkpointing"]
    assert probabilities["cleanup"] <= probabilities["metrics_logging"]
    
    # Get final metrics
    metrics = await circuit.get_circuit_metrics()
    assert metrics["num_tasks"] == 8
    assert metrics["available_qubits"] == 4  # 12 - 8 = 4
    
    print(f"✅ Quantum circuit integration test completed successfully")
    print(f"   - {metrics['num_tasks']} tasks simulated")
    print(f"   - {metrics['circuit_operations']} quantum operations")
    print(f"   - {simulation_result['quantum_state_entropy']:.3f} quantum entropy")
    print(f"   - {metrics['measurement_history_size']} measurements recorded")


if __name__ == "__main__":
    # Run integration test directly
    asyncio.run(test_quantum_circuit_integration())