"""Quantum Circuit Simulator for Task Dependency Modeling.

Implements quantum circuit simulation for modeling complex task dependencies
and execution patterns in HPU cluster environments.
"""

import asyncio
import logging
import cmath
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import time

from ..exceptions import QuantumCircuitError, ValidationError
from ..validation import DataValidator

logger = logging.getLogger(__name__)


class QuantumGate(Enum):
    """Quantum gates for circuit construction."""
    IDENTITY = "I"
    PAULI_X = "X" 
    PAULI_Y = "Y"
    PAULI_Z = "Z"
    HADAMARD = "H"
    PHASE = "S"
    T_GATE = "T"
    CNOT = "CNOT"
    CONTROLLED_Z = "CZ"
    TOFFOLI = "TOFFOLI"
    ROTATION_X = "RX"
    ROTATION_Y = "RY"
    ROTATION_Z = "RZ"


@dataclass
class QuantumGateOperation:
    """Quantum gate operation in circuit."""
    gate: QuantumGate
    target_qubits: List[int]
    control_qubits: List[int] = field(default_factory=list)
    parameter: Optional[float] = None  # For parameterized gates
    
    def __post_init__(self):
        # Validation
        if not self.target_qubits:
            raise ValueError("Target qubits cannot be empty")
        
        # Check for qubit conflicts
        all_qubits = set(self.target_qubits + self.control_qubits)
        if len(all_qubits) != len(self.target_qubits) + len(self.control_qubits):
            raise ValueError("Qubits cannot be both target and control")


@dataclass 
class TaskQuantumState:
    """Quantum state representation of a task."""
    task_id: str
    qubit_index: int
    state_vector: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j, 0.0 + 0j]))
    dependencies: List[str] = field(default_factory=list)
    execution_probability: float = 0.0
    
    def __post_init__(self):
        # Normalize state vector
        norm = np.linalg.norm(self.state_vector)
        if norm > 0:
            self.state_vector = self.state_vector / norm
        
        # Calculate execution probability |α|² where α is amplitude of |1⟩ state
        self.execution_probability = abs(self.state_vector[1]) ** 2


class QuantumCircuitSimulator:
    """Quantum circuit simulator for task dependency modeling."""
    
    def __init__(self, num_qubits: int = 16):
        """Initialize quantum circuit simulator.
        
        Args:
            num_qubits: Number of qubits in the quantum system
        """
        if num_qubits <= 0 or num_qubits > 32:  # Practical limit for simulation
            raise ValueError("Number of qubits must be between 1 and 32")
        
        self.num_qubits = num_qubits
        self.state_dim = 2 ** num_qubits
        
        # Initialize quantum state |00...0⟩
        self.state_vector = np.zeros(self.state_dim, dtype=complex)
        self.state_vector[0] = 1.0 + 0j
        
        # Circuit operations
        self.circuit_operations: List[QuantumGateOperation] = []
        
        # Task mapping
        self.task_qubit_mapping: Dict[str, int] = {}
        self.qubit_task_mapping: Dict[int, str] = {}
        self.task_states: Dict[str, TaskQuantumState] = {}
        
        # Measurement results
        self.measurement_history: List[Dict[str, Any]] = []
        
        # Gate matrices (precomputed for efficiency)
        self._initialize_gate_matrices()
        
        logger.info(f"Initialized quantum circuit simulator with {num_qubits} qubits")
    
    def _initialize_gate_matrices(self):
        """Initialize quantum gate matrices."""
        self.gate_matrices = {
            QuantumGate.IDENTITY: np.array([[1, 0], [0, 1]], dtype=complex),
            QuantumGate.PAULI_X: np.array([[0, 1], [1, 0]], dtype=complex),
            QuantumGate.PAULI_Y: np.array([[0, -1j], [1j, 0]], dtype=complex),
            QuantumGate.PAULI_Z: np.array([[1, 0], [0, -1]], dtype=complex),
            QuantumGate.HADAMARD: np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            QuantumGate.PHASE: np.array([[1, 0], [0, 1j]], dtype=complex),
            QuantumGate.T_GATE: np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
        }
    
    async def add_task(self, task_id: str, dependencies: List[str] = None) -> int:
        """Add a task to the quantum circuit.
        
        Args:
            task_id: Unique task identifier
            dependencies: List of task IDs this task depends on
            
        Returns:
            Assigned qubit index
        """
        if task_id in self.task_qubit_mapping:
            raise ValidationError(f"Task {task_id} already exists in circuit")
        
        # Find available qubit
        used_qubits = set(self.task_qubit_mapping.values())
        available_qubits = [i for i in range(self.num_qubits) if i not in used_qubits]
        
        if not available_qubits:
            raise QuantumCircuitError(f"No available qubits for task {task_id}")
        
        qubit_index = available_qubits[0]
        
        # Create task-qubit mapping
        self.task_qubit_mapping[task_id] = qubit_index
        self.qubit_task_mapping[qubit_index] = task_id
        
        # Initialize task quantum state
        self.task_states[task_id] = TaskQuantumState(
            task_id=task_id,
            qubit_index=qubit_index,
            dependencies=dependencies or []
        )
        
        logger.info(f"Added task {task_id} to qubit {qubit_index}")
        return qubit_index
    
    async def create_dependency_circuit(self) -> List[QuantumGateOperation]:
        """Create quantum circuit representing task dependencies."""
        
        operations = []
        
        # Initialize all task qubits in superposition (ready to execute)
        for task_id, qubit_index in self.task_qubit_mapping.items():
            # Apply Hadamard gate to create superposition |0⟩ + |1⟩
            operations.append(QuantumGateOperation(
                gate=QuantumGate.HADAMARD,
                target_qubits=[qubit_index]
            ))
        
        # Create entanglement for dependencies
        for task_id, task_state in self.task_states.items():
            task_qubit = task_state.qubit_index
            
            for dependency_id in task_state.dependencies:
                if dependency_id in self.task_qubit_mapping:
                    dependency_qubit = self.task_qubit_mapping[dependency_id]
                    
                    # Create controlled dependency: task can only execute if dependency is complete
                    # CNOT gate: |dependency⟩|task⟩ → |dependency⟩|dependency ⊕ task⟩
                    operations.append(QuantumGateOperation(
                        gate=QuantumGate.CNOT,
                        target_qubits=[task_qubit],
                        control_qubits=[dependency_qubit]
                    ))
        
        # Add priority-based phase gates
        await self._add_priority_phases(operations)
        
        return operations
    
    async def _add_priority_phases(self, operations: List[QuantumGateOperation]):
        """Add phase gates based on task priorities."""
        
        # Sort tasks by estimated execution time and add phase rotations
        task_priorities = {}
        for task_id in self.task_states.keys():
            # Simple priority: inverse of number of dependencies
            num_deps = len(self.task_states[task_id].dependencies)
            priority = 1.0 / (num_deps + 1)
            task_priorities[task_id] = priority
        
        # Sort by priority (highest first)
        sorted_tasks = sorted(task_priorities.items(), key=lambda x: x[1], reverse=True)
        
        for i, (task_id, priority) in enumerate(sorted_tasks):
            qubit_index = self.task_qubit_mapping[task_id]
            
            # Apply rotation based on priority ranking
            # Higher priority tasks get smaller phase rotation (execute sooner)
            phase_angle = (i + 1) * np.pi / (4 * len(sorted_tasks))
            
            operations.append(QuantumGateOperation(
                gate=QuantumGate.ROTATION_Z,
                target_qubits=[qubit_index],
                parameter=phase_angle
            ))
    
    async def simulate_circuit(self, operations: List[QuantumGateOperation] = None) -> Dict[str, Any]:
        """Simulate quantum circuit execution."""
        
        if operations is None:
            operations = await self.create_dependency_circuit()
        
        # Store operations
        self.circuit_operations = operations
        
        # Reset to initial state
        self.state_vector = np.zeros(self.state_dim, dtype=complex)
        self.state_vector[0] = 1.0 + 0j
        
        start_time = time.time()
        
        # Apply each quantum operation
        for operation in operations:
            await self._apply_quantum_operation(operation)
        
        simulation_time = time.time() - start_time
        
        # Calculate task execution probabilities
        task_probabilities = await self._calculate_task_probabilities()
        
        # Update task states
        for task_id, probability in task_probabilities.items():
            if task_id in self.task_states:
                self.task_states[task_id].execution_probability = probability
        
        simulation_result = {
            "simulation_time": simulation_time,
            "task_probabilities": task_probabilities,
            "total_operations": len(operations),
            "quantum_state_entropy": await self._calculate_state_entropy(),
            "circuit_depth": await self._calculate_circuit_depth(operations)
        }
        
        logger.info(f"Quantum circuit simulation completed in {simulation_time:.4f}s")
        return simulation_result
    
    async def _apply_quantum_operation(self, operation: QuantumGateOperation):
        """Apply a quantum operation to the state vector."""
        
        try:
            if operation.gate in [QuantumGate.IDENTITY, QuantumGate.PAULI_X, 
                                QuantumGate.PAULI_Y, QuantumGate.PAULI_Z,
                                QuantumGate.HADAMARD, QuantumGate.PHASE, QuantumGate.T_GATE]:
                
                # Single-qubit gates
                if len(operation.target_qubits) != 1:
                    raise QuantumCircuitError(f"Gate {operation.gate} requires exactly 1 target qubit")
                
                await self._apply_single_qubit_gate(operation.gate, operation.target_qubits[0])
                
            elif operation.gate in [QuantumGate.ROTATION_X, QuantumGate.ROTATION_Y, QuantumGate.ROTATION_Z]:
                
                # Parameterized rotation gates
                if len(operation.target_qubits) != 1 or operation.parameter is None:
                    raise QuantumCircuitError(f"Rotation gate requires 1 target qubit and parameter")
                
                await self._apply_rotation_gate(operation.gate, operation.target_qubits[0], operation.parameter)
                
            elif operation.gate == QuantumGate.CNOT:
                
                # Two-qubit CNOT gate
                if len(operation.control_qubits) != 1 or len(operation.target_qubits) != 1:
                    raise QuantumCircuitError("CNOT requires 1 control and 1 target qubit")
                
                await self._apply_cnot_gate(operation.control_qubits[0], operation.target_qubits[0])
                
            elif operation.gate == QuantumGate.CONTROLLED_Z:
                
                # Controlled-Z gate
                if len(operation.control_qubits) != 1 or len(operation.target_qubits) != 1:
                    raise QuantumCircuitError("Controlled-Z requires 1 control and 1 target qubit")
                
                await self._apply_controlled_z_gate(operation.control_qubits[0], operation.target_qubits[0])
                
            else:
                raise QuantumCircuitError(f"Unsupported quantum gate: {operation.gate}")
                
        except Exception as e:
            logger.error(f"Error applying quantum operation {operation.gate}: {e}")
            raise QuantumCircuitError(f"Failed to apply quantum operation: {e}")
    
    async def _apply_single_qubit_gate(self, gate: QuantumGate, qubit: int):
        """Apply single-qubit gate to state vector."""
        
        gate_matrix = self.gate_matrices[gate]
        
        # Create full system matrix using Kronecker products
        full_matrix = np.eye(1, dtype=complex)
        
        for i in range(self.num_qubits):
            if i == qubit:
                full_matrix = np.kron(full_matrix, gate_matrix)
            else:
                full_matrix = np.kron(full_matrix, self.gate_matrices[QuantumGate.IDENTITY])
        
        # Apply transformation
        self.state_vector = full_matrix @ self.state_vector
    
    async def _apply_rotation_gate(self, gate: QuantumGate, qubit: int, angle: float):
        """Apply parameterized rotation gate."""
        
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        
        if gate == QuantumGate.ROTATION_X:
            rotation_matrix = np.array([
                [cos_half, -1j * sin_half],
                [-1j * sin_half, cos_half]
            ], dtype=complex)
        elif gate == QuantumGate.ROTATION_Y:
            rotation_matrix = np.array([
                [cos_half, -sin_half],
                [sin_half, cos_half]
            ], dtype=complex)
        elif gate == QuantumGate.ROTATION_Z:
            rotation_matrix = np.array([
                [np.exp(-1j * angle / 2), 0],
                [0, np.exp(1j * angle / 2)]
            ], dtype=complex)
        else:
            raise QuantumCircuitError(f"Unknown rotation gate: {gate}")
        
        # Create full system matrix
        full_matrix = np.eye(1, dtype=complex)
        
        for i in range(self.num_qubits):
            if i == qubit:
                full_matrix = np.kron(full_matrix, rotation_matrix)
            else:
                full_matrix = np.kron(full_matrix, self.gate_matrices[QuantumGate.IDENTITY])
        
        # Apply transformation
        self.state_vector = full_matrix @ self.state_vector
    
    async def _apply_cnot_gate(self, control_qubit: int, target_qubit: int):
        """Apply CNOT (controlled-X) gate."""
        
        # CNOT matrix for two qubits
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        # For multi-qubit system, we need to construct the full matrix
        # This is a simplified implementation for demonstration
        # In practice, you'd use more efficient tensor product operations
        
        new_state = self.state_vector.copy()
        
        for i in range(self.state_dim):
            # Get bit representation
            bits = [(i >> j) & 1 for j in range(self.num_qubits)]
            
            # Apply CNOT logic: if control is 1, flip target
            if bits[control_qubit] == 1:
                # Flip target bit
                new_bits = bits.copy()
                new_bits[target_qubit] = 1 - new_bits[target_qubit]
                
                # Calculate new state index
                new_index = sum(bit * (2 ** j) for j, bit in enumerate(new_bits))
                
                # Swap amplitudes
                new_state[new_index] = self.state_vector[i]
                new_state[i] = 0
        
        self.state_vector = new_state
    
    async def _apply_controlled_z_gate(self, control_qubit: int, target_qubit: int):
        """Apply controlled-Z gate."""
        
        # Controlled-Z applies Z gate to target if control is |1⟩
        for i in range(self.state_dim):
            bits = [(i >> j) & 1 for j in range(self.num_qubits)]
            
            # If control is 1 and target is 1, apply phase flip
            if bits[control_qubit] == 1 and bits[target_qubit] == 1:
                self.state_vector[i] *= -1
    
    async def _calculate_task_probabilities(self) -> Dict[str, float]:
        """Calculate execution probabilities for each task."""
        
        probabilities = {}
        
        for task_id, qubit_index in self.task_qubit_mapping.items():
            # Calculate probability of qubit being in |1⟩ state (ready to execute)
            probability = 0.0
            
            for i in range(self.state_dim):
                # Check if this state has the task qubit in |1⟩
                if (i >> qubit_index) & 1 == 1:
                    probability += abs(self.state_vector[i]) ** 2
            
            probabilities[task_id] = probability
        
        return probabilities
    
    async def _calculate_state_entropy(self) -> float:
        """Calculate von Neumann entropy of the quantum state."""
        
        # Calculate density matrix diagonal elements (probabilities)
        probabilities = np.abs(self.state_vector) ** 2
        
        # Remove zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 1e-16]
        
        if len(probabilities) == 0:
            return 0.0
        
        # Calculate entropy: S = -Σ p_i * log2(p_i)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return float(entropy)
    
    async def _calculate_circuit_depth(self, operations: List[QuantumGateOperation]) -> int:
        """Calculate circuit depth (maximum number of gates on any qubit)."""
        
        qubit_depths = {i: 0 for i in range(self.num_qubits)}
        
        for operation in operations:
            # Find maximum depth among involved qubits
            involved_qubits = operation.target_qubits + operation.control_qubits
            max_depth = max(qubit_depths[q] for q in involved_qubits) if involved_qubits else 0
            
            # Increment depth for all involved qubits
            for qubit in involved_qubits:
                qubit_depths[qubit] = max_depth + 1
        
        return max(qubit_depths.values()) if qubit_depths else 0
    
    async def measure_tasks(self, task_ids: List[str] = None) -> Dict[str, bool]:
        """Measure quantum states to determine task execution outcomes."""
        
        if task_ids is None:
            task_ids = list(self.task_qubit_mapping.keys())
        
        measurement_results = {}
        
        # Calculate measurement probabilities
        for task_id in task_ids:
            if task_id not in self.task_qubit_mapping:
                continue
            
            qubit_index = self.task_qubit_mapping[task_id]
            probability = await self._get_qubit_probability(qubit_index)
            
            # Quantum measurement: probabilistic outcome
            measurement_results[task_id] = np.random.random() < probability
        
        # Store measurement in history
        self.measurement_history.append({
            "timestamp": time.time(),
            "measurements": measurement_results.copy(),
            "state_entropy": await self._calculate_state_entropy()
        })
        
        # Collapse quantum state after measurement
        await self._collapse_measured_qubits(measurement_results)
        
        return measurement_results
    
    async def _get_qubit_probability(self, qubit_index: int) -> float:
        """Get probability of qubit being in |1⟩ state."""
        
        probability = 0.0
        
        for i in range(self.state_dim):
            if (i >> qubit_index) & 1 == 1:
                probability += abs(self.state_vector[i]) ** 2
        
        return probability
    
    async def _collapse_measured_qubits(self, measurements: Dict[str, bool]):
        """Collapse quantum state after measurement."""
        
        # Create new state vector with collapsed measurements
        new_state = np.zeros_like(self.state_vector)
        
        for i in range(self.state_dim):
            # Check if this state is consistent with measurements
            consistent = True
            
            for task_id, measured_value in measurements.items():
                if task_id in self.task_qubit_mapping:
                    qubit_index = self.task_qubit_mapping[task_id]
                    qubit_value = (i >> qubit_index) & 1
                    
                    if (qubit_value == 1) != measured_value:
                        consistent = False
                        break
            
            if consistent:
                new_state[i] = self.state_vector[i]
        
        # Renormalize
        norm = np.linalg.norm(new_state)
        if norm > 0:
            self.state_vector = new_state / norm
        
        logger.info(f"Collapsed quantum state after measurements: {measurements}")
    
    async def get_circuit_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit metrics."""
        
        return {
            "num_qubits": self.num_qubits,
            "num_tasks": len(self.task_qubit_mapping),
            "circuit_operations": len(self.circuit_operations),
            "state_vector_norm": float(np.linalg.norm(self.state_vector)),
            "state_entropy": await self._calculate_state_entropy(),
            "task_probabilities": await self._calculate_task_probabilities(),
            "measurement_history_size": len(self.measurement_history),
            "available_qubits": self.num_qubits - len(self.task_qubit_mapping)
        }
    
    async def reset_circuit(self):
        """Reset quantum circuit to initial state."""
        
        # Reset state vector to |00...0⟩
        self.state_vector = np.zeros(self.state_dim, dtype=complex)
        self.state_vector[0] = 1.0 + 0j
        
        # Clear circuit operations
        self.circuit_operations.clear()
        
        # Clear task mappings
        self.task_qubit_mapping.clear()
        self.qubit_task_mapping.clear()
        self.task_states.clear()
        
        # Clear measurement history
        self.measurement_history.clear()
        
        logger.info("Reset quantum circuit to initial state")