"""Quantum-Inspired Task Planner for Optimal HPU Cluster Scheduling.

Implements quantum algorithms for intelligent task planning:
- Quantum superposition for exploring multiple execution paths
- Quantum interference for priority optimization
- Quantum entanglement for distributed coordination
- Quantum annealing for resource allocation
"""

import asyncio
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import time

from ..exceptions import TaskPlanningError, ValidationError
from ..validation import DataValidator
from ..monitoring.performance import PerformanceMonitor

logger = logging.getLogger(__name__)


class QuantumState(Enum):
    """Quantum states for task execution."""
    SUPERPOSITION = "superposition"  # Exploring multiple paths
    ENTANGLED = "entangled"          # Coordinated with other tasks
    COLLAPSED = "collapsed"           # Determined execution path
    INTERFERENCE = "interference"     # Priority optimization


@dataclass
class QuantumTask:
    """Quantum representation of a computational task."""
    task_id: str
    amplitude: complex = field(default_factory=lambda: complex(1.0, 0.0))
    phase: float = 0.0
    entangled_tasks: Set[str] = field(default_factory=set)
    priority_weight: float = 1.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: float = 0.0
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    
    @property
    def probability_amplitude(self) -> float:
        """Calculate probability amplitude |ψ|²."""
        return abs(self.amplitude) ** 2
    
    def apply_quantum_gate(self, gate: str, angle: float = 0.0):
        """Apply quantum gate transformation."""
        if gate == "hadamard":
            # Create superposition
            self.amplitude = (self.amplitude + self.amplitude * complex(0, 1)) / math.sqrt(2)
        elif gate == "phase":
            # Apply phase shift
            self.phase += angle
            self.amplitude *= complex(math.cos(angle), math.sin(angle))
        elif gate == "pauli_x":
            # Bit flip
            self.amplitude = complex(-self.amplitude.imag, self.amplitude.real)
        
    def collapse_to_state(self, state: QuantumState):
        """Collapse quantum superposition to definite state."""
        self.quantum_state = state
        # Normalize amplitude after collapse
        if self.quantum_state == QuantumState.COLLAPSED:
            self.amplitude = complex(1.0, 0.0) if self.probability_amplitude > 0.5 else complex(0.0, 0.0)


class QuantumTaskPlanner:
    """Quantum-inspired task planner for HPU clusters."""
    
    def __init__(self, 
                 cluster_nodes: int = 8,
                 hpu_per_node: int = 8,
                 quantum_coherence_time: float = 10.0,
                 enable_entanglement: bool = True):
        self.cluster_nodes = cluster_nodes
        self.hpu_per_node = hpu_per_node
        self.total_hpus = cluster_nodes * hpu_per_node
        self.quantum_coherence_time = quantum_coherence_time
        self.enable_entanglement = enable_entanglement
        
        # Quantum system state
        self.quantum_tasks: Dict[str, QuantumTask] = {}
        self.entanglement_graph: Dict[str, Set[str]] = {}
        self.interference_matrix: np.ndarray = np.eye(0)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Resource tracking
        self.available_resources = {
            "hpu_cores": self.total_hpus,
            "memory_gb": self.total_hpus * 96,  # 96GB HBM per Gaudi 3
            "network_bandwidth": cluster_nodes * 200  # 200Gb/s per node
        }
        
        logger.info(f"Initialized QuantumTaskPlanner with {self.total_hpus} HPUs across {cluster_nodes} nodes")
    
    async def add_task(self, 
                      task_id: str,
                      resource_requirements: Dict[str, float],
                      dependencies: Set[str] = None,
                      priority: float = 1.0,
                      estimated_duration: float = 0.0) -> QuantumTask:
        """Add a new task to the quantum planning system."""
        
        # Validate inputs
        validator = DataValidator()
        if not validator.validate_string(task_id, min_length=1):
            raise ValidationError(f"Invalid task_id: {task_id}")
        
        if task_id in self.quantum_tasks:
            raise TaskPlanningError(f"Task {task_id} already exists")
        
        # Create quantum task
        quantum_task = QuantumTask(
            task_id=task_id,
            resource_requirements=resource_requirements,
            dependencies=dependencies or set(),
            priority_weight=priority,
            estimated_duration=estimated_duration
        )
        
        # Initialize in quantum superposition
        quantum_task.apply_quantum_gate("hadamard")
        
        # Add to quantum system
        self.quantum_tasks[task_id] = quantum_task
        
        # Setup entanglement if enabled
        if self.enable_entanglement:
            await self._create_entanglement_patterns(task_id)
        
        logger.info(f"Added quantum task {task_id} with amplitude {quantum_task.amplitude}")
        return quantum_task
    
    async def _create_entanglement_patterns(self, task_id: str):
        """Create quantum entanglement patterns between related tasks."""
        task = self.quantum_tasks[task_id]
        
        # Entangle with dependency tasks
        for dep_id in task.dependencies:
            if dep_id in self.quantum_tasks:
                await self._entangle_tasks(task_id, dep_id)
        
        # Entangle tasks with similar resource requirements
        for other_id, other_task in self.quantum_tasks.items():
            if other_id != task_id:
                resource_similarity = self._calculate_resource_similarity(
                    task.resource_requirements, 
                    other_task.resource_requirements
                )
                
                # Create entanglement if high similarity
                if resource_similarity > 0.8:
                    await self._entangle_tasks(task_id, other_id)
    
    async def _entangle_tasks(self, task1_id: str, task2_id: str):
        """Create quantum entanglement between two tasks."""
        task1 = self.quantum_tasks[task1_id]
        task2 = self.quantum_tasks[task2_id]
        
        # Add to entanglement sets
        task1.entangled_tasks.add(task2_id)
        task2.entangled_tasks.add(task1_id)
        
        # Update entanglement graph
        if task1_id not in self.entanglement_graph:
            self.entanglement_graph[task1_id] = set()
        if task2_id not in self.entanglement_graph:
            self.entanglement_graph[task2_id] = set()
        
        self.entanglement_graph[task1_id].add(task2_id)
        self.entanglement_graph[task2_id].add(task1_id)
        
        # Apply entanglement transformation
        # Create Bell state: |00⟩ + |11⟩
        entanglement_phase = math.pi / 4  # 45-degree phase
        task1.apply_quantum_gate("phase", entanglement_phase)
        task2.apply_quantum_gate("phase", -entanglement_phase)  # Anti-correlated
        
        task1.quantum_state = QuantumState.ENTANGLED
        task2.quantum_state = QuantumState.ENTANGLED
        
        logger.info(f"Created quantum entanglement between {task1_id} and {task2_id}")
    
    def _calculate_resource_similarity(self, req1: Dict[str, float], req2: Dict[str, float]) -> float:
        """Calculate resource requirement similarity using cosine similarity."""
        # Get all unique resource types
        all_resources = set(req1.keys()) | set(req2.keys())
        
        # Create vectors
        vec1 = np.array([req1.get(resource, 0.0) for resource in all_resources])
        vec2 = np.array([req2.get(resource, 0.0) for resource in all_resources])
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        return dot_product / norms
    
    async def optimize_task_schedule(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Optimize task schedule using quantum interference patterns."""
        
        if not self.quantum_tasks:
            return []
        
        # Create interference matrix for priority optimization
        await self._build_interference_matrix()
        
        # Apply quantum interference to optimize priorities
        optimized_priorities = await self._apply_quantum_interference()
        
        # Collapse quantum states to determine execution order
        execution_plan = await self._collapse_to_execution_plan(optimized_priorities)
        
        logger.info(f"Generated quantum-optimized execution plan for {len(execution_plan)} tasks")
        return execution_plan
    
    async def _build_interference_matrix(self):
        """Build quantum interference matrix for task interactions."""
        task_ids = list(self.quantum_tasks.keys())
        n_tasks = len(task_ids)
        
        self.interference_matrix = np.zeros((n_tasks, n_tasks), dtype=complex)
        
        for i, task1_id in enumerate(task_ids):
            for j, task2_id in enumerate(task_ids):
                if i == j:
                    # Self-interference (identity)
                    self.interference_matrix[i, j] = 1.0 + 0j
                else:
                    task1 = self.quantum_tasks[task1_id]
                    task2 = self.quantum_tasks[task2_id]
                    
                    # Calculate interference amplitude
                    phase_diff = task1.phase - task2.phase
                    interference = task1.amplitude * np.conj(task2.amplitude) * np.exp(1j * phase_diff)
                    
                    # Apply entanglement correlation
                    if task2_id in task1.entangled_tasks:
                        interference *= 1.5  # Constructive interference for entangled tasks
                    
                    # Apply resource conflict destructive interference
                    resource_conflict = self._calculate_resource_conflict(
                        task1.resource_requirements, 
                        task2.resource_requirements
                    )
                    if resource_conflict > 0.5:
                        interference *= -0.5  # Destructive interference for conflicting tasks
                    
                    self.interference_matrix[i, j] = interference
    
    def _calculate_resource_conflict(self, req1: Dict[str, float], req2: Dict[str, float]) -> float:
        """Calculate resource conflict level between two tasks."""
        total_conflict = 0.0
        total_resources = 0
        
        for resource, amount1 in req1.items():
            if resource in req2:
                amount2 = req2[resource]
                available = self.available_resources.get(resource, 0.0)
                
                if available > 0:
                    conflict = min(1.0, (amount1 + amount2) / available - 1.0)
                    total_conflict += max(0.0, conflict)
                    total_resources += 1
        
        return total_conflict / max(1, total_resources)
    
    async def _apply_quantum_interference(self) -> Dict[str, float]:
        """Apply quantum interference to optimize task priorities."""
        task_ids = list(self.quantum_tasks.keys())
        
        # Get initial priority vector
        initial_priorities = np.array([
            self.quantum_tasks[task_id].priority_weight 
            for task_id in task_ids
        ])
        
        # Apply interference transformation
        # Priority evolution: P' = |M · P|² where M is interference matrix
        evolved_amplitudes = self.interference_matrix @ initial_priorities
        optimized_priorities = np.abs(evolved_amplitudes) ** 2
        
        # Normalize priorities
        optimized_priorities = optimized_priorities / np.sum(optimized_priorities)
        
        # Create task priority mapping
        priority_map = {}
        for i, task_id in enumerate(task_ids):
            priority_map[task_id] = float(optimized_priorities[i])
        
        logger.info("Applied quantum interference optimization to task priorities")
        return priority_map
    
    async def _collapse_to_execution_plan(self, optimized_priorities: Dict[str, float]) -> List[Tuple[str, Dict[str, Any]]]:
        """Collapse quantum superposition to concrete execution plan."""
        
        # Sort tasks by optimized priority (highest first)
        sorted_tasks = sorted(
            optimized_priorities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        execution_plan = []
        scheduled_tasks = set()
        current_time = 0.0
        
        for task_id, priority in sorted_tasks:
            task = self.quantum_tasks[task_id]
            
            # Check if dependencies are satisfied
            if not task.dependencies.issubset(scheduled_tasks):
                # Skip for now, will be scheduled when dependencies are ready
                continue
            
            # Collapse quantum state
            task.collapse_to_state(QuantumState.COLLAPSED)
            
            # Calculate optimal start time considering entangled tasks
            start_time = await self._calculate_optimal_start_time(task_id, current_time)
            
            # Add to execution plan
            execution_info = {
                "task_id": task_id,
                "start_time": start_time,
                "estimated_duration": task.estimated_duration,
                "priority": priority,
                "quantum_amplitude": abs(task.amplitude),
                "resource_requirements": task.resource_requirements,
                "entangled_tasks": list(task.entangled_tasks),
                "quantum_state": task.quantum_state.value
            }
            
            execution_plan.append((task_id, execution_info))
            scheduled_tasks.add(task_id)
            current_time = max(current_time, start_time + task.estimated_duration)
        
        # Handle any remaining unscheduled tasks (dependency cycles, etc.)
        unscheduled = set(self.quantum_tasks.keys()) - scheduled_tasks
        for task_id in unscheduled:
            task = self.quantum_tasks[task_id]
            task.collapse_to_state(QuantumState.COLLAPSED)
            
            execution_info = {
                "task_id": task_id,
                "start_time": current_time,
                "estimated_duration": task.estimated_duration,
                "priority": optimized_priorities.get(task_id, 0.0),
                "quantum_amplitude": abs(task.amplitude),
                "resource_requirements": task.resource_requirements,
                "entangled_tasks": list(task.entangled_tasks),
                "quantum_state": task.quantum_state.value,
                "note": "Scheduled after dependency resolution"
            }
            
            execution_plan.append((task_id, execution_info))
            current_time += task.estimated_duration
        
        return execution_plan
    
    async def _calculate_optimal_start_time(self, task_id: str, current_time: float) -> float:
        """Calculate optimal start time considering quantum entanglement."""
        task = self.quantum_tasks[task_id]
        
        # Base start time
        optimal_time = current_time
        
        # Consider entangled tasks for coordinated execution
        if task.entangled_tasks and task.quantum_state == QuantumState.ENTANGLED:
            entangled_times = []
            
            for entangled_id in task.entangled_tasks:
                if entangled_id in self.quantum_tasks:
                    entangled_task = self.quantum_tasks[entangled_id]
                    # Quantum correlation suggests synchronized execution
                    correlation_delay = abs(task.phase - entangled_task.phase) * 0.1
                    entangled_times.append(current_time + correlation_delay)
            
            if entangled_times:
                # Use quantum superposition principle - average of entangled states
                optimal_time = np.mean(entangled_times)
        
        return optimal_time
    
    async def get_quantum_metrics(self) -> Dict[str, Any]:
        """Get quantum system metrics and statistics."""
        total_tasks = len(self.quantum_tasks)
        
        if total_tasks == 0:
            return {"total_tasks": 0, "quantum_coherence": 0.0}
        
        # Calculate quantum coherence
        total_amplitude = sum(abs(task.amplitude) for task in self.quantum_tasks.values())
        quantum_coherence = total_amplitude / total_tasks if total_tasks > 0 else 0.0
        
        # Count tasks by quantum state
        state_counts = {}
        for state in QuantumState:
            state_counts[state.value] = sum(
                1 for task in self.quantum_tasks.values() 
                if task.quantum_state == state
            )
        
        # Calculate entanglement metrics
        total_entanglements = sum(len(task.entangled_tasks) for task in self.quantum_tasks.values()) // 2
        entanglement_density = total_entanglements / max(1, total_tasks * (total_tasks - 1) // 2)
        
        return {
            "total_tasks": total_tasks,
            "quantum_coherence": quantum_coherence,
            "state_distribution": state_counts,
            "total_entanglements": total_entanglements,
            "entanglement_density": entanglement_density,
            "interference_matrix_size": self.interference_matrix.shape,
            "available_resources": self.available_resources.copy()
        }
    
    async def reset_quantum_system(self):
        """Reset quantum system to initial state."""
        # Clear all quantum tasks
        for task in self.quantum_tasks.values():
            task.quantum_state = QuantumState.SUPERPOSITION
            task.amplitude = complex(1.0, 0.0)
            task.phase = 0.0
            task.entangled_tasks.clear()
        
        # Clear entanglement graph
        self.entanglement_graph.clear()
        
        # Reset interference matrix
        self.interference_matrix = np.eye(0)
        
        logger.info("Reset quantum task planning system")