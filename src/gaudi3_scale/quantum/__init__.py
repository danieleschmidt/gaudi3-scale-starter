"""Quantum-Inspired Task Planning for Gaudi 3 Scale.

This module implements quantum-inspired algorithms for intelligent task scheduling,
resource allocation, and distributed coordination in HPU clusters.

Features:
- Quantum circuit simulation for task dependency modeling
- Quantum annealing for optimal resource allocation  
- Quantum entanglement patterns for cluster coordination
- Quantum superposition for parallel task exploration
- Quantum interference for task priority optimization
"""

from .task_planner import QuantumTaskPlanner
from .resource_allocator import QuantumResourceAllocator
from .circuit_simulator import QuantumCircuitSimulator
from .annealing_optimizer import QuantumAnnealingOptimizer
from .entanglement_coordinator import EntanglementCoordinator
from .auto_scaler import QuantumAutoScaler
from .multi_dimensional_deployer import MultiDimensionalQuantumDeployer

__all__ = [
    "QuantumTaskPlanner",
    "QuantumResourceAllocator", 
    "QuantumCircuitSimulator",
    "QuantumAnnealingOptimizer",
    "EntanglementCoordinator",
    "QuantumAutoScaler",
    "MultiDimensionalQuantumDeployer"
]