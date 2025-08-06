"""Quantum Resource Allocator for Optimal HPU Cluster Resource Management.

Implements quantum annealing and optimization algorithms for:
- Dynamic resource allocation across HPU clusters
- Quantum-inspired load balancing 
- Energy-efficient resource distribution
- Predictive resource scaling based on quantum states
"""

import asyncio
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor

from ..exceptions import ResourceAllocationError, ValidationError
from ..validation import DataValidator
from ..monitoring.performance import PerformanceMonitor

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources in HPU cluster."""
    HPU_CORES = "hpu_cores"
    MEMORY_GB = "memory_gb"
    NETWORK_BANDWIDTH = "network_bandwidth"
    STORAGE_GB = "storage_gb"
    POWER_WATTS = "power_watts"


@dataclass
class QuantumResource:
    """Quantum representation of cluster resources."""
    resource_type: ResourceType
    total_capacity: float
    allocated: float = 0.0
    quantum_efficiency: complex = field(default_factory=lambda: complex(1.0, 0.0))
    energy_state: float = 0.0  # Quantum energy level
    entanglement_links: Set[str] = field(default_factory=set)
    
    @property
    def available(self) -> float:
        """Available resource capacity."""
        return max(0.0, self.total_capacity - self.allocated)
    
    @property
    def utilization(self) -> float:
        """Resource utilization percentage."""
        if self.total_capacity == 0:
            return 0.0
        return (self.allocated / self.total_capacity) * 100
    
    @property
    def quantum_state_amplitude(self) -> float:
        """Quantum state amplitude for resource optimization."""
        return abs(self.quantum_efficiency) ** 2


@dataclass
class ResourceRequest:
    """Resource allocation request."""
    request_id: str
    task_id: str
    requirements: Dict[ResourceType, float]
    priority: float = 1.0
    duration_estimate: float = 0.0
    min_requirements: Optional[Dict[ResourceType, float]] = None
    max_requirements: Optional[Dict[ResourceType, float]] = None
    deadline: Optional[float] = None
    
    def __post_init__(self):
        if self.min_requirements is None:
            # Default minimum is 50% of requirements
            self.min_requirements = {
                resource: amount * 0.5 
                for resource, amount in self.requirements.items()
            }
        
        if self.max_requirements is None:
            # Default maximum is 150% of requirements
            self.max_requirements = {
                resource: amount * 1.5 
                for resource, amount in self.requirements.items()
            }


class QuantumResourceAllocator:
    """Quantum-inspired resource allocator for HPU clusters."""
    
    def __init__(self, 
                 cluster_nodes: int = 8,
                 hpu_per_node: int = 8,
                 memory_per_hpu: float = 96.0,  # GB
                 network_bandwidth_per_node: float = 200.0,  # Gbps
                 power_per_node: float = 1500.0):  # Watts
        
        self.cluster_nodes = cluster_nodes
        self.hpu_per_node = hpu_per_node
        
        # Initialize quantum resource pool
        self.quantum_resources: Dict[str, QuantumResource] = {}
        self._initialize_quantum_resources(
            cluster_nodes, hpu_per_node, memory_per_hpu, 
            network_bandwidth_per_node, power_per_node
        )
        
        # Resource allocation tracking
        self.active_allocations: Dict[str, Dict[ResourceType, float]] = {}
        self.pending_requests: List[ResourceRequest] = []
        
        # Quantum annealing parameters
        self.annealing_temperature = 100.0
        self.cooling_rate = 0.95
        self.min_temperature = 0.01
        self.max_iterations = 1000
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Thread pool for parallel quantum computations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized QuantumResourceAllocator with {self.cluster_nodes} nodes")
    
    def _initialize_quantum_resources(self, 
                                    cluster_nodes: int, 
                                    hpu_per_node: int,
                                    memory_per_hpu: float,
                                    network_bandwidth_per_node: float,
                                    power_per_node: float):
        """Initialize quantum resource pool."""
        
        # HPU cores
        total_hpus = cluster_nodes * hpu_per_node
        self.quantum_resources["hpu_cores"] = QuantumResource(
            resource_type=ResourceType.HPU_CORES,
            total_capacity=float(total_hpus),
            quantum_efficiency=complex(1.0, 0.0)
        )
        
        # Memory
        total_memory = total_hpus * memory_per_hpu
        self.quantum_resources["memory_gb"] = QuantumResource(
            resource_type=ResourceType.MEMORY_GB,
            total_capacity=total_memory,
            quantum_efficiency=complex(0.9, 0.1)  # Slight phase for memory optimization
        )
        
        # Network bandwidth
        total_network = cluster_nodes * network_bandwidth_per_node
        self.quantum_resources["network_bandwidth"] = QuantumResource(
            resource_type=ResourceType.NETWORK_BANDWIDTH,
            total_capacity=total_network,
            quantum_efficiency=complex(0.8, 0.2)  # Higher phase for network coordination
        )
        
        # Storage (assumed 1TB NVMe per node)
        total_storage = cluster_nodes * 1000.0
        self.quantum_resources["storage_gb"] = QuantumResource(
            resource_type=ResourceType.STORAGE_GB,
            total_capacity=total_storage,
            quantum_efficiency=complex(0.7, 0.3)
        )
        
        # Power
        total_power = cluster_nodes * power_per_node
        self.quantum_resources["power_watts"] = QuantumResource(
            resource_type=ResourceType.POWER_WATTS,
            total_capacity=total_power,
            quantum_efficiency=complex(0.6, 0.4)  # High phase for power optimization
        )
        
        # Create quantum entanglements between related resources
        self._create_resource_entanglements()
    
    def _create_resource_entanglements(self):
        """Create quantum entanglements between related resources."""
        # HPU cores entangled with memory
        self.quantum_resources["hpu_cores"].entanglement_links.add("memory_gb")
        self.quantum_resources["memory_gb"].entanglement_links.add("hpu_cores")
        
        # Network entangled with storage and power
        self.quantum_resources["network_bandwidth"].entanglement_links.add("storage_gb")
        self.quantum_resources["storage_gb"].entanglement_links.add("network_bandwidth")
        
        # Power entangled with all compute resources
        for resource_name in ["hpu_cores", "memory_gb", "network_bandwidth"]:
            self.quantum_resources["power_watts"].entanglement_links.add(resource_name)
            self.quantum_resources[resource_name].entanglement_links.add("power_watts")
    
    async def request_resources(self, 
                              task_id: str,
                              requirements: Dict[str, float],
                              priority: float = 1.0,
                              duration_estimate: float = 0.0,
                              deadline: Optional[float] = None) -> str:
        """Request resource allocation using quantum optimization."""
        
        # Validate inputs
        validator = DataValidator()
        if not validator.validate_string(task_id, min_length=1):
            raise ValidationError(f"Invalid task_id: {task_id}")
        
        # Convert string keys to ResourceType enum
        typed_requirements = {}
        for resource_name, amount in requirements.items():
            try:
                resource_type = ResourceType(resource_name)
                typed_requirements[resource_type] = float(amount)
            except (ValueError, TypeError):
                raise ValidationError(f"Invalid resource type or amount: {resource_name}={amount}")
        
        # Create resource request
        request_id = f"{task_id}_{int(time.time() * 1000000)}"  # Microsecond precision
        
        request = ResourceRequest(
            request_id=request_id,
            task_id=task_id,
            requirements=typed_requirements,
            priority=priority,
            duration_estimate=duration_estimate,
            deadline=deadline
        )
        
        # Attempt immediate allocation using quantum annealing
        allocation_result = await self._quantum_allocate(request)
        
        if allocation_result["success"]:
            # Record allocation
            allocation_amounts = {}
            for resource_type, amount in allocation_result["allocation"].items():
                resource_name = resource_type.value
                allocation_amounts[resource_type] = amount
                
                # Update resource allocation
                self.quantum_resources[resource_name].allocated += amount
            
            self.active_allocations[request_id] = allocation_amounts
            
            logger.info(f"Successfully allocated resources for task {task_id} (request {request_id})")
            return request_id
        else:
            # Add to pending queue for batch optimization
            self.pending_requests.append(request)
            logger.info(f"Added task {task_id} to pending resource queue")
            return request_id
    
    async def _quantum_allocate(self, request: ResourceRequest) -> Dict[str, Any]:
        """Perform quantum annealing optimization for resource allocation."""
        
        # Check if basic requirements can be met
        if not await self._can_satisfy_requirements(request.requirements):
            return {
                "success": False,
                "reason": "Insufficient total resources",
                "allocation": {}
            }
        
        # Perform quantum annealing optimization
        optimal_allocation = await self._quantum_annealing_optimization(request)
        
        if optimal_allocation is not None:
            return {
                "success": True,
                "allocation": optimal_allocation,
                "quantum_energy": self._calculate_allocation_energy(optimal_allocation)
            }
        else:
            return {
                "success": False,
                "reason": "Quantum optimization failed",
                "allocation": {}
            }
    
    async def _can_satisfy_requirements(self, requirements: Dict[ResourceType, float]) -> bool:
        """Check if requirements can be satisfied by available resources."""
        for resource_type, required_amount in requirements.items():
            resource_name = resource_type.value
            if resource_name not in self.quantum_resources:
                return False
            
            resource = self.quantum_resources[resource_name]
            if resource.available < required_amount:
                return False
        
        return True
    
    async def _quantum_annealing_optimization(self, request: ResourceRequest) -> Optional[Dict[ResourceType, float]]:
        """Quantum annealing algorithm for optimal resource allocation."""
        
        # Initialize quantum state
        current_temperature = self.annealing_temperature
        
        # Start with minimum requirements as initial state
        current_allocation = request.min_requirements.copy()
        current_energy = self._calculate_allocation_energy(current_allocation)
        
        best_allocation = current_allocation.copy()
        best_energy = current_energy
        
        iteration = 0
        
        while current_temperature > self.min_temperature and iteration < self.max_iterations:
            # Generate neighbor state using quantum tunneling
            neighbor_allocation = await self._generate_neighbor_state(
                current_allocation, request, current_temperature
            )
            
            if neighbor_allocation is None:
                break
            
            neighbor_energy = self._calculate_allocation_energy(neighbor_allocation)
            
            # Quantum acceptance probability
            delta_energy = neighbor_energy - current_energy
            
            if delta_energy < 0:
                # Better state - always accept
                acceptance_probability = 1.0
            else:
                # Worse state - quantum tunneling probability
                acceptance_probability = math.exp(-delta_energy / current_temperature)
            
            # Accept or reject using quantum superposition
            if np.random.random() < acceptance_probability:
                current_allocation = neighbor_allocation
                current_energy = neighbor_energy
                
                # Update best solution
                if current_energy < best_energy:
                    best_allocation = current_allocation.copy()
                    best_energy = current_energy
            
            # Cool down (quantum decoherence)
            current_temperature *= self.cooling_rate
            iteration += 1
        
        # Verify final allocation is valid
        if await self._is_allocation_valid(best_allocation, request):
            return best_allocation
        else:
            return None
    
    async def _generate_neighbor_state(self, 
                                     current_allocation: Dict[ResourceType, float],
                                     request: ResourceRequest,
                                     temperature: float) -> Optional[Dict[ResourceType, float]]:
        """Generate neighbor state using quantum fluctuations."""
        
        neighbor = current_allocation.copy()
        
        # Apply quantum fluctuations to one random resource
        resource_types = list(request.requirements.keys())
        if not resource_types:
            return None
        
        selected_resource = np.random.choice(resource_types)
        
        # Quantum fluctuation amplitude based on temperature
        fluctuation_amplitude = temperature / self.annealing_temperature * 0.2
        
        # Get current amount and constraints
        current_amount = neighbor[selected_resource]
        min_amount = request.min_requirements[selected_resource]
        max_amount = request.max_requirements[selected_resource]
        
        # Apply quantum fluctuation
        quantum_delta = np.random.normal(0, fluctuation_amplitude) * request.requirements[selected_resource]
        new_amount = current_amount + quantum_delta
        
        # Enforce constraints
        new_amount = max(min_amount, min(max_amount, new_amount))
        
        # Check resource availability
        resource_name = selected_resource.value
        if resource_name in self.quantum_resources:
            resource = self.quantum_resources[resource_name]
            available_capacity = resource.available + current_allocation.get(selected_resource, 0.0)
            
            if new_amount <= available_capacity:
                neighbor[selected_resource] = new_amount
                return neighbor
        
        return None
    
    def _calculate_allocation_energy(self, allocation: Dict[ResourceType, float]) -> float:
        """Calculate quantum energy of resource allocation state."""
        
        total_energy = 0.0
        
        for resource_type, allocated_amount in allocation.items():
            resource_name = resource_type.value
            
            if resource_name in self.quantum_resources:
                resource = self.quantum_resources[resource_name]
                
                # Base energy from utilization
                utilization = (resource.allocated + allocated_amount) / resource.total_capacity
                utilization_energy = utilization ** 2  # Quadratic penalty for high utilization
                
                # Quantum efficiency factor
                quantum_factor = abs(resource.quantum_efficiency) ** 2
                
                # Entanglement energy - lower energy for balanced entangled resources
                entanglement_energy = 0.0
                for entangled_resource_name in resource.entanglement_links:
                    if entangled_resource_name in self.quantum_resources:
                        entangled_resource = self.quantum_resources[entangled_resource_name]
                        entangled_utilization = entangled_resource.allocated / entangled_resource.total_capacity
                        
                        # Penalty for imbalanced entangled resources
                        imbalance = abs(utilization - entangled_utilization)
                        entanglement_energy += imbalance ** 2
                
                # Combine energy components
                resource_energy = (utilization_energy / quantum_factor) + (0.1 * entanglement_energy)
                total_energy += resource_energy
        
        return total_energy
    
    async def _is_allocation_valid(self, 
                                 allocation: Dict[ResourceType, float],
                                 request: ResourceRequest) -> bool:
        """Validate resource allocation against constraints."""
        
        # Check minimum requirements
        for resource_type, min_amount in request.min_requirements.items():
            if allocation.get(resource_type, 0.0) < min_amount:
                return False
        
        # Check maximum limits
        for resource_type, max_amount in request.max_requirements.items():
            if allocation.get(resource_type, 0.0) > max_amount:
                return False
        
        # Check resource availability
        for resource_type, amount in allocation.items():
            resource_name = resource_type.value
            if resource_name in self.quantum_resources:
                resource = self.quantum_resources[resource_name]
                if amount > resource.available:
                    return False
        
        return True
    
    async def release_resources(self, request_id: str) -> bool:
        """Release allocated resources."""
        
        if request_id not in self.active_allocations:
            logger.warning(f"Attempt to release unknown allocation: {request_id}")
            return False
        
        allocation = self.active_allocations[request_id]
        
        # Release resources
        for resource_type, amount in allocation.items():
            resource_name = resource_type.value
            if resource_name in self.quantum_resources:
                self.quantum_resources[resource_name].allocated -= amount
                # Ensure no negative allocation
                self.quantum_resources[resource_name].allocated = max(
                    0.0, self.quantum_resources[resource_name].allocated
                )
        
        # Remove from active allocations
        del self.active_allocations[request_id]
        
        logger.info(f"Released resources for allocation {request_id}")
        return True
    
    async def optimize_pending_requests(self) -> List[str]:
        """Batch optimize pending resource requests using quantum algorithms."""
        
        if not self.pending_requests:
            return []
        
        logger.info(f"Optimizing {len(self.pending_requests)} pending resource requests")
        
        # Sort by priority and deadline
        sorted_requests = sorted(
            self.pending_requests,
            key=lambda r: (r.priority * -1, r.deadline or float('inf'))
        )
        
        allocated_request_ids = []
        
        for request in sorted_requests:
            allocation_result = await self._quantum_allocate(request)
            
            if allocation_result["success"]:
                # Record allocation
                allocation_amounts = {}
                for resource_type, amount in allocation_result["allocation"].items():
                    resource_name = resource_type.value
                    allocation_amounts[resource_type] = amount
                    self.quantum_resources[resource_name].allocated += amount
                
                self.active_allocations[request.request_id] = allocation_amounts
                allocated_request_ids.append(request.request_id)
        
        # Remove allocated requests from pending
        self.pending_requests = [
            req for req in self.pending_requests 
            if req.request_id not in allocated_request_ids
        ]
        
        logger.info(f"Successfully allocated resources for {len(allocated_request_ids)} requests")
        return allocated_request_ids
    
    async def get_resource_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resource metrics and quantum states."""
        
        metrics = {
            "timestamp": time.time(),
            "total_resources": {},
            "active_allocations_count": len(self.active_allocations),
            "pending_requests_count": len(self.pending_requests),
            "quantum_states": {}
        }
        
        for resource_name, resource in self.quantum_resources.items():
            metrics["total_resources"][resource_name] = {
                "total_capacity": resource.total_capacity,
                "allocated": resource.allocated,
                "available": resource.available,
                "utilization_percent": resource.utilization,
                "quantum_efficiency": {
                    "real": resource.quantum_efficiency.real,
                    "imag": resource.quantum_efficiency.imag,
                    "amplitude": abs(resource.quantum_efficiency)
                },
                "energy_state": resource.energy_state,
                "entanglement_links": list(resource.entanglement_links)
            }
            
            metrics["quantum_states"][resource_name] = {
                "quantum_amplitude": resource.quantum_state_amplitude,
                "phase": math.atan2(
                    resource.quantum_efficiency.imag,
                    resource.quantum_efficiency.real
                )
            }
        
        return metrics
    
    async def predict_resource_needs(self, 
                                   forecast_horizon: float = 3600.0) -> Dict[str, float]:
        """Predict future resource needs using quantum pattern analysis."""
        
        # Analyze current allocation patterns
        current_patterns = {}
        total_allocations = len(self.active_allocations)
        
        if total_allocations == 0:
            return {resource.value: 0.0 for resource in ResourceType}
        
        # Calculate average allocation patterns
        for resource_type in ResourceType:
            total_allocated = 0.0
            count = 0
            
            for allocation in self.active_allocations.values():
                if resource_type in allocation:
                    total_allocated += allocation[resource_type]
                    count += 1
            
            average_allocation = total_allocated / max(1, count)
            current_patterns[resource_type.value] = average_allocation
        
        # Apply quantum prediction using current quantum states
        predicted_needs = {}
        
        for resource_name, resource in self.quantum_resources.items():
            current_avg = current_patterns.get(resource_name, 0.0)
            
            # Quantum amplification based on resource efficiency
            quantum_factor = abs(resource.quantum_efficiency)
            
            # Predict based on quantum state evolution
            # Resources with higher phase tend to show more dynamic allocation patterns
            phase = math.atan2(resource.quantum_efficiency.imag, resource.quantum_efficiency.real)
            volatility_factor = 1.0 + abs(math.sin(phase)) * 0.5
            
            # Forecast considering pending requests
            pending_demand = sum(
                req.requirements.get(ResourceType(resource_name), 0.0)
                for req in self.pending_requests
            )
            
            # Quantum prediction formula
            predicted_allocation = (
                current_avg * quantum_factor * volatility_factor +
                pending_demand * 0.7  # 70% probability of pending requests being fulfilled
            )
            
            predicted_needs[resource_name] = predicted_allocation
        
        return predicted_needs
    
    async def rebalance_quantum_resources(self):
        """Rebalance quantum resource states for optimal efficiency."""
        
        logger.info("Starting quantum resource rebalancing")
        
        # Calculate target quantum states based on current utilization
        for resource_name, resource in self.quantum_resources.items():
            utilization = resource.utilization / 100.0  # Convert to 0-1 scale
            
            # Adjust quantum efficiency based on utilization patterns
            # High utilization resources get higher real component for stability
            # Low utilization resources get higher imaginary component for flexibility
            target_real = 0.5 + (utilization * 0.5)
            target_imag = 0.5 - (utilization * 0.3)
            
            # Normalize to unit circle
            magnitude = math.sqrt(target_real**2 + target_imag**2)
            if magnitude > 0:
                target_real /= magnitude
                target_imag /= magnitude
            
            # Gradually adjust quantum efficiency (avoid sudden state changes)
            current_real = resource.quantum_efficiency.real
            current_imag = resource.quantum_efficiency.imag
            
            # Quantum state interpolation
            alpha = 0.1  # Interpolation factor
            new_real = current_real * (1 - alpha) + target_real * alpha
            new_imag = current_imag * (1 - alpha) + target_imag * alpha
            
            resource.quantum_efficiency = complex(new_real, new_imag)
            
            # Update energy state
            resource.energy_state = abs(resource.quantum_efficiency) ** 2 * utilization
        
        logger.info("Completed quantum resource rebalancing")
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)