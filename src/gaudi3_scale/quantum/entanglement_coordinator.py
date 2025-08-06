"""Quantum Entanglement Coordinator for Distributed HPU Cluster Management.

Implements quantum entanglement patterns for coordinating distributed
operations across HPU clusters with quantum-inspired synchronization.
"""

import asyncio
import logging
import cmath
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from collections import defaultdict
import uuid

from ..exceptions import EntanglementError, ValidationError
from ..validation import DataValidator
from ..monitoring.performance import PerformanceMonitor

logger = logging.getLogger(__name__)


class EntanglementType(Enum):
    """Types of quantum entanglement between cluster components."""
    TASK_DEPENDENCY = "task_dependency"
    RESOURCE_CORRELATION = "resource_correlation" 
    NODE_SYNCHRONIZATION = "node_synchronization"
    DATA_COHERENCE = "data_coherence"
    LOAD_BALANCING = "load_balancing"
    FAULT_TOLERANCE = "fault_tolerance"


class EntanglementState(Enum):
    """States of quantum entanglement."""
    UNENTANGLED = "unentangled"
    CREATING = "creating"
    ENTANGLED = "entangled"
    MEASURING = "measuring"
    COLLAPSED = "collapsed"
    DECOHERENT = "decoherent"


@dataclass
class QuantumEntity:
    """Quantum representation of cluster entity (node, task, resource)."""
    entity_id: str
    entity_type: str  # "node", "task", "resource", "data_partition"
    quantum_state: complex = field(default_factory=lambda: complex(1.0, 0.0))
    phase: float = 0.0
    entanglement_partners: Set[str] = field(default_factory=set)
    entanglement_strength: Dict[str, float] = field(default_factory=dict)
    last_interaction: float = 0.0
    coherence_time: float = 300.0  # Quantum coherence time in seconds
    
    @property
    def probability_amplitude(self) -> float:
        """Calculate probability amplitude |ψ|²."""
        return abs(self.quantum_state) ** 2
    
    @property
    def is_coherent(self) -> bool:
        """Check if quantum state is still coherent."""
        return (time.time() - self.last_interaction) < self.coherence_time
    
    def apply_phase_shift(self, phase_shift: float):
        """Apply quantum phase shift."""
        self.phase += phase_shift
        self.quantum_state *= complex(np.cos(phase_shift), np.sin(phase_shift))
        self.last_interaction = time.time()


@dataclass
class EntanglementPair:
    """Quantum entanglement between two entities."""
    entanglement_id: str
    entity1_id: str
    entity2_id: str
    entanglement_type: EntanglementType
    state: EntanglementState = EntanglementState.UNENTANGLED
    strength: float = 1.0  # Entanglement strength (0-1)
    created_at: float = field(default_factory=time.time)
    last_synchronized: float = field(default_factory=time.time)
    bell_state: str = "phi_plus"  # Bell state type: phi_plus, phi_minus, psi_plus, psi_minus
    correlation_coefficient: float = 1.0
    decoherence_rate: float = 0.01  # Per second
    
    @property
    def is_active(self) -> bool:
        """Check if entanglement is active."""
        return self.state in [EntanglementState.ENTANGLED, EntanglementState.CREATING]
    
    @property
    def age(self) -> float:
        """Age of entanglement in seconds."""
        return time.time() - self.created_at
    
    def update_strength(self):
        """Update entanglement strength considering decoherence."""
        if self.is_active:
            # Apply decoherence over time
            time_factor = np.exp(-self.decoherence_rate * self.age)
            self.strength *= time_factor


class EntanglementCoordinator:
    """Quantum entanglement coordinator for distributed HPU clusters."""
    
    def __init__(self, 
                 cluster_size: int = 8,
                 default_coherence_time: float = 300.0,
                 enable_decoherence: bool = True,
                 synchronization_period: float = 1.0):
        """Initialize entanglement coordinator.
        
        Args:
            cluster_size: Number of nodes in cluster
            default_coherence_time: Default quantum coherence time (seconds)
            enable_decoherence: Enable quantum decoherence effects
            synchronization_period: Period for entanglement synchronization
        """
        self.cluster_size = cluster_size
        self.default_coherence_time = default_coherence_time
        self.enable_decoherence = enable_decoherence
        self.synchronization_period = synchronization_period
        
        # Quantum entity registry
        self.quantum_entities: Dict[str, QuantumEntity] = {}
        
        # Entanglement registry
        self.entanglements: Dict[str, EntanglementPair] = {}
        self.entanglement_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Bell state patterns for different entanglement types
        self.bell_states = {
            "phi_plus": np.array([1, 0, 0, 1]) / np.sqrt(2),    # |00⟩ + |11⟩
            "phi_minus": np.array([1, 0, 0, -1]) / np.sqrt(2),  # |00⟩ - |11⟩
            "psi_plus": np.array([0, 1, 1, 0]) / np.sqrt(2),    # |01⟩ + |10⟩
            "psi_minus": np.array([0, 1, -1, 0]) / np.sqrt(2),  # |01⟩ - |10⟩
        }
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Synchronization task
        self._sync_task = None
        self._running = False
        
        logger.info(f"Initialized EntanglementCoordinator for {cluster_size} node cluster")
    
    async def start(self):
        """Start the entanglement coordination system."""
        if self._running:
            return
        
        self._running = True
        
        # Start background synchronization task
        self._sync_task = asyncio.create_task(self._synchronization_loop())
        
        logger.info("Started quantum entanglement coordination")
    
    async def stop(self):
        """Stop the entanglement coordination system."""
        self._running = False
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped quantum entanglement coordination")
    
    async def register_entity(self, 
                            entity_id: str,
                            entity_type: str,
                            initial_state: complex = None,
                            coherence_time: float = None) -> QuantumEntity:
        """Register quantum entity in the system."""
        
        # Validate inputs
        validator = DataValidator()
        if not validator.validate_string(entity_id, min_length=1):
            raise ValidationError(f"Invalid entity_id: {entity_id}")
        
        if entity_id in self.quantum_entities:
            raise ValidationError(f"Entity {entity_id} already registered")
        
        # Create quantum entity
        entity = QuantumEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            quantum_state=initial_state or complex(1.0, 0.0),
            coherence_time=coherence_time or self.default_coherence_time
        )
        
        self.quantum_entities[entity_id] = entity
        self.entanglement_graph[entity_id] = set()
        
        logger.info(f"Registered quantum entity {entity_id} of type {entity_type}")
        return entity
    
    async def create_entanglement(self,
                                entity1_id: str,
                                entity2_id: str,
                                entanglement_type: EntanglementType,
                                strength: float = 1.0,
                                bell_state: str = "phi_plus") -> str:
        """Create quantum entanglement between two entities."""
        
        # Validate entities exist
        if entity1_id not in self.quantum_entities:
            raise ValidationError(f"Entity {entity1_id} not found")
        if entity2_id not in self.quantum_entities:
            raise ValidationError(f"Entity {entity2_id} not found")
        
        if entity1_id == entity2_id:
            raise ValidationError("Cannot entangle entity with itself")
        
        # Check if entanglement already exists
        existing_entanglement = await self._find_entanglement(entity1_id, entity2_id)
        if existing_entanglement:
            logger.warning(f"Entanglement already exists between {entity1_id} and {entity2_id}")
            return existing_entanglement.entanglement_id
        
        # Create entanglement
        entanglement_id = f"entangle_{entity1_id}_{entity2_id}_{uuid.uuid4().hex[:8]}"
        
        entanglement = EntanglementPair(
            entanglement_id=entanglement_id,
            entity1_id=entity1_id,
            entity2_id=entity2_id,
            entanglement_type=entanglement_type,
            state=EntanglementState.CREATING,
            strength=strength,
            bell_state=bell_state
        )
        
        # Initialize entanglement process
        await self._initialize_entanglement(entanglement)
        
        # Register entanglement
        self.entanglements[entanglement_id] = entanglement
        self.entanglement_graph[entity1_id].add(entity2_id)
        self.entanglement_graph[entity2_id].add(entity1_id)
        
        # Update entity entanglement records
        entity1 = self.quantum_entities[entity1_id]
        entity2 = self.quantum_entities[entity2_id]
        
        entity1.entanglement_partners.add(entity2_id)
        entity2.entanglement_partners.add(entity1_id)
        entity1.entanglement_strength[entity2_id] = strength
        entity2.entanglement_strength[entity1_id] = strength
        
        logger.info(f"Created {entanglement_type.value} entanglement between {entity1_id} and {entity2_id}")
        return entanglement_id
    
    async def _initialize_entanglement(self, entanglement: EntanglementPair):
        """Initialize quantum entanglement between two entities."""
        
        entity1 = self.quantum_entities[entanglement.entity1_id]
        entity2 = self.quantum_entities[entanglement.entity2_id]
        
        # Create Bell state based on entanglement type
        if entanglement.entanglement_type == EntanglementType.TASK_DEPENDENCY:
            # Sequential dependency: |01⟩ + |10⟩ (one task executes before other)
            bell_state = "psi_plus"
        elif entanglement.entanglement_type == EntanglementType.NODE_SYNCHRONIZATION:
            # Synchronization: |00⟩ + |11⟩ (nodes in same state)
            bell_state = "phi_plus"  
        elif entanglement.entanglement_type == EntanglementType.RESOURCE_CORRELATION:
            # Resource sharing: |00⟩ - |11⟩ (anti-correlated resources)
            bell_state = "phi_minus"
        elif entanglement.entanglement_type == EntanglementType.LOAD_BALANCING:
            # Load balancing: |01⟩ - |10⟩ (complementary loads)
            bell_state = "psi_minus"
        else:
            bell_state = entanglement.bell_state
        
        entanglement.bell_state = bell_state
        
        # Calculate entangled quantum states
        # For simplicity, use phase correlation
        base_phase = np.random.uniform(0, 2 * np.pi)
        
        if bell_state in ["phi_plus", "phi_minus"]:
            # Same phase for both entities (correlated)
            entity1.phase = base_phase
            entity2.phase = base_phase
            correlation_sign = 1.0 if bell_state == "phi_plus" else -1.0
        else:
            # Opposite phases (anti-correlated)
            entity1.phase = base_phase
            entity2.phase = base_phase + np.pi
            correlation_sign = 1.0 if bell_state == "psi_plus" else -1.0
        
        entanglement.correlation_coefficient = correlation_sign * entanglement.strength
        
        # Update quantum states
        entity1.quantum_state = complex(
            np.cos(entity1.phase),
            np.sin(entity1.phase)
        ) * np.sqrt(entanglement.strength)
        
        entity2.quantum_state = complex(
            np.cos(entity2.phase),
            np.sin(entity2.phase) * correlation_sign
        ) * np.sqrt(entanglement.strength)
        
        # Mark entanglement as active
        entanglement.state = EntanglementState.ENTANGLED
        entanglement.last_synchronized = time.time()
    
    async def _find_entanglement(self, entity1_id: str, entity2_id: str) -> Optional[EntanglementPair]:
        """Find existing entanglement between two entities."""
        
        for entanglement in self.entanglements.values():
            if ((entanglement.entity1_id == entity1_id and entanglement.entity2_id == entity2_id) or
                (entanglement.entity1_id == entity2_id and entanglement.entity2_id == entity1_id)):
                return entanglement
        
        return None
    
    async def synchronize_entangled_entities(self, entity_id: str) -> Dict[str, Any]:
        """Synchronize an entity with all its entangled partners."""
        
        if entity_id not in self.quantum_entities:
            raise ValidationError(f"Entity {entity_id} not found")
        
        entity = self.quantum_entities[entity_id]
        synchronization_results = {}
        
        # Synchronize with each entangled partner
        for partner_id in entity.entanglement_partners:
            if partner_id in self.quantum_entities:
                sync_result = await self._synchronize_pair(entity_id, partner_id)
                synchronization_results[partner_id] = sync_result
        
        logger.info(f"Synchronized entity {entity_id} with {len(synchronization_results)} partners")
        return synchronization_results
    
    async def _synchronize_pair(self, entity1_id: str, entity2_id: str) -> Dict[str, Any]:
        """Synchronize a pair of entangled entities."""
        
        entanglement = await self._find_entanglement(entity1_id, entity2_id)
        if not entanglement or not entanglement.is_active:
            return {"synchronized": False, "reason": "No active entanglement"}
        
        entity1 = self.quantum_entities[entity1_id]
        entity2 = self.quantum_entities[entity2_id]
        
        # Check coherence
        if not entity1.is_coherent or not entity2.is_coherent:
            entanglement.state = EntanglementState.DECOHERENT
            return {"synchronized": False, "reason": "Decoherent states"}
        
        # Apply entanglement correlation
        correlation = entanglement.correlation_coefficient
        
        # Calculate phase synchronization
        phase_diff = entity1.phase - entity2.phase
        target_phase_diff = 0.0 if correlation > 0 else np.pi
        
        # Synchronize phases gradually to maintain stability
        sync_rate = 0.1  # Gradual synchronization
        phase_correction = (target_phase_diff - phase_diff) * sync_rate
        
        entity1.apply_phase_shift(phase_correction / 2)
        entity2.apply_phase_shift(-phase_correction / 2)
        
        # Update entanglement strength (apply decoherence if enabled)
        if self.enable_decoherence:
            entanglement.update_strength()
            
            # Update entity entanglement strengths
            entity1.entanglement_strength[entity2_id] = entanglement.strength
            entity2.entanglement_strength[entity1_id] = entanglement.strength
            
            # Check for complete decoherence
            if entanglement.strength < 0.01:
                entanglement.state = EntanglementState.DECOHERENT
        
        entanglement.last_synchronized = time.time()
        
        return {
            "synchronized": True,
            "phase_correction": phase_correction,
            "entanglement_strength": entanglement.strength,
            "correlation_coefficient": correlation
        }
    
    async def measure_entanglement(self, entity1_id: str, entity2_id: str) -> Dict[str, Any]:
        """Perform quantum measurement on entangled pair."""
        
        entanglement = await self._find_entanglement(entity1_id, entity2_id)
        if not entanglement:
            raise EntanglementError(f"No entanglement found between {entity1_id} and {entity2_id}")
        
        entity1 = self.quantum_entities[entity1_id]
        entity2 = self.quantum_entities[entity2_id]
        
        # Mark as measuring
        entanglement.state = EntanglementState.MEASURING
        
        # Quantum measurement probabilities based on Bell state
        bell_state_vector = self.bell_states[entanglement.bell_state]
        probabilities = np.abs(bell_state_vector) ** 2
        
        # Perform measurement
        measurement_outcome = np.random.choice(4, p=probabilities)
        
        # Decode measurement outcome
        measurement_states = {
            0: (0, 0),  # |00⟩
            1: (0, 1),  # |01⟩
            2: (1, 0),  # |10⟩
            3: (1, 1),  # |11⟩
        }
        
        entity1_state, entity2_state = measurement_states[measurement_outcome]
        
        # Collapse quantum states
        entity1.quantum_state = complex(1.0, 0.0) if entity1_state == 0 else complex(0.0, 1.0)
        entity2.quantum_state = complex(1.0, 0.0) if entity2_state == 0 else complex(0.0, 1.0)
        
        # Update phases
        entity1.phase = 0.0 if entity1_state == 0 else np.pi / 2
        entity2.phase = 0.0 if entity2_state == 0 else np.pi / 2
        
        # Mark entanglement as collapsed
        entanglement.state = EntanglementState.COLLAPSED
        
        measurement_result = {
            "measurement_outcome": measurement_outcome,
            "entity1_state": entity1_state,
            "entity2_state": entity2_state,
            "bell_state": entanglement.bell_state,
            "measurement_probabilities": probabilities.tolist(),
            "correlation_observed": (entity1_state == entity2_state) if entanglement.correlation_coefficient > 0 else (entity1_state != entity2_state)
        }
        
        logger.info(f"Measured entanglement between {entity1_id} and {entity2_id}: outcome {measurement_outcome}")
        return measurement_result
    
    async def break_entanglement(self, entity1_id: str, entity2_id: str) -> bool:
        """Break quantum entanglement between two entities."""
        
        entanglement = await self._find_entanglement(entity1_id, entity2_id)
        if not entanglement:
            return False
        
        # Remove from registries
        if entanglement.entanglement_id in self.entanglements:
            del self.entanglements[entanglement.entanglement_id]
        
        # Update entanglement graph
        self.entanglement_graph[entity1_id].discard(entity2_id)
        self.entanglement_graph[entity2_id].discard(entity1_id)
        
        # Update entities
        if entity1_id in self.quantum_entities:
            entity1 = self.quantum_entities[entity1_id]
            entity1.entanglement_partners.discard(entity2_id)
            entity1.entanglement_strength.pop(entity2_id, None)
        
        if entity2_id in self.quantum_entities:
            entity2 = self.quantum_entities[entity2_id]
            entity2.entanglement_partners.discard(entity1_id)
            entity2.entanglement_strength.pop(entity1_id, None)
        
        logger.info(f"Broke entanglement between {entity1_id} and {entity2_id}")
        return True
    
    async def _synchronization_loop(self):
        """Background synchronization loop for maintaining entanglements."""
        
        while self._running:
            try:
                # Synchronize all active entanglements
                synchronization_tasks = []
                
                for entanglement in self.entanglements.values():
                    if entanglement.is_active:
                        task = asyncio.create_task(
                            self._synchronize_pair(entanglement.entity1_id, entanglement.entity2_id)
                        )
                        synchronization_tasks.append(task)
                
                # Wait for all synchronizations
                if synchronization_tasks:
                    await asyncio.gather(*synchronization_tasks)
                
                # Clean up decoherent entanglements
                await self._cleanup_decoherent_entanglements()
                
                # Wait for next synchronization period
                await asyncio.sleep(self.synchronization_period)
                
            except Exception as e:
                logger.error(f"Error in synchronization loop: {e}")
                await asyncio.sleep(self.synchronization_period)
    
    async def _cleanup_decoherent_entanglements(self):
        """Remove decoherent entanglements from the system."""
        
        decoherent_entanglements = [
            entanglement_id for entanglement_id, entanglement in self.entanglements.items()
            if entanglement.state == EntanglementState.DECOHERENT
        ]
        
        for entanglement_id in decoherent_entanglements:
            entanglement = self.entanglements[entanglement_id]
            await self.break_entanglement(entanglement.entity1_id, entanglement.entity2_id)
            logger.info(f"Cleaned up decoherent entanglement {entanglement_id}")
    
    async def get_entanglement_metrics(self) -> Dict[str, Any]:
        """Get comprehensive entanglement system metrics."""
        
        total_entities = len(self.quantum_entities)
        total_entanglements = len(self.entanglements)
        
        # Count entanglements by state
        state_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for entanglement in self.entanglements.values():
            state_counts[entanglement.state.value] += 1
            type_counts[entanglement.entanglement_type.value] += 1
        
        # Calculate average entanglement strength
        active_entanglements = [e for e in self.entanglements.values() if e.is_active]
        avg_strength = np.mean([e.strength for e in active_entanglements]) if active_entanglements else 0.0
        
        # Calculate network connectivity
        total_possible_entanglements = total_entities * (total_entities - 1) // 2
        connectivity = total_entanglements / max(1, total_possible_entanglements)
        
        # Calculate coherent entities
        coherent_entities = sum(1 for entity in self.quantum_entities.values() if entity.is_coherent)
        
        return {
            "total_entities": total_entities,
            "coherent_entities": coherent_entities,
            "total_entanglements": total_entanglements,
            "active_entanglements": len(active_entanglements),
            "entanglement_states": dict(state_counts),
            "entanglement_types": dict(type_counts),
            "average_entanglement_strength": avg_strength,
            "network_connectivity": connectivity,
            "synchronization_period": self.synchronization_period,
            "decoherence_enabled": self.enable_decoherence,
            "system_running": self._running
        }
    
    async def get_entity_status(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed status of a quantum entity."""
        
        if entity_id not in self.quantum_entities:
            raise ValidationError(f"Entity {entity_id} not found")
        
        entity = self.quantum_entities[entity_id]
        
        # Get entanglement details
        entanglement_details = {}
        for partner_id in entity.entanglement_partners:
            entanglement = await self._find_entanglement(entity_id, partner_id)
            if entanglement:
                entanglement_details[partner_id] = {
                    "entanglement_type": entanglement.entanglement_type.value,
                    "strength": entanglement.strength,
                    "state": entanglement.state.value,
                    "bell_state": entanglement.bell_state,
                    "age": entanglement.age,
                    "last_synchronized": entanglement.last_synchronized
                }
        
        return {
            "entity_id": entity_id,
            "entity_type": entity.entity_type,
            "quantum_state": {
                "real": entity.quantum_state.real,
                "imag": entity.quantum_state.imag,
                "amplitude": abs(entity.quantum_state),
                "phase": entity.phase
            },
            "probability_amplitude": entity.probability_amplitude,
            "is_coherent": entity.is_coherent,
            "coherence_time": entity.coherence_time,
            "entanglement_partners": list(entity.entanglement_partners),
            "entanglement_details": entanglement_details,
            "last_interaction": entity.last_interaction
        }
    
    async def simulate_quantum_teleportation(self, 
                                           source_entity_id: str,
                                           target_entity_id: str,
                                           auxiliary_entity_id: str) -> Dict[str, Any]:
        """Simulate quantum teleportation protocol between entities."""
        
        # Validate entities
        for entity_id in [source_entity_id, target_entity_id, auxiliary_entity_id]:
            if entity_id not in self.quantum_entities:
                raise ValidationError(f"Entity {entity_id} not found")
        
        # Create entanglement between auxiliary and target
        entanglement_id = await self.create_entanglement(
            auxiliary_entity_id, target_entity_id,
            EntanglementType.DATA_COHERENCE,
            strength=1.0, bell_state="phi_plus"
        )
        
        # Get source entity state
        source_entity = self.quantum_entities[source_entity_id]
        original_state = source_entity.quantum_state
        original_phase = source_entity.phase
        
        # Perform Bell measurement on source and auxiliary
        auxiliary_entity = self.quantum_entities[auxiliary_entity_id]
        
        # Simplified teleportation: transfer state information
        measurement_result = await self.measure_entanglement(source_entity_id, auxiliary_entity_id)
        
        # Apply correction based on measurement outcome
        target_entity = self.quantum_entities[target_entity_id]
        
        # Transfer quantum state (simplified)
        target_entity.quantum_state = original_state
        target_entity.phase = original_phase
        target_entity.last_interaction = time.time()
        
        # Destroy source state (no-cloning theorem)
        source_entity.quantum_state = complex(0.0, 0.0)
        source_entity.phase = 0.0
        
        teleportation_result = {
            "success": True,
            "source_entity": source_entity_id,
            "target_entity": target_entity_id,
            "auxiliary_entity": auxiliary_entity_id,
            "original_state": {
                "real": original_state.real,
                "imag": original_state.imag,
                "phase": original_phase
            },
            "measurement_result": measurement_result,
            "entanglement_used": entanglement_id
        }
        
        logger.info(f"Quantum teleportation from {source_entity_id} to {target_entity_id} completed")
        return teleportation_result
    
    def __del__(self):
        """Clean up resources."""
        if self._running:
            asyncio.create_task(self.stop())