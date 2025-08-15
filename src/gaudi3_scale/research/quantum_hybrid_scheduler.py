"""Quantum-Hybrid Task Scheduler for Multi-Node Gaudi 3 Clusters.

This module implements a novel quantum-hybrid scheduling algorithm that combines
classical distributed scheduling with quantum-inspired optimization to achieve
optimal resource allocation across Gaudi 3 HPU clusters.
"""

import logging
import time
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from enum import Enum
from collections import defaultdict, deque
import json
import numpy as np

try:
    import torch
    import torch.distributed as dist
    _torch_available = True
except ImportError:
    torch = None
    dist = None
    _torch_available = False

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ResourceType(Enum):
    """Resource types in the cluster."""
    HPU = "hpu"
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Resource:
    """Resource specification."""
    resource_type: ResourceType
    amount: float
    unit: str = ""
    
    def __str__(self) -> str:
        return f"{self.amount}{self.unit} {self.resource_type.value}"


@dataclass
class Node:
    """Cluster node representation."""
    node_id: str
    node_type: str = "gaudi3"
    available_resources: Dict[ResourceType, float] = field(default_factory=dict)
    total_resources: Dict[ResourceType, float] = field(default_factory=dict)
    current_load: float = 0.0
    health_score: float = 1.0
    last_heartbeat: float = field(default_factory=time.time)
    
    def get_utilization(self, resource_type: ResourceType) -> float:
        """Get resource utilization percentage."""
        total = self.total_resources.get(resource_type, 0.0)
        available = self.available_resources.get(resource_type, 0.0)
        if total == 0:
            return 0.0
        return (total - available) / total
    
    def can_accommodate(self, required_resources: Dict[ResourceType, float]) -> bool:
        """Check if node can accommodate required resources."""
        for resource_type, amount in required_resources.items():
            available = self.available_resources.get(resource_type, 0.0)
            if available < amount:
                return False
        return True
    
    def allocate_resources(self, required_resources: Dict[ResourceType, float]) -> bool:
        """Allocate resources if available."""
        if not self.can_accommodate(required_resources):
            return False
        
        for resource_type, amount in required_resources.items():
            self.available_resources[resource_type] -= amount
        
        return True
    
    def release_resources(self, resources: Dict[ResourceType, float]) -> None:
        """Release allocated resources."""
        for resource_type, amount in resources.items():
            current = self.available_resources.get(resource_type, 0.0)
            total = self.total_resources.get(resource_type, 0.0)
            self.available_resources[resource_type] = min(total, current + amount)


@dataclass
class Task:
    """Training task representation."""
    task_id: str
    name: str
    priority: TaskPriority
    required_resources: Dict[ResourceType, float]
    estimated_duration: float
    dependencies: List[str] = field(default_factory=list)
    affinity_constraints: Dict[str, Any] = field(default_factory=dict)
    state: TaskState = TaskState.PENDING
    assigned_nodes: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    quantum_weight: float = 1.0
    
    def get_resource_demand(self, resource_type: ResourceType) -> float:
        """Get demand for specific resource type."""
        return self.required_resources.get(resource_type, 0.0)
    
    def is_ready_to_schedule(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def get_priority_score(self) -> float:
        """Get numerical priority score (lower is higher priority)."""
        base_score = self.priority.value
        
        # Apply quantum weight adjustment
        quantum_adjustment = (self.quantum_weight - 1.0) * 0.1
        
        return base_score + quantum_adjustment


@dataclass
class SchedulingDecision:
    """Represents a scheduling decision."""
    task_id: str
    assigned_nodes: List[str]
    start_time: float
    expected_completion_time: float
    confidence_score: float
    quantum_score: float = 0.0


class QuantumInspiredOptimizer:
    """Quantum-inspired optimization for task scheduling."""
    
    def __init__(
        self,
        temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.1,
        max_iterations: int = 1000
    ):
        """Initialize quantum optimizer.
        
        Args:
            temperature: Initial temperature for simulated annealing
            cooling_rate: Rate of temperature decrease
            min_temperature: Minimum temperature before stopping
            max_iterations: Maximum optimization iterations
        """
        self.initial_temperature = temperature
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        
    def optimize_schedule(
        self,
        tasks: List[Task],
        nodes: List[Node],
        current_schedule: Dict[str, SchedulingDecision]
    ) -> Dict[str, SchedulingDecision]:
        """Optimize task schedule using quantum-inspired approach.
        
        Args:
            tasks: List of tasks to schedule
            nodes: Available cluster nodes
            current_schedule: Current scheduling decisions
            
        Returns:
            Optimized scheduling decisions
        """
        best_schedule = current_schedule.copy()
        best_score = self._evaluate_schedule(best_schedule, tasks, nodes)
        current_schedule_work = current_schedule.copy()
        
        self.temperature = self.initial_temperature
        
        for iteration in range(self.max_iterations):
            if self.temperature < self.min_temperature:
                break
            
            # Generate neighbor solution
            neighbor_schedule = self._generate_neighbor(current_schedule_work, tasks, nodes)
            neighbor_score = self._evaluate_schedule(neighbor_schedule, tasks, nodes)
            
            # Accept or reject based on quantum-inspired probability
            if self._should_accept(best_score, neighbor_score, self.temperature):
                current_schedule_work = neighbor_schedule
                
                if neighbor_score > best_score:
                    best_schedule = neighbor_schedule.copy()
                    best_score = neighbor_score
                    logger.debug(f"Quantum optimization improved score to {best_score:.4f}")
            
            # Cool down
            self.temperature *= self.cooling_rate
        
        logger.info(f"Quantum optimization completed: final score {best_score:.4f}")
        return best_schedule
    
    def _evaluate_schedule(
        self,
        schedule: Dict[str, SchedulingDecision],
        tasks: List[Task],
        nodes: List[Node]
    ) -> float:
        """Evaluate quality of a schedule."""
        if not schedule:
            return 0.0
        
        # Multi-objective scoring
        scores = []
        
        # 1. Resource utilization efficiency
        utilization_score = self._calculate_utilization_score(schedule, nodes)
        scores.append(0.3 * utilization_score)
        
        # 2. Task priority satisfaction
        priority_score = self._calculate_priority_score(schedule, tasks)
        scores.append(0.25 * priority_score)
        
        # 3. Load balancing
        balance_score = self._calculate_balance_score(schedule, nodes)
        scores.append(0.2 * balance_score)
        
        # 4. Completion time optimization
        time_score = self._calculate_time_score(schedule)
        scores.append(0.15 * time_score)
        
        # 5. Quantum coherence (novel metric for quantum-inspired optimization)
        quantum_score = self._calculate_quantum_score(schedule, tasks)
        scores.append(0.1 * quantum_score)
        
        return sum(scores)
    
    def _calculate_utilization_score(
        self,
        schedule: Dict[str, SchedulingDecision],
        nodes: List[Node]
    ) -> float:
        """Calculate resource utilization efficiency score."""
        if not schedule:
            return 0.0
        
        total_utilization = 0.0
        for node in nodes:
            hpu_util = node.get_utilization(ResourceType.HPU)
            memory_util = node.get_utilization(ResourceType.MEMORY)
            
            # Optimal utilization is around 85% for stability
            hpu_score = 1.0 - abs(hpu_util - 0.85)
            memory_score = 1.0 - abs(memory_util - 0.8)
            
            total_utilization += (hpu_score + memory_score) / 2
        
        return total_utilization / len(nodes) if nodes else 0.0
    
    def _calculate_priority_score(
        self,
        schedule: Dict[str, SchedulingDecision],
        tasks: List[Task]
    ) -> float:
        """Calculate priority satisfaction score."""
        if not tasks:
            return 1.0
        
        priority_satisfaction = 0.0
        for task in tasks:
            if task.task_id in schedule:
                # Higher priority tasks get better scores for being scheduled
                priority_value = 4 - task.priority.value  # Invert priority
                priority_satisfaction += priority_value / 4.0
        
        return priority_satisfaction / len(tasks)
    
    def _calculate_balance_score(
        self,
        schedule: Dict[str, SchedulingDecision],
        nodes: List[Node]
    ) -> float:
        """Calculate load balancing score."""
        if not nodes:
            return 1.0
        
        node_loads = [node.current_load for node in nodes]
        if not node_loads:
            return 1.0
        
        mean_load = np.mean(node_loads)
        load_variance = np.var(node_loads)
        
        # Lower variance is better
        balance_score = 1.0 / (1.0 + load_variance)
        return balance_score
    
    def _calculate_time_score(self, schedule: Dict[str, SchedulingDecision]) -> float:
        """Calculate completion time optimization score."""
        if not schedule:
            return 1.0
        
        completion_times = [decision.expected_completion_time for decision in schedule.values()]
        if not completion_times:
            return 1.0
        
        # Prefer schedules with earlier completion times
        max_completion = max(completion_times)
        avg_completion = np.mean(completion_times)
        
        # Normalize to 0-1 range
        time_score = 1.0 - (avg_completion / max_completion) if max_completion > 0 else 1.0
        return max(0.0, time_score)
    
    def _calculate_quantum_score(
        self,
        schedule: Dict[str, SchedulingDecision],
        tasks: List[Task]
    ) -> float:
        """Calculate quantum coherence score (novel metric)."""
        if not schedule or not tasks:
            return 0.0
        
        # Quantum coherence measures how well task quantum weights align with scheduling
        coherence_sum = 0.0
        for task in tasks:
            if task.task_id in schedule:
                decision = schedule[task.task_id]
                # Higher quantum weights should get better scheduling decisions
                quantum_alignment = task.quantum_weight * decision.confidence_score
                coherence_sum += quantum_alignment
        
        return coherence_sum / len(tasks) if tasks else 0.0
    
    def _generate_neighbor(
        self,
        current_schedule: Dict[str, SchedulingDecision],
        tasks: List[Task],
        nodes: List[Node]
    ) -> Dict[str, SchedulingDecision]:
        """Generate neighbor solution for optimization."""
        neighbor = current_schedule.copy()
        
        if not neighbor:
            return neighbor
        
        # Randomly select a task to reschedule
        task_ids = list(neighbor.keys())
        if not task_ids:
            return neighbor
        
        task_id = np.random.choice(task_ids)
        
        # Find alternative nodes for the task
        task = next((t for t in tasks if t.task_id == task_id), None)
        if not task:
            return neighbor
        
        available_nodes = [n for n in nodes if n.can_accommodate(task.required_resources)]
        if available_nodes:
            # Reassign to a different node
            new_node = np.random.choice(available_nodes)
            decision = neighbor[task_id]
            decision.assigned_nodes = [new_node.node_id]
            decision.confidence_score *= 0.9  # Slightly reduce confidence for change
        
        return neighbor
    
    def _should_accept(self, current_score: float, new_score: float, temperature: float) -> bool:
        """Determine if new solution should be accepted."""
        if new_score > current_score:
            return True
        
        # Accept worse solutions with quantum-inspired probability
        delta = new_score - current_score
        probability = math.exp(delta / temperature)
        return np.random.random() < probability


class QuantumHybridScheduler:
    """Advanced quantum-hybrid scheduler for Gaudi 3 clusters."""
    
    def __init__(
        self,
        cluster_name: str = "gaudi3-cluster",
        scheduling_interval: float = 5.0,
        enable_quantum_optimization: bool = True,
        max_concurrent_tasks: int = 100,
        load_balancing_threshold: float = 0.8
    ):
        """Initialize quantum-hybrid scheduler.
        
        Args:
            cluster_name: Name of the cluster
            scheduling_interval: Scheduling cycle interval in seconds
            enable_quantum_optimization: Enable quantum optimization
            max_concurrent_tasks: Maximum concurrent tasks
            load_balancing_threshold: Load balancing threshold
        """
        self.cluster_name = cluster_name
        self.scheduling_interval = scheduling_interval
        self.enable_quantum_optimization = enable_quantum_optimization
        self.max_concurrent_tasks = max_concurrent_tasks
        self.load_balancing_threshold = load_balancing_threshold
        
        # Core data structures
        self.nodes: Dict[str, Node] = {}
        self.tasks: Dict[str, Task] = {}
        self.pending_tasks: deque = deque()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Dict[str, str] = {}  # task_id -> error_message
        
        # Scheduling state
        self.current_schedule: Dict[str, SchedulingDecision] = {}
        self.scheduling_history: List[Dict[str, Any]] = []
        
        # Quantum optimizer
        self.quantum_optimizer = QuantumInspiredOptimizer() if enable_quantum_optimization else None
        
        # Thread control
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.RLock()
        
        logger.info(f"Initialized QuantumHybridScheduler for cluster '{cluster_name}'")
    
    def add_node(self, node: Node) -> None:
        """Add a node to the cluster.
        
        Args:
            node: Node to add
        """
        with self.lock:
            self.nodes[node.node_id] = node
            logger.info(f"Added node {node.node_id} to cluster")
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the cluster.
        
        Args:
            node_id: ID of node to remove
        """
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                # Reschedule tasks assigned to this node
                self._reschedule_node_tasks(node_id)
                logger.info(f"Removed node {node_id} from cluster")
    
    def submit_task(self, task: Task) -> bool:
        """Submit a task for scheduling.
        
        Args:
            task: Task to submit
            
        Returns:
            True if task was accepted, False otherwise
        """
        with self.lock:
            if len(self.tasks) >= self.max_concurrent_tasks:
                logger.warning(f"Task {task.task_id} rejected: max concurrent tasks reached")
                return False
            
            self.tasks[task.task_id] = task
            self.pending_tasks.append(task.task_id)
            logger.info(f"Submitted task {task.task_id} (priority: {task.priority.name})")
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled, False if not found or already completed
        """
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.state in [TaskState.PENDING, TaskState.SCHEDULED, TaskState.RUNNING]:
                    task.state = TaskState.CANCELLED
                    self._cleanup_task_resources(task_id)
                    logger.info(f"Cancelled task {task_id}")
                    return True
            return False
    
    def start_scheduling(self) -> None:
        """Start the scheduler daemon."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduling_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Started quantum-hybrid scheduler")
    
    def stop_scheduling(self) -> None:
        """Stop the scheduler daemon."""
        if not self.running:
            return
        
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("Stopped quantum-hybrid scheduler")
    
    def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while self.running:
            try:
                self._schedule_pending_tasks()
                self._monitor_running_tasks()
                self._cleanup_completed_tasks()
                
                if self.enable_quantum_optimization:
                    self._run_quantum_optimization()
                
                time.sleep(self.scheduling_interval)
                
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                time.sleep(1.0)  # Brief pause on error
    
    def _schedule_pending_tasks(self) -> None:
        """Schedule pending tasks to available nodes."""
        with self.lock:
            scheduled_count = 0
            
            while self.pending_tasks and scheduled_count < 10:  # Limit per cycle
                task_id = self.pending_tasks.popleft()
                
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                
                # Check if dependencies are satisfied
                if not task.is_ready_to_schedule(self.completed_tasks):
                    # Put back at end of queue
                    self.pending_tasks.append(task_id)
                    continue
                
                # Find suitable nodes
                suitable_nodes = self._find_suitable_nodes(task)
                
                if suitable_nodes:
                    # Create scheduling decision
                    decision = SchedulingDecision(
                        task_id=task_id,
                        assigned_nodes=[node.node_id for node in suitable_nodes],
                        start_time=time.time(),
                        expected_completion_time=time.time() + task.estimated_duration,
                        confidence_score=self._calculate_confidence(task, suitable_nodes)
                    )
                    
                    # Allocate resources
                    for node in suitable_nodes:
                        node.allocate_resources(task.required_resources)
                    
                    # Update task state
                    task.state = TaskState.SCHEDULED
                    task.assigned_nodes = decision.assigned_nodes
                    task.start_time = decision.start_time
                    
                    # Store decision
                    self.current_schedule[task_id] = decision
                    self.running_tasks[task_id] = task
                    
                    scheduled_count += 1
                    logger.info(f"Scheduled task {task_id} on nodes {decision.assigned_nodes}")
                else:
                    # No suitable nodes, try again later
                    self.pending_tasks.append(task_id)
                    break
    
    def _find_suitable_nodes(self, task: Task) -> List[Node]:
        """Find nodes suitable for running the task.
        
        Args:
            task: Task to find nodes for
            
        Returns:
            List of suitable nodes
        """
        suitable_nodes = []
        
        for node in self.nodes.values():
            # Check resource availability
            if not node.can_accommodate(task.required_resources):
                continue
            
            # Check health
            if node.health_score < 0.8:
                continue
            
            # Check load
            if node.current_load > self.load_balancing_threshold:
                continue
            
            # Check affinity constraints
            if not self._check_affinity_constraints(task, node):
                continue
            
            suitable_nodes.append(node)
        
        # Sort by suitability score
        suitable_nodes.sort(key=lambda n: self._calculate_node_suitability(task, n), reverse=True)
        
        # Return required number of nodes (for now, just 1)
        return suitable_nodes[:1] if suitable_nodes else []
    
    def _calculate_node_suitability(self, task: Task, node: Node) -> float:
        """Calculate how suitable a node is for a task.
        
        Args:
            task: Task to evaluate
            node: Node to evaluate
            
        Returns:
            Suitability score (higher is better)
        """
        score = 0.0
        
        # Resource availability score
        hpu_ratio = task.get_resource_demand(ResourceType.HPU) / node.total_resources.get(ResourceType.HPU, 1.0)
        memory_ratio = task.get_resource_demand(ResourceType.MEMORY) / node.total_resources.get(ResourceType.MEMORY, 1.0)
        
        resource_score = 1.0 - max(hpu_ratio, memory_ratio)
        score += 0.4 * resource_score
        
        # Health score
        score += 0.3 * node.health_score
        
        # Load balancing score (prefer less loaded nodes)
        load_score = 1.0 - node.current_load
        score += 0.2 * load_score
        
        # Quantum affinity (novel scoring based on task quantum weight)
        quantum_score = task.quantum_weight * (1.0 - abs(node.current_load - 0.5))
        score += 0.1 * quantum_score
        
        return score
    
    def _check_affinity_constraints(self, task: Task, node: Node) -> bool:
        """Check if node satisfies task affinity constraints.
        
        Args:
            task: Task with constraints
            node: Node to check
            
        Returns:
            True if constraints are satisfied
        """
        constraints = task.affinity_constraints
        
        # Node type constraint
        if "node_type" in constraints:
            if node.node_type != constraints["node_type"]:
                return False
        
        # Zone constraint
        if "zone" in constraints:
            node_zone = getattr(node, "zone", None)
            if node_zone != constraints["zone"]:
                return False
        
        return True
    
    def _calculate_confidence(self, task: Task, nodes: List[Node]) -> float:
        """Calculate confidence score for scheduling decision.
        
        Args:
            task: Task being scheduled
            nodes: Nodes assigned to task
            
        Returns:
            Confidence score between 0 and 1
        """
        if not nodes:
            return 0.0
        
        # Base confidence on resource availability
        total_confidence = 0.0
        for node in nodes:
            resource_confidence = 1.0
            for resource_type, required in task.required_resources.items():
                available = node.available_resources.get(resource_type, 0.0)
                if available > 0:
                    resource_confidence *= min(1.0, available / required)
                else:
                    resource_confidence = 0.0
                    break
            
            total_confidence += resource_confidence * node.health_score
        
        return min(1.0, total_confidence / len(nodes))
    
    def _monitor_running_tasks(self) -> None:
        """Monitor running tasks and update their state."""
        with self.lock:
            current_time = time.time()
            completed_tasks = []
            
            for task_id, task in self.running_tasks.items():
                if task.state == TaskState.RUNNING:
                    # Check if task should be completed (mock completion for now)
                    if (task.start_time and 
                        current_time - task.start_time > task.estimated_duration):
                        task.state = TaskState.COMPLETED
                        task.completion_time = current_time
                        completed_tasks.append(task_id)
                        logger.info(f"Task {task_id} completed")
                
                elif task.state == TaskState.SCHEDULED:
                    # Transition to running
                    task.state = TaskState.RUNNING
                    logger.info(f"Task {task_id} started running")
            
            # Move completed tasks
            for task_id in completed_tasks:
                self.completed_tasks.add(task_id)
                del self.running_tasks[task_id]
                self._cleanup_task_resources(task_id)
    
    def _cleanup_completed_tasks(self) -> None:
        """Cleanup completed tasks from memory."""
        with self.lock:
            # Keep recent history but clean up old completed tasks
            current_time = time.time()
            tasks_to_remove = []
            
            for task_id in list(self.tasks.keys()):
                task = self.tasks[task_id]
                if (task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED] and
                    task.completion_time and
                    current_time - task.completion_time > 3600):  # 1 hour retention
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                self.completed_tasks.discard(task_id)
                if task_id in self.current_schedule:
                    del self.current_schedule[task_id]
    
    def _cleanup_task_resources(self, task_id: str) -> None:
        """Release resources allocated to a task.
        
        Args:
            task_id: ID of task to cleanup
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            for node_id in task.assigned_nodes:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    node.release_resources(task.required_resources)
    
    def _run_quantum_optimization(self) -> None:
        """Run quantum optimization on current schedule."""
        if not self.quantum_optimizer or not self.current_schedule:
            return
        
        with self.lock:
            try:
                # Run quantum optimization
                active_tasks = [task for task in self.tasks.values() 
                              if task.state in [TaskState.SCHEDULED, TaskState.RUNNING]]
                active_nodes = list(self.nodes.values())
                
                if active_tasks and active_nodes:
                    optimized_schedule = self.quantum_optimizer.optimize_schedule(
                        active_tasks, active_nodes, self.current_schedule
                    )
                    
                    # Apply optimizations (simplified - would need proper migration logic)
                    self.current_schedule.update(optimized_schedule)
                    
            except Exception as e:
                logger.error(f"Error in quantum optimization: {e}")
    
    def _reschedule_node_tasks(self, node_id: str) -> None:
        """Reschedule tasks assigned to a removed node.
        
        Args:
            node_id: ID of removed node
        """
        tasks_to_reschedule = []
        
        for task_id, task in self.running_tasks.items():
            if node_id in task.assigned_nodes:
                tasks_to_reschedule.append(task_id)
        
        for task_id in tasks_to_reschedule:
            task = self.tasks[task_id]
            task.state = TaskState.PENDING
            task.assigned_nodes = []
            self.pending_tasks.append(task_id)
            
            if task_id in self.current_schedule:
                del self.current_schedule[task_id]
            
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            logger.info(f"Rescheduled task {task_id} due to node {node_id} removal")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status.
        
        Returns:
            Dictionary containing cluster status information
        """
        with self.lock:
            node_stats = {}
            for node_id, node in self.nodes.items():
                node_stats[node_id] = {
                    "type": node.node_type,
                    "health_score": node.health_score,
                    "current_load": node.current_load,
                    "hpu_utilization": node.get_utilization(ResourceType.HPU),
                    "memory_utilization": node.get_utilization(ResourceType.MEMORY),
                    "last_heartbeat": node.last_heartbeat
                }
            
            task_counts = {
                "pending": len(self.pending_tasks),
                "running": len(self.running_tasks),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "total": len(self.tasks)
            }
            
            return {
                "cluster_name": self.cluster_name,
                "timestamp": time.time(),
                "nodes": node_stats,
                "tasks": task_counts,
                "scheduling_enabled": self.running,
                "quantum_optimization_enabled": self.enable_quantum_optimization,
                "active_schedule_size": len(self.current_schedule)
            }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task.
        
        Args:
            task_id: ID of task to query
            
        Returns:
            Task status dictionary or None if not found
        """
        with self.lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            status = {
                "task_id": task_id,
                "name": task.name,
                "state": task.state.value,
                "priority": task.priority.name,
                "assigned_nodes": task.assigned_nodes,
                "start_time": task.start_time,
                "completion_time": task.completion_time,
                "estimated_duration": task.estimated_duration,
                "quantum_weight": task.quantum_weight
            }
            
            if task_id in self.current_schedule:
                decision = self.current_schedule[task_id]
                status["scheduling_decision"] = {
                    "confidence_score": decision.confidence_score,
                    "expected_completion_time": decision.expected_completion_time,
                    "quantum_score": decision.quantum_score
                }
            
            return status
    
    def save_state(self, filepath: str) -> None:
        """Save scheduler state to file.
        
        Args:
            filepath: File path to save state
        """
        with self.lock:
            state = {
                "cluster_name": self.cluster_name,
                "timestamp": time.time(),
                "nodes": {node_id: {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "available_resources": {rt.value: amount for rt, amount in node.available_resources.items()},
                    "total_resources": {rt.value: amount for rt, amount in node.total_resources.items()},
                    "current_load": node.current_load,
                    "health_score": node.health_score
                } for node_id, node in self.nodes.items()},
                "completed_tasks": list(self.completed_tasks),
                "failed_tasks": self.failed_tasks
            }
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(state, f, indent=2)
                logger.info(f"Scheduler state saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save scheduler state: {e}")


# Factory functions

def create_gaudi3_node(
    node_id: str,
    hpu_count: int = 8,
    memory_gb: int = 96,
    cpu_cores: int = 96
) -> Node:
    """Create a standard Gaudi 3 node configuration.
    
    Args:
        node_id: Unique node identifier
        hpu_count: Number of HPU devices
        memory_gb: Memory in GB
        cpu_cores: Number of CPU cores
        
    Returns:
        Configured Node object
    """
    resources = {
        ResourceType.HPU: float(hpu_count),
        ResourceType.MEMORY: float(memory_gb),
        ResourceType.CPU: float(cpu_cores),
        ResourceType.NETWORK: 200.0,  # 200 Gbps
        ResourceType.STORAGE: 1000.0  # 1TB
    }
    
    return Node(
        node_id=node_id,
        node_type="gaudi3",
        available_resources=resources.copy(),
        total_resources=resources.copy(),
        health_score=1.0
    )


def create_training_task(
    task_id: str,
    name: str,
    hpu_requirement: int = 1,
    memory_gb: int = 8,
    estimated_hours: float = 1.0,
    priority: TaskPriority = TaskPriority.NORMAL
) -> Task:
    """Create a training task with standard requirements.
    
    Args:
        task_id: Unique task identifier
        name: Human-readable task name
        hpu_requirement: Number of HPUs required
        memory_gb: Memory requirement in GB
        estimated_hours: Estimated duration in hours
        priority: Task priority level
        
    Returns:
        Configured Task object
    """
    required_resources = {
        ResourceType.HPU: float(hpu_requirement),
        ResourceType.MEMORY: float(memory_gb),
        ResourceType.CPU: float(hpu_requirement * 4),  # 4 CPU cores per HPU
    }
    
    return Task(
        task_id=task_id,
        name=name,
        priority=priority,
        required_resources=required_resources,
        estimated_duration=estimated_hours * 3600,  # Convert to seconds
        quantum_weight=1.0 + np.random.uniform(-0.2, 0.2)  # Add quantum variation
    )


# Export main classes and functions
__all__ = [
    'QuantumHybridScheduler',
    'Task',
    'Node',
    'TaskPriority',
    'ResourceType',
    'TaskState',
    'SchedulingDecision',
    'QuantumInspiredOptimizer',
    'create_gaudi3_node',
    'create_training_task'
]