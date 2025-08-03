"""Job scheduling and resource allocation algorithms."""

import logging
import heapq
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ResourceRequirement:
    """Resource requirements for a job."""
    hpus: int = 1
    memory_gb: int = 32
    storage_gb: int = 100
    network_bandwidth_gbps: float = 10.0
    max_runtime_hours: float = 24.0


@dataclass
class Job:
    """Training job representation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    config: Dict[str, Any] = field(default_factory=dict)
    submitted_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration_hours: float = 1.0
    user_id: str = ""
    cost_budget: float = 1000.0
    
    @property
    def runtime_hours(self) -> float:
        """Calculate current runtime in hours."""
        if not self.started_at:
            return 0.0
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds() / 3600
    
    @property
    def wait_time_hours(self) -> float:
        """Calculate wait time in queue."""
        start_time = self.started_at or datetime.now()
        return (start_time - self.submitted_at).total_seconds() / 3600


@dataclass
class ResourceNode:
    """Compute node resource representation."""
    id: str
    hpus: int = 8
    memory_gb: int = 256
    available_hpus: int = 8
    available_memory_gb: int = 256
    status: str = "available"  # available, busy, maintenance
    instance_type: str = "dl2q.24xlarge"
    cost_per_hour: float = 32.77
    location: str = "us-east-1"
    
    @property
    def utilization(self) -> float:
        """Calculate current utilization percentage."""
        hpu_util = (self.hpus - self.available_hpus) / self.hpus
        memory_util = (self.memory_gb - self.available_memory_gb) / self.memory_gb
        return max(hpu_util, memory_util) * 100


class JobScheduler:
    """Intelligent job scheduler for training workloads.
    
    Implements various scheduling algorithms including priority-based,
    fair-share, and cost-optimized scheduling.
    """
    
    def __init__(self, scheduling_policy: str = "priority_fair"):
        """Initialize job scheduler.
        
        Args:
            scheduling_policy: Scheduling policy (priority_fair, fifo, cost_optimized)
        """
        self.scheduling_policy = scheduling_policy
        self.job_queue: List[Job] = []
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: List[Job] = []
        self.user_quotas: Dict[str, Dict[str, float]] = {}
        
    def submit_job(self, job: Job) -> str:
        """Submit a job to the scheduler.
        
        Args:
            job: Job to submit
            
        Returns:
            Job ID
        """
        job.submitted_at = datetime.now()
        job.status = JobStatus.PENDING
        
        # Validate resource requirements
        if not self._validate_job_requirements(job):
            raise ValueError(f"Invalid resource requirements for job {job.id}")
        
        # Check user quotas
        if not self._check_user_quota(job):
            raise ValueError(f"User {job.user_id} exceeds quota limits")
        
        # Add to queue based on scheduling policy
        if self.scheduling_policy == "priority_fair":
            self._insert_by_priority(job)
        elif self.scheduling_policy == "cost_optimized":
            self._insert_by_cost_efficiency(job)
        else:  # fifo
            self.job_queue.append(job)
        
        logger.info(f"Job {job.id} submitted to queue (position: {len(self.job_queue)})")
        return job.id
    
    def schedule_jobs(self, available_resources: List[ResourceNode]) -> List[Tuple[Job, ResourceNode]]:
        """Schedule pending jobs to available resources.
        
        Args:
            available_resources: List of available compute nodes
            
        Returns:
            List of (job, node) assignments
        """
        assignments = []
        available_nodes = [node for node in available_resources if node.status == "available"]
        
        # Sort jobs based on scheduling policy
        scheduled_jobs = self._get_scheduled_jobs()
        
        for job in scheduled_jobs:
            # Find suitable node for job
            suitable_node = self._find_suitable_node(job, available_nodes)
            
            if suitable_node:
                # Assign job to node
                if self._assign_job_to_node(job, suitable_node):
                    assignments.append((job, suitable_node))
                    available_nodes.remove(suitable_node)
                    self.job_queue.remove(job)
                    self.running_jobs[job.id] = job
                    
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now()
                    
                    logger.info(f"Scheduled job {job.id} on node {suitable_node.id}")
        
        return assignments
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                         completion_info: Optional[Dict[str, Any]] = None) -> bool:
        """Update job status.
        
        Args:
            job_id: Job identifier
            status: New job status
            completion_info: Optional completion information
            
        Returns:
            True if update successful
        """
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]
            job.status = status
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.completed_at = datetime.now()
                self.completed_jobs.append(job)
                del self.running_jobs[job_id]
                
                if completion_info:
                    job.config.update(completion_info)
                
                logger.info(f"Job {job_id} completed with status {status.value}")
            
            return True
        
        return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status.
        
        Returns:
            Queue status information
        """
        pending_jobs = len(self.job_queue)
        running_jobs = len(self.running_jobs)
        completed_jobs = len(self.completed_jobs)
        
        # Calculate average wait times
        if self.completed_jobs:
            avg_wait_time = sum(job.wait_time_hours for job in self.completed_jobs) / len(self.completed_jobs)
            avg_runtime = sum(job.runtime_hours for job in self.completed_jobs) / len(self.completed_jobs)
        else:
            avg_wait_time = 0
            avg_runtime = 0
        
        # Queue statistics by priority
        priority_stats = {}
        for priority in JobPriority:
            priority_jobs = [job for job in self.job_queue if job.priority == priority]
            priority_stats[priority.name] = len(priority_jobs)
        
        return {
            "pending_jobs": pending_jobs,
            "running_jobs": running_jobs,
            "completed_jobs": completed_jobs,
            "avg_wait_time_hours": avg_wait_time,
            "avg_runtime_hours": avg_runtime,
            "queue_by_priority": priority_stats,
            "scheduling_policy": self.scheduling_policy
        }
    
    def get_user_quota_status(self, user_id: str) -> Dict[str, Any]:
        """Get user quota status.
        
        Args:
            user_id: User identifier
            
        Returns:
            User quota information
        """
        if user_id not in self.user_quotas:
            return {"error": "User not found"}
        
        quota = self.user_quotas[user_id]
        
        # Calculate current usage
        user_running_jobs = [job for job in self.running_jobs.values() if job.user_id == user_id]
        current_hpus = sum(job.requirements.hpus for job in user_running_jobs)
        current_cost = sum(job.cost_budget for job in user_running_jobs)
        
        return {
            "user_id": user_id,
            "quota_hpus": quota.get("max_hpus", 0),
            "quota_cost_per_month": quota.get("max_cost_per_month", 0),
            "current_hpus": current_hpus,
            "current_monthly_cost": current_cost,
            "hpu_utilization": current_hpus / quota.get("max_hpus", 1) * 100,
            "cost_utilization": current_cost / quota.get("max_cost_per_month", 1) * 100
        }
    
    def _validate_job_requirements(self, job: Job) -> bool:
        """Validate job resource requirements."""
        req = job.requirements
        
        # Check reasonable bounds
        if req.hpus < 1 or req.hpus > 128:
            return False
        if req.memory_gb < 1 or req.memory_gb > 1024:
            return False
        if req.max_runtime_hours < 0.1 or req.max_runtime_hours > 168:  # 1 week max
            return False
        
        return True
    
    def _check_user_quota(self, job: Job) -> bool:
        """Check if job fits within user quotas."""
        if job.user_id not in self.user_quotas:
            # No quota set, allow job
            return True
        
        quota = self.user_quotas[job.user_id]
        
        # Check HPU quota
        user_running_jobs = [j for j in self.running_jobs.values() if j.user_id == job.user_id]
        current_hpus = sum(j.requirements.hpus for j in user_running_jobs)
        
        if current_hpus + job.requirements.hpus > quota.get("max_hpus", float('inf')):
            return False
        
        # Check cost quota
        current_cost = sum(j.cost_budget for j in user_running_jobs)
        if current_cost + job.cost_budget > quota.get("max_cost_per_month", float('inf')):
            return False
        
        return True
    
    def _insert_by_priority(self, job: Job) -> None:
        """Insert job by priority and fair-share."""
        # Calculate user fair share
        user_running_jobs = [j for j in self.running_jobs.values() if j.user_id == job.user_id]
        user_resource_usage = sum(j.requirements.hpus for j in user_running_jobs)
        
        # Insert position based on priority and fairness
        insert_pos = len(self.job_queue)
        
        for i, queued_job in enumerate(self.job_queue):
            # Higher priority jobs go first
            if job.priority.value > queued_job.priority.value:
                insert_pos = i
                break
            
            # Within same priority, consider fairness
            if job.priority == queued_job.priority:
                other_user_running = [j for j in self.running_jobs.values() if j.user_id == queued_job.user_id]
                other_user_usage = sum(j.requirements.hpus for j in other_user_running)
                
                if user_resource_usage < other_user_usage:
                    insert_pos = i
                    break
        
        self.job_queue.insert(insert_pos, job)
    
    def _insert_by_cost_efficiency(self, job: Job) -> None:
        """Insert job by cost efficiency."""
        # Calculate cost efficiency (performance per dollar)
        estimated_tokens = job.config.get("estimated_tokens", 1_000_000)
        cost_efficiency = estimated_tokens / job.cost_budget
        
        # Insert by cost efficiency
        insert_pos = len(self.job_queue)
        
        for i, queued_job in enumerate(self.job_queue):
            other_tokens = queued_job.config.get("estimated_tokens", 1_000_000)
            other_efficiency = other_tokens / queued_job.cost_budget
            
            if cost_efficiency > other_efficiency:
                insert_pos = i
                break
        
        self.job_queue.insert(insert_pos, job)
    
    def _get_scheduled_jobs(self) -> List[Job]:
        """Get jobs in scheduling order."""
        if self.scheduling_policy == "cost_optimized":
            # Sort by cost efficiency
            return sorted(self.job_queue, 
                         key=lambda j: j.config.get("estimated_tokens", 1) / j.cost_budget, 
                         reverse=True)
        else:
            # Return queue order (already sorted)
            return self.job_queue.copy()
    
    def _find_suitable_node(self, job: Job, nodes: List[ResourceNode]) -> Optional[ResourceNode]:
        """Find suitable node for job."""
        req = job.requirements
        
        suitable_nodes = []
        for node in nodes:
            if (node.available_hpus >= req.hpus and 
                node.available_memory_gb >= req.memory_gb and
                node.status == "available"):
                suitable_nodes.append(node)
        
        if not suitable_nodes:
            return None
        
        # Select best node based on policy
        if self.scheduling_policy == "cost_optimized":
            # Choose cheapest node
            return min(suitable_nodes, key=lambda n: n.cost_per_hour)
        else:
            # Choose node with best fit (least waste)
            return min(suitable_nodes, 
                      key=lambda n: (n.available_hpus - req.hpus) + 
                                   (n.available_memory_gb - req.memory_gb))
    
    def _assign_job_to_node(self, job: Job, node: ResourceNode) -> bool:
        """Assign job to node."""
        req = job.requirements
        
        if (node.available_hpus >= req.hpus and 
            node.available_memory_gb >= req.memory_gb):
            
            node.available_hpus -= req.hpus
            node.available_memory_gb -= req.memory_gb
            
            if node.available_hpus == 0:
                node.status = "busy"
            
            return True
        
        return False


class ResourceScheduler:
    """Manages resource allocation and optimization."""
    
    def __init__(self):
        """Initialize resource scheduler."""
        self.nodes: Dict[str, ResourceNode] = {}
        self.resource_pools: Dict[str, List[str]] = {
            "training": [],
            "inference": [],
            "development": []
        }
    
    def add_node(self, node: ResourceNode, pool: str = "training") -> None:
        """Add node to resource pool.
        
        Args:
            node: Resource node to add
            pool: Resource pool name
        """
        self.nodes[node.id] = node
        if pool in self.resource_pools:
            self.resource_pools[pool].append(node.id)
        
        logger.info(f"Added node {node.id} to {pool} pool")
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from all pools.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if node was removed
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            
            # Remove from all pools
            for pool_nodes in self.resource_pools.values():
                if node_id in pool_nodes:
                    pool_nodes.remove(node_id)
            
            logger.info(f"Removed node {node_id}")
            return True
        
        return False
    
    def get_available_resources(self, pool: str = "training") -> List[ResourceNode]:
        """Get available resources in pool.
        
        Args:
            pool: Resource pool name
            
        Returns:
            List of available nodes
        """
        if pool not in self.resource_pools:
            return []
        
        available_nodes = []
        for node_id in self.resource_pools[pool]:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.status in ["available", "busy"] and node.available_hpus > 0:
                    available_nodes.append(node)
        
        return available_nodes
    
    def optimize_resource_allocation(self, jobs: List[Job]) -> Dict[str, Any]:
        """Optimize resource allocation for given jobs.
        
        Args:
            jobs: List of jobs to optimize for
            
        Returns:
            Optimization recommendations
        """
        total_hpu_demand = sum(job.requirements.hpus for job in jobs)
        total_memory_demand = sum(job.requirements.memory_gb for job in jobs)
        
        # Calculate current capacity
        total_hpus = sum(node.hpus for node in self.nodes.values())
        total_memory = sum(node.memory_gb for node in self.nodes.values())
        
        hpu_utilization = (total_hpu_demand / total_hpus * 100) if total_hpus > 0 else 0
        memory_utilization = (total_memory_demand / total_memory * 100) if total_memory > 0 else 0
        
        recommendations = []
        
        # Check if we need more resources
        if hpu_utilization > 90:
            additional_nodes = math.ceil((total_hpu_demand - total_hpus * 0.9) / 8)
            recommendations.append({
                "type": "scale_up",
                "description": f"Add {additional_nodes} nodes to handle HPU demand",
                "urgency": "high" if hpu_utilization > 100 else "medium"
            })
        
        # Check for underutilization
        if hpu_utilization < 30 and len(self.nodes) > 1:
            nodes_to_remove = math.floor((30 - hpu_utilization) / 100 * len(self.nodes))
            recommendations.append({
                "type": "scale_down",
                "description": f"Consider removing {nodes_to_remove} underutilized nodes",
                "urgency": "low"
            })
        
        # Check for imbalanced pools
        pool_utilizations = {}
        for pool_name, node_ids in self.resource_pools.items():
            pool_nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
            if pool_nodes:
                pool_util = sum(n.utilization for n in pool_nodes) / len(pool_nodes)
                pool_utilizations[pool_name] = pool_util
        
        if len(pool_utilizations) > 1:
            max_util_pool = max(pool_utilizations, key=pool_utilizations.get)
            min_util_pool = min(pool_utilizations, key=pool_utilizations.get)
            
            if pool_utilizations[max_util_pool] - pool_utilizations[min_util_pool] > 40:
                recommendations.append({
                    "type": "rebalance",
                    "description": f"Rebalance resources from {min_util_pool} to {max_util_pool} pool",
                    "urgency": "medium"
                })
        
        return {
            "current_utilization": {
                "hpu_percent": hpu_utilization,
                "memory_percent": memory_utilization
            },
            "pool_utilizations": pool_utilizations,
            "recommendations": recommendations,
            "total_capacity": {
                "hpus": total_hpus,
                "memory_gb": total_memory,
                "nodes": len(self.nodes)
            }
        }