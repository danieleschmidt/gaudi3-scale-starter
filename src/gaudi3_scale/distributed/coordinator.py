"""Distributed training coordination for multi-node Gaudi 3 clusters."""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
import uuid
import json

from ..models.cluster import ClusterConfig, NodeConfig
from ..exceptions import TrainingError, HPUNotAvailableError
from ..logging_utils import get_logger


logger = get_logger(__name__)


class TrainingPhase(str, Enum):
    """Training phases for distributed coordination."""
    INITIALIZATION = "initialization"
    DATA_PREPARATION = "data_preparation"
    MODEL_SETUP = "model_setup"
    TRAINING = "training"
    SYNCHRONIZATION = "synchronization"
    VALIDATION = "validation"
    CHECKPOINTING = "checkpointing"
    COMPLETION = "completion"
    ERROR = "error"


class NodeRole(str, Enum):
    """Node roles in distributed training."""
    COORDINATOR = "coordinator"      # Master node that orchestrates training
    WORKER = "worker"               # Worker node that performs training
    PARAMETER_SERVER = "ps"         # Parameter server for gradient aggregation
    DATA_LOADER = "data_loader"     # Specialized data loading node
    MONITOR = "monitor"             # Monitoring and health check node


class SynchronizationMode(str, Enum):
    """Gradient synchronization modes."""
    ALLREDUCE = "allreduce"         # All-reduce synchronization
    PARAMETER_SERVER = "ps"         # Parameter server synchronization
    RING_ALLREDUCE = "ring"         # Ring all-reduce
    HIERARCHICAL = "hierarchical"   # Hierarchical synchronization
    ASYNC = "async"                 # Asynchronous updates


@dataclass
class NodeStatus:
    """Status of a training node."""
    node_id: str
    role: NodeRole
    phase: TrainingPhase
    hpu_utilization: float
    memory_usage_gb: float
    network_bandwidth_mbps: float
    last_heartbeat: datetime
    error_message: Optional[str] = None
    is_healthy: bool = True
    training_step: int = 0
    batch_processed: int = 0


@dataclass
class TrainingJob:
    """Distributed training job configuration."""
    job_id: str
    model_name: str
    dataset_path: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    sync_mode: SynchronizationMode
    checkpoint_interval: int
    nodes: List[str]
    coordinator_node: str
    created_at: datetime
    status: TrainingPhase = TrainingPhase.INITIALIZATION


class DistributedTrainingCoordinator:
    """Coordinates distributed training across multiple Gaudi 3 nodes."""
    
    def __init__(self, cluster_config: ClusterConfig):
        """Initialize the distributed training coordinator.
        
        Args:
            cluster_config: Cluster configuration
        """
        self.cluster_config = cluster_config
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Node management
        self.nodes: Dict[str, NodeStatus] = {}
        self.node_roles: Dict[str, NodeRole] = {}
        self.coordinator_node: Optional[str] = None
        
        # Job management
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.job_history: List[TrainingJob] = []
        
        # Communication
        self.heartbeat_interval = 30.0  # seconds
        self.heartbeat_timeout = 90.0   # seconds
        self.sync_timeout = 300.0       # seconds
        
        # Performance tracking
        self.global_step = 0
        self.total_batches_processed = 0
        self.training_start_time: Optional[datetime] = None
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        self.logger.info(f"Initialized coordinator for cluster: {cluster_config.cluster_name}")
    
    async def initialize_cluster(self) -> bool:
        """Initialize the distributed training cluster.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing distributed training cluster")
            
            # Assign node roles
            await self._assign_node_roles()
            
            # Initialize nodes
            for node_config in self.cluster_config.nodes:
                await self._initialize_node(node_config)
            
            # Start heartbeat monitoring
            asyncio.create_task(self._heartbeat_monitor())
            
            # Verify cluster readiness
            if not await self._verify_cluster_readiness():
                raise TrainingError("Cluster initialization failed - not all nodes ready")
            
            self.logger.info("Distributed training cluster initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Cluster initialization failed: {e}")
            await self._emit_event("cluster_initialization_failed", {"error": str(e)})
            return False
    
    async def start_distributed_training(
        self,
        model_name: str,
        dataset_path: str,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        sync_mode: SynchronizationMode = SynchronizationMode.ALLREDUCE,
        checkpoint_interval: int = 1000
    ) -> str:
        """Start distributed training job.
        
        Args:
            model_name: Name of the model to train
            dataset_path: Path to training dataset
            batch_size: Training batch size per node
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            sync_mode: Gradient synchronization mode
            checkpoint_interval: Steps between checkpoints
            
        Returns:
            Job ID for the training job
        """
        job_id = str(uuid.uuid4())
        
        try:
            # Create training job
            job = TrainingJob(
                job_id=job_id,
                model_name=model_name,
                dataset_path=dataset_path,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                sync_mode=sync_mode,
                checkpoint_interval=checkpoint_interval,
                nodes=list(self.nodes.keys()),
                coordinator_node=self.coordinator_node,
                created_at=datetime.now()
            )
            
            self.active_jobs[job_id] = job
            self.training_start_time = datetime.now()
            self.global_step = 0
            self.total_batches_processed = 0
            
            self.logger.info(f"Starting distributed training job {job_id}")
            await self._emit_event("training_job_started", {"job_id": job_id, "job": job})
            
            # Initialize training on all nodes
            await self._initialize_training_job(job)
            
            # Start training coordination
            asyncio.create_task(self._coordinate_training(job))
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to start training job: {e}")
            if job_id in self.active_jobs:
                self.active_jobs[job_id].status = TrainingPhase.ERROR
                await self._emit_event("training_job_failed", {"job_id": job_id, "error": str(e)})
            raise TrainingError(f"Failed to start distributed training: {e}")
    
    async def stop_training_job(self, job_id: str, graceful: bool = True) -> bool:
        """Stop a distributed training job.
        
        Args:
            job_id: Training job ID
            graceful: Whether to stop gracefully with final checkpoint
            
        Returns:
            True if stopped successfully
        """
        if job_id not in self.active_jobs:
            raise TrainingError(f"Training job {job_id} not found")
        
        try:
            job = self.active_jobs[job_id]
            self.logger.info(f"Stopping training job {job_id} (graceful={graceful})")
            
            if graceful:
                # Save final checkpoint
                await self._create_checkpoint(job)
            
            # Stop training on all nodes
            await self._stop_training_on_nodes(job.nodes)
            
            # Update job status
            job.status = TrainingPhase.COMPLETION
            self.job_history.append(job)
            del self.active_jobs[job_id]
            
            await self._emit_event("training_job_stopped", {"job_id": job_id, "graceful": graceful})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop training job {job_id}: {e}")
            return False
    
    async def scale_training_cluster(self, target_nodes: int) -> bool:
        """Scale the training cluster up or down.
        
        Args:
            target_nodes: Target number of training nodes
            
        Returns:
            True if scaling successful
        """
        current_nodes = len(self.nodes)
        
        if target_nodes == current_nodes:
            return True
        
        try:
            if target_nodes > current_nodes:
                # Scale up - add nodes
                await self._scale_up_cluster(target_nodes - current_nodes)
            else:
                # Scale down - remove nodes
                await self._scale_down_cluster(current_nodes - target_nodes)
            
            await self._emit_event("cluster_scaled", {
                "from_nodes": current_nodes,
                "to_nodes": target_nodes
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to scale cluster: {e}")
            return False
    
    async def get_training_status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current training status.
        
        Args:
            job_id: Specific job ID, or None for all jobs
            
        Returns:
            Training status information
        """
        if job_id:
            if job_id not in self.active_jobs:
                raise TrainingError(f"Training job {job_id} not found")
            
            job = self.active_jobs[job_id]
            return await self._get_job_status(job)
        else:
            # Return status for all active jobs
            status = {}
            for jid, job in self.active_jobs.items():
                status[jid] = await self._get_job_status(job)
            return status
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status.
        
        Returns:
            Cluster status information
        """
        healthy_nodes = sum(1 for node in self.nodes.values() if node.is_healthy)
        total_hpu_utilization = sum(node.hpu_utilization for node in self.nodes.values())
        avg_hpu_utilization = total_hpu_utilization / len(self.nodes) if self.nodes else 0
        
        return {
            "cluster_name": self.cluster_config.cluster_name,
            "total_nodes": len(self.nodes),
            "healthy_nodes": healthy_nodes,
            "coordinator_node": self.coordinator_node,
            "average_hpu_utilization": avg_hpu_utilization,
            "active_jobs": len(self.active_jobs),
            "global_training_step": self.global_step,
            "total_batches_processed": self.total_batches_processed,
            "uptime": datetime.now() - self.training_start_time if self.training_start_time else None,
            "nodes": {node_id: {
                "role": status.role.value,
                "phase": status.phase.value,
                "hpu_utilization": status.hpu_utilization,
                "memory_usage_gb": status.memory_usage_gb,
                "is_healthy": status.is_healthy,
                "training_step": status.training_step
            } for node_id, status in self.nodes.items()}
        }
    
    def register_event_callback(self, event_type: str, callback: Callable[[Dict], None]):
        """Register callback for training events.
        
        Args:
            event_type: Type of event to listen for
            callback: Callback function to execute
        """
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    async def _assign_node_roles(self):
        """Assign roles to nodes in the cluster."""
        nodes = [node.node_id for node in self.cluster_config.nodes]
        
        if not nodes:
            raise TrainingError("No nodes available for role assignment")
        
        # First node becomes coordinator
        self.coordinator_node = nodes[0]
        self.node_roles[nodes[0]] = NodeRole.COORDINATOR
        
        # Remaining nodes are workers
        for node_id in nodes[1:]:
            self.node_roles[node_id] = NodeRole.WORKER
        
        # For large clusters, assign specialized roles
        if len(nodes) >= 8:
            # Assign parameter servers (every 4th node starting from index 2)
            for i in range(2, len(nodes), 4):
                self.node_roles[nodes[i]] = NodeRole.PARAMETER_SERVER
        
        if len(nodes) >= 16:
            # Assign dedicated data loaders
            self.node_roles[nodes[-1]] = NodeRole.DATA_LOADER
            if len(nodes) >= 32:
                self.node_roles[nodes[-2]] = NodeRole.MONITOR
        
        self.logger.info(f"Assigned node roles: {self.node_roles}")
    
    async def _initialize_node(self, node_config: NodeConfig):
        """Initialize a single node for distributed training.
        
        Args:
            node_config: Node configuration
        """
        node_id = node_config.node_id
        role = self.node_roles.get(node_id, NodeRole.WORKER)
        
        node_status = NodeStatus(
            node_id=node_id,
            role=role,
            phase=TrainingPhase.INITIALIZATION,
            hpu_utilization=0.0,
            memory_usage_gb=0.0,
            network_bandwidth_mbps=0.0,
            last_heartbeat=datetime.now(),
            training_step=0,
            batch_processed=0
        )
        
        self.nodes[node_id] = node_status
        self.logger.info(f"Initialized node {node_id} with role {role.value}")
    
    async def _verify_cluster_readiness(self) -> bool:
        """Verify all nodes are ready for training.
        
        Returns:
            True if all nodes are ready
        """
        ready_nodes = 0
        for node_id, status in self.nodes.items():
            if status.is_healthy and status.phase == TrainingPhase.INITIALIZATION:
                ready_nodes += 1
        
        return ready_nodes == len(self.nodes)
    
    async def _heartbeat_monitor(self):
        """Monitor node heartbeats and health."""
        while True:
            try:
                current_time = datetime.now()
                
                for node_id, status in self.nodes.items():
                    # Check for heartbeat timeout
                    if (current_time - status.last_heartbeat).total_seconds() > self.heartbeat_timeout:
                        if status.is_healthy:
                            self.logger.warning(f"Node {node_id} heartbeat timeout")
                            status.is_healthy = False
                            await self._handle_node_failure(node_id)
                    
                    # Simulate heartbeat updates (in real implementation, nodes would send these)
                    if status.is_healthy:
                        status.last_heartbeat = current_time
                        status.hpu_utilization = min(95.0, status.hpu_utilization + (time.time() % 10) / 100)
                        status.memory_usage_gb = min(
                            node_config.memory_gb * 0.9,
                            status.memory_usage_gb + (time.time() % 5) / 10
                        ) if (node_config := next((n for n in self.cluster_config.nodes if n.node_id == node_id), None)) else status.memory_usage_gb
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _initialize_training_job(self, job: TrainingJob):
        """Initialize training job on all nodes.
        
        Args:
            job: Training job configuration
        """
        self.logger.info(f"Initializing training job {job.job_id} on {len(job.nodes)} nodes")
        
        # Update all nodes to data preparation phase
        for node_id in job.nodes:
            if node_id in self.nodes:
                self.nodes[node_id].phase = TrainingPhase.DATA_PREPARATION
        
        # Simulate data preparation
        await asyncio.sleep(2)
        
        # Update to model setup phase
        for node_id in job.nodes:
            if node_id in self.nodes:
                self.nodes[node_id].phase = TrainingPhase.MODEL_SETUP
        
        # Simulate model setup
        await asyncio.sleep(3)
        
        # Ready for training
        for node_id in job.nodes:
            if node_id in self.nodes:
                self.nodes[node_id].phase = TrainingPhase.TRAINING
        
        job.status = TrainingPhase.TRAINING
    
    async def _coordinate_training(self, job: TrainingJob):
        """Coordinate training execution across nodes.
        
        Args:
            job: Training job to coordinate
        """
        try:
            step = 0
            epoch = 0
            
            while epoch < job.num_epochs and job.status == TrainingPhase.TRAINING:
                # Simulate training step
                await asyncio.sleep(0.1)  # Simulate training time
                
                step += 1
                self.global_step = step
                
                # Update node progress
                for node_id in job.nodes:
                    if node_id in self.nodes and self.nodes[node_id].is_healthy:
                        self.nodes[node_id].training_step = step
                        self.nodes[node_id].batch_processed += 1
                        self.total_batches_processed += 1
                
                # Synchronization phase
                if step % 100 == 0:  # Sync every 100 steps
                    await self._synchronize_gradients(job)
                
                # Checkpointing
                if step % job.checkpoint_interval == 0:
                    await self._create_checkpoint(job)
                
                # Validation
                if step % (job.checkpoint_interval * 2) == 0:
                    await self._run_validation(job)
                
                # Epoch completion
                if step % 1000 == 0:
                    epoch += 1
                    self.logger.info(f"Completed epoch {epoch}/{job.num_epochs}")
            
            # Training completion
            job.status = TrainingPhase.COMPLETION
            await self._emit_event("training_job_completed", {"job_id": job.job_id})
            
        except Exception as e:
            self.logger.error(f"Training coordination failed: {e}")
            job.status = TrainingPhase.ERROR
            await self._emit_event("training_job_failed", {"job_id": job.job_id, "error": str(e)})
    
    async def _synchronize_gradients(self, job: TrainingJob):
        """Synchronize gradients across nodes.
        
        Args:
            job: Training job
        """
        # Update nodes to synchronization phase
        healthy_nodes = [
            node_id for node_id in job.nodes
            if node_id in self.nodes and self.nodes[node_id].is_healthy
        ]
        
        for node_id in healthy_nodes:
            self.nodes[node_id].phase = TrainingPhase.SYNCHRONIZATION
        
        # Simulate gradient synchronization based on sync mode
        if job.sync_mode == SynchronizationMode.ALLREDUCE:
            await asyncio.sleep(0.05 * len(healthy_nodes))  # Simulate allreduce
        elif job.sync_mode == SynchronizationMode.PARAMETER_SERVER:
            await asyncio.sleep(0.1)  # Simulate PS communication
        
        # Return nodes to training phase
        for node_id in healthy_nodes:
            self.nodes[node_id].phase = TrainingPhase.TRAINING
    
    async def _create_checkpoint(self, job: TrainingJob):
        """Create training checkpoint.
        
        Args:
            job: Training job
        """
        # Update nodes to checkpointing phase
        for node_id in job.nodes:
            if node_id in self.nodes and self.nodes[node_id].is_healthy:
                self.nodes[node_id].phase = TrainingPhase.CHECKPOINTING
        
        # Simulate checkpoint creation
        await asyncio.sleep(1)
        
        self.logger.info(f"Created checkpoint for job {job.job_id} at step {self.global_step}")
        
        # Return nodes to training phase
        for node_id in job.nodes:
            if node_id in self.nodes and self.nodes[node_id].is_healthy:
                self.nodes[node_id].phase = TrainingPhase.TRAINING
    
    async def _run_validation(self, job: TrainingJob):
        """Run validation on training job.
        
        Args:
            job: Training job
        """
        # Use coordinator node for validation
        if self.coordinator_node in self.nodes:
            self.nodes[self.coordinator_node].phase = TrainingPhase.VALIDATION
        
        # Simulate validation
        await asyncio.sleep(0.5)
        
        self.logger.info(f"Completed validation for job {job.job_id} at step {self.global_step}")
        
        # Return to training phase
        if self.coordinator_node in self.nodes:
            self.nodes[self.coordinator_node].phase = TrainingPhase.TRAINING
    
    async def _handle_node_failure(self, node_id: str):
        """Handle node failure during training.
        
        Args:
            node_id: Failed node ID
        """
        self.logger.error(f"Handling failure of node {node_id}")
        
        # Mark node as unhealthy
        if node_id in self.nodes:
            self.nodes[node_id].is_healthy = False
            self.nodes[node_id].phase = TrainingPhase.ERROR
        
        # If coordinator fails, elect new coordinator
        if node_id == self.coordinator_node:
            await self._elect_new_coordinator()
        
        await self._emit_event("node_failed", {"node_id": node_id})
    
    async def _elect_new_coordinator(self):
        """Elect new coordinator node after failure."""
        healthy_workers = [
            node_id for node_id, status in self.nodes.items()
            if status.is_healthy and status.role == NodeRole.WORKER
        ]
        
        if healthy_workers:
            new_coordinator = healthy_workers[0]
            self.coordinator_node = new_coordinator
            self.nodes[new_coordinator].role = NodeRole.COORDINATOR
            self.node_roles[new_coordinator] = NodeRole.COORDINATOR
            
            self.logger.info(f"Elected new coordinator: {new_coordinator}")
            await self._emit_event("coordinator_elected", {"node_id": new_coordinator})
    
    async def _scale_up_cluster(self, additional_nodes: int):
        """Scale up cluster by adding nodes.
        
        Args:
            additional_nodes: Number of nodes to add
        """
        # Simulate node addition (in real implementation, would provision infrastructure)
        for i in range(additional_nodes):
            node_id = f"scale-up-node-{uuid.uuid4().hex[:8]}"
            await self._initialize_node(NodeConfig(
                node_id=node_id,
                instance_type=self.cluster_config.nodes[0].instance_type,
                hpu_count=8,
                memory_gb=96,
                storage_gb=100,
                network_bandwidth_gbps=25
            ))
        
        self.logger.info(f"Scaled up cluster by {additional_nodes} nodes")
    
    async def _scale_down_cluster(self, nodes_to_remove: int):
        """Scale down cluster by removing nodes.
        
        Args:
            nodes_to_remove: Number of nodes to remove
        """
        # Remove worker nodes first (keep coordinator)
        worker_nodes = [
            node_id for node_id, status in self.nodes.items()
            if status.role == NodeRole.WORKER and status.is_healthy
        ]
        
        nodes_removed = 0
        for node_id in worker_nodes:
            if nodes_removed >= nodes_to_remove:
                break
            
            # Gracefully stop node
            del self.nodes[node_id]
            del self.node_roles[node_id]
            nodes_removed += 1
        
        self.logger.info(f"Scaled down cluster by {nodes_removed} nodes")
    
    async def _stop_training_on_nodes(self, node_ids: List[str]):
        """Stop training on specified nodes.
        
        Args:
            node_ids: List of node IDs to stop training on
        """
        for node_id in node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].phase = TrainingPhase.COMPLETION
    
    async def _get_job_status(self, job: TrainingJob) -> Dict[str, Any]:
        """Get status for a specific job.
        
        Args:
            job: Training job
            
        Returns:
            Job status information
        """
        active_nodes = [
            node_id for node_id in job.nodes
            if node_id in self.nodes and self.nodes[node_id].is_healthy
        ]
        
        total_batches = sum(
            self.nodes[node_id].batch_processed
            for node_id in active_nodes
        )
        
        return {
            "job_id": job.job_id,
            "model_name": job.model_name,
            "status": job.status.value,
            "global_step": self.global_step,
            "total_batches_processed": total_batches,
            "active_nodes": len(active_nodes),
            "total_nodes": len(job.nodes),
            "sync_mode": job.sync_mode.value,
            "created_at": job.created_at,
            "running_time": datetime.now() - job.created_at,
            "coordinator_node": job.coordinator_node
        }
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered callbacks.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Event callback error: {e}")