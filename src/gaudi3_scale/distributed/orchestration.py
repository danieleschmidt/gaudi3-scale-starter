"""Deployment orchestration and automation for distributed Gaudi 3 clusters."""

import asyncio
import json
import logging
import subprocess
import tempfile
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
import uuid
import shutil
import os

from .discovery import ServiceRegistry, ServiceInfo, ServiceType, ServiceStatus
from .config_management import DistributedConfigManager
from ..models.cluster import ClusterConfig
from ..logging_utils import get_logger
from ..exceptions import Gaudi3ScaleError

logger = get_logger(__name__)


class DeploymentStatus(str, Enum):
    """Deployment status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLING_BACK = "rolling_back"


class DeploymentStrategy(str, Enum):
    """Deployment strategies."""
    RECREATE = "recreate"           # Stop all, then start new
    ROLLING_UPDATE = "rolling"      # Gradual replacement
    BLUE_GREEN = "blue_green"       # Switch between two environments
    CANARY = "canary"              # Gradual traffic shift
    A_B_TESTING = "a_b"            # Parallel versions for testing


class ResourceType(str, Enum):
    """Types of resources that can be deployed."""
    KUBERNETES_MANIFEST = "k8s_manifest"
    TERRAFORM_CONFIG = "terraform"
    DOCKER_COMPOSE = "docker_compose"
    ANSIBLE_PLAYBOOK = "ansible"
    HELM_CHART = "helm"
    CUSTOM_SCRIPT = "custom_script"


@dataclass
class DeploymentStep:
    """Represents a single step in a deployment."""
    step_id: str
    name: str
    resource_type: ResourceType
    resource_path: str
    dependencies: List[str]  # Step IDs this depends on
    timeout_minutes: int = 30
    retry_count: int = 3
    rollback_command: Optional[str] = None
    environment_variables: Dict[str, str] = None
    
    # State tracking
    status: DeploymentStatus = DeploymentStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    attempt_count: int = 0
    
    def __post_init__(self):
        if self.environment_variables is None:
            self.environment_variables = {}
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_complete(self) -> bool:
        return self.status in [DeploymentStatus.COMPLETED, DeploymentStatus.FAILED, DeploymentStatus.CANCELLED]


@dataclass
class DeploymentPlan:
    """Represents a complete deployment plan."""
    deployment_id: str
    name: str
    description: str
    cluster_config: ClusterConfig
    strategy: DeploymentStrategy
    steps: List[DeploymentStep]
    environment: str = "production"
    
    # Rollback configuration
    rollback_enabled: bool = True
    rollback_timeout_minutes: int = 60
    
    # State tracking
    status: DeploymentStatus = DeploymentStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.started_at:
            end_time = self.completed_at or datetime.now()
            return end_time - self.started_at
        return None
    
    @property
    def progress_percent(self) -> float:
        if not self.steps:
            return 0.0
        
        completed_steps = len([s for s in self.steps if s.status == DeploymentStatus.COMPLETED])
        return (completed_steps / len(self.steps)) * 100


@dataclass
class AutomationRule:
    """Represents an automation rule for deployments."""
    rule_id: str
    name: str
    trigger_type: str  # schedule, event, metric_threshold
    trigger_config: Dict[str, Any]
    deployment_template: Dict[str, Any]
    enabled: bool = True
    
    # State tracking
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    last_error: Optional[str] = None


class DeploymentOrchestrator:
    """Orchestrates deployment of distributed Gaudi 3 cluster resources."""
    
    def __init__(self, 
                 service_registry: ServiceRegistry,
                 config_manager: DistributedConfigManager):
        """Initialize deployment orchestrator.
        
        Args:
            service_registry: Service registry for managing services
            config_manager: Configuration manager for deployment configs
        """
        self.service_registry = service_registry
        self.config_manager = config_manager
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Deployment state
        self.active_deployments: Dict[str, DeploymentPlan] = {}
        self.deployment_history: List[DeploymentPlan] = []
        self.max_history_size = 100
        
        # Resource management
        self.deployment_templates: Dict[str, Dict[str, Any]] = {}
        self.resource_generators: Dict[ResourceType, callable] = {}
        
        # Execution tracking
        self.step_executors: Dict[ResourceType, callable] = {}
        self.max_concurrent_deployments = 5
        
        # Initialize default executors
        self._initialize_executors()
        self._initialize_templates()
    
    async def create_deployment_plan(self, 
                                   name: str,
                                   cluster_config: ClusterConfig,
                                   strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
                                   description: str = "") -> str:
        """Create a new deployment plan.
        
        Args:
            name: Deployment name
            cluster_config: Target cluster configuration
            strategy: Deployment strategy
            description: Deployment description
            
        Returns:
            Deployment ID
        """
        deployment_id = str(uuid.uuid4())
        
        # Generate deployment steps based on cluster config
        steps = await self._generate_deployment_steps(cluster_config, strategy)
        
        deployment_plan = DeploymentPlan(
            deployment_id=deployment_id,
            name=name,
            description=description,
            cluster_config=cluster_config,
            strategy=strategy,
            steps=steps
        )
        
        self.active_deployments[deployment_id] = deployment_plan
        
        self.logger.info(f"Created deployment plan: {name} ({deployment_id}) with {len(steps)} steps")
        
        return deployment_id
    
    async def execute_deployment(self, deployment_id: str) -> bool:
        """Execute a deployment plan.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            True if deployment successful
        """
        if deployment_id not in self.active_deployments:
            raise Gaudi3ScaleError(f"Deployment {deployment_id} not found")
        
        deployment_plan = self.active_deployments[deployment_id]
        
        if deployment_plan.status != DeploymentStatus.PENDING:
            raise Gaudi3ScaleError(f"Deployment {deployment_id} is not in pending state")
        
        try:
            deployment_plan.status = DeploymentStatus.RUNNING
            deployment_plan.started_at = datetime.now()
            
            self.logger.info(f"Starting deployment: {deployment_plan.name}")
            
            # Execute steps based on strategy
            if deployment_plan.strategy == DeploymentStrategy.ROLLING_UPDATE:
                success = await self._execute_rolling_deployment(deployment_plan)
            elif deployment_plan.strategy == DeploymentStrategy.BLUE_GREEN:
                success = await self._execute_blue_green_deployment(deployment_plan)
            elif deployment_plan.strategy == DeploymentStrategy.RECREATE:
                success = await self._execute_recreate_deployment(deployment_plan)
            elif deployment_plan.strategy == DeploymentStrategy.CANARY:
                success = await self._execute_canary_deployment(deployment_plan)
            else:
                success = await self._execute_sequential_deployment(deployment_plan)
            
            # Update final status
            deployment_plan.status = DeploymentStatus.COMPLETED if success else DeploymentStatus.FAILED
            deployment_plan.completed_at = datetime.now()
            
            # Move to history
            self._move_to_history(deployment_id)
            
            self.logger.info(
                f"Deployment {deployment_plan.name} {'completed' if success else 'failed'} "
                f"in {deployment_plan.duration}"
            )
            
            return success
            
        except Exception as e:
            deployment_plan.status = DeploymentStatus.FAILED
            deployment_plan.completed_at = datetime.now()
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            
            # Attempt rollback if enabled
            if deployment_plan.rollback_enabled:
                await self.rollback_deployment(deployment_id)
            
            return False
    
    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            True if rollback successful
        """
        # Find deployment (check both active and history)
        deployment_plan = self.active_deployments.get(deployment_id)
        if not deployment_plan:
            deployment_plan = next(
                (d for d in self.deployment_history if d.deployment_id == deployment_id),
                None
            )
        
        if not deployment_plan:
            raise Gaudi3ScaleError(f"Deployment {deployment_id} not found")
        
        try:
            deployment_plan.status = DeploymentStatus.ROLLING_BACK
            
            self.logger.info(f"Rolling back deployment: {deployment_plan.name}")
            
            # Execute rollback commands in reverse order
            rollback_steps = [step for step in reversed(deployment_plan.steps) 
                            if step.status == DeploymentStatus.COMPLETED and step.rollback_command]
            
            for step in rollback_steps:
                try:
                    await self._execute_rollback_step(step)
                    self.logger.info(f"Rolled back step: {step.name}")
                except Exception as e:
                    self.logger.error(f"Rollback failed for step {step.name}: {e}")
                    return False
            
            deployment_plan.status = DeploymentStatus.COMPLETED
            
            self.logger.info(f"Rollback completed for deployment: {deployment_plan.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed for deployment {deployment_id}: {e}")
            return False
    
    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel an active deployment.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            True if cancellation successful
        """
        if deployment_id not in self.active_deployments:
            return False
        
        deployment_plan = self.active_deployments[deployment_id]
        
        if deployment_plan.status == DeploymentStatus.RUNNING:
            deployment_plan.status = DeploymentStatus.CANCELLED
            deployment_plan.completed_at = datetime.now()
            
            self.logger.info(f"Cancelled deployment: {deployment_plan.name}")
            
            # Move to history
            self._move_to_history(deployment_id)
            
            return True
        
        return False
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Deployment status information
        """
        # Check active deployments first
        deployment = self.active_deployments.get(deployment_id)
        if not deployment:
            # Check history
            deployment = next(
                (d for d in self.deployment_history if d.deployment_id == deployment_id),
                None
            )
        
        if not deployment:
            return None
        
        return {
            "deployment_id": deployment.deployment_id,
            "name": deployment.name,
            "status": deployment.status.value,
            "strategy": deployment.strategy.value,
            "progress_percent": deployment.progress_percent,
            "created_at": deployment.created_at,
            "started_at": deployment.started_at,
            "completed_at": deployment.completed_at,
            "duration": deployment.duration.total_seconds() if deployment.duration else None,
            "current_step": deployment.current_step,
            "total_steps": len(deployment.steps),
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "status": step.status.value,
                    "duration": step.duration.total_seconds() if step.duration else None,
                    "attempt_count": step.attempt_count,
                    "error_message": step.error_message
                }
                for step in deployment.steps
            ]
        }
    
    def list_deployments(self, include_history: bool = True) -> List[Dict[str, Any]]:
        """List all deployments.
        
        Args:
            include_history: Whether to include completed deployments
            
        Returns:
            List of deployment summaries
        """
        deployments = list(self.active_deployments.values())
        
        if include_history:
            deployments.extend(self.deployment_history)
        
        # Sort by creation time (newest first)
        deployments.sort(key=lambda d: d.created_at, reverse=True)
        
        return [
            {
                "deployment_id": d.deployment_id,
                "name": d.name,
                "status": d.status.value,
                "strategy": d.strategy.value,
                "progress_percent": d.progress_percent,
                "created_at": d.created_at,
                "duration": d.duration.total_seconds() if d.duration else None,
                "cluster_name": d.cluster_config.cluster_name
            }
            for d in deployments
        ]
    
    async def _generate_deployment_steps(self, 
                                        cluster_config: ClusterConfig,
                                        strategy: DeploymentStrategy) -> List[DeploymentStep]:
        """Generate deployment steps based on cluster configuration.
        
        Args:
            cluster_config: Cluster configuration
            strategy: Deployment strategy
            
        Returns:
            List of deployment steps
        """
        steps = []
        
        # Infrastructure provisioning steps
        if cluster_config.provider.value in ["aws", "azure", "gcp"]:
            # Terraform infrastructure
            steps.append(DeploymentStep(
                step_id="terraform_plan",
                name="Generate Terraform Plan",
                resource_type=ResourceType.TERRAFORM_CONFIG,
                resource_path="terraform/infrastructure",
                dependencies=[],
                timeout_minutes=10
            ))
            
            steps.append(DeploymentStep(
                step_id="terraform_apply",
                name="Apply Terraform Configuration",
                resource_type=ResourceType.TERRAFORM_CONFIG,
                resource_path="terraform/infrastructure",
                dependencies=["terraform_plan"],
                timeout_minutes=45,
                rollback_command="terraform destroy -auto-approve"
            ))
        
        # Kubernetes cluster setup
        steps.append(DeploymentStep(
            step_id="k8s_namespace",
            name="Create Kubernetes Namespace",
            resource_type=ResourceType.KUBERNETES_MANIFEST,
            resource_path="k8s/namespace.yaml",
            dependencies=["terraform_apply"] if cluster_config.provider.value != "onprem" else [],
            timeout_minutes=5
        ))
        
        # Configuration setup
        steps.append(DeploymentStep(
            step_id="configmaps",
            name="Deploy Configuration Maps",
            resource_type=ResourceType.KUBERNETES_MANIFEST,
            resource_path="k8s/configmaps",
            dependencies=["k8s_namespace"],
            timeout_minutes=10
        ))
        
        # Storage setup
        steps.append(DeploymentStep(
            step_id="storage_setup",
            name="Setup Distributed Storage",
            resource_type=ResourceType.KUBERNETES_MANIFEST,
            resource_path="k8s/storage",
            dependencies=["configmaps"],
            timeout_minutes=15
        ))
        
        # Service mesh setup
        steps.append(DeploymentStep(
            step_id="service_mesh",
            name="Deploy Service Mesh",
            resource_type=ResourceType.HELM_CHART,
            resource_path="charts/service-mesh",
            dependencies=["storage_setup"],
            timeout_minutes=20
        ))
        
        # Node deployment based on strategy
        if strategy == DeploymentStrategy.ROLLING_UPDATE:
            # Deploy nodes in small batches
            batch_size = max(1, len(cluster_config.nodes) // 3)
            for i in range(0, len(cluster_config.nodes), batch_size):
                batch_nodes = cluster_config.nodes[i:i + batch_size]
                
                steps.append(DeploymentStep(
                    step_id=f"deploy_nodes_batch_{i // batch_size}",
                    name=f"Deploy Node Batch {i // batch_size + 1}",
                    resource_type=ResourceType.KUBERNETES_MANIFEST,
                    resource_path=f"k8s/nodes/batch-{i // batch_size}",
                    dependencies=["service_mesh"],
                    timeout_minutes=30
                ))
        
        elif strategy == DeploymentStrategy.BLUE_GREEN:
            # Deploy all nodes in green environment
            steps.append(DeploymentStep(
                step_id="deploy_green_environment",
                name="Deploy Green Environment",
                resource_type=ResourceType.KUBERNETES_MANIFEST,
                resource_path="k8s/nodes/green",
                dependencies=["service_mesh"],
                timeout_minutes=45
            ))
            
            steps.append(DeploymentStep(
                step_id="switch_traffic",
                name="Switch Traffic to Green",
                resource_type=ResourceType.CUSTOM_SCRIPT,
                resource_path="scripts/switch-traffic.sh",
                dependencies=["deploy_green_environment"],
                timeout_minutes=10
            ))
        
        else:
            # Default: deploy all nodes at once
            steps.append(DeploymentStep(
                step_id="deploy_all_nodes",
                name="Deploy All Nodes",
                resource_type=ResourceType.KUBERNETES_MANIFEST,
                resource_path="k8s/nodes",
                dependencies=["service_mesh"],
                timeout_minutes=60
            ))
        
        # Monitoring setup
        steps.append(DeploymentStep(
            step_id="monitoring_setup",
            name="Setup Monitoring Stack",
            resource_type=ResourceType.HELM_CHART,
            resource_path="charts/monitoring",
            dependencies=[s.step_id for s in steps if "deploy" in s.step_id],
            timeout_minutes=15
        ))
        
        # Health checks and validation
        steps.append(DeploymentStep(
            step_id="health_validation",
            name="Validate Cluster Health",
            resource_type=ResourceType.CUSTOM_SCRIPT,
            resource_path="scripts/validate-cluster.sh",
            dependencies=["monitoring_setup"],
            timeout_minutes=10
        ))
        
        return steps
    
    async def _execute_sequential_deployment(self, deployment_plan: DeploymentPlan) -> bool:
        """Execute deployment steps sequentially.
        
        Args:
            deployment_plan: Deployment plan to execute
            
        Returns:
            True if all steps successful
        """
        for step in deployment_plan.steps:
            # Check if deployment was cancelled
            if deployment_plan.status == DeploymentStatus.CANCELLED:
                return False
            
            # Wait for dependencies
            if not await self._wait_for_dependencies(deployment_plan, step):
                return False
            
            # Execute step
            deployment_plan.current_step = step.step_id
            success = await self._execute_step(step)
            
            if not success:
                self.logger.error(f"Step {step.name} failed")
                return False
        
        return True
    
    async def _execute_rolling_deployment(self, deployment_plan: DeploymentPlan) -> bool:
        """Execute rolling deployment strategy.
        
        Args:
            deployment_plan: Deployment plan to execute
            
        Returns:
            True if deployment successful
        """
        # Group steps by dependencies to find parallelizable batches
        batches = self._group_steps_into_batches(deployment_plan.steps)
        
        for batch in batches:
            # Execute batch steps in parallel
            batch_tasks = []
            for step in batch:
                if deployment_plan.status == DeploymentStatus.CANCELLED:
                    return False
                
                deployment_plan.current_step = step.step_id
                task = asyncio.create_task(self._execute_step(step))
                batch_tasks.append(task)
            
            # Wait for batch completion
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Check if all steps in batch succeeded
            for result in results:
                if isinstance(result, Exception) or not result:
                    return False
        
        return True
    
    async def _execute_blue_green_deployment(self, deployment_plan: DeploymentPlan) -> bool:
        """Execute blue-green deployment strategy.
        
        Args:
            deployment_plan: Deployment plan to execute
            
        Returns:
            True if deployment successful
        """
        # Execute infrastructure and green environment steps
        for step in deployment_plan.steps:
            if deployment_plan.status == DeploymentStatus.CANCELLED:
                return False
            
            deployment_plan.current_step = step.step_id
            success = await self._execute_step(step)
            
            if not success:
                return False
            
            # Add delay after green environment deployment for health checks
            if step.step_id == "deploy_green_environment":
                await asyncio.sleep(30)  # Allow time for services to start
        
        return True
    
    async def _execute_recreate_deployment(self, deployment_plan: DeploymentPlan) -> bool:
        """Execute recreate deployment strategy.
        
        Args:
            deployment_plan: Deployment plan to execute
            
        Returns:
            True if deployment successful
        """
        # First, stop existing services
        await self._stop_existing_services(deployment_plan.cluster_config)
        
        # Wait for complete shutdown
        await asyncio.sleep(10)
        
        # Execute all steps sequentially
        return await self._execute_sequential_deployment(deployment_plan)
    
    async def _execute_canary_deployment(self, deployment_plan: DeploymentPlan) -> bool:
        """Execute canary deployment strategy.
        
        Args:
            deployment_plan: Deployment plan to execute
            
        Returns:
            True if deployment successful
        """
        # Deploy infrastructure and canary nodes (subset)
        canary_steps = [step for step in deployment_plan.steps 
                       if not step.step_id.startswith("deploy_nodes_batch") or 
                       step.step_id == "deploy_nodes_batch_0"]
        
        for step in canary_steps:
            if deployment_plan.status == DeploymentStatus.CANCELLED:
                return False
            
            deployment_plan.current_step = step.step_id
            success = await self._execute_step(step)
            
            if not success:
                return False
        
        # Monitor canary for a period
        await asyncio.sleep(300)  # 5 minutes monitoring
        
        # If canary is healthy, deploy remaining nodes
        remaining_steps = [step for step in deployment_plan.steps 
                          if step.step_id.startswith("deploy_nodes_batch") and 
                          step.step_id != "deploy_nodes_batch_0"]
        
        for step in remaining_steps:
            if deployment_plan.status == DeploymentStatus.CANCELLED:
                return False
            
            deployment_plan.current_step = step.step_id
            success = await self._execute_step(step)
            
            if not success:
                return False
        
        return True
    
    async def _execute_step(self, step: DeploymentStep) -> bool:
        """Execute a single deployment step.
        
        Args:
            step: Deployment step to execute
            
        Returns:
            True if step successful
        """
        step.status = DeploymentStatus.RUNNING
        step.started_at = datetime.now()
        step.attempt_count += 1
        
        try:
            # Get appropriate executor
            executor = self.step_executors.get(step.resource_type)
            if not executor:
                raise Gaudi3ScaleError(f"No executor found for resource type: {step.resource_type}")
            
            # Execute step with timeout
            success = await asyncio.wait_for(
                executor(step),
                timeout=step.timeout_minutes * 60
            )
            
            if success:
                step.status = DeploymentStatus.COMPLETED
            else:
                step.status = DeploymentStatus.FAILED
                
            step.completed_at = datetime.now()
            
            return success
            
        except asyncio.TimeoutError:
            step.status = DeploymentStatus.FAILED
            step.error_message = f"Step timed out after {step.timeout_minutes} minutes"
            step.completed_at = datetime.now()
            
            return False
            
        except Exception as e:
            step.status = DeploymentStatus.FAILED
            step.error_message = str(e)
            step.completed_at = datetime.now()
            
            # Retry if attempts remaining
            if step.attempt_count < step.retry_count:
                self.logger.warning(f"Step {step.name} failed, retrying (attempt {step.attempt_count}/{step.retry_count})")
                await asyncio.sleep(5)  # Brief delay before retry
                return await self._execute_step(step)
            
            return False
    
    async def _wait_for_dependencies(self, 
                                   deployment_plan: DeploymentPlan, 
                                   step: DeploymentStep) -> bool:
        """Wait for step dependencies to complete.
        
        Args:
            deployment_plan: Deployment plan
            step: Step to check dependencies for
            
        Returns:
            True if dependencies satisfied
        """
        if not step.dependencies:
            return True
        
        max_wait_time = 600  # 10 minutes
        check_interval = 5   # 5 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            # Check if all dependencies are complete
            dependencies_complete = True
            
            for dep_step_id in step.dependencies:
                dep_step = next((s for s in deployment_plan.steps if s.step_id == dep_step_id), None)
                if not dep_step or dep_step.status != DeploymentStatus.COMPLETED:
                    dependencies_complete = False
                    break
            
            if dependencies_complete:
                return True
            
            # Check if deployment was cancelled
            if deployment_plan.status == DeploymentStatus.CANCELLED:
                return False
            
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        
        return False
    
    def _group_steps_into_batches(self, steps: List[DeploymentStep]) -> List[List[DeploymentStep]]:
        """Group steps into batches based on dependencies.
        
        Args:
            steps: List of deployment steps
            
        Returns:
            List of step batches that can be executed in parallel
        """
        batches = []
        remaining_steps = steps.copy()
        completed_steps = set()
        
        while remaining_steps:
            # Find steps with satisfied dependencies
            current_batch = []
            
            for step in remaining_steps[:]:
                dependencies_satisfied = all(dep in completed_steps for dep in step.dependencies)
                
                if dependencies_satisfied:
                    current_batch.append(step)
                    remaining_steps.remove(step)
            
            if not current_batch:
                # No steps can be executed - circular dependency or missing step
                raise Gaudi3ScaleError("Circular dependency or missing step detected in deployment plan")
            
            batches.append(current_batch)
            completed_steps.update(step.step_id for step in current_batch)
        
        return batches
    
    async def _stop_existing_services(self, cluster_config: ClusterConfig):
        """Stop existing services before recreate deployment.
        
        Args:
            cluster_config: Cluster configuration
        """
        services = self.service_registry.discover_services()
        
        for service in services:
            try:
                # Stop service (implementation would depend on service type)
                self.logger.info(f"Stopping service: {service.service_name}")
                await asyncio.sleep(1)  # Simulate stopping
            except Exception as e:
                self.logger.error(f"Failed to stop service {service.service_name}: {e}")
    
    async def _execute_rollback_step(self, step: DeploymentStep):
        """Execute rollback command for a step.
        
        Args:
            step: Step to rollback
        """
        if not step.rollback_command:
            return
        
        # Execute rollback command
        process = await asyncio.create_subprocess_shell(
            step.rollback_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **step.environment_variables}
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Gaudi3ScaleError(f"Rollback command failed: {stderr.decode()}")
    
    def _move_to_history(self, deployment_id: str):
        """Move deployment from active to history.
        
        Args:
            deployment_id: Deployment to move
        """
        if deployment_id in self.active_deployments:
            deployment = self.active_deployments[deployment_id]
            self.deployment_history.append(deployment)
            del self.active_deployments[deployment_id]
            
            # Limit history size
            if len(self.deployment_history) > self.max_history_size:
                self.deployment_history.pop(0)
    
    def _initialize_executors(self):
        """Initialize step executors for different resource types."""
        self.step_executors[ResourceType.KUBERNETES_MANIFEST] = self._execute_k8s_manifest
        self.step_executors[ResourceType.TERRAFORM_CONFIG] = self._execute_terraform
        self.step_executors[ResourceType.HELM_CHART] = self._execute_helm_chart
        self.step_executors[ResourceType.CUSTOM_SCRIPT] = self._execute_custom_script
        self.step_executors[ResourceType.DOCKER_COMPOSE] = self._execute_docker_compose
    
    def _initialize_templates(self):
        """Initialize deployment templates."""
        self.deployment_templates["basic_cluster"] = {
            "strategy": DeploymentStrategy.ROLLING_UPDATE.value,
            "steps": [
                {
                    "name": "Infrastructure Setup",
                    "resource_type": ResourceType.TERRAFORM_CONFIG.value,
                    "resource_path": "terraform/basic"
                },
                {
                    "name": "Node Deployment",
                    "resource_type": ResourceType.KUBERNETES_MANIFEST.value,
                    "resource_path": "k8s/nodes"
                }
            ]
        }
    
    async def _execute_k8s_manifest(self, step: DeploymentStep) -> bool:
        """Execute Kubernetes manifest deployment.
        
        Args:
            step: Deployment step
            
        Returns:
            True if successful
        """
        try:
            # Generate manifest content
            manifest_content = await self._generate_k8s_manifest(step)
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(manifest_content)
                manifest_path = f.name
            
            try:
                # Apply manifest
                process = await asyncio.create_subprocess_exec(
                    'kubectl', 'apply', '-f', manifest_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, **step.environment_variables}
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    self.logger.info(f"Successfully applied K8s manifest: {step.name}")
                    return True
                else:
                    self.logger.error(f"kubectl apply failed: {stderr.decode()}")
                    return False
            
            finally:
                # Clean up temporary file
                os.unlink(manifest_path)
                
        except Exception as e:
            self.logger.error(f"K8s manifest execution failed: {e}")
            return False
    
    async def _execute_terraform(self, step: DeploymentStep) -> bool:
        """Execute Terraform configuration.
        
        Args:
            step: Deployment step
            
        Returns:
            True if successful
        """
        try:
            # Generate Terraform configuration
            tf_config = await self._generate_terraform_config(step)
            
            # Create temporary directory for Terraform files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write main configuration
                main_tf_path = os.path.join(temp_dir, 'main.tf')
                with open(main_tf_path, 'w') as f:
                    f.write(tf_config)
                
                # Initialize Terraform
                process = await asyncio.create_subprocess_exec(
                    'terraform', 'init',
                    cwd=temp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, **step.environment_variables}
                )
                
                await process.communicate()
                
                if process.returncode != 0:
                    return False
                
                # Plan or Apply based on step
                command = 'plan' if step.step_id.endswith('_plan') else 'apply'
                args = ['terraform', command]
                if command == 'apply':
                    args.append('-auto-approve')
                
                process = await asyncio.create_subprocess_exec(
                    *args,
                    cwd=temp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, **step.environment_variables}
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    self.logger.info(f"Terraform {command} successful: {step.name}")
                    return True
                else:
                    self.logger.error(f"Terraform {command} failed: {stderr.decode()}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Terraform execution failed: {e}")
            return False
    
    async def _execute_helm_chart(self, step: DeploymentStep) -> bool:
        """Execute Helm chart deployment.
        
        Args:
            step: Deployment step
            
        Returns:
            True if successful
        """
        try:
            chart_name = os.path.basename(step.resource_path)
            
            # Install or upgrade chart
            process = await asyncio.create_subprocess_exec(
                'helm', 'upgrade', '--install', chart_name, step.resource_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **step.environment_variables}
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Helm chart deployed successfully: {step.name}")
                return True
            else:
                self.logger.error(f"Helm deployment failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Helm execution failed: {e}")
            return False
    
    async def _execute_custom_script(self, step: DeploymentStep) -> bool:
        """Execute custom script.
        
        Args:
            step: Deployment step
            
        Returns:
            True if successful
        """
        try:
            # Make script executable
            os.chmod(step.resource_path, 0o755)
            
            # Execute script
            process = await asyncio.create_subprocess_exec(
                step.resource_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **step.environment_variables}
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Custom script executed successfully: {step.name}")
                return True
            else:
                self.logger.error(f"Custom script failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Custom script execution failed: {e}")
            return False
    
    async def _execute_docker_compose(self, step: DeploymentStep) -> bool:
        """Execute Docker Compose deployment.
        
        Args:
            step: Deployment step
            
        Returns:
            True if successful
        """
        try:
            # Deploy using docker-compose
            process = await asyncio.create_subprocess_exec(
                'docker-compose', '-f', step.resource_path, 'up', '-d',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **step.environment_variables}
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Docker Compose deployed successfully: {step.name}")
                return True
            else:
                self.logger.error(f"Docker Compose deployment failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Docker Compose execution failed: {e}")
            return False
    
    async def _generate_k8s_manifest(self, step: DeploymentStep) -> str:
        """Generate Kubernetes manifest content.
        
        Args:
            step: Deployment step
            
        Returns:
            YAML manifest content
        """
        # This would generate actual K8s manifests based on step configuration
        # For now, return a simple example
        manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": "gaudi3-scale"
            }
        }
        
        return yaml.dump(manifest)
    
    async def _generate_terraform_config(self, step: DeploymentStep) -> str:
        """Generate Terraform configuration.
        
        Args:
            step: Deployment step
            
        Returns:
            Terraform configuration content
        """
        # This would generate actual Terraform configuration
        # For now, return a simple example
        return """
provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "gaudi3_node" {
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "dl2q.24xlarge"
  
  tags = {
    Name = "Gaudi3-Training-Node"
  }
}
"""


class AutomationEngine:
    """Automation engine for triggered deployments and operations."""
    
    def __init__(self, orchestrator: DeploymentOrchestrator):
        """Initialize automation engine.
        
        Args:
            orchestrator: Deployment orchestrator
        """
        self.orchestrator = orchestrator
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Automation rules
        self.automation_rules: Dict[str, AutomationRule] = {}
        
        # Scheduler state
        self.scheduler_running = False
        
        # Metric thresholds
        self.metric_watchers: Dict[str, Dict[str, Any]] = {}
        
    def add_automation_rule(self, rule: AutomationRule):
        """Add an automation rule.
        
        Args:
            rule: Automation rule to add
        """
        self.automation_rules[rule.rule_id] = rule
        self.logger.info(f"Added automation rule: {rule.name}")
    
    async def start_automation(self):
        """Start the automation engine."""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        
        # Start scheduler tasks
        asyncio.create_task(self._schedule_processor())
        asyncio.create_task(self._metric_monitor())
        
        self.logger.info("Automation engine started")
    
    async def stop_automation(self):
        """Stop the automation engine."""
        self.scheduler_running = False
        self.logger.info("Automation engine stopped")
    
    async def trigger_rule(self, rule_id: str, context: Dict[str, Any] = None) -> bool:
        """Manually trigger an automation rule.
        
        Args:
            rule_id: Rule identifier
            context: Additional context for the trigger
            
        Returns:
            True if rule executed successfully
        """
        if rule_id not in self.automation_rules:
            return False
        
        rule = self.automation_rules[rule_id]
        
        try:
            success = await self._execute_rule(rule, context or {})
            
            rule.last_triggered = datetime.now()
            rule.trigger_count += 1
            
            if success:
                rule.last_error = None
            
            return success
            
        except Exception as e:
            rule.last_error = str(e)
            self.logger.error(f"Rule execution failed for {rule.name}: {e}")
            return False
    
    def get_automation_statistics(self) -> Dict[str, Any]:
        """Get automation engine statistics.
        
        Returns:
            Statistics dictionary
        """
        total_rules = len(self.automation_rules)
        enabled_rules = len([r for r in self.automation_rules.values() if r.enabled])
        total_triggers = sum(r.trigger_count for r in self.automation_rules.values())
        
        failed_rules = len([r for r in self.automation_rules.values() if r.last_error])
        
        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "disabled_rules": total_rules - enabled_rules,
            "total_triggers": total_triggers,
            "failed_rules": failed_rules,
            "scheduler_running": self.scheduler_running,
            "metric_watchers": len(self.metric_watchers)
        }
    
    async def _schedule_processor(self):
        """Process scheduled automation rules."""
        while self.scheduler_running:
            try:
                current_time = datetime.now()
                
                for rule in self.automation_rules.values():
                    if not rule.enabled or rule.trigger_type != "schedule":
                        continue
                    
                    # Check schedule configuration
                    schedule_config = rule.trigger_config
                    
                    if self._should_trigger_schedule(rule, current_time, schedule_config):
                        await self.trigger_rule(rule.rule_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Schedule processor error: {e}")
                await asyncio.sleep(30)
    
    async def _metric_monitor(self):
        """Monitor metrics for threshold-based triggers."""
        while self.scheduler_running:
            try:
                for rule in self.automation_rules.values():
                    if not rule.enabled or rule.trigger_type != "metric_threshold":
                        continue
                    
                    # Check metric thresholds (simplified)
                    threshold_config = rule.trigger_config
                    
                    if self._should_trigger_metric(rule, threshold_config):
                        await self.trigger_rule(rule.rule_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Metric monitor error: {e}")
                await asyncio.sleep(30)
    
    def _should_trigger_schedule(self, 
                               rule: AutomationRule, 
                               current_time: datetime,
                               schedule_config: Dict[str, Any]) -> bool:
        """Check if scheduled rule should trigger.
        
        Args:
            rule: Automation rule
            current_time: Current time
            schedule_config: Schedule configuration
            
        Returns:
            True if should trigger
        """
        # Simple cron-like scheduling (simplified implementation)
        interval_minutes = schedule_config.get("interval_minutes", 60)
        
        if rule.last_triggered:
            next_trigger = rule.last_triggered + timedelta(minutes=interval_minutes)
            return current_time >= next_trigger
        else:
            return True  # First trigger
    
    def _should_trigger_metric(self, 
                             rule: AutomationRule,
                             threshold_config: Dict[str, Any]) -> bool:
        """Check if metric threshold rule should trigger.
        
        Args:
            rule: Automation rule
            threshold_config: Threshold configuration
            
        Returns:
            True if should trigger
        """
        # Simplified metric threshold checking
        # In real implementation, would check actual metrics
        return False
    
    async def _execute_rule(self, rule: AutomationRule, context: Dict[str, Any]) -> bool:
        """Execute an automation rule.
        
        Args:
            rule: Rule to execute
            context: Execution context
            
        Returns:
            True if execution successful
        """
        deployment_template = rule.deployment_template
        
        # Create deployment from template
        deployment_name = f"auto_{rule.name}_{int(datetime.now().timestamp())}"
        
        # This would create and execute a deployment based on the template
        # For now, just log the action
        self.logger.info(f"Executing automation rule: {rule.name}")
        
        return True