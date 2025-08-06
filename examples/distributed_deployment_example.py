#!/usr/bin/env python3
"""
Comprehensive example demonstrating distributed deployment capabilities
for Intel Gaudi 3 HPU clusters using the gaudi3_scale package.

This example shows how to:
1. Set up a distributed cluster
2. Coordinate multi-node training
3. Manage distributed storage
4. Monitor cluster health
5. Handle fault tolerance
6. Orchestrate deployments
7. Manage configurations

Usage:
    python distributed_deployment_example.py
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

from gaudi3_scale.distributed import (
    # Core components
    NodeDiscoveryService,
    ServiceRegistry,
    DistributedTrainingCoordinator,
    ServiceMesh,
    DistributedStorageManager,
    DataManager,
    FaultToleranceManager,
    FailoverCoordinator,
    DistributedMonitor,
    ObservabilityStack,
    DeploymentOrchestrator,
    AutomationEngine,
    DistributedConfigManager,
    ConfigSynchronizer,
    
    # Configuration types
    StorageConfig,
    StorageBackend,
    ReplicationStrategy,
    ConsistencyLevel,
    CompressionType,
    ConfigScope,
    ConfigFormat,
    DeploymentStrategy,
    ResourceType,
    
    # Service types
    ServiceType,
    ServiceStatus,
    ServiceInfo,
    ServiceEndpoint,
    
    # Communication
    ProtocolType,
    MessageType,
    Message,
    
    # Monitoring
    MetricType,
    AlertSeverity
)

from gaudi3_scale.models.cluster import ClusterConfig, NodeConfig, CloudProvider, InstanceType
from gaudi3_scale.logging_utils import get_logger

logger = get_logger(__name__)


class DistributedClusterManager:
    """Main distributed cluster manager orchestrating all components."""
    
    def __init__(self, cluster_config: ClusterConfig):
        """Initialize distributed cluster manager."""
        self.cluster_config = cluster_config
        self.node_id = f"manager_{int(time.time())}"
        
        # Initialize core services
        self.discovery_service = NodeDiscoveryService(
            node_id=self.node_id,
            discovery_port=8500
        )
        
        self.service_registry = ServiceRegistry(self.discovery_service)
        
        # Storage configuration
        storage_config = StorageConfig(
            backend=StorageBackend.LOCAL_FS,
            base_path="/tmp/gaudi3_storage",
            replication_strategy=ReplicationStrategy.MIRROR,
            replication_factor=2,
            consistency_level=ConsistencyLevel.STRONG,
            compression=CompressionType.GZIP,
            encryption_enabled=True
        )
        
        self.storage_manager = DistributedStorageManager(
            storage_config, 
            self.service_registry
        )
        
        self.data_manager = DataManager(self.storage_manager)
        
        # Training coordination
        self.training_coordinator = DistributedTrainingCoordinator(cluster_config)
        
        # Service mesh
        self.service_mesh = ServiceMesh(self.service_registry)
        
        # Fault tolerance
        self.fault_tolerance = FaultToleranceManager(
            self.service_registry,
            self.data_manager
        )
        
        self.failover_coordinator = FailoverCoordinator(
            self.training_coordinator,
            self.fault_tolerance
        )
        
        # Monitoring and observability
        self.observability = ObservabilityStack(
            self.service_registry,
            self.training_coordinator,
            self.data_manager
        )
        
        # Configuration management
        self.config_manager = DistributedConfigManager(
            self.service_registry,
            self.data_manager,
            self.node_id
        )
        
        self.config_sync = ConfigSynchronizer(self.config_manager)
        
        # Deployment orchestration
        self.orchestrator = DeploymentOrchestrator(
            self.service_registry,
            self.config_manager
        )
        
        self.automation_engine = AutomationEngine(self.orchestrator)
        
        logger.info(f"Distributed cluster manager initialized for {cluster_config.cluster_name}")
    
    async def start_cluster(self):
        """Start the distributed cluster."""
        logger.info("Starting distributed Gaudi 3 cluster...")
        
        try:
            # 1. Start node discovery
            logger.info("Starting node discovery service...")
            await self.discovery_service.start()
            
            # Register this node as a coordinator service
            coordinator_service = ServiceInfo(
                service_id=f"coordinator_{self.node_id}",
                service_name="training_coordinator", 
                service_type=ServiceType.TRAINING_COORDINATOR,
                node_id=self.node_id,
                endpoints=[
                    ServiceEndpoint(host="localhost", port=8501, protocol="http")
                ],
                status=ServiceStatus.HEALTHY,
                metadata={"role": "coordinator", "capabilities": ["training", "orchestration"]}
            )
            
            await self.service_registry.register_service(coordinator_service)
            
            # 2. Initialize cluster for training
            logger.info("Initializing training cluster...")
            await self.training_coordinator.initialize_cluster()
            
            # 3. Setup monitoring
            logger.info("Starting monitoring stack...")
            await self.observability.monitor.record_metric(
                "cluster_startup", 1, MetricType.COUNTER,
                labels={"cluster": self.cluster_config.cluster_name}
            )
            
            # 4. Configure fault tolerance
            logger.info("Setting up fault tolerance...")
            
            # Register health checks
            await self.fault_tolerance.register_health_check(
                health_check=type('HealthCheck', (), {
                    'name': 'cluster_connectivity',
                    'check_function': lambda: len(self.discovery_service.nodes) > 0,
                    'interval_seconds': 30,
                    'timeout_seconds': 5,
                    'failure_threshold': 3,
                    'success_threshold': 1,
                    'enabled': True
                })()
            )
            
            # 5. Set initial configurations
            logger.info("Setting up distributed configuration...")
            await self._setup_initial_configs()
            
            # 6. Start automation engine
            logger.info("Starting automation engine...")
            await self.automation_engine.start_automation()
            
            logger.info("Distributed cluster startup completed successfully!")
            
        except Exception as e:
            logger.error(f"Failed to start cluster: {e}")
            raise
    
    async def run_distributed_training(self):
        """Run a distributed training job."""
        logger.info("Starting distributed training job...")
        
        try:
            # 1. Register training dataset
            dataset_id = await self.data_manager.register_dataset(
                name="example_dataset",
                data_path="/tmp/example_data.txt", 
                format_type="text",
                metadata={"description": "Example training dataset"}
            )
            
            logger.info(f"Registered dataset: {dataset_id}")
            
            # 2. Start distributed training
            job_id = await self.training_coordinator.start_distributed_training(
                model_name="example_model",
                dataset_path="/tmp/example_data.txt",
                batch_size=32,
                learning_rate=0.001,
                num_epochs=5
            )
            
            logger.info(f"Started training job: {job_id}")
            
            # 3. Monitor training progress
            start_time = time.time()
            max_training_time = 300  # 5 minutes max
            
            while time.time() - start_time < max_training_time:
                status = await self.training_coordinator.get_training_status(job_id)
                
                if status:
                    logger.info(f"Training progress: {status.get('progress_percent', 0):.1f}%")
                    
                    if status.get("status") == "completed":
                        logger.info("Training completed successfully!")
                        break
                    elif status.get("status") == "failed":
                        logger.error("Training failed!")
                        break
                
                await asyncio.sleep(10)
            
            # 4. Save training checkpoint
            checkpoint_id = await self.data_manager.save_checkpoint(
                training_job_id=job_id,
                model_state={"model_weights": "example_weights"},
                optimizer_state={"optimizer_state": "example_state"},
                step=1000,
                metadata={"final_checkpoint": True}
            )
            
            logger.info(f"Saved final checkpoint: {checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Distributed training failed: {e}")
            raise
    
    async def demonstrate_fault_tolerance(self):
        """Demonstrate fault tolerance capabilities."""
        logger.info("Demonstrating fault tolerance...")
        
        try:
            # 1. Create a backup before simulating failure
            backup_id = await self.fault_tolerance.create_backup(
                backup_type="system_state",
                component="training_coordinator",
                data={"node_count": len(self.training_coordinator.nodes)},
                metadata={"demo_backup": True}
            )
            
            logger.info(f"Created backup: {backup_id}")
            
            # 2. Simulate a node failure
            if self.training_coordinator.nodes:
                failing_node = list(self.training_coordinator.nodes.keys())[0]
                
                await self.fault_tolerance.report_failure(
                    failure_type=self.fault_tolerance.__class__.__module__.split('.')[-1] + ".FailureType.NODE_FAILURE",
                    affected_component=failing_node,
                    description="Simulated node failure for demonstration",
                    severity=self.fault_tolerance.__class__.__module__.split('.')[-1] + ".FailureSeverity.HIGH"
                )
                
                logger.info(f"Simulated failure for node: {failing_node}")
            
            # 3. Check fault tolerance statistics
            stats = self.fault_tolerance.get_failure_statistics()
            logger.info(f"Fault tolerance stats: {stats}")
            
            # 4. Demonstrate recovery
            await asyncio.sleep(5)  # Allow time for recovery
            
            health_status = self.fault_tolerance.get_system_health_status()
            logger.info(f"System health: {health_status['overall_status']}")
            
        except Exception as e:
            logger.error(f"Fault tolerance demonstration failed: {e}")
    
    async def demonstrate_orchestration(self):
        """Demonstrate deployment orchestration."""
        logger.info("Demonstrating deployment orchestration...")
        
        try:
            # 1. Create deployment plan
            deployment_id = await self.orchestrator.create_deployment_plan(
                name="example_deployment",
                cluster_config=self.cluster_config,
                strategy=DeploymentStrategy.ROLLING_UPDATE,
                description="Example distributed deployment"
            )
            
            logger.info(f"Created deployment plan: {deployment_id}")
            
            # 2. Get deployment status
            status = self.orchestrator.get_deployment_status(deployment_id)
            if status:
                logger.info(f"Deployment status: {status['status']}")
                logger.info(f"Total steps: {status['total_steps']}")
            
            # 3. List all deployments
            deployments = self.orchestrator.list_deployments()
            logger.info(f"Found {len(deployments)} deployments")
            
        except Exception as e:
            logger.error(f"Orchestration demonstration failed: {e}")
    
    async def demonstrate_config_management(self):
        """Demonstrate distributed configuration management."""
        logger.info("Demonstrating configuration management...")
        
        try:
            # 1. Set various configurations
            await self.config_manager.set_config(
                "training.batch_size",
                32,
                scope=ConfigScope.GLOBAL,
                format=ConfigFormat.JSON,
                description="Global training batch size"
            )
            
            await self.config_manager.set_config(
                "cluster.max_nodes", 
                8,
                scope=ConfigScope.GLOBAL,
                description="Maximum cluster nodes"
            )
            
            await self.config_manager.set_config(
                "monitoring.metrics_interval",
                30,
                scope=ConfigScope.SERVICE,
                description="Metrics collection interval"
            )
            
            # 2. Retrieve configurations
            batch_size = await self.config_manager.get_config("training.batch_size")
            max_nodes = await self.config_manager.get_config("cluster.max_nodes")
            
            logger.info(f"Retrieved configs - batch_size: {batch_size}, max_nodes: {max_nodes}")
            
            # 3. List all configurations
            configs = await self.config_manager.list_configs()
            logger.info(f"Total configurations: {len(configs)}")
            
            for config in configs[:3]:  # Show first 3
                logger.info(f"  - {config['key_path']}: {config.get('current_value')}")
            
            # 4. Export configurations
            config_export = await self.config_manager.export_configs(
                scope=ConfigScope.GLOBAL,
                format=ConfigFormat.YAML
            )
            
            logger.info("Configuration export preview:")
            logger.info(config_export[:200] + "..." if len(config_export) > 200 else config_export)
            
        except Exception as e:
            logger.error(f"Configuration management demonstration failed: {e}")
    
    async def get_cluster_metrics(self):
        """Get comprehensive cluster metrics."""
        logger.info("Collecting cluster metrics...")
        
        try:
            # 1. Training metrics
            cluster_status = await self.training_coordinator.get_cluster_status()
            logger.info(f"Cluster status: {cluster_status['total_nodes']} nodes, "
                       f"{cluster_status['active_jobs']} active jobs")
            
            # 2. Storage metrics
            storage_stats = self.data_manager.get_storage_stats()
            logger.info(f"Storage: {storage_stats['total_objects']} objects, "
                       f"{storage_stats['utilization_percent']:.1f}% utilized")
            
            # 3. Monitoring dashboard
            dashboard_data = self.observability.monitor.get_monitoring_dashboard_data()
            logger.info(f"System health - CPU: {dashboard_data['system']['cpu_utilization_percent']:.1f}%, "
                       f"Memory: {dashboard_data['system']['memory_utilization_percent']:.1f}%")
            
            # 4. Service registry status
            all_services = self.service_registry.discover_services()
            healthy_services = self.service_registry.discover_services(status=ServiceStatus.HEALTHY)
            
            logger.info(f"Services: {len(healthy_services)}/{len(all_services)} healthy")
            
            # 5. Fault tolerance statistics
            ft_stats = self.fault_tolerance.get_failure_statistics()
            logger.info(f"Fault tolerance: {ft_stats['total_failures']} total failures, "
                       f"MTTR: {ft_stats['mttr_minutes']:.1f} minutes")
            
            # 6. Configuration sync status
            sync_status = self.config_manager.get_sync_status()
            logger.info(f"Config sync: {sync_status['config_count']} configs, "
                       f"last sync: {sync_status['last_sync_time']}")
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    async def _setup_initial_configs(self):
        """Set up initial cluster configurations."""
        initial_configs = {
            "cluster.name": self.cluster_config.cluster_name,
            "cluster.provider": self.cluster_config.provider.value,
            "cluster.region": self.cluster_config.region,
            "training.default_batch_size": 32,
            "training.default_learning_rate": 0.001,
            "monitoring.enabled": True,
            "monitoring.metrics_interval": 30,
            "storage.replication_factor": 2,
            "storage.compression_enabled": True,
            "fault_tolerance.enabled": True,
            "fault_tolerance.backup_interval": 3600
        }
        
        for key, value in initial_configs.items():
            await self.config_manager.set_config(
                key, value, 
                scope=ConfigScope.GLOBAL,
                description=f"Initial cluster configuration for {key}"
            )
    
    async def shutdown_cluster(self):
        """Gracefully shutdown the cluster."""
        logger.info("Shutting down distributed cluster...")
        
        try:
            # 1. Stop automation
            await self.automation_engine.stop_automation()
            
            # 2. Stop active training jobs
            for job_id in list(self.training_coordinator.active_jobs.keys()):
                await self.training_coordinator.stop_training_job(job_id, graceful=True)
            
            # 3. Perform final sync
            await self.config_sync.perform_full_sync()
            
            # 4. Create final backup
            await self.fault_tolerance.create_backup(
                "final_state",
                "cluster_manager",
                {"shutdown_time": datetime.now().isoformat()},
                {"final_backup": True}
            )
            
            # 5. Stop discovery service
            await self.discovery_service.stop()
            
            logger.info("Cluster shutdown completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Main example function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=== Gaudi 3 Scale Distributed Deployment Demo ===")
    
    # Create example cluster configuration
    cluster_config = ClusterConfig(
        cluster_name="gaudi3-distributed-demo",
        provider=CloudProvider.ONPREM,
        region="local",
        nodes=[
            NodeConfig(
                node_id="demo-node-1",
                instance_type=InstanceType.ONPREM_GAUDI3_NODE,
                hpu_count=8,
                memory_gb=96,
                storage_gb=1000,
                network_bandwidth_gbps=25
            ),
            NodeConfig(
                node_id="demo-node-2", 
                instance_type=InstanceType.ONPREM_GAUDI3_NODE,
                hpu_count=8,
                memory_gb=96,
                storage_gb=1000,
                network_bandwidth_gbps=25
            )
        ]
    )
    
    # Initialize cluster manager
    cluster_manager = DistributedClusterManager(cluster_config)
    
    try:
        # 1. Start the cluster
        await cluster_manager.start_cluster()
        await asyncio.sleep(2)  # Allow services to initialize
        
        # 2. Demonstrate distributed training
        logger.info("\n=== Distributed Training Demo ===")
        
        # Create dummy training data
        Path("/tmp/example_data.txt").write_text("example training data\n" * 100)
        
        await cluster_manager.run_distributed_training()
        await asyncio.sleep(2)
        
        # 3. Demonstrate fault tolerance
        logger.info("\n=== Fault Tolerance Demo ===")
        await cluster_manager.demonstrate_fault_tolerance()
        await asyncio.sleep(2)
        
        # 4. Demonstrate orchestration
        logger.info("\n=== Orchestration Demo ===")
        await cluster_manager.demonstrate_orchestration()
        await asyncio.sleep(2)
        
        # 5. Demonstrate configuration management
        logger.info("\n=== Configuration Management Demo ===")
        await cluster_manager.demonstrate_config_management()
        await asyncio.sleep(2)
        
        # 6. Show final metrics
        logger.info("\n=== Final Cluster Metrics ===")
        await cluster_manager.get_cluster_metrics()
        
        # 7. Let the system run for a bit to collect metrics
        logger.info("\nLetting system run for 10 seconds to collect metrics...")
        await asyncio.sleep(10)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
        
    finally:
        # 8. Shutdown gracefully
        logger.info("\n=== Shutting Down ===")
        await cluster_manager.shutdown_cluster()
    
    logger.info("=== Demo completed successfully! ===")


if __name__ == "__main__":
    asyncio.run(main())