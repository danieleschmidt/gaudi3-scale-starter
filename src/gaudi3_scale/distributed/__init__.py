"""Distributed deployment capabilities for Gaudi 3 Scale.

This module provides comprehensive distributed system capabilities including:
- Multi-node distributed training coordination
- Cluster management and node discovery
- Service mesh and communication protocols
- Distributed storage and data management
- Fault tolerance and failover mechanisms
- Distributed monitoring and observability
- Deployment orchestration and automation
- Distributed configuration management

Generation 3 Feature: Distributed Deployment
"""

from .coordinator import DistributedTrainingCoordinator
from .discovery import NodeDiscoveryService, ServiceRegistry
from .mesh import ServiceMesh, CommunicationProtocol
from .storage import DistributedStorageManager, DataManager
from .fault_tolerance import FaultToleranceManager, FailoverCoordinator
from .observability import DistributedMonitor, ObservabilityStack
from .orchestration import DeploymentOrchestrator, AutomationEngine
from .config_management import DistributedConfigManager, ConfigSynchronizer

__all__ = [
    # Training coordination
    "DistributedTrainingCoordinator",
    
    # Discovery and registry
    "NodeDiscoveryService",
    "ServiceRegistry", 
    
    # Communication
    "ServiceMesh",
    "CommunicationProtocol",
    
    # Storage
    "DistributedStorageManager", 
    "DataManager",
    
    # Fault tolerance
    "FaultToleranceManager",
    "FailoverCoordinator",
    
    # Monitoring
    "DistributedMonitor",
    "ObservabilityStack",
    
    # Orchestration
    "DeploymentOrchestrator", 
    "AutomationEngine",
    
    # Configuration
    "DistributedConfigManager",
    "ConfigSynchronizer"
]