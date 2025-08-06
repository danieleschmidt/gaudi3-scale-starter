"""Node discovery and service registry for distributed Gaudi 3 clusters."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
import hashlib
import socket
import uuid

from ..logging_utils import get_logger
from ..exceptions import Gaudi3ScaleError

logger = get_logger(__name__)


class ServiceType(str, Enum):
    """Types of services in the cluster."""
    TRAINING_COORDINATOR = "training_coordinator"
    TRAINING_WORKER = "training_worker"
    PARAMETER_SERVER = "parameter_server"
    DATA_LOADER = "data_loader"
    MODEL_SERVER = "model_server"
    MONITORING = "monitoring"
    STORAGE = "storage"
    API_GATEWAY = "api_gateway"
    MESSAGE_BROKER = "message_broker"
    CONFIG_SERVER = "config_server"


class ServiceStatus(str, Enum):
    """Service status states."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass
class ServiceEndpoint:
    """Service endpoint information."""
    host: str
    port: int
    protocol: str = "tcp"
    path: str = "/"
    
    @property
    def url(self) -> str:
        """Get full URL for the endpoint."""
        if self.protocol in ["http", "https"]:
            return f"{self.protocol}://{self.host}:{self.port}{self.path}"
        else:
            return f"{self.host}:{self.port}"


@dataclass
class ServiceInfo:
    """Information about a service in the cluster."""
    service_id: str
    service_name: str
    service_type: ServiceType
    node_id: str
    endpoints: List[ServiceEndpoint]
    status: ServiceStatus
    metadata: Dict[str, Any]
    health_check_url: Optional[str] = None
    tags: Set[str] = None
    registered_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    version: str = "1.0.0"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.registered_at is None:
            self.registered_at = datetime.now()
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    capabilities: List[str]
    resources: Dict[str, Any]
    status: ServiceStatus
    services: Dict[str, ServiceInfo]
    metadata: Dict[str, Any]
    last_seen: datetime
    discovery_port: int = 8500
    
    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = datetime.now()


class NodeDiscoveryService:
    """Service for discovering and managing cluster nodes."""
    
    def __init__(self, 
                 node_id: str,
                 discovery_port: int = 8500,
                 heartbeat_interval: int = 30,
                 node_timeout: int = 90):
        """Initialize node discovery service.
        
        Args:
            node_id: Unique identifier for this node
            discovery_port: Port for discovery communication
            heartbeat_interval: Heartbeat interval in seconds
            node_timeout: Node timeout in seconds
        """
        self.node_id = node_id
        self.discovery_port = discovery_port
        self.heartbeat_interval = heartbeat_interval
        self.node_timeout = node_timeout
        
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Node tracking
        self.nodes: Dict[str, NodeInfo] = {}
        self.local_node: Optional[NodeInfo] = None
        
        # Discovery state
        self.is_running = False
        self.discovery_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.node_joined_callbacks: List[Callable[[NodeInfo], None]] = []
        self.node_left_callbacks: List[Callable[[str], None]] = []
        self.node_updated_callbacks: List[Callable[[NodeInfo], None]] = []
        
        self.logger.info(f"Initialized node discovery for {node_id}")
    
    async def start(self, bind_address: str = "0.0.0.0") -> bool:
        """Start the node discovery service.
        
        Args:
            bind_address: Address to bind the discovery service to
            
        Returns:
            True if started successfully
        """
        try:
            # Initialize local node information
            await self._initialize_local_node(bind_address)
            
            # Start discovery server
            self.discovery_task = asyncio.create_task(
                self._discovery_server(bind_address)
            )
            
            # Start heartbeat process
            self.heartbeat_task = asyncio.create_task(
                self._heartbeat_process()
            )
            
            # Start node health monitoring
            asyncio.create_task(self._monitor_node_health())
            
            self.is_running = True
            self.logger.info(f"Node discovery started on {bind_address}:{self.discovery_port}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start node discovery: {e}")
            return False
    
    async def stop(self):
        """Stop the node discovery service."""
        self.is_running = False
        
        if self.discovery_task:
            self.discovery_task.cancel()
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        self.logger.info("Node discovery service stopped")
    
    async def discover_peers(self, seed_addresses: List[str]) -> List[NodeInfo]:
        """Discover peer nodes using seed addresses.
        
        Args:
            seed_addresses: List of seed node addresses
            
        Returns:
            List of discovered nodes
        """
        discovered_nodes = []
        
        for address in seed_addresses:
            try:
                nodes = await self._contact_seed_node(address)
                discovered_nodes.extend(nodes)
            except Exception as e:
                self.logger.warning(f"Failed to contact seed node {address}: {e}")
        
        # Remove duplicates and add to known nodes
        unique_nodes = {}
        for node in discovered_nodes:
            if node.node_id not in unique_nodes:
                unique_nodes[node.node_id] = node
                self.nodes[node.node_id] = node
                await self._emit_node_joined(node)
        
        self.logger.info(f"Discovered {len(unique_nodes)} peer nodes")
        return list(unique_nodes.values())
    
    async def join_cluster(self, seed_addresses: List[str]) -> bool:
        """Join an existing cluster.
        
        Args:
            seed_addresses: List of seed node addresses
            
        Returns:
            True if successfully joined cluster
        """
        try:
            # Discover existing nodes
            await self.discover_peers(seed_addresses)
            
            # Announce this node to all known nodes
            await self._announce_to_cluster()
            
            self.logger.info(f"Successfully joined cluster with {len(self.nodes)} nodes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to join cluster: {e}")
            return False
    
    def get_nodes(self, 
                  service_type: Optional[ServiceType] = None,
                  status: Optional[ServiceStatus] = None,
                  tags: Optional[Set[str]] = None) -> List[NodeInfo]:
        """Get nodes matching criteria.
        
        Args:
            service_type: Filter by service type
            status: Filter by node status  
            tags: Filter by tags
            
        Returns:
            List of matching nodes
        """
        filtered_nodes = []
        
        for node in self.nodes.values():
            # Status filter
            if status and node.status != status:
                continue
            
            # Service type filter
            if service_type:
                has_service_type = any(
                    service.service_type == service_type
                    for service in node.services.values()
                )
                if not has_service_type:
                    continue
            
            # Tags filter (simplified - would need more sophisticated matching)
            if tags:
                node_tags = set()
                for service in node.services.values():
                    node_tags.update(service.tags)
                if not tags.issubset(node_tags):
                    continue
            
            filtered_nodes.append(node)
        
        return filtered_nodes
    
    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Get specific node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node information or None if not found
        """
        return self.nodes.get(node_id)
    
    async def update_node_metadata(self, metadata: Dict[str, Any]):
        """Update local node metadata.
        
        Args:
            metadata: Metadata to update
        """
        if self.local_node:
            self.local_node.metadata.update(metadata)
            await self._broadcast_node_update()
    
    async def update_node_resources(self, resources: Dict[str, Any]):
        """Update local node resource information.
        
        Args:
            resources: Resource information to update
        """
        if self.local_node:
            self.local_node.resources.update(resources)
            await self._broadcast_node_update()
    
    def register_node_joined_callback(self, callback: Callable[[NodeInfo], None]):
        """Register callback for node joined events."""
        self.node_joined_callbacks.append(callback)
    
    def register_node_left_callback(self, callback: Callable[[str], None]):
        """Register callback for node left events."""
        self.node_left_callbacks.append(callback)
    
    def register_node_updated_callback(self, callback: Callable[[NodeInfo], None]):
        """Register callback for node updated events."""
        self.node_updated_callbacks.append(callback)
    
    async def _initialize_local_node(self, bind_address: str):
        """Initialize local node information."""
        hostname = socket.gethostname()
        
        # Get local IP if bind_address is 0.0.0.0
        if bind_address == "0.0.0.0":
            ip_address = socket.gethostbyname(hostname)
        else:
            ip_address = bind_address
        
        # Get system resources (simplified)
        resources = {
            "cpu_cores": 32,  # Would get from system
            "memory_gb": 96,
            "hpu_count": 8,
            "disk_space_gb": 1000,
            "network_bandwidth_gbps": 25
        }
        
        capabilities = [
            "training",
            "inference", 
            "monitoring",
            "storage"
        ]
        
        self.local_node = NodeInfo(
            node_id=self.node_id,
            hostname=hostname,
            ip_address=ip_address,
            port=self.discovery_port,
            capabilities=capabilities,
            resources=resources,
            status=ServiceStatus.HEALTHY,
            services={},
            metadata={
                "version": "1.0.0",
                "platform": "gaudi3",
                "started_at": datetime.now().isoformat()
            },
            last_seen=datetime.now(),
            discovery_port=self.discovery_port
        )
        
        # Add to nodes registry
        self.nodes[self.node_id] = self.local_node
    
    async def _discovery_server(self, bind_address: str):
        """Run the discovery server."""
        server = await asyncio.start_server(
            self._handle_discovery_connection,
            bind_address,
            self.discovery_port
        )
        
        self.logger.info(f"Discovery server listening on {bind_address}:{self.discovery_port}")
        
        async with server:
            await server.serve_forever()
    
    async def _handle_discovery_connection(self, reader, writer):
        """Handle incoming discovery connection."""
        try:
            # Read request
            data = await reader.read(4096)
            request = json.loads(data.decode())
            
            response = await self._process_discovery_request(request)
            
            # Send response
            writer.write(json.dumps(response).encode())
            await writer.drain()
            
        except Exception as e:
            self.logger.error(f"Error handling discovery connection: {e}")
            error_response = {"error": str(e)}
            writer.write(json.dumps(error_response).encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _process_discovery_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process discovery request."""
        request_type = request.get("type")
        
        if request_type == "get_nodes":
            return {
                "type": "nodes_response",
                "nodes": [self._node_to_dict(node) for node in self.nodes.values()]
            }
        
        elif request_type == "announce_node":
            node_data = request.get("node")
            if node_data:
                node = self._dict_to_node(node_data)
                await self._handle_node_announcement(node)
                return {"type": "announce_response", "status": "accepted"}
        
        elif request_type == "heartbeat":
            node_id = request.get("node_id")
            if node_id in self.nodes:
                self.nodes[node_id].last_seen = datetime.now()
                return {"type": "heartbeat_response", "status": "acknowledged"}
        
        elif request_type == "node_update":
            node_data = request.get("node")
            if node_data:
                node = self._dict_to_node(node_data)
                await self._handle_node_update(node)
                return {"type": "update_response", "status": "accepted"}
        
        return {"type": "error", "message": "Unknown request type"}
    
    async def _heartbeat_process(self):
        """Send periodic heartbeats to known nodes."""
        while self.is_running:
            try:
                # Send heartbeat to all known nodes
                for node in list(self.nodes.values()):
                    if node.node_id != self.node_id:
                        try:
                            await self._send_heartbeat_to_node(node)
                        except Exception as e:
                            self.logger.warning(f"Failed to send heartbeat to {node.node_id}: {e}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Heartbeat process error: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_node_health(self):
        """Monitor health of known nodes."""
        while self.is_running:
            try:
                current_time = datetime.now()
                nodes_to_remove = []
                
                for node_id, node in self.nodes.items():
                    if node_id == self.node_id:
                        continue  # Skip local node
                    
                    # Check if node has timed out
                    time_since_seen = (current_time - node.last_seen).total_seconds()
                    
                    if time_since_seen > self.node_timeout:
                        if node.status != ServiceStatus.UNHEALTHY:
                            node.status = ServiceStatus.UNHEALTHY
                            await self._emit_node_updated(node)
                        
                        # Remove node after extended timeout
                        if time_since_seen > self.node_timeout * 2:
                            nodes_to_remove.append(node_id)
                
                # Remove timed-out nodes
                for node_id in nodes_to_remove:
                    del self.nodes[node_id]
                    await self._emit_node_left(node_id)
                    self.logger.info(f"Removed timed-out node: {node_id}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Node health monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _contact_seed_node(self, address: str) -> List[NodeInfo]:
        """Contact a seed node to discover peers."""
        try:
            host, port = address.split(":")
            port = int(port)
            
            reader, writer = await asyncio.open_connection(host, port)
            
            # Send discovery request
            request = {"type": "get_nodes"}
            writer.write(json.dumps(request).encode())
            await writer.drain()
            
            # Read response
            data = await reader.read(8192)
            response = json.loads(data.decode())
            
            writer.close()
            await writer.wait_closed()
            
            if response.get("type") == "nodes_response":
                nodes = []
                for node_data in response.get("nodes", []):
                    node = self._dict_to_node(node_data)
                    nodes.append(node)
                return nodes
            else:
                self.logger.warning(f"Unexpected response from seed node {address}")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to contact seed node {address}: {e}")
            return []
    
    async def _announce_to_cluster(self):
        """Announce this node to all known nodes."""
        if not self.local_node:
            return
        
        announcement = {
            "type": "announce_node",
            "node": self._node_to_dict(self.local_node)
        }
        
        for node in list(self.nodes.values()):
            if node.node_id != self.node_id:
                try:
                    await self._send_request_to_node(node, announcement)
                except Exception as e:
                    self.logger.warning(f"Failed to announce to {node.node_id}: {e}")
    
    async def _send_heartbeat_to_node(self, node: NodeInfo):
        """Send heartbeat to a specific node."""
        request = {
            "type": "heartbeat",
            "node_id": self.node_id,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._send_request_to_node(node, request)
    
    async def _send_request_to_node(self, node: NodeInfo, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to a node."""
        try:
            reader, writer = await asyncio.open_connection(
                node.ip_address, 
                node.discovery_port
            )
            
            writer.write(json.dumps(request).encode())
            await writer.drain()
            
            data = await reader.read(4096)
            response = json.loads(data.decode())
            
            writer.close()
            await writer.wait_closed()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to send request to {node.node_id}: {e}")
            raise
    
    async def _broadcast_node_update(self):
        """Broadcast node update to all known nodes."""
        if not self.local_node:
            return
        
        update = {
            "type": "node_update", 
            "node": self._node_to_dict(self.local_node)
        }
        
        for node in list(self.nodes.values()):
            if node.node_id != self.node_id:
                try:
                    await self._send_request_to_node(node, update)
                except Exception as e:
                    self.logger.warning(f"Failed to send update to {node.node_id}: {e}")
    
    async def _handle_node_announcement(self, node: NodeInfo):
        """Handle node announcement."""
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node
            await self._emit_node_joined(node)
            self.logger.info(f"New node joined: {node.node_id}")
        else:
            # Update existing node
            self.nodes[node.node_id] = node
            await self._emit_node_updated(node)
    
    async def _handle_node_update(self, node: NodeInfo):
        """Handle node update."""
        if node.node_id in self.nodes:
            self.nodes[node.node_id] = node
            await self._emit_node_updated(node)
    
    def _node_to_dict(self, node: NodeInfo) -> Dict[str, Any]:
        """Convert NodeInfo to dictionary."""
        node_dict = asdict(node)
        
        # Convert datetime objects to ISO strings
        if node_dict.get("last_seen"):
            node_dict["last_seen"] = node.last_seen.isoformat()
        
        # Convert services to dict format
        services_dict = {}
        for service_id, service in node.services.items():
            service_dict = asdict(service)
            if service_dict.get("registered_at"):
                service_dict["registered_at"] = service.registered_at.isoformat()
            if service_dict.get("last_heartbeat"):
                service_dict["last_heartbeat"] = service.last_heartbeat.isoformat()
            if service_dict.get("tags"):
                service_dict["tags"] = list(service.tags)
            services_dict[service_id] = service_dict
        
        node_dict["services"] = services_dict
        return node_dict
    
    def _dict_to_node(self, node_dict: Dict[str, Any]) -> NodeInfo:
        """Convert dictionary to NodeInfo."""
        # Convert ISO strings back to datetime
        if node_dict.get("last_seen"):
            node_dict["last_seen"] = datetime.fromisoformat(node_dict["last_seen"])
        
        # Convert services from dict format
        services = {}
        for service_id, service_dict in node_dict.get("services", {}).items():
            if service_dict.get("registered_at"):
                service_dict["registered_at"] = datetime.fromisoformat(service_dict["registered_at"])
            if service_dict.get("last_heartbeat"):
                service_dict["last_heartbeat"] = datetime.fromisoformat(service_dict["last_heartbeat"])
            if service_dict.get("tags"):
                service_dict["tags"] = set(service_dict["tags"])
            
            # Convert endpoints
            endpoints = []
            for ep in service_dict.get("endpoints", []):
                endpoints.append(ServiceEndpoint(**ep))
            service_dict["endpoints"] = endpoints
            
            services[service_id] = ServiceInfo(**service_dict)
        
        node_dict["services"] = services
        return NodeInfo(**node_dict)
    
    async def _emit_node_joined(self, node: NodeInfo):
        """Emit node joined event."""
        for callback in self.node_joined_callbacks:
            try:
                callback(node)
            except Exception as e:
                self.logger.error(f"Node joined callback error: {e}")
    
    async def _emit_node_left(self, node_id: str):
        """Emit node left event."""
        for callback in self.node_left_callbacks:
            try:
                callback(node_id)
            except Exception as e:
                self.logger.error(f"Node left callback error: {e}")
    
    async def _emit_node_updated(self, node: NodeInfo):
        """Emit node updated event.""" 
        for callback in self.node_updated_callbacks:
            try:
                callback(node)
            except Exception as e:
                self.logger.error(f"Node updated callback error: {e}")


class ServiceRegistry:
    """Registry for services running in the cluster."""
    
    def __init__(self, discovery_service: NodeDiscoveryService):
        """Initialize service registry.
        
        Args:
            discovery_service: Node discovery service
        """
        self.discovery_service = discovery_service
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Service tracking
        self.services: Dict[str, ServiceInfo] = {}
        self.service_types: Dict[ServiceType, List[str]] = {}
        
        # Health checking
        self.health_check_interval = 60  # seconds
        self.health_check_timeout = 10   # seconds
        
        # Start health monitoring
        asyncio.create_task(self._monitor_service_health())
    
    async def register_service(self, service: ServiceInfo) -> bool:
        """Register a service.
        
        Args:
            service: Service information
            
        Returns:
            True if registered successfully
        """
        try:
            # Add service to registry
            self.services[service.service_id] = service
            
            # Update service type mapping
            if service.service_type not in self.service_types:
                self.service_types[service.service_type] = []
            self.service_types[service.service_type].append(service.service_id)
            
            # Add service to local node
            if self.discovery_service.local_node:
                self.discovery_service.local_node.services[service.service_id] = service
                await self.discovery_service._broadcast_node_update()
            
            self.logger.info(f"Registered service: {service.service_name} ({service.service_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register service {service.service_id}: {e}")
            return False
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if deregistered successfully
        """
        try:
            if service_id not in self.services:
                return False
            
            service = self.services[service_id]
            
            # Remove from service type mapping
            if service.service_type in self.service_types:
                if service_id in self.service_types[service.service_type]:
                    self.service_types[service.service_type].remove(service_id)
                if not self.service_types[service.service_type]:
                    del self.service_types[service.service_type]
            
            # Remove from registry
            del self.services[service_id]
            
            # Remove from local node
            if self.discovery_service.local_node:
                if service_id in self.discovery_service.local_node.services:
                    del self.discovery_service.local_node.services[service_id]
                await self.discovery_service._broadcast_node_update()
            
            self.logger.info(f"Deregistered service: {service.service_name} ({service_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deregister service {service_id}: {e}")
            return False
    
    def discover_services(self, 
                         service_type: Optional[ServiceType] = None,
                         status: Optional[ServiceStatus] = None,
                         node_id: Optional[str] = None,
                         tags: Optional[Set[str]] = None) -> List[ServiceInfo]:
        """Discover services matching criteria.
        
        Args:
            service_type: Filter by service type
            status: Filter by service status
            node_id: Filter by node ID
            tags: Filter by tags
            
        Returns:
            List of matching services
        """
        services = []
        
        # Get services from all nodes
        for node in self.discovery_service.nodes.values():
            for service in node.services.values():
                # Apply filters
                if service_type and service.service_type != service_type:
                    continue
                if status and service.status != status:
                    continue
                if node_id and service.node_id != node_id:
                    continue
                if tags and not tags.issubset(service.tags):
                    continue
                
                services.append(service)
        
        return services
    
    def get_service_instances(self, service_type: ServiceType) -> List[ServiceInfo]:
        """Get all instances of a service type.
        
        Args:
            service_type: Type of service
            
        Returns:
            List of service instances
        """
        return self.discover_services(service_type=service_type, status=ServiceStatus.HEALTHY)
    
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """Get service by ID.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Service information or None if not found
        """
        # Check local services first
        if service_id in self.services:
            return self.services[service_id]
        
        # Check services on other nodes
        for node in self.discovery_service.nodes.values():
            if service_id in node.services:
                return node.services[service_id]
        
        return None
    
    async def update_service_status(self, service_id: str, status: ServiceStatus):
        """Update service status.
        
        Args:
            service_id: Service identifier
            status: New status
        """
        if service_id in self.services:
            self.services[service_id].status = status
            self.services[service_id].last_heartbeat = datetime.now()
            
            # Update in local node
            if self.discovery_service.local_node:
                if service_id in self.discovery_service.local_node.services:
                    self.discovery_service.local_node.services[service_id].status = status
                    self.discovery_service.local_node.services[service_id].last_heartbeat = datetime.now()
                await self.discovery_service._broadcast_node_update()
    
    async def _monitor_service_health(self):
        """Monitor health of registered services."""
        while True:
            try:
                for service_id, service in list(self.services.items()):
                    try:
                        if service.health_check_url:
                            # In a real implementation, would make HTTP request
                            # For now, simulate health check
                            is_healthy = True  # Simulate health check result
                            
                            new_status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
                            if new_status != service.status:
                                await self.update_service_status(service_id, new_status)
                    
                    except Exception as e:
                        self.logger.warning(f"Health check failed for service {service_id}: {e}")
                        await self.update_service_status(service_id, ServiceStatus.UNHEALTHY)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Service health monitoring error: {e}")
                await asyncio.sleep(10)