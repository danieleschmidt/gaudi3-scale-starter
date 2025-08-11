"""Service mesh and communication protocols for distributed Gaudi 3 clusters."""

import asyncio
import json
import logging
import ssl
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Union
import hashlib
import uuid
try:
    import aiohttp
except ImportError:
    # Fallback for environments without aiohttp
    aiohttp = None
import websockets
from cryptography.fernet import Fernet

from .discovery import ServiceInfo, ServiceRegistry, ServiceType, ServiceStatus
from ..logging_utils import get_logger
from ..exceptions import Gaudi3ScaleError

logger = get_logger(__name__)


class ProtocolType(str, Enum):
    """Communication protocol types."""
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    TCP = "tcp"
    UDP = "udp"
    CUSTOM = "custom"


class MessageType(str, Enum):
    """Message types for service communication."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    LOCALITY_AWARE = "locality_aware"


@dataclass
class Message:
    """Message structure for service communication."""
    message_id: str
    message_type: MessageType
    source_service: str
    target_service: Optional[str]
    payload: Dict[str, Any]
    headers: Dict[str, str]
    timestamp: datetime
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for service reliability."""
    service_id: str
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    
    @property
    def is_open(self) -> bool:
        return self.state == "OPEN"
    
    @property
    def is_half_open(self) -> bool:
        return self.state == "HALF_OPEN"
    
    @property
    def should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return False
        return (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout


@dataclass
class LoadBalancerState:
    """Load balancer state for service instances."""
    current_index: int = 0
    connection_counts: Dict[str, int] = None
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.connection_counts is None:
            self.connection_counts = {}
        if self.weights is None:
            self.weights = {}


class CommunicationProtocol:
    """Base communication protocol for service mesh."""
    
    def __init__(self, protocol_type: ProtocolType, encryption_enabled: bool = True):
        """Initialize communication protocol.
        
        Args:
            protocol_type: Type of communication protocol
            encryption_enabled: Whether to enable encryption
        """
        self.protocol_type = protocol_type
        self.encryption_enabled = encryption_enabled
        self.logger = logger.getChild(f"{self.__class__.__name__}_{protocol_type.value}")
        
        # Encryption
        self.cipher = Fernet(Fernet.generate_key()) if encryption_enabled else None
        
        # Connection management
        self.connections: Dict[str, Any] = {}
        self.connection_pool_size = 10
        
        # Message tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_timeout = 30  # seconds
        
    async def send_message(self, 
                          target_endpoint: str, 
                          message: Message) -> Optional[Message]:
        """Send message to target endpoint.
        
        Args:
            target_endpoint: Target service endpoint
            message: Message to send
            
        Returns:
            Response message if request type, None for notifications
        """
        try:
            # Encrypt message if enabled
            payload = self._encrypt_payload(message) if self.encryption_enabled else message.payload
            
            if self.protocol_type == ProtocolType.HTTP:
                return await self._send_http_message(target_endpoint, message, payload)
            elif self.protocol_type == ProtocolType.WEBSOCKET:
                return await self._send_websocket_message(target_endpoint, message, payload)
            elif self.protocol_type == ProtocolType.TCP:
                return await self._send_tcp_message(target_endpoint, message, payload)
            else:
                raise Gaudi3ScaleError(f"Protocol {self.protocol_type} not implemented")
                
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise
    
    async def start_server(self, bind_address: str, port: int):
        """Start protocol server.
        
        Args:
            bind_address: Address to bind to
            port: Port to listen on
        """
        if self.protocol_type == ProtocolType.HTTP:
            await self._start_http_server(bind_address, port)
        elif self.protocol_type == ProtocolType.WEBSOCKET:
            await self._start_websocket_server(bind_address, port)
        elif self.protocol_type == ProtocolType.TCP:
            await self._start_tcp_server(bind_address, port)
    
    def register_message_handler(self, handler: Callable[[Message], Message]):
        """Register message handler."""
        self.message_handler = handler
    
    async def _send_http_message(self, endpoint: str, message: Message, payload: Any) -> Optional[Message]:
        """Send HTTP message."""
        async with aiohttp.ClientSession() as session:
            headers = message.headers.copy()
            headers.update({
                'Content-Type': 'application/json',
                'X-Message-ID': message.message_id,
                'X-Message-Type': message.message_type.value,
                'X-Source-Service': message.source_service
            })
            
            if message.correlation_id:
                headers['X-Correlation-ID'] = message.correlation_id
            
            async with session.post(endpoint, json=payload, headers=headers) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return Message(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.RESPONSE,
                        source_service=message.target_service or "unknown",
                        target_service=message.source_service,
                        payload=response_data,
                        headers=dict(response.headers),
                        timestamp=datetime.now(),
                        correlation_id=message.message_id
                    )
                else:
                    raise Gaudi3ScaleError(f"HTTP request failed: {response.status}")
    
    async def _send_websocket_message(self, endpoint: str, message: Message, payload: Any) -> Optional[Message]:
        """Send WebSocket message."""
        # Get or create WebSocket connection
        if endpoint not in self.connections:
            try:
                self.connections[endpoint] = await websockets.connect(endpoint)
            except Exception as e:
                raise Gaudi3ScaleError(f"Failed to connect to WebSocket: {e}")
        
        websocket = self.connections[endpoint]
        
        # Prepare message
        ws_message = {
            "id": message.message_id,
            "type": message.message_type.value,
            "source": message.source_service,
            "target": message.target_service,
            "payload": payload,
            "headers": message.headers,
            "timestamp": message.timestamp.isoformat()
        }
        
        # Send message
        await websocket.send(json.dumps(ws_message))
        
        # Wait for response if request type
        if message.message_type == MessageType.REQUEST:
            response_data = await websocket.recv()
            response_message = json.loads(response_data)
            
            return Message(
                message_id=response_message["id"],
                message_type=MessageType(response_message["type"]),
                source_service=response_message["source"],
                target_service=response_message["target"],
                payload=response_message["payload"],
                headers=response_message["headers"],
                timestamp=datetime.fromisoformat(response_message["timestamp"]),
                correlation_id=message.message_id
            )
    
    async def _send_tcp_message(self, endpoint: str, message: Message, payload: Any) -> Optional[Message]:
        """Send TCP message."""
        host, port = endpoint.split(":")
        port = int(port)
        
        reader, writer = await asyncio.open_connection(host, port)
        
        try:
            # Prepare message
            tcp_message = {
                "id": message.message_id,
                "type": message.message_type.value,
                "source": message.source_service,
                "target": message.target_service,
                "payload": payload,
                "headers": message.headers,
                "timestamp": message.timestamp.isoformat()
            }
            
            # Send message
            message_data = json.dumps(tcp_message).encode()
            writer.write(len(message_data).to_bytes(4, 'big'))  # Send length first
            writer.write(message_data)
            await writer.drain()
            
            # Wait for response if request type
            if message.message_type == MessageType.REQUEST:
                # Read response length
                length_data = await reader.read(4)
                response_length = int.from_bytes(length_data, 'big')
                
                # Read response
                response_data = await reader.read(response_length)
                response_message = json.loads(response_data.decode())
                
                return Message(
                    message_id=response_message["id"],
                    message_type=MessageType(response_message["type"]),
                    source_service=response_message["source"],
                    target_service=response_message["target"],
                    payload=response_message["payload"],
                    headers=response_message["headers"],
                    timestamp=datetime.fromisoformat(response_message["timestamp"]),
                    correlation_id=message.message_id
                )
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _start_http_server(self, bind_address: str, port: int):
        """Start HTTP server."""
        from aiohttp import web, web_request
        
        async def handle_request(request: web_request.Request):
            try:
                payload = await request.json()
                
                # Decrypt payload if encryption enabled
                if self.encryption_enabled:
                    payload = self._decrypt_payload(payload)
                
                # Create message from request
                message = Message(
                    message_id=request.headers.get('X-Message-ID', str(uuid.uuid4())),
                    message_type=MessageType(request.headers.get('X-Message-Type', 'request')),
                    source_service=request.headers.get('X-Source-Service', 'unknown'),
                    target_service=None,
                    payload=payload,
                    headers=dict(request.headers),
                    timestamp=datetime.now(),
                    correlation_id=request.headers.get('X-Correlation-ID')
                )
                
                # Handle message
                response_message = await self._handle_message(message)
                
                if response_message:
                    return web.json_response(response_message.payload)
                else:
                    return web.Response(status=204)  # No content
                    
            except Exception as e:
                self.logger.error(f"HTTP request handling error: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        app = web.Application()
        app.router.add_post('/', handle_request)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, bind_address, port)
        await site.start()
        
        self.logger.info(f"HTTP server started on {bind_address}:{port}")
    
    async def _start_websocket_server(self, bind_address: str, port: int):
        """Start WebSocket server."""
        async def handle_websocket(websocket, path):
            try:
                async for message_data in websocket:
                    message_dict = json.loads(message_data)
                    
                    # Decrypt payload if encryption enabled
                    payload = message_dict["payload"]
                    if self.encryption_enabled:
                        payload = self._decrypt_payload(payload)
                    
                    # Create message
                    message = Message(
                        message_id=message_dict["id"],
                        message_type=MessageType(message_dict["type"]),
                        source_service=message_dict["source"],
                        target_service=message_dict["target"],
                        payload=payload,
                        headers=message_dict["headers"],
                        timestamp=datetime.fromisoformat(message_dict["timestamp"]),
                        correlation_id=message_dict.get("correlation_id")
                    )
                    
                    # Handle message
                    response_message = await self._handle_message(message)
                    
                    if response_message:
                        response_dict = {
                            "id": response_message.message_id,
                            "type": response_message.message_type.value,
                            "source": response_message.source_service,
                            "target": response_message.target_service,
                            "payload": response_message.payload,
                            "headers": response_message.headers,
                            "timestamp": response_message.timestamp.isoformat()
                        }
                        await websocket.send(json.dumps(response_dict))
                        
            except Exception as e:
                self.logger.error(f"WebSocket handling error: {e}")
        
        server = await websockets.serve(handle_websocket, bind_address, port)
        self.logger.info(f"WebSocket server started on {bind_address}:{port}")
        
        return server
    
    async def _start_tcp_server(self, bind_address: str, port: int):
        """Start TCP server."""
        async def handle_client(reader, writer):
            try:
                while True:
                    # Read message length
                    length_data = await reader.read(4)
                    if not length_data:
                        break
                    
                    message_length = int.from_bytes(length_data, 'big')
                    
                    # Read message
                    message_data = await reader.read(message_length)
                    message_dict = json.loads(message_data.decode())
                    
                    # Decrypt payload if encryption enabled
                    payload = message_dict["payload"]
                    if self.encryption_enabled:
                        payload = self._decrypt_payload(payload)
                    
                    # Create message
                    message = Message(
                        message_id=message_dict["id"],
                        message_type=MessageType(message_dict["type"]),
                        source_service=message_dict["source"],
                        target_service=message_dict["target"],
                        payload=payload,
                        headers=message_dict["headers"],
                        timestamp=datetime.fromisoformat(message_dict["timestamp"]),
                        correlation_id=message_dict.get("correlation_id")
                    )
                    
                    # Handle message
                    response_message = await self._handle_message(message)
                    
                    if response_message:
                        response_dict = {
                            "id": response_message.message_id,
                            "type": response_message.message_type.value,
                            "source": response_message.source_service,
                            "target": response_message.target_service,
                            "payload": response_message.payload,
                            "headers": response_message.headers,
                            "timestamp": response_message.timestamp.isoformat()
                        }
                        
                        response_data = json.dumps(response_dict).encode()
                        writer.write(len(response_data).to_bytes(4, 'big'))
                        writer.write(response_data)
                        await writer.drain()
                        
            except Exception as e:
                self.logger.error(f"TCP client handling error: {e}")
            finally:
                writer.close()
                await writer.wait_closed()
        
        server = await asyncio.start_server(handle_client, bind_address, port)
        self.logger.info(f"TCP server started on {bind_address}:{port}")
        
        return server
    
    def _encrypt_payload(self, message: Message) -> bytes:
        """Encrypt message payload."""
        if self.cipher:
            payload_json = json.dumps(message.payload).encode()
            return self.cipher.encrypt(payload_json)
        return message.payload
    
    def _decrypt_payload(self, encrypted_payload: Union[bytes, Any]) -> Dict[str, Any]:
        """Decrypt message payload."""
        if self.cipher and isinstance(encrypted_payload, bytes):
            decrypted_data = self.cipher.decrypt(encrypted_payload)
            return json.loads(decrypted_data.decode())
        return encrypted_payload
    
    async def _handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message."""
        if hasattr(self, 'message_handler'):
            return await self.message_handler(message)
        else:
            self.logger.warning(f"No message handler registered for {message.message_type}")
            return None


class ServiceMesh:
    """Service mesh for managing distributed service communication."""
    
    def __init__(self, service_registry: ServiceRegistry):
        """Initialize service mesh.
        
        Args:
            service_registry: Service registry for service discovery
        """
        self.service_registry = service_registry
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Communication protocols
        self.protocols: Dict[ProtocolType, CommunicationProtocol] = {}
        
        # Load balancing
        self.load_balancer_states: Dict[str, LoadBalancerState] = {}
        self.load_balancing_strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Metrics
        self.request_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Middleware
        self.middleware_stack: List[Callable] = []
        
        # Initialize default protocols
        self._initialize_protocols()
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_circuit_breakers())
        asyncio.create_task(self._collect_metrics())
    
    def _initialize_protocols(self):
        """Initialize communication protocols."""
        self.protocols[ProtocolType.HTTP] = CommunicationProtocol(ProtocolType.HTTP)
        self.protocols[ProtocolType.WEBSOCKET] = CommunicationProtocol(ProtocolType.WEBSOCKET)
        self.protocols[ProtocolType.TCP] = CommunicationProtocol(ProtocolType.TCP)
    
    async def send_request(self, 
                          target_service_type: ServiceType,
                          message: Message,
                          protocol: ProtocolType = ProtocolType.HTTP,
                          timeout: int = 30) -> Optional[Message]:
        """Send request to service through mesh.
        
        Args:
            target_service_type: Type of target service
            message: Message to send
            protocol: Communication protocol to use
            timeout: Request timeout in seconds
            
        Returns:
            Response message
        """
        try:
            # Discover target service instances
            service_instances = self.service_registry.get_service_instances(target_service_type)
            if not service_instances:
                raise Gaudi3ScaleError(f"No healthy instances found for {target_service_type}")
            
            # Select service instance using load balancing
            selected_service = self._select_service_instance(service_instances, target_service_type)
            if not selected_service:
                raise Gaudi3ScaleError(f"No available instances for {target_service_type}")
            
            # Check circuit breaker
            circuit_breaker = self._get_circuit_breaker(selected_service.service_id)
            if circuit_breaker.is_open:
                if circuit_breaker.should_attempt_reset:
                    circuit_breaker.state = "HALF_OPEN"
                else:
                    raise Gaudi3ScaleError(f"Circuit breaker open for {selected_service.service_id}")
            
            # Get protocol and endpoint
            communication_protocol = self.protocols[protocol]
            endpoint = self._get_service_endpoint(selected_service, protocol)
            
            # Update message target
            message.target_service = selected_service.service_id
            
            # Apply middleware
            for middleware in self.middleware_stack:
                message = await middleware(message)
            
            # Send request
            start_time = time.time()
            response = await asyncio.wait_for(
                communication_protocol.send_message(endpoint, message),
                timeout=timeout
            )
            
            # Update metrics and circuit breaker
            duration = time.time() - start_time
            self._record_success_metrics(selected_service.service_id, duration)
            self._record_circuit_breaker_success(selected_service.service_id)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            if 'selected_service' in locals():
                self._record_failure_metrics(selected_service.service_id, str(e))
                self._record_circuit_breaker_failure(selected_service.service_id)
            raise
    
    async def broadcast_message(self, 
                               message: Message,
                               service_type: Optional[ServiceType] = None,
                               protocol: ProtocolType = ProtocolType.HTTP):
        """Broadcast message to all services of a type.
        
        Args:
            message: Message to broadcast
            service_type: Service type to broadcast to (None for all)
            protocol: Communication protocol to use
        """
        # Get target services
        if service_type:
            target_services = self.service_registry.get_service_instances(service_type)
        else:
            target_services = self.service_registry.discover_services(status=ServiceStatus.HEALTHY)
        
        # Send to all services
        tasks = []
        communication_protocol = self.protocols[protocol]
        
        for service in target_services:
            endpoint = self._get_service_endpoint(service, protocol)
            message.target_service = service.service_id
            message.message_type = MessageType.BROADCAST
            
            task = asyncio.create_task(
                communication_protocol.send_message(endpoint, message)
            )
            tasks.append(task)
        
        # Wait for all broadcasts to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info(f"Broadcast sent to {len(target_services)} services")
    
    def add_middleware(self, middleware: Callable[[Message], Message]):
        """Add middleware to the service mesh.
        
        Args:
            middleware: Middleware function
        """
        self.middleware_stack.append(middleware)
    
    def set_load_balancing_strategy(self, strategy: LoadBalancingStrategy):
        """Set load balancing strategy.
        
        Args:
            strategy: Load balancing strategy
        """
        self.load_balancing_strategy = strategy
    
    def get_service_metrics(self, service_id: str) -> Dict[str, Any]:
        """Get metrics for a service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Service metrics
        """
        return self.request_metrics.get(service_id, {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "circuit_breaker_state": "CLOSED"
        })
    
    def _select_service_instance(self, 
                                services: List[ServiceInfo], 
                                service_type: ServiceType) -> Optional[ServiceInfo]:
        """Select service instance using load balancing strategy.
        
        Args:
            services: Available service instances
            service_type: Type of service
            
        Returns:
            Selected service instance
        """
        if not services:
            return None
        
        # Filter out services with open circuit breakers (except in half-open state)
        available_services = []
        for service in services:
            circuit_breaker = self._get_circuit_breaker(service.service_id)
            if not circuit_breaker.is_open or circuit_breaker.is_half_open:
                available_services.append(service)
        
        if not available_services:
            return None
        
        service_type_key = service_type.value
        
        # Get or create load balancer state
        if service_type_key not in self.load_balancer_states:
            self.load_balancer_states[service_type_key] = LoadBalancerState()
        
        lb_state = self.load_balancer_states[service_type_key]
        
        # Select based on strategy
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected = available_services[lb_state.current_index % len(available_services)]
            lb_state.current_index += 1
            return selected
            
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select service with least connections
            min_connections = float('inf')
            selected = None
            for service in available_services:
                connections = lb_state.connection_counts.get(service.service_id, 0)
                if connections < min_connections:
                    min_connections = connections
                    selected = service
            return selected
            
        elif self.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            import random
            return random.choice(available_services)
            
        else:
            # Default to round robin
            selected = available_services[lb_state.current_index % len(available_services)]
            lb_state.current_index += 1
            return selected
    
    def _get_service_endpoint(self, service: ServiceInfo, protocol: ProtocolType) -> str:
        """Get service endpoint for protocol.
        
        Args:
            service: Service information
            protocol: Communication protocol
            
        Returns:
            Service endpoint URL
        """
        for endpoint in service.endpoints:
            if protocol == ProtocolType.HTTP and endpoint.protocol in ["http", "https"]:
                return endpoint.url
            elif protocol == ProtocolType.WEBSOCKET and endpoint.protocol == "ws":
                return endpoint.url
            elif protocol == ProtocolType.TCP and endpoint.protocol == "tcp":
                return f"{endpoint.host}:{endpoint.port}"
        
        # Fallback to first endpoint
        if service.endpoints:
            return service.endpoints[0].url
        
        raise Gaudi3ScaleError(f"No suitable endpoint found for {protocol}")
    
    def _get_circuit_breaker(self, service_id: str) -> CircuitBreakerState:
        """Get or create circuit breaker for service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Circuit breaker state
        """
        if service_id not in self.circuit_breakers:
            self.circuit_breakers[service_id] = CircuitBreakerState(service_id)
        return self.circuit_breakers[service_id]
    
    def _record_success_metrics(self, service_id: str, duration: float):
        """Record successful request metrics.
        
        Args:
            service_id: Service identifier
            duration: Request duration
        """
        if service_id not in self.request_metrics:
            self.request_metrics[service_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_response_time": 0.0,
                "average_response_time": 0.0
            }
        
        metrics = self.request_metrics[service_id]
        metrics["total_requests"] += 1
        metrics["successful_requests"] += 1
        metrics["total_response_time"] += duration
        metrics["average_response_time"] = metrics["total_response_time"] / metrics["total_requests"]
    
    def _record_failure_metrics(self, service_id: str, error: str):
        """Record failed request metrics.
        
        Args:
            service_id: Service identifier
            error: Error message
        """
        if service_id not in self.request_metrics:
            self.request_metrics[service_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_response_time": 0.0,
                "average_response_time": 0.0
            }
        
        metrics = self.request_metrics[service_id]
        metrics["total_requests"] += 1
        metrics["failed_requests"] += 1
    
    def _record_circuit_breaker_success(self, service_id: str):
        """Record circuit breaker success.
        
        Args:
            service_id: Service identifier
        """
        circuit_breaker = self._get_circuit_breaker(service_id)
        
        if circuit_breaker.is_half_open:
            # Reset circuit breaker
            circuit_breaker.state = "CLOSED"
            circuit_breaker.failure_count = 0
            circuit_breaker.last_failure_time = None
    
    def _record_circuit_breaker_failure(self, service_id: str):
        """Record circuit breaker failure.
        
        Args:
            service_id: Service identifier
        """
        circuit_breaker = self._get_circuit_breaker(service_id)
        circuit_breaker.failure_count += 1
        circuit_breaker.last_failure_time = datetime.now()
        
        if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
            circuit_breaker.state = "OPEN"
    
    async def _monitor_circuit_breakers(self):
        """Monitor and update circuit breaker states."""
        while True:
            try:
                for service_id, circuit_breaker in self.circuit_breakers.items():
                    if circuit_breaker.is_open and circuit_breaker.should_attempt_reset:
                        circuit_breaker.state = "HALF_OPEN"
                        self.logger.info(f"Circuit breaker for {service_id} moved to HALF_OPEN")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Circuit breaker monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_metrics(self):
        """Collect and log service metrics."""
        while True:
            try:
                # Log metrics summary
                total_services = len(self.request_metrics)
                total_requests = sum(m["total_requests"] for m in self.request_metrics.values())
                
                self.logger.info(f"Service mesh metrics: {total_services} services, {total_requests} total requests")
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(10)