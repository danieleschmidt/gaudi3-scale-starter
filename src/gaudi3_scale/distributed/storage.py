"""Distributed storage and data management for Gaudi 3 clusters."""

import asyncio
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, BinaryIO, AsyncIterator
import uuid
import aiofiles
import aioboto3
from concurrent.futures import ThreadPoolExecutor

from .discovery import ServiceRegistry, ServiceInfo, ServiceType, ServiceStatus
from ..logging_utils import get_logger
from ..exceptions import Gaudi3ScaleError

logger = get_logger(__name__)


class StorageBackend(str, Enum):
    """Supported storage backends."""
    LOCAL_FS = "local_fs"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    NFS = "nfs"
    CEPH = "ceph"
    HDFS = "hdfs"
    DISTRIBUTED_FS = "distributed_fs"


class ReplicationStrategy(str, Enum):
    """Data replication strategies."""
    NONE = "none"
    MIRROR = "mirror"           # Full replication
    ERASURE_CODE = "erasure"    # Erasure coding
    RAID = "raid"              # RAID-like replication
    SHARDED = "sharded"        # Sharded with replication factor


class ConsistencyLevel(str, Enum):
    """Data consistency levels."""
    EVENTUAL = "eventual"       # Eventual consistency
    STRONG = "strong"          # Strong consistency
    SESSION = "session"        # Session consistency
    BOUNDED = "bounded"        # Bounded staleness


class CompressionType(str, Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"


@dataclass
class StorageConfig:
    """Storage configuration."""
    backend: StorageBackend
    base_path: str
    replication_strategy: ReplicationStrategy = ReplicationStrategy.MIRROR
    replication_factor: int = 3
    consistency_level: ConsistencyLevel = ConsistencyLevel.STRONG
    compression: CompressionType = CompressionType.GZIP
    encryption_enabled: bool = True
    max_file_size_mb: int = 1024
    chunk_size_mb: int = 64
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DataObject:
    """Represents a data object in distributed storage."""
    object_id: str
    name: str
    path: str
    size_bytes: int
    content_type: str
    checksum: str
    created_at: datetime
    modified_at: datetime
    replicas: List[str]  # Node IDs where replicas exist
    metadata: Dict[str, Any]
    version: int = 1
    is_deleted: bool = False
    access_count: int = 0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StorageNode:
    """Represents a storage node in the cluster."""
    node_id: str
    endpoint: str
    capacity_gb: int
    used_gb: int
    available_gb: int
    health_status: ServiceStatus
    last_heartbeat: datetime
    data_objects: Set[str]  # Object IDs stored on this node
    
    @property
    def utilization_percent(self) -> float:
        return (self.used_gb / self.capacity_gb) * 100 if self.capacity_gb > 0 else 0
    
    @property
    def is_healthy(self) -> bool:
        return self.health_status == ServiceStatus.HEALTHY


class DistributedStorageManager:
    """Manages distributed storage across cluster nodes."""
    
    def __init__(self, 
                 config: StorageConfig,
                 service_registry: ServiceRegistry):
        """Initialize distributed storage manager.
        
        Args:
            config: Storage configuration
            service_registry: Service registry for node discovery
        """
        self.config = config
        self.service_registry = service_registry
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Storage state
        self.storage_nodes: Dict[str, StorageNode] = {}
        self.data_objects: Dict[str, DataObject] = {}
        self.pending_operations: Dict[str, asyncio.Future] = {}
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.write_cache: Dict[str, Any] = {}
        self.read_cache: Dict[str, Any] = {}
        self.cache_size_limit = 1000
        
        # Monitoring
        self.operation_metrics: Dict[str, Any] = {
            "total_reads": 0,
            "total_writes": 0,
            "total_deletes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "replication_operations": 0
        }
        
        # Initialize storage
        asyncio.create_task(self._initialize_storage())
        asyncio.create_task(self._monitor_storage_health())
        asyncio.create_task(self._cleanup_cache())
    
    async def put_object(self, 
                        name: str,
                        data: Union[bytes, str, BinaryIO],
                        content_type: str = "application/octet-stream",
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store object in distributed storage.
        
        Args:
            name: Object name
            data: Object data
            content_type: MIME content type
            metadata: Additional metadata
            
        Returns:
            Object ID
        """
        try:
            object_id = str(uuid.uuid4())
            
            # Prepare data
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif hasattr(data, 'read'):
                data_bytes = data.read()
            else:
                data_bytes = data
            
            # Compress data if configured
            if self.config.compression != CompressionType.NONE:
                data_bytes = await self._compress_data(data_bytes)
            
            # Calculate checksum
            checksum = hashlib.sha256(data_bytes).hexdigest()
            
            # Create data object
            data_object = DataObject(
                object_id=object_id,
                name=name,
                path=f"{self.config.base_path}/{object_id}",
                size_bytes=len(data_bytes),
                content_type=content_type,
                checksum=checksum,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                replicas=[],
                metadata=metadata or {}
            )
            
            # Select storage nodes for replication
            target_nodes = await self._select_storage_nodes(len(data_bytes))
            if len(target_nodes) < self.config.replication_factor:
                self.logger.warning(
                    f"Only {len(target_nodes)} nodes available, "
                    f"replication factor is {self.config.replication_factor}"
                )
            
            # Store data on selected nodes
            storage_tasks = []
            for node in target_nodes[:self.config.replication_factor]:
                task = asyncio.create_task(
                    self._store_on_node(node, object_id, data_bytes)
                )
                storage_tasks.append(task)
            
            # Wait for storage operations
            results = await asyncio.gather(*storage_tasks, return_exceptions=True)
            
            # Check results and update replicas
            successful_nodes = []
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    successful_nodes.append(target_nodes[i].node_id)
                    data_object.replicas.append(target_nodes[i].node_id)
            
            if not successful_nodes:
                raise Gaudi3ScaleError("Failed to store object on any node")
            
            # Store object metadata
            self.data_objects[object_id] = data_object
            
            # Update cache
            self.write_cache[object_id] = data_bytes
            if len(self.write_cache) > self.cache_size_limit:
                # Remove oldest entries
                oldest_key = next(iter(self.write_cache))
                del self.write_cache[oldest_key]
            
            # Update metrics
            self.operation_metrics["total_writes"] += 1
            self.operation_metrics["replication_operations"] += len(successful_nodes)
            
            self.logger.info(
                f"Stored object {name} ({object_id}) on {len(successful_nodes)} nodes"
            )
            
            return object_id
            
        except Exception as e:
            self.logger.error(f"Failed to store object {name}: {e}")
            raise Gaudi3ScaleError(f"Storage operation failed: {e}")
    
    async def get_object(self, object_id: str) -> Optional[bytes]:
        """Retrieve object from distributed storage.
        
        Args:
            object_id: Object identifier
            
        Returns:
            Object data or None if not found
        """
        try:
            # Check cache first
            if object_id in self.read_cache:
                self.operation_metrics["cache_hits"] += 1
                return self.read_cache[object_id]
            
            self.operation_metrics["cache_misses"] += 1
            
            # Get object metadata
            if object_id not in self.data_objects:
                return None
            
            data_object = self.data_objects[object_id]
            if data_object.is_deleted:
                return None
            
            # Find healthy replica nodes
            healthy_replicas = []
            for replica_node_id in data_object.replicas:
                if (replica_node_id in self.storage_nodes and 
                    self.storage_nodes[replica_node_id].is_healthy):
                    healthy_replicas.append(self.storage_nodes[replica_node_id])
            
            if not healthy_replicas:
                raise Gaudi3ScaleError(f"No healthy replicas found for object {object_id}")
            
            # Try to read from replicas
            for node in healthy_replicas:
                try:
                    data = await self._read_from_node(node, object_id)
                    if data:
                        # Verify checksum
                        computed_checksum = hashlib.sha256(data).hexdigest()
                        if computed_checksum != data_object.checksum:
                            self.logger.warning(
                                f"Checksum mismatch for object {object_id} on node {node.node_id}"
                            )
                            continue
                        
                        # Decompress if needed
                        if self.config.compression != CompressionType.NONE:
                            data = await self._decompress_data(data)
                        
                        # Update cache and metrics
                        self.read_cache[object_id] = data
                        if len(self.read_cache) > self.cache_size_limit:
                            oldest_key = next(iter(self.read_cache))
                            del self.read_cache[oldest_key]
                        
                        data_object.access_count += 1
                        self.operation_metrics["total_reads"] += 1
                        
                        return data
                        
                except Exception as e:
                    self.logger.warning(f"Failed to read from node {node.node_id}: {e}")
                    continue
            
            raise Gaudi3ScaleError(f"Failed to read object {object_id} from any replica")
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve object {object_id}: {e}")
            raise
    
    async def delete_object(self, object_id: str) -> bool:
        """Delete object from distributed storage.
        
        Args:
            object_id: Object identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            if object_id not in self.data_objects:
                return False
            
            data_object = self.data_objects[object_id]
            
            # Delete from replica nodes
            delete_tasks = []
            for replica_node_id in data_object.replicas:
                if replica_node_id in self.storage_nodes:
                    node = self.storage_nodes[replica_node_id]
                    task = asyncio.create_task(
                        self._delete_from_node(node, object_id)
                    )
                    delete_tasks.append(task)
            
            # Wait for deletion operations
            await asyncio.gather(*delete_tasks, return_exceptions=True)
            
            # Mark as deleted
            data_object.is_deleted = True
            data_object.modified_at = datetime.now()
            
            # Remove from caches
            self.read_cache.pop(object_id, None)
            self.write_cache.pop(object_id, None)
            
            self.operation_metrics["total_deletes"] += 1
            
            self.logger.info(f"Deleted object {object_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete object {object_id}: {e}")
            return False
    
    async def list_objects(self, 
                          prefix: str = "",
                          limit: int = 1000,
                          include_deleted: bool = False) -> List[DataObject]:
        """List objects in storage.
        
        Args:
            prefix: Name prefix filter
            limit: Maximum number of objects to return
            include_deleted: Whether to include deleted objects
            
        Returns:
            List of data objects
        """
        objects = []
        count = 0
        
        for data_object in self.data_objects.values():
            if count >= limit:
                break
            
            # Filter by deletion status
            if not include_deleted and data_object.is_deleted:
                continue
            
            # Filter by prefix
            if prefix and not data_object.name.startswith(prefix):
                continue
            
            objects.append(data_object)
            count += 1
        
        # Sort by creation time
        objects.sort(key=lambda obj: obj.created_at, reverse=True)
        return objects
    
    async def get_object_metadata(self, object_id: str) -> Optional[DataObject]:
        """Get object metadata.
        
        Args:
            object_id: Object identifier
            
        Returns:
            Data object metadata or None if not found
        """
        return self.data_objects.get(object_id)
    
    async def replicate_object(self, object_id: str, target_nodes: int = None) -> bool:
        """Ensure object has sufficient replicas.
        
        Args:
            object_id: Object identifier
            target_nodes: Target number of replicas (default: replication_factor)
            
        Returns:
            True if replication successful
        """
        if target_nodes is None:
            target_nodes = self.config.replication_factor
        
        if object_id not in self.data_objects:
            return False
        
        data_object = self.data_objects[object_id]
        
        # Count healthy replicas
        healthy_replicas = 0
        healthy_nodes = []
        for replica_node_id in data_object.replicas:
            if (replica_node_id in self.storage_nodes and 
                self.storage_nodes[replica_node_id].is_healthy):
                healthy_replicas += 1
                healthy_nodes.append(self.storage_nodes[replica_node_id])
        
        if healthy_replicas >= target_nodes:
            return True  # Already sufficiently replicated
        
        # Need more replicas
        needed_replicas = target_nodes - healthy_replicas
        
        # Get object data from a healthy replica
        if not healthy_nodes:
            self.logger.error(f"No healthy replicas for object {object_id}")
            return False
        
        source_node = healthy_nodes[0]
        data = await self._read_from_node(source_node, object_id)
        if not data:
            return False
        
        # Select new nodes for replication
        available_nodes = await self._select_storage_nodes(
            len(data), 
            exclude_nodes={node.node_id for node in healthy_nodes}
        )
        
        if len(available_nodes) < needed_replicas:
            self.logger.warning(
                f"Only {len(available_nodes)} nodes available for replication, "
                f"need {needed_replicas}"
            )
        
        # Replicate to new nodes
        replication_tasks = []
        for node in available_nodes[:needed_replicas]:
            task = asyncio.create_task(
                self._store_on_node(node, object_id, data)
            )
            replication_tasks.append(task)
        
        results = await asyncio.gather(*replication_tasks, return_exceptions=True)
        
        # Update replica list
        successful_replications = 0
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                data_object.replicas.append(available_nodes[i].node_id)
                successful_replications += 1
        
        self.operation_metrics["replication_operations"] += successful_replications
        
        self.logger.info(
            f"Replicated object {object_id} to {successful_replications} additional nodes"
        )
        
        return successful_replications > 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Storage statistics
        """
        total_capacity = sum(node.capacity_gb for node in self.storage_nodes.values())
        total_used = sum(node.used_gb for node in self.storage_nodes.values())
        total_objects = len([obj for obj in self.data_objects.values() if not obj.is_deleted])
        total_size = sum(
            obj.size_bytes for obj in self.data_objects.values() 
            if not obj.is_deleted
        )
        
        healthy_nodes = sum(1 for node in self.storage_nodes.values() if node.is_healthy)
        
        return {
            "total_nodes": len(self.storage_nodes),
            "healthy_nodes": healthy_nodes,
            "total_capacity_gb": total_capacity,
            "total_used_gb": total_used,
            "utilization_percent": (total_used / total_capacity * 100) if total_capacity > 0 else 0,
            "total_objects": total_objects,
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "cache_hit_rate": (
                self.operation_metrics["cache_hits"] / 
                (self.operation_metrics["cache_hits"] + self.operation_metrics["cache_misses"])
            ) if (self.operation_metrics["cache_hits"] + self.operation_metrics["cache_misses"]) > 0 else 0,
            **self.operation_metrics
        }
    
    async def _initialize_storage(self):
        """Initialize storage system."""
        try:
            # Discover storage nodes
            await self._discover_storage_nodes()
            
            # Create base directory if using local filesystem
            if self.config.backend == StorageBackend.LOCAL_FS:
                os.makedirs(self.config.base_path, exist_ok=True)
            
            self.logger.info(f"Storage initialized with {len(self.storage_nodes)} nodes")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {e}")
    
    async def _discover_storage_nodes(self):
        """Discover available storage nodes."""
        # Get storage services from registry
        storage_services = self.service_registry.discover_services(
            service_type=ServiceType.STORAGE,
            status=ServiceStatus.HEALTHY
        )
        
        for service in storage_services:
            # Create storage node from service info
            node = StorageNode(
                node_id=service.node_id,
                endpoint=service.endpoints[0].url if service.endpoints else "",
                capacity_gb=service.metadata.get("capacity_gb", 1000),
                used_gb=service.metadata.get("used_gb", 0),
                available_gb=service.metadata.get("available_gb", 1000),
                health_status=service.status,
                last_heartbeat=service.last_heartbeat,
                data_objects=set()
            )
            
            self.storage_nodes[service.node_id] = node
    
    async def _select_storage_nodes(self, 
                                   data_size: int, 
                                   exclude_nodes: Set[str] = None) -> List[StorageNode]:
        """Select storage nodes for data placement.
        
        Args:
            data_size: Size of data to store
            exclude_nodes: Nodes to exclude from selection
            
        Returns:
            List of selected storage nodes
        """
        if exclude_nodes is None:
            exclude_nodes = set()
        
        # Filter available nodes
        available_nodes = []
        required_space_gb = data_size / (1024**3) + 0.1  # Add buffer
        
        for node in self.storage_nodes.values():
            if (node.is_healthy and 
                node.node_id not in exclude_nodes and
                node.available_gb >= required_space_gb):
                available_nodes.append(node)
        
        # Sort by utilization (prefer less utilized nodes)
        available_nodes.sort(key=lambda n: n.utilization_percent)
        
        return available_nodes
    
    async def _store_on_node(self, node: StorageNode, object_id: str, data: bytes) -> bool:
        """Store data on a specific node.
        
        Args:
            node: Storage node
            object_id: Object identifier
            data: Data to store
            
        Returns:
            True if successful
        """
        try:
            if self.config.backend == StorageBackend.LOCAL_FS:
                # Store on local filesystem
                file_path = Path(self.config.base_path) / object_id
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(data)
                
                # Update node state
                node.used_gb += len(data) / (1024**3)
                node.available_gb -= len(data) / (1024**3)
                node.data_objects.add(object_id)
                
                return True
            
            elif self.config.backend == StorageBackend.S3:
                # Store on S3 (simplified implementation)
                return await self._store_s3(node.endpoint, object_id, data)
            
            else:
                # Other backends would be implemented here
                self.logger.warning(f"Backend {self.config.backend} not implemented")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to store on node {node.node_id}: {e}")
            return False
    
    async def _read_from_node(self, node: StorageNode, object_id: str) -> Optional[bytes]:
        """Read data from a specific node.
        
        Args:
            node: Storage node
            object_id: Object identifier
            
        Returns:
            Data bytes or None if not found
        """
        try:
            if self.config.backend == StorageBackend.LOCAL_FS:
                file_path = Path(self.config.base_path) / object_id
                if file_path.exists():
                    async with aiofiles.open(file_path, 'rb') as f:
                        return await f.read()
                return None
            
            elif self.config.backend == StorageBackend.S3:
                return await self._read_s3(node.endpoint, object_id)
            
            else:
                self.logger.warning(f"Backend {self.config.backend} not implemented")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to read from node {node.node_id}: {e}")
            return None
    
    async def _delete_from_node(self, node: StorageNode, object_id: str) -> bool:
        """Delete data from a specific node.
        
        Args:
            node: Storage node
            object_id: Object identifier
            
        Returns:
            True if successful
        """
        try:
            if self.config.backend == StorageBackend.LOCAL_FS:
                file_path = Path(self.config.base_path) / object_id
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    
                    # Update node state
                    node.used_gb -= file_size / (1024**3)
                    node.available_gb += file_size / (1024**3)
                    node.data_objects.discard(object_id)
                
                return True
            
            elif self.config.backend == StorageBackend.S3:
                return await self._delete_s3(node.endpoint, object_id)
            
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete from node {node.node_id}: {e}")
            return False
    
    async def _compress_data(self, data: bytes) -> bytes:
        """Compress data using configured algorithm.
        
        Args:
            data: Data to compress
            
        Returns:
            Compressed data
        """
        if self.config.compression == CompressionType.GZIP:
            import gzip
            return gzip.compress(data)
        elif self.config.compression == CompressionType.LZ4:
            import lz4.frame
            return lz4.frame.compress(data)
        elif self.config.compression == CompressionType.ZSTD:
            import zstd
            return zstd.compress(data)
        else:
            return data
    
    async def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using configured algorithm.
        
        Args:
            data: Compressed data
            
        Returns:
            Decompressed data
        """
        if self.config.compression == CompressionType.GZIP:
            import gzip
            return gzip.decompress(data)
        elif self.config.compression == CompressionType.LZ4:
            import lz4.frame
            return lz4.frame.decompress(data)
        elif self.config.compression == CompressionType.ZSTD:
            import zstd
            return zstd.decompress(data)
        else:
            return data
    
    async def _monitor_storage_health(self):
        """Monitor storage node health."""
        while True:
            try:
                # Rediscover storage nodes
                await self._discover_storage_nodes()
                
                # Check for failed replications
                for data_object in self.data_objects.values():
                    if data_object.is_deleted:
                        continue
                    
                    healthy_replicas = sum(
                        1 for replica_id in data_object.replicas
                        if (replica_id in self.storage_nodes and 
                            self.storage_nodes[replica_id].is_healthy)
                    )
                    
                    if healthy_replicas < self.config.replication_factor:
                        self.logger.warning(
                            f"Object {data_object.object_id} has only {healthy_replicas} "
                            f"healthy replicas (target: {self.config.replication_factor})"
                        )
                        # Trigger re-replication
                        asyncio.create_task(
                            self.replicate_object(data_object.object_id)
                        )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Storage health monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_cache(self):
        """Clean up old cache entries."""
        while True:
            try:
                # Clean up read cache
                if len(self.read_cache) > self.cache_size_limit:
                    # Remove 25% of entries
                    remove_count = len(self.read_cache) // 4
                    for _ in range(remove_count):
                        oldest_key = next(iter(self.read_cache))
                        del self.read_cache[oldest_key]
                
                # Clean up write cache
                if len(self.write_cache) > self.cache_size_limit:
                    remove_count = len(self.write_cache) // 4
                    for _ in range(remove_count):
                        oldest_key = next(iter(self.write_cache))
                        del self.write_cache[oldest_key]
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(30)


class DataManager:
    """High-level data management for distributed training."""
    
    def __init__(self, storage_manager: DistributedStorageManager):
        """Initialize data manager.
        
        Args:
            storage_manager: Distributed storage manager
        """
        self.storage_manager = storage_manager
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Dataset management
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.checkpoints: Dict[str, List[str]] = {}  # training_job -> checkpoint_ids
        self.models: Dict[str, str] = {}  # model_name -> object_id
    
    async def register_dataset(self, 
                              name: str,
                              data_path: str,
                              format_type: str = "auto",
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a dataset for distributed training.
        
        Args:
            name: Dataset name
            data_path: Path to dataset
            format_type: Data format (auto, tfrecord, parquet, etc.)
            metadata: Additional metadata
            
        Returns:
            Dataset ID
        """
        try:
            # Read and store dataset
            if os.path.isfile(data_path):
                # Single file
                with open(data_path, 'rb') as f:
                    data = f.read()
                
                object_id = await self.storage_manager.put_object(
                    name=f"dataset_{name}",
                    data=data,
                    content_type="application/octet-stream",
                    metadata={
                        "type": "dataset",
                        "format": format_type,
                        "original_path": data_path,
                        **(metadata or {})
                    }
                )
                
                dataset_info = {
                    "id": object_id,
                    "name": name,
                    "format": format_type,
                    "size": len(data),
                    "files": [object_id],
                    "created_at": datetime.now(),
                    "metadata": metadata or {}
                }
                
            else:
                # Directory - store multiple files
                file_objects = []
                total_size = 0
                
                for root, dirs, files in os.walk(data_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, data_path)
                        
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        object_id = await self.storage_manager.put_object(
                            name=f"dataset_{name}/{relative_path}",
                            data=file_data,
                            content_type="application/octet-stream",
                            metadata={
                                "type": "dataset_file",
                                "dataset": name,
                                "relative_path": relative_path,
                                "format": format_type
                            }
                        )
                        
                        file_objects.append(object_id)
                        total_size += len(file_data)
                
                dataset_id = str(uuid.uuid4())
                dataset_info = {
                    "id": dataset_id,
                    "name": name,
                    "format": format_type,
                    "size": total_size,
                    "files": file_objects,
                    "created_at": datetime.now(),
                    "metadata": metadata or {}
                }
            
            self.datasets[name] = dataset_info
            
            self.logger.info(f"Registered dataset {name} with {len(dataset_info['files'])} files")
            return dataset_info["id"]
            
        except Exception as e:
            self.logger.error(f"Failed to register dataset {name}: {e}")
            raise Gaudi3ScaleError(f"Dataset registration failed: {e}")
    
    async def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dataset information.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset information or None if not found
        """
        return self.datasets.get(name)
    
    async def save_checkpoint(self, 
                             training_job_id: str,
                             model_state: Dict[str, Any],
                             optimizer_state: Dict[str, Any],
                             step: int,
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save training checkpoint.
        
        Args:
            training_job_id: Training job identifier
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            step: Training step number
            metadata: Additional metadata
            
        Returns:
            Checkpoint ID
        """
        try:
            checkpoint_data = {
                "model_state": model_state,
                "optimizer_state": optimizer_state,
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "training_job_id": training_job_id,
                "metadata": metadata or {}
            }
            
            checkpoint_json = json.dumps(checkpoint_data).encode('utf-8')
            
            checkpoint_id = await self.storage_manager.put_object(
                name=f"checkpoint_{training_job_id}_step_{step}",
                data=checkpoint_json,
                content_type="application/json",
                metadata={
                    "type": "checkpoint",
                    "training_job_id": training_job_id,
                    "step": step
                }
            )
            
            # Track checkpoint
            if training_job_id not in self.checkpoints:
                self.checkpoints[training_job_id] = []
            self.checkpoints[training_job_id].append(checkpoint_id)
            
            self.logger.info(f"Saved checkpoint for job {training_job_id} at step {step}")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise Gaudi3ScaleError(f"Checkpoint save failed: {e}")
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load training checkpoint.
        
        Args:
            checkpoint_id: Checkpoint identifier
            
        Returns:
            Checkpoint data or None if not found
        """
        try:
            checkpoint_data = await self.storage_manager.get_object(checkpoint_id)
            if checkpoint_data:
                return json.loads(checkpoint_data.decode('utf-8'))
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    async def list_checkpoints(self, training_job_id: str) -> List[str]:
        """List checkpoints for a training job.
        
        Args:
            training_job_id: Training job identifier
            
        Returns:
            List of checkpoint IDs
        """
        return self.checkpoints.get(training_job_id, [])
    
    async def save_model(self, 
                        name: str,
                        model_data: bytes,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save trained model.
        
        Args:
            name: Model name
            model_data: Serialized model data
            metadata: Model metadata
            
        Returns:
            Model object ID
        """
        try:
            object_id = await self.storage_manager.put_object(
                name=f"model_{name}",
                data=model_data,
                content_type="application/octet-stream",
                metadata={
                    "type": "model",
                    "model_name": name,
                    **(metadata or {})
                }
            )
            
            self.models[name] = object_id
            
            self.logger.info(f"Saved model {name}")
            return object_id
            
        except Exception as e:
            self.logger.error(f"Failed to save model {name}: {e}")
            raise Gaudi3ScaleError(f"Model save failed: {e}")
    
    async def load_model(self, name: str) -> Optional[bytes]:
        """Load trained model.
        
        Args:
            name: Model name
            
        Returns:
            Model data or None if not found
        """
        try:
            if name in self.models:
                return await self.storage_manager.get_object(self.models[name])
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load model {name}: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get data management statistics.
        
        Returns:
            Statistics dictionary
        """
        storage_stats = self.storage_manager.get_storage_stats()
        
        total_checkpoints = sum(len(checkpoints) for checkpoints in self.checkpoints.values())
        
        return {
            **storage_stats,
            "total_datasets": len(self.datasets),
            "total_models": len(self.models),
            "total_checkpoints": total_checkpoints,
            "active_training_jobs": len(self.checkpoints)
        }