"""Distributed configuration management for Gaudi 3 clusters."""

import asyncio
import json
import logging
import hashlib
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Callable
import uuid
import yaml
from pathlib import Path

from .discovery import ServiceRegistry, ServiceInfo, ServiceType, ServiceStatus
from .storage import DataManager
from ..logging_utils import get_logger
from ..exceptions import Gaudi3ScaleError

logger = get_logger(__name__)


class ConfigScope(str, Enum):
    """Configuration scope levels."""
    GLOBAL = "global"           # Cluster-wide configuration
    SERVICE = "service"         # Service-specific configuration
    NODE = "node"              # Node-specific configuration
    USER = "user"              # User-specific configuration
    APPLICATION = "application" # Application-specific configuration


class ConfigFormat(str, Enum):
    """Configuration format types."""
    YAML = "yaml"
    JSON = "json"
    PROPERTIES = "properties"
    ENV = "env"
    TOML = "toml"
    INI = "ini"


class ConfigSource(str, Enum):
    """Configuration source types."""
    FILE = "file"
    DATABASE = "database"
    ENVIRONMENT = "environment"
    REMOTE_CONFIG = "remote"
    VAULT = "vault"
    CONSUL = "consul"
    ETCD = "etcd"


class ConfigStatus(str, Enum):
    """Configuration status states."""
    ACTIVE = "active"
    DRAFT = "draft"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ConfigVersion:
    """Represents a version of configuration."""
    version_id: str
    version_number: int
    config_data: Dict[str, Any]
    created_at: datetime
    created_by: str
    description: str
    checksum: str
    status: ConfigStatus = ConfigStatus.DRAFT
    
    @classmethod
    def create(cls, 
               config_data: Dict[str, Any], 
               created_by: str,
               description: str = "") -> 'ConfigVersion':
        """Create a new configuration version.
        
        Args:
            config_data: Configuration data
            created_by: User who created the version
            description: Version description
            
        Returns:
            New configuration version
        """
        config_json = json.dumps(config_data, sort_keys=True)
        checksum = hashlib.sha256(config_json.encode()).hexdigest()
        
        return cls(
            version_id=str(uuid.uuid4()),
            version_number=1,  # Will be set properly when added to config
            config_data=config_data,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            checksum=checksum
        )


@dataclass
class ConfigKey:
    """Represents a configuration key with metadata."""
    key_id: str
    key_path: str  # Hierarchical key path like "database.connection.timeout"
    scope: ConfigScope
    format: ConfigFormat
    source: ConfigSource
    encrypted: bool = False
    versions: List[ConfigVersion] = field(default_factory=list)
    active_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Access control
    read_permissions: Set[str] = field(default_factory=set)
    write_permissions: Set[str] = field(default_factory=set)
    
    # Synchronization
    last_sync: Optional[datetime] = None
    sync_enabled: bool = True
    
    @property
    def current_value(self) -> Optional[Any]:
        """Get current active configuration value."""
        if self.active_version:
            version = next(
                (v for v in self.versions if v.version_id == self.active_version),
                None
            )
            if version:
                return self._extract_value_from_path(version.config_data, self.key_path)
        return None
    
    def _extract_value_from_path(self, data: Dict[str, Any], path: str) -> Any:
        """Extract value from nested dictionary using dot notation path."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


@dataclass
class ConfigChangeEvent:
    """Represents a configuration change event."""
    event_id: str
    key_path: str
    scope: ConfigScope
    change_type: str  # created, updated, deleted
    old_value: Optional[Any]
    new_value: Optional[Any]
    changed_by: str
    timestamp: datetime
    version_id: str
    reason: str = ""


class DistributedConfigManager:
    """Manages distributed configuration across the Gaudi 3 cluster."""
    
    def __init__(self, 
                 service_registry: ServiceRegistry,
                 data_manager: DataManager,
                 node_id: str):
        """Initialize distributed configuration manager.
        
        Args:
            service_registry: Service registry for discovering config services
            data_manager: Data manager for storing configuration
            node_id: Current node identifier
        """
        self.service_registry = service_registry
        self.data_manager = data_manager
        self.node_id = node_id
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Configuration state
        self.config_keys: Dict[str, ConfigKey] = {}
        self.change_history: List[ConfigChangeEvent] = []
        self.max_history_size = 1000
        
        # Watchers and callbacks
        self.config_watchers: Dict[str, List[Callable]] = {}
        self.change_callbacks: List[Callable[[ConfigChangeEvent], None]] = []
        
        # Synchronization
        self.sync_enabled = True
        self.sync_interval = 30  # seconds
        self.last_sync_time: Optional[datetime] = None
        
        # Caching
        self.config_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Validation and schema
        self.config_schemas: Dict[str, Dict[str, Any]] = {}
        self.validation_enabled = True
        
        # Start background tasks
        asyncio.create_task(self._sync_configurations())
        asyncio.create_task(self._cleanup_cache())
        
        self.logger.info(f"Distributed config manager initialized for node {node_id}")
    
    async def set_config(self, 
                        key_path: str,
                        value: Any,
                        scope: ConfigScope = ConfigScope.GLOBAL,
                        format: ConfigFormat = ConfigFormat.JSON,
                        user: str = "system",
                        description: str = "",
                        encrypt: bool = False) -> str:
        """Set configuration value.
        
        Args:
            key_path: Hierarchical configuration key path
            value: Configuration value
            scope: Configuration scope
            format: Value format
            user: User making the change
            description: Change description
            encrypt: Whether to encrypt the value
            
        Returns:
            Version ID of the new configuration
        """
        try:
            # Validate configuration if schema exists
            if self.validation_enabled:
                await self._validate_config(key_path, value, scope)
            
            # Get or create config key
            if key_path not in self.config_keys:
                config_key = ConfigKey(
                    key_id=str(uuid.uuid4()),
                    key_path=key_path,
                    scope=scope,
                    format=format,
                    source=ConfigSource.REMOTE_CONFIG,
                    encrypted=encrypt
                )
                self.config_keys[key_path] = config_key
            else:
                config_key = self.config_keys[key_path]
            
            # Get old value for change tracking
            old_value = config_key.current_value
            
            # Create configuration data structure
            config_data = self._create_nested_dict(key_path, value)
            
            # Create new version
            version = ConfigVersion.create(
                config_data=config_data,
                created_by=user,
                description=description
            )
            version.version_number = len(config_key.versions) + 1
            version.status = ConfigStatus.ACTIVE
            
            # Encrypt if requested
            if encrypt:
                version.config_data = await self._encrypt_config_data(version.config_data)
                config_key.encrypted = True
            
            # Add version and set as active
            config_key.versions.append(version)
            config_key.active_version = version.version_id
            
            # Store in persistent storage
            await self._persist_config(config_key)
            
            # Update cache
            self.config_cache[key_path] = value
            self.cache_timestamps[key_path] = datetime.now()
            
            # Create change event
            change_event = ConfigChangeEvent(
                event_id=str(uuid.uuid4()),
                key_path=key_path,
                scope=scope,
                change_type="updated" if old_value is not None else "created",
                old_value=old_value,
                new_value=value,
                changed_by=user,
                timestamp=datetime.now(),
                version_id=version.version_id,
                reason=description
            )
            
            self.change_history.append(change_event)
            
            # Notify watchers and callbacks
            await self._notify_config_change(change_event)
            
            # Sync to other nodes if enabled
            if self.sync_enabled:
                asyncio.create_task(self._sync_to_peers(key_path, value, scope))
            
            self.logger.info(f"Set configuration: {key_path} = {value} (version {version.version_number})")
            
            return version.version_id
            
        except Exception as e:
            self.logger.error(f"Failed to set configuration {key_path}: {e}")
            raise Gaudi3ScaleError(f"Configuration set failed: {e}")
    
    async def get_config(self, 
                        key_path: str,
                        scope: ConfigScope = ConfigScope.GLOBAL,
                        default: Any = None,
                        use_cache: bool = True) -> Any:
        """Get configuration value.
        
        Args:
            key_path: Configuration key path
            scope: Configuration scope
            default: Default value if not found
            use_cache: Whether to use cached value
            
        Returns:
            Configuration value or default
        """
        try:
            # Check cache first if enabled
            if use_cache and key_path in self.config_cache:
                cache_time = self.cache_timestamps.get(key_path, datetime.min)
                if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                    return self.config_cache[key_path]
            
            # Get from config keys
            if key_path in self.config_keys:
                config_key = self.config_keys[key_path]
                
                # Check scope match
                if config_key.scope != scope:
                    return default
                
                value = config_key.current_value
                
                # Decrypt if needed
                if config_key.encrypted and value is not None:
                    value = await self._decrypt_config_value(value)
                
                # Update cache
                if use_cache and value is not None:
                    self.config_cache[key_path] = value
                    self.cache_timestamps[key_path] = datetime.now()
                
                return value if value is not None else default
            
            # Try to load from persistent storage
            config_key = await self._load_config_from_storage(key_path, scope)
            if config_key:
                self.config_keys[key_path] = config_key
                return await self.get_config(key_path, scope, default, use_cache)
            
            return default
            
        except Exception as e:
            self.logger.error(f"Failed to get configuration {key_path}: {e}")
            return default
    
    async def delete_config(self, 
                           key_path: str,
                           user: str = "system",
                           reason: str = "") -> bool:
        """Delete configuration key.
        
        Args:
            key_path: Configuration key path
            user: User performing deletion
            reason: Deletion reason
            
        Returns:
            True if deleted successfully
        """
        try:
            if key_path not in self.config_keys:
                return False
            
            config_key = self.config_keys[key_path]
            old_value = config_key.current_value
            
            # Mark all versions as archived
            for version in config_key.versions:
                version.status = ConfigStatus.ARCHIVED
            
            # Remove from active configs
            del self.config_keys[key_path]
            
            # Remove from cache
            self.config_cache.pop(key_path, None)
            self.cache_timestamps.pop(key_path, None)
            
            # Delete from persistent storage
            await self._delete_from_storage(key_path)
            
            # Create change event
            change_event = ConfigChangeEvent(
                event_id=str(uuid.uuid4()),
                key_path=key_path,
                scope=config_key.scope,
                change_type="deleted",
                old_value=old_value,
                new_value=None,
                changed_by=user,
                timestamp=datetime.now(),
                version_id="",
                reason=reason
            )
            
            self.change_history.append(change_event)
            
            # Notify watchers
            await self._notify_config_change(change_event)
            
            self.logger.info(f"Deleted configuration: {key_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete configuration {key_path}: {e}")
            return False
    
    async def list_configs(self, 
                          scope: Optional[ConfigScope] = None,
                          prefix: str = "",
                          include_encrypted: bool = False) -> List[Dict[str, Any]]:
        """List configuration keys.
        
        Args:
            scope: Filter by scope
            prefix: Filter by key prefix
            include_encrypted: Whether to include encrypted configs
            
        Returns:
            List of configuration summaries
        """
        configs = []
        
        for key_path, config_key in self.config_keys.items():
            # Apply filters
            if scope and config_key.scope != scope:
                continue
            
            if prefix and not key_path.startswith(prefix):
                continue
            
            if not include_encrypted and config_key.encrypted:
                continue
            
            # Build summary
            config_summary = {
                "key_path": key_path,
                "scope": config_key.scope.value,
                "format": config_key.format.value,
                "source": config_key.source.value,
                "encrypted": config_key.encrypted,
                "version_count": len(config_key.versions),
                "current_version": config_key.active_version,
                "last_modified": max(v.created_at for v in config_key.versions) if config_key.versions else None,
                "last_sync": config_key.last_sync
            }
            
            # Include current value if not encrypted or specifically requested
            if not config_key.encrypted or include_encrypted:
                config_summary["current_value"] = config_key.current_value
            
            configs.append(config_summary)
        
        # Sort by key path
        configs.sort(key=lambda c: c["key_path"])
        
        return configs
    
    async def watch_config(self, 
                          key_path: str,
                          callback: Callable[[str, Any, Any], None]):
        """Watch for configuration changes.
        
        Args:
            key_path: Configuration key to watch
            callback: Callback function (key, old_value, new_value)
        """
        if key_path not in self.config_watchers:
            self.config_watchers[key_path] = []
        
        self.config_watchers[key_path].append(callback)
        self.logger.info(f"Added watcher for configuration: {key_path}")
    
    def unwatch_config(self, 
                      key_path: str,
                      callback: Callable[[str, Any, Any], None]):
        """Stop watching configuration changes.
        
        Args:
            key_path: Configuration key
            callback: Callback function to remove
        """
        if key_path in self.config_watchers:
            try:
                self.config_watchers[key_path].remove(callback)
                if not self.config_watchers[key_path]:
                    del self.config_watchers[key_path]
                self.logger.info(f"Removed watcher for configuration: {key_path}")
            except ValueError:
                pass
    
    async def get_config_history(self, 
                                key_path: str,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Get configuration change history.
        
        Args:
            key_path: Configuration key path
            limit: Maximum number of changes to return
            
        Returns:
            List of configuration changes
        """
        changes = [
            change for change in self.change_history
            if change.key_path == key_path
        ]
        
        # Sort by timestamp (newest first)
        changes.sort(key=lambda c: c.timestamp, reverse=True)
        
        # Limit results
        changes = changes[:limit]
        
        return [
            {
                "event_id": change.event_id,
                "change_type": change.change_type,
                "old_value": change.old_value,
                "new_value": change.new_value,
                "changed_by": change.changed_by,
                "timestamp": change.timestamp,
                "version_id": change.version_id,
                "reason": change.reason
            }
            for change in changes
        ]
    
    async def rollback_config(self, 
                             key_path: str,
                             version_id: str,
                             user: str = "system",
                             reason: str = "") -> bool:
        """Rollback configuration to a previous version.
        
        Args:
            key_path: Configuration key path
            version_id: Version to rollback to
            user: User performing rollback
            reason: Rollback reason
            
        Returns:
            True if rollback successful
        """
        try:
            if key_path not in self.config_keys:
                return False
            
            config_key = self.config_keys[key_path]
            
            # Find the target version
            target_version = next(
                (v for v in config_key.versions if v.version_id == version_id),
                None
            )
            
            if not target_version:
                return False
            
            old_value = config_key.current_value
            
            # Set target version as active
            config_key.active_version = version_id
            target_version.status = ConfigStatus.ACTIVE
            
            # Deactivate other versions
            for version in config_key.versions:
                if version.version_id != version_id:
                    version.status = ConfigStatus.DEPRECATED
            
            # Update persistent storage
            await self._persist_config(config_key)
            
            # Update cache
            new_value = target_version.config_data
            if config_key.encrypted:
                new_value = await self._decrypt_config_value(new_value)
            
            self.config_cache[key_path] = new_value
            self.cache_timestamps[key_path] = datetime.now()
            
            # Create change event
            change_event = ConfigChangeEvent(
                event_id=str(uuid.uuid4()),
                key_path=key_path,
                scope=config_key.scope,
                change_type="rollback",
                old_value=old_value,
                new_value=new_value,
                changed_by=user,
                timestamp=datetime.now(),
                version_id=version_id,
                reason=f"Rollback: {reason}"
            )
            
            self.change_history.append(change_event)
            
            # Notify watchers
            await self._notify_config_change(change_event)
            
            self.logger.info(f"Rolled back configuration {key_path} to version {version_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback configuration {key_path}: {e}")
            return False
    
    def add_config_schema(self, 
                         key_path: str,
                         schema: Dict[str, Any]):
        """Add validation schema for configuration key.
        
        Args:
            key_path: Configuration key path
            schema: JSON schema for validation
        """
        self.config_schemas[key_path] = schema
        self.logger.info(f"Added schema for configuration: {key_path}")
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status.
        
        Returns:
            Sync status information
        """
        return {
            "sync_enabled": self.sync_enabled,
            "last_sync_time": self.last_sync_time,
            "sync_interval": self.sync_interval,
            "config_count": len(self.config_keys),
            "cache_size": len(self.config_cache),
            "watchers": sum(len(watchers) for watchers in self.config_watchers.values()),
            "change_history_size": len(self.change_history)
        }
    
    async def export_configs(self, 
                            scope: Optional[ConfigScope] = None,
                            format: ConfigFormat = ConfigFormat.YAML) -> str:
        """Export configurations to specified format.
        
        Args:
            scope: Filter by scope
            format: Export format
            
        Returns:
            Exported configuration string
        """
        configs = {}
        
        for key_path, config_key in self.config_keys.items():
            if scope and config_key.scope != scope:
                continue
            
            if not config_key.encrypted:  # Don't export encrypted configs
                value = config_key.current_value
                if value is not None:
                    # Create nested structure
                    nested_config = self._create_nested_dict(key_path, value)
                    self._merge_dict(configs, nested_config)
        
        # Format output
        if format == ConfigFormat.YAML:
            return yaml.dump(configs, default_flow_style=False)
        elif format == ConfigFormat.JSON:
            return json.dumps(configs, indent=2)
        else:
            return str(configs)
    
    async def import_configs(self, 
                            config_data: str,
                            format: ConfigFormat = ConfigFormat.YAML,
                            scope: ConfigScope = ConfigScope.GLOBAL,
                            user: str = "system",
                            dry_run: bool = False) -> List[str]:
        """Import configurations from data.
        
        Args:
            config_data: Configuration data string
            format: Data format
            scope: Import scope
            user: User performing import
            dry_run: If True, validate but don't apply changes
            
        Returns:
            List of imported/would-be-imported key paths
        """
        try:
            # Parse configuration data
            if format == ConfigFormat.YAML:
                parsed_data = yaml.safe_load(config_data)
            elif format == ConfigFormat.JSON:
                parsed_data = json.loads(config_data)
            else:
                raise Gaudi3ScaleError(f"Unsupported import format: {format}")
            
            # Flatten nested configuration
            flattened = self._flatten_dict(parsed_data)
            
            imported_keys = []
            
            for key_path, value in flattened.items():
                if not dry_run:
                    await self.set_config(
                        key_path=key_path,
                        value=value,
                        scope=scope,
                        user=user,
                        description=f"Imported from {format} data"
                    )
                
                imported_keys.append(key_path)
            
            if not dry_run:
                self.logger.info(f"Imported {len(imported_keys)} configurations")
            
            return imported_keys
            
        except Exception as e:
            self.logger.error(f"Failed to import configurations: {e}")
            raise Gaudi3ScaleError(f"Configuration import failed: {e}")
    
    def _create_nested_dict(self, key_path: str, value: Any) -> Dict[str, Any]:
        """Create nested dictionary from dot-notation key path.
        
        Args:
            key_path: Dot-notation key path
            value: Value to set
            
        Returns:
            Nested dictionary
        """
        keys = key_path.split('.')
        result = {}
        current = result
        
        for key in keys[:-1]:
            current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        return result
    
    def _flatten_dict(self, 
                     data: Dict[str, Any], 
                     parent_key: str = "", 
                     sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary to dot-notation keys.
        
        Args:
            data: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Key separator
            
        Returns:
            Flattened dictionary
        """
        items = []
        
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    def _merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Merge source dictionary into target dictionary.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dict(target[key], value)
            else:
                target[key] = value
    
    async def _validate_config(self, 
                              key_path: str, 
                              value: Any, 
                              scope: ConfigScope):
        """Validate configuration value against schema.
        
        Args:
            key_path: Configuration key path
            value: Value to validate
            scope: Configuration scope
        """
        if key_path in self.config_schemas:
            schema = self.config_schemas[key_path]
            # Simplified validation - in production would use jsonschema library
            # For now, just check basic type constraints
            if "type" in schema:
                expected_type = schema["type"]
                if expected_type == "string" and not isinstance(value, str):
                    raise Gaudi3ScaleError(f"Configuration {key_path} must be a string")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    raise Gaudi3ScaleError(f"Configuration {key_path} must be a number")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    raise Gaudi3ScaleError(f"Configuration {key_path} must be a boolean")
    
    async def _encrypt_config_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt configuration data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        # Placeholder for encryption - in production would use proper encryption
        # For now, just return the data as-is
        return data
    
    async def _decrypt_config_value(self, value: Any) -> Any:
        """Decrypt configuration value.
        
        Args:
            value: Encrypted value
            
        Returns:
            Decrypted value
        """
        # Placeholder for decryption - in production would use proper decryption
        return value
    
    async def _persist_config(self, config_key: ConfigKey):
        """Persist configuration to storage.
        
        Args:
            config_key: Configuration key to persist
        """
        try:
            # Serialize configuration key
            config_data = {
                "key_id": config_key.key_id,
                "key_path": config_key.key_path,
                "scope": config_key.scope.value,
                "format": config_key.format.value,
                "source": config_key.source.value,
                "encrypted": config_key.encrypted,
                "active_version": config_key.active_version,
                "metadata": config_key.metadata,
                "versions": [
                    {
                        "version_id": v.version_id,
                        "version_number": v.version_number,
                        "config_data": v.config_data,
                        "created_at": v.created_at.isoformat(),
                        "created_by": v.created_by,
                        "description": v.description,
                        "checksum": v.checksum,
                        "status": v.status.value
                    }
                    for v in config_key.versions
                ]
            }
            
            # Store in data manager
            await self.data_manager.storage_manager.put_object(
                name=f"config_{config_key.key_path}",
                data=json.dumps(config_data).encode('utf-8'),
                content_type="application/json",
                metadata={
                    "type": "configuration",
                    "scope": config_key.scope.value,
                    "key_path": config_key.key_path
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to persist configuration {config_key.key_path}: {e}")
    
    async def _load_config_from_storage(self, 
                                       key_path: str, 
                                       scope: ConfigScope) -> Optional[ConfigKey]:
        """Load configuration from persistent storage.
        
        Args:
            key_path: Configuration key path
            scope: Configuration scope
            
        Returns:
            Configuration key or None if not found
        """
        try:
            # Get from data manager
            config_data = await self.data_manager.storage_manager.get_object(f"config_{key_path}")
            if not config_data:
                return None
            
            # Deserialize
            config_dict = json.loads(config_data.decode('utf-8'))
            
            # Reconstruct ConfigKey
            config_key = ConfigKey(
                key_id=config_dict["key_id"],
                key_path=config_dict["key_path"],
                scope=ConfigScope(config_dict["scope"]),
                format=ConfigFormat(config_dict["format"]),
                source=ConfigSource(config_dict["source"]),
                encrypted=config_dict["encrypted"],
                active_version=config_dict["active_version"],
                metadata=config_dict["metadata"],
                versions=[]
            )
            
            # Reconstruct versions
            for v_dict in config_dict["versions"]:
                version = ConfigVersion(
                    version_id=v_dict["version_id"],
                    version_number=v_dict["version_number"],
                    config_data=v_dict["config_data"],
                    created_at=datetime.fromisoformat(v_dict["created_at"]),
                    created_by=v_dict["created_by"],
                    description=v_dict["description"],
                    checksum=v_dict["checksum"],
                    status=ConfigStatus(v_dict["status"])
                )
                config_key.versions.append(version)
            
            return config_key
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration {key_path}: {e}")
            return None
    
    async def _delete_from_storage(self, key_path: str):
        """Delete configuration from persistent storage.
        
        Args:
            key_path: Configuration key path to delete
        """
        try:
            await self.data_manager.storage_manager.delete_object(f"config_{key_path}")
        except Exception as e:
            self.logger.error(f"Failed to delete configuration {key_path} from storage: {e}")
    
    async def _notify_config_change(self, change_event: ConfigChangeEvent):
        """Notify watchers and callbacks of configuration change.
        
        Args:
            change_event: Configuration change event
        """
        key_path = change_event.key_path
        
        # Notify specific key watchers
        if key_path in self.config_watchers:
            for callback in self.config_watchers[key_path]:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, 
                        callback, 
                        key_path, 
                        change_event.old_value, 
                        change_event.new_value
                    )
                except Exception as e:
                    self.logger.error(f"Config watcher callback failed: {e}")
        
        # Notify global change callbacks
        for callback in self.change_callbacks:
            try:
                await asyncio.get_event_loop().run_in_executor(None, callback, change_event)
            except Exception as e:
                self.logger.error(f"Config change callback failed: {e}")
    
    async def _sync_configurations(self):
        """Synchronize configurations with other nodes."""
        while self.sync_enabled:
            try:
                if self.sync_enabled:
                    await self._perform_sync()
                    self.last_sync_time = datetime.now()
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                self.logger.error(f"Configuration sync error: {e}")
                await asyncio.sleep(30)
    
    async def _perform_sync(self):
        """Perform configuration synchronization with peer nodes."""
        try:
            # Get config services from registry
            config_services = self.service_registry.discover_services(
                service_type=ServiceType.CONFIG_SERVER,
                status=ServiceStatus.HEALTHY
            )
            
            # Sync with each peer
            for service in config_services:
                if service.node_id != self.node_id:
                    await self._sync_with_peer(service)
                    
        except Exception as e:
            self.logger.error(f"Sync operation failed: {e}")
    
    async def _sync_with_peer(self, peer_service: ServiceInfo):
        """Sync configurations with a peer node.
        
        Args:
            peer_service: Peer service information
        """
        try:
            # Simplified sync - in production would implement conflict resolution
            # For now, just log the sync attempt
            self.logger.debug(f"Syncing with peer: {peer_service.node_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to sync with peer {peer_service.node_id}: {e}")
    
    async def _sync_to_peers(self, key_path: str, value: Any, scope: ConfigScope):
        """Sync configuration change to peer nodes.
        
        Args:
            key_path: Configuration key path
            value: New value
            scope: Configuration scope
        """
        try:
            # Get config services
            config_services = self.service_registry.discover_services(
                service_type=ServiceType.CONFIG_SERVER,
                status=ServiceStatus.HEALTHY
            )
            
            # Notify peers of change (simplified)
            for service in config_services:
                if service.node_id != self.node_id:
                    self.logger.debug(f"Notifying peer {service.node_id} of config change: {key_path}")
                    
        except Exception as e:
            self.logger.error(f"Failed to sync to peers: {e}")
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries."""
        while True:
            try:
                current_time = datetime.now()
                expired_keys = []
                
                for key, timestamp in self.cache_timestamps.items():
                    if (current_time - timestamp).total_seconds() > self.cache_ttl:
                        expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    self.config_cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
                
                # Limit change history
                if len(self.change_history) > self.max_history_size:
                    self.change_history = self.change_history[-self.max_history_size:]
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)


class ConfigSynchronizer:
    """Handles configuration synchronization between cluster nodes."""
    
    def __init__(self, config_manager: DistributedConfigManager):
        """Initialize configuration synchronizer.
        
        Args:
            config_manager: Distributed configuration manager
        """
        self.config_manager = config_manager
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Sync state
        self.sync_in_progress = False
        self.last_full_sync: Optional[datetime] = None
        self.sync_errors: List[str] = []
        
    async def perform_full_sync(self) -> bool:
        """Perform full configuration synchronization.
        
        Returns:
            True if sync successful
        """
        if self.sync_in_progress:
            return False
        
        try:
            self.sync_in_progress = True
            self.sync_errors.clear()
            
            self.logger.info("Starting full configuration synchronization")
            
            # Get all peer config services
            peers = self.config_manager.service_registry.discover_services(
                service_type=ServiceType.CONFIG_SERVER,
                status=ServiceStatus.HEALTHY
            )
            
            peers = [p for p in peers if p.node_id != self.config_manager.node_id]
            
            if not peers:
                self.logger.info("No peers found for synchronization")
                return True
            
            # Sync with each peer
            for peer in peers:
                try:
                    await self._sync_with_peer_full(peer)
                except Exception as e:
                    error_msg = f"Failed to sync with peer {peer.node_id}: {e}"
                    self.sync_errors.append(error_msg)
                    self.logger.error(error_msg)
            
            self.last_full_sync = datetime.now()
            
            success = len(self.sync_errors) == 0
            status = "completed" if success else "completed with errors"
            
            self.logger.info(f"Full synchronization {status}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Full sync failed: {e}")
            return False
        finally:
            self.sync_in_progress = False
    
    async def _sync_with_peer_full(self, peer: ServiceInfo):
        """Perform full sync with a specific peer.
        
        Args:
            peer: Peer service information
        """
        # Placeholder for peer-to-peer configuration sync
        # In production, would implement actual sync protocol
        self.logger.debug(f"Full sync with peer {peer.node_id}")
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status.
        
        Returns:
            Sync status information
        """
        return {
            "sync_in_progress": self.sync_in_progress,
            "last_full_sync": self.last_full_sync,
            "sync_errors": self.sync_errors,
            "error_count": len(self.sync_errors)
        }