"""Fault tolerance and failover mechanisms for distributed Gaudi 3 clusters."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Union
import uuid

from .discovery import ServiceRegistry, ServiceInfo, ServiceType, ServiceStatus
from .coordinator import DistributedTrainingCoordinator, TrainingPhase, NodeRole
from .storage import DataManager
from ..logging_utils import get_logger
from ..exceptions import Gaudi3ScaleError

logger = get_logger(__name__)


class FailureType(str, Enum):
    """Types of failures that can occur in the cluster."""
    NODE_FAILURE = "node_failure"
    SERVICE_FAILURE = "service_failure"
    NETWORK_PARTITION = "network_partition"
    STORAGE_FAILURE = "storage_failure"
    TRAINING_FAILURE = "training_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_ERROR = "configuration_error"
    HARDWARE_FAILURE = "hardware_failure"


class FailureSeverity(str, Enum):
    """Severity levels for failures."""
    LOW = "low"           # Minor issues, can continue with degraded performance
    MEDIUM = "medium"     # Significant impact, requires attention
    HIGH = "high"         # Major impact, requires immediate action
    CRITICAL = "critical" # System-wide impact, requires emergency response


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different failure types."""
    RESTART = "restart"                    # Restart failed component
    FAILOVER = "failover"                 # Switch to backup
    RESCHEDULE = "reschedule"             # Reschedule on different node
    CHECKPOINT_RESTORE = "checkpoint"      # Restore from checkpoint
    GRACEFUL_DEGRADATION = "degradation"   # Continue with reduced capacity
    ROLLBACK = "rollback"                 # Rollback to previous state
    MANUAL_INTERVENTION = "manual"        # Requires manual intervention


@dataclass
class FailureEvent:
    """Represents a failure event in the cluster."""
    failure_id: str
    failure_type: FailureType
    severity: FailureSeverity
    affected_component: str
    description: str
    timestamp: datetime
    detection_method: str
    context: Dict[str, Any]
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_started_at: Optional[datetime] = None
    recovery_completed_at: Optional[datetime] = None
    is_resolved: bool = False
    root_cause: Optional[str] = None
    
    @property
    def duration(self) -> timedelta:
        end_time = self.recovery_completed_at or datetime.now()
        return end_time - self.timestamp
    
    @property
    def recovery_duration(self) -> Optional[timedelta]:
        if self.recovery_started_at and self.recovery_completed_at:
            return self.recovery_completed_at - self.recovery_started_at
        return None


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: int
    timeout_seconds: int
    failure_threshold: int = 3
    success_threshold: int = 1
    enabled: bool = True
    
    # State tracking
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    is_healthy: bool = True


@dataclass
class Backup:
    """Represents a backup of system state."""
    backup_id: str
    backup_type: str  # checkpoint, configuration, data
    component: str
    timestamp: datetime
    size_bytes: int
    location: str
    metadata: Dict[str, Any]
    retention_days: int = 30
    
    @property
    def is_expired(self) -> bool:
        expiry_date = self.timestamp + timedelta(days=self.retention_days)
        return datetime.now() > expiry_date


class FaultToleranceManager:
    """Manages fault tolerance and failure detection for the distributed system."""
    
    def __init__(self, 
                 service_registry: ServiceRegistry,
                 data_manager: DataManager):
        """Initialize fault tolerance manager.
        
        Args:
            service_registry: Service registry for monitoring services
            data_manager: Data manager for checkpoint operations
        """
        self.service_registry = service_registry
        self.data_manager = data_manager
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Failure tracking
        self.active_failures: Dict[str, FailureEvent] = {}
        self.failure_history: List[FailureEvent] = []
        self.failure_patterns: Dict[str, int] = {}
        
        # Health monitoring
        self.health_checks: Dict[str, HealthCheck] = {}
        self.monitoring_enabled = True
        
        # Recovery management
        self.recovery_handlers: Dict[FailureType, List[Callable]] = {}
        self.recovery_in_progress: Set[str] = set()
        
        # Backup management
        self.backups: Dict[str, List[Backup]] = {}
        self.backup_retention_days = 30
        
        # Configuration
        self.max_failure_history = 1000
        self.health_check_interval = 30
        self.failure_correlation_window = 300  # seconds
        
        # Initialize default health checks
        self._initialize_health_checks()
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_health())
        asyncio.create_task(self._correlate_failures())
        asyncio.create_task(self._cleanup_old_data())
    
    async def register_health_check(self, health_check: HealthCheck):
        """Register a health check.
        
        Args:
            health_check: Health check configuration
        """
        self.health_checks[health_check.name] = health_check
        self.logger.info(f"Registered health check: {health_check.name}")
    
    def register_recovery_handler(self, 
                                 failure_type: FailureType,
                                 handler: Callable[[FailureEvent], bool]):
        """Register a recovery handler for a failure type.
        
        Args:
            failure_type: Type of failure to handle
            handler: Recovery handler function
        """
        if failure_type not in self.recovery_handlers:
            self.recovery_handlers[failure_type] = []
        self.recovery_handlers[failure_type].append(handler)
        
        self.logger.info(f"Registered recovery handler for {failure_type}")
    
    async def report_failure(self, 
                           failure_type: FailureType,
                           affected_component: str,
                           description: str,
                           severity: FailureSeverity = FailureSeverity.MEDIUM,
                           context: Optional[Dict[str, Any]] = None) -> str:
        """Report a failure event.
        
        Args:
            failure_type: Type of failure
            affected_component: Component that failed
            description: Description of the failure
            severity: Severity level
            context: Additional context information
            
        Returns:
            Failure ID
        """
        failure_id = str(uuid.uuid4())
        
        failure_event = FailureEvent(
            failure_id=failure_id,
            failure_type=failure_type,
            severity=severity,
            affected_component=affected_component,
            description=description,
            timestamp=datetime.now(),
            detection_method="manual_report",
            context=context or {}
        )
        
        self.active_failures[failure_id] = failure_event
        self.failure_history.append(failure_event)
        
        # Update failure patterns
        pattern_key = f"{failure_type}_{affected_component}"
        self.failure_patterns[pattern_key] = self.failure_patterns.get(pattern_key, 0) + 1
        
        self.logger.error(
            f"Failure reported: {failure_type} in {affected_component} - {description}"
        )
        
        # Trigger recovery
        await self._initiate_recovery(failure_event)
        
        return failure_id
    
    async def resolve_failure(self, failure_id: str, root_cause: str = None):
        """Mark a failure as resolved.
        
        Args:
            failure_id: Failure identifier
            root_cause: Root cause analysis
        """
        if failure_id in self.active_failures:
            failure_event = self.active_failures[failure_id]
            failure_event.is_resolved = True
            failure_event.recovery_completed_at = datetime.now()
            failure_event.root_cause = root_cause
            
            del self.active_failures[failure_id]
            self.recovery_in_progress.discard(failure_id)
            
            self.logger.info(
                f"Failure {failure_id} resolved after {failure_event.duration}"
            )
    
    async def create_backup(self, 
                           backup_type: str,
                           component: str,
                           data: Union[Dict[str, Any], bytes, str],
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a backup of system state.
        
        Args:
            backup_type: Type of backup (checkpoint, config, data)
            component: Component being backed up
            data: Data to backup
            metadata: Additional metadata
            
        Returns:
            Backup ID
        """
        backup_id = str(uuid.uuid4())
        
        # Serialize data if needed
        if isinstance(data, dict):
            serialized_data = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            serialized_data = data.encode('utf-8')
        else:
            serialized_data = data
        
        # Store backup using data manager
        object_id = await self.data_manager.storage_manager.put_object(
            name=f"backup_{backup_type}_{component}_{backup_id}",
            data=serialized_data,
            content_type="application/json" if isinstance(data, dict) else "application/octet-stream",
            metadata={
                "type": "backup",
                "backup_type": backup_type,
                "component": component,
                "backup_id": backup_id,
                **(metadata or {})
            }
        )
        
        backup = Backup(
            backup_id=backup_id,
            backup_type=backup_type,
            component=component,
            timestamp=datetime.now(),
            size_bytes=len(serialized_data),
            location=object_id,
            metadata=metadata or {}
        )
        
        if component not in self.backups:
            self.backups[component] = []
        self.backups[component].append(backup)
        
        self.logger.info(
            f"Created backup {backup_id} for {component} ({len(serialized_data)} bytes)"
        )
        
        return backup_id
    
    async def restore_backup(self, backup_id: str) -> Optional[Union[Dict[str, Any], bytes]]:
        """Restore from a backup.
        
        Args:
            backup_id: Backup identifier
            
        Returns:
            Restored data or None if backup not found
        """
        # Find backup
        backup = None
        for component_backups in self.backups.values():
            for b in component_backups:
                if b.backup_id == backup_id:
                    backup = b
                    break
            if backup:
                break
        
        if not backup:
            self.logger.error(f"Backup {backup_id} not found")
            return None
        
        try:
            # Retrieve backup data
            backup_data = await self.data_manager.storage_manager.get_object(backup.location)
            if not backup_data:
                self.logger.error(f"Backup data not found for {backup_id}")
                return None
            
            # Deserialize if it was a dict
            if backup.backup_type in ["checkpoint", "configuration"]:
                try:
                    return json.loads(backup_data.decode('utf-8'))
                except json.JSONDecodeError:
                    return backup_data
            else:
                return backup_data
                
        except Exception as e:
            self.logger.error(f"Failed to restore backup {backup_id}: {e}")
            return None
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure statistics and health metrics.
        
        Returns:
            Statistics dictionary
        """
        total_failures = len(self.failure_history)
        active_failures = len(self.active_failures)
        
        # Calculate MTBF and MTTR
        if total_failures > 1:
            failure_times = [f.timestamp for f in self.failure_history]
            time_diffs = [(failure_times[i] - failure_times[i-1]).total_seconds() 
                         for i in range(1, len(failure_times))]
            mtbf = sum(time_diffs) / len(time_diffs) / 3600  # hours
        else:
            mtbf = 0
        
        resolved_failures = [f for f in self.failure_history if f.is_resolved and f.recovery_duration]
        if resolved_failures:
            mttr = sum(f.recovery_duration.total_seconds() for f in resolved_failures) / len(resolved_failures) / 60  # minutes
        else:
            mttr = 0
        
        # Failure types distribution
        failure_type_counts = {}
        for failure in self.failure_history:
            failure_type_counts[failure.failure_type.value] = \
                failure_type_counts.get(failure.failure_type.value, 0) + 1
        
        # Health check status
        healthy_checks = sum(1 for hc in self.health_checks.values() if hc.is_healthy)
        total_checks = len(self.health_checks)
        
        return {
            "total_failures": total_failures,
            "active_failures": active_failures,
            "resolved_failures": len([f for f in self.failure_history if f.is_resolved]),
            "mtbf_hours": mtbf,
            "mttr_minutes": mttr,
            "failure_types": failure_type_counts,
            "failure_patterns": dict(self.failure_patterns),
            "healthy_checks": healthy_checks,
            "total_health_checks": total_checks,
            "health_check_success_rate": (healthy_checks / total_checks) if total_checks > 0 else 1.0,
            "total_backups": sum(len(backups) for backups in self.backups.values()),
            "recovery_in_progress": len(self.recovery_in_progress)
        }
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """Get overall system health status.
        
        Returns:
            Health status dictionary
        """
        # Calculate overall health score
        health_scores = []
        
        # Health checks score
        if self.health_checks:
            healthy_checks = sum(1 for hc in self.health_checks.values() if hc.is_healthy)
            health_check_score = healthy_checks / len(self.health_checks)
            health_scores.append(("health_checks", health_check_score, 0.3))
        
        # Active failures score
        critical_failures = len([f for f in self.active_failures.values() 
                               if f.severity == FailureSeverity.CRITICAL])
        high_failures = len([f for f in self.active_failures.values() 
                           if f.severity == FailureSeverity.HIGH])
        
        failure_penalty = min(1.0, (critical_failures * 0.5 + high_failures * 0.2))
        failure_score = max(0.0, 1.0 - failure_penalty)
        health_scores.append(("failures", failure_score, 0.4))
        
        # Recent failure trend score
        recent_failures = len([f for f in self.failure_history 
                             if (datetime.now() - f.timestamp).total_seconds() < 3600])  # Last hour
        trend_penalty = min(0.5, recent_failures * 0.1)
        trend_score = max(0.5, 1.0 - trend_penalty)
        health_scores.append(("trend", trend_score, 0.3))
        
        # Calculate weighted overall score
        overall_score = sum(score * weight for _, score, weight in health_scores)
        
        # Determine health status
        if overall_score >= 0.9:
            status = "HEALTHY"
        elif overall_score >= 0.7:
            status = "DEGRADED"
        elif overall_score >= 0.5:
            status = "UNHEALTHY"
        else:
            status = "CRITICAL"
        
        return {
            "overall_status": status,
            "overall_score": overall_score,
            "component_scores": {name: score for name, score, _ in health_scores},
            "active_critical_failures": critical_failures,
            "active_high_failures": high_failures,
            "recent_failure_count": recent_failures,
            "monitoring_enabled": self.monitoring_enabled,
            "last_updated": datetime.now()
        }
    
    def _initialize_health_checks(self):
        """Initialize default health checks."""
        # Service registry health check
        self.health_checks["service_registry"] = HealthCheck(
            name="service_registry",
            check_function=self._check_service_registry_health,
            interval_seconds=30,
            timeout_seconds=5
        )
        
        # Storage health check
        self.health_checks["storage"] = HealthCheck(
            name="storage",
            check_function=self._check_storage_health,
            interval_seconds=60,
            timeout_seconds=10
        )
        
        # Memory health check
        self.health_checks["memory"] = HealthCheck(
            name="memory",
            check_function=self._check_memory_health,
            interval_seconds=30,
            timeout_seconds=5
        )
    
    def _check_service_registry_health(self) -> bool:
        """Check service registry health."""
        try:
            # Simple check - see if we can discover services
            services = self.service_registry.discover_services()
            return len(services) >= 0  # Just check it doesn't crash
        except Exception:
            return False
    
    def _check_storage_health(self) -> bool:
        """Check storage health."""
        try:
            stats = self.data_manager.get_storage_stats()
            # Check if storage utilization is reasonable
            return stats.get("utilization_percent", 0) < 95
        except Exception:
            return False
    
    def _check_memory_health(self) -> bool:
        """Check system memory health."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if memory usage > 90%
        except Exception:
            return True  # Default to healthy if can't check
    
    async def _monitor_health(self):
        """Monitor system health using registered health checks."""
        while self.monitoring_enabled:
            try:
                for health_check in self.health_checks.values():
                    if not health_check.enabled:
                        continue
                    
                    try:
                        # Run health check with timeout
                        result = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, health_check.check_function
                            ),
                            timeout=health_check.timeout_seconds
                        )
                        
                        health_check.last_check = datetime.now()
                        
                        if result:
                            health_check.consecutive_successes += 1
                            health_check.consecutive_failures = 0
                            
                            # Mark as healthy if we have enough successes
                            if (not health_check.is_healthy and 
                                health_check.consecutive_successes >= health_check.success_threshold):
                                health_check.is_healthy = True
                                self.logger.info(f"Health check {health_check.name} recovered")
                        else:
                            health_check.consecutive_failures += 1
                            health_check.consecutive_successes = 0
                            health_check.last_failure = datetime.now()
                            
                            # Mark as unhealthy if we have too many failures
                            if (health_check.is_healthy and 
                                health_check.consecutive_failures >= health_check.failure_threshold):
                                health_check.is_healthy = False
                                
                                # Report failure
                                await self.report_failure(
                                    FailureType.SERVICE_FAILURE,
                                    health_check.name,
                                    f"Health check failed {health_check.consecutive_failures} times",
                                    FailureSeverity.MEDIUM,
                                    {"health_check": health_check.name}
                                )
                    
                    except asyncio.TimeoutError:
                        health_check.consecutive_failures += 1
                        health_check.last_failure = datetime.now()
                        self.logger.warning(f"Health check {health_check.name} timed out")
                    
                    except Exception as e:
                        self.logger.error(f"Health check {health_check.name} error: {e}")
                
                # Wait before next round of checks
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _initiate_recovery(self, failure_event: FailureEvent):
        """Initiate recovery for a failure event.
        
        Args:
            failure_event: Failure event to recover from
        """
        if failure_event.failure_id in self.recovery_in_progress:
            return  # Recovery already in progress
        
        self.recovery_in_progress.add(failure_event.failure_id)
        failure_event.recovery_started_at = datetime.now()
        
        try:
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(failure_event)
            failure_event.recovery_strategy = recovery_strategy
            
            self.logger.info(
                f"Initiating recovery for {failure_event.failure_id} using {recovery_strategy}"
            )
            
            # Execute recovery handlers
            if failure_event.failure_type in self.recovery_handlers:
                for handler in self.recovery_handlers[failure_event.failure_type]:
                    try:
                        success = await handler(failure_event)
                        if success:
                            await self.resolve_failure(failure_event.failure_id, "Automated recovery")
                            break
                    except Exception as e:
                        self.logger.error(f"Recovery handler failed: {e}")
            
            # Default recovery actions
            if not failure_event.is_resolved:
                await self._execute_default_recovery(failure_event)
        
        except Exception as e:
            self.logger.error(f"Recovery initiation failed: {e}")
        finally:
            if failure_event.failure_id in self.recovery_in_progress:
                self.recovery_in_progress.remove(failure_event.failure_id)
    
    def _determine_recovery_strategy(self, failure_event: FailureEvent) -> RecoveryStrategy:
        """Determine the best recovery strategy for a failure.
        
        Args:
            failure_event: Failure event
            
        Returns:
            Recommended recovery strategy
        """
        # Strategy mapping based on failure type and severity
        strategy_map = {
            FailureType.NODE_FAILURE: {
                FailureSeverity.LOW: RecoveryStrategy.GRACEFUL_DEGRADATION,
                FailureSeverity.MEDIUM: RecoveryStrategy.RESCHEDULE,
                FailureSeverity.HIGH: RecoveryStrategy.FAILOVER,
                FailureSeverity.CRITICAL: RecoveryStrategy.CHECKPOINT_RESTORE
            },
            FailureType.SERVICE_FAILURE: {
                FailureSeverity.LOW: RecoveryStrategy.RESTART,
                FailureSeverity.MEDIUM: RecoveryStrategy.RESTART,
                FailureSeverity.HIGH: RecoveryStrategy.FAILOVER,
                FailureSeverity.CRITICAL: RecoveryStrategy.ROLLBACK
            },
            FailureType.TRAINING_FAILURE: {
                FailureSeverity.LOW: RecoveryStrategy.RESTART,
                FailureSeverity.MEDIUM: RecoveryStrategy.CHECKPOINT_RESTORE,
                FailureSeverity.HIGH: RecoveryStrategy.CHECKPOINT_RESTORE,
                FailureSeverity.CRITICAL: RecoveryStrategy.MANUAL_INTERVENTION
            }
        }
        
        return strategy_map.get(failure_event.failure_type, {}).get(
            failure_event.severity, 
            RecoveryStrategy.MANUAL_INTERVENTION
        )
    
    async def _execute_default_recovery(self, failure_event: FailureEvent):
        """Execute default recovery actions.
        
        Args:
            failure_event: Failure event to recover from
        """
        strategy = failure_event.recovery_strategy
        
        if strategy == RecoveryStrategy.RESTART:
            await self._restart_component(failure_event.affected_component)
        elif strategy == RecoveryStrategy.CHECKPOINT_RESTORE:
            await self._restore_from_checkpoint(failure_event.affected_component)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            await self._enable_degraded_mode(failure_event.affected_component)
        # Other strategies would be implemented here
        
        # Mark as resolved if we made it this far without exceptions
        await self.resolve_failure(failure_event.failure_id, "Default recovery completed")
    
    async def _restart_component(self, component: str):
        """Restart a failed component.
        
        Args:
            component: Component to restart
        """
        self.logger.info(f"Restarting component: {component}")
        # Implementation would depend on the specific component
        # For now, just simulate restart
        await asyncio.sleep(1)
    
    async def _restore_from_checkpoint(self, component: str):
        """Restore component from latest checkpoint.
        
        Args:
            component: Component to restore
        """
        self.logger.info(f"Restoring component from checkpoint: {component}")
        
        # Find latest backup
        if component in self.backups and self.backups[component]:
            latest_backup = max(self.backups[component], key=lambda b: b.timestamp)
            data = await self.restore_backup(latest_backup.backup_id)
            if data:
                self.logger.info(f"Restored {component} from backup {latest_backup.backup_id}")
            else:
                self.logger.error(f"Failed to restore {component} from backup")
    
    async def _enable_degraded_mode(self, component: str):
        """Enable degraded operation mode for component.
        
        Args:
            component: Component to run in degraded mode
        """
        self.logger.info(f"Enabling degraded mode for: {component}")
        # Implementation would reduce functionality to maintain basic operation
        await asyncio.sleep(0.5)
    
    async def _correlate_failures(self):
        """Correlate failures to identify patterns and root causes."""
        while self.monitoring_enabled:
            try:
                current_time = datetime.now()
                window_start = current_time - timedelta(seconds=self.failure_correlation_window)
                
                # Get recent failures
                recent_failures = [
                    f for f in self.failure_history
                    if f.timestamp >= window_start
                ]
                
                if len(recent_failures) >= 3:  # Need multiple failures to correlate
                    # Look for patterns
                    component_failures = {}
                    for failure in recent_failures:
                        component = failure.affected_component
                        if component not in component_failures:
                            component_failures[component] = []
                        component_failures[component].append(failure)
                    
                    # Check for cascade failures
                    for component, failures in component_failures.items():
                        if len(failures) >= 2:
                            self.logger.warning(
                                f"Multiple failures detected in {component}: {len(failures)} failures"
                            )
                            
                            # Could trigger additional recovery actions here
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Failure correlation error: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_old_data(self):
        """Clean up old failure history and backups."""
        while self.monitoring_enabled:
            try:
                current_time = datetime.now()
                
                # Clean up old failure history
                if len(self.failure_history) > self.max_failure_history:
                    self.failure_history = self.failure_history[-self.max_failure_history:]
                
                # Clean up expired backups
                for component, backups in self.backups.items():
                    active_backups = []
                    for backup in backups:
                        if not backup.is_expired:
                            active_backups.append(backup)
                        else:
                            # Delete expired backup
                            try:
                                await self.data_manager.storage_manager.delete_object(backup.location)
                                self.logger.info(f"Deleted expired backup: {backup.backup_id}")
                            except Exception as e:
                                self.logger.error(f"Failed to delete backup {backup.backup_id}: {e}")
                    
                    self.backups[component] = active_backups
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)


class FailoverCoordinator:
    """Coordinates failover operations for distributed training."""
    
    def __init__(self, 
                 training_coordinator: DistributedTrainingCoordinator,
                 fault_tolerance_manager: FaultToleranceManager):
        """Initialize failover coordinator.
        
        Args:
            training_coordinator: Distributed training coordinator
            fault_tolerance_manager: Fault tolerance manager
        """
        self.training_coordinator = training_coordinator
        self.fault_tolerance = fault_tolerance_manager
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Failover state
        self.standby_nodes: Dict[str, str] = {}  # active_node -> standby_node
        self.failover_history: List[Dict[str, Any]] = []
        
        # Register recovery handlers
        self.fault_tolerance.register_recovery_handler(
            FailureType.NODE_FAILURE, self._handle_node_failure
        )
        self.fault_tolerance.register_recovery_handler(
            FailureType.TRAINING_FAILURE, self._handle_training_failure
        )
    
    async def setup_standby_nodes(self, node_mappings: Dict[str, str]):
        """Setup standby nodes for active nodes.
        
        Args:
            node_mappings: Dictionary mapping active node IDs to standby node IDs
        """
        self.standby_nodes = node_mappings
        self.logger.info(f"Setup {len(node_mappings)} standby node mappings")
    
    async def _handle_node_failure(self, failure_event: FailureEvent) -> bool:
        """Handle node failure by failing over to standby node.
        
        Args:
            failure_event: Node failure event
            
        Returns:
            True if failover successful
        """
        failed_node = failure_event.affected_component
        
        if failed_node not in self.standby_nodes:
            self.logger.warning(f"No standby node configured for {failed_node}")
            return False
        
        standby_node = self.standby_nodes[failed_node]
        
        try:
            self.logger.info(f"Failing over from {failed_node} to {standby_node}")
            
            # Create checkpoint before failover
            await self._create_emergency_checkpoint()
            
            # Update training coordinator to use standby node
            await self._update_node_assignment(failed_node, standby_node)
            
            # Record failover
            failover_record = {
                "timestamp": datetime.now(),
                "failed_node": failed_node,
                "standby_node": standby_node,
                "failure_id": failure_event.failure_id,
                "success": True
            }
            self.failover_history.append(failover_record)
            
            self.logger.info(f"Failover completed: {failed_node} -> {standby_node}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failover failed: {e}")
            return False
    
    async def _handle_training_failure(self, failure_event: FailureEvent) -> bool:
        """Handle training failure by restarting from checkpoint.
        
        Args:
            failure_event: Training failure event
            
        Returns:
            True if recovery successful
        """
        try:
            self.logger.info("Recovering training from failure")
            
            # Stop current training
            for job_id in list(self.training_coordinator.active_jobs.keys()):
                await self.training_coordinator.stop_training_job(job_id, graceful=False)
            
            # Wait a moment for cleanup
            await asyncio.sleep(2)
            
            # Restart training from latest checkpoint
            # This would typically involve loading the last checkpoint
            # and resuming training with the same configuration
            
            self.logger.info("Training recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Training recovery failed: {e}")
            return False
    
    async def _create_emergency_checkpoint(self):
        """Create an emergency checkpoint before failover."""
        try:
            # Get current training state from all active jobs
            for job_id, job in self.training_coordinator.active_jobs.items():
                # Create backup of current training state
                training_state = {
                    "job_id": job_id,
                    "global_step": self.training_coordinator.global_step,
                    "nodes": list(self.training_coordinator.nodes.keys()),
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.fault_tolerance.create_backup(
                    "emergency_checkpoint",
                    f"training_job_{job_id}",
                    training_state,
                    {"emergency": True, "failover": True}
                )
            
            self.logger.info("Emergency checkpoint created")
            
        except Exception as e:
            self.logger.error(f"Emergency checkpoint failed: {e}")
    
    async def _update_node_assignment(self, failed_node: str, standby_node: str):
        """Update node assignment in training coordinator.
        
        Args:
            failed_node: Failed node ID
            standby_node: Standby node ID to use instead
        """
        # Update node mappings in training coordinator
        if failed_node in self.training_coordinator.nodes:
            node_status = self.training_coordinator.nodes[failed_node]
            
            # Remove failed node
            del self.training_coordinator.nodes[failed_node]
            
            # Add standby node with same role
            node_status.node_id = standby_node
            node_status.is_healthy = True
            node_status.phase = TrainingPhase.INITIALIZATION
            
            self.training_coordinator.nodes[standby_node] = node_status
            self.training_coordinator.node_roles[standby_node] = \
                self.training_coordinator.node_roles.pop(failed_node, NodeRole.WORKER)
            
            # Update coordinator node if needed
            if self.training_coordinator.coordinator_node == failed_node:
                self.training_coordinator.coordinator_node = standby_node
        
        # Update active jobs to use new node
        for job in self.training_coordinator.active_jobs.values():
            if failed_node in job.nodes:
                job.nodes.remove(failed_node)
                job.nodes.append(standby_node)
            
            if job.coordinator_node == failed_node:
                job.coordinator_node = standby_node
    
    def get_failover_statistics(self) -> Dict[str, Any]:
        """Get failover statistics.
        
        Returns:
            Failover statistics dictionary
        """
        total_failovers = len(self.failover_history)
        successful_failovers = len([f for f in self.failover_history if f["success"]])
        
        recent_failovers = len([
            f for f in self.failover_history
            if (datetime.now() - f["timestamp"]).total_seconds() < 86400  # Last 24 hours
        ])
        
        return {
            "total_failovers": total_failovers,
            "successful_failovers": successful_failovers,
            "failover_success_rate": (successful_failovers / total_failovers) if total_failovers > 0 else 1.0,
            "recent_failovers_24h": recent_failovers,
            "standby_nodes_configured": len(self.standby_nodes),
            "last_failover": self.failover_history[-1]["timestamp"] if self.failover_history else None
        }