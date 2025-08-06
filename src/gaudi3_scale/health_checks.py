"""Health checks and monitoring hooks for Gaudi 3 Scale.

This module provides comprehensive health checking capabilities
for all system components, with monitoring hooks for integration
with external monitoring systems.
"""

import asyncio
import json
import os
import psutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .exceptions import (
    HealthCheckError, HPUError, NetworkError, ServiceUnavailableError,
    ResourceError, Gaudi3ScaleError, ErrorCode
)
from .logging_utils import get_logger
from .retry_utils import execute_with_retry, RetryStrategy

logger = get_logger('health_checks')


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of components that can be health checked."""
    HPU_DEVICE = "hpu_device"
    HPU_DRIVER = "hpu_driver"
    NETWORK = "network"
    STORAGE = "storage"
    SERVICE = "service"
    DATABASE = "database"
    TRAINING_PROCESS = "training_process"
    MONITORING = "monitoring"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    check_duration: float = 0.0
    error: Optional[Exception] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'component_name': self.component_name,
            'component_type': self.component_type.value,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'check_duration': self.check_duration,
            'error': str(self.error) if self.error else None
        }
    
    def is_healthy(self) -> bool:
        """Check if the component is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    def is_critical(self) -> bool:
        """Check if the component has critical issues."""
        return self.status == HealthStatus.CRITICAL


class HealthCheck(ABC):
    """Abstract base class for health checks."""
    
    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        timeout: float = 30.0,
        critical_threshold: Optional[float] = None,
        warning_threshold: Optional[float] = None
    ):
        """Initialize health check.
        
        Args:
            name: Name of the component being checked
            component_type: Type of component
            timeout: Timeout for the health check
            critical_threshold: Threshold for critical status
            warning_threshold: Threshold for warning status
        """
        self.name = name
        self.component_type = component_type
        self.timeout = timeout
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
    
    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform the health check.
        
        Returns:
            HealthCheckResult containing the check outcome
        """
        pass
    
    def _create_result(
        self,
        status: HealthStatus,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        duration: float = 0.0
    ) -> HealthCheckResult:
        """Create a health check result.
        
        Args:
            status: Health status
            message: Status message
            details: Additional details
            error: Exception if check failed
            duration: Check duration in seconds
            
        Returns:
            HealthCheckResult instance
        """
        return HealthCheckResult(
            component_name=self.name,
            component_type=self.component_type,
            status=status,
            message=message,
            details=details or {},
            error=error,
            check_duration=duration
        )


class HPUHealthCheck(HealthCheck):
    """Health check for HPU devices."""
    
    def __init__(
        self,
        device_id: int = 0,
        check_memory: bool = True,
        check_temperature: bool = True,
        check_utilization: bool = True,
        **kwargs
    ):
        """Initialize HPU health check.
        
        Args:
            device_id: HPU device ID to check
            check_memory: Whether to check memory usage
            check_temperature: Whether to check temperature
            check_utilization: Whether to check utilization
            **kwargs: Additional HealthCheck parameters
        """
        super().__init__(
            name=f"hpu_device_{device_id}",
            component_type=ComponentType.HPU_DEVICE,
            **kwargs
        )
        self.device_id = device_id
        self.check_memory = check_memory
        self.check_temperature = check_temperature
        self.check_utilization = check_utilization
    
    def check(self) -> HealthCheckResult:
        """Check HPU device health."""
        start_time = time.time()
        
        try:
            # Try to import Habana frameworks
            try:
                import habana_frameworks.torch as htorch
            except ImportError:
                return self._create_result(
                    HealthStatus.CRITICAL,
                    "Habana frameworks not available",
                    duration=time.time() - start_time
                )
            
            # Check if HPU is available
            if not htorch.hpu.is_available():
                return self._create_result(
                    HealthStatus.CRITICAL,
                    "HPU devices not available",
                    duration=time.time() - start_time
                )
            
            # Check device count
            device_count = htorch.hpu.device_count()
            if self.device_id >= device_count:
                return self._create_result(
                    HealthStatus.CRITICAL,
                    f"HPU device {self.device_id} not found (available: 0-{device_count-1})",
                    duration=time.time() - start_time
                )
            
            details = {
                'device_id': self.device_id,
                'total_devices': device_count
            }
            
            # Try to get device properties
            try:
                device_name = htorch.hpu.get_device_name(self.device_id)
                details['device_name'] = device_name
            except Exception:
                details['device_name'] = f"HPU {self.device_id}"
            
            # Memory check
            if self.check_memory:
                try:
                    memory_allocated = htorch.hpu.memory_allocated(self.device_id)
                    memory_reserved = htorch.hpu.memory_reserved(self.device_id)
                    
                    details.update({
                        'memory_allocated_bytes': memory_allocated,
                        'memory_reserved_bytes': memory_reserved,
                        'memory_allocated_gb': memory_allocated / (1024**3),
                        'memory_reserved_gb': memory_reserved / (1024**3)
                    })
                except Exception as e:
                    logger.warning(f"Failed to get HPU memory info: {e}")
            
            # Basic device functionality test
            try:
                # Try to set current device
                current_device = htorch.hpu.current_device()
                htorch.hpu.set_device(self.device_id)
                htorch.hpu.set_device(current_device)  # Restore
                details['device_accessible'] = True
            except Exception as e:
                return self._create_result(
                    HealthStatus.CRITICAL,
                    f"HPU device {self.device_id} is not accessible: {e}",
                    details=details,
                    duration=time.time() - start_time
                )
            
            return self._create_result(
                HealthStatus.HEALTHY,
                f"HPU device {self.device_id} is healthy",
                details=details,
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return self._create_result(
                HealthStatus.CRITICAL,
                f"HPU health check failed: {str(e)}",
                error=e,
                duration=time.time() - start_time
            )


class SystemHealthCheck(HealthCheck):
    """Health check for system resources."""
    
    def __init__(
        self,
        check_cpu: bool = True,
        check_memory: bool = True,
        check_disk: bool = True,
        check_network: bool = True,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0,
        **kwargs
    ):
        """Initialize system health check.
        
        Args:
            check_cpu: Whether to check CPU usage
            check_memory: Whether to check memory usage
            check_disk: Whether to check disk usage
            check_network: Whether to check network connectivity
            cpu_threshold: CPU usage threshold for warnings (%)
            memory_threshold: Memory usage threshold for warnings (%)
            disk_threshold: Disk usage threshold for warnings (%)
            **kwargs: Additional HealthCheck parameters
        """
        super().__init__(
            name="system_resources",
            component_type=ComponentType.SYSTEM,
            **kwargs
        )
        self.check_cpu = check_cpu
        self.check_memory = check_memory
        self.check_disk = check_disk
        self.check_network = check_network
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    def check(self) -> HealthCheckResult:
        """Check system health."""
        start_time = time.time()
        
        try:
            details = {}
            status = HealthStatus.HEALTHY
            messages = []
            
            # CPU check
            if self.check_cpu:
                cpu_percent = psutil.cpu_percent(interval=1)
                details['cpu_usage_percent'] = cpu_percent
                details['cpu_count'] = psutil.cpu_count()
                
                if cpu_percent > self.cpu_threshold:
                    status = HealthStatus.WARNING
                    messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Memory check
            if self.check_memory:
                memory = psutil.virtual_memory()
                details.update({
                    'memory_usage_percent': memory.percent,
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_used_gb': memory.used / (1024**3)
                })
                
                if memory.percent > self.memory_threshold:
                    if memory.percent > 95:
                        status = HealthStatus.CRITICAL
                    elif status == HealthStatus.HEALTHY:
                        status = HealthStatus.WARNING
                    messages.append(f"High memory usage: {memory.percent:.1f}%")
            
            # Disk check
            if self.check_disk:
                disk_usage = psutil.disk_usage('/')
                disk_percent = (disk_usage.used / disk_usage.total) * 100
                details.update({
                    'disk_usage_percent': disk_percent,
                    'disk_total_gb': disk_usage.total / (1024**3),
                    'disk_free_gb': disk_usage.free / (1024**3),
                    'disk_used_gb': disk_usage.used / (1024**3)
                })
                
                if disk_percent > self.disk_threshold:
                    if disk_percent > 95:
                        status = HealthStatus.CRITICAL
                    elif status == HealthStatus.HEALTHY:
                        status = HealthStatus.WARNING
                    messages.append(f"High disk usage: {disk_percent:.1f}%")
            
            # Network check (basic connectivity)
            if self.check_network:
                try:
                    network_io = psutil.net_io_counters()
                    details.update({
                        'network_bytes_sent': network_io.bytes_sent,
                        'network_bytes_recv': network_io.bytes_recv,
                        'network_packets_sent': network_io.packets_sent,
                        'network_packets_recv': network_io.packets_recv
                    })
                except Exception as e:
                    messages.append(f"Network check failed: {e}")
                    if status == HealthStatus.HEALTHY:
                        status = HealthStatus.WARNING
            
            # Load average (on Unix systems)
            try:
                if hasattr(os, 'getloadavg'):
                    load_avg = os.getloadavg()
                    details['load_average'] = {
                        '1min': load_avg[0],
                        '5min': load_avg[1],
                        '15min': load_avg[2]
                    }
                    
                    # Check if load is too high
                    cpu_count = psutil.cpu_count()
                    if load_avg[0] > cpu_count * 2:
                        if status == HealthStatus.HEALTHY:
                            status = HealthStatus.WARNING
                        messages.append(f"High system load: {load_avg[0]:.2f}")
            except Exception:
                pass
            
            message = "System resources healthy"
            if messages:
                message = "; ".join(messages)
            
            return self._create_result(
                status,
                message,
                details=details,
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return self._create_result(
                HealthStatus.CRITICAL,
                f"System health check failed: {str(e)}",
                error=e,
                duration=time.time() - start_time
            )


class ServiceHealthCheck(HealthCheck):
    """Health check for external services."""
    
    def __init__(
        self,
        service_name: str,
        url: str,
        method: str = "GET",
        expected_status: int = 200,
        expected_content: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Initialize service health check.
        
        Args:
            service_name: Name of the service
            url: Service URL to check
            method: HTTP method to use
            expected_status: Expected HTTP status code
            expected_content: Expected content in response
            headers: HTTP headers to send
            **kwargs: Additional HealthCheck parameters
        """
        super().__init__(
            name=service_name,
            component_type=ComponentType.SERVICE,
            **kwargs
        )
        self.url = url
        self.method = method.upper()
        self.expected_status = expected_status
        self.expected_content = expected_content
        self.headers = headers or {}
    
    def check(self) -> HealthCheckResult:
        """Check service health via HTTP request."""
        if not REQUESTS_AVAILABLE:
            return self._create_result(
                HealthStatus.CRITICAL,
                "requests library not available for service health check"
            )
        
        start_time = time.time()
        
        try:
            # Make HTTP request
            response = requests.request(
                method=self.method,
                url=self.url,
                headers=self.headers,
                timeout=self.timeout
            )
            
            details = {
                'url': self.url,
                'method': self.method,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'content_length': len(response.content) if response.content else 0
            }
            
            # Check status code
            if response.status_code != self.expected_status:
                return self._create_result(
                    HealthStatus.CRITICAL,
                    f"Unexpected status code: {response.status_code} (expected {self.expected_status})",
                    details=details,
                    duration=time.time() - start_time
                )
            
            # Check content if specified
            if self.expected_content and self.expected_content not in response.text:
                return self._create_result(
                    HealthStatus.CRITICAL,
                    f"Expected content not found in response",
                    details=details,
                    duration=time.time() - start_time
                )
            
            return self._create_result(
                HealthStatus.HEALTHY,
                f"Service {self.name} is responding normally",
                details=details,
                duration=time.time() - start_time
            )
            
        except requests.exceptions.Timeout:
            return self._create_result(
                HealthStatus.CRITICAL,
                f"Service {self.name} request timed out",
                duration=time.time() - start_time
            )
        except requests.exceptions.ConnectionError as e:
            return self._create_result(
                HealthStatus.CRITICAL,
                f"Cannot connect to service {self.name}: {str(e)}",
                error=e,
                duration=time.time() - start_time
            )
        except Exception as e:
            return self._create_result(
                HealthStatus.CRITICAL,
                f"Service health check failed: {str(e)}",
                error=e,
                duration=time.time() - start_time
            )


class FileSystemHealthCheck(HealthCheck):
    """Health check for file system paths and storage."""
    
    def __init__(
        self,
        path: Union[str, Path],
        check_writable: bool = False,
        check_readable: bool = True,
        min_free_space_gb: Optional[float] = None,
        **kwargs
    ):
        """Initialize file system health check.
        
        Args:
            path: Path to check
            check_writable: Whether to check if path is writable
            check_readable: Whether to check if path is readable
            min_free_space_gb: Minimum required free space in GB
            **kwargs: Additional HealthCheck parameters
        """
        super().__init__(
            name=f"filesystem_{Path(path).name}",
            component_type=ComponentType.STORAGE,
            **kwargs
        )
        self.path = Path(path)
        self.check_writable = check_writable
        self.check_readable = check_readable
        self.min_free_space_gb = min_free_space_gb
    
    def check(self) -> HealthCheckResult:
        """Check file system health."""
        start_time = time.time()
        
        try:
            details = {'path': str(self.path)}
            
            # Check if path exists
            if not self.path.exists():
                return self._create_result(
                    HealthStatus.CRITICAL,
                    f"Path does not exist: {self.path}",
                    details=details,
                    duration=time.time() - start_time
                )
            
            # Check permissions
            if self.check_readable and not os.access(self.path, os.R_OK):
                return self._create_result(
                    HealthStatus.CRITICAL,
                    f"Path is not readable: {self.path}",
                    details=details,
                    duration=time.time() - start_time
                )
            
            if self.check_writable and not os.access(self.path, os.W_OK):
                return self._create_result(
                    HealthStatus.CRITICAL,
                    f"Path is not writable: {self.path}",
                    details=details,
                    duration=time.time() - start_time
                )
            
            # Check disk space
            if self.path.is_dir():
                try:
                    disk_usage = psutil.disk_usage(str(self.path))
                    free_space_gb = disk_usage.free / (1024**3)
                    total_space_gb = disk_usage.total / (1024**3)
                    used_space_gb = disk_usage.used / (1024**3)
                    
                    details.update({
                        'free_space_gb': free_space_gb,
                        'total_space_gb': total_space_gb,
                        'used_space_gb': used_space_gb,
                        'usage_percent': (used_space_gb / total_space_gb) * 100
                    })
                    
                    if self.min_free_space_gb and free_space_gb < self.min_free_space_gb:
                        return self._create_result(
                            HealthStatus.CRITICAL,
                            f"Insufficient free space: {free_space_gb:.1f}GB (required: {self.min_free_space_gb}GB)",
                            details=details,
                            duration=time.time() - start_time
                        )
                
                except Exception as e:
                    logger.warning(f"Failed to get disk usage for {self.path}: {e}")
            
            return self._create_result(
                HealthStatus.HEALTHY,
                f"File system path {self.path} is healthy",
                details=details,
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return self._create_result(
                HealthStatus.CRITICAL,
                f"File system health check failed: {str(e)}",
                error=e,
                duration=time.time() - start_time
            )


class TrainingProcessHealthCheck(HealthCheck):
    """Health check for training processes."""
    
    def __init__(
        self,
        process_name: str,
        pid: Optional[int] = None,
        check_memory: bool = True,
        check_cpu: bool = True,
        max_memory_gb: Optional[float] = None,
        max_cpu_percent: Optional[float] = None,
        **kwargs
    ):
        """Initialize training process health check.
        
        Args:
            process_name: Name of the training process
            pid: Process ID (if known)
            check_memory: Whether to check memory usage
            check_cpu: Whether to check CPU usage
            max_memory_gb: Maximum allowed memory usage in GB
            max_cpu_percent: Maximum allowed CPU usage percentage
            **kwargs: Additional HealthCheck parameters
        """
        super().__init__(
            name=f"training_process_{process_name}",
            component_type=ComponentType.TRAINING_PROCESS,
            **kwargs
        )
        self.process_name = process_name
        self.pid = pid
        self.check_memory = check_memory
        self.check_cpu = check_cpu
        self.max_memory_gb = max_memory_gb
        self.max_cpu_percent = max_cpu_percent
    
    def check(self) -> HealthCheckResult:
        """Check training process health."""
        start_time = time.time()
        
        try:
            # Find the process
            process = None
            if self.pid:
                try:
                    process = psutil.Process(self.pid)
                    if not process.is_running():
                        process = None
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    process = None
            
            if not process:
                # Search by name
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if (self.process_name in proc.info['name'] or 
                            any(self.process_name in arg for arg in (proc.info['cmdline'] or []))):
                            process = proc
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            
            if not process:
                return self._create_result(
                    HealthStatus.CRITICAL,
                    f"Training process '{self.process_name}' not found",
                    duration=time.time() - start_time
                )
            
            details = {
                'pid': process.pid,
                'name': process.name(),
                'status': process.status(),
                'create_time': process.create_time(),
                'running_time': time.time() - process.create_time()
            }
            
            # Check if process is responsive
            try:
                cpu_percent = process.cpu_percent(interval=1)
                details['cpu_percent'] = cpu_percent
                
                if self.max_cpu_percent and cpu_percent > self.max_cpu_percent:
                    return self._create_result(
                        HealthStatus.WARNING,
                        f"High CPU usage: {cpu_percent:.1f}%",
                        details=details,
                        duration=time.time() - start_time
                    )
            except Exception as e:
                logger.warning(f"Failed to get CPU usage for process {process.pid}: {e}")
            
            # Check memory usage
            if self.check_memory:
                try:
                    memory_info = process.memory_info()
                    memory_gb = memory_info.rss / (1024**3)
                    
                    details.update({
                        'memory_rss_gb': memory_gb,
                        'memory_vms_gb': memory_info.vms / (1024**3),
                        'memory_percent': process.memory_percent()
                    })
                    
                    if self.max_memory_gb and memory_gb > self.max_memory_gb:
                        return self._create_result(
                            HealthStatus.WARNING,
                            f"High memory usage: {memory_gb:.1f}GB",
                            details=details,
                            duration=time.time() - start_time
                        )
                except Exception as e:
                    logger.warning(f"Failed to get memory info for process {process.pid}: {e}")
            
            # Check for zombie or stopped processes
            if process.status() in [psutil.STATUS_ZOMBIE, psutil.STATUS_STOPPED]:
                return self._create_result(
                    HealthStatus.CRITICAL,
                    f"Process is in {process.status()} state",
                    details=details,
                    duration=time.time() - start_time
                )
            
            return self._create_result(
                HealthStatus.HEALTHY,
                f"Training process '{self.process_name}' is running normally",
                details=details,
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return self._create_result(
                HealthStatus.CRITICAL,
                f"Training process health check failed: {str(e)}",
                error=e,
                duration=time.time() - start_time
            )


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(
        self,
        checks: Optional[List[HealthCheck]] = None,
        check_interval: float = 60.0,
        enable_background_monitoring: bool = True,
        max_concurrent_checks: int = 10,
        history_retention_hours: int = 24
    ):
        """Initialize health monitor.
        
        Args:
            checks: List of health checks to run
            check_interval: Interval between checks in seconds
            enable_background_monitoring: Whether to run checks in background
            max_concurrent_checks: Maximum concurrent health checks
            history_retention_hours: How long to keep check history
        """
        self.checks = checks or []
        self.check_interval = check_interval
        self.enable_background_monitoring = enable_background_monitoring
        self.max_concurrent_checks = max_concurrent_checks
        self.history_retention_hours = history_retention_hours
        
        # Results storage
        self.latest_results: Dict[str, HealthCheckResult] = {}
        self.history: List[HealthCheckResult] = []
        self._history_lock = threading.Lock()
        
        # Background monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Health check hooks
        self.on_status_change_callbacks: List[Callable[[HealthCheckResult, HealthCheckResult], None]] = []
        self.on_critical_callbacks: List[Callable[[HealthCheckResult], None]] = []
        
        if self.enable_background_monitoring:
            self.start_monitoring()
    
    def add_check(self, check: HealthCheck) -> None:
        """Add a health check.
        
        Args:
            check: Health check to add
        """
        self.checks.append(check)
        logger.info(f"Added health check: {check.name}")
    
    def remove_check(self, check_name: str) -> None:
        """Remove a health check by name.
        
        Args:
            check_name: Name of the check to remove
        """
        self.checks = [c for c in self.checks if c.name != check_name]
        if check_name in self.latest_results:
            del self.latest_results[check_name]
        logger.info(f"Removed health check: {check_name}")
    
    def run_check(self, check: HealthCheck) -> HealthCheckResult:
        """Run a single health check.
        
        Args:
            check: Health check to run
            
        Returns:
            HealthCheckResult
        """
        logger.debug(f"Running health check: {check.name}")
        
        try:
            result = check.check()
            self._store_result(result)
            return result
        except Exception as e:
            logger.exception(f"Health check {check.name} raised an exception")
            result = HealthCheckResult(
                component_name=check.name,
                component_type=check.component_type,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed with exception: {str(e)}",
                error=e
            )
            self._store_result(result)
            return result
    
    def run_all_checks(self, parallel: bool = True) -> Dict[str, HealthCheckResult]:
        """Run all health checks.
        
        Args:
            parallel: Whether to run checks in parallel
            
        Returns:
            Dictionary of check results by check name
        """
        if not self.checks:
            return {}
        
        if parallel:
            return self._run_checks_parallel()
        else:
            return self._run_checks_sequential()
    
    def _run_checks_parallel(self) -> Dict[str, HealthCheckResult]:
        """Run health checks in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_checks) as executor:
            future_to_check = {
                executor.submit(self.run_check, check): check
                for check in self.checks
            }
            
            for future in as_completed(future_to_check):
                check = future_to_check[future]
                try:
                    result = future.result()
                    results[check.name] = result
                except Exception as e:
                    logger.exception(f"Error running health check {check.name}")
                    results[check.name] = HealthCheckResult(
                        component_name=check.name,
                        component_type=check.component_type,
                        status=HealthStatus.CRITICAL,
                        message=f"Health check execution failed: {str(e)}",
                        error=e
                    )
        
        return results
    
    def _run_checks_sequential(self) -> Dict[str, HealthCheckResult]:
        """Run health checks sequentially."""
        results = {}
        for check in self.checks:
            results[check.name] = self.run_check(check)
        return results
    
    def _store_result(self, result: HealthCheckResult) -> None:
        """Store health check result."""
        with self._history_lock:
            # Update latest results
            previous_result = self.latest_results.get(result.component_name)
            self.latest_results[result.component_name] = result
            
            # Add to history
            self.history.append(result)
            
            # Clean old history
            cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
            self.history = [r for r in self.history if r.timestamp > cutoff_time]
        
        # Call status change callbacks
        if previous_result and previous_result.status != result.status:
            for callback in self.on_status_change_callbacks:
                try:
                    callback(previous_result, result)
                except Exception as e:
                    logger.exception(f"Error in status change callback: {e}")
        
        # Call critical callbacks
        if result.status == HealthStatus.CRITICAL:
            for callback in self.on_critical_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.exception(f"Error in critical status callback: {e}")
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status.
        
        Returns:
            Overall health status based on all checks
        """
        if not self.latest_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.latest_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report.
        
        Returns:
            Health report dictionary
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': self.get_overall_status().value,
            'total_checks': len(self.checks),
            'check_results': {
                name: result.to_dict()
                for name, result in self.latest_results.items()
            },
            'status_summary': self._get_status_summary(),
            'unhealthy_components': [
                result.component_name
                for result in self.latest_results.values()
                if not result.is_healthy()
            ]
        }
    
    def _get_status_summary(self) -> Dict[str, int]:
        """Get summary of health statuses."""
        summary = {status.value: 0 for status in HealthStatus}
        for result in self.latest_results.values():
            summary[result.status.value] += 1
        return summary
    
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Started background health monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped background health monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                self.run_all_checks(parallel=True)
            except Exception as e:
                logger.exception(f"Error in health monitoring loop: {e}")
            
            self._stop_monitoring.wait(self.check_interval)
    
    def add_status_change_callback(self, callback: Callable[[HealthCheckResult, HealthCheckResult], None]) -> None:
        """Add callback for status changes.
        
        Args:
            callback: Function to call when status changes
        """
        self.on_status_change_callbacks.append(callback)
    
    def add_critical_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """Add callback for critical status.
        
        Args:
            callback: Function to call when status becomes critical
        """
        self.on_critical_callbacks.append(callback)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Predefined health check configurations

def create_default_hpu_health_checks(device_count: int = 8) -> List[HealthCheck]:
    """Create default HPU health checks.
    
    Args:
        device_count: Number of HPU devices to check
        
    Returns:
        List of HPU health checks
    """
    checks = []
    for device_id in range(device_count):
        check = HPUHealthCheck(
            device_id=device_id,
            timeout=10.0,
            check_memory=True,
            check_temperature=True
        )
        checks.append(check)
    return checks


def create_system_health_checks(
    cpu_threshold: float = 85.0,
    memory_threshold: float = 90.0,
    disk_threshold: float = 85.0
) -> List[HealthCheck]:
    """Create default system health checks.
    
    Args:
        cpu_threshold: CPU usage warning threshold
        memory_threshold: Memory usage warning threshold  
        disk_threshold: Disk usage warning threshold
        
    Returns:
        List of system health checks
    """
    return [
        SystemHealthCheck(
            cpu_threshold=cpu_threshold,
            memory_threshold=memory_threshold,
            disk_threshold=disk_threshold,
            timeout=30.0
        )
    ]


def create_storage_health_checks(paths: List[str]) -> List[HealthCheck]:
    """Create health checks for storage paths.
    
    Args:
        paths: List of paths to check
        
    Returns:
        List of file system health checks
    """
    checks = []
    for path in paths:
        check = FileSystemHealthCheck(
            path=path,
            check_writable=True,
            check_readable=True,
            min_free_space_gb=10.0,
            timeout=15.0
        )
        checks.append(check)
    return checks