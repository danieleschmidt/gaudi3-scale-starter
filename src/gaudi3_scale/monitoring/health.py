"""Health checking and monitoring for Gaudi 3 infrastructure."""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float


class HealthCheck:
    """Base health check class."""
    
    def __init__(self, name: str, timeout_seconds: float = 30.0):
        """Initialize health check.
        
        Args:
            name: Name of the health check
            timeout_seconds: Timeout for the check
        """
        self.name = name
        self.timeout_seconds = timeout_seconds
    
    async def check(self) -> HealthCheckResult:
        """Perform health check.
        
        Returns:
            Health check result
        """
        start_time = time.time()
        
        try:
            result = await self._perform_check()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=result.get("status", HealthStatus.UNKNOWN),
                message=result.get("message", ""),
                details=result.get("details", {}),
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Health check {self.name} failed: {e}")
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Perform the actual health check.
        
        Returns:
            Health check result data
        """
        raise NotImplementedError("Subclasses must implement _perform_check")


class DatabaseHealthCheck(HealthCheck):
    """Database connectivity health check."""
    
    def __init__(self, database_connection, timeout_seconds: float = 10.0):
        """Initialize database health check.
        
        Args:
            database_connection: Database connection instance
            timeout_seconds: Timeout for the check
        """
        super().__init__("database", timeout_seconds)
        self.database_connection = database_connection
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check database connectivity.
        
        Returns:
            Database health status
        """
        try:
            # Test database connection
            if self.database_connection.test_connection():
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "Database connection successful",
                    "details": {
                        "connection_pool_size": getattr(self.database_connection, 'pool_size', 'unknown'),
                        "active_connections": getattr(self.database_connection, 'active_connections', 'unknown')
                    }
                }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": "Database connection failed",
                    "details": {}
                }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Database check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class HPUHealthCheck(HealthCheck):
    """HPU device health check."""
    
    def __init__(self, timeout_seconds: float = 15.0):
        """Initialize HPU health check.
        
        Args:
            timeout_seconds: Timeout for the check
        """
        super().__init__("hpu", timeout_seconds)
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check HPU device status.
        
        Returns:
            HPU health status
        """
        try:
            import habana_frameworks.torch as htorch
            
            # Check if HPUs are available
            if not htorch.hpu.is_available():
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": "No HPU devices available",
                    "details": {}
                }
            
            device_count = htorch.hpu.device_count()
            device_details = {}
            
            for i in range(device_count):
                try:
                    # Get device properties
                    memory_allocated = htorch.hpu.memory_allocated(i)
                    memory_reserved = htorch.hpu.memory_reserved(i)
                    
                    device_details[f"hpu_{i}"] = {
                        "memory_allocated_mb": memory_allocated / (1024 * 1024),
                        "memory_reserved_mb": memory_reserved / (1024 * 1024),
                        "status": "available"
                    }
                except Exception as e:
                    device_details[f"hpu_{i}"] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": f"{device_count} HPU device(s) available",
                "details": {
                    "device_count": device_count,
                    "devices": device_details
                }
            }
        
        except ImportError:
            return {
                "status": HealthStatus.DEGRADED,
                "message": "Habana frameworks not available",
                "details": {"error": "habana_frameworks not installed"}
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"HPU check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class SystemResourcesHealthCheck(HealthCheck):
    """System resources health check."""
    
    def __init__(self, cpu_threshold: float = 90.0, memory_threshold: float = 85.0, 
                 disk_threshold: float = 90.0, timeout_seconds: float = 5.0):
        """Initialize system resources health check.
        
        Args:
            cpu_threshold: CPU usage threshold for degraded status
            memory_threshold: Memory usage threshold for degraded status
            disk_threshold: Disk usage threshold for degraded status
            timeout_seconds: Timeout for the check
        """
        super().__init__("system_resources", timeout_seconds)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check system resource usage.
        
        Returns:
            System resources health status
        """
        if not PSUTIL_AVAILABLE:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "psutil not available for system monitoring",
                "details": {}
            }
        
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            issues = []
            status = HealthStatus.HEALTHY
            
            if cpu_usage > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
                status = HealthStatus.DEGRADED
            
            if memory.percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                status = HealthStatus.DEGRADED
            
            if disk.percent > self.disk_threshold:
                issues.append(f"High disk usage: {disk.percent:.1f}%")
                status = HealthStatus.DEGRADED
            
            message = "System resources normal"
            if issues:
                message = "; ".join(issues)
                if len(issues) > 1:
                    status = HealthStatus.UNHEALTHY
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "cpu_usage_percent": cpu_usage,
                    "memory_usage_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_usage_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else []
                }
            }
        
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"System resources check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class TrainingJobHealthCheck(HealthCheck):
    """Training job health check."""
    
    def __init__(self, training_service, timeout_seconds: float = 10.0):
        """Initialize training job health check.
        
        Args:
            training_service: Training service instance
            timeout_seconds: Timeout for the check
        """
        super().__init__("training_jobs", timeout_seconds)
        self.training_service = training_service
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check training job status.
        
        Returns:
            Training jobs health status
        """
        try:
            # Get active training jobs
            active_jobs = getattr(self.training_service, 'get_active_jobs', lambda: [])()
            failed_jobs = getattr(self.training_service, 'get_failed_jobs', lambda: [])()
            
            total_jobs = len(active_jobs)
            failed_count = len(failed_jobs)
            
            # Determine status
            if failed_count == 0:
                status = HealthStatus.HEALTHY
                message = f"{total_jobs} training job(s) running successfully"
            elif failed_count < total_jobs / 2:
                status = HealthStatus.DEGRADED
                message = f"{failed_count} of {total_jobs} training jobs failed"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"High failure rate: {failed_count} of {total_jobs} training jobs failed"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "active_jobs": total_jobs,
                    "failed_jobs": failed_count,
                    "success_rate": (total_jobs - failed_count) / max(total_jobs, 1) * 100
                }
            }
        
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Training jobs check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class ExternalServiceHealthCheck(HealthCheck):
    """External service connectivity health check."""
    
    def __init__(self, service_name: str, check_function: Callable, timeout_seconds: float = 10.0):
        """Initialize external service health check.
        
        Args:
            service_name: Name of the external service
            check_function: Function to check service health
            timeout_seconds: Timeout for the check
        """
        super().__init__(f"external_service_{service_name}", timeout_seconds)
        self.service_name = service_name
        self.check_function = check_function
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check external service connectivity.
        
        Returns:
            External service health status
        """
        try:
            result = await self.check_function()
            
            if result.get("available", False):
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": f"{self.service_name} service available",
                    "details": result.get("details", {})
                }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"{self.service_name} service unavailable",
                    "details": result.get("details", {})
                }
        
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"{self.service_name} service check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class HealthChecker:
    """Main health checker that orchestrates all health checks."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: List[HealthCheck] = []
        self.last_results: Dict[str, HealthCheckResult] = {}
    
    def add_check(self, health_check: HealthCheck) -> None:
        """Add a health check.
        
        Args:
            health_check: Health check to add
        """
        self.checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_check(self, name: str) -> bool:
        """Remove a health check by name.
        
        Args:
            name: Name of the health check to remove
            
        Returns:
            True if check was removed
        """
        original_count = len(self.checks)
        self.checks = [check for check in self.checks if check.name != name]
        
        if name in self.last_results:
            del self.last_results[name]
        
        return len(self.checks) < original_count
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks.
        
        Returns:
            Overall health status and individual check results
        """
        results = []
        overall_status = HealthStatus.HEALTHY
        
        for check in self.checks:
            try:
                result = await check.check()
                results.append(result)
                self.last_results[check.name] = result
                
                # Update overall status
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                
            except Exception as e:
                logger.error(f"Health check {check.name} raised exception: {e}")
                error_result = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed with exception: {str(e)}",
                    details={"error": str(e)},
                    timestamp=datetime.now(),
                    duration_ms=0.0
                )
                results.append(error_result)
                self.last_results[check.name] = error_result
                overall_status = HealthStatus.UNHEALTHY
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": [
                {
                    "name": result.name,
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "duration_ms": result.duration_ms
                }
                for result in results
            ],
            "summary": {
                "total_checks": len(results),
                "healthy": len([r for r in results if r.status == HealthStatus.HEALTHY]),
                "degraded": len([r for r in results if r.status == HealthStatus.DEGRADED]),
                "unhealthy": len([r for r in results if r.status == HealthStatus.UNHEALTHY])
            }
        }
    
    async def check_specific(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check.
        
        Args:
            name: Name of the health check to run
            
        Returns:
            Health check result or None if not found
        """
        for check in self.checks:
            if check.name == name:
                result = await check.check()
                self.last_results[name] = result
                return result
        
        return None
    
    def get_last_results(self) -> Dict[str, Dict[str, Any]]:
        """Get last health check results.
        
        Returns:
            Dictionary of last health check results
        """
        return {
            name: {
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "timestamp": result.timestamp.isoformat(),
                "duration_ms": result.duration_ms
            }
            for name, result in self.last_results.items()
        }
    
    def is_healthy(self) -> bool:
        """Check if system is overall healthy.
        
        Returns:
            True if all checks are healthy or degraded
        """
        if not self.last_results:
            return False
        
        return all(
            result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            for result in self.last_results.values()
        )
    
    def setup_default_checks(self, database_connection=None, training_service=None) -> None:
        """Setup default health checks.
        
        Args:
            database_connection: Database connection instance
            training_service: Training service instance
        """
        # Always add system resources check
        self.add_check(SystemResourcesHealthCheck())
        
        # Add HPU check
        self.add_check(HPUHealthCheck())
        
        # Add database check if connection provided
        if database_connection:
            self.add_check(DatabaseHealthCheck(database_connection))
        
        # Add training jobs check if service provided
        if training_service:
            self.add_check(TrainingJobHealthCheck(training_service))
        
        logger.info(f"Setup {len(self.checks)} default health checks")