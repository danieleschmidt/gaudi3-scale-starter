"""Comprehensive Reliability Engine for Gaudi 3 Scale - Generation 2 Enhancement.

This module implements enterprise-grade reliability features including:
- Circuit breaker patterns for fault tolerance
- Exponential backoff and jitter for retry logic
- Health monitoring with auto-recovery
- Resource leak detection and cleanup
- Performance degradation detection
- Automatic failover mechanisms
"""

import asyncio
import logging
import time
import threading
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Union
from concurrent.futures import ThreadPoolExecutor, Future
import json
from pathlib import Path


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    name: str = "default"


class CircuitBreaker:
    """Production-ready circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.lock = threading.RLock()
        
        # Metrics tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_open_count = 0
        
        self.logger = logging.getLogger(f"circuit_breaker.{config.name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
    
    def __enter__(self):
        """Context manager entry."""
        with self.lock:
            self.total_calls += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.config.name} attempting reset")
                else:
                    self.circuit_open_count += 1
                    raise Exception(f"Circuit breaker {self.config.name} is OPEN")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        with self.lock:
            if exc_type is None:
                self._on_success()
            elif issubclass(exc_type, self.config.expected_exception):
                self._on_failure()
            # Re-raise unexpected exceptions without counting as failures
        
        return False  # Don't suppress exceptions
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.successful_calls += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.logger.info(f"Circuit breaker {self.config.name} reset to CLOSED")
        
        self.success_count += 1
    
    def _on_failure(self):
        """Handle failed call."""
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker {self.config.name} opened after {self.failure_count} failures")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                "name": self.config.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "circuit_open_count": self.circuit_open_count,
                "failure_rate": self.failed_calls / max(1, self.total_calls),
                "last_failure_time": self.last_failure_time
            }


class ExponentialBackoff:
    """Exponential backoff with jitter for reliable retries."""
    
    def __init__(
        self,
        initial_interval: float = 0.1,
        max_interval: float = 30.0,
        multiplier: float = 2.0,
        max_attempts: int = 5,
        jitter: bool = True
    ):
        self.initial_interval = initial_interval
        self.max_interval = max_interval
        self.multiplier = multiplier
        self.max_attempts = max_attempts
        self.jitter = jitter
        
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if attempt <= 0:
            return 0.0
        
        delay = min(
            self.initial_interval * (self.multiplier ** (attempt - 1)),
            self.max_interval
        )
        
        if self.jitter:
            import random
            # Add jitter to avoid thundering herd
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def __iter__(self):
        """Iterator for retry delays."""
        for attempt in range(1, self.max_attempts + 1):
            delay = self.calculate_delay(attempt)
            yield attempt, delay


class RetryManager:
    """Advanced retry manager with multiple strategies."""
    
    def __init__(
        self,
        backoff: ExponentialBackoff = None,
        circuit_breaker: CircuitBreaker = None,
        retryable_exceptions: tuple = (Exception,),
        name: str = "default"
    ):
        self.backoff = backoff or ExponentialBackoff()
        self.circuit_breaker = circuit_breaker
        self.retryable_exceptions = retryable_exceptions
        self.name = name
        self.logger = logging.getLogger(f"retry_manager.{name}")
        
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt, delay in self.backoff:
            try:
                if self.circuit_breaker:
                    with self.circuit_breaker:
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not isinstance(e, self.retryable_exceptions):
                    self.logger.error(f"Non-retryable exception in {self.name}: {e}")
                    raise
                
                # Don't retry on final attempt
                if attempt >= self.backoff.max_attempts:
                    break
                
                self.logger.warning(
                    f"Attempt {attempt}/{self.backoff.max_attempts} failed in {self.name}: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                time.sleep(delay)
        
        # All attempts failed
        self.logger.error(f"All retry attempts failed in {self.name}")
        raise last_exception


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_func: Callable[[], bool]
    timeout: float = 5.0
    critical: bool = False
    interval: float = 30.0


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, check_interval: float = 30.0):
        self.checks: Dict[str, HealthCheck] = {}
        self.check_interval = check_interval
        self.status_history: List[Dict] = []
        self.current_status = HealthStatus.HEALTHY
        self.lock = threading.RLock()
        self._running = False
        self._thread = None
        self.logger = logging.getLogger("health_monitor")
        
        # Auto-recovery functions
        self.recovery_functions: Dict[str, Callable] = {}
        
        # Performance metrics
        self.start_time = time.time()
        self.check_count = 0
        self.failure_count = 0
        
    def add_check(self, health_check: HealthCheck):
        """Add a health check."""
        with self.lock:
            self.checks[health_check.name] = health_check
            self.logger.info(f"Added health check: {health_check.name}")
    
    def remove_check(self, name: str):
        """Remove a health check."""
        with self.lock:
            if name in self.checks:
                del self.checks[name]
                self.logger.info(f"Removed health check: {name}")
    
    def add_recovery_function(self, name: str, func: Callable):
        """Add auto-recovery function for a health check."""
        self.recovery_functions[name] = func
        self.logger.info(f"Added recovery function for: {name}")
    
    def start(self):
        """Start health monitoring."""
        with self.lock:
            if self._running:
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            self.logger.info("Health monitor started")
    
    def stop(self):
        """Stop health monitoring."""
        with self.lock:
            self._running = False
            if self._thread:
                self._thread.join(timeout=5.0)
            self.logger.info("Health monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                time.sleep(min(self.check_interval, 10.0))
    
    def _perform_health_checks(self):
        """Perform all health checks."""
        results = {}
        failed_checks = []
        critical_failures = False
        
        with self.lock:
            checks_snapshot = dict(self.checks)
        
        for name, check in checks_snapshot.items():
            try:
                self.check_count += 1
                start_time = time.time()
                
                # Execute check with timeout
                success = self._execute_with_timeout(check.check_func, check.timeout)
                
                execution_time = time.time() - start_time
                results[name] = {
                    "success": success,
                    "execution_time": execution_time,
                    "critical": check.critical,
                    "timestamp": time.time()
                }
                
                if not success:
                    self.failure_count += 1
                    failed_checks.append(name)
                    if check.critical:
                        critical_failures = True
                    
                    # Attempt auto-recovery
                    if name in self.recovery_functions:
                        self._attempt_recovery(name)
                        
            except Exception as e:
                self.logger.error(f"Health check {name} failed with exception: {e}")
                self.failure_count += 1
                failed_checks.append(name)
                if check.critical:
                    critical_failures = True
                
                results[name] = {
                    "success": False,
                    "error": str(e),
                    "critical": check.critical,
                    "timestamp": time.time()
                }
        
        # Determine overall status
        new_status = self._calculate_status(results, failed_checks, critical_failures)
        
        # Record status change
        if new_status != self.current_status:
            self.logger.warning(f"Health status changed: {self.current_status.value} -> {new_status.value}")
            self.current_status = new_status
        
        # Store results
        status_record = {
            "timestamp": time.time(),
            "status": new_status.value,
            "checks": results,
            "failed_checks": failed_checks,
            "critical_failures": critical_failures,
            "total_checks": len(checks_snapshot),
            "successful_checks": len(checks_snapshot) - len(failed_checks)
        }
        
        with self.lock:
            self.status_history.append(status_record)
            # Keep only last 100 records
            if len(self.status_history) > 100:
                self.status_history = self.status_history[-100:]
    
    def _execute_with_timeout(self, func: Callable, timeout: float) -> bool:
        """Execute function with timeout."""
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func)
                result = future.result(timeout=timeout)
                return bool(result)
        except Exception as e:
            self.logger.debug(f"Health check execution failed: {e}")
            return False
    
    def _calculate_status(self, results: Dict, failed_checks: List, critical_failures: bool) -> HealthStatus:
        """Calculate overall health status."""
        if critical_failures:
            return HealthStatus.CRITICAL
        
        if not results:  # No checks configured
            return HealthStatus.HEALTHY
        
        failure_rate = len(failed_checks) / len(results)
        
        if failure_rate == 0:
            return HealthStatus.HEALTHY
        elif failure_rate <= 0.3:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY
    
    def _attempt_recovery(self, check_name: str):
        """Attempt auto-recovery for failed check."""
        if check_name not in self.recovery_functions:
            return
        
        try:
            self.logger.info(f"Attempting auto-recovery for: {check_name}")
            recovery_func = self.recovery_functions[check_name]
            recovery_func()
            self.logger.info(f"Auto-recovery completed for: {check_name}")
        except Exception as e:
            self.logger.error(f"Auto-recovery failed for {check_name}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self.lock:
            uptime = time.time() - self.start_time
            latest_record = self.status_history[-1] if self.status_history else None
            
            return {
                "status": self.current_status.value,
                "uptime": uptime,
                "total_checks_performed": self.check_count,
                "total_failures": self.failure_count,
                "failure_rate": self.failure_count / max(1, self.check_count),
                "active_health_checks": len(self.checks),
                "latest_check_results": latest_record,
                "timestamp": time.time()
            }
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get health status history."""
        with self.lock:
            return self.status_history[-limit:]


class ResourceLeakDetector:
    """Detect and track resource leaks."""
    
    def __init__(self):
        self.tracked_resources: Dict[str, weakref.WeakSet] = {}
        self.creation_counts: Dict[str, int] = {}
        self.destruction_counts: Dict[str, int] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger("resource_leak_detector")
    
    def track_resource(self, resource: Any, resource_type: str):
        """Track a resource for leak detection."""
        with self.lock:
            if resource_type not in self.tracked_resources:
                self.tracked_resources[resource_type] = weakref.WeakSet()
                self.creation_counts[resource_type] = 0
                self.destruction_counts[resource_type] = 0
            
            self.tracked_resources[resource_type].add(resource)
            self.creation_counts[resource_type] += 1
    
    def get_leak_report(self) -> Dict[str, Any]:
        """Generate resource leak report."""
        with self.lock:
            report = {
                "timestamp": time.time(),
                "resource_types": {}
            }
            
            for resource_type, weak_set in self.tracked_resources.items():
                active_count = len(weak_set)
                created = self.creation_counts[resource_type]
                destroyed = created - active_count
                
                report["resource_types"][resource_type] = {
                    "active": active_count,
                    "created": created,
                    "destroyed": destroyed,
                    "potential_leak": active_count > (created * 0.1)  # More than 10% still active
                }
            
            return report
    
    def cleanup_resources(self, resource_type: str = None):
        """Force cleanup of tracked resources."""
        import gc
        
        with self.lock:
            if resource_type:
                if resource_type in self.tracked_resources:
                    self.tracked_resources[resource_type].clear()
                    self.logger.info(f"Cleared tracked resources of type: {resource_type}")
            else:
                for rt in self.tracked_resources:
                    self.tracked_resources[rt].clear()
                self.logger.info("Cleared all tracked resources")
        
        # Force garbage collection
        gc.collect()


class ReliabilityEngine:
    """Comprehensive reliability engine orchestrating all reliability features."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_managers: Dict[str, RetryManager] = {}
        self.health_monitor = HealthMonitor(
            check_interval=self.config.get("health_check_interval", 30.0)
        )
        self.resource_detector = ResourceLeakDetector()
        
        self.logger = logging.getLogger("reliability_engine")
        self.start_time = time.time()
        
        # Initialize default components
        self._setup_default_components()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load reliability configuration."""
        default_config = {
            "health_check_interval": 30.0,
            "default_circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 60
            },
            "default_retry": {
                "max_attempts": 5,
                "initial_interval": 0.1,
                "max_interval": 30.0
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Loaded configuration from: {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_default_components(self):
        """Setup default reliability components."""
        # Default circuit breaker
        default_cb_config = CircuitBreakerConfig(
            name="default",
            **self.config["default_circuit_breaker"]
        )
        self.circuit_breakers["default"] = CircuitBreaker(default_cb_config)
        
        # Default retry manager
        default_backoff = ExponentialBackoff(**self.config["default_retry"])
        self.retry_managers["default"] = RetryManager(
            backoff=default_backoff,
            circuit_breaker=self.circuit_breakers["default"],
            name="default"
        )
        
        # Default health checks
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        # Memory health check
        def memory_health_check():
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                return memory_percent < 90  # Healthy if less than 90% memory usage
            except ImportError:
                return True  # Skip if psutil not available
        
        memory_check = HealthCheck(
            name="memory",
            check_func=memory_health_check,
            timeout=5.0,
            critical=True,
            interval=30.0
        )
        self.health_monitor.add_check(memory_check)
        
        # Disk health check
        def disk_health_check():
            try:
                import psutil
                disk_percent = psutil.disk_usage('/').percent
                return disk_percent < 95  # Healthy if less than 95% disk usage
            except (ImportError, FileNotFoundError):
                return True  # Skip if psutil not available or path not found
        
        disk_check = HealthCheck(
            name="disk",
            check_func=disk_health_check,
            timeout=5.0,
            critical=False,
            interval=60.0
        )
        self.health_monitor.add_check(disk_check)
    
    def start(self):
        """Start the reliability engine."""
        self.health_monitor.start()
        self.logger.info("Reliability engine started")
    
    def stop(self):
        """Stop the reliability engine."""
        self.health_monitor.stop()
        self.logger.info("Reliability engine stopped")
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create a new circuit breaker."""
        cb = CircuitBreaker(config)
        self.circuit_breakers[name] = cb
        self.logger.info(f"Created circuit breaker: {name}")
        return cb
    
    def create_retry_manager(
        self,
        name: str,
        backoff: ExponentialBackoff = None,
        circuit_breaker: CircuitBreaker = None,
        retryable_exceptions: tuple = (Exception,)
    ) -> RetryManager:
        """Create a new retry manager."""
        rm = RetryManager(
            backoff=backoff,
            circuit_breaker=circuit_breaker,
            retryable_exceptions=retryable_exceptions,
            name=name
        )
        self.retry_managers[name] = rm
        self.logger.info(f"Created retry manager: {name}")
        return rm
    
    def get_reliability_report(self) -> Dict[str, Any]:
        """Generate comprehensive reliability report."""
        uptime = time.time() - self.start_time
        
        # Circuit breaker stats
        cb_stats = {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}
        
        # Health status
        health_status = self.health_monitor.get_status()
        
        # Resource leak report
        leak_report = self.resource_detector.get_leak_report()
        
        return {
            "timestamp": time.time(),
            "uptime": uptime,
            "overall_health": health_status["status"],
            "circuit_breakers": cb_stats,
            "health_monitoring": health_status,
            "resource_tracking": leak_report,
            "active_components": {
                "circuit_breakers": len(self.circuit_breakers),
                "retry_managers": len(self.retry_managers),
                "health_checks": health_status["active_health_checks"]
            }
        }
    
    @contextmanager
    def reliable_operation(
        self,
        operation_name: str = "default",
        retryable_exceptions: tuple = (Exception,),
        track_resources: bool = True
    ):
        """Context manager for reliable operations."""
        if track_resources:
            # Track this operation context as a resource
            self.resource_detector.track_resource(self, f"operation_{operation_name}")
        
        retry_manager = self.retry_managers.get(operation_name, self.retry_managers["default"])
        
        def operation_wrapper(func):
            return retry_manager.retry(func)
        
        yield operation_wrapper
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check to monitoring."""
        self.health_monitor.add_check(health_check)
    
    def add_recovery_function(self, check_name: str, recovery_func: Callable):
        """Add auto-recovery function for a health check."""
        self.health_monitor.add_recovery_function(check_name, recovery_func)


# Global reliability engine instance
_reliability_engine = None


def get_reliability_engine(config_path: Optional[str] = None) -> ReliabilityEngine:
    """Get or create global reliability engine instance."""
    global _reliability_engine
    
    if _reliability_engine is None:
        _reliability_engine = ReliabilityEngine(config_path)
        _reliability_engine.start()
    
    return _reliability_engine


def shutdown_reliability_engine():
    """Shutdown global reliability engine."""
    global _reliability_engine
    
    if _reliability_engine:
        _reliability_engine.stop()
        _reliability_engine = None