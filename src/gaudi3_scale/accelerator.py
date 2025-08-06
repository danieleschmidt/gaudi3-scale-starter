"""Enhanced Gaudi 3 accelerator integration for PyTorch Lightning.

This module provides production-ready HPU acceleration with comprehensive
error handling, validation, logging, and monitoring capabilities.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager

try:
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.accelerators import Accelerator
    _torch_available = True
except ImportError:
    torch = None
    pl = None
    Accelerator = object
    _torch_available = False

# Import habana frameworks for easier mocking in tests
try:
    import habana_frameworks.torch as htorch
    _habana_available = True
except ImportError:
    htorch = None
    _habana_available = False

# Import enhanced components
from .exceptions import (
    HPUNotAvailableError, HPUInitializationError, HPUMemoryError, 
    HPUDriverError, ParameterValidationError, create_validation_error
)
from .logging_utils import get_logger, log_performance, log_function_call
from .validation import DataValidator
from .retry_utils import retry_on_failure, execute_with_retry, get_hpu_retry_config
from .health_checks import HPUHealthCheck, HealthStatus

logger = get_logger('accelerator')


class GaudiAccelerator(Accelerator):
    """Production-ready PyTorch Lightning accelerator for Intel Gaudi HPUs.
    
    This accelerator provides optimized training on Intel Gaudi 3 devices
    with comprehensive error handling, validation, logging, and monitoring.
    
    Features:
        - Automatic HPU device detection and management
        - Optimized environment configuration with validation
        - Memory monitoring and statistics
        - Health checks and error recovery
        - Comprehensive logging and performance monitoring
        - Input validation and sanitization
        - Retry logic for transient failures
        - Integration with PyTorch Lightning training workflows
    
    Example:
        >>> accelerator = GaudiAccelerator(enable_monitoring=True)
        >>> trainer = pl.Trainer(accelerator=accelerator, devices=8)
        >>> # Accelerator automatically handles HPU setup, validation, and monitoring
    """
    
    def __init__(
        self,
        enable_monitoring: bool = True,
        enable_health_checks: bool = True,
        enable_retry: bool = True,
        max_retry_attempts: int = 3,
        validate_environment: bool = True,
        custom_env_vars: Optional[Dict[str, str]] = None
    ) -> None:
        """Initialize Gaudi accelerator with enhanced capabilities.
        
        Args:
            enable_monitoring: Enable performance monitoring
            enable_health_checks: Enable automated health checks
            enable_retry: Enable retry logic for transient failures
            max_retry_attempts: Maximum retry attempts for operations
            validate_environment: Validate environment setup
            custom_env_vars: Custom environment variables to set
        
        Raises:
            HPUNotAvailableError: If Habana frameworks are not available
            HPUInitializationError: If HPU initialization fails
        """
        logger.info("Initializing GaudiAccelerator with enhanced capabilities")
        
        # Store configuration
        self.enable_monitoring = enable_monitoring
        self.enable_health_checks = enable_health_checks
        self.enable_retry = enable_retry
        self.max_retry_attempts = max_retry_attempts
        self.validate_environment = validate_environment
        self.custom_env_vars = custom_env_vars or {}
        
        # Initialize monitoring data
        self._performance_data = {}
        self._health_checker = None
        self._validator = DataValidator()
        
        try:
            super().__init__()
            
            # Enhanced initialization with error handling
            with logger.context_manager(component="accelerator_init"):
                self._check_habana_availability()
                self._setup_environment_enhanced()
                
                if self.enable_health_checks:
                    self._setup_health_checks()
                
                if self.enable_monitoring:
                    self._setup_monitoring()
                
                logger.info("GaudiAccelerator initialization completed successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize GaudiAccelerator: {str(e)}")
            raise HPUInitializationError(
                f"Gaudi accelerator initialization failed: {str(e)}",
                context={"enable_monitoring": enable_monitoring, "enable_health_checks": enable_health_checks}
            ) from e
    
    def _check_habana_availability(self) -> None:
        """Check if Habana frameworks are available with enhanced validation.
        
        Raises:
            HPUNotAvailableError: If habana-torch-plugin is not installed or HPU devices not found
        """
        logger.debug("Checking Habana frameworks availability")
        
        if htorch is None:
            raise HPUNotAvailableError(
                requested_devices=1,
                available_devices=0,
                context={"error_type": "missing_habana_frameworks"},
                cause=ImportError("habana-torch-plugin not installed")
            )
        
        # Additional checks for HPU availability
        try:
            if not htorch.hpu.is_available():
                raise HPUNotAvailableError(
                    requested_devices=1,
                    available_devices=0,
                    context={"error_type": "hpu_not_available"},
                    cause=RuntimeError("HPU devices not detected")
                )
            
            device_count = htorch.hpu.device_count()
            if device_count == 0:
                raise HPUNotAvailableError(
                    requested_devices=1,
                    available_devices=0,
                    context={"error_type": "no_hpu_devices"}
                )
            
            logger.info(f"Habana frameworks available with {device_count} HPU device(s)")
            
        except Exception as e:
            if isinstance(e, HPUNotAvailableError):
                raise
            
            logger.error(f"Error checking HPU availability: {str(e)}")
            raise HPUNotAvailableError(
                requested_devices=1,
                available_devices=0,
                context={"error_type": "availability_check_failed"},
                cause=e
            )
    
    def _setup_environment_enhanced(self) -> None:
        """Setup optimal Gaudi environment variables with validation and logging.
        
        Configures Habana graph compiler and memory optimization settings
        for best performance on Gaudi 3 devices, with comprehensive validation.
        
        Raises:
            HPUInitializationError: If environment setup fails
        """
        logger.info("Setting up Gaudi environment variables")
        
        try:
            # Default optimal settings
            default_env_vars = {
                'PT_HPU_LAZY_MODE': '1',
                'PT_HPU_ENABLE_LAZY_COMPILATION': '1',
                'PT_HPU_GRAPH_COMPILER_OPT_LEVEL': '3',
                'PT_HPU_MAX_COMPOUND_OP_SIZE': '256',
                'PT_HPU_ENABLE_SYNAPSE_LAYOUT_OPT': '1',
                'PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE': '1',
                'PT_HPU_POOL_STRATEGY': 'OPTIMIZE_UTILIZATION'
            }
            
            # Merge with custom environment variables
            env_vars = {**default_env_vars, **self.custom_env_vars}
            
            # Validate and set environment variables
            for var, value in env_vars.items():
                if self.validate_environment:
                    self._validate_env_var(var, value)
                
                # Set environment variable if not already set or if custom
                if var not in os.environ or var in self.custom_env_vars:
                    os.environ[var] = value
                    logger.debug(f"Set environment variable {var}={value}")
                else:
                    logger.debug(f"Environment variable {var} already set to {os.environ[var]}")
            
            # Verify critical environment variables
            self._verify_environment_setup()
            
            logger.info("Gaudi environment setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Gaudi environment: {str(e)}")
            raise HPUInitializationError(
                f"Environment setup failed: {str(e)}",
                context={"custom_env_vars": self.custom_env_vars}
            ) from e
    
    @staticmethod
    @log_function_call(level=20)  # INFO level
    def is_available() -> bool:
        """Check if Gaudi accelerator is available with enhanced validation.
        
        Returns:
            bool: True if Habana HPU devices are available, False otherwise.
        """
        try:
            if htorch is None:
                logger.debug("Habana frameworks not available")
                return False
            
            available = htorch.hpu.is_available()
            device_count = htorch.hpu.device_count() if available else 0
            
            logger.debug(
                f"HPU availability check: available={available}, devices={device_count}",
                extra={"hpu_available": available, "hpu_device_count": device_count}
            )
            
            return available and device_count > 0
            
        except Exception as e:
            logger.warning(f"Error checking HPU availability: {str(e)}")
            return False
    
    @log_function_call(include_args=True)
    def parse_devices(self, devices: Union[int, str, List[int]]) -> Union[int, List[int]]:
        """Parse device specification for HPU devices with enhanced validation.
        
        Args:
            devices: Device specification - int, string, or list of device indices.
        
        Returns:
            Union[int, List[int]]: Parsed device specification.
            
        Raises:
            ParameterValidationError: If device specification is invalid.
        """
        logger.debug(f"Parsing device specification: {devices}")
        
        try:
            # Validate input type
            if not isinstance(devices, (int, str, list)):
                raise ParameterValidationError(
                    f"Invalid device specification type: {type(devices).__name__}",
                    parameter_name="devices",
                    parameter_value=devices,
                    expected_type=Union[int, str, List[int]]
                )
            
            # Handle "auto" specification
            if devices == "auto":
                auto_count = self.auto_device_count()
                logger.info(f"Auto-detected {auto_count} HPU devices")
                return auto_count
            
            # Handle string specification
            if isinstance(devices, str):
                if devices.isdigit():
                    parsed_devices = int(devices)
                    if parsed_devices < 0:
                        raise ParameterValidationError(
                            "Device count must be non-negative",
                            parameter_name="devices",
                            parameter_value=parsed_devices
                        )
                    return parsed_devices
                else:
                    raise ParameterValidationError(
                        f"Invalid string device specification: '{devices}' (must be 'auto' or numeric)",
                        parameter_name="devices",
                        parameter_value=devices
                    )
            
            # Handle integer specification
            if isinstance(devices, int):
                if devices < 0:
                    raise ParameterValidationError(
                        "Device count must be non-negative",
                        parameter_name="devices",
                        parameter_value=devices
                    )
                
                # Validate against available devices
                available_count = self.auto_device_count()
                if devices > available_count:
                    logger.warning(
                        f"Requested {devices} devices but only {available_count} available"
                    )
                
                return devices
            
            # Handle list specification
            if isinstance(devices, list):
                if not devices:
                    raise ParameterValidationError(
                        "Device list cannot be empty",
                        parameter_name="devices",
                        parameter_value=devices
                    )
                
                # Validate all elements are non-negative integers
                for i, device_id in enumerate(devices):
                    if not isinstance(device_id, int) or device_id < 0:
                        raise ParameterValidationError(
                            f"Device index at position {i} must be a non-negative integer",
                            parameter_name=f"devices[{i}]",
                            parameter_value=device_id
                        )
                
                # Check for duplicates
                if len(set(devices)) != len(devices):
                    raise ParameterValidationError(
                        "Device list contains duplicate indices",
                        parameter_name="devices",
                        parameter_value=devices
                    )
                
                # Validate against available devices
                available_count = self.auto_device_count()
                invalid_devices = [d for d in devices if d >= available_count]
                if invalid_devices:
                    raise ParameterValidationError(
                        f"Device indices {invalid_devices} not available (max: {available_count-1})",
                        parameter_name="devices",
                        parameter_value=devices
                    )
                
                logger.info(f"Parsed device list: {devices}")
                return devices
            
        except Exception as e:
            if isinstance(e, ParameterValidationError):
                raise
            
            logger.error(f"Unexpected error parsing devices: {str(e)}")
            raise ParameterValidationError(
                f"Failed to parse device specification: {str(e)}",
                parameter_name="devices",
                parameter_value=devices
            ) from e
    
    @retry_on_failure(max_attempts=3, base_delay=1.0)
    @log_function_call(include_args=True)
    def get_parallel_devices(self, devices: Union[int, List[int]]) -> List[Any]:
        """Convert device indices to HPU device objects with enhanced validation.
        
        Args:
            devices: Device count or list of device indices.
        
        Returns:
            List[torch.device]: List of HPU device objects.
            
        Raises:
            HPUNotAvailableError: If PyTorch is not available or devices invalid
            ParameterValidationError: If device specification is invalid.
        """
        if torch is None:
            raise HPUNotAvailableError(
                requested_devices=1,
                available_devices=0,
                context={"error_type": "pytorch_not_available"},
                cause=ImportError("PyTorch not available")
            )
        
        logger.debug(f"Converting devices to parallel device objects: {devices}")
        
        try:
            if isinstance(devices, int):
                if devices <= 0:
                    raise ParameterValidationError(
                        "Device count must be positive",
                        parameter_name="devices",
                        parameter_value=devices
                    )
                
                available_count = self.auto_device_count()
                if devices > available_count:
                    raise HPUNotAvailableError(
                        requested_devices=devices,
                        available_devices=available_count
                    )
                
                device_objects = [torch.device(f"hpu:{i}") for i in range(devices)]
                logger.info(f"Created {len(device_objects)} HPU device objects")
                return device_objects
            
            if isinstance(devices, list):
                available_count = self.auto_device_count()
                
                # Validate all device indices
                for device_idx in devices:
                    if device_idx >= available_count:
                        raise HPUNotAvailableError(
                            requested_devices=max(devices) + 1,
                            available_devices=available_count,
                            context={"invalid_device_index": device_idx}
                        )
                
                device_objects = [torch.device(f"hpu:{i}") for i in devices]
                logger.info(f"Created device objects for indices {devices}")
                return device_objects
            
            raise ParameterValidationError(
                f"Unsupported device specification type: {type(devices).__name__}",
                parameter_name="devices",
                parameter_value=devices
            )
            
        except Exception as e:
            if isinstance(e, (HPUNotAvailableError, ParameterValidationError)):
                raise
            
            logger.error(f"Error creating parallel devices: {str(e)}")
            raise HPUInitializationError(
                f"Failed to create parallel devices: {str(e)}",
                context={"devices": devices}
            ) from e
    
    @log_performance()
    def auto_device_count(self) -> int:
        """Get the number of available HPU devices with caching and validation.
        
        Returns:
            int: Number of available HPU devices.
        """
        try:
            if htorch is None:
                logger.debug("Habana frameworks not available, returning 0 devices")
                return 0
            
            if not htorch.hpu.is_available():
                logger.debug("HPU not available, returning 0 devices")
                return 0
            
            device_count = htorch.hpu.device_count()
            logger.debug(f"Auto-detected {device_count} HPU devices")
            
            # Cache the result for performance
            if not hasattr(self, '_cached_device_count'):
                self._cached_device_count = device_count
                logger.info(f"Cached HPU device count: {device_count}")
            
            return device_count
            
        except Exception as e:
            logger.warning(f"Error getting HPU device count: {str(e)}")
            return 0
    
    @log_performance()
    @retry_on_failure(max_attempts=2, base_delay=0.5)
    def get_device_stats(self, device: Union[Any, str, int]) -> Dict[str, Any]:
        """Get HPU device statistics for monitoring with enhanced error handling.
        
        Args:
            device: Device to get statistics for.
        
        Returns:
            Dict[str, Any]: Dictionary containing device statistics.
        """
        if htorch is None:
            logger.warning("Habana frameworks not available for device stats")
            return {"error": "habana_frameworks_unavailable", "device": str(device)}
        
        try:
            # Enhanced device parsing with validation
            device_idx = self._parse_device_index(device)
            
            # Validate device index
            available_count = self.auto_device_count()
            if device_idx >= available_count:
                raise HPUNotAvailableError(
                    requested_devices=device_idx + 1,
                    available_devices=available_count,
                    context={"requested_device_idx": device_idx}
                )
            
            # Create device object - handle case where torch is None
            if torch is not None:
                device_obj = torch.device(f"hpu:{device_idx}")
            else:
                device_obj = f"hpu:{device_idx}"
            
            # Collect comprehensive statistics
            stats = {
                "device_index": device_idx,
                "timestamp": time.time(),
                "hpu_device_count": htorch.hpu.device_count(),
                "hpu_current_device": htorch.hpu.current_device(),
            }
            
            # Memory statistics with error handling
            try:
                memory_allocated = htorch.hpu.memory_allocated(device_obj)
                memory_reserved = htorch.hpu.memory_reserved(device_obj)
                
                stats.update({
                    "hpu_memory_allocated_bytes": memory_allocated,
                    "hpu_memory_reserved_bytes": memory_reserved,
                    "hpu_memory_allocated_gb": memory_allocated / (1024**3),
                    "hpu_memory_reserved_gb": memory_reserved / (1024**3),
                    "memory_utilization_percent": (memory_allocated / memory_reserved * 100) if memory_reserved > 0 else 0
                })
            except Exception as e:
                logger.warning(f"Failed to get memory stats for device {device_idx}: {e}")
                stats["memory_error"] = str(e)
            
            # Device name with fallback
            try:
                device_name = htorch.hpu.get_device_name(device_obj)
                stats["hpu_device_name"] = device_name
            except (AttributeError, RuntimeError) as e:
                logger.debug(f"Could not get device name for device {device_idx}: {e}")
                stats["hpu_device_name"] = f"HPU {device_idx}"
            
            # Additional system information
            stats.update({
                "driver_version": self._get_driver_version(),
                "is_available": htorch.hpu.is_available(),
                "stats_collection_time": time.time() - stats["timestamp"]
            })
            
            logger.debug(f"Collected stats for HPU device {device_idx}")
            return stats
            
        except Exception as e:
            error_context = {
                "device": str(device),
                "available_devices": self.auto_device_count()
            }
            
            logger.error(
                f"Failed to get device stats: {str(e)}",
                extra={"error_context": error_context}
            )
            
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "device": str(device),
                "timestamp": time.time(),
                "context": error_context
            }
    
    @classmethod
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        """Register the Gaudi accelerator with Lightning CLI.
        
        Args:
            accelerator_registry: Lightning accelerator registry.
        """
        try:
            accelerator_registry("hpu", cls)
            logger.info("Successfully registered GaudiAccelerator with Lightning CLI")
        except Exception as e:
            logger.error(f"Failed to register GaudiAccelerator: {str(e)}")
            raise HPUInitializationError(
                f"Failed to register accelerator: {str(e)}"
            ) from e
    
    # Enhanced private methods
    
    def _validate_env_var(self, var_name: str, value: str) -> None:
        """Validate environment variable value.
        
        Args:
            var_name: Environment variable name
            value: Environment variable value
            
        Raises:
            ParameterValidationError: If validation fails
        """
        # Validate variable name
        name_result = self._validator.validate_string(
            var_name, "env_var_name",
            min_length=1, max_length=100,
            pattern=re.compile(r'^[A-Z][A-Z0-9_]*$')
        )
        if not name_result.is_valid:
            raise ParameterValidationError(
                f"Invalid environment variable name: {var_name}",
                parameter_name="env_var_name",
                parameter_value=var_name
            )
        
        # Validate variable value
        value_result = self._validator.validate_string(
            value, "env_var_value",
            max_length=1000
        )
        if not value_result.is_valid:
            raise ParameterValidationError(
                f"Invalid environment variable value for {var_name}",
                parameter_name="env_var_value",
                parameter_value=value
            )
    
    def _verify_environment_setup(self) -> None:
        """Verify that critical environment variables are properly set.
        
        Raises:
            HPUInitializationError: If critical environment variables are missing or invalid
        """
        critical_vars = ['PT_HPU_LAZY_MODE', 'PT_HPU_ENABLE_LAZY_COMPILATION']
        
        for var in critical_vars:
            if var not in os.environ:
                raise HPUInitializationError(
                    f"Critical environment variable {var} not set",
                    context={"missing_var": var}
                )
        
        logger.debug("Environment setup verification completed")
    
    def _setup_health_checks(self) -> None:
        """Setup automated health checks for HPU devices."""
        try:
            device_count = self.auto_device_count()
            self._health_checker = HPUHealthCheck(
                device_id=0,  # Check primary device
                timeout=10.0
            )
            
            # Run initial health check
            health_result = self._health_checker.check()
            if health_result.status != HealthStatus.HEALTHY:
                logger.warning(
                    f"Initial HPU health check failed: {health_result.message}",
                    extra={"health_status": health_result.status.value}
                )
            else:
                logger.info("Initial HPU health check passed")
                
        except Exception as e:
            logger.warning(f"Failed to setup health checks: {str(e)}")
    
    def _setup_monitoring(self) -> None:
        """Setup performance monitoring."""
        try:
            self._performance_data = {
                "initialization_time": time.time(),
                "device_stats_calls": 0,
                "parse_devices_calls": 0,
                "errors": []
            }
            
            logger.info("Performance monitoring enabled")
            
        except Exception as e:
            logger.warning(f"Failed to setup monitoring: {str(e)}")
    
    def _parse_device_index(self, device: Union[Any, str, int]) -> int:
        """Parse device specification to get device index.
        
        Args:
            device: Device specification
        
        Returns:
            int: Device index
            
        Raises:
            ParameterValidationError: If device specification is invalid
        """
        try:
            if isinstance(device, int):
                return device
            elif isinstance(device, str):
                return int(device.replace('hpu:', ''))
            elif hasattr(device, 'index'):
                return device.index
            elif hasattr(device, 'type') and 'hpu' in str(device):
                # Extract index from torch.device object
                device_str = str(device)
                if ':' in device_str:
                    return int(device_str.split(':')[1])
                return 0
            else:
                return 0
                
        except (ValueError, AttributeError) as e:
            raise ParameterValidationError(
                f"Cannot parse device index from {device}",
                parameter_name="device",
                parameter_value=device
            ) from e
    
    def _get_driver_version(self) -> str:
        """Get HPU driver version.
        
        Returns:
            str: Driver version or 'unknown' if not available
        """
        try:
            # This would be implementation-specific
            # For now, return a placeholder
            return "unknown"
        except Exception:
            return "unknown"
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations.
        
        Args:
            operation_name: Name of the operation to monitor
        """
        if not self.enable_monitoring:
            yield
            return
        
        start_time = time.time()
        operation_id = logger.log_operation_start(operation_name)
        
        try:
            yield
            duration = time.time() - start_time
            logger.log_operation_end(operation_name, operation_id, True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            logger.log_operation_end(operation_name, operation_id, False, duration)
            
            if hasattr(self, '_performance_data'):
                self._performance_data["errors"].append({
                    "operation": operation_name,
                    "error": str(e),
                    "timestamp": time.time()
                })
            
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary.
        
        Returns:
            Dict containing performance data
        """
        if not self.enable_monitoring or not hasattr(self, '_performance_data'):
            return {}
        
        current_time = time.time()
        uptime = current_time - self._performance_data.get("initialization_time", current_time)
        
        return {
            "uptime_seconds": uptime,
            "device_stats_calls": self._performance_data.get("device_stats_calls", 0),
            "parse_devices_calls": self._performance_data.get("parse_devices_calls", 0),
            "error_count": len(self._performance_data.get("errors", [])),
            "recent_errors": self._performance_data.get("errors", [])[-5:],  # Last 5 errors
            "health_status": self.get_health_status()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the accelerator.
        
        Returns:
            Dict containing health information
        """
        if not self.enable_health_checks or not self._health_checker:
            return {"status": "unknown", "reason": "health_checks_disabled"}
        
        try:
            health_result = self._health_checker.check()
            return {
                "status": health_result.status.value,
                "message": health_result.message,
                "details": health_result.details,
                "last_check": health_result.timestamp.isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "error": str(e)
            }