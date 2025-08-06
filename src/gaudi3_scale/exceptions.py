"""Custom exception classes for Gaudi 3 Scale with Quantum State Monitoring.

This module defines a comprehensive hierarchy of custom exceptions for
different types of errors and failure modes in the Gaudi 3 Scale system.
These exceptions provide structured error handling with context and
recovery suggestions, enhanced with quantum state monitoring for
advanced error tracking and correlation analysis.
"""

import time
import cmath
import numpy as np
from typing import Any, Dict, List, Optional, Union, Set
from enum import Enum


class QuantumErrorState(Enum):
    """Quantum error states for enhanced error tracking."""
    COHERENT = "coherent"           # Normal operation
    DECOHERENT = "decoherent"      # Quantum decoherence detected
    ENTANGLED = "entangled"        # Error entangled with other systems
    SUPERPOSITION = "superposition" # Error in quantum superposition
    COLLAPSED = "collapsed"        # Error state collapsed to classical


class ErrorCode(Enum):
    """Standardized error codes for different failure types."""
    
    # Configuration Errors (1000-1099)
    INVALID_CONFIG = 1000
    MISSING_REQUIRED_CONFIG = 1001
    INVALID_MODEL_CONFIG = 1002
    INVALID_DATASET_CONFIG = 1003
    INVALID_TRAINING_CONFIG = 1004
    INVALID_CLUSTER_CONFIG = 1005
    
    # Hardware Errors (2000-2099)
    HPU_NOT_AVAILABLE = 2000
    HPU_INITIALIZATION_FAILED = 2001
    HPU_MEMORY_ERROR = 2002
    HPU_DRIVER_ERROR = 2003
    HPU_DEVICE_ERROR = 2004
    HPU_COMMUNICATION_ERROR = 2005
    
    # Training Errors (3000-3099)
    TRAINING_INITIALIZATION_FAILED = 3000
    TRAINING_STEP_FAILED = 3001
    GRADIENT_OVERFLOW = 3002
    OPTIMIZER_ERROR = 3003
    CHECKPOINT_SAVE_FAILED = 3004
    CHECKPOINT_LOAD_FAILED = 3005
    DATA_LOADING_ERROR = 3006
    MODEL_COMPILATION_FAILED = 3007
    
    # Validation Errors (4000-4099)
    VALIDATION_FAILED = 4000
    PARAMETER_VALIDATION_ERROR = 4001
    DATA_VALIDATION_ERROR = 4002
    MODEL_VALIDATION_ERROR = 4003
    INPUT_SANITIZATION_ERROR = 4004
    
    # Resource Errors (5000-5099)
    INSUFFICIENT_MEMORY = 5000
    INSUFFICIENT_STORAGE = 5001
    RESOURCE_ALLOCATION_FAILED = 5002
    RESOURCE_CLEANUP_FAILED = 5003
    QUOTA_EXCEEDED = 5004
    
    # Network/Infrastructure Errors (6000-6099)
    NETWORK_CONNECTION_FAILED = 6000
    CLUSTER_DEPLOYMENT_FAILED = 6001
    SERVICE_UNAVAILABLE = 6002
    TIMEOUT_ERROR = 6003
    AUTHENTICATION_ERROR = 6004
    AUTHORIZATION_ERROR = 6005
    
    # Monitoring Errors (7000-7099)
    METRICS_COLLECTION_FAILED = 7000
    HEALTH_CHECK_FAILED = 7001
    ALERTING_FAILED = 7002
    MONITORING_SERVICE_DOWN = 7003
    
    # Unknown/Generic Errors (9000-9099)
    UNKNOWN_ERROR = 9000
    INTERNAL_ERROR = 9001
    UNEXPECTED_STATE = 9002
    
    # Quantum-Enhanced Error Codes (8000-8099)
    QUANTUM_DECOHERENCE_ERROR = 8000
    QUANTUM_ENTANGLEMENT_ERROR = 8001
    QUANTUM_STATE_COLLAPSE_ERROR = 8002
    QUANTUM_CIRCUIT_ERROR = 8003
    QUANTUM_MEASUREMENT_ERROR = 8004
    TASK_PLANNING_ERROR = 8010
    RESOURCE_ALLOCATION_ERROR = 8011
    ANNEALING_OPTIMIZATION_ERROR = 8012
    ENTANGLEMENT_COORDINATOR_ERROR = 8013


class QuantumEnhancedError(Exception):
    """Quantum-enhanced error with state monitoring and entanglement tracking."""
    
    def __init__(self,
                 message: str,
                 quantum_state: QuantumErrorState = QuantumErrorState.COHERENT,
                 amplitude: complex = complex(1.0, 0.0),
                 phase: float = 0.0,
                 entangled_errors: Set[str] = None,
                 decoherence_rate: float = 0.01,
                 timestamp: float = None):
        """Initialize quantum-enhanced error.
        
        Args:
            message: Error message
            quantum_state: Current quantum error state
            amplitude: Quantum amplitude of error state
            phase: Quantum phase of error state
            entangled_errors: Set of entangled error IDs
            decoherence_rate: Rate of quantum decoherence
            timestamp: Error occurrence timestamp
        """
        super().__init__(message)
        self.quantum_state = quantum_state
        self.amplitude = amplitude
        self.phase = phase
        self.entangled_errors = entangled_errors or set()
        self.decoherence_rate = decoherence_rate
        self.timestamp = timestamp or time.time()
        self.error_id = f"qerr_{int(self.timestamp * 1000000)}"  # Unique error ID
        
    @property
    def probability_amplitude(self) -> float:
        """Calculate probability amplitude |ψ|²."""
        return abs(self.amplitude) ** 2
    
    @property
    def coherence_time_remaining(self) -> float:
        """Calculate remaining coherence time."""
        elapsed = time.time() - self.timestamp
        return max(0.0, (1.0 / self.decoherence_rate) - elapsed)
    
    @property
    def is_coherent(self) -> bool:
        """Check if error state is still coherent."""
        return self.coherence_time_remaining > 0
    
    def apply_quantum_evolution(self, time_delta: float):
        """Apply quantum evolution to error state."""
        if self.quantum_state == QuantumErrorState.COHERENT:
            # Apply decoherence
            decoherence_factor = np.exp(-self.decoherence_rate * time_delta)
            self.amplitude *= decoherence_factor
            
            if abs(self.amplitude) < 0.1:
                self.quantum_state = QuantumErrorState.DECOHERENT
        
        elif self.quantum_state == QuantumErrorState.SUPERPOSITION:
            # Phase evolution in superposition
            self.phase += 0.1 * time_delta
            self.amplitude *= complex(np.cos(self.phase), np.sin(self.phase))
    
    def entangle_with_error(self, other_error: 'QuantumEnhancedError'):
        """Create quantum entanglement with another error."""
        self.entangled_errors.add(other_error.error_id)
        other_error.entangled_errors.add(self.error_id)
        
        # Both errors enter entangled state
        self.quantum_state = QuantumErrorState.ENTANGLED
        other_error.quantum_state = QuantumErrorState.ENTANGLED
        
        # Create Bell state correlation
        entanglement_phase = np.pi / 4
        self.phase += entanglement_phase
        other_error.phase -= entanglement_phase
    
    def collapse_quantum_state(self) -> bool:
        """Collapse quantum error state to classical."""
        measurement_outcome = np.random.random() < self.probability_amplitude
        
        self.quantum_state = QuantumErrorState.COLLAPSED
        self.amplitude = complex(1.0, 0.0) if measurement_outcome else complex(0.0, 0.0)
        
        return measurement_outcome
    
    def to_quantum_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with quantum state information."""
        return {
            "error_id": self.error_id,
            "message": str(self),
            "quantum_state": self.quantum_state.value,
            "amplitude": {
                "real": self.amplitude.real,
                "imag": self.amplitude.imag,
                "magnitude": abs(self.amplitude)
            },
            "phase": self.phase,
            "probability_amplitude": self.probability_amplitude,
            "entangled_errors": list(self.entangled_errors),
            "coherence_time_remaining": self.coherence_time_remaining,
            "is_coherent": self.is_coherent,
            "timestamp": self.timestamp,
            "decoherence_rate": self.decoherence_rate
        }


class Gaudi3ScaleError(Exception):
    """Base exception class for all Gaudi 3 Scale errors.
    
    Provides structured error information including error codes,
    context, and recovery suggestions.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        recovery_suggestions: Optional[List[str]] = None
    ) -> None:
        """Initialize Gaudi3ScaleError.
        
        Args:
            message: Human-readable error message
            error_code: Standardized error code
            context: Additional error context
            cause: Original exception that caused this error
            recovery_suggestions: List of suggested recovery actions
        """
        super().__init__(message)
        self.error_code = error_code or ErrorCode.UNKNOWN_ERROR
        self.context = context or {}
        self.cause = cause
        self.recovery_suggestions = recovery_suggestions or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "error_code": self.error_code.name,
            "error_code_value": self.error_code.value,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "recovery_suggestions": self.recovery_suggestions
        }
        
    def __str__(self) -> str:
        """String representation with context."""
        base_msg = super().__str__()
        if self.error_code:
            base_msg = f"[{self.error_code.name}] {base_msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (Context: {context_str})"
        return base_msg


# Configuration Errors

class ConfigurationError(Gaudi3ScaleError):
    """Base class for configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if config_key:
            context["config_key"] = config_key
        super().__init__(message, context=context, **kwargs)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid or malformed."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.INVALID_CONFIG,
            recovery_suggestions=[
                "Verify configuration file syntax",
                "Check configuration schema documentation",
                "Use configuration validation tools"
            ],
            **kwargs
        )


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, config_key: str, **kwargs):
        message = f"Required configuration '{config_key}' is missing"
        super().__init__(
            message,
            config_key=config_key,
            error_code=ErrorCode.MISSING_REQUIRED_CONFIG,
            recovery_suggestions=[
                f"Add '{config_key}' to your configuration",
                "Check configuration file completeness",
                "Review configuration templates"
            ],
            **kwargs
        )


class ModelConfigurationError(ConfigurationError):
    """Raised when model configuration is invalid."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.INVALID_MODEL_CONFIG,
            recovery_suggestions=[
                "Verify model name and path",
                "Check model size specification",
                "Validate model parameters"
            ],
            **kwargs
        )


class DatasetConfigurationError(ConfigurationError):
    """Raised when dataset configuration is invalid."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.INVALID_DATASET_CONFIG,
            recovery_suggestions=[
                "Verify dataset path or name",
                "Check dataset access permissions",
                "Validate dataset format"
            ],
            **kwargs
        )


class TrainingConfigurationError(ConfigurationError):
    """Raised when training configuration is invalid."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.INVALID_TRAINING_CONFIG,
            recovery_suggestions=[
                "Check training hyperparameters",
                "Verify resource requirements",
                "Validate training strategy"
            ],
            **kwargs
        )


# Hardware Errors

class HardwareError(Gaudi3ScaleError):
    """Base class for hardware-related errors."""
    pass


class HPUError(HardwareError):
    """Base class for HPU-specific errors.
    
    This is the base class for all Intel Gaudi HPU-related errors.
    Specific HPU error types inherit from this class.
    """
    pass


class HPUNotAvailableError(HPUError):
    """Raised when HPU devices are not available."""
    
    def __init__(self, requested_devices: int = 1, available_devices: int = 0, **kwargs):
        message = f"HPU devices not available (requested: {requested_devices}, available: {available_devices})"
        context = kwargs.pop("context", {})
        context.update({
            "requested_devices": requested_devices,
            "available_devices": available_devices
        })
        super().__init__(
            message,
            error_code=ErrorCode.HPU_NOT_AVAILABLE,
            context=context,
            recovery_suggestions=[
                "Check HPU driver installation",
                "Verify Habana software stack",
                "Ensure HPU devices are not in use",
                "Check system compatibility"
            ],
            **kwargs
        )


class HPUInitializationError(HPUError):
    """Raised when HPU initialization fails."""
    
    def __init__(self, message: str, device_id: Optional[int] = None, **kwargs):
        context = kwargs.pop("context", {})
        if device_id is not None:
            context["device_id"] = device_id
        super().__init__(
            message,
            error_code=ErrorCode.HPU_INITIALIZATION_FAILED,
            context=context,
            recovery_suggestions=[
                "Restart HPU driver services",
                "Check device permissions",
                "Verify system resources",
                "Review HPU device logs"
            ],
            **kwargs
        )


class HPUMemoryError(HPUError):
    """Raised when HPU memory operations fail."""
    
    def __init__(
        self,
        message: str,
        required_memory: Optional[int] = None,
        available_memory: Optional[int] = None,
        device_id: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if required_memory is not None:
            context["required_memory_gb"] = required_memory / (1024**3)
        if available_memory is not None:
            context["available_memory_gb"] = available_memory / (1024**3)
        if device_id is not None:
            context["device_id"] = device_id
            
        super().__init__(
            message,
            error_code=ErrorCode.HPU_MEMORY_ERROR,
            context=context,
            recovery_suggestions=[
                "Reduce batch size",
                "Use gradient accumulation",
                "Enable gradient checkpointing",
                "Clear HPU memory cache",
                "Use model parallelism"
            ],
            **kwargs
        )


class HPUDriverError(HPUError):
    """Raised when HPU driver errors occur."""
    
    def __init__(self, message: str, driver_version: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if driver_version:
            context["driver_version"] = driver_version
        super().__init__(
            message,
            error_code=ErrorCode.HPU_DRIVER_ERROR,
            context=context,
            recovery_suggestions=[
                "Update HPU drivers",
                "Restart driver services",
                "Check system compatibility",
                "Review driver logs"
            ],
            **kwargs
        )


# Training Errors

class TrainingError(Gaudi3ScaleError):
    """Base class for training-related errors."""
    pass


class TrainingInitializationError(TrainingError):
    """Raised when training initialization fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.TRAINING_INITIALIZATION_FAILED,
            recovery_suggestions=[
                "Check model and data configurations",
                "Verify resource availability",
                "Review training parameters",
                "Check for conflicting processes"
            ],
            **kwargs
        )


class TrainingStepError(TrainingError):
    """Raised when a training step fails."""
    
    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if epoch is not None:
            context["epoch"] = epoch
        if step is not None:
            context["step"] = step
            
        super().__init__(
            message,
            error_code=ErrorCode.TRAINING_STEP_FAILED,
            context=context,
            recovery_suggestions=[
                "Check for data corruption",
                "Reduce learning rate",
                "Enable gradient clipping",
                "Review model parameters"
            ],
            **kwargs
        )


class GradientOverflowError(TrainingError):
    """Raised when gradient overflow is detected."""
    
    def __init__(self, message: str = "Gradient overflow detected", **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.GRADIENT_OVERFLOW,
            recovery_suggestions=[
                "Reduce learning rate",
                "Enable gradient scaling",
                "Use gradient clipping",
                "Reduce batch size",
                "Check for extreme values in data"
            ],
            **kwargs
        )


class OptimizerError(TrainingError):
    """Raised when optimizer operations fail."""
    
    def __init__(self, message: str, optimizer_type: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if optimizer_type:
            context["optimizer_type"] = optimizer_type
        super().__init__(
            message,
            error_code=ErrorCode.OPTIMIZER_ERROR,
            context=context,
            recovery_suggestions=[
                "Check optimizer parameters",
                "Verify parameter updates",
                "Review learning rate schedule",
                "Try different optimizer"
            ],
            **kwargs
        )


class CheckpointError(TrainingError):
    """Base class for checkpoint-related errors."""
    pass


class CheckpointSaveError(CheckpointError):
    """Raised when checkpoint saving fails."""
    
    def __init__(self, message: str, checkpoint_path: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if checkpoint_path:
            context["checkpoint_path"] = checkpoint_path
        super().__init__(
            message,
            error_code=ErrorCode.CHECKPOINT_SAVE_FAILED,
            context=context,
            recovery_suggestions=[
                "Check disk space",
                "Verify write permissions",
                "Ensure directory exists",
                "Check file system health"
            ],
            **kwargs
        )


class CheckpointLoadError(CheckpointError):
    """Raised when checkpoint loading fails."""
    
    def __init__(self, message: str, checkpoint_path: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if checkpoint_path:
            context["checkpoint_path"] = checkpoint_path
        super().__init__(
            message,
            error_code=ErrorCode.CHECKPOINT_LOAD_FAILED,
            context=context,
            recovery_suggestions=[
                "Verify checkpoint file exists",
                "Check file permissions",
                "Validate checkpoint format",
                "Try loading different checkpoint"
            ],
            **kwargs
        )


class DataLoadingError(TrainingError):
    """Raised when data loading fails."""
    
    def __init__(self, message: str, dataset_path: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if dataset_path:
            context["dataset_path"] = dataset_path
        super().__init__(
            message,
            error_code=ErrorCode.DATA_LOADING_ERROR,
            context=context,
            recovery_suggestions=[
                "Verify dataset path",
                "Check data format",
                "Validate data integrity",
                "Increase data loading workers"
            ],
            **kwargs
        )


# Validation Errors

class ValidationError(Gaudi3ScaleError):
    """Base class for validation errors."""
    pass


class ParameterValidationError(ValidationError):
    """Raised when parameter validation fails."""
    
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        expected_type: Optional[type] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if parameter_name:
            context["parameter_name"] = parameter_name
        if parameter_value is not None:
            context["parameter_value"] = str(parameter_value)
        if expected_type:
            context["expected_type"] = expected_type.__name__
            
        super().__init__(
            message,
            error_code=ErrorCode.PARAMETER_VALIDATION_ERROR,
            context=context,
            recovery_suggestions=[
                "Check parameter type and range",
                "Review parameter documentation",
                "Validate input values"
            ],
            **kwargs
        )


class DataValidationError(ValidationError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.DATA_VALIDATION_ERROR,
            recovery_suggestions=[
                "Verify data format",
                "Check data integrity",
                "Validate data schema",
                "Remove corrupted data"
            ],
            **kwargs
        )


class InputSanitizationError(ValidationError):
    """Raised when input sanitization fails."""
    
    def __init__(self, message: str, input_name: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if input_name:
            context["input_name"] = input_name
        super().__init__(
            message,
            error_code=ErrorCode.INPUT_SANITIZATION_ERROR,
            context=context,
            recovery_suggestions=[
                "Remove malicious characters",
                "Validate input format",
                "Apply input filters",
                "Use safe input methods"
            ],
            **kwargs
        )


class SecurityValidationError(ValidationError):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, security_check: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if security_check:
            context["security_check"] = security_check
        super().__init__(
            message,
            error_code=ErrorCode.VALIDATION_FAILED,
            context=context,
            recovery_suggestions=[
                "Review security requirements",
                "Check security policies",
                "Validate security configuration",
                "Contact security team"
            ],
            **kwargs
        )


# Resource Errors

class ResourceError(Gaudi3ScaleError):
    """Base class for resource-related errors."""
    pass


class InsufficientMemoryError(ResourceError):
    """Raised when there is insufficient memory."""
    
    def __init__(
        self,
        message: str,
        required_memory: Optional[int] = None,
        available_memory: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if required_memory is not None:
            context["required_memory_gb"] = required_memory / (1024**3)
        if available_memory is not None:
            context["available_memory_gb"] = available_memory / (1024**3)
            
        super().__init__(
            message,
            error_code=ErrorCode.INSUFFICIENT_MEMORY,
            context=context,
            recovery_suggestions=[
                "Reduce batch size",
                "Use gradient accumulation",
                "Enable model parallelism",
                "Clear memory caches",
                "Close other applications"
            ],
            **kwargs
        )


class InsufficientStorageError(ResourceError):
    """Raised when there is insufficient storage space."""
    
    def __init__(
        self,
        message: str,
        required_space: Optional[int] = None,
        available_space: Optional[int] = None,
        path: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if required_space is not None:
            context["required_space_gb"] = required_space / (1024**3)
        if available_space is not None:
            context["available_space_gb"] = available_space / (1024**3)
        if path:
            context["path"] = path
            
        super().__init__(
            message,
            error_code=ErrorCode.INSUFFICIENT_STORAGE,
            context=context,
            recovery_suggestions=[
                "Free up disk space",
                "Move files to external storage",
                "Clean up temporary files",
                "Use compression",
                "Choose different storage location"
            ],
            **kwargs
        )


# Network/Infrastructure Errors

class NetworkError(Gaudi3ScaleError):
    """Base class for network-related errors."""
    pass


class NetworkConnectionError(NetworkError):
    """Raised when network connections fail."""
    
    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if host:
            context["host"] = host
        if port:
            context["port"] = port
            
        super().__init__(
            message,
            error_code=ErrorCode.NETWORK_CONNECTION_FAILED,
            context=context,
            recovery_suggestions=[
                "Check network connectivity",
                "Verify firewall settings",
                "Test DNS resolution",
                "Check service availability"
            ],
            **kwargs
        )


class ServiceUnavailableError(NetworkError):
    """Raised when a service is unavailable."""
    
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if service_name:
            context["service_name"] = service_name
        super().__init__(
            message,
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            context=context,
            recovery_suggestions=[
                "Check service status",
                "Restart service",
                "Verify service configuration",
                "Check service logs"
            ],
            **kwargs
        )


class TimeoutError(NetworkError):
    """Raised when operations timeout."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        if operation:
            context["operation"] = operation
            
        super().__init__(
            message,
            error_code=ErrorCode.TIMEOUT_ERROR,
            context=context,
            recovery_suggestions=[
                "Increase timeout value",
                "Check network latency",
                "Optimize operation performance",
                "Use async operations"
            ],
            **kwargs
        )


class AuthenticationError(NetworkError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, username: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if username:
            context["username"] = username
        super().__init__(
            message,
            error_code=ErrorCode.AUTHENTICATION_ERROR,
            context=context,
            recovery_suggestions=[
                "Check credentials",
                "Verify authentication method",
                "Review authentication configuration",
                "Contact administrator"
            ],
            **kwargs
        )


class AuthorizationError(NetworkError):
    """Raised when authorization fails."""
    
    def __init__(
        self,
        message: str,
        username: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if username:
            context["username"] = username
        if resource:
            context["resource"] = resource
        if action:
            context["action"] = action
            
        super().__init__(
            message,
            error_code=ErrorCode.AUTHORIZATION_ERROR,
            context=context,
            recovery_suggestions=[
                "Check user permissions",
                "Verify access control policies",
                "Review resource permissions",
                "Contact administrator"
            ],
            **kwargs
        )


# Monitoring Errors

class MonitoringError(Gaudi3ScaleError):
    """Base class for monitoring-related errors."""
    pass


class MetricsCollectionError(MonitoringError):
    """Raised when metrics collection fails."""
    
    def __init__(self, message: str, metric_name: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if metric_name:
            context["metric_name"] = metric_name
        super().__init__(
            message,
            error_code=ErrorCode.METRICS_COLLECTION_FAILED,
            context=context,
            recovery_suggestions=[
                "Check monitoring agent status",
                "Verify metric definitions",
                "Review collection permissions",
                "Restart monitoring service"
            ],
            **kwargs
        )


class HealthCheckError(MonitoringError):
    """Raised when health checks fail."""
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        context = kwargs.pop("context", {})
        if component:
            context["component"] = component
        super().__init__(
            message,
            error_code=ErrorCode.HEALTH_CHECK_FAILED,
            context=context,
            recovery_suggestions=[
                "Check component status",
                "Review component logs",
                "Restart failed component",
                "Verify configuration"
            ],
            **kwargs
        )


# Utility functions for exception handling

def create_validation_error(
    field_name: str,
    value: Any,
    expected_type: Optional[type] = None,
    valid_range: Optional[tuple] = None,
    valid_values: Optional[list] = None
) -> ParameterValidationError:
    """Create a parameter validation error with standard message format.
    
    Args:
        field_name: Name of the field being validated
        value: The invalid value
        expected_type: Expected type for the value
        valid_range: Valid range for numeric values (min, max)
        valid_values: List of valid values
        
    Returns:
        ParameterValidationError with appropriate message and context
    """
    message_parts = [f"Invalid value for '{field_name}': {value}"]
    
    if expected_type:
        message_parts.append(f"expected {expected_type.__name__}")
    
    if valid_range:
        message_parts.append(f"valid range: {valid_range[0]} to {valid_range[1]}")
    
    if valid_values:
        message_parts.append(f"valid values: {valid_values}")
    
    message = ". ".join(message_parts)
    
    return ParameterValidationError(
        message,
        parameter_name=field_name,
        parameter_value=value,
        expected_type=expected_type
    )


def wrap_exception(
    original_exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    recovery_suggestions: Optional[List[str]] = None
) -> Gaudi3ScaleError:
    """Wrap a generic exception in a Gaudi3ScaleError.
    
    Args:
        original_exception: The original exception to wrap
        context: Additional context information
        recovery_suggestions: Suggested recovery actions
        
    Returns:
        Gaudi3ScaleError wrapping the original exception
    """
    message = f"Wrapped exception: {str(original_exception)}"
    
    return Gaudi3ScaleError(
        message,
        error_code=ErrorCode.INTERNAL_ERROR,
        context=context,
        cause=original_exception,
        recovery_suggestions=recovery_suggestions or [
            "Check original exception details",
            "Review operation logs",
            "Try operation again",
            "Contact support if issue persists"
        ]
    )


# Quantum-Enhanced Exception Classes

class QuantumCircuitError(QuantumEnhancedError):
    """Quantum circuit simulation errors."""
    
    def __init__(self, message: str, circuit_qubits: int = None, **kwargs):
        context = {"circuit_qubits": circuit_qubits} if circuit_qubits else {}
        kwargs.setdefault("quantum_state", QuantumErrorState.SUPERPOSITION)
        super().__init__(message, **kwargs)
        self.context = context


class TaskPlanningError(QuantumEnhancedError):
    """Quantum task planning errors."""
    
    def __init__(self, message: str, task_count: int = None, **kwargs):
        context = {"task_count": task_count} if task_count else {}
        kwargs.setdefault("quantum_state", QuantumErrorState.COHERENT)
        super().__init__(message, **kwargs)
        self.context = context


class ResourceAllocationError(QuantumEnhancedError):
    """Quantum resource allocation errors."""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        context = {"resource_type": resource_type} if resource_type else {}
        kwargs.setdefault("quantum_state", QuantumErrorState.DECOHERENT)
        super().__init__(message, **kwargs)
        self.context = context


class QuantumOptimizationError(QuantumEnhancedError):
    """Quantum annealing optimization errors."""
    
    def __init__(self, message: str, iteration: int = None, temperature: float = None, **kwargs):
        context = {}
        if iteration is not None:
            context["iteration"] = iteration
        if temperature is not None:
            context["temperature"] = temperature
        kwargs.setdefault("quantum_state", QuantumErrorState.SUPERPOSITION)
        super().__init__(message, **kwargs)
        self.context = context


class EntanglementError(QuantumEnhancedError):
    """Quantum entanglement coordination errors."""
    
    def __init__(self, message: str, entity_count: int = None, **kwargs):
        context = {"entity_count": entity_count} if entity_count else {}
        kwargs.setdefault("quantum_state", QuantumErrorState.ENTANGLED)
        super().__init__(message, **kwargs)
        self.context = context


class QuantumDecoherenceError(QuantumEnhancedError):
    """Quantum decoherence detection error."""
    
    def __init__(self, message: str, decoherence_time: float = None, **kwargs):
        context = {"decoherence_time": decoherence_time} if decoherence_time else {}
        kwargs.setdefault("quantum_state", QuantumErrorState.DECOHERENT)
        super().__init__(message, **kwargs)
        self.context = context


# Quantum Error Manager for system-wide error monitoring

class QuantumErrorManager:
    """Manager for tracking and analyzing quantum-enhanced errors."""
    
    def __init__(self):
        self.active_errors: Dict[str, QuantumEnhancedError] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.entanglement_graph: Dict[str, Set[str]] = {}
    
    def register_error(self, error: QuantumEnhancedError):
        """Register a new quantum error in the system."""
        self.active_errors[error.error_id] = error
        
        # Add to history
        self.error_history.append(error.to_quantum_dict())
        
        # Update entanglement graph
        if error.error_id not in self.entanglement_graph:
            self.entanglement_graph[error.error_id] = set()
    
    def entangle_errors(self, error1_id: str, error2_id: str):
        """Create entanglement between two errors."""
        if error1_id in self.active_errors and error2_id in self.active_errors:
            error1 = self.active_errors[error1_id]
            error2 = self.active_errors[error2_id]
            
            error1.entangle_with_error(error2)
            
            # Update entanglement graph
            self.entanglement_graph[error1_id].add(error2_id)
            self.entanglement_graph[error2_id].add(error1_id)
    
    def evolve_error_states(self, time_delta: float = 1.0):
        """Apply quantum evolution to all active errors."""
        for error in self.active_errors.values():
            error.apply_quantum_evolution(time_delta)
            
            # Remove decoherent errors
            if error.quantum_state == QuantumErrorState.DECOHERENT and not error.is_coherent:
                self._cleanup_error(error.error_id)
    
    def _cleanup_error(self, error_id: str):
        """Remove error from active tracking."""
        if error_id in self.active_errors:
            del self.active_errors[error_id]
        
        # Remove from entanglement graph
        if error_id in self.entanglement_graph:
            # Remove connections
            for connected_id in self.entanglement_graph[error_id]:
                if connected_id in self.entanglement_graph:
                    self.entanglement_graph[connected_id].discard(error_id)
            del self.entanglement_graph[error_id]
    
    def get_error_analytics(self) -> Dict[str, Any]:
        """Get analytics on quantum error patterns."""
        total_errors = len(self.active_errors)
        
        if total_errors == 0:
            return {"total_active_errors": 0, "quantum_coherence": 1.0}
        
        # Analyze quantum states
        state_distribution = {}
        coherent_errors = 0
        total_amplitude = 0.0
        
        for error in self.active_errors.values():
            state = error.quantum_state.value
            state_distribution[state] = state_distribution.get(state, 0) + 1
            
            if error.is_coherent:
                coherent_errors += 1
            
            total_amplitude += error.probability_amplitude
        
        # Calculate system-wide quantum coherence
        quantum_coherence = coherent_errors / total_errors if total_errors > 0 else 1.0
        
        # Entanglement metrics
        total_entanglements = sum(len(connections) for connections in self.entanglement_graph.values()) // 2
        max_possible_entanglements = total_errors * (total_errors - 1) // 2
        entanglement_density = total_entanglements / max(1, max_possible_entanglements)
        
        return {
            "total_active_errors": total_errors,
            "coherent_errors": coherent_errors,
            "quantum_coherence": quantum_coherence,
            "state_distribution": state_distribution,
            "total_entanglements": total_entanglements,
            "entanglement_density": entanglement_density,
            "average_amplitude": total_amplitude / total_errors if total_errors > 0 else 0.0,
            "error_history_size": len(self.error_history)
        }


# Global quantum error manager instance
_quantum_error_manager = QuantumErrorManager()