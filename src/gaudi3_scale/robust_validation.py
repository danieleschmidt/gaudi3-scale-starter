"""Enhanced validation framework for Generation 2 robustness.

This module provides comprehensive validation capabilities including:
- Input/output validation with detailed error reporting
- Schema validation for configuration files
- Runtime validation for training parameters
- Data integrity checks
- Performance validation
- Security validation
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .logging_utils import get_logger
from .exceptions import ValidationError, ConfigurationError, create_validation_error

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ValidationCategory(Enum):
    """Categories of validation."""
    CONFIGURATION = "configuration"
    DATA = "data"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SYSTEM = "system"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    valid: bool
    category: ValidationCategory
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    error_code: Optional[str] = None


class RobustValidator:
    """Enhanced validation framework with comprehensive checks."""
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_performance_validation: bool = True,
        enable_security_validation: bool = True,
        custom_validators: Optional[Dict[str, Callable]] = None
    ):
        """Initialize robust validator.
        
        Args:
            validation_level: Strictness level for validation
            enable_performance_validation: Enable performance checks
            enable_security_validation: Enable security checks
            custom_validators: Custom validation functions
        """
        self.validation_level = validation_level
        self.enable_performance_validation = enable_performance_validation
        self.enable_security_validation = enable_security_validation
        self.custom_validators = custom_validators or {}
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'by_category': {cat.value: {'passed': 0, 'failed': 0} for cat in ValidationCategory},
            'by_level': {lvl.value: {'passed': 0, 'failed': 0} for lvl in ValidationLevel}
        }
        
        logger.info(f"RobustValidator initialized with {validation_level.value} level")
        
    def validate_training_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Comprehensive training configuration validation."""
        self._update_stats('total_validations')
        
        try:
            # Basic structure validation
            required_fields = ['model_name', 'batch_size', 'learning_rate', 'max_epochs']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                return self._create_failed_result(
                    ValidationCategory.CONFIGURATION,
                    f"Missing required fields: {', '.join(missing_fields)}",
                    details={'missing_fields': missing_fields},
                    suggestions=[f"Add '{field}' to configuration" for field in missing_fields],
                    error_code="MISSING_REQUIRED_FIELDS"
                )
            
            # Type validation
            type_checks = [
                ('model_name', str, config.get('model_name')),
                ('batch_size', int, config.get('batch_size')),
                ('learning_rate', (int, float), config.get('learning_rate')),
                ('max_epochs', int, config.get('max_epochs'))
            ]
            
            for field_name, expected_type, value in type_checks:
                if not isinstance(value, expected_type):
                    return self._create_failed_result(
                        ValidationCategory.CONFIGURATION,
                        f"Field '{field_name}' must be {expected_type.__name__ if not isinstance(expected_type, tuple) else ' or '.join(t.__name__ for t in expected_type)}",
                        details={'field': field_name, 'expected_type': str(expected_type), 'actual_type': type(value).__name__},
                        error_code="INVALID_TYPE"
                    )
            
            # Range validation
            range_checks = [
                ('batch_size', config.get('batch_size'), 1, 10000, "Batch size must be between 1 and 10000"),
                ('learning_rate', config.get('learning_rate'), 1e-8, 1.0, "Learning rate must be between 1e-8 and 1.0"),
                ('max_epochs', config.get('max_epochs'), 1, 10000, "Max epochs must be between 1 and 10000")
            ]
            
            for field_name, value, min_val, max_val, message in range_checks:
                if not (min_val <= value <= max_val):
                    return self._create_failed_result(
                        ValidationCategory.CONFIGURATION,
                        message,
                        details={'field': field_name, 'value': value, 'min': min_val, 'max': max_val},
                        suggestions=[f"Set {field_name} to a value between {min_val} and {max_val}"],
                        error_code="VALUE_OUT_OF_RANGE"
                    )
            
            # Advanced validation based on level
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                advanced_result = self._validate_advanced_training_config(config)
                if not advanced_result.valid:
                    return advanced_result
                    
            # Performance validation
            if self.enable_performance_validation:
                perf_result = self._validate_performance_config(config)
                if not perf_result.valid:
                    return perf_result
                    
            # Security validation
            if self.enable_security_validation:
                sec_result = self._validate_security_config(config)
                if not sec_result.valid:
                    return sec_result
                    
            return self._create_passed_result(
                ValidationCategory.CONFIGURATION,
                "Training configuration is valid",
                details={'validated_fields': list(config.keys())}
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return self._create_failed_result(
                ValidationCategory.CONFIGURATION,
                f"Validation failed with error: {str(e)}",
                error_code="VALIDATION_ERROR"
            )
            
    def _validate_advanced_training_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Advanced configuration validation for strict levels."""
        
        # Validate model name format
        model_name = config.get('model_name', '')
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-_]*[a-zA-Z0-9])?$', model_name):
            return self._create_failed_result(
                ValidationCategory.CONFIGURATION,
                "Model name must contain only alphanumeric characters, hyphens, and underscores",
                details={'model_name': model_name},
                suggestions=["Use only letters, numbers, hyphens, and underscores in model name"],
                error_code="INVALID_MODEL_NAME"
            )
        
        # Validate precision if specified
        if 'precision' in config:
            valid_precisions = ['fp32', 'fp16', 'bf16', 'mixed']
            if config['precision'] not in valid_precisions:
                return self._create_failed_result(
                    ValidationCategory.CONFIGURATION,
                    f"Precision must be one of: {', '.join(valid_precisions)}",
                    details={'precision': config['precision'], 'valid_options': valid_precisions},
                    error_code="INVALID_PRECISION"
                )
        
        # Validate optimizer settings if specified
        if 'optimizer' in config:
            opt_result = self._validate_optimizer_config(config['optimizer'])
            if not opt_result.valid:
                return opt_result
                
        return self._create_passed_result(
            ValidationCategory.CONFIGURATION,
            "Advanced configuration validation passed"
        )
        
    def _validate_optimizer_config(self, optimizer_config: Dict[str, Any]) -> ValidationResult:
        """Validate optimizer configuration."""
        if not isinstance(optimizer_config, dict):
            return self._create_failed_result(
                ValidationCategory.CONFIGURATION,
                "Optimizer configuration must be a dictionary",
                error_code="INVALID_OPTIMIZER_CONFIG"
            )
            
        # Validate optimizer type
        if 'type' in optimizer_config:
            valid_optimizers = ['adamw', 'adam', 'sgd', 'rmsprop', 'adagrad']
            if optimizer_config['type'].lower() not in valid_optimizers:
                return self._create_failed_result(
                    ValidationCategory.CONFIGURATION,
                    f"Optimizer type must be one of: {', '.join(valid_optimizers)}",
                    details={'optimizer_type': optimizer_config['type']},
                    error_code="INVALID_OPTIMIZER_TYPE"
                )
        
        # Validate optimizer parameters
        if 'weight_decay' in optimizer_config:
            weight_decay = optimizer_config['weight_decay']
            if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
                return self._create_failed_result(
                    ValidationCategory.CONFIGURATION,
                    "Weight decay must be a non-negative number",
                    details={'weight_decay': weight_decay},
                    error_code="INVALID_WEIGHT_DECAY"
                )
                
        return self._create_passed_result(
            ValidationCategory.CONFIGURATION,
            "Optimizer configuration is valid"
        )
        
    def _validate_performance_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration for performance implications."""
        warnings = []
        
        # Check for performance-impacting settings
        batch_size = config.get('batch_size', 32)
        if batch_size < 8:
            warnings.append("Small batch size may reduce training efficiency")
            
        # Check learning rate for stability
        learning_rate = config.get('learning_rate', 0.001)
        if learning_rate > 0.01:
            warnings.append("High learning rate may cause training instability")
        elif learning_rate < 1e-6:
            warnings.append("Very low learning rate may slow convergence")
            
        # Check epoch count for efficiency
        max_epochs = config.get('max_epochs', 10)
        if max_epochs > 1000:
            warnings.append("Very high epoch count may indicate inefficient training")
            
        if warnings and self.validation_level == ValidationLevel.PARANOID:
            return self._create_failed_result(
                ValidationCategory.PERFORMANCE,
                "Performance validation failed",
                details={'warnings': warnings},
                suggestions=["Review configuration for optimal performance"],
                error_code="PERFORMANCE_WARNINGS"
            )
            
        return self._create_passed_result(
            ValidationCategory.PERFORMANCE,
            "Performance validation passed",
            details={'warnings': warnings} if warnings else None
        )
        
    def _validate_security_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration for security issues."""
        security_issues = []
        
        # Check for potentially unsafe paths
        for key, value in config.items():
            if isinstance(value, str) and ('output_dir' in key or 'path' in key.lower()):
                if '..' in value or value.startswith('/'):
                    security_issues.append(f"Potentially unsafe path in {key}: {value}")
                    
        # Check for sensitive information in model name
        model_name = config.get('model_name', '').lower()
        sensitive_keywords = ['password', 'secret', 'key', 'token', 'private']
        if any(keyword in model_name for keyword in sensitive_keywords):
            security_issues.append("Model name may contain sensitive information")
            
        if security_issues:
            return self._create_failed_result(
                ValidationCategory.SECURITY,
                "Security validation failed",
                details={'issues': security_issues},
                suggestions=["Review configuration for security best practices"],
                error_code="SECURITY_ISSUES"
            )
            
        return self._create_passed_result(
            ValidationCategory.SECURITY,
            "Security validation passed"
        )
        
    def validate_data_integrity(self, data: Any, schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data integrity and format."""
        self._update_stats('total_validations')
        
        try:
            if data is None:
                return self._create_failed_result(
                    ValidationCategory.DATA,
                    "Data cannot be None",
                    error_code="NULL_DATA"
                )
                
            # Basic data checks
            if isinstance(data, (list, tuple)):
                if len(data) == 0:
                    return self._create_failed_result(
                        ValidationCategory.DATA,
                        "Data list cannot be empty",
                        error_code="EMPTY_DATA"
                    )
                    
            # Schema validation if provided
            if schema:
                schema_result = self._validate_against_schema(data, schema)
                if not schema_result.valid:
                    return schema_result
                    
            return self._create_passed_result(
                ValidationCategory.DATA,
                "Data integrity validation passed",
                details={'data_type': type(data).__name__, 'data_size': len(data) if hasattr(data, '__len__') else 'unknown'}
            )
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return self._create_failed_result(
                ValidationCategory.DATA,
                f"Data validation failed: {str(e)}",
                error_code="DATA_VALIDATION_ERROR"
            )
            
    def _validate_against_schema(self, data: Any, schema: Dict[str, Any]) -> ValidationResult:
        """Validate data against a schema."""
        # Simple schema validation implementation
        try:
            if 'type' in schema:
                expected_type = schema['type']
                if expected_type == 'dict' and not isinstance(data, dict):
                    return self._create_failed_result(
                        ValidationCategory.DATA,
                        f"Expected dict, got {type(data).__name__}",
                        error_code="SCHEMA_TYPE_MISMATCH"
                    )
                elif expected_type == 'list' and not isinstance(data, list):
                    return self._create_failed_result(
                        ValidationCategory.DATA,
                        f"Expected list, got {type(data).__name__}",
                        error_code="SCHEMA_TYPE_MISMATCH"
                    )
                    
            if 'required_keys' in schema and isinstance(data, dict):
                missing_keys = [key for key in schema['required_keys'] if key not in data]
                if missing_keys:
                    return self._create_failed_result(
                        ValidationCategory.DATA,
                        f"Missing required keys: {', '.join(missing_keys)}",
                        details={'missing_keys': missing_keys},
                        error_code="MISSING_KEYS"
                    )
                    
            return self._create_passed_result(
                ValidationCategory.DATA,
                "Schema validation passed"
            )
            
        except Exception as e:
            return self._create_failed_result(
                ValidationCategory.DATA,
                f"Schema validation error: {str(e)}",
                error_code="SCHEMA_VALIDATION_ERROR"
            )
            
    def validate_system_resources(self) -> ValidationResult:
        """Validate system resources and readiness."""
        self._update_stats('total_validations')
        
        try:
            import psutil
            
            # Check available memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return self._create_failed_result(
                    ValidationCategory.SYSTEM,
                    f"High memory usage: {memory.percent}%",
                    details={'memory_percent': memory.percent, 'available_gb': memory.available / (1024**3)},
                    suggestions=["Free up memory before starting training"],
                    error_code="HIGH_MEMORY_USAGE"
                )
                
            # Check available disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                return self._create_failed_result(
                    ValidationCategory.SYSTEM,
                    f"High disk usage: {disk.percent}%",
                    details={'disk_percent': disk.percent, 'free_gb': disk.free / (1024**3)},
                    suggestions=["Free up disk space before starting training"],
                    error_code="HIGH_DISK_USAGE"
                )
                
            # Check CPU load
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:
                return self._create_failed_result(
                    ValidationCategory.SYSTEM,
                    f"High CPU usage: {cpu_percent}%",
                    details={'cpu_percent': cpu_percent},
                    suggestions=["Wait for CPU load to decrease or stop other processes"],
                    error_code="HIGH_CPU_USAGE"
                )
                
            return self._create_passed_result(
                ValidationCategory.SYSTEM,
                "System resources validation passed",
                details={
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'cpu_percent': cpu_percent
                }
            )
            
        except ImportError:
            logger.warning("psutil not available, skipping system resource validation")
            return self._create_passed_result(
                ValidationCategory.SYSTEM,
                "System validation skipped (psutil not available)"
            )
        except Exception as e:
            logger.error(f"System validation error: {e}")
            return self._create_failed_result(
                ValidationCategory.SYSTEM,
                f"System validation failed: {str(e)}",
                error_code="SYSTEM_VALIDATION_ERROR"
            )
            
    def validate_file_path(self, path: Union[str, Path], must_exist: bool = False, must_be_writable: bool = False) -> ValidationResult:
        """Validate file path accessibility and permissions."""
        self._update_stats('total_validations')
        
        try:
            path_obj = Path(path)
            
            # Check if path exists when required
            if must_exist and not path_obj.exists():
                return self._create_failed_result(
                    ValidationCategory.SYSTEM,
                    f"Path does not exist: {path}",
                    details={'path': str(path)},
                    suggestions=["Create the path or check the path spelling"],
                    error_code="PATH_NOT_FOUND"
                )
                
            # Check parent directory exists for new files
            if not must_exist and not path_obj.parent.exists():
                return self._create_failed_result(
                    ValidationCategory.SYSTEM,
                    f"Parent directory does not exist: {path_obj.parent}",
                    details={'parent_path': str(path_obj.parent)},
                    suggestions=["Create parent directory first"],
                    error_code="PARENT_PATH_NOT_FOUND"
                )
                
            # Check write permissions
            if must_be_writable:
                if path_obj.exists() and not path_obj.is_file():
                    return self._create_failed_result(
                        ValidationCategory.SYSTEM,
                        f"Path is not a file: {path}",
                        error_code="NOT_A_FILE"
                    )
                    
                # Test write access
                test_path = path_obj if path_obj.exists() else path_obj.parent
                if not test_path.exists() or not os.access(test_path, os.W_OK):
                    return self._create_failed_result(
                        ValidationCategory.SYSTEM,
                        f"No write permission for path: {path}",
                        details={'path': str(path)},
                        suggestions=["Check file permissions or change output directory"],
                        error_code="NO_WRITE_PERMISSION"
                    )
                    
            return self._create_passed_result(
                ValidationCategory.SYSTEM,
                f"Path validation passed: {path}",
                details={'path': str(path), 'exists': path_obj.exists()}
            )
            
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return self._create_failed_result(
                ValidationCategory.SYSTEM,
                f"Path validation failed: {str(e)}",
                error_code="PATH_VALIDATION_ERROR"
            )
            
    def run_comprehensive_validation(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Run all available validations."""
        results = []
        
        logger.info("Running comprehensive validation suite")
        
        # Configuration validation
        results.append(self.validate_training_config(config))
        
        # System validation
        results.append(self.validate_system_resources())
        
        # File path validation
        if 'output_dir' in config:
            results.append(self.validate_file_path(config['output_dir'], must_be_writable=True))
            
        # Custom validators
        for name, validator in self.custom_validators.items():
            try:
                result = validator(config)
                if isinstance(result, ValidationResult):
                    results.append(result)
                else:
                    logger.warning(f"Custom validator {name} returned invalid result type")
            except Exception as e:
                logger.error(f"Custom validator {name} failed: {e}")
                results.append(self._create_failed_result(
                    ValidationCategory.CONFIGURATION,
                    f"Custom validator {name} failed: {str(e)}",
                    error_code="CUSTOM_VALIDATOR_ERROR"
                ))
                
        # Log summary
        passed = sum(1 for r in results if r.valid)
        failed = len(results) - passed
        
        logger.info(f"Comprehensive validation completed: {passed} passed, {failed} failed")
        
        return results
        
    def _create_passed_result(self, category: ValidationCategory, message: str, details: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Create a passed validation result."""
        self._update_stats('passed_validations')
        self._update_stats(f'by_category.{category.value}.passed')
        self._update_stats(f'by_level.{self.validation_level.value}.passed')
        
        return ValidationResult(
            valid=True,
            category=category,
            level=self.validation_level,
            message=message,
            details=details
        )
        
    def _create_failed_result(
        self,
        category: ValidationCategory,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        error_code: Optional[str] = None
    ) -> ValidationResult:
        """Create a failed validation result."""
        self._update_stats('failed_validations')
        self._update_stats(f'by_category.{category.value}.failed')
        self._update_stats(f'by_level.{self.validation_level.value}.failed')
        
        return ValidationResult(
            valid=False,
            category=category,
            level=self.validation_level,
            message=message,
            details=details,
            suggestions=suggestions,
            error_code=error_code
        )
        
    def _update_stats(self, stat_path: str):
        """Update validation statistics."""
        keys = stat_path.split('.')
        current = self.validation_stats
        
        for key in keys[:-1]:
            current = current[key]
            
        current[keys[-1]] += 1
        
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()
        
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'by_category': {cat.value: {'passed': 0, 'failed': 0} for cat in ValidationCategory},
            'by_level': {lvl.value: {'passed': 0, 'failed': 0} for lvl in ValidationLevel}
        }
        

# Helper functions
def validate_config_file(config_path: Union[str, Path]) -> ValidationResult:
    """Validate a configuration file."""
    validator = RobustValidator()
    
    try:
        with open(config_path, 'r') as f:
            if str(config_path).endswith('.json'):
                config = json.load(f)
            else:
                # Assume YAML
                import yaml
                config = yaml.safe_load(f)
                
        return validator.validate_training_config(config)
        
    except FileNotFoundError:
        return ValidationResult(
            valid=False,
            category=ValidationCategory.SYSTEM,
            level=ValidationLevel.STANDARD,
            message=f"Configuration file not found: {config_path}",
            error_code="CONFIG_FILE_NOT_FOUND"
        )
    except json.JSONDecodeError as e:
        return ValidationResult(
            valid=False,
            category=ValidationCategory.CONFIGURATION,
            level=ValidationLevel.STANDARD,
            message=f"Invalid JSON in configuration file: {str(e)}",
            error_code="INVALID_JSON"
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            category=ValidationCategory.CONFIGURATION,
            level=ValidationLevel.STANDARD,
            message=f"Failed to validate configuration file: {str(e)}",
            error_code="CONFIG_VALIDATION_ERROR"
        )


def quick_validate(config: Dict[str, Any]) -> bool:
    """Quick validation for basic use cases."""
    validator = RobustValidator(ValidationLevel.BASIC)
    result = validator.validate_training_config(config)
    return result.valid