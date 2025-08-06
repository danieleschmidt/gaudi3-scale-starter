"""Input validation and sanitization utilities for Gaudi 3 Scale.

This module provides comprehensive validation and sanitization functions
for all types of inputs, configurations, and data to ensure system
security and reliability.
"""

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from urllib.parse import urlparse
import logging

try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from .exceptions import (
    ParameterValidationError,
    DataValidationError,
    InputSanitizationError,
    ConfigurationError,
    create_validation_error
)

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(
        self,
        is_valid: bool = True,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        sanitized_value: Optional[Any] = None
    ):
        """Initialize validation result.
        
        Args:
            is_valid: Whether the validation passed
            errors: List of error messages
            warnings: List of warning messages
            sanitized_value: The sanitized/cleaned value (if applicable)
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.sanitized_value = sanitized_value
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False


class DataValidator:
    """Comprehensive data validator with sanitization capabilities."""
    
    # Dangerous patterns that should be rejected
    DANGEROUS_PATTERNS = [
        r'<script.*?>.*?</script>',  # Script tags
        r'javascript:',             # JavaScript protocol
        r'on\w+\s*=',              # Event handlers
        r'eval\s*\(',              # Eval function
        r'exec\s*\(',              # Exec function
        r'\$\{.*?\}',              # Template literals
        r'<%.*?%>',                # Server-side templates
        r'\{\{.*?\}\}',            # Template expressions
    ]
    
    # File extensions considered safe for uploads
    SAFE_FILE_EXTENSIONS = {
        '.json', '.yaml', '.yml', '.txt', '.csv', '.tsv', '.log',
        '.py', '.sh', '.md', '.rst', '.cfg', '.conf', '.ini',
        '.pkl', '.pth', '.pt', '.safetensors', '.bin'
    }
    
    # Characters allowed in identifiers (names, IDs, etc.)
    SAFE_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    
    @staticmethod
    def validate_string(
        value: Any,
        field_name: str,
        min_length: int = 0,
        max_length: int = 1000,
        pattern: Optional[re.Pattern] = None,
        allowed_chars: Optional[Set[str]] = None,
        disallowed_chars: Optional[Set[str]] = None,
        sanitize: bool = True
    ) -> ValidationResult:
        """Validate and sanitize string input.
        
        Args:
            value: Value to validate
            field_name: Name of the field being validated
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            pattern: Regex pattern the string must match
            allowed_chars: Set of allowed characters
            disallowed_chars: Set of disallowed characters
            sanitize: Whether to sanitize the input
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Type validation
        if not isinstance(value, str):
            result.add_error(f"{field_name} must be a string, got {type(value).__name__}")
            return result
        
        original_value = value
        
        # Sanitization
        if sanitize:
            value = DataValidator._sanitize_string(value)
            result.sanitized_value = value
        
        # Length validation
        if len(value) < min_length:
            result.add_error(f"{field_name} must be at least {min_length} characters long")
        
        if len(value) > max_length:
            result.add_error(f"{field_name} must not exceed {max_length} characters")
        
        # Pattern validation
        if pattern and not pattern.match(value):
            result.add_error(f"{field_name} does not match required pattern")
        
        # Character validation
        if allowed_chars:
            invalid_chars = set(value) - allowed_chars
            if invalid_chars:
                result.add_error(f"{field_name} contains invalid characters: {invalid_chars}")
        
        if disallowed_chars:
            found_chars = set(value) & disallowed_chars
            if found_chars:
                result.add_error(f"{field_name} contains disallowed characters: {found_chars}")
        
        # Security validation
        security_result = DataValidator._validate_string_security(value, field_name)
        result.merge(security_result)
        
        # Warning if value was modified during sanitization
        if sanitize and original_value != value:
            result.add_warning(f"{field_name} was sanitized during validation")
        
        return result
    
    @staticmethod
    def validate_number(
        value: Any,
        field_name: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        expected_type: type = float,
        allow_zero: bool = True,
        allow_negative: bool = True
    ) -> ValidationResult:
        """Validate numeric input.
        
        Args:
            value: Value to validate
            field_name: Name of the field being validated
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            expected_type: Expected numeric type (int or float)
            allow_zero: Whether zero is allowed
            allow_negative: Whether negative values are allowed
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Type validation and conversion
        try:
            if expected_type == int:
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                elif not isinstance(value, int):
                    value = int(value)
            else:
                value = float(value)
            
            result.sanitized_value = value
            
        except (ValueError, TypeError) as e:
            result.add_error(f"{field_name} must be a valid {expected_type.__name__}: {e}")
            return result
        
        # Range validation
        if min_value is not None and value < min_value:
            result.add_error(f"{field_name} must be at least {min_value}")
        
        if max_value is not None and value > max_value:
            result.add_error(f"{field_name} must not exceed {max_value}")
        
        # Zero validation
        if not allow_zero and value == 0:
            result.add_error(f"{field_name} cannot be zero")
        
        # Negative validation
        if not allow_negative and value < 0:
            result.add_error(f"{field_name} cannot be negative")
        
        # Special numeric validations
        if isinstance(value, float):
            if not DataValidator._is_finite_number(value):
                result.add_error(f"{field_name} must be a finite number")
        
        return result
    
    @staticmethod
    def validate_path(
        value: Any,
        field_name: str,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        must_be_readable: bool = False,
        must_be_writable: bool = False,
        allowed_extensions: Optional[Set[str]] = None,
        max_path_length: int = 4096
    ) -> ValidationResult:
        """Validate file/directory path.
        
        Args:
            value: Path to validate
            field_name: Name of the field being validated
            must_exist: Whether the path must exist
            must_be_file: Whether the path must be a file
            must_be_dir: Whether the path must be a directory
            must_be_readable: Whether the path must be readable
            must_be_writable: Whether the path must be writable
            allowed_extensions: Set of allowed file extensions
            max_path_length: Maximum allowed path length
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Type validation
        if not isinstance(value, (str, Path)):
            result.add_error(f"{field_name} must be a string or Path object")
            return result
        
        # Convert to Path object
        try:
            path = Path(value).resolve()
            result.sanitized_value = str(path)
        except (ValueError, OSError) as e:
            result.add_error(f"{field_name} is not a valid path: {e}")
            return result
        
        # Security validation
        security_result = DataValidator._validate_path_security(path, field_name)
        result.merge(security_result)
        
        if not result.is_valid:
            return result
        
        # Length validation
        if len(str(path)) > max_path_length:
            result.add_error(f"{field_name} path is too long (max {max_path_length} characters)")
        
        # Existence validation
        if must_exist and not path.exists():
            result.add_error(f"{field_name} path does not exist: {path}")
            return result
        
        if path.exists():
            # Type validation
            if must_be_file and not path.is_file():
                result.add_error(f"{field_name} must be a file: {path}")
            
            if must_be_dir and not path.is_dir():
                result.add_error(f"{field_name} must be a directory: {path}")
            
            # Permission validation
            if must_be_readable and not os.access(path, os.R_OK):
                result.add_error(f"{field_name} is not readable: {path}")
            
            if must_be_writable and not os.access(path, os.W_OK):
                result.add_error(f"{field_name} is not writable: {path}")
        
        # Extension validation
        if allowed_extensions and path.suffix.lower() not in allowed_extensions:
            result.add_error(
                f"{field_name} has invalid extension '{path.suffix}', "
                f"allowed: {allowed_extensions}"
            )
        
        return result
    
    @staticmethod
    def validate_url(
        value: Any,
        field_name: str,
        allowed_schemes: Optional[Set[str]] = None,
        require_netloc: bool = True
    ) -> ValidationResult:
        """Validate URL.
        
        Args:
            value: URL to validate
            field_name: Name of the field being validated
            allowed_schemes: Set of allowed URL schemes
            require_netloc: Whether a network location is required
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Type validation
        if not isinstance(value, str):
            result.add_error(f"{field_name} must be a string")
            return result
        
        # Parse URL
        try:
            parsed = urlparse(value)
            result.sanitized_value = value
        except Exception as e:
            result.add_error(f"{field_name} is not a valid URL: {e}")
            return result
        
        # Scheme validation
        if allowed_schemes and parsed.scheme not in allowed_schemes:
            result.add_error(
                f"{field_name} has invalid scheme '{parsed.scheme}', "
                f"allowed: {allowed_schemes}"
            )
        
        # Network location validation
        if require_netloc and not parsed.netloc:
            result.add_error(f"{field_name} must include a network location")
        
        return result
    
    @staticmethod
    def validate_json(value: Any, field_name: str) -> ValidationResult:
        """Validate JSON data.
        
        Args:
            value: JSON string or data to validate
            field_name: Name of the field being validated
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        if isinstance(value, str):
            # Parse JSON string
            try:
                parsed_data = json.loads(value)
                result.sanitized_value = parsed_data
            except json.JSONDecodeError as e:
                result.add_error(f"{field_name} is not valid JSON: {e}")
                return result
        else:
            # Verify data is JSON-serializable
            try:
                json.dumps(value)
                result.sanitized_value = value
            except (TypeError, ValueError) as e:
                result.add_error(f"{field_name} is not JSON-serializable: {e}")
                return result
        
        return result
    
    @staticmethod
    def validate_list(
        value: Any,
        field_name: str,
        min_length: int = 0,
        max_length: int = 1000,
        item_validator: Optional[callable] = None,
        unique_items: bool = False
    ) -> ValidationResult:
        """Validate list/array input.
        
        Args:
            value: List to validate
            field_name: Name of the field being validated
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            item_validator: Function to validate individual items
            unique_items: Whether all items must be unique
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Type validation
        if not isinstance(value, (list, tuple)):
            result.add_error(f"{field_name} must be a list or tuple")
            return result
        
        value = list(value)  # Ensure it's a list
        result.sanitized_value = value
        
        # Length validation
        if len(value) < min_length:
            result.add_error(f"{field_name} must have at least {min_length} items")
        
        if len(value) > max_length:
            result.add_error(f"{field_name} must not have more than {max_length} items")
        
        # Uniqueness validation
        if unique_items and len(value) != len(set(str(item) for item in value)):
            result.add_error(f"{field_name} must contain unique items")
        
        # Item validation
        if item_validator:
            for i, item in enumerate(value):
                item_result = item_validator(item, f"{field_name}[{i}]")
                if isinstance(item_result, ValidationResult):
                    result.merge(item_result)
                elif not item_result:
                    result.add_error(f"{field_name}[{i}] failed validation")
        
        return result
    
    @staticmethod
    def validate_dict(
        value: Any,
        field_name: str,
        required_keys: Optional[Set[str]] = None,
        optional_keys: Optional[Set[str]] = None,
        key_validators: Optional[Dict[str, callable]] = None,
        allow_extra_keys: bool = True
    ) -> ValidationResult:
        """Validate dictionary input.
        
        Args:
            value: Dictionary to validate
            field_name: Name of the field being validated
            required_keys: Set of required keys
            optional_keys: Set of optional keys
            key_validators: Dictionary of key -> validator function mappings
            allow_extra_keys: Whether extra keys are allowed
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Type validation
        if not isinstance(value, dict):
            result.add_error(f"{field_name} must be a dictionary")
            return result
        
        result.sanitized_value = value.copy()
        
        # Required keys validation
        if required_keys:
            missing_keys = required_keys - set(value.keys())
            if missing_keys:
                result.add_error(f"{field_name} missing required keys: {missing_keys}")
        
        # Extra keys validation
        if not allow_extra_keys:
            allowed_keys = (required_keys or set()) | (optional_keys or set())
            extra_keys = set(value.keys()) - allowed_keys
            if extra_keys:
                result.add_error(f"{field_name} contains unexpected keys: {extra_keys}")
        
        # Key validation
        if key_validators:
            for key, validator in key_validators.items():
                if key in value:
                    key_result = validator(value[key], f"{field_name}.{key}")
                    if isinstance(key_result, ValidationResult):
                        result.merge(key_result)
                        if key_result.sanitized_value is not None:
                            result.sanitized_value[key] = key_result.sanitized_value
                    elif not key_result:
                        result.add_error(f"{field_name}.{key} failed validation")
        
        return result
    
    @staticmethod
    def _sanitize_string(value: str) -> str:
        """Sanitize string input by removing/escaping dangerous content.
        
        Args:
            value: String to sanitize
            
        Returns:
            Sanitized string
        """
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Remove dangerous patterns
        for pattern in DataValidator.DANGEROUS_PATTERNS:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE | re.DOTALL)
        
        # Normalize whitespace
        value = re.sub(r'\s+', ' ', value).strip()
        
        # Remove control characters (except newlines and tabs)
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\t')
        
        return value
    
    @staticmethod
    def _validate_string_security(value: str, field_name: str) -> ValidationResult:
        """Validate string for security issues.
        
        Args:
            value: String to validate
            field_name: Name of the field being validated
            
        Returns:
            ValidationResult with security validation outcome
        """
        result = ValidationResult()
        
        # Check for dangerous patterns
        for pattern in DataValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE | re.DOTALL):
                result.add_error(f"{field_name} contains potentially dangerous content")
                break
        
        # Check for null bytes
        if '\x00' in value:
            result.add_error(f"{field_name} contains null bytes")
        
        # Check for excessive length (potential DoS)
        if len(value) > 100000:  # 100KB limit
            result.add_warning(f"{field_name} is very long and may cause performance issues")
        
        return result
    
    @staticmethod
    def _validate_path_security(path: Path, field_name: str) -> ValidationResult:
        """Validate path for security issues.
        
        Args:
            path: Path to validate
            field_name: Name of the field being validated
            
        Returns:
            ValidationResult with security validation outcome
        """
        result = ValidationResult()
        
        path_str = str(path)
        
        # Check for path traversal
        if '..' in path.parts:
            result.add_error(f"{field_name} contains path traversal sequences")
        
        # Check for dangerous paths
        dangerous_paths = ['/etc/', '/proc/', '/sys/', '/dev/', '/root/', '/boot/']
        if any(path_str.startswith(dangerous) for dangerous in dangerous_paths):
            result.add_error(f"{field_name} points to a restricted system directory")
        
        # Check for hidden files (may be intentional, so warning only)
        if path.name.startswith('.') and len(path.name) > 1:
            result.add_warning(f"{field_name} points to a hidden file/directory")
        
        return result
    
    @staticmethod
    def _is_finite_number(value: float) -> bool:
        """Check if a float value is finite (not inf or NaN).
        
        Args:
            value: Float value to check
            
        Returns:
            True if the value is finite
        """
        import math
        return math.isfinite(value)


class ConfigurationValidator:
    """Validates configuration objects and files."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize configuration validator.
        
        Args:
            strict_mode: Whether to use strict validation (errors vs warnings)
        """
        self.strict_mode = strict_mode
        self.validator = DataValidator()
    
    def validate_training_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate training configuration.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Define required and optional fields with their validators
        field_validators = {
            'batch_size': lambda v, n: self.validator.validate_number(
                v, n, min_value=1, max_value=512, expected_type=int
            ),
            'learning_rate': lambda v, n: self.validator.validate_number(
                v, n, min_value=1e-8, max_value=1.0, allow_zero=False
            ),
            'max_epochs': lambda v, n: self.validator.validate_number(
                v, n, min_value=1, max_value=1000, expected_type=int, allow_zero=False
            ),
            'gradient_accumulation_steps': lambda v, n: self.validator.validate_number(
                v, n, min_value=1, max_value=128, expected_type=int, allow_zero=False
            ),
            'warmup_steps': lambda v, n: self.validator.validate_number(
                v, n, min_value=0, max_value=10000, expected_type=int
            ),
            'weight_decay': lambda v, n: self.validator.validate_number(
                v, n, min_value=0.0, max_value=1.0
            ),
            'gradient_clip_val': lambda v, n: self.validator.validate_number(
                v, n, min_value=0.0, max_value=100.0
            ),
            'output_dir': lambda v, n: self.validator.validate_path(
                v, n, must_be_dir=False, must_be_writable=False
            )
        }
        
        # Validate each field
        for field, validator in field_validators.items():
            if field in config:
                field_result = validator(config[field], field)
                result.merge(field_result)
        
        # Cross-field validation
        cross_validation_result = self._validate_training_config_cross_fields(config)
        result.merge(cross_validation_result)
        
        return result
    
    def validate_model_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Model name validation
        if 'model_name' in config:
            name_result = self.validator.validate_string(
                config['model_name'], 'model_name',
                min_length=1, max_length=100,
                pattern=self.validator.SAFE_IDENTIFIER_PATTERN
            )
            result.merge(name_result)
        
        # Model size validation
        if 'model_size' in config:
            valid_sizes = {'7B', '13B', '30B', '65B', '70B', '175B'}
            if config['model_size'] not in valid_sizes:
                result.add_error(f"model_size must be one of {valid_sizes}")
        
        # Model path validation
        if 'pretrained_path' in config:
            path_result = self.validator.validate_path(
                config['pretrained_path'], 'pretrained_path',
                must_exist=False, must_be_readable=False
            )
            result.merge(path_result)
        
        return result
    
    def validate_cluster_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate cluster configuration.
        
        Args:
            config: Cluster configuration dictionary
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Required fields
        required_fields = {'cluster_name', 'provider', 'region'}
        missing_fields = required_fields - set(config.keys())
        if missing_fields:
            result.add_error(f"Missing required cluster config fields: {missing_fields}")
        
        # Cluster name validation
        if 'cluster_name' in config:
            name_result = self.validator.validate_string(
                config['cluster_name'], 'cluster_name',
                min_length=3, max_length=50,
                pattern=re.compile(r'^[a-zA-Z0-9\-]+$')
            )
            result.merge(name_result)
        
        # Provider validation
        if 'provider' in config:
            valid_providers = {'aws', 'azure', 'gcp', 'onprem'}
            if config['provider'] not in valid_providers:
                result.add_error(f"provider must be one of {valid_providers}")
        
        # Nodes validation
        if 'nodes' in config:
            nodes_result = self.validator.validate_list(
                config['nodes'], 'nodes',
                min_length=1, max_length=100,
                item_validator=self._validate_node_config
            )
            result.merge(nodes_result)
        
        return result
    
    def _validate_training_config_cross_fields(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate cross-field relationships in training config.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            ValidationResult with cross-validation outcome
        """
        result = ValidationResult()
        
        # Effective batch size validation
        if 'batch_size' in config and 'gradient_accumulation_steps' in config:
            effective_batch_size = config['batch_size'] * config['gradient_accumulation_steps']
            if effective_batch_size > 2048:
                result.add_warning(
                    f"Effective batch size ({effective_batch_size}) is very large "
                    "and may cause memory issues"
                )
        
        # Warmup steps vs max steps validation
        if 'warmup_steps' in config and 'max_steps' in config:
            if config['warmup_steps'] >= config['max_steps']:
                result.add_error("warmup_steps must be less than max_steps")
        
        # Learning rate validation based on model size
        if 'learning_rate' in config and 'model_size' in config:
            lr = config['learning_rate']
            model_size = config.get('model_size', '7B')
            
            # Suggested learning rates for different model sizes
            lr_suggestions = {
                '7B': (1e-5, 5e-4),
                '13B': (5e-6, 3e-4),
                '70B': (1e-6, 1e-4)
            }
            
            if model_size in lr_suggestions:
                min_lr, max_lr = lr_suggestions[model_size]
                if lr < min_lr:
                    result.add_warning(
                        f"Learning rate {lr} may be too low for {model_size} model "
                        f"(suggested range: {min_lr} - {max_lr})"
                    )
                elif lr > max_lr:
                    result.add_warning(
                        f"Learning rate {lr} may be too high for {model_size} model "
                        f"(suggested range: {min_lr} - {max_lr})"
                    )
        
        return result
    
    def _validate_node_config(self, node_config: Any, field_name: str) -> ValidationResult:
        """Validate individual node configuration.
        
        Args:
            node_config: Node configuration dictionary
            field_name: Name of the field being validated
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        if not isinstance(node_config, dict):
            result.add_error(f"{field_name} must be a dictionary")
            return result
        
        # Required node fields
        required_fields = {'name', 'instance_type', 'hpu_count'}
        missing_fields = required_fields - set(node_config.keys())
        if missing_fields:
            result.add_error(f"{field_name} missing required fields: {missing_fields}")
        
        # HPU count validation
        if 'hpu_count' in node_config:
            hpu_result = self.validator.validate_number(
                node_config['hpu_count'], f"{field_name}.hpu_count",
                min_value=1, max_value=8, expected_type=int, allow_zero=False
            )
            result.merge(hpu_result)
        
        return result


class InputSanitizer:
    """Sanitizes various types of input to prevent security issues."""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to be safe for filesystem operations.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        if not isinstance(filename, str):
            raise InputSanitizationError("Filename must be a string")
        
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        dangerous_chars = '<>:"|?*\x00'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure filename is not empty
        if not filename:
            filename = 'sanitized_file'
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename
    
    @staticmethod
    def sanitize_command_arg(arg: str) -> str:
        """Sanitize command line argument.
        
        Args:
            arg: Original argument
            
        Returns:
            Sanitized argument
        """
        if not isinstance(arg, str):
            raise InputSanitizationError("Command argument must be a string")
        
        # Remove null bytes
        arg = arg.replace('\x00', '')
        
        # Remove command injection patterns
        dangerous_patterns = [';', '&&', '||', '|', '`', '$', '(', ')']
        for pattern in dangerous_patterns:
            if pattern in arg:
                raise InputSanitizationError(f"Command argument contains dangerous pattern: {pattern}")
        
        return arg
    
    @staticmethod
    def sanitize_log_message(message: str) -> str:
        """Sanitize log message to prevent log injection.
        
        Args:
            message: Original log message
            
        Returns:
            Sanitized log message
        """
        if not isinstance(message, str):
            message = str(message)
        
        # Remove control characters except newlines and tabs
        message = ''.join(char for char in message if ord(char) >= 32 or char in '\n\t')
        
        # Limit message length
        if len(message) > 10000:
            message = message[:9997] + '...'
        
        # Escape ANSI sequences
        message = re.sub(r'\x1b\[[0-9;]*m', '', message)
        
        return message


def validate_pydantic_model(
    data: Dict[str, Any],
    model_class: type,
    field_name: str = "config"
) -> ValidationResult:
    """Validate data against a Pydantic model.
    
    Args:
        data: Data to validate
        model_class: Pydantic model class
        field_name: Name of the field being validated
        
    Returns:
        ValidationResult with validation outcome
    """
    result = ValidationResult()
    
    if not PYDANTIC_AVAILABLE:
        result.add_warning("Pydantic not available, skipping model validation")
        result.sanitized_value = data
        return result
    
    try:
        validated_model = model_class(**data)
        result.sanitized_value = validated_model.dict()
    except PydanticValidationError as e:
        for error in e.errors():
            field_path = '.'.join(str(loc) for loc in error['loc'])
            result.add_error(f"{field_name}.{field_path}: {error['msg']}")
    except Exception as e:
        result.add_error(f"Unexpected validation error for {field_name}: {str(e)}")
    
    return result


def create_comprehensive_validator(strict_mode: bool = True) -> ConfigurationValidator:
    """Create a comprehensive validator instance.
    
    Args:
        strict_mode: Whether to use strict validation
        
    Returns:
        ConfigurationValidator instance
    """
    return ConfigurationValidator(strict_mode=strict_mode)