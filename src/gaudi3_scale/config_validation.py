"""Configuration validation and schema checking for Gaudi 3 Scale.

This module provides comprehensive validation for all configuration types
with schema validation, cross-field validation, and security checks.
"""

import json
import os

try:
    import yaml
except ImportError:
    # Fallback for environments without PyYAML
    yaml = None
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Tuple, Set
from enum import Enum
import re

try:
    from jsonschema import validate, ValidationError as JsonSchemaValidationError, Draft7Validator
    from jsonschema.exceptions import SchemaError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from .exceptions import (
    ConfigurationError, InvalidConfigurationError, MissingConfigurationError,
    ModelConfigurationError, DatasetConfigurationError, TrainingConfigurationError,
    ParameterValidationError
)
from .validation import ValidationResult, DataValidator, ConfigurationValidator
from .logging_utils import get_logger

logger = get_logger('config_validation')


class ConfigurationType(Enum):
    """Types of configurations that can be validated."""
    TRAINING = "training"
    MODEL = "model"
    DATASET = "dataset"
    CLUSTER = "cluster"
    OPTIMIZER = "optimizer"
    MONITORING = "monitoring"
    SECURITY = "security"


class ConfigurationSchema:
    """Holds JSON schemas for different configuration types."""
    
    TRAINING_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "batch_size": {
                "type": "integer",
                "minimum": 1,
                "maximum": 512,
                "description": "Training batch size per device"
            },
            "learning_rate": {
                "type": "number",
                "minimum": 1e-8,
                "maximum": 1.0,
                "description": "Initial learning rate"
            },
            "max_epochs": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "description": "Maximum training epochs"
            },
            "max_steps": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": "Maximum training steps"
            },
            "gradient_accumulation_steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 128,
                "description": "Gradient accumulation steps"
            },
            "warmup_steps": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10000,
                "description": "Learning rate warmup steps"
            },
            "weight_decay": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Weight decay coefficient"
            },
            "gradient_clip_val": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 100.0,
                "description": "Gradient clipping value"
            },
            "precision": {
                "type": "string",
                "enum": ["fp32", "fp16", "bf16", "bf16-mixed"],
                "description": "Training precision"
            },
            "optimizer_type": {
                "type": "string",
                "enum": ["adamw", "fused_adamw", "sgd", "adam"],
                "description": "Optimizer type"
            },
            "lr_scheduler_type": {
                "type": "string",
                "enum": ["linear", "cosine", "constant", "polynomial"],
                "description": "Learning rate scheduler type"
            },
            "save_steps": {
                "type": "integer",
                "minimum": 1,
                "description": "Save checkpoint every N steps"
            },
            "eval_steps": {
                "type": "integer",
                "minimum": 1,
                "description": "Evaluate every N steps"
            },
            "logging_steps": {
                "type": "integer",
                "minimum": 1,
                "description": "Log every N steps"
            },
            "output_dir": {
                "type": "string",
                "minLength": 1,
                "description": "Output directory path"
            },
            "use_lazy_mode": {
                "type": "boolean",
                "description": "Enable HPU lazy mode"
            },
            "use_hpu_graphs": {
                "type": "boolean",
                "description": "Enable HPU graphs"
            },
            "mixed_precision": {
                "type": "boolean",
                "description": "Enable mixed precision training"
            }
        },
        "required": ["batch_size", "learning_rate", "max_epochs", "output_dir"],
        "additionalProperties": True
    }
    
    MODEL_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "model_type": {
                "type": "string",
                "enum": ["llama", "bert", "gpt", "t5", "stable_diffusion", "mixtral"],
                "description": "Model architecture type"
            },
            "model_name": {
                "type": "string",
                "minLength": 1,
                "pattern": "^[a-zA-Z0-9_\\-\\.]+$",
                "description": "Model name or identifier"
            },
            "model_size": {
                "type": "string",
                "enum": ["7B", "13B", "30B", "65B", "70B", "175B"],
                "description": "Model size"
            },
            "pretrained_path": {
                "type": ["string", "null"],
                "description": "Path to pretrained model"
            },
            "checkpoint_path": {
                "type": ["string", "null"],
                "description": "Path to model checkpoint"
            },
            "trust_remote_code": {
                "type": "boolean",
                "default": False,
                "description": "Trust remote code execution"
            },
            "gradient_checkpointing": {
                "type": "boolean",
                "default": True,
                "description": "Enable gradient checkpointing"
            },
            "use_cache": {
                "type": "boolean",
                "default": False,
                "description": "Use KV cache during training"
            },
            "vocab_size": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": "Vocabulary size"
            },
            "hidden_size": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": "Hidden dimension size"
            },
            "num_layers": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": "Number of layers"
            },
            "num_heads": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": "Number of attention heads"
            },
            "sequence_length": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 8192,
                "description": "Maximum sequence length"
            }
        },
        "required": ["model_type", "model_name", "model_size"],
        "additionalProperties": True
    }
    
    DATASET_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "dataset_name": {
                "type": "string",
                "minLength": 1,
                "description": "Dataset name or path"
            },
            "dataset_type": {
                "type": "string",
                "enum": ["huggingface", "local", "s3", "gcs", "azure"],
                "default": "huggingface",
                "description": "Dataset type"
            },
            "tokenizer_name": {
                "type": ["string", "null"],
                "description": "Tokenizer name or path"
            },
            "max_length": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8192,
                "default": 2048,
                "description": "Maximum sequence length"
            },
            "padding": {
                "type": "string",
                "enum": ["max_length", "longest", "do_not_pad"],
                "default": "max_length",
                "description": "Padding strategy"
            },
            "truncation": {
                "type": "boolean",
                "default": True,
                "description": "Enable truncation"
            },
            "streaming": {
                "type": "boolean",
                "default": False,
                "description": "Enable streaming dataset"
            },
            "num_proc": {
                "type": "integer",
                "minimum": 1,
                "maximum": 64,
                "default": 8,
                "description": "Number of preprocessing processes"
            },
            "cache_dir": {
                "type": ["string", "null"],
                "description": "Cache directory path"
            },
            "train_split": {
                "type": "string",
                "default": "train",
                "description": "Training split name"
            },
            "validation_split": {
                "type": ["string", "null"],
                "default": "validation",
                "description": "Validation split name"
            },
            "test_split": {
                "type": ["string", "null"],
                "default": "test",
                "description": "Test split name"
            },
            "min_length": {
                "type": "integer",
                "minimum": 1,
                "default": 10,
                "description": "Minimum sequence length"
            }
        },
        "required": ["dataset_name"],
        "additionalProperties": True
    }
    
    CLUSTER_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "cluster_name": {
                "type": "string",
                "minLength": 3,
                "maxLength": 50,
                "pattern": "^[a-zA-Z0-9\\-]+$",
                "description": "Cluster name"
            },
            "provider": {
                "type": "string",
                "enum": ["aws", "azure", "gcp", "onprem"],
                "description": "Cloud provider"
            },
            "region": {
                "type": "string",
                "minLength": 1,
                "description": "Cloud region"
            },
            "enable_monitoring": {
                "type": "boolean",
                "default": True,
                "description": "Enable monitoring stack"
            },
            "enable_spot_instances": {
                "type": "boolean",
                "default": False,
                "description": "Use spot instances"
            },
            "nodes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 100,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Node name"
                        },
                        "instance_type": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Instance type"
                        },
                        "hpu_count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 8,
                            "description": "Number of HPUs per node"
                        },
                        "memory_gb": {
                            "type": "integer",
                            "minimum": 8,
                            "maximum": 1024,
                            "description": "Memory in GB"
                        },
                        "storage_gb": {
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 10000,
                            "description": "Storage in GB"
                        }
                    },
                    "required": ["name", "instance_type", "hpu_count"],
                    "additionalProperties": True
                },
                "description": "Cluster nodes configuration"
            },
            "storage": {
                "type": "object",
                "properties": {
                    "data_volume_size_gb": {
                        "type": "integer",
                        "minimum": 100,
                        "maximum": 10000,
                        "description": "Data volume size in GB"
                    },
                    "backup_enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable backups"
                    }
                },
                "additionalProperties": True
            },
            "network": {
                "type": "object",
                "properties": {
                    "enable_efa": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable Elastic Fabric Adapter"
                    },
                    "vpc_cidr": {
                        "type": "string",
                        "pattern": "^\\d+\\.\\d+\\.\\d+\\.\\d+/\\d+$",
                        "description": "VPC CIDR block"
                    }
                },
                "additionalProperties": True
            },
            "tags": {
                "type": "object",
                "patternProperties": {
                    "^[a-zA-Z0-9_\\-\\.]+$": {
                        "type": "string"
                    }
                },
                "description": "Resource tags"
            }
        },
        "required": ["cluster_name", "provider", "region", "nodes"],
        "additionalProperties": True
    }
    
    OPTIMIZER_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "optimizer_type": {
                "type": "string",
                "enum": ["adamw", "fused_adamw", "sgd", "adam", "lamb"],
                "description": "Optimizer type"
            },
            "learning_rate": {
                "type": "number",
                "minimum": 1e-8,
                "maximum": 1.0,
                "description": "Learning rate"
            },
            "weight_decay": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Weight decay"
            },
            "betas": {
                "type": "array",
                "items": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "minItems": 2,
                "maxItems": 2,
                "description": "Beta coefficients for Adam-style optimizers"
            },
            "eps": {
                "type": "number",
                "minimum": 1e-12,
                "maximum": 1e-3,
                "default": 1e-8,
                "description": "Epsilon for numerical stability"
            },
            "mixed_precision": {
                "type": "boolean",
                "default": True,
                "description": "Enable mixed precision"
            },
            "gradient_scaling": {
                "type": "string",
                "enum": ["none", "static", "dynamic", "auto"],
                "default": "auto",
                "description": "Gradient scaling strategy"
            },
            "use_habana": {
                "type": "boolean",
                "default": True,
                "description": "Use Habana-optimized optimizers"
            }
        },
        "required": ["optimizer_type", "learning_rate"],
        "additionalProperties": True
    }


class ConfigValidator:
    """Comprehensive configuration validator with schema and business logic validation."""
    
    def __init__(
        self,
        use_json_schema: bool = True,
        strict_mode: bool = True,
        enable_cross_validation: bool = True
    ):
        """Initialize configuration validator.
        
        Args:
            use_json_schema: Whether to use JSON schema validation
            strict_mode: Whether to use strict validation (errors vs warnings)
            enable_cross_validation: Whether to enable cross-field validation
        """
        self.use_json_schema = use_json_schema and JSONSCHEMA_AVAILABLE
        self.strict_mode = strict_mode
        self.enable_cross_validation = enable_cross_validation
        self.data_validator = DataValidator()
        self.config_validator = ConfigurationValidator(strict_mode)
        
        if not JSONSCHEMA_AVAILABLE and use_json_schema:
            logger.warning("jsonschema not available, falling back to basic validation")
    
    def validate_config(
        self,
        config: Dict[str, Any],
        config_type: ConfigurationType,
        schema: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate configuration against schema and business rules.
        
        Args:
            config: Configuration dictionary to validate
            config_type: Type of configuration
            schema: Custom JSON schema (optional)
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # JSON Schema validation
        if self.use_json_schema:
            schema_result = self._validate_json_schema(config, config_type, schema)
            result.merge(schema_result)
        
        # Business logic validation
        business_result = self._validate_business_logic(config, config_type)
        result.merge(business_result)
        
        # Cross-field validation
        if self.enable_cross_validation:
            cross_result = self._validate_cross_fields(config, config_type)
            result.merge(cross_result)
        
        # Security validation
        security_result = self._validate_security(config, config_type)
        result.merge(security_result)
        
        # Set sanitized value
        if result.is_valid or not self.strict_mode:
            result.sanitized_value = config.copy()
        
        return result
    
    def _validate_json_schema(
        self,
        config: Dict[str, Any],
        config_type: ConfigurationType,
        custom_schema: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate configuration against JSON schema."""
        result = ValidationResult()
        
        # Get schema
        schema = custom_schema or self._get_schema(config_type)
        if not schema:
            result.add_warning(f"No schema available for {config_type.value} configuration")
            return result
        
        try:
            # Validate schema itself first
            Draft7Validator.check_schema(schema)
            
            # Validate configuration
            validator = Draft7Validator(schema)
            errors = list(validator.iter_errors(config))
            
            for error in errors:
                field_path = '.'.join(str(part) for part in error.absolute_path)
                field_name = field_path or 'root'
                
                if self.strict_mode:
                    result.add_error(f"{field_name}: {error.message}")
                else:
                    result.add_warning(f"{field_name}: {error.message}")
            
        except JsonSchemaValidationError as e:
            result.add_error(f"JSON schema validation failed: {e.message}")
        except SchemaError as e:
            result.add_error(f"Invalid JSON schema: {e.message}")
        except Exception as e:
            result.add_error(f"Schema validation error: {str(e)}")
        
        return result
    
    def _validate_business_logic(
        self,
        config: Dict[str, Any],
        config_type: ConfigurationType
    ) -> ValidationResult:
        """Validate configuration business logic."""
        if config_type == ConfigurationType.TRAINING:
            return self.config_validator.validate_training_config(config)
        elif config_type == ConfigurationType.MODEL:
            return self.config_validator.validate_model_config(config)
        elif config_type == ConfigurationType.CLUSTER:
            return self.config_validator.validate_cluster_config(config)
        else:
            # Generic validation for other types
            return self._validate_generic_config(config)
    
    def _validate_generic_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Generic configuration validation."""
        result = ValidationResult()
        
        # Check for common issues
        for key, value in config.items():
            # Check for None values in required fields
            if value is None and key in ['name', 'type', 'path', 'url']:
                result.add_error(f"Required field '{key}' cannot be None")
            
            # Check for empty strings
            if isinstance(value, str) and not value.strip():
                result.add_error(f"Field '{key}' cannot be empty")
            
            # Check for negative values where they don't make sense
            if isinstance(value, (int, float)) and value < 0 and key in [
                'batch_size', 'epochs', 'timeout', 'port', 'workers', 'size'
            ]:
                result.add_error(f"Field '{key}' cannot be negative")
        
        return result
    
    def _validate_cross_fields(
        self,
        config: Dict[str, Any],
        config_type: ConfigurationType
    ) -> ValidationResult:
        """Validate relationships between fields."""
        result = ValidationResult()
        
        if config_type == ConfigurationType.TRAINING:
            result = self._validate_training_cross_fields(config)
        elif config_type == ConfigurationType.CLUSTER:
            result = self._validate_cluster_cross_fields(config)
        elif config_type == ConfigurationType.OPTIMIZER:
            result = self._validate_optimizer_cross_fields(config)
        
        return result
    
    def _validate_training_cross_fields(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate training configuration cross-field relationships."""
        result = ValidationResult()
        
        # Effective batch size validation
        if 'batch_size' in config and 'gradient_accumulation_steps' in config:
            effective_batch_size = config['batch_size'] * config['gradient_accumulation_steps']
            if effective_batch_size > 2048:
                result.add_warning(
                    f"Very large effective batch size ({effective_batch_size}) may cause memory issues"
                )
            elif effective_batch_size < 8:
                result.add_warning(
                    f"Very small effective batch size ({effective_batch_size}) may slow training"
                )
        
        # Learning rate and batch size relationship
        if 'learning_rate' in config and 'batch_size' in config:
            lr = config['learning_rate']
            batch_size = config['batch_size']
            
            # Rule of thumb: learning rate should roughly scale with sqrt(batch_size)
            if batch_size >= 64 and lr < 1e-4:
                result.add_warning(
                    f"Learning rate {lr} may be too low for batch size {batch_size}"
                )
            elif batch_size <= 16 and lr > 1e-3:
                result.add_warning(
                    f"Learning rate {lr} may be too high for batch size {batch_size}"
                )
        
        # Warmup and total steps relationship
        if 'warmup_steps' in config and 'max_steps' in config:
            if config['warmup_steps'] >= config['max_steps']:
                result.add_error("warmup_steps must be less than max_steps")
            elif config['warmup_steps'] > config['max_steps'] * 0.1:
                result.add_warning(
                    "warmup_steps is more than 10% of max_steps, which is unusual"
                )
        
        # Precision and mixed precision consistency
        if 'precision' in config and 'mixed_precision' in config:
            precision = config['precision']
            mixed_precision = config['mixed_precision']
            
            if precision in ['fp16', 'bf16', 'bf16-mixed'] and not mixed_precision:
                result.add_warning(
                    f"Using {precision} precision but mixed_precision is disabled"
                )
            elif precision == 'fp32' and mixed_precision:
                result.add_warning(
                    "mixed_precision enabled but using fp32 precision"
                )
        
        return result
    
    def _validate_cluster_cross_fields(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate cluster configuration cross-field relationships."""
        result = ValidationResult()
        
        if 'nodes' not in config:
            return result
        
        nodes = config['nodes']
        total_hpus = sum(node.get('hpu_count', 0) for node in nodes)
        
        # Check total HPU count
        if total_hpus == 0:
            result.add_error("Cluster must have at least one HPU")
        elif total_hpus > 512:
            result.add_warning(f"Very large cluster ({total_hpus} HPUs) may be expensive")
        
        # Check node consistency
        instance_types = set(node.get('instance_type') for node in nodes)
        if len(instance_types) > 1:
            result.add_warning("Mixed instance types may cause performance issues")
        
        hpu_counts = set(node.get('hpu_count', 0) for node in nodes)
        if len(hpu_counts) > 1:
            result.add_warning("Mixed HPU counts per node may complicate scheduling")
        
        # Provider-specific validations
        provider = config.get('provider')
        if provider == 'aws':
            result.merge(self._validate_aws_cluster_config(config))
        elif provider == 'azure':
            result.merge(self._validate_azure_cluster_config(config))
        
        return result
    
    def _validate_optimizer_cross_fields(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate optimizer configuration cross-field relationships."""
        result = ValidationResult()
        
        optimizer_type = config.get('optimizer_type')
        
        # Beta coefficients validation for Adam-style optimizers
        if optimizer_type in ['adamw', 'fused_adamw', 'adam'] and 'betas' in config:
            betas = config['betas']
            if len(betas) == 2:
                if betas[0] >= betas[1]:
                    result.add_warning("First beta should typically be smaller than second beta")
                if betas[1] >= 1.0:
                    result.add_error("Second beta should be less than 1.0")
        
        # Learning rate and weight decay relationship
        if 'learning_rate' in config and 'weight_decay' in config:
            lr = config['learning_rate']
            wd = config['weight_decay']
            
            if lr > 1e-2 and wd > 0.1:
                result.add_warning(
                    "High learning rate with high weight decay may cause instability"
                )
        
        # Mixed precision and gradient scaling
        if config.get('mixed_precision') and 'gradient_scaling' in config:
            scaling = config['gradient_scaling']
            if scaling == 'none':
                result.add_warning(
                    "Mixed precision enabled but gradient scaling disabled"
                )
        
        return result
    
    def _validate_aws_cluster_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate AWS-specific cluster configuration."""
        result = ValidationResult()
        
        # Check EFA settings
        if config.get('network', {}).get('enable_efa'):
            # EFA requires specific instance types
            nodes = config.get('nodes', [])
            for node in nodes:
                instance_type = node.get('instance_type', '')
                if not instance_type.startswith(('p4d', 'p4de', 'p5', 'dl2q')):
                    result.add_warning(
                        f"EFA may not be supported on instance type {instance_type}"
                    )
        
        # Check region compatibility
        region = config.get('region', '')
        if 'dl2q' in str(config.get('nodes', [])) and not region.startswith('us-'):
            result.add_warning("DL2q instances may not be available in all regions")
        
        return result
    
    def _validate_azure_cluster_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate Azure-specific cluster configuration."""
        result = ValidationResult()
        
        # Azure-specific validations can be added here
        nodes = config.get('nodes', [])
        for node in nodes:
            instance_type = node.get('instance_type', '')
            if 'HX' not in instance_type:
                result.add_warning(
                    f"Instance type {instance_type} may not support Gaudi HPUs on Azure"
                )
        
        return result
    
    def _validate_security(
        self,
        config: Dict[str, Any],
        config_type: ConfigurationType
    ) -> ValidationResult:
        """Validate configuration for security issues."""
        result = ValidationResult()
        
        # Check for sensitive information in config
        sensitive_keys = {
            'password', 'passwd', 'secret', 'key', 'token', 'credential',
            'api_key', 'access_key', 'private_key', 'auth_token'
        }
        
        def check_sensitive_data(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check key names
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        result.add_warning(
                            f"Potentially sensitive data in config at {current_path}"
                        )
                    
                    check_sensitive_data(value, current_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_sensitive_data(item, f"{path}[{i}]")
            
            elif isinstance(obj, str):
                # Check for patterns that look like secrets
                if re.match(r'^[A-Za-z0-9+/]{20,}={0,2}$', obj):  # Base64-like
                    result.add_warning(f"Possible encoded secret at {path}")
                elif re.match(r'^[a-fA-F0-9]{32,}$', obj):  # Hex string
                    result.add_warning(f"Possible hex-encoded secret at {path}")
        
        check_sensitive_data(config)
        
        # Check for unsafe configurations
        if config_type == ConfigurationType.MODEL:
            if config.get('trust_remote_code'):
                result.add_warning(
                    "trust_remote_code=True is a security risk"
                )
        
        # Check for insecure paths
        for key, value in config.items():
            if isinstance(value, str) and ('path' in key.lower() or 'dir' in key.lower()):
                if value.startswith('/tmp/') or value.startswith('/var/tmp/'):
                    result.add_warning(f"Using potentially insecure temporary path: {value}")
        
        return result
    
    def _get_schema(self, config_type: ConfigurationType) -> Optional[Dict[str, Any]]:
        """Get JSON schema for configuration type."""
        schema_map = {
            ConfigurationType.TRAINING: ConfigurationSchema.TRAINING_SCHEMA,
            ConfigurationType.MODEL: ConfigurationSchema.MODEL_SCHEMA,
            ConfigurationType.DATASET: ConfigurationSchema.DATASET_SCHEMA,
            ConfigurationType.CLUSTER: ConfigurationSchema.CLUSTER_SCHEMA,
            ConfigurationType.OPTIMIZER: ConfigurationSchema.OPTIMIZER_SCHEMA,
        }
        return schema_map.get(config_type)
    
    def validate_config_file(
        self,
        config_path: str,
        config_type: ConfigurationType,
        encoding: str = 'utf-8'
    ) -> ValidationResult:
        """Validate configuration file.
        
        Args:
            config_path: Path to configuration file
            config_type: Type of configuration
            encoding: File encoding
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Validate file path
        path_result = self.data_validator.validate_path(
            config_path,
            'config_file',
            must_exist=True,
            must_be_file=True,
            must_be_readable=True,
            allowed_extensions={'.json', '.yaml', '.yml'}
        )
        result.merge(path_result)
        
        if not result.is_valid:
            return result
        
        # Load and parse configuration file
        try:
            config_file = Path(config_path)
            with open(config_file, 'r', encoding=encoding) as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    if yaml is None:
                        result.add_error("YAML configuration files not supported - PyYAML not installed")
                        return result
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Validate configuration
            validation_result = self.validate_config(config_data, config_type)
            result.merge(validation_result)
            
            if validation_result.sanitized_value:
                result.sanitized_value = validation_result.sanitized_value
            
        except Exception as yaml_error:
            if yaml and "YAMLError" in str(type(yaml_error)):
                result.add_error(f"YAML parsing error: {str(yaml_error)}")
            elif yaml is None and "yaml" in str(yaml_error).lower():
                result.add_error(f"YAML not supported - PyYAML not installed")
            else:
                # Re-raise if not YAML-related
                raise yaml_error
        except json.JSONDecodeError as e:
            result.add_error(f"JSON parsing error: {str(e)}")
        except UnicodeDecodeError as e:
            result.add_error(f"File encoding error: {str(e)}")
        except Exception as e:
            result.add_error(f"Failed to load configuration file: {str(e)}")
        
        return result
    
    def get_validation_report(
        self,
        config: Dict[str, Any],
        config_type: ConfigurationType
    ) -> Dict[str, Any]:
        """Get comprehensive validation report.
        
        Args:
            config: Configuration to validate
            config_type: Type of configuration
            
        Returns:
            Validation report dictionary
        """
        result = self.validate_config(config, config_type)
        
        return {
            'config_type': config_type.value,
            'is_valid': result.is_valid,
            'validation_time': result.timestamp if hasattr(result, 'timestamp') else None,
            'errors': result.errors,
            'warnings': result.warnings,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings),
            'recommendations': self._get_recommendations(config, config_type, result),
            'schema_used': self._get_schema(config_type) is not None,
            'validation_features': {
                'json_schema': self.use_json_schema,
                'business_logic': True,
                'cross_validation': self.enable_cross_validation,
                'security_checks': True
            }
        }
    
    def _get_recommendations(
        self,
        config: Dict[str, Any],
        config_type: ConfigurationType,
        result: ValidationResult
    ) -> List[str]:
        """Get recommendations for improving configuration."""
        recommendations = []
        
        if config_type == ConfigurationType.TRAINING:
            # Performance recommendations
            batch_size = config.get('batch_size', 32)
            if batch_size < 16:
                recommendations.append("Consider increasing batch_size for better GPU utilization")
            
            if not config.get('mixed_precision', False):
                recommendations.append("Enable mixed_precision for faster training and lower memory usage")
            
            if not config.get('gradient_checkpointing', False):
                recommendations.append("Enable gradient_checkpointing to reduce memory usage")
        
        elif config_type == ConfigurationType.CLUSTER:
            node_count = len(config.get('nodes', []))
            if node_count == 1:
                recommendations.append("Consider using multiple nodes for better fault tolerance")
            
            if not config.get('enable_monitoring', True):
                recommendations.append("Enable monitoring for better observability")
        
        # Add recommendations based on validation errors/warnings
        if result.errors:
            recommendations.append("Fix all validation errors before deploying to production")
        
        if result.warnings:
            recommendations.append("Review validation warnings to improve configuration quality")
        
        return recommendations


def create_validator(
    config_type: ConfigurationType,
    strict_mode: bool = True
) -> ConfigValidator:
    """Create a validator for specific configuration type.
    
    Args:
        config_type: Type of configuration to validate
        strict_mode: Whether to use strict validation
        
    Returns:
        Configured ConfigValidator instance
    """
    return ConfigValidator(
        use_json_schema=True,
        strict_mode=strict_mode,
        enable_cross_validation=True
    )


def validate_training_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate training configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        ValidationResult
    """
    validator = create_validator(ConfigurationType.TRAINING)
    return validator.validate_config(config, ConfigurationType.TRAINING)


def validate_model_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        ValidationResult
    """
    validator = create_validator(ConfigurationType.MODEL)
    return validator.validate_config(config, ConfigurationType.MODEL)


def validate_cluster_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate cluster configuration.
    
    Args:
        config: Cluster configuration dictionary
        
    Returns:
        ValidationResult
    """
    validator = create_validator(ConfigurationType.CLUSTER)
    return validator.validate_config(config, ConfigurationType.CLUSTER)