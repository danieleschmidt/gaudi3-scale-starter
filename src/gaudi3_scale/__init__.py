"""Gaudi 3 Scale Starter - Production Infrastructure for Intel Gaudi 3 HPU Clusters.

This package provides comprehensive tools and utilities for deploying and managing
large-scale machine learning training on Intel Gaudi 3 accelerators with
enterprise-grade reliability, monitoring, and error handling.

Generation 3 Features (Distributed Deployment):
- Multi-node distributed training coordination
- Cluster management and node discovery
- Service mesh and communication protocols
- Distributed storage and data management
- Fault tolerance and failover mechanisms
- Distributed monitoring and observability
- Deployment orchestration and automation
- Distributed configuration management

Generation 3 Features (Performance Optimization):
- Advanced multi-level caching (L1/L2 with distributed cache)
- High-performance connection pooling and resource management
- Async/await patterns for optimal I/O performance
- Comprehensive performance monitoring and profiling
- Intelligent auto-scaling and load balancing
- Priority-based batch processing and queue management
- Memory optimization and garbage collection tuning
- Performance benchmarking and testing frameworks

Generation 2 Features:
- Comprehensive error handling and custom exception hierarchy
- Input validation and sanitization for security
- Structured logging with performance monitoring
- Health checks and system monitoring
- Configuration validation with JSON schema support
- Retry logic with multiple backoff strategies
- Production-ready deployment capabilities
- Enterprise-grade security hardening and compliance
- Advanced threat detection and incident response
- Secure configuration and secrets management
- Rate limiting and DoS protection
- Comprehensive audit logging and monitoring
"""

import logging
import warnings

__version__ = "0.5.0"  # Updated for Generation 3 Distributed Deployment
__author__ = "Daniel Schmidt"

# Set up logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import optional dependency helper first
from . import optional_deps

# Core components (always available)
from .accelerator import GaudiAccelerator
from .optimizer import GaudiOptimizer
from .trainer import GaudiTrainer

# Enhanced reliability components (always available)
from . import exceptions
from . import validation
from . import logging_utils
from . import retry_utils
from . import health_checks
from . import config_validation

# Security components (Generation 2 - Enterprise Security)
from . import security

# Performance optimization components (Generation 3) - import conditionally
_optional_modules = {}

# Cache module
try:
    from . import cache
    _optional_modules['cache'] = cache
except ImportError as e:
    logger.debug(f"Cache module import failed: {e}")
    cache = None

# Database module
try:
    from . import database
    _optional_modules['database'] = database
except ImportError as e:
    logger.debug(f"Database module import failed: {e}")
    database = None

# Services module
try:
    from . import services
    _optional_modules['services'] = services
except ImportError as e:
    logger.debug(f"Services module import failed: {e}")
    services = None

# Algorithms module
try:
    from . import algorithms
    _optional_modules['algorithms'] = algorithms
except ImportError as e:
    logger.debug(f"Algorithms module import failed: {e}")
    algorithms = None

# Monitoring module
try:
    from . import monitoring
    _optional_modules['monitoring'] = monitoring
except ImportError as e:
    logger.debug(f"Monitoring module import failed: {e}")
    monitoring = None

# Optimization module
try:
    from . import optimization
    _optional_modules['optimization'] = optimization
except ImportError as e:
    logger.debug(f"Optimization module import failed: {e}")
    optimization = None

# Benchmarks module
try:
    from . import benchmarks
    _optional_modules['benchmarks'] = benchmarks
except ImportError as e:
    logger.debug(f"Benchmarks module import failed: {e}")
    benchmarks = None

# Distributed deployment components (Generation 3 - Distributed)
try:
    from . import distributed
    _optional_modules['distributed'] = distributed
except ImportError as e:
    logger.debug(f"Distributed module import failed: {e}")
    distributed = None

# API module
try:
    from . import api
    _optional_modules['api'] = api
except ImportError as e:
    logger.debug(f"API module import failed: {e}")
    api = None

# Integrations module
try:
    from . import integrations
    _optional_modules['integrations'] = integrations
except ImportError as e:
    logger.debug(f"Integrations module import failed: {e}")
    integrations = None

# Common exception classes for convenience
from .exceptions import (
    Gaudi3ScaleError,
    HPUError,
    HPUNotAvailableError,
    HPUInitializationError,
    HPUMemoryError,
    TrainingError,
    ConfigurationError,
    ValidationError
)

# Common validation functions
from .validation import DataValidator, ValidationResult
from .config_validation import (
    validate_training_config,
    validate_model_config,
    validate_cluster_config
)

# Logging utilities
from .logging_utils import get_logger, LoggerFactory

# Health monitoring
from .health_checks import HealthMonitor, HealthStatus

# Build __all__ dynamically based on available modules
__all__ = [
    # Core components
    "GaudiAccelerator",
    "GaudiOptimizer", 
    "GaudiTrainer",
    
    # Enhanced modules (always available)
    "exceptions",
    "validation",
    "logging_utils",
    "retry_utils",
    "health_checks",
    "config_validation",
    "security",
    "optional_deps",
    
    # Common exceptions
    "Gaudi3ScaleError",
    "HPUError",
    "HPUNotAvailableError",
    "HPUInitializationError", 
    "HPUMemoryError",
    "TrainingError",
    "ConfigurationError",
    "ValidationError",
    
    # Validation utilities
    "DataValidator",
    "ValidationResult",
    "validate_training_config",
    "validate_model_config",
    "validate_cluster_config",
    
    # Logging utilities
    "get_logger",
    "LoggerFactory",
    
    # Health monitoring
    "HealthMonitor",
    "HealthStatus",
    
    # Utility functions
    "get_version_info",
    "configure_global_settings",
    "get_available_features",
    "check_optional_dependencies"
]

# Add optional modules to __all__ if they're available
for module_name in _optional_modules:
    __all__.append(module_name)

# Package metadata
__title__ = "gaudi3-scale"
__description__ = "Production Infrastructure for Intel Gaudi 3 HPU Clusters"
__url__ = "https://github.com/danieleschmidt/gaudi3-scale-starter"
__license__ = "MIT"
__status__ = "Production"
__generation__ = "3.0"  # Performance Optimization Generation

# Feature flags for backward compatibility
ENABLE_ENHANCED_FEATURES = True
ENABLE_STRICT_VALIDATION = False  # Can be enabled for stricter validation
ENABLE_PERFORMANCE_MONITORING = True
ENABLE_SECURITY_HARDENING = True  # Enterprise security features
ENABLE_PERFORMANCE_OPTIMIZATION = True  # Generation 3 performance features
ENABLE_DISTRIBUTED_DEPLOYMENT = True  # Generation 3 distributed features


def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "title": __title__,
        "description": __description__,
        "author": __author__,
        "status": __status__,
        "generation": __generation__,
        "enhanced_features": ENABLE_ENHANCED_FEATURES,
        "strict_validation": ENABLE_STRICT_VALIDATION,
        "performance_monitoring": ENABLE_PERFORMANCE_MONITORING,
        "security_hardening": ENABLE_SECURITY_HARDENING,
        "distributed_deployment": ENABLE_DISTRIBUTED_DEPLOYMENT,
        "available_optional_modules": list(_optional_modules.keys())
    }


def configure_global_settings(
    strict_validation: bool = None,
    performance_monitoring: bool = None,
    log_level: str = "INFO"
):
    """Configure global package settings.
    
    Args:
        strict_validation: Enable strict validation mode
        performance_monitoring: Enable performance monitoring
        log_level: Default logging level
    """
    global ENABLE_STRICT_VALIDATION, ENABLE_PERFORMANCE_MONITORING
    
    if strict_validation is not None:
        ENABLE_STRICT_VALIDATION = strict_validation
    
    if performance_monitoring is not None:
        ENABLE_PERFORMANCE_MONITORING = performance_monitoring
    
    # Configure default logging
    if log_level:
        try:
            from .logging_utils import LoggerFactory
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL
            }
            level = level_map.get(log_level.upper(), logging.INFO)
            LoggerFactory.configure_defaults(level=level)
        except ImportError:
            logger.warning("LoggerFactory not available for configuration")


def get_available_features():
    """Get information about available features and their dependency status.
    
    Returns:
        Dict with feature availability information
    """
    from .optional_deps import get_available_deps, get_missing_deps
    
    available_deps = get_available_deps()
    missing_deps = get_missing_deps()
    
    features = {
        "core": {
            "available": True,
            "description": "Core Gaudi 3 training functionality",
            "modules": ["accelerator", "optimizer", "trainer"]
        },
        "api": {
            "available": optional_deps.FASTAPI is not None,
            "description": "REST API server functionality",
            "dependencies": ["fastapi", "uvicorn", "pydantic"],
            "install_command": "pip install gaudi3-scale-starter[api]"
        },
        "caching": {
            "available": optional_deps.REDIS is not None,
            "description": "Redis-based caching system",
            "dependencies": ["redis"],
            "install_command": "pip install gaudi3-scale-starter[caching]"
        },
        "database": {
            "available": optional_deps.SQLALCHEMY is not None,
            "description": "Database connectivity and ORM",
            "dependencies": ["sqlalchemy", "asyncpg", "psycopg2"],
            "install_command": "pip install gaudi3-scale-starter[database]"
        },
        "monitoring": {
            "available": optional_deps.PROMETHEUS_CLIENT is not None,
            "description": "Metrics and monitoring",
            "dependencies": ["prometheus_client"],
            "install_command": "pip install gaudi3-scale-starter[monitoring]"
        },
        "async": {
            "available": optional_deps.AIOHTTP is not None,
            "description": "Async functionality and performance optimization", 
            "dependencies": ["aiohttp", "aiofiles"],
            "install_command": "pip install gaudi3-scale-starter[async]"
        }
    }
    
    return {
        "features": features,
        "available_modules": list(_optional_modules.keys()),
        "available_dependencies": available_deps,
        "missing_dependencies": missing_deps
    }


def check_optional_dependencies(feature: str = None):
    """Check status of optional dependencies and provide installation guidance.
    
    Args:
        feature: Specific feature to check (e.g., 'api', 'caching'). If None, check all.
    """
    features_info = get_available_features()
    
    if feature:
        if feature not in features_info["features"]:
            print(f"Unknown feature: {feature}")
            print(f"Available features: {list(features_info['features'].keys())}")
            return
        
        feature_info = features_info["features"][feature]
        status = "✓ Available" if feature_info["available"] else "✗ Not Available"
        print(f"{feature.title()} Feature: {status}")
        print(f"Description: {feature_info['description']}")
        
        if not feature_info["available"] and "install_command" in feature_info:
            print(f"To enable: {feature_info['install_command']}")
    else:
        print("Gaudi 3 Scale Optional Dependencies Status")
        print("=" * 45)
        
        for feature_name, feature_info in features_info["features"].items():
            status = "✓" if feature_info["available"] else "✗"
            print(f"{status} {feature_name.title()}: {feature_info['description']}")
            
            if not feature_info["available"] and "install_command" in feature_info:
                print(f"   Install with: {feature_info['install_command']}")
        
        print(f"\nAvailable optional modules: {', '.join(features_info['available_modules'])}")
        
        if features_info["missing_dependencies"]:
            print(f"\nMissing dependencies:")
            for dep, error in features_info["missing_dependencies"].items():
                print(f"  - {dep}: {error}")


# Show warnings for missing critical optional dependencies on import
def _show_import_warnings():
    """Show warnings for missing critical optional dependencies."""
    try:
        missing_deps = optional_deps.get_missing_deps()
        
        # Only warn about truly optional dependencies that affect major features
        critical_deps = {
            'fastapi': 'API server functionality',
            'redis': 'High-performance caching',
            'sqlalchemy': 'Database connectivity'
        }
        
        for dep_name, feature_name in critical_deps.items():
            if dep_name in missing_deps:
                warnings.warn(
                    f"Optional dependency '{dep_name}' not found. "
                    f"{feature_name} will be disabled. "
                    f"Install with: pip install gaudi3-scale-starter[{optional_deps.get_extra_name(dep_name)}]",
                    UserWarning,
                    stacklevel=2
                )
    except Exception as e:
        logger.debug(f"Error checking optional dependencies: {e}")

# Run import warnings (only once)
_show_import_warnings()