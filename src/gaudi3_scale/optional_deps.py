"""Optional dependency management for gaudi3_scale package.

This module provides utilities for graceful handling of optional dependencies,
allowing the core package functionality to work even when optional dependencies
are not installed.
"""

import logging
import warnings
from typing import Any, Dict, Optional, Type, Union, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# Track which optional dependencies are available
_AVAILABLE_DEPS: Dict[str, bool] = {}
_MISSING_DEPS: Dict[str, str] = {}


def try_import(module_name: str, package_name: str = None) -> Union[Any, None]:
    """Try to import a module, returning None if not available.
    
    Args:
        module_name: Name of the module to import
        package_name: Optional package name for better error messages
        
    Returns:
        Imported module or None if not available
    """
    package_name = package_name or module_name
    
    if module_name in _AVAILABLE_DEPS:
        if not _AVAILABLE_DEPS[module_name]:
            return None
    
    try:
        if '.' in module_name:
            # Handle submodule imports like 'fastapi.routing'
            parts = module_name.split('.')
            module = __import__(module_name)
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            module = __import__(module_name)
        
        _AVAILABLE_DEPS[module_name] = True
        logger.debug(f"Successfully imported optional dependency: {module_name}")
        return module
        
    except ImportError as e:
        _AVAILABLE_DEPS[module_name] = False
        _MISSING_DEPS[module_name] = str(e)
        logger.debug(f"Optional dependency not available: {package_name} ({e})")
        return None


def require_optional_dep(dep_name: str, feature_name: str = None) -> Callable:
    """Decorator to require an optional dependency for a function/class.
    
    Args:
        dep_name: Name of the required dependency
        feature_name: Human-readable name of the feature
        
    Returns:
        Decorator function
    """
    def decorator(func_or_class):
        feature = feature_name or func_or_class.__name__
        
        @wraps(func_or_class)
        def wrapper(*args, **kwargs):
            if not is_available(dep_name):
                missing_dep_error = _MISSING_DEPS.get(dep_name, f"'{dep_name}' not found")
                raise ImportError(
                    f"Optional dependency '{dep_name}' is required for {feature}. "
                    f"Install it with: pip install gaudi3-scale-starter[{get_extra_name(dep_name)}] "
                    f"or pip install {dep_name}. Error: {missing_dep_error}"
                )
            return func_or_class(*args, **kwargs)
        return wrapper
    return decorator


def is_available(dep_name: str) -> bool:
    """Check if an optional dependency is available.
    
    Args:
        dep_name: Name of the dependency to check
        
    Returns:
        True if dependency is available
    """
    return _AVAILABLE_DEPS.get(dep_name, False)


def get_missing_deps() -> Dict[str, str]:
    """Get dictionary of missing dependencies and their error messages.
    
    Returns:
        Dictionary mapping dependency names to error messages
    """
    return _MISSING_DEPS.copy()


def get_available_deps() -> Dict[str, bool]:
    """Get dictionary of all checked dependencies and their availability.
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    return _AVAILABLE_DEPS.copy()


def get_extra_name(dep_name: str) -> str:
    """Get the extra name for a dependency in pyproject.toml.
    
    Args:
        dep_name: Name of the dependency
        
    Returns:
        Extra name for pip install
    """
    extra_mapping = {
        'fastapi': 'api',
        'redis': 'caching',
        'sqlalchemy': 'database',
        'aiohttp': 'async',
        'prometheus_client': 'monitoring',
        'uvicorn': 'api',
        'pydantic': 'api',
        'starlette': 'api',
    }
    return extra_mapping.get(dep_name, dep_name)


def warn_missing_dependency(dep_name: str, feature_name: str, fallback_msg: str = None):
    """Issue a warning about a missing optional dependency.
    
    Args:
        dep_name: Name of the missing dependency
        feature_name: Name of the feature that won't work
        fallback_msg: Optional message about fallback behavior
    """
    extra_name = get_extra_name(dep_name)
    msg = (
        f"Optional dependency '{dep_name}' not found. "
        f"{feature_name} functionality will be limited. "
        f"Install with: pip install gaudi3-scale-starter[{extra_name}] "
        f"or pip install {dep_name}"
    )
    
    if fallback_msg:
        msg += f". {fallback_msg}"
    
    warnings.warn(msg, UserWarning, stacklevel=2)
    logger.warning(msg)


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required but not available."""
    
    def __init__(self, dep_name: str, feature_name: str):
        extra_name = get_extra_name(dep_name)
        msg = (
            f"Optional dependency '{dep_name}' is required for {feature_name}. "
            f"Install it with: pip install gaudi3-scale-starter[{extra_name}] "
            f"or pip install {dep_name}"
        )
        super().__init__(msg)
        self.dep_name = dep_name
        self.feature_name = feature_name
        self.extra_name = extra_name


# Pre-check common optional dependencies
FASTAPI = try_import('fastapi', 'FastAPI')
REDIS = try_import('redis', 'redis-py')
SQLALCHEMY = try_import('sqlalchemy', 'SQLAlchemy')
AIOHTTP = try_import('aiohttp', 'aiohttp')
UVICORN = try_import('uvicorn', 'uvicorn')
PROMETHEUS_CLIENT = try_import('prometheus_client', 'prometheus-client')
PYDANTIC = try_import('pydantic', 'pydantic')
STARLETTE = try_import('starlette', 'starlette')

# FastAPI submodules
if FASTAPI:
    FASTAPI_ROUTING = try_import('fastapi.routing')
    FASTAPI_MIDDLEWARE = try_import('fastapi.middleware')
    FASTAPI_SECURITY = try_import('fastapi.security')
    FASTAPI_RESPONSES = try_import('fastapi.responses')
else:
    FASTAPI_ROUTING = None
    FASTAPI_MIDDLEWARE = None
    FASTAPI_SECURITY = None
    FASTAPI_RESPONSES = None


def create_mock_class(class_name: str, dep_name: str, feature_name: str = None):
    """Create a mock class that raises an error when instantiated.
    
    Args:
        class_name: Name of the class to mock
        dep_name: Name of the missing dependency
        feature_name: Name of the feature (defaults to class_name)
        
    Returns:
        Mock class that raises OptionalDependencyError
    """
    feature_name = feature_name or class_name
    
    class MockClass:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError(dep_name, feature_name)
        
        def __getattr__(self, name):
            raise OptionalDependencyError(dep_name, feature_name)
        
        @classmethod
        def __class_getitem__(cls, item):
            # Support for generic types like List[str]
            return cls
    
    MockClass.__name__ = class_name
    MockClass.__qualname__ = class_name
    return MockClass


def create_mock_function(func_name: str, dep_name: str, feature_name: str = None):
    """Create a mock function that raises an error when called.
    
    Args:
        func_name: Name of the function to mock
        dep_name: Name of the missing dependency
        feature_name: Name of the feature (defaults to func_name)
        
    Returns:
        Mock function that raises OptionalDependencyError
    """
    feature_name = feature_name or func_name
    
    def mock_func(*args, **kwargs):
        raise OptionalDependencyError(dep_name, feature_name)
    
    mock_func.__name__ = func_name
    mock_func.__qualname__ = func_name
    return mock_func