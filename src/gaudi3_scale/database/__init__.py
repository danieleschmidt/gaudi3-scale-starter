"""Database layer for Gaudi 3 Scale infrastructure."""

from .connection import DatabaseConnection, get_database_url

# Import optional database components
try:
    from .migrations import MigrationManager
    _has_migrations = True
except ImportError:
    MigrationManager = None
    _has_migrations = False

try:
    from .models import Base, ClusterModel, TrainingJobModel, MetricModel
    _has_models = True
except ImportError:
    Base = None
    ClusterModel = None
    TrainingJobModel = None
    MetricModel = None
    _has_models = False

# Build __all__ dynamically
__all__ = [
    "DatabaseConnection",
    "get_database_url",
]

if _has_migrations:
    __all__.append("MigrationManager")

if _has_models:
    __all__.extend([
        "Base",
        "ClusterModel", 
        "TrainingJobModel",
        "MetricModel",
    ])

# Utility functions
def has_migrations() -> bool:
    """Check if database migrations are available."""
    return _has_migrations

def has_models() -> bool:
    """Check if database models are available."""
    return _has_models