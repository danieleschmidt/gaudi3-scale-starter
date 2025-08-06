"""REST API for Gaudi 3 Scale infrastructure management."""

from ..optional_deps import FASTAPI

if FASTAPI:
    from .main import app, get_application, is_api_available, get_api_info
    
    __all__ = ["app", "get_application", "is_api_available", "get_api_info"]
else:
    # Provide stub implementations when FastAPI is not available
    app = None
    
    def get_application():
        from ..optional_deps import OptionalDependencyError
        raise OptionalDependencyError('fastapi', 'API server')
    
    def is_api_available():
        return False
    
    def get_api_info():
        return {
            "fastapi_available": False,
            "app_created": False,
            "api_available": False,
            "message": "API server functionality requires FastAPI. Install with: pip install gaudi3-scale-starter[api]"
        }
    
    __all__ = ["app", "get_application", "is_api_available", "get_api_info"]