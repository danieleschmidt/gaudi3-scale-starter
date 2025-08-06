"""Main FastAPI application for Gaudi 3 Scale."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, Union

from ..optional_deps import FASTAPI, OptionalDependencyError, require_optional_dep

# Import FastAPI components conditionally
if FASTAPI:
    from fastapi import FastAPI, HTTPException, Depends, status, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    
    from .routers import clusters, training, metrics, health
    from .dependencies import get_database, get_cache_manager

logger = logging.getLogger(__name__)


if FASTAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        # Startup
        logger.info("Starting Gaudi 3 Scale API")
        
        # Initialize database connection (if available)
        try:
            from ..database.connection import get_database as get_db_connection
            db = get_db_connection()
            if hasattr(db, 'test_connection') and db.test_connection():
                logger.info("Database connection established")
            else:
                logger.warning("Database connection test failed or not implemented")
        except ImportError:
            logger.warning("Database connection not available - continuing without database")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
        
        yield
        
        # Shutdown
        logger.info("Shutting down Gaudi 3 Scale API")


@require_optional_dep('fastapi', 'API server')
def create_application() -> 'FastAPI':
    """Create FastAPI application with all configurations."""
    if not FASTAPI:
        raise OptionalDependencyError('fastapi', 'API server')
    
    app = FastAPI(
        title="Gaudi 3 Scale API",
        description="Production Infrastructure API for Intel Gaudi 3 HPU Clusters",
        version="0.5.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add security middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    )
    
    # Security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
    
    # Include routers
    app.include_router(
        clusters.router,
        prefix="/api/v1/clusters",
        tags=["clusters"]
    )
    
    app.include_router(
        training.router,
        prefix="/api/v1/training",
        tags=["training"]
    )
    
    app.include_router(
        metrics.router,
        prefix="/api/v1/metrics",
        tags=["metrics"]
    )
    
    app.include_router(
        health.router,
        prefix="/health",
        tags=["health"]
    )
    
    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "status_code": 500
            }
        )
    
    # Root endpoint
    @app.get("/", response_model=Dict[str, Any])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Gaudi 3 Scale API",
            "version": "0.5.0",
            "description": "Production Infrastructure API for Intel Gaudi 3 HPU Clusters",
            "docs_url": "/docs",
            "health_check": "/health"
        }
    
    return app


# Create application instance only if FastAPI is available
if FASTAPI:
    try:
        app = create_application()
    except Exception as e:
        logger.warning(f"Failed to create FastAPI application: {e}")
        app = None
else:
    app = None
    logger.info("FastAPI not available - API functionality disabled")


@require_optional_dep('fastapi', 'API server')
def get_application() -> 'FastAPI':
    """Get application instance for testing."""
    if app is None:
        return create_application()
    return app


def is_api_available() -> bool:
    """Check if API functionality is available."""
    return FASTAPI is not None and app is not None


def get_api_info() -> Dict[str, Any]:
    """Get API availability information."""
    return {
        "fastapi_available": FASTAPI is not None,
        "app_created": app is not None,
        "api_available": is_api_available(),
        "message": "API server functionality requires FastAPI. Install with: pip install gaudi3-scale-starter[api]"
    }