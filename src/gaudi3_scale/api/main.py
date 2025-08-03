"""Main FastAPI application for Gaudi 3 Scale."""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .routers import clusters, training, metrics, health
from .dependencies import get_database, get_cache_manager
from ..database.connection import DatabaseConnection, get_database as get_db_connection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Gaudi 3 Scale API")
    
    # Initialize database connection
    try:
        db = get_db_connection()
        if db.test_connection():
            logger.info("Database connection established")
        else:
            logger.error("Failed to establish database connection")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Gaudi 3 Scale API")


def create_application() -> FastAPI:
    """Create FastAPI application with all configurations."""
    
    app = FastAPI(
        title="Gaudi 3 Scale API",
        description="Production Infrastructure API for Intel Gaudi 3 HPU Clusters",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
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
            "version": "0.1.0",
            "description": "Production Infrastructure API for Intel Gaudi 3 HPU Clusters",
            "docs_url": "/docs",
            "health_check": "/health"
        }
    
    return app


# Create application instance
app = create_application()


def get_application() -> FastAPI:
    """Get application instance for testing."""
    return app