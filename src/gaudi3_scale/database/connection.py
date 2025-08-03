"""Database connection management for Gaudi 3 Scale."""

import os
import logging
from typing import Optional
from urllib.parse import quote_plus

import asyncpg
import redis
from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import Base

logger = logging.getLogger(__name__)


def get_database_url(async_mode: bool = False) -> str:
    """Get database URL from environment variables.
    
    Args:
        async_mode: Whether to return async database URL
        
    Returns:
        Database connection URL
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "gaudi3_scale")
    username = os.getenv("POSTGRES_USER", "gaudi3_user")
    password = os.getenv("POSTGRES_PASSWORD", "")
    
    # URL encode password to handle special characters
    encoded_password = quote_plus(password) if password else ""
    
    if async_mode:
        driver = "postgresql+asyncpg"
    else:
        driver = "postgresql+psycopg2"
    
    if encoded_password:
        return f"{driver}://{username}:{encoded_password}@{host}:{port}/{database}"
    else:
        return f"{driver}://{username}@{host}:{port}/{database}"


class DatabaseConnection:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection.
        
        Args:
            database_url: Optional database URL override
        """
        self.database_url = database_url or get_database_url()
        self.async_database_url = database_url or get_database_url(async_mode=True)
        
        # Create engines
        self.engine = self._create_sync_engine()
        self.async_engine = self._create_async_engine()
        
        # Create session makers
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            class_=Session
        )
        
        self.AsyncSessionLocal = sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False
        )
        
        self.logger = logger.getChild(self.__class__.__name__)
        
    def _create_sync_engine(self) -> Engine:
        """Create synchronous database engine."""
        return create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=os.getenv("DEBUG", "false").lower() == "true"
        )
    
    def _create_async_engine(self) -> AsyncEngine:
        """Create asynchronous database engine."""
        return create_async_engine(
            self.async_database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=os.getenv("DEBUG", "false").lower() == "true"
        )
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def create_tables_async(self):
        """Create all database tables asynchronously."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database tables created successfully (async)")
        except Exception as e:
            self.logger.error(f"Failed to create database tables (async): {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a database session.
        
        Returns:
            Database session
        """
        return self.SessionLocal()
    
    def get_async_session(self) -> AsyncSession:
        """Get an async database session.
        
        Returns:
            Async database session
        """
        return self.AsyncSessionLocal()
    
    def test_connection(self) -> bool:
        """Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            self.logger.info("Database connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    async def test_connection_async(self) -> bool:
        """Test database connection asynchronously.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            async with self.async_engine.connect() as conn:
                await conn.execute("SELECT 1")
            self.logger.info("Database connection test successful (async)")
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed (async): {e}")
            return False
    
    def close(self):
        """Close database connections."""
        self.engine.dispose()
        self.logger.info("Database connections closed")
    
    async def close_async(self):
        """Close database connections asynchronously."""
        await self.async_engine.dispose()
        self.logger.info("Database connections closed (async)")


class RedisConnection:
    """Manages Redis connections for caching and session storage."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis connection.
        
        Args:
            redis_url: Optional Redis URL override
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_password = os.getenv("REDIS_PASSWORD")
        
        # Create Redis connection
        self.redis_client = self._create_redis_client()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client."""
        return redis.from_url(
            self.redis_url,
            password=self.redis_password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
    
    def test_connection(self) -> bool:
        """Test Redis connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.redis_client.ping()
            self.logger.info("Redis connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Redis connection test failed: {e}")
            return False
    
    def get_client(self) -> redis.Redis:
        """Get Redis client.
        
        Returns:
            Redis client instance
        """
        return self.redis_client
    
    def close(self):
        """Close Redis connection."""
        self.redis_client.close()
        self.logger.info("Redis connection closed")


# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None
_redis_connection: Optional[RedisConnection] = None


def get_database() -> DatabaseConnection:
    """Get global database connection instance.
    
    Returns:
        Database connection instance
    """
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection


def get_redis() -> RedisConnection:
    """Get global Redis connection instance.
    
    Returns:
        Redis connection instance
    """
    global _redis_connection
    if _redis_connection is None:
        _redis_connection = RedisConnection()
    return _redis_connection


def close_connections():
    """Close all global connections."""
    global _db_connection, _redis_connection
    
    if _db_connection:
        _db_connection.close()
        _db_connection = None
    
    if _redis_connection:
        _redis_connection.close()
        _redis_connection = None