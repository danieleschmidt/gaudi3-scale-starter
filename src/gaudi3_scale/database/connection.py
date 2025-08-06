"""Database connection management for Gaudi 3 Scale."""

import os
import logging
from typing import Optional, Union
from urllib.parse import quote_plus

from ..optional_deps import (
    REDIS, SQLALCHEMY, AIOHTTP,
    OptionalDependencyError, require_optional_dep, 
    warn_missing_dependency, try_import
)

logger = logging.getLogger(__name__)

# Try to import database dependencies
if SQLALCHEMY:
    try:
        from sqlalchemy import create_engine, Engine
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
        from sqlalchemy.orm import sessionmaker, Session
        from sqlalchemy.pool import QueuePool
    except ImportError as e:
        logger.warning(f"Some SQLAlchemy components not available: {e}")
        SQLALCHEMY = None

# Try to import async postgres driver
ASYNCPG = try_import('asyncpg')

# Try to import database models
try:
    from .models import Base
    HAS_MODELS = True
except ImportError:
    logger.debug("Database models not available")
    HAS_MODELS = False


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
    
    @require_optional_dep('sqlalchemy', 'Database connections')
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection.
        
        Args:
            database_url: Optional database URL override
        """
        if not SQLALCHEMY:
            raise OptionalDependencyError('sqlalchemy', 'Database connections')
            
        self.database_url = database_url or get_database_url()
        self.async_database_url = database_url or get_database_url(async_mode=True)
        
        # Create engines
        try:
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
        except Exception as e:
            logger.error(f"Failed to initialize database engines: {e}")
            # Set fallback attributes
            self.engine = None
            self.async_engine = None
            self.SessionLocal = None
            self.AsyncSessionLocal = None
            
        self.logger = logger.getChild(self.__class__.__name__)
        
    def _create_sync_engine(self) -> 'Engine':
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
    
    def _create_async_engine(self) -> 'AsyncEngine':
        """Create asynchronous database engine."""
        if not ASYNCPG:
            warn_missing_dependency('asyncpg', 'Async database operations',
                                   'Async database functionality will be limited')
            return None
        
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
        if not self.engine:
            raise OptionalDependencyError('sqlalchemy', 'Database table creation')
        if not HAS_MODELS:
            logger.warning("Database models not available - cannot create tables")
            return
            
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def create_tables_async(self):
        """Create all database tables asynchronously."""
        if not self.async_engine:
            raise OptionalDependencyError('asyncpg', 'Async database table creation')
        if not HAS_MODELS:
            logger.warning("Database models not available - cannot create tables")
            return
            
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database tables created successfully (async)")
        except Exception as e:
            self.logger.error(f"Failed to create database tables (async): {e}")
            raise
    
    def get_session(self) -> 'Session':
        """Get a database session.
        
        Returns:
            Database session
        """
        if not self.SessionLocal:
            raise OptionalDependencyError('sqlalchemy', 'Database sessions')
        return self.SessionLocal()
    
    def get_async_session(self) -> 'AsyncSession':
        """Get an async database session.
        
        Returns:
            Async database session
        """
        if not self.AsyncSessionLocal:
            raise OptionalDependencyError('asyncpg', 'Async database sessions')
        return self.AsyncSessionLocal()
    
    def test_connection(self) -> bool:
        """Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.engine:
            logger.warning("Database engine not available")
            return False
            
        try:
            with self.engine.connect() as conn:
                if hasattr(conn, 'execute'):
                    # SQLAlchemy 1.4+ style
                    from sqlalchemy import text
                    conn.execute(text("SELECT 1"))
                else:
                    # Older SQLAlchemy style
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
        if not self.async_engine:
            logger.warning("Async database engine not available")
            return False
            
        try:
            async with self.async_engine.connect() as conn:
                from sqlalchemy import text
                await conn.execute(text("SELECT 1"))
            self.logger.info("Database connection test successful (async)")
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed (async): {e}")
            return False
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connections closed")
    
    async def close_async(self):
        """Close database connections asynchronously."""
        if self.async_engine:
            await self.async_engine.dispose()
            self.logger.info("Database connections closed (async)")


class RedisConnection:
    """Manages Redis connections for caching and session storage."""
    
    @require_optional_dep('redis', 'Redis connections')
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis connection.
        
        Args:
            redis_url: Optional Redis URL override
        """
        if not REDIS:
            raise OptionalDependencyError('redis', 'Redis connections')
            
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_password = os.getenv("REDIS_PASSWORD")
        
        # Create Redis connection
        try:
            self.redis_client = self._create_redis_client()
        except Exception as e:
            logger.error(f"Failed to create Redis client: {e}")
            self.redis_client = None
            
        self.logger = logger.getChild(self.__class__.__name__)
    
    def _create_redis_client(self) -> 'redis.Redis':
        """Create Redis client."""
        import redis
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
        if not self.redis_client:
            logger.warning("Redis client not available")
            return False
            
        try:
            self.redis_client.ping()
            self.logger.info("Redis connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Redis connection test failed: {e}")
            return False
    
    def get_client(self) -> 'redis.Redis':
        """Get Redis client.
        
        Returns:
            Redis client instance
        """
        if not self.redis_client:
            raise OptionalDependencyError('redis', 'Redis client operations')
        return self.redis_client
    
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
            self.logger.info("Redis connection closed")


class FallbackDatabaseConnection:
    """Fallback database connection for when SQLAlchemy is not available."""
    
    def __init__(self, *args, **kwargs):
        """Initialize fallback database connection."""
        self.logger = logger.getChild(self.__class__.__name__)
        warn_missing_dependency('sqlalchemy', 'Database connections',
                               'Database functionality is disabled')
    
    def create_tables(self):
        raise OptionalDependencyError('sqlalchemy', 'Database table creation')
    
    async def create_tables_async(self):
        raise OptionalDependencyError('sqlalchemy', 'Async database table creation')
    
    def get_session(self):
        raise OptionalDependencyError('sqlalchemy', 'Database sessions')
    
    def get_async_session(self):
        raise OptionalDependencyError('sqlalchemy', 'Async database sessions')
    
    def test_connection(self) -> bool:
        return False
    
    async def test_connection_async(self) -> bool:
        return False
    
    def close(self):
        pass
    
    async def close_async(self):
        pass


class FallbackRedisConnection:
    """Fallback Redis connection for when Redis is not available."""
    
    def __init__(self, *args, **kwargs):
        """Initialize fallback Redis connection."""
        self.logger = logger.getChild(self.__class__.__name__)
        warn_missing_dependency('redis', 'Redis connections',
                               'Redis functionality is disabled')
    
    def test_connection(self) -> bool:
        return False
    
    def get_client(self):
        raise OptionalDependencyError('redis', 'Redis client operations')
    
    def close(self):
        pass


# Global database connection instance
_db_connection: Optional[Union[DatabaseConnection, FallbackDatabaseConnection]] = None
_redis_connection: Optional[Union[RedisConnection, FallbackRedisConnection]] = None


def get_database() -> Union[DatabaseConnection, FallbackDatabaseConnection]:
    """Get global database connection instance.
    
    Returns:
        Database connection instance (or fallback if SQLAlchemy unavailable)
    """
    global _db_connection
    if _db_connection is None:
        if SQLALCHEMY:
            try:
                _db_connection = DatabaseConnection()
            except Exception as e:
                logger.warning(f"Failed to create database connection: {e}")
                _db_connection = FallbackDatabaseConnection()
        else:
            _db_connection = FallbackDatabaseConnection()
    return _db_connection


def get_redis() -> Union[RedisConnection, FallbackRedisConnection]:
    """Get global Redis connection instance.
    
    Returns:
        Redis connection instance (or fallback if Redis unavailable)
    """
    global _redis_connection
    if _redis_connection is None:
        if REDIS:
            try:
                _redis_connection = RedisConnection()
            except Exception as e:
                logger.warning(f"Failed to create Redis connection: {e}")
                _redis_connection = FallbackRedisConnection()
        else:
            _redis_connection = FallbackRedisConnection()
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


def is_database_available() -> bool:
    """Check if database functionality is available."""
    return SQLALCHEMY is not None


def is_redis_available() -> bool:
    """Check if Redis functionality is available."""
    return REDIS is not None


def get_connection_info() -> dict:
    """Get information about available connections."""
    return {
        "database_available": is_database_available(),
        "redis_available": is_redis_available(),
        "asyncpg_available": ASYNCPG is not None,
        "models_available": HAS_MODELS
    }