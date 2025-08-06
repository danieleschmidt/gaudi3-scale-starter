"""Advanced connection pooling and resource management for databases and external services."""

import asyncio
import threading
import time
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
import weakref
from abc import ABC, abstractmethod
from queue import Queue, Empty, Full
import psutil

logger = logging.getLogger(__name__)

class PoolState(Enum):
    """Connection pool states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"

class ConnectionState(Enum):
    """Connection states."""
    IDLE = "idle"
    ACTIVE = "active"
    STALE = "stale"
    BROKEN = "broken"
    RETIRED = "retired"

@dataclass
class PoolConfig:
    """Configuration for connection pool."""
    min_size: int = 2
    max_size: int = 10
    max_overflow: int = 5
    pool_timeout: float = 30.0
    pool_recycle: int = 3600  # seconds
    pool_pre_ping: bool = True
    pool_reset_on_return: str = "commit"  # "commit", "rollback", "none"
    health_check_interval: float = 60.0
    max_idle_time: float = 1800.0  # 30 minutes
    connection_retry_attempts: int = 3
    connection_retry_delay: float = 1.0
    enable_metrics: bool = True

@dataclass 
class ConnectionWrapper:
    """Wrapper for database connections with metadata."""
    connection: Any
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    state: ConnectionState = ConnectionState.IDLE
    pool_id: Optional[str] = None
    thread_id: Optional[int] = None
    
    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used = time.time()
        self.use_count += 1
    
    def is_stale(self, max_age: float) -> bool:
        """Check if connection is stale."""
        return time.time() - self.created_at > max_age
    
    def is_idle_too_long(self, max_idle: float) -> bool:
        """Check if connection has been idle too long."""
        return time.time() - self.last_used > max_idle
    
    def age(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at
    
    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used

class PoolMetrics:
    """Connection pool metrics collector."""
    
    def __init__(self):
        self.reset()
        self._lock = threading.Lock()
    
    def reset(self) -> None:
        """Reset all metrics."""
        with getattr(self, '_lock', threading.Lock()):
            self.connections_created = 0
            self.connections_closed = 0
            self.connections_failed = 0
            self.connections_recycled = 0
            self.checkouts = 0
            self.checkins = 0
            self.checkout_timeouts = 0
            self.health_check_failures = 0
            self.pool_overflows = 0
            self.total_checkout_time = 0.0
            self.max_checkout_time = 0.0
    
    def record_checkout(self, duration: float) -> None:
        """Record connection checkout."""
        with self._lock:
            self.checkouts += 1
            self.total_checkout_time += duration
            self.max_checkout_time = max(self.max_checkout_time, duration)
    
    def record_checkin(self) -> None:
        """Record connection checkin."""
        with self._lock:
            self.checkins += 1
    
    def record_timeout(self) -> None:
        """Record checkout timeout."""
        with self._lock:
            self.checkout_timeouts += 1
    
    def record_creation(self) -> None:
        """Record connection creation."""
        with self._lock:
            self.connections_created += 1
    
    def record_closure(self) -> None:
        """Record connection closure."""
        with self._lock:
            self.connections_closed += 1
    
    def record_failure(self) -> None:
        """Record connection failure."""
        with self._lock:
            self.connections_failed += 1
    
    def record_recycle(self) -> None:
        """Record connection recycle."""
        with self._lock:
            self.connections_recycled += 1
    
    def record_health_check_failure(self) -> None:
        """Record health check failure."""
        with self._lock:
            self.health_check_failures += 1
    
    def record_overflow(self) -> None:
        """Record pool overflow."""
        with self._lock:
            self.pool_overflows += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            avg_checkout_time = (
                self.total_checkout_time / self.checkouts 
                if self.checkouts > 0 else 0.0
            )
            
            return {
                'connections_created': self.connections_created,
                'connections_closed': self.connections_closed,
                'connections_failed': self.connections_failed,
                'connections_recycled': self.connections_recycled,
                'checkouts': self.checkouts,
                'checkins': self.checkins,
                'checkout_timeouts': self.checkout_timeouts,
                'health_check_failures': self.health_check_failures,
                'pool_overflows': self.pool_overflows,
                'avg_checkout_time': avg_checkout_time,
                'max_checkout_time': self.max_checkout_time,
                'checkout_success_rate': (
                    (self.checkouts - self.checkout_timeouts) / self.checkouts * 100
                    if self.checkouts > 0 else 100.0
                )
            }

class ConnectionFactory(ABC):
    """Abstract factory for creating connections."""
    
    @abstractmethod
    def create_connection(self) -> Any:
        """Create a new connection."""
        pass
    
    @abstractmethod
    def validate_connection(self, connection: Any) -> bool:
        """Validate if connection is still usable."""
        pass
    
    @abstractmethod
    def close_connection(self, connection: Any) -> None:
        """Close a connection."""
        pass
    
    def reset_connection(self, connection: Any) -> None:
        """Reset connection state (optional)."""
        pass

class ConnectionPool:
    """High-performance connection pool with advanced features."""
    
    def __init__(self, 
                 connection_factory: ConnectionFactory,
                 config: Optional[PoolConfig] = None,
                 name: str = "default"):
        """Initialize connection pool.
        
        Args:
            connection_factory: Factory for creating connections
            config: Pool configuration
            name: Pool name for identification
        """
        self.factory = connection_factory
        self.config = config or PoolConfig()
        self.name = name
        self.logger = logger.getChild(f"{self.__class__.__name__}.{name}")
        
        # Pool state
        self._state = PoolState.INITIALIZING
        self._pool: Queue[ConnectionWrapper] = Queue(maxsize=self.config.max_size)
        self._overflow_connections: weakref.WeakSet = weakref.WeakSet()
        self._active_connections: Dict[int, ConnectionWrapper] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._creation_lock = threading.Lock()
        
        # Metrics
        self.metrics = PoolMetrics() if self.config.enable_metrics else None
        
        # Health checking
        self._health_check_task: Optional[threading.Timer] = None
        self._shutdown_event = threading.Event()
        
        # Initialize pool
        self._initialize_pool()
        
        # Start health checking
        if self.config.health_check_interval > 0:
            self._start_health_checks()
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        try:
            # Create minimum number of connections
            for _ in range(self.config.min_size):
                try:
                    conn = self._create_connection()
                    self._pool.put_nowait(conn)
                except Exception as e:
                    self.logger.error(f"Failed to create initial connection: {e}")
                    # Continue with fewer connections if some fail
            
            self._state = PoolState.HEALTHY
            self.logger.info(f"Initialized pool '{self.name}' with {self._pool.qsize()} connections")
            
        except Exception as e:
            self._state = PoolState.CRITICAL
            self.logger.error(f"Failed to initialize pool '{self.name}': {e}")
            raise
    
    def _create_connection(self) -> ConnectionWrapper:
        """Create a new connection."""
        try:
            connection = self.factory.create_connection()
            wrapper = ConnectionWrapper(
                connection=connection,
                pool_id=self.name,
                thread_id=threading.get_ident()
            )
            
            if self.metrics:
                self.metrics.record_creation()
            
            self.logger.debug(f"Created new connection {id(wrapper)}")
            return wrapper
            
        except Exception as e:
            if self.metrics:
                self.metrics.record_failure()
            self.logger.error(f"Failed to create connection: {e}")
            raise
    
    def _validate_connection(self, wrapper: ConnectionWrapper) -> bool:
        """Validate connection and update state."""
        try:
            # Check if connection is too old
            if wrapper.is_stale(self.config.pool_recycle):
                wrapper.state = ConnectionState.STALE
                return False
            
            # Ping connection if enabled
            if self.config.pool_pre_ping:
                if not self.factory.validate_connection(wrapper.connection):
                    wrapper.state = ConnectionState.BROKEN
                    return False
            
            wrapper.state = ConnectionState.IDLE
            return True
            
        except Exception as e:
            self.logger.warning(f"Connection validation failed: {e}")
            wrapper.state = ConnectionState.BROKEN
            return False
    
    def _close_connection(self, wrapper: ConnectionWrapper) -> None:
        """Close a connection."""
        try:
            self.factory.close_connection(wrapper.connection)
            wrapper.state = ConnectionState.RETIRED
            
            if self.metrics:
                self.metrics.record_closure()
            
            self.logger.debug(f"Closed connection {id(wrapper)}")
            
        except Exception as e:
            self.logger.warning(f"Error closing connection: {e}")
    
    def _reset_connection(self, wrapper: ConnectionWrapper) -> None:
        """Reset connection state."""
        try:
            if self.config.pool_reset_on_return != "none":
                self.factory.reset_connection(wrapper.connection)
        except Exception as e:
            self.logger.warning(f"Error resetting connection: {e}")
            wrapper.state = ConnectionState.BROKEN
    
    def get_connection(self, timeout: Optional[float] = None) -> ConnectionWrapper:
        """Get a connection from the pool.
        
        Args:
            timeout: Checkout timeout in seconds
            
        Returns:
            Connection wrapper
            
        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If pool is shutdown
        """
        if self._state == PoolState.SHUTDOWN:
            raise RuntimeError("Connection pool is shutdown")
        
        start_time = time.time()
        effective_timeout = timeout or self.config.pool_timeout
        
        try:
            # Try to get connection from pool
            with self._lock:
                while True:
                    try:
                        wrapper = self._pool.get_nowait()
                        
                        # Validate connection
                        if self._validate_connection(wrapper):
                            wrapper.state = ConnectionState.ACTIVE
                            wrapper.touch()
                            self._active_connections[id(wrapper)] = wrapper
                            
                            if self.metrics:
                                checkout_time = time.time() - start_time
                                self.metrics.record_checkout(checkout_time)
                            
                            return wrapper
                        else:
                            # Connection is invalid, close it
                            self._close_connection(wrapper)
                            continue
                            
                    except Empty:
                        break
                
                # No connections available, try to create new one
                total_connections = len(self._active_connections) + self._pool.qsize()
                
                if total_connections < self.config.max_size:
                    # Can create new connection within pool limit
                    with self._creation_lock:
                        wrapper = self._create_connection()
                        wrapper.state = ConnectionState.ACTIVE
                        wrapper.touch()
                        self._active_connections[id(wrapper)] = wrapper
                        
                        if self.metrics:
                            checkout_time = time.time() - start_time
                            self.metrics.record_checkout(checkout_time)
                        
                        return wrapper
                
                elif len(self._overflow_connections) < self.config.max_overflow:
                    # Create overflow connection
                    with self._creation_lock:
                        wrapper = self._create_connection()
                        wrapper.state = ConnectionState.ACTIVE
                        wrapper.touch()
                        self._overflow_connections.add(wrapper)
                        
                        if self.metrics:
                            self.metrics.record_overflow()
                            checkout_time = time.time() - start_time
                            self.metrics.record_checkout(checkout_time)
                        
                        return wrapper
                
                # Pool is full, wait for connection to be returned
                elapsed = time.time() - start_time
                if elapsed >= effective_timeout:
                    if self.metrics:
                        self.metrics.record_timeout()
                    raise TimeoutError(f"Pool checkout timed out after {effective_timeout}s")
                
                # Wait a bit before retrying
                time.sleep(0.1)
                
        except Exception as e:
            if self.metrics:
                self.metrics.record_failure()
            raise
    
    def return_connection(self, wrapper: ConnectionWrapper, discard: bool = False) -> None:
        """Return a connection to the pool.
        
        Args:
            wrapper: Connection wrapper to return
            discard: Whether to discard the connection
        """
        with self._lock:
            # Remove from active connections
            self._active_connections.pop(id(wrapper), None)
            
            if discard or wrapper.state == ConnectionState.BROKEN:
                # Discard broken or unwanted connection
                self._close_connection(wrapper)
                
                # Replace with new connection if below minimum
                if (self._pool.qsize() + len(self._active_connections) < self.config.min_size and
                    self._state != PoolState.SHUTDOWN):
                    try:
                        new_wrapper = self._create_connection()
                        self._pool.put_nowait(new_wrapper)
                    except Exception as e:
                        self.logger.warning(f"Failed to replace discarded connection: {e}")
                
                if self.metrics:
                    self.metrics.record_checkin()
                return
            
            # Check if this is an overflow connection
            if wrapper in self._overflow_connections:
                self._overflow_connections.discard(wrapper)
                self._close_connection(wrapper)
                if self.metrics:
                    self.metrics.record_checkin()
                return
            
            # Reset connection state
            try:
                self._reset_connection(wrapper)
                
                if wrapper.state != ConnectionState.BROKEN:
                    # Connection is good, return to pool
                    wrapper.state = ConnectionState.IDLE
                    
                    try:
                        self._pool.put_nowait(wrapper)
                        if self.metrics:
                            self.metrics.record_checkin()
                    except Full:
                        # Pool is full, close excess connection
                        self._close_connection(wrapper)
                else:
                    # Connection is broken after reset
                    self._close_connection(wrapper)
                    
            except Exception as e:
                self.logger.warning(f"Error returning connection: {e}")
                self._close_connection(wrapper)
            
            if self.metrics:
                self.metrics.record_checkin()
    
    @contextmanager
    def connection(self, timeout: Optional[float] = None) -> Iterator[Any]:
        """Context manager for getting/returning connections.
        
        Args:
            timeout: Checkout timeout in seconds
            
        Yields:
            Database connection
        """
        wrapper = None
        try:
            wrapper = self.get_connection(timeout)
            yield wrapper.connection
        except Exception:
            if wrapper:
                self.return_connection(wrapper, discard=True)
            raise
        else:
            if wrapper:
                self.return_connection(wrapper)
    
    def _health_check(self) -> None:
        """Perform health check on pool connections."""
        if self._shutdown_event.is_set():
            return
        
        try:
            with self._lock:
                # Check idle connections in pool
                connections_to_remove = []
                temp_connections = []
                
                # Get all connections from pool
                while not self._pool.empty():
                    try:
                        wrapper = self._pool.get_nowait()
                        
                        if (wrapper.is_stale(self.config.pool_recycle) or 
                            wrapper.is_idle_too_long(self.config.max_idle_time) or
                            not self._validate_connection(wrapper)):
                            
                            connections_to_remove.append(wrapper)
                            if self.metrics:
                                self.metrics.record_recycle()
                        else:
                            temp_connections.append(wrapper)
                    except Empty:
                        break
                
                # Put valid connections back
                for wrapper in temp_connections:
                    try:
                        self._pool.put_nowait(wrapper)
                    except Full:
                        connections_to_remove.append(wrapper)
                
                # Close invalid connections
                for wrapper in connections_to_remove:
                    self._close_connection(wrapper)
                
                # Maintain minimum pool size
                current_size = self._pool.qsize() + len(self._active_connections)
                needed = max(0, self.config.min_size - current_size)
                
                for _ in range(needed):
                    try:
                        new_wrapper = self._create_connection()
                        self._pool.put_nowait(new_wrapper)
                    except Exception as e:
                        self.logger.warning(f"Failed to create replacement connection: {e}")
                        if self.metrics:
                            self.metrics.record_health_check_failure()
                        break
                
                # Update pool state
                healthy_connections = self._pool.qsize() + len(self._active_connections)
                if healthy_connections >= self.config.min_size:
                    self._state = PoolState.HEALTHY
                elif healthy_connections > 0:
                    self._state = PoolState.DEGRADED
                else:
                    self._state = PoolState.CRITICAL
            
        except Exception as e:
            self.logger.error(f"Error during health check: {e}")
            if self.metrics:
                self.metrics.record_health_check_failure()
        
        # Schedule next health check
        if not self._shutdown_event.is_set() and self.config.health_check_interval > 0:
            self._health_check_task = threading.Timer(
                self.config.health_check_interval, 
                self._health_check
            )
            self._health_check_task.start()
    
    def _start_health_checks(self) -> None:
        """Start health checking."""
        if self.config.health_check_interval > 0:
            self._health_check_task = threading.Timer(
                self.config.health_check_interval,
                self._health_check
            )
            self._health_check_task.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        with self._lock:
            pool_size = self._pool.qsize()
            active_size = len(self._active_connections)
            overflow_size = len(self._overflow_connections)
            
            stats = {
                'pool_name': self.name,
                'state': self._state.value,
                'pool_size': pool_size,
                'active_connections': active_size,
                'overflow_connections': overflow_size,
                'total_connections': pool_size + active_size + overflow_size,
                'config': {
                    'min_size': self.config.min_size,
                    'max_size': self.config.max_size,
                    'max_overflow': self.config.max_overflow,
                    'pool_timeout': self.config.pool_timeout,
                    'pool_recycle': self.config.pool_recycle,
                }
            }
            
            if self.metrics:
                stats['metrics'] = self.metrics.get_stats()
            
            return stats
    
    def resize(self, min_size: int, max_size: int) -> None:
        """Resize pool limits.
        
        Args:
            min_size: New minimum pool size
            max_size: New maximum pool size
        """
        with self._lock:
            old_min = self.config.min_size
            old_max = self.config.max_size
            
            self.config.min_size = min_size
            self.config.max_size = max_size
            
            # Adjust pool size if needed
            current_size = self._pool.qsize() + len(self._active_connections)
            
            if current_size < min_size:
                # Need to create more connections
                needed = min_size - current_size
                for _ in range(needed):
                    try:
                        wrapper = self._create_connection()
                        self._pool.put_nowait(wrapper)
                    except Exception as e:
                        self.logger.warning(f"Failed to create connection during resize: {e}")
                        break
            
            elif self._pool.qsize() > max_size:
                # Need to remove excess idle connections
                excess = self._pool.qsize() - max_size
                removed_connections = []
                
                for _ in range(excess):
                    try:
                        wrapper = self._pool.get_nowait()
                        removed_connections.append(wrapper)
                    except Empty:
                        break
                
                # Close removed connections
                for wrapper in removed_connections:
                    self._close_connection(wrapper)
            
            self.logger.info(
                f"Resized pool '{self.name}' from ({old_min}, {old_max}) to ({min_size}, {max_size})"
            )
    
    def invalidate(self) -> None:
        """Invalidate all connections in the pool."""
        with self._lock:
            # Close all idle connections
            while not self._pool.empty():
                try:
                    wrapper = self._pool.get_nowait()
                    self._close_connection(wrapper)
                except Empty:
                    break
            
            # Mark active connections for discard on return
            for wrapper in self._active_connections.values():
                wrapper.state = ConnectionState.BROKEN
            
            # Close overflow connections
            overflow_list = list(self._overflow_connections)
            for wrapper in overflow_list:
                self._overflow_connections.discard(wrapper)
                self._close_connection(wrapper)
            
            self.logger.info(f"Invalidated all connections in pool '{self.name}'")
    
    def close(self) -> None:
        """Close the connection pool."""
        self.logger.info(f"Closing connection pool '{self.name}'")
        
        # Signal shutdown
        self._state = PoolState.SHUTDOWN
        self._shutdown_event.set()
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
        
        with self._lock:
            # Close all idle connections
            while not self._pool.empty():
                try:
                    wrapper = self._pool.get_nowait()
                    self._close_connection(wrapper)
                except Empty:
                    break
            
            # Close active connections (they will be discarded when returned)
            active_list = list(self._active_connections.values())
            for wrapper in active_list:
                wrapper.state = ConnectionState.BROKEN
            
            # Close overflow connections
            overflow_list = list(self._overflow_connections)
            for wrapper in overflow_list:
                self._overflow_connections.discard(wrapper)
                self._close_connection(wrapper)
        
        self.logger.info(f"Closed connection pool '{self.name}'")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PoolManager:
    """Manager for multiple connection pools."""
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self._lock = threading.RLock()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def create_pool(self, 
                    name: str, 
                    factory: ConnectionFactory,
                    config: Optional[PoolConfig] = None) -> ConnectionPool:
        """Create a new connection pool.
        
        Args:
            name: Pool name
            factory: Connection factory
            config: Pool configuration
            
        Returns:
            Created connection pool
        """
        with self._lock:
            if name in self.pools:
                raise ValueError(f"Pool '{name}' already exists")
            
            pool = ConnectionPool(factory, config, name)
            self.pools[name] = pool
            
            self.logger.info(f"Created pool '{name}'")
            return pool
    
    def get_pool(self, name: str) -> ConnectionPool:
        """Get connection pool by name.
        
        Args:
            name: Pool name
            
        Returns:
            Connection pool
            
        Raises:
            KeyError: If pool not found
        """
        with self._lock:
            if name not in self.pools:
                raise KeyError(f"Pool '{name}' not found")
            return self.pools[name]
    
    def remove_pool(self, name: str) -> None:
        """Remove and close connection pool.
        
        Args:
            name: Pool name
        """
        with self._lock:
            if name in self.pools:
                pool = self.pools.pop(name)
                pool.close()
                self.logger.info(f"Removed pool '{name}'")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools.
        
        Returns:
            Dictionary mapping pool names to statistics
        """
        with self._lock:
            return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all pools.
        
        Returns:
            Dictionary mapping pool names to health status
        """
        results = {}
        with self._lock:
            for name, pool in self.pools.items():
                try:
                    stats = pool.get_stats()
                    results[name] = stats['state'] in ['healthy', 'degraded']
                except Exception as e:
                    self.logger.error(f"Health check failed for pool '{name}': {e}")
                    results[name] = False
        return results
    
    def close_all(self) -> None:
        """Close all connection pools."""
        with self._lock:
            pool_names = list(self.pools.keys())
            for name in pool_names:
                self.remove_pool(name)
        
        self.logger.info("Closed all connection pools")


# Global pool manager
_pool_manager: Optional[PoolManager] = None

def get_pool_manager() -> PoolManager:
    """Get global pool manager instance.
    
    Returns:
        Pool manager instance
    """
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = PoolManager()
    return _pool_manager