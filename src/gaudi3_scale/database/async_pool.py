"""Async connection pooling with advanced resource management."""

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator, Set
from dataclasses import dataclass, field
from enum import Enum
import weakref
from abc import ABC, abstractmethod

from .connection_pool import PoolState, ConnectionState, PoolConfig, PoolMetrics, ConnectionWrapper

logger = logging.getLogger(__name__)

class AsyncConnectionFactory(ABC):
    """Abstract factory for creating async connections."""
    
    @abstractmethod
    async def create_connection(self) -> Any:
        """Create a new async connection."""
        pass
    
    @abstractmethod
    async def validate_connection(self, connection: Any) -> bool:
        """Validate if connection is still usable."""
        pass
    
    @abstractmethod
    async def close_connection(self, connection: Any) -> None:
        """Close a connection."""
        pass
    
    async def reset_connection(self, connection: Any) -> None:
        """Reset connection state (optional)."""
        pass

class AsyncConnectionPool:
    """High-performance async connection pool."""
    
    def __init__(self, 
                 connection_factory: AsyncConnectionFactory,
                 config: Optional[PoolConfig] = None,
                 name: str = "async_default"):
        """Initialize async connection pool.
        
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
        self._pool: asyncio.Queue[ConnectionWrapper] = asyncio.Queue(maxsize=self.config.max_size)
        self._overflow_connections: weakref.WeakSet = weakref.WeakSet()
        self._active_connections: Dict[int, ConnectionWrapper] = {}
        
        # Async synchronization
        self._lock = asyncio.Lock()
        self._creation_semaphore = asyncio.Semaphore(5)  # Limit concurrent creations
        
        # Metrics
        self.metrics = PoolMetrics() if self.config.enable_metrics else None
        
        # Health checking
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Pool initialization will be done lazily
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def _ensure_initialized(self) -> None:
        """Ensure pool is initialized."""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:
                return
            
            await self._initialize_pool()
            self._initialized = True
    
    async def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        try:
            # Create minimum number of connections
            for _ in range(self.config.min_size):
                try:
                    conn = await self._create_connection()
                    await self._pool.put(conn)
                except Exception as e:
                    self.logger.error(f"Failed to create initial connection: {e}")
                    # Continue with fewer connections if some fail
            
            self._state = PoolState.HEALTHY
            self.logger.info(f"Initialized async pool '{self.name}' with {self._pool.qsize()} connections")
            
            # Start health checking
            if self.config.health_check_interval > 0:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            
        except Exception as e:
            self._state = PoolState.CRITICAL
            self.logger.error(f"Failed to initialize async pool '{self.name}': {e}")
            raise
    
    async def _create_connection(self) -> ConnectionWrapper:
        """Create a new connection."""
        async with self._creation_semaphore:
            try:
                connection = await self.factory.create_connection()
                wrapper = ConnectionWrapper(
                    connection=connection,
                    pool_id=self.name
                )
                
                if self.metrics:
                    self.metrics.record_creation()
                
                self.logger.debug(f"Created new async connection {id(wrapper)}")
                return wrapper
                
            except Exception as e:
                if self.metrics:
                    self.metrics.record_failure()
                self.logger.error(f"Failed to create async connection: {e}")
                raise
    
    async def _validate_connection(self, wrapper: ConnectionWrapper) -> bool:
        """Validate connection and update state."""
        try:
            # Check if connection is too old
            if wrapper.is_stale(self.config.pool_recycle):
                wrapper.state = ConnectionState.STALE
                return False
            
            # Ping connection if enabled
            if self.config.pool_pre_ping:
                if not await self.factory.validate_connection(wrapper.connection):
                    wrapper.state = ConnectionState.BROKEN
                    return False
            
            wrapper.state = ConnectionState.IDLE
            return True
            
        except Exception as e:
            self.logger.warning(f"Async connection validation failed: {e}")
            wrapper.state = ConnectionState.BROKEN
            return False
    
    async def _close_connection(self, wrapper: ConnectionWrapper) -> None:
        """Close a connection."""
        try:
            await self.factory.close_connection(wrapper.connection)
            wrapper.state = ConnectionState.RETIRED
            
            if self.metrics:
                self.metrics.record_closure()
            
            self.logger.debug(f"Closed async connection {id(wrapper)}")
            
        except Exception as e:
            self.logger.warning(f"Error closing async connection: {e}")
    
    async def _reset_connection(self, wrapper: ConnectionWrapper) -> None:
        """Reset connection state."""
        try:
            if self.config.pool_reset_on_return != "none":
                await self.factory.reset_connection(wrapper.connection)
        except Exception as e:
            self.logger.warning(f"Error resetting async connection: {e}")
            wrapper.state = ConnectionState.BROKEN
    
    async def get_connection(self, timeout: Optional[float] = None) -> ConnectionWrapper:
        """Get a connection from the pool.
        
        Args:
            timeout: Checkout timeout in seconds
            
        Returns:
            Connection wrapper
            
        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If pool is shutdown
        """
        await self._ensure_initialized()
        
        if self._state == PoolState.SHUTDOWN:
            raise RuntimeError("Async connection pool is shutdown")
        
        start_time = time.time()
        effective_timeout = timeout or self.config.pool_timeout
        
        try:
            # Try to get connection from pool with timeout
            try:
                wrapper = await asyncio.wait_for(
                    self._pool.get(), 
                    timeout=effective_timeout
                )
                
                # Validate connection
                if await self._validate_connection(wrapper):
                    async with self._lock:
                        wrapper.state = ConnectionState.ACTIVE
                        wrapper.touch()
                        self._active_connections[id(wrapper)] = wrapper
                    
                    if self.metrics:
                        checkout_time = time.time() - start_time
                        self.metrics.record_checkout(checkout_time)
                    
                    return wrapper
                else:
                    # Connection is invalid, close it and try again
                    await self._close_connection(wrapper)
                    # Fall through to create new connection
                    
            except asyncio.TimeoutError:
                pass  # No connection available, try to create new one
            
            # Try to create new connection
            async with self._lock:
                total_connections = len(self._active_connections) + self._pool.qsize()
                
                if total_connections < self.config.max_size:
                    # Can create new connection within pool limit
                    wrapper = await self._create_connection()
                    wrapper.state = ConnectionState.ACTIVE
                    wrapper.touch()
                    self._active_connections[id(wrapper)] = wrapper
                    
                    if self.metrics:
                        checkout_time = time.time() - start_time
                        self.metrics.record_checkout(checkout_time)
                    
                    return wrapper
                
                elif len(self._overflow_connections) < self.config.max_overflow:
                    # Create overflow connection
                    wrapper = await self._create_connection()
                    wrapper.state = ConnectionState.ACTIVE
                    wrapper.touch()
                    self._overflow_connections.add(wrapper)
                    
                    if self.metrics:
                        self.metrics.record_overflow()
                        checkout_time = time.time() - start_time
                        self.metrics.record_checkout(checkout_time)
                    
                    return wrapper
            
            # Pool is full and no overflow allowed
            if self.metrics:
                self.metrics.record_timeout()
            raise TimeoutError(f"Async pool checkout timed out after {effective_timeout}s")
                
        except Exception as e:
            if self.metrics:
                self.metrics.record_failure()
            raise
    
    async def return_connection(self, wrapper: ConnectionWrapper, discard: bool = False) -> None:
        """Return a connection to the pool.
        
        Args:
            wrapper: Connection wrapper to return
            discard: Whether to discard the connection
        """
        async with self._lock:
            # Remove from active connections
            self._active_connections.pop(id(wrapper), None)
            
            if discard or wrapper.state == ConnectionState.BROKEN:
                # Discard broken or unwanted connection
                await self._close_connection(wrapper)
                
                # Replace with new connection if below minimum
                if (self._pool.qsize() + len(self._active_connections) < self.config.min_size and
                    self._state != PoolState.SHUTDOWN):
                    try:
                        new_wrapper = await self._create_connection()
                        await self._pool.put(new_wrapper)
                    except Exception as e:
                        self.logger.warning(f"Failed to replace discarded async connection: {e}")
                
                if self.metrics:
                    self.metrics.record_checkin()
                return
            
            # Check if this is an overflow connection
            if wrapper in self._overflow_connections:
                self._overflow_connections.discard(wrapper)
                await self._close_connection(wrapper)
                if self.metrics:
                    self.metrics.record_checkin()
                return
            
            # Reset connection state
            try:
                await self._reset_connection(wrapper)
                
                if wrapper.state != ConnectionState.BROKEN:
                    # Connection is good, return to pool
                    wrapper.state = ConnectionState.IDLE
                    
                    try:
                        await self._pool.put(wrapper)
                        if self.metrics:
                            self.metrics.record_checkin()
                    except asyncio.QueueFull:
                        # Pool is full, close excess connection
                        await self._close_connection(wrapper)
                else:
                    # Connection is broken after reset
                    await self._close_connection(wrapper)
                    
            except Exception as e:
                self.logger.warning(f"Error returning async connection: {e}")
                await self._close_connection(wrapper)
            
            if self.metrics:
                self.metrics.record_checkin()
    
    @asynccontextmanager
    async def connection(self, timeout: Optional[float] = None) -> AsyncIterator[Any]:
        """Async context manager for getting/returning connections.
        
        Args:
            timeout: Checkout timeout in seconds
            
        Yields:
            Database connection
        """
        wrapper = None
        try:
            wrapper = await self.get_connection(timeout)
            yield wrapper.connection
        except Exception:
            if wrapper:
                await self.return_connection(wrapper, discard=True)
            raise
        else:
            if wrapper:
                await self.return_connection(wrapper)
    
    async def _health_check_loop(self) -> None:
        """Health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in async health check loop: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_health_check(self) -> None:
        """Perform health check on pool connections."""
        if self._shutdown_event.is_set():
            return
        
        try:
            async with self._lock:
                # Check idle connections in pool
                connections_to_remove = []
                temp_connections = []
                
                # Get all connections from pool
                while not self._pool.empty():
                    try:
                        wrapper = self._pool.get_nowait()
                        
                        if (wrapper.is_stale(self.config.pool_recycle) or 
                            wrapper.is_idle_too_long(self.config.max_idle_time) or
                            not await self._validate_connection(wrapper)):
                            
                            connections_to_remove.append(wrapper)
                            if self.metrics:
                                self.metrics.record_recycle()
                        else:
                            temp_connections.append(wrapper)
                    except asyncio.QueueEmpty:
                        break
                
                # Put valid connections back
                for wrapper in temp_connections:
                    try:
                        self._pool.put_nowait(wrapper)
                    except asyncio.QueueFull:
                        connections_to_remove.append(wrapper)
                
                # Close invalid connections
                for wrapper in connections_to_remove:
                    await self._close_connection(wrapper)
                
                # Maintain minimum pool size
                current_size = self._pool.qsize() + len(self._active_connections)
                needed = max(0, self.config.min_size - current_size)
                
                for _ in range(needed):
                    try:
                        new_wrapper = await self._create_connection()
                        await self._pool.put(new_wrapper)
                    except Exception as e:
                        self.logger.warning(f"Failed to create replacement async connection: {e}")
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
            self.logger.error(f"Error during async health check: {e}")
            if self.metrics:
                self.metrics.record_health_check_failure()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        pool_size = self._pool.qsize()
        active_size = len(self._active_connections)
        overflow_size = len(self._overflow_connections)
        
        stats = {
            'pool_name': self.name,
            'pool_type': 'async',
            'state': self._state.value,
            'pool_size': pool_size,
            'active_connections': active_size,
            'overflow_connections': overflow_size,
            'total_connections': pool_size + active_size + overflow_size,
            'initialized': self._initialized,
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
    
    async def resize(self, min_size: int, max_size: int) -> None:
        """Resize pool limits.
        
        Args:
            min_size: New minimum pool size
            max_size: New maximum pool size
        """
        async with self._lock:
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
                        wrapper = await self._create_connection()
                        await self._pool.put(wrapper)
                    except Exception as e:
                        self.logger.warning(f"Failed to create connection during async resize: {e}")
                        break
            
            elif self._pool.qsize() > max_size:
                # Need to remove excess idle connections
                excess = self._pool.qsize() - max_size
                removed_connections = []
                
                for _ in range(excess):
                    try:
                        wrapper = self._pool.get_nowait()
                        removed_connections.append(wrapper)
                    except asyncio.QueueEmpty:
                        break
                
                # Close removed connections
                for wrapper in removed_connections:
                    await self._close_connection(wrapper)
            
            self.logger.info(
                f"Resized async pool '{self.name}' from ({old_min}, {old_max}) to ({min_size}, {max_size})"
            )
    
    async def invalidate(self) -> None:
        """Invalidate all connections in the pool."""
        async with self._lock:
            # Close all idle connections
            while not self._pool.empty():
                try:
                    wrapper = self._pool.get_nowait()
                    await self._close_connection(wrapper)
                except asyncio.QueueEmpty:
                    break
            
            # Mark active connections for discard on return
            for wrapper in self._active_connections.values():
                wrapper.state = ConnectionState.BROKEN
            
            # Close overflow connections
            overflow_list = list(self._overflow_connections)
            for wrapper in overflow_list:
                self._overflow_connections.discard(wrapper)
                await self._close_connection(wrapper)
            
            self.logger.info(f"Invalidated all connections in async pool '{self.name}'")
    
    async def close(self) -> None:
        """Close the async connection pool."""
        self.logger.info(f"Closing async connection pool '{self.name}'")
        
        # Signal shutdown
        self._state = PoolState.SHUTDOWN
        self._shutdown_event.set()
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            # Close all idle connections
            while not self._pool.empty():
                try:
                    wrapper = self._pool.get_nowait()
                    await self._close_connection(wrapper)
                except asyncio.QueueEmpty:
                    break
            
            # Close active connections (they will be discarded when returned)
            active_list = list(self._active_connections.values())
            for wrapper in active_list:
                wrapper.state = ConnectionState.BROKEN
            
            # Close overflow connections
            overflow_list = list(self._overflow_connections)
            for wrapper in overflow_list:
                self._overflow_connections.discard(wrapper)
                await self._close_connection(wrapper)
        
        self.logger.info(f"Closed async connection pool '{self.name}'")
    
    async def __aenter__(self):
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class AsyncPoolManager:
    """Manager for multiple async connection pools."""
    
    def __init__(self):
        self.pools: Dict[str, AsyncConnectionPool] = {}
        self._lock = asyncio.Lock()
        self.logger = logger.getChild(self.__class__.__name__)
    
    async def create_pool(self, 
                         name: str, 
                         factory: AsyncConnectionFactory,
                         config: Optional[PoolConfig] = None) -> AsyncConnectionPool:
        """Create a new async connection pool.
        
        Args:
            name: Pool name
            factory: Connection factory
            config: Pool configuration
            
        Returns:
            Created connection pool
        """
        async with self._lock:
            if name in self.pools:
                raise ValueError(f"Async pool '{name}' already exists")
            
            pool = AsyncConnectionPool(factory, config, name)
            self.pools[name] = pool
            
            self.logger.info(f"Created async pool '{name}'")
            return pool
    
    async def get_pool(self, name: str) -> AsyncConnectionPool:
        """Get async connection pool by name.
        
        Args:
            name: Pool name
            
        Returns:
            Connection pool
            
        Raises:
            KeyError: If pool not found
        """
        async with self._lock:
            if name not in self.pools:
                raise KeyError(f"Async pool '{name}' not found")
            return self.pools[name]
    
    async def remove_pool(self, name: str) -> None:
        """Remove and close async connection pool.
        
        Args:
            name: Pool name
        """
        async with self._lock:
            if name in self.pools:
                pool = self.pools.pop(name)
                await pool.close()
                self.logger.info(f"Removed async pool '{name}'")
    
    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all async pools.
        
        Returns:
            Dictionary mapping pool names to statistics
        """
        async with self._lock:
            return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all async pools.
        
        Returns:
            Dictionary mapping pool names to health status
        """
        results = {}
        async with self._lock:
            for name, pool in self.pools.items():
                try:
                    stats = pool.get_stats()
                    results[name] = stats['state'] in ['healthy', 'degraded']
                except Exception as e:
                    self.logger.error(f"Async health check failed for pool '{name}': {e}")
                    results[name] = False
        return results
    
    async def close_all(self) -> None:
        """Close all async connection pools."""
        async with self._lock:
            pool_names = list(self.pools.keys())
            for name in pool_names:
                await self.remove_pool(name)
        
        self.logger.info("Closed all async connection pools")


# Global async pool manager
_async_pool_manager: Optional[AsyncPoolManager] = None

async def get_async_pool_manager() -> AsyncPoolManager:
    """Get global async pool manager instance.
    
    Returns:
        Async pool manager instance
    """
    global _async_pool_manager
    if _async_pool_manager is None:
        _async_pool_manager = AsyncPoolManager()
    return _async_pool_manager