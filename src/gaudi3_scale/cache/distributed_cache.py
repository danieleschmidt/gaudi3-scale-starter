"""Distributed caching with L2 cache support and advanced features."""

import asyncio
import hashlib
import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading

from .memory_cache import MemoryCache, EvictionPolicy
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)

class ConsistencyLevel(Enum):
    """Cache consistency levels."""
    EVENTUAL = "eventual"
    STRONG = "strong"
    WEAK = "weak"

class CacheStrategy(Enum):
    """Cache strategy types."""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"

@dataclass
class CacheConfig:
    """Configuration for distributed cache."""
    l1_max_size: int = 1000
    l1_ttl: float = 300.0
    l2_ttl: float = 3600.0
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH
    enable_compression: bool = True
    compression_threshold: int = 1024  # bytes
    enable_batching: bool = True
    batch_size: int = 100
    batch_timeout: float = 0.1  # seconds

class DistributedCache:
    """Multi-level distributed cache with L1 (memory) and L2 (Redis) layers."""
    
    def __init__(self, 
                 config: Optional[CacheConfig] = None,
                 redis_cache: Optional[RedisCache] = None,
                 memory_cache: Optional[MemoryCache] = None):
        """Initialize distributed cache.
        
        Args:
            config: Cache configuration
            redis_cache: Redis cache instance
            memory_cache: Memory cache instance
        """
        self.config = config or CacheConfig()
        
        # L1 Cache (Memory)
        self.l1_cache = memory_cache or MemoryCache(
            max_size=self.config.l1_max_size,
            default_ttl=self.config.l1_ttl,
            eviction_policy=EvictionPolicy.LRU,
            enable_stats=True
        )
        
        # L2 Cache (Redis)
        self.l2_cache = redis_cache or RedisCache()
        
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Statistics
        self._stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'total_gets': 0,
            'total_sets': 0,
            'cache_refreshes': 0,
            'compression_saves': 0,
            'batch_operations': 0
        }
        
        # Batch operations
        self._batch_operations: List[Tuple[str, str, Any, Optional[float]]] = []
        self._batch_lock = threading.Lock()
        self._batch_timer: Optional[threading.Timer] = None
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="cache_")
        
        # Cache invalidation tracking
        self._invalidation_callbacks: Dict[str, List[Callable]] = {}
        
        # Compression support
        self._enable_compression = self.config.enable_compression
        self._compression_threshold = self.config.compression_threshold
        
        if self._enable_compression:
            try:
                import zlib
                self._compressor = zlib
            except ImportError:
                self.logger.warning("zlib not available, disabling compression")
                self._enable_compression = False
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if beneficial.
        
        Args:
            data: Data to compress
            
        Returns:
            Compressed data or original if not beneficial
        """
        if not self._enable_compression or len(data) < self._compression_threshold:
            return data
        
        try:
            compressed = self._compressor.compress(data)
            if len(compressed) < len(data) * 0.9:  # Only use if >10% savings
                self._stats['compression_saves'] += 1
                return b'compressed:' + compressed
            return data
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if compressed.
        
        Args:
            data: Data to decompress
            
        Returns:
            Decompressed data
        """
        if not self._enable_compression or not data.startswith(b'compressed:'):
            return data
        
        try:
            compressed_data = data[11:]  # Remove 'compressed:' prefix
            return self._compressor.decompress(compressed_data)
        except Exception as e:
            self.logger.warning(f"Decompression failed: {e}")
            return data
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with L1/L2 lookup.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        self._stats['total_gets'] += 1
        
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            self._stats['l1_hits'] += 1
            return value
        
        self._stats['l1_misses'] += 1
        
        # Try L2 cache
        l2_value = self.l2_cache.get(key)
        if l2_value is not None:
            self._stats['l2_hits'] += 1
            
            # Store in L1 for faster future access
            self.l1_cache.set(key, l2_value, ttl=self.config.l1_ttl)
            return l2_value
        
        self._stats['l2_misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache with L1/L2 storage.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if set successfully
        """
        self._stats['total_sets'] += 1
        
        effective_ttl = ttl or self.config.l2_ttl
        
        if self.config.strategy == CacheStrategy.WRITE_THROUGH:
            # Write to both L1 and L2 immediately
            l1_success = self.l1_cache.set(key, value, ttl=min(effective_ttl, self.config.l1_ttl))
            l2_success = self.l2_cache.set(key, value, ttl=int(effective_ttl))
            
            # Trigger invalidation callbacks
            self._trigger_invalidation(key)
            
            return l1_success and l2_success
        
        elif self.config.strategy == CacheStrategy.WRITE_BACK:
            # Write to L1 immediately, L2 asynchronously
            l1_success = self.l1_cache.set(key, value, ttl=min(effective_ttl, self.config.l1_ttl))
            
            if self.config.enable_batching:
                self._add_to_batch('set', key, value, effective_ttl)
            else:
                # Async write to L2
                self._executor.submit(self.l2_cache.set, key, value, int(effective_ttl))
            
            return l1_success
        
        elif self.config.strategy == CacheStrategy.WRITE_AROUND:
            # Write only to L2, bypass L1
            l2_success = self.l2_cache.set(key, value, ttl=int(effective_ttl))
            self._trigger_invalidation(key)
            return l2_success
        
        return False
    
    def _add_to_batch(self, operation: str, key: str, value: Any, ttl: Optional[float]) -> None:
        """Add operation to batch queue.
        
        Args:
            operation: Operation type
            key: Cache key
            value: Value
            ttl: TTL in seconds
        """
        with self._batch_lock:
            self._batch_operations.append((operation, key, value, ttl))
            
            if len(self._batch_operations) >= self.config.batch_size:
                self._flush_batch()
            elif self._batch_timer is None:
                self._batch_timer = threading.Timer(self.config.batch_timeout, self._flush_batch)
                self._batch_timer.start()
    
    def _flush_batch(self) -> None:
        """Flush batched operations to L2 cache."""
        with self._batch_lock:
            if not self._batch_operations:
                return
            
            operations = self._batch_operations.copy()
            self._batch_operations.clear()
            
            if self._batch_timer:
                self._batch_timer.cancel()
                self._batch_timer = None
        
        # Execute batched operations
        try:
            for operation, key, value, ttl in operations:
                if operation == 'set':
                    self.l2_cache.set(key, value, ttl=int(ttl) if ttl else None)
                elif operation == 'delete':
                    self.l2_cache.delete(key)
            
            self._stats['batch_operations'] += 1
            self.logger.debug(f"Flushed batch of {len(operations)} operations")
        
        except Exception as e:
            self.logger.error(f"Error flushing batch operations: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from both cache levels.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted from at least one level
        """
        l1_deleted = self.l1_cache.delete(key)
        
        if self.config.enable_batching and self.config.strategy == CacheStrategy.WRITE_BACK:
            self._add_to_batch('delete', key, None, None)
            l2_deleted = True  # Assume will succeed
        else:
            l2_deleted = self.l2_cache.delete(key)
        
        self._trigger_invalidation(key)
        return l1_deleted or l2_deleted
    
    def exists(self, key: str) -> bool:
        """Check if key exists in any cache level.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        return self.l1_cache.exists(key) or self.l2_cache.exists(key)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern.
        
        Args:
            pattern: Key pattern with wildcards
            
        Returns:
            Number of keys invalidated
        """
        # Get matching keys from L2
        matching_keys = self.l2_cache.scan_keys(pattern)
        
        count = 0
        for key in matching_keys:
            if self.delete(key):
                count += 1
        
        return count
    
    def bulk_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values efficiently.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to values
        """
        result = {}
        missing_keys = []
        
        # Check L1 cache first
        for key in keys:
            value = self.l1_cache.get(key)
            if value is not None:
                result[key] = value
                self._stats['l1_hits'] += 1
            else:
                missing_keys.append(key)
                self._stats['l1_misses'] += 1
        
        # Batch get from L2 for missing keys
        if missing_keys:
            for key in missing_keys:
                l2_value = self.l2_cache.get(key)
                if l2_value is not None:
                    result[key] = l2_value
                    self._stats['l2_hits'] += 1
                    # Cache in L1
                    self.l1_cache.set(key, l2_value, ttl=self.config.l1_ttl)
                else:
                    self._stats['l2_misses'] += 1
        
        return result
    
    def bulk_set(self, data: Dict[str, Any], ttl: Optional[float] = None) -> Dict[str, bool]:
        """Set multiple values efficiently.
        
        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
            
        Returns:
            Dictionary mapping keys to success status
        """
        results = {}
        
        for key, value in data.items():
            results[key] = self.set(key, value, ttl)
        
        return results
    
    def refresh_cache(self, key: str, loader_func: Callable[[], Any], 
                     ttl: Optional[float] = None) -> Any:
        """Refresh cache entry using loader function.
        
        Args:
            key: Cache key
            loader_func: Function to load fresh data
            ttl: Time to live in seconds
            
        Returns:
            Fresh value from loader function
        """
        try:
            fresh_value = loader_func()
            self.set(key, fresh_value, ttl)
            self._stats['cache_refreshes'] += 1
            return fresh_value
        except Exception as e:
            self.logger.error(f"Error refreshing cache for key {key}: {e}")
            # Return stale value if available
            return self.get(key)
    
    def get_or_set(self, key: str, loader_func: Callable[[], Any], 
                   ttl: Optional[float] = None) -> Any:
        """Get value or set using loader function if not found.
        
        Args:
            key: Cache key
            loader_func: Function to load data if not cached
            ttl: Time to live in seconds
            
        Returns:
            Cached or fresh value
        """
        value = self.get(key)
        if value is not None:
            return value
        
        # Load and cache
        fresh_value = loader_func()
        self.set(key, fresh_value, ttl)
        return fresh_value
    
    def add_invalidation_callback(self, key_pattern: str, callback: Callable[[str], None]) -> None:
        """Add callback for cache invalidation events.
        
        Args:
            key_pattern: Pattern to match keys
            callback: Callback function
        """
        if key_pattern not in self._invalidation_callbacks:
            self._invalidation_callbacks[key_pattern] = []
        self._invalidation_callbacks[key_pattern].append(callback)
    
    def _trigger_invalidation(self, key: str) -> None:
        """Trigger invalidation callbacks for key.
        
        Args:
            key: Cache key that was invalidated
        """
        for pattern, callbacks in self._invalidation_callbacks.items():
            # Simple pattern matching (could be enhanced)
            if pattern == key or (pattern.endswith('*') and key.startswith(pattern[:-1])):
                for callback in callbacks:
                    try:
                        callback(key)
                    except Exception as e:
                        self.logger.error(f"Error in invalidation callback: {e}")
    
    def warm_up(self, data: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Warm up cache with initial data.
        
        Args:
            data: Dictionary of key-value pairs to warm up
            ttl: Time to live in seconds
        """
        self.bulk_set(data, ttl)
        self.logger.info(f"Warmed up cache with {len(data)} entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        total_requests = self._stats['total_gets']
        l1_hit_ratio = (self._stats['l1_hits'] / total_requests * 100) if total_requests > 0 else 0.0
        l2_hit_ratio = (self._stats['l2_hits'] / total_requests * 100) if total_requests > 0 else 0.0
        overall_hit_ratio = ((self._stats['l1_hits'] + self._stats['l2_hits']) / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'overall': {
                'total_requests': total_requests,
                'overall_hit_ratio': overall_hit_ratio,
                'l1_hit_ratio': l1_hit_ratio,
                'l2_hit_ratio': l2_hit_ratio,
                **self._stats
            },
            'l1_cache': l1_stats,
            'l2_cache': l2_stats,
            'config': {
                'consistency_level': self.config.consistency_level.value,
                'strategy': self.config.strategy.value,
                'compression_enabled': self._enable_compression,
                'batching_enabled': self.config.enable_batching
            }
        }
    
    def clear_all(self) -> None:
        """Clear all cache levels."""
        self.l1_cache.clear()
        # Note: Not clearing L2 cache to preserve shared data
        self.logger.warning("Cleared L1 cache (L2 cache preserved)")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache systems.
        
        Returns:
            Health check results
        """
        health = {
            'l1_cache': {'healthy': True, 'error': None},
            'l2_cache': {'healthy': True, 'error': None},
            'overall': {'healthy': True}
        }
        
        # Test L1 cache
        try:
            test_key = f"health_check_{time.time()}"
            self.l1_cache.set(test_key, "test", ttl=1.0)
            result = self.l1_cache.get(test_key)
            if result != "test":
                raise Exception("L1 cache test failed")
            self.l1_cache.delete(test_key)
        except Exception as e:
            health['l1_cache'] = {'healthy': False, 'error': str(e)}
            health['overall']['healthy'] = False
        
        # Test L2 cache
        try:
            test_key = f"health_check_{time.time()}"
            self.l2_cache.set(test_key, "test", ttl=1)
            result = self.l2_cache.get(test_key)
            if result != "test":
                raise Exception("L2 cache test failed")
            self.l2_cache.delete(test_key)
        except Exception as e:
            health['l2_cache'] = {'healthy': False, 'error': str(e)}
            health['overall']['healthy'] = False
        
        return health
    
    def close(self) -> None:
        """Close cache and cleanup resources."""
        self._flush_batch()  # Flush any pending operations
        self.l1_cache.close()
        self._executor.shutdown(wait=True)
        
        if self._batch_timer:
            self._batch_timer.cancel()


# Global distributed cache instance
_distributed_cache: Optional[DistributedCache] = None

def get_distributed_cache() -> DistributedCache:
    """Get global distributed cache instance.
    
    Returns:
        Distributed cache instance
    """
    global _distributed_cache
    if _distributed_cache is None:
        _distributed_cache = DistributedCache()
    return _distributed_cache