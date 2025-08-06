"""High-performance memory caching with LRU eviction and TTL support."""

import asyncio
import time
import threading
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import gc
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')

class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    RANDOM = "random"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self) -> None:
        """Update access information."""
        self.last_accessed = time.time()
        self.access_count += 1

class MemoryCache(Generic[T]):
    """High-performance memory cache with advanced features."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: Optional[float] = None,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 enable_stats: bool = True,
                 cleanup_interval: float = 60.0,
                 max_memory_mb: Optional[int] = None):
        """Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            eviction_policy: Cache eviction policy
            enable_stats: Enable statistics collection
            cleanup_interval: Background cleanup interval in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._eviction_policy = eviction_policy
        self._enable_stats = enable_stats
        self._cleanup_interval = cleanup_interval
        self._max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_cleanups': 0,
            'memory_cleanups': 0,
            'total_gets': 0,
            'total_sets': 0
        }
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_cleanup = False
        
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Start background cleanup if running in async context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._start_background_cleanup()
        except RuntimeError:
            # Not in async context, cleanup will be manual
            pass
    
    def _start_background_cleanup(self) -> None:
        """Start background cleanup task."""
        async def cleanup_task():
            while not self._stop_cleanup:
                try:
                    self.cleanup_expired()
                    await asyncio.sleep(self._cleanup_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in background cleanup: {e}")
                    await asyncio.sleep(self._cleanup_interval)
        
        self._cleanup_task = asyncio.create_task(cleanup_task())
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if self._enable_stats:
                self._stats['total_gets'] += 1
            
            if key not in self._cache:
                if self._enable_stats:
                    self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                if self._enable_stats:
                    self._stats['misses'] += 1
                    self._stats['expired_cleanups'] += 1
                return None
            
            # Update access info and move to end (LRU)
            entry.touch()
            if self._eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(key)
            
            if self._enable_stats:
                self._stats['hits'] += 1
            
            return entry.value
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if set successfully
        """
        with self._lock:
            if self._enable_stats:
                self._stats['total_sets'] += 1
            
            # Calculate entry size
            try:
                import sys
                size_bytes = sys.getsizeof(value)
                if hasattr(value, '__dict__'):
                    size_bytes += sys.getsizeof(value.__dict__)
            except Exception:
                size_bytes = 0
            
            # Check memory limit
            if self._max_memory_bytes and self._get_total_memory() + size_bytes > self._max_memory_bytes:
                self._evict_for_memory()
            
            # Create entry
            effective_ttl = ttl or self._default_ttl
            entry = CacheEntry(
                value=value,
                ttl=effective_ttl,
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = entry
            
            # Evict if over size limit
            if len(self._cache) > self._max_size:
                self._evict_one()
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            if self._enable_stats:
                self._stats['evictions'] += len(self._cache)
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and is valid
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return False
            
            return True
    
    def keys(self) -> List[str]:
        """Get all valid cache keys.
        
        Returns:
            List of cache keys
        """
        with self._lock:
            valid_keys = []
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
                else:
                    valid_keys.append(key)
            
            # Clean up expired keys
            for key in expired_keys:
                del self._cache[key]
            
            return valid_keys
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries.
        
        Returns:
            Number of entries cleaned up
        """
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            if self._enable_stats:
                self._stats['expired_cleanups'] += len(expired_keys)
            
            return len(expired_keys)
    
    def _evict_one(self) -> None:
        """Evict one entry based on eviction policy."""
        if not self._cache:
            return
        
        if self._eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used (first in OrderedDict)
            key = next(iter(self._cache))
            del self._cache[key]
        
        elif self._eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            min_access_count = min(entry.access_count for entry in self._cache.values())
            for key, entry in self._cache.items():
                if entry.access_count == min_access_count:
                    del self._cache[key]
                    break
        
        elif self._eviction_policy == EvictionPolicy.TTL:
            # Remove entry with shortest remaining TTL
            current_time = time.time()
            min_remaining_ttl = float('inf')
            key_to_remove = None
            
            for key, entry in self._cache.items():
                if entry.ttl:
                    remaining_ttl = (entry.created_at + entry.ttl) - current_time
                    if remaining_ttl < min_remaining_ttl:
                        min_remaining_ttl = remaining_ttl
                        key_to_remove = key
            
            if key_to_remove:
                del self._cache[key_to_remove]
            else:
                # Fallback to LRU
                key = next(iter(self._cache))
                del self._cache[key]
        
        elif self._eviction_policy == EvictionPolicy.RANDOM:
            # Remove random entry
            import random
            key = random.choice(list(self._cache.keys()))
            del self._cache[key]
        
        if self._enable_stats:
            self._stats['evictions'] += 1
    
    def _evict_for_memory(self) -> None:
        """Evict entries to free memory."""
        target_size = int(self._max_memory_bytes * 0.8)  # Target 80% of max memory
        
        while self._get_total_memory() > target_size and self._cache:
            self._evict_one()
            if self._enable_stats:
                self._stats['memory_cleanups'] += 1
    
    def _get_total_memory(self) -> int:
        """Get total memory usage of cache.
        
        Returns:
            Total memory in bytes
        """
        return sum(entry.size_bytes for entry in self._cache.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_ratio = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0.0
            
            return {
                'entries': len(self._cache),
                'max_size': self._max_size,
                'memory_usage_bytes': self._get_total_memory(),
                'max_memory_bytes': self._max_memory_bytes,
                'hit_ratio': hit_ratio,
                'eviction_policy': self._eviction_policy.value,
                **self._stats
            }
    
    def resize(self, new_max_size: int) -> None:
        """Resize cache maximum size.
        
        Args:
            new_max_size: New maximum size
        """
        with self._lock:
            self._max_size = new_max_size
            
            # Evict entries if over new limit
            while len(self._cache) > self._max_size:
                self._evict_one()
    
    def set_ttl(self, key: str, ttl: float) -> bool:
        """Set TTL for existing key.
        
        Args:
            key: Cache key
            ttl: New TTL in seconds
            
        Returns:
            True if TTL was set
        """
        with self._lock:
            if key in self._cache:
                self._cache[key].ttl = ttl
                self._cache[key].created_at = time.time()  # Reset creation time
                return True
            return False
    
    def extend_ttl(self, key: str, additional_seconds: float) -> bool:
        """Extend TTL for existing key.
        
        Args:
            key: Cache key
            additional_seconds: Additional seconds to add to TTL
            
        Returns:
            True if TTL was extended
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.ttl:
                    entry.ttl += additional_seconds
                return True
            return False
    
    def get_with_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value with metadata.
        
        Args:
            key: Cache key
            
        Returns:
            Dictionary with value and metadata or None
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            entry.touch()
            if self._eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(key)
            
            return {
                'value': entry.value,
                'created_at': entry.created_at,
                'last_accessed': entry.last_accessed,
                'access_count': entry.access_count,
                'ttl': entry.ttl,
                'size_bytes': entry.size_bytes,
                'age_seconds': time.time() - entry.created_at
            }
    
    def close(self) -> None:
        """Close cache and cleanup resources."""
        self._stop_cleanup = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        with self._lock:
            self._cache.clear()


def cache_decorator(cache: MemoryCache, ttl: Optional[float] = None, 
                   key_func: Optional[Callable] = None):
    """Decorator for caching function results.
    
    Args:
        cache: Cache instance to use
        ttl: TTL for cached results
        key_func: Function to generate cache key
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        wrapper.cache = cache
        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    
    return decorator


# Global memory cache instance
_memory_cache: Optional[MemoryCache] = None

def get_memory_cache() -> MemoryCache:
    """Get global memory cache instance.
    
    Returns:
        Memory cache instance
    """
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = MemoryCache(
            max_size=10000,
            default_ttl=3600.0,
            eviction_policy=EvictionPolicy.LRU,
            enable_stats=True,
            cleanup_interval=300.0,
            max_memory_mb=100
        )
    return _memory_cache