"""Advanced memory optimization and garbage collection tuning system."""

import gc
import sys
import os
import time
import threading
import asyncio
import logging
import weakref
import psutil
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import tracemalloc
import resource
from contextlib import contextmanager
import ctypes
import mmap

logger = logging.getLogger(__name__)

class MemoryStrategy(Enum):
    """Memory optimization strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

class GCMode(Enum):
    """Garbage collection modes."""
    AUTO = "auto"
    MANUAL = "manual"
    ADAPTIVE = "adaptive"
    DISABLED = "disabled"

@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    # Memory limits
    max_memory_mb: Optional[int] = None
    warning_threshold: float = 0.8
    critical_threshold: float = 0.9
    
    # Garbage collection
    gc_strategy: MemoryStrategy = MemoryStrategy.BALANCED
    gc_mode: GCMode = GCMode.ADAPTIVE
    gc_threshold_0: int = 700
    gc_threshold_1: int = 10
    gc_threshold_2: int = 10
    auto_gc_interval: float = 60.0
    
    # Object pool settings
    enable_object_pools: bool = True
    pool_sizes: Dict[str, int] = field(default_factory=lambda: {
        'small': 1000,
        'medium': 500,
        'large': 100
    })
    
    # Memory profiling
    enable_memory_tracking: bool = True
    track_allocations: bool = False
    allocation_limit: int = 10000
    
    # Cache optimization
    enable_memory_cache_compression: bool = False
    cache_cleanup_interval: float = 300.0
    
    # Advanced settings
    enable_memory_mapping: bool = False
    mmap_threshold_mb: int = 50
    enable_copy_on_write: bool = True

class MemoryStats:
    """Memory usage statistics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.total_allocated = 0
        self.total_freed = 0
        self.peak_usage = 0
        self.current_usage = 0
        self.gc_collections = [0, 0, 0]
        self.gc_time_spent = 0.0
        self.objects_created = 0
        self.objects_destroyed = 0
        self.memory_leaks_detected = 0
        self.pool_hits = 0
        self.pool_misses = 0
    
    def update_from_gc_stats(self):
        """Update statistics from garbage collector."""
        try:
            gc_stats = gc.get_stats()
            for i, stats in enumerate(gc_stats):
                self.gc_collections[i] = stats.get('collections', 0)
        except Exception:
            pass
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent(),
                'peak_mb': self.peak_usage / (1024 * 1024),
                'available_mb': psutil.virtual_memory().available / (1024 * 1024)
            }
        except Exception as e:
            logger.warning(f"Error getting memory usage: {e}")
            return {}

class ObjectPool:
    """Generic object pool for memory optimization."""
    
    def __init__(self, factory: Callable, max_size: int = 100, reset_func: Optional[Callable] = None):
        self.factory = factory
        self.max_size = max_size
        self.reset_func = reset_func
        self._pool = deque(maxlen=max_size)
        self._created_count = 0
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
    
    def get(self):
        """Get object from pool or create new one."""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._hits += 1
                return obj
            else:
                obj = self.factory()
                self._created_count += 1
                self._misses += 1
                return obj
    
    def put(self, obj):
        """Return object to pool."""
        if obj is None:
            return
        
        with self._lock:
            if len(self._pool) < self.max_size:
                if self.reset_func:
                    try:
                        self.reset_func(obj)
                    except Exception as e:
                        logger.warning(f"Error resetting pooled object: {e}")
                        return
                
                self._pool.append(obj)
    
    @contextmanager
    def get_object(self):
        """Context manager for automatic object return."""
        obj = self.get()
        try:
            yield obj
        finally:
            self.put(obj)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            
            return {
                'size': len(self._pool),
                'max_size': self.max_size,
                'created_count': self._created_count,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate
            }
    
    def clear(self):
        """Clear all objects from pool."""
        with self._lock:
            self._pool.clear()

class MemoryLeakDetector:
    """Memory leak detection and reporting."""
    
    def __init__(self, sample_interval: float = 60.0):
        self.sample_interval = sample_interval
        self._snapshots = deque(maxlen=100)
        self._object_counts = defaultdict(int)
        self._weak_refs = weakref.WeakSet()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    def start_monitoring(self):
        """Start memory leak monitoring."""
        if not self._monitoring:
            self._monitoring = True
            if tracemalloc.is_tracing():
                self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Memory leak detection started")
    
    def stop_monitoring(self):
        """Stop memory leak monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Memory leak detection stopped")
    
    async def _monitor_loop(self):
        """Monitoring loop for memory leaks."""
        while self._monitoring:
            try:
                await asyncio.sleep(self.sample_interval)
                await self._take_snapshot()
                await self._analyze_leaks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory leak monitoring: {e}")
    
    async def _take_snapshot(self):
        """Take memory snapshot."""
        if not tracemalloc.is_tracing():
            return
        
        try:
            snapshot = tracemalloc.take_snapshot()
            self._snapshots.append({
                'timestamp': time.time(),
                'snapshot': snapshot,
                'object_count': len(gc.get_objects()),
                'memory_usage': psutil.Process().memory_info().rss
            })
        except Exception as e:
            logger.warning(f"Error taking memory snapshot: {e}")
    
    async def _analyze_leaks(self):
        """Analyze snapshots for potential memory leaks."""
        if len(self._snapshots) < 2:
            return
        
        try:
            current = self._snapshots[-1]
            previous = self._snapshots[-2]
            
            # Compare memory usage
            memory_growth = current['memory_usage'] - previous['memory_usage']
            object_growth = current['object_count'] - previous['object_count']
            
            # Check for significant growth
            if memory_growth > 50 * 1024 * 1024:  # 50MB growth
                logger.warning(f"Potential memory leak detected: {memory_growth / (1024*1024):.1f}MB growth")
                
                # Get top memory consumers
                top_stats = current['snapshot'].statistics('lineno')[:10]
                for stat in top_stats[:3]:
                    logger.warning(f"Top memory consumer: {stat}")
            
            if object_growth > 10000:  # 10k objects growth
                logger.warning(f"Object count increased by {object_growth}")
        
        except Exception as e:
            logger.error(f"Error analyzing memory leaks: {e}")
    
    def register_object(self, obj):
        """Register object for leak tracking."""
        try:
            self._weak_refs.add(obj)
            obj_type = type(obj).__name__
            self._object_counts[obj_type] += 1
        except TypeError:
            pass  # Object doesn't support weak references
    
    def get_object_counts(self) -> Dict[str, int]:
        """Get current object counts by type."""
        return dict(self._object_counts)

class MemoryCompressor:
    """Memory compression utilities."""
    
    def __init__(self):
        self._compression_available = self._check_compression()
    
    def _check_compression(self) -> bool:
        """Check if compression is available."""
        try:
            import zlib
            return True
        except ImportError:
            return False
    
    def compress_data(self, data: bytes) -> Optional[bytes]:
        """Compress data if beneficial."""
        if not self._compression_available or len(data) < 1024:
            return None
        
        try:
            import zlib
            compressed = zlib.compress(data, level=6)
            
            # Only return compressed if we save at least 20%
            if len(compressed) < len(data) * 0.8:
                return compressed
            return None
        except Exception:
            return None
    
    def decompress_data(self, compressed_data: bytes) -> Optional[bytes]:
        """Decompress data."""
        if not self._compression_available:
            return None
        
        try:
            import zlib
            return zlib.decompress(compressed_data)
        except Exception:
            return None

class GarbageCollectionTuner:
    """Advanced garbage collection tuning."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._gc_stats = MemoryStats()
        self._last_collection = [0.0, 0.0, 0.0]
        self._collection_times = [deque(maxlen=100) for _ in range(3)]
        self._adaptive_thresholds = [config.gc_threshold_0, config.gc_threshold_1, config.gc_threshold_2]
        self._auto_tune_enabled = True
        
    def configure_gc(self):
        """Configure garbage collection based on strategy."""
        if self.config.gc_mode == GCMode.DISABLED:
            gc.disable()
            logger.info("Garbage collection disabled")
            return
        
        gc.enable()
        
        # Set thresholds based on strategy
        if self.config.gc_strategy == MemoryStrategy.CONSERVATIVE:
            thresholds = (1400, 20, 20)
        elif self.config.gc_strategy == MemoryStrategy.AGGRESSIVE:
            thresholds = (350, 5, 5)
        elif self.config.gc_strategy == MemoryStrategy.BALANCED:
            thresholds = (700, 10, 10)
        else:
            thresholds = (self.config.gc_threshold_0, self.config.gc_threshold_1, self.config.gc_threshold_2)
        
        gc.set_threshold(*thresholds)
        self._adaptive_thresholds = list(thresholds)
        
        logger.info(f"GC configured with thresholds: {thresholds}")
    
    def force_collection(self, generation: Optional[int] = None) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        start_time = time.time()
        
        if generation is not None:
            collected = gc.collect(generation)
        else:
            collected = gc.collect()
        
        collection_time = time.time() - start_time
        
        # Update statistics
        if generation is not None:
            self._collection_times[generation].append(collection_time)
        else:
            # Full collection affects all generations
            for i in range(3):
                self._collection_times[i].append(collection_time / 3)
        
        return {
            'collected': collected,
            'time': collection_time,
            'generation': generation
        }
    
    def adaptive_tuning(self):
        """Perform adaptive GC tuning based on performance."""
        if not self._auto_tune_enabled or self.config.gc_mode != GCMode.ADAPTIVE:
            return
        
        try:
            # Get current GC stats
            counts = gc.get_count()
            
            # Adjust thresholds based on collection frequency and times
            for generation in range(3):
                if self._collection_times[generation]:
                    avg_time = sum(self._collection_times[generation]) / len(self._collection_times[generation])
                    
                    # If collections are taking too long, increase threshold
                    if avg_time > 0.1:  # 100ms
                        self._adaptive_thresholds[generation] = int(self._adaptive_thresholds[generation] * 1.1)
                    elif avg_time < 0.01:  # 10ms
                        self._adaptive_thresholds[generation] = int(self._adaptive_thresholds[generation] * 0.9)
            
            # Apply new thresholds
            new_thresholds = tuple(max(100, t) for t in self._adaptive_thresholds)
            gc.set_threshold(*new_thresholds)
            
        except Exception as e:
            logger.warning(f"Error in adaptive GC tuning: {e}")
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        self._gc_stats.update_from_gc_stats()
        
        return {
            'thresholds': gc.get_threshold(),
            'counts': gc.get_count(),
            'stats': gc.get_stats(),
            'collections': self._gc_stats.gc_collections,
            'avg_collection_times': [
                sum(times) / len(times) if times else 0.0
                for times in self._collection_times
            ],
            'adaptive_thresholds': self._adaptive_thresholds
        }

class MemoryOptimizer:
    """Main memory optimization system."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Components
        self._gc_tuner = GarbageCollectionTuner(self.config)
        self._leak_detector = MemoryLeakDetector()
        self._compressor = MemoryCompressor()
        self._stats = MemoryStats()
        
        # Object pools
        self._pools: Dict[str, ObjectPool] = {}
        
        # Memory monitoring
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._memory_alerts_enabled = True
        
        # Memory mapping for large objects
        self._memory_maps: List[mmap.mmap] = []
        
        # Initialize memory tracking if enabled
        if self.config.enable_memory_tracking:
            if not tracemalloc.is_tracing():
                tracemalloc.start(self.config.allocation_limit)
    
    def initialize(self):
        """Initialize memory optimization system."""
        # Configure garbage collection
        self._gc_tuner.configure_gc()
        
        # Start leak detection
        if self.config.enable_memory_tracking:
            self._leak_detector.start_monitoring()
        
        # Create default object pools
        if self.config.enable_object_pools:
            self._create_default_pools()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info("Memory optimizer initialized")
    
    def _create_default_pools(self):
        """Create default object pools."""
        # List pool
        self._pools['list'] = ObjectPool(
            factory=list,
            max_size=self.config.pool_sizes.get('small', 1000),
            reset_func=lambda x: x.clear()
        )
        
        # Dict pool
        self._pools['dict'] = ObjectPool(
            factory=dict,
            max_size=self.config.pool_sizes.get('medium', 500),
            reset_func=lambda x: x.clear()
        )
        
        # Set pool
        self._pools['set'] = ObjectPool(
            factory=set,
            max_size=self.config.pool_sizes.get('small', 1000),
            reset_func=lambda x: x.clear()
        )
    
    def get_pool(self, pool_name: str) -> Optional[ObjectPool]:
        """Get object pool by name."""
        return self._pools.get(pool_name)
    
    def create_pool(self, name: str, factory: Callable, max_size: int = 100, 
                   reset_func: Optional[Callable] = None) -> ObjectPool:
        """Create custom object pool."""
        pool = ObjectPool(factory, max_size, reset_func)
        self._pools[name] = pool
        logger.info(f"Created object pool '{name}' with max size {max_size}")
        return pool
    
    def start_monitoring(self):
        """Start memory monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
        self._leak_detector.stop_monitoring()
        logger.info("Memory monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await asyncio.sleep(self.config.auto_gc_interval)
                
                # Check memory usage
                await self._check_memory_limits()
                
                # Perform adaptive GC tuning
                self._gc_tuner.adaptive_tuning()
                
                # Cleanup object pools
                await self._cleanup_pools()
                
                # Update statistics
                self._update_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
    
    async def _check_memory_limits(self):
        """Check memory usage against limits."""
        if not self.config.max_memory_mb:
            return
        
        try:
            memory_usage = self._stats.get_memory_usage()
            current_mb = memory_usage.get('rss_mb', 0)
            limit_mb = self.config.max_memory_mb
            
            usage_ratio = current_mb / limit_mb
            
            if usage_ratio >= self.config.critical_threshold:
                logger.critical(f"Critical memory usage: {current_mb:.1f}MB / {limit_mb}MB ({usage_ratio*100:.1f}%)")
                await self._emergency_cleanup()
                
            elif usage_ratio >= self.config.warning_threshold:
                logger.warning(f"High memory usage: {current_mb:.1f}MB / {limit_mb}MB ({usage_ratio*100:.1f}%)")
                await self._proactive_cleanup()
                
        except Exception as e:
            logger.error(f"Error checking memory limits: {e}")
    
    async def _proactive_cleanup(self):
        """Perform proactive memory cleanup."""
        logger.info("Performing proactive memory cleanup")
        
        # Force garbage collection
        self._gc_tuner.force_collection()
        
        # Clear object pools
        for pool in self._pools.values():
            pool.clear()
        
        # Compress cached data if enabled
        if self.config.enable_memory_cache_compression:
            await self._compress_caches()
    
    async def _emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        logger.warning("Performing emergency memory cleanup")
        
        # Aggressive garbage collection
        for generation in range(3):
            self._gc_tuner.force_collection(generation)
        
        # Clear all object pools
        for pool in self._pools.values():
            pool.clear()
        
        # Close memory maps
        for mmap_obj in self._memory_maps[:]:
            try:
                mmap_obj.close()
                self._memory_maps.remove(mmap_obj)
            except Exception:
                pass
        
        # Force system memory cleanup
        if hasattr(os, 'sync'):
            os.sync()
    
    async def _cleanup_pools(self):
        """Cleanup object pools."""
        for name, pool in self._pools.items():
            try:
                # Shrink pools that are too large
                stats = pool.get_stats()
                if stats['hit_rate'] < 50 and stats['size'] > 10:
                    # Pool not effective, reduce size
                    pool.clear()
                    logger.debug(f"Cleared ineffective pool '{name}'")
            except Exception as e:
                logger.warning(f"Error cleaning pool '{name}': {e}")
    
    async def _compress_caches(self):
        """Compress cached data to save memory."""
        try:
            from ..cache.distributed_cache import get_distributed_cache
            cache = get_distributed_cache()
            
            # This would need to be implemented in the cache layer
            # cache.compress_stored_data()
            
        except Exception as e:
            logger.warning(f"Error compressing caches: {e}")
    
    def _update_stats(self):
        """Update memory statistics."""
        try:
            self._stats.update_from_gc_stats()
            
            # Update peak usage
            current_usage = psutil.Process().memory_info().rss
            self._stats.current_usage = current_usage
            if current_usage > self._stats.peak_usage:
                self._stats.peak_usage = current_usage
            
            # Update pool stats
            total_hits = sum(pool.get_stats()['hits'] for pool in self._pools.values())
            total_misses = sum(pool.get_stats()['misses'] for pool in self._pools.values())
            self._stats.pool_hits = total_hits
            self._stats.pool_misses = total_misses
            
        except Exception as e:
            logger.warning(f"Error updating memory stats: {e}")
    
    def create_memory_map(self, size: int, access: int = mmap.ACCESS_WRITE) -> Optional[mmap.mmap]:
        """Create memory-mapped file for large data."""
        if not self.config.enable_memory_mapping or size < self.config.mmap_threshold_mb * 1024 * 1024:
            return None
        
        try:
            # Create temporary file
            import tempfile
            fd = tempfile.NamedTemporaryFile(delete=False).fileno()
            
            # Create memory map
            memory_map = mmap.mmap(fd, size, access=access)
            self._memory_maps.append(memory_map)
            
            logger.debug(f"Created memory map of size {size / (1024*1024):.1f}MB")
            return memory_map
            
        except Exception as e:
            logger.error(f"Error creating memory map: {e}")
            return None
    
    def optimize_for_low_memory(self):
        """Optimize settings for low-memory environment."""
        logger.info("Optimizing for low-memory environment")
        
        # Reduce GC thresholds for more frequent collection
        gc.set_threshold(200, 3, 3)
        
        # Clear all pools
        for pool in self._pools.values():
            pool.clear()
        
        # Disable object pools to save memory
        self.config.enable_object_pools = False
        
        # Force aggressive garbage collection
        self._gc_tuner.force_collection()
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get memory optimization suggestions."""
        suggestions = []
        
        try:
            memory_usage = self._stats.get_memory_usage()
            gc_stats = self._gc_tuner.get_gc_stats()
            
            # Check memory usage
            if memory_usage.get('percent', 0) > 80:
                suggestions.append("Memory usage is high, consider reducing cache sizes or enabling compression")
            
            # Check GC performance
            avg_times = gc_stats.get('avg_collection_times', [0, 0, 0])
            if any(t > 0.1 for t in avg_times):
                suggestions.append("GC collection times are high, consider increasing thresholds")
            
            # Check pool effectiveness
            for name, pool in self._pools.items():
                stats = pool.get_stats()
                if stats['hit_rate'] < 30:
                    suggestions.append(f"Object pool '{name}' has low hit rate, consider adjusting size")
            
            # Check for potential leaks
            object_counts = self._leak_detector.get_object_counts()
            if any(count > 10000 for count in object_counts.values()):
                suggestions.append("High object counts detected, check for potential memory leaks")
            
        except Exception as e:
            logger.warning(f"Error generating optimization suggestions: {e}")
        
        return suggestions
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            'memory_usage': self._stats.get_memory_usage(),
            'gc_stats': self._gc_tuner.get_gc_stats(),
            'pool_stats': {name: pool.get_stats() for name, pool in self._pools.items()},
            'leak_detection': {
                'object_counts': self._leak_detector.get_object_counts(),
                'snapshots_taken': len(self._leak_detector._snapshots)
            },
            'config': {
                'max_memory_mb': self.config.max_memory_mb,
                'gc_strategy': self.config.gc_strategy.value,
                'gc_mode': self.config.gc_mode.value,
                'object_pools_enabled': self.config.enable_object_pools,
                'memory_tracking_enabled': self.config.enable_memory_tracking
            },
            'optimization_suggestions': self.get_optimization_suggestions()
        }
    
    def cleanup(self):
        """Cleanup optimizer resources."""
        self.stop_monitoring()
        
        # Close memory maps
        for mmap_obj in self._memory_maps:
            try:
                mmap_obj.close()
            except Exception:
                pass
        self._memory_maps.clear()
        
        # Clear pools
        for pool in self._pools.values():
            pool.clear()
        
        logger.info("Memory optimizer cleaned up")


# Global memory optimizer
_memory_optimizer: Optional[MemoryOptimizer] = None

def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
        _memory_optimizer.initialize()
    return _memory_optimizer


# Decorator for automatic memory management
def optimize_memory(pool_name: Optional[str] = None):
    """Decorator for automatic memory optimization."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_memory_optimizer()
            
            if pool_name:
                pool = optimizer.get_pool(pool_name)
                if pool:
                    with pool.get_object() as obj:
                        # Use pooled object somehow - this is context specific
                        return func(*args, **kwargs)
            
            # Register objects for leak detection
            result = func(*args, **kwargs)
            if hasattr(result, '__dict__'):
                optimizer._leak_detector.register_object(result)
            
            return result
        return wrapper
    return decorator