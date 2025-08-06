"""Cache management for Gaudi 3 Scale infrastructure with advanced distributed caching."""

import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta

from .redis_cache import RedisCache
from .memory_cache import MemoryCache
from .distributed_cache import DistributedCache, CacheConfig, CacheStrategy, ConsistencyLevel

logger = logging.getLogger(__name__)


class CacheManager:
    """High-level cache management with distributed caching and performance optimization."""
    
    def __init__(self, 
                 distributed_cache: Optional[DistributedCache] = None,
                 cache_backend: Optional[RedisCache] = None,
                 enable_distributed: bool = True):
        """Initialize cache manager.
        
        Args:
            distributed_cache: Optional distributed cache instance
            cache_backend: Optional Redis cache backend (for backward compatibility)
            enable_distributed: Whether to use distributed caching
        """
        self.enable_distributed = enable_distributed
        
        if enable_distributed:
            # Use distributed cache with L1 (memory) + L2 (Redis) layers
            config = CacheConfig(
                l1_max_size=5000,
                l1_ttl=300.0,
                l2_ttl=3600.0,
                strategy=CacheStrategy.WRITE_THROUGH,
                consistency_level=ConsistencyLevel.EVENTUAL,
                enable_compression=True,
                enable_batching=True
            )
            self.cache = distributed_cache or DistributedCache(config=config)
            self.legacy_cache = None
        else:
            # Use legacy Redis cache for backward compatibility
            self.cache = cache_backend or RedisCache()
            self.legacy_cache = self.cache
            
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Cache key prefixes for different data types
        self.prefixes = {
            "cluster": "cluster:",
            "training_job": "training_job:",
            "metrics": "metrics:",
            "node": "node:",
            "cost": "cost:",
            "health": "health:",
            "config": "config:",
            "session": "session:"
        }
    
    def _make_key(self, category: str, identifier: str) -> str:
        """Make cache key with prefix.
        
        Args:
            category: Data category
            identifier: Unique identifier
            
        Returns:
            Formatted cache key
        """
        prefix = self.prefixes.get(category, f"{category}:")
        return f"{prefix}{identifier}"
    
    def set_cluster_data(self, cluster_id: str, data: Dict[str, Any], 
                        ttl: int = 3600) -> bool:
        """Cache cluster data.
        
        Args:
            cluster_id: Cluster identifier
            data: Cluster data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        key = self._make_key("cluster", cluster_id)
        return self.cache.set(key, data, ttl)
    
    def get_cluster_data(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Get cached cluster data.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Cached cluster data or None
        """
        key = self._make_key("cluster", cluster_id)
        return self.cache.get(key)
    
    def invalidate_cluster_data(self, cluster_id: str) -> bool:
        """Invalidate cached cluster data.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            True if invalidated successfully
        """
        key = self._make_key("cluster", cluster_id)
        return self.cache.delete(key)
    
    def set_training_job_data(self, job_id: str, data: Dict[str, Any],
                             ttl: int = 1800) -> bool:
        """Cache training job data.
        
        Args:
            job_id: Training job identifier
            data: Job data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        key = self._make_key("training_job", job_id)
        return self.cache.set(key, data, ttl)
    
    def get_training_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get cached training job data.
        
        Args:
            job_id: Training job identifier
            
        Returns:
            Cached job data or None
        """
        key = self._make_key("training_job", job_id)
        return self.cache.get(key)
    
    def cache_metrics_batch(self, cluster_id: str, metrics: List[Dict[str, Any]],
                           ttl: int = 300) -> bool:
        """Cache batch of metrics data.
        
        Args:
            cluster_id: Cluster identifier
            metrics: List of metrics to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        timestamp = datetime.utcnow().isoformat()
        key = self._make_key("metrics", f"{cluster_id}:{timestamp}")
        
        metrics_data = {
            "timestamp": timestamp,
            "cluster_id": cluster_id,
            "metrics": metrics
        }
        
        return self.cache.set(key, metrics_data, ttl)
    
    def get_recent_metrics(self, cluster_id: str, 
                          minutes: int = 30) -> List[Dict[str, Any]]:
        """Get recent metrics for a cluster.
        
        Args:
            cluster_id: Cluster identifier
            minutes: Number of minutes to look back
            
        Returns:
            List of recent metrics
        """
        pattern = self._make_key("metrics", f"{cluster_id}:*")
        metric_keys = self.cache.scan_keys(pattern)
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = []
        
        for key in metric_keys:
            data = self.cache.get(key)
            if data and 'timestamp' in data:
                try:
                    metric_time = datetime.fromisoformat(data['timestamp'])
                    if metric_time >= cutoff_time:
                        recent_metrics.extend(data.get('metrics', []))
                except ValueError:
                    # Invalid timestamp format, skip
                    continue
        
        return sorted(recent_metrics, key=lambda x: x.get('timestamp', ''))
    
    def set_node_health(self, cluster_id: str, node_id: str, 
                       health_data: Dict[str, Any], ttl: int = 120) -> bool:
        """Cache node health data.
        
        Args:
            cluster_id: Cluster identifier
            node_id: Node identifier
            health_data: Health data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        key = self._make_key("health", f"{cluster_id}:{node_id}")
        health_data['last_updated'] = datetime.utcnow().isoformat()
        return self.cache.set(key, health_data, ttl)
    
    def get_cluster_health(self, cluster_id: str) -> Dict[str, Dict[str, Any]]:
        """Get health data for all nodes in a cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Dictionary mapping node IDs to health data
        """
        pattern = self._make_key("health", f"{cluster_id}:*")
        health_keys = self.cache.scan_keys(pattern)
        
        cluster_health = {}
        for key in health_keys:
            # Extract node ID from key
            node_id = key.split(':')[-1]
            health_data = self.cache.get(key)
            if health_data:
                cluster_health[node_id] = health_data
        
        return cluster_health
    
    def cache_cost_analysis(self, analysis_id: str, analysis_data: Dict[str, Any],
                           ttl: int = 7200) -> bool:
        """Cache cost analysis results.
        
        Args:
            analysis_id: Analysis identifier
            analysis_data: Cost analysis data
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        key = self._make_key("cost", analysis_id)
        analysis_data['cached_at'] = datetime.utcnow().isoformat()
        return self.cache.set(key, analysis_data, ttl)
    
    def get_cost_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get cached cost analysis.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            Cached analysis data or None
        """
        key = self._make_key("cost", analysis_id)
        return self.cache.get(key)
    
    def cache_configuration(self, config_id: str, config_data: Dict[str, Any],
                           ttl: int = 3600) -> bool:
        """Cache configuration data.
        
        Args:
            config_id: Configuration identifier
            config_data: Configuration data
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        key = self._make_key("config", config_id)
        return self.cache.set(key, config_data, ttl)
    
    def get_configuration(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get cached configuration.
        
        Args:
            config_id: Configuration identifier
            
        Returns:
            Cached configuration or None
        """
        key = self._make_key("config", config_id)
        return self.cache.get(key)
    
    def set_session_data(self, session_id: str, session_data: Dict[str, Any],
                        ttl: int = 1800) -> bool:
        """Cache session data.
        
        Args:
            session_id: Session identifier
            session_data: Session data
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        key = self._make_key("session", session_id)
        return self.cache.set(key, session_data, ttl)
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Cached session data or None
        """
        key = self._make_key("session", session_id)
        return self.cache.get(key)
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if invalidated successfully
        """
        key = self._make_key("session", session_id)
        return self.cache.delete(key)
    
    def bulk_invalidate(self, category: str, identifiers: List[str]) -> int:
        """Bulk invalidate cache entries.
        
        Args:
            category: Data category
            identifiers: List of identifiers to invalidate
            
        Returns:
            Number of entries invalidated
        """
        keys = [self._make_key(category, identifier) for identifier in identifiers]
        return self.cache.delete_many(keys)
    
    def clear_category(self, category: str) -> int:
        """Clear all cache entries for a category.
        
        Args:
            category: Data category to clear
            
        Returns:
            Number of entries cleared
        """
        prefix = self.prefixes.get(category, f"{category}:")
        pattern = f"{prefix}*"
        keys = self.cache.scan_keys(pattern)
        
        if keys:
            return self.cache.delete_many(keys)
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.cache.get_stats()
        
        # Add category-specific statistics
        category_stats = {}
        for category, prefix in self.prefixes.items():
            pattern = f"{prefix}*"
            keys = self.cache.scan_keys(pattern)
            category_stats[category] = len(keys)
        
        stats['category_counts'] = category_stats
        return stats
    
    def warm_up_cache(self, cluster_ids: List[str]) -> Dict[str, bool]:
        """Warm up cache for specified clusters.
        
        Args:
            cluster_ids: List of cluster IDs to warm up
            
        Returns:
            Dictionary mapping cluster IDs to success status
        """
        results = {}
        
        for cluster_id in cluster_ids:
            try:
                # This would typically fetch from database and cache
                # For now, we'll just mark as warmed up
                warm_data = {
                    "warmed_up": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                success = self.set_cluster_data(cluster_id, warm_data, ttl=3600)
                results[cluster_id] = success
                
                if success:
                    self.logger.debug(f"Warmed up cache for cluster {cluster_id}")
                else:
                    self.logger.warning(f"Failed to warm up cache for cluster {cluster_id}")
                    
            except Exception as e:
                self.logger.error(f"Error warming up cache for cluster {cluster_id}: {e}")
                results[cluster_id] = False
        
        return results
    
    def bulk_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple cache entries efficiently.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to values
        """
        if self.enable_distributed:
            return self.cache.bulk_get(keys)
        else:
            # Fallback implementation for legacy cache
            result = {}
            for key in keys:
                value = self.cache.get(key)
                if value is not None:
                    result[key] = value
            return result
    
    def bulk_set(self, data: Dict[str, Any], ttl: int = 3600) -> Dict[str, bool]:
        """Set multiple cache entries efficiently.
        
        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
            
        Returns:
            Dictionary mapping keys to success status
        """
        if self.enable_distributed:
            return self.cache.bulk_set(data, ttl=float(ttl))
        else:
            # Fallback implementation for legacy cache
            results = {}
            for key, value in data.items():
                results[key] = self.cache.set(key, value, ttl)
            return results
    
    def get_or_set(self, key: str, loader_func: Callable[[], Any], ttl: int = 3600) -> Any:
        """Get value from cache or set using loader function.
        
        Args:
            key: Cache key
            loader_func: Function to load data if not cached
            ttl: Time to live in seconds
            
        Returns:
            Cached or fresh value
        """
        if self.enable_distributed:
            return self.cache.get_or_set(key, loader_func, ttl=float(ttl))
        else:
            # Fallback implementation for legacy cache
            value = self.cache.get(key)
            if value is not None:
                return value
            
            fresh_value = loader_func()
            self.cache.set(key, fresh_value, ttl)
            return fresh_value
    
    def refresh_cache(self, key: str, loader_func: Callable[[], Any], ttl: int = 3600) -> Any:
        """Refresh cache entry using loader function.
        
        Args:
            key: Cache key
            loader_func: Function to load fresh data
            ttl: Time to live in seconds
            
        Returns:
            Fresh value from loader function
        """
        if self.enable_distributed:
            return self.cache.refresh_cache(key, loader_func, ttl=float(ttl))
        else:
            # Fallback implementation for legacy cache
            try:
                fresh_value = loader_func()
                self.cache.set(key, fresh_value, ttl)
                return fresh_value
            except Exception as e:
                self.logger.error(f"Error refreshing cache for key {key}: {e}")
                return self.cache.get(key)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern.
        
        Args:
            pattern: Key pattern with wildcards
            
        Returns:
            Number of keys invalidated
        """
        if self.enable_distributed:
            return self.cache.invalidate_pattern(pattern)
        else:
            # Fallback implementation for legacy cache
            keys = self.cache.scan_keys(pattern)
            count = 0
            for key in keys:
                if self.cache.delete(key):
                    count += 1
            return count
    
    def add_invalidation_callback(self, key_pattern: str, callback: Callable[[str], None]) -> None:
        """Add callback for cache invalidation events.
        
        Args:
            key_pattern: Pattern to match keys
            callback: Callback function
        """
        if self.enable_distributed:
            self.cache.add_invalidation_callback(key_pattern, callback)
        else:
            self.logger.warning("Invalidation callbacks not supported in legacy cache mode")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache systems.
        
        Returns:
            Health check results
        """
        if self.enable_distributed:
            return self.cache.health_check()
        else:
            # Basic health check for legacy cache
            try:
                test_key = f"health_check_{datetime.utcnow().timestamp()}"
                self.cache.set(test_key, "test", ttl=1)
                result = self.cache.get(test_key)
                success = result == "test"
                if success:
                    self.cache.delete(test_key)
                
                return {
                    'cache': {'healthy': success, 'error': None if success else 'Health check failed'},
                    'overall': {'healthy': success}
                }
            except Exception as e:
                return {
                    'cache': {'healthy': False, 'error': str(e)},
                    'overall': {'healthy': False}
                }
    
    def close(self) -> None:
        """Close cache manager and cleanup resources."""
        if self.enable_distributed:
            self.cache.close()
        # Legacy cache doesn't need explicit cleanup


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(enable_distributed: bool = True) -> CacheManager:
    """Get global cache manager instance.
    
    Args:
        enable_distributed: Whether to use distributed caching
        
    Returns:
        Cache manager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(enable_distributed=enable_distributed)
    return _cache_manager