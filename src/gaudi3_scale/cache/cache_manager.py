"""Cache management for Gaudi 3 Scale infrastructure."""

import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from .redis_cache import RedisCache

logger = logging.getLogger(__name__)


class CacheManager:
    """High-level cache management with multiple backend support."""
    
    def __init__(self, cache_backend: Optional[RedisCache] = None):
        """Initialize cache manager.
        
        Args:
            cache_backend: Optional cache backend override
        """
        self.cache = cache_backend or RedisCache()
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


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance.
    
    Returns:
        Cache manager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager