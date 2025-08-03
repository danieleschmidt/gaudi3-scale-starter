"""Unit tests for cache layer components."""

import json
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.gaudi3_scale.cache.redis_cache import RedisCache
from src.gaudi3_scale.cache.cache_manager import CacheManager


class TestRedisCache:
    """Test RedisCache functionality."""
    
    def test_set_and_get_json(self, redis_cache, mock_redis):
        """Test setting and getting JSON data."""
        test_data = {"key": "value", "number": 42}
        
        # Configure mock to return serialized data
        mock_redis.get.return_value = json.dumps(test_data)
        
        # Test set
        result = redis_cache.set("test_key", test_data, ttl=3600)
        assert result is True
        
        # Verify set was called correctly
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "test_key"
        assert call_args[0][1] == 3600
        assert json.loads(call_args[0][2]) == test_data
        
        # Test get
        retrieved_data = redis_cache.get("test_key")
        assert retrieved_data == test_data
    
    def test_set_and_get_pickle(self, redis_cache, mock_redis):
        """Test setting and getting data with pickle."""
        import pickle
        
        test_data = {"complex": {"nested": "data"}}
        
        # Configure mock to return pickled data
        mock_redis.get.return_value = pickle.dumps(test_data)
        
        # Test set with pickle
        result = redis_cache.set("test_key", test_data, ttl=3600, use_pickle=True)
        assert result is True
        
        # Test get with pickle
        retrieved_data = redis_cache.get("test_key", use_pickle=True)
        assert retrieved_data == test_data
    
    def test_delete(self, redis_cache, mock_redis):
        """Test deleting cache entries."""
        mock_redis.delete.return_value = 1
        
        result = redis_cache.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")
    
    def test_delete_many(self, redis_cache, mock_redis):
        """Test deleting multiple cache entries."""
        mock_redis.delete.return_value = 3
        
        keys = ["key1", "key2", "key3"]
        result = redis_cache.delete_many(keys)
        
        assert result == 3
        mock_redis.delete.assert_called_once_with(*keys)
    
    def test_exists(self, redis_cache, mock_redis):
        """Test checking if key exists."""
        mock_redis.exists.return_value = 1
        
        result = redis_cache.exists("test_key")
        assert result is True
        mock_redis.exists.assert_called_once_with("test_key")
    
    def test_expire(self, redis_cache, mock_redis):
        """Test setting expiration time."""
        mock_redis.expire.return_value = True
        
        result = redis_cache.expire("test_key", 3600)
        assert result is True
        mock_redis.expire.assert_called_once_with("test_key", 3600)
    
    def test_ttl(self, redis_cache, mock_redis):
        """Test getting time to live."""
        mock_redis.ttl.return_value = 1800
        
        result = redis_cache.ttl("test_key")
        assert result == 1800
        mock_redis.ttl.assert_called_once_with("test_key")
    
    def test_scan_keys(self, redis_cache, mock_redis):
        """Test scanning for keys with pattern."""
        mock_redis.scan.side_effect = [
            (0, ["key1", "key2", "key3"]),  # First call returns cursor 0 (end)
        ]
        
        result = redis_cache.scan_keys("test_*")
        assert result == ["key1", "key2", "key3"]
        mock_redis.scan.assert_called_with(0, match="test_*")
    
    def test_increment(self, redis_cache, mock_redis):
        """Test incrementing numeric values."""
        mock_redis.incrby.return_value = 5
        
        result = redis_cache.increment("counter", 2)
        assert result == 5
        mock_redis.incrby.assert_called_once_with("counter", 2)
    
    def test_hash_operations(self, redis_cache, mock_redis):
        """Test hash operations."""
        test_data = {"field": "value"}
        
        # Test set hash
        mock_redis.hset.return_value = 1
        result = redis_cache.set_hash("hash_key", "field1", test_data)
        assert result is True
        
        # Test get hash
        mock_redis.hget.return_value = json.dumps(test_data)
        retrieved = redis_cache.get_hash("hash_key", "field1")
        assert retrieved == test_data
        
        # Test get all hash
        mock_redis.hgetall.return_value = {"field1": json.dumps(test_data)}
        all_data = redis_cache.get_all_hash("hash_key")
        assert all_data == {"field1": test_data}
    
    def test_list_operations(self, redis_cache, mock_redis):
        """Test list operations."""
        test_item = {"item": "data"}
        
        # Test append to list
        mock_redis.rpush.return_value = 1
        result = redis_cache.list_append("list_key", test_item)
        assert result == 1
        
        # Test get range
        mock_redis.lrange.return_value = [json.dumps(test_item)]
        retrieved = redis_cache.list_get_range("list_key")
        assert retrieved == [test_item]
    
    def test_get_stats(self, redis_cache, mock_redis):
        """Test getting cache statistics."""
        stats = redis_cache.get_stats()
        
        assert "redis_version" in stats
        assert "used_memory" in stats
        assert "hit_ratio" in stats
        assert stats["hit_ratio"] == 80.0  # Based on mock data: 80/(80+20) * 100
    
    def test_error_handling(self, redis_cache, mock_redis):
        """Test error handling in cache operations."""
        # Test JSON decode error
        mock_redis.get.return_value = "invalid json"
        result = redis_cache.get("test_key")
        assert result is None
        
        # Test Redis connection error
        mock_redis.set.side_effect = Exception("Connection error")
        result = redis_cache.set("test_key", {"data": "value"})
        assert result is False


class TestCacheManager:
    """Test CacheManager functionality."""
    
    def test_cluster_data_caching(self, mock_redis):
        """Test cluster data caching operations."""
        with patch('src.gaudi3_scale.cache.cache_manager.RedisCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            mock_cache.set.return_value = True
            mock_cache.get.return_value = {"cluster_id": "test-cluster"}
            mock_cache.delete.return_value = True
            
            cache_manager = CacheManager()
            
            # Test set cluster data
            test_data = {"name": "test-cluster", "status": "running"}
            result = cache_manager.set_cluster_data("cluster-123", test_data)
            assert result is True
            
            # Test get cluster data
            retrieved = cache_manager.get_cluster_data("cluster-123")
            assert retrieved == {"cluster_id": "test-cluster"}
            
            # Test invalidate cluster data
            result = cache_manager.invalidate_cluster_data("cluster-123")
            assert result is True
    
    def test_training_job_caching(self, mock_redis):
        """Test training job caching operations."""
        with patch('src.gaudi3_scale.cache.cache_manager.RedisCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            mock_cache.set.return_value = True
            mock_cache.get.return_value = {"job_id": "test-job"}
            
            cache_manager = CacheManager()
            
            # Test set training job data
            job_data = {"name": "test-job", "status": "running", "progress": 0.5}
            result = cache_manager.set_training_job_data("job-123", job_data)
            assert result is True
            
            # Test get training job data
            retrieved = cache_manager.get_training_job_data("job-123")
            assert retrieved == {"job_id": "test-job"}
    
    def test_metrics_caching(self, mock_redis):
        """Test metrics caching operations."""
        with patch('src.gaudi3_scale.cache.cache_manager.RedisCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            mock_cache.set.return_value = True
            mock_cache.scan_keys.return_value = ["metrics:cluster-123:2025-01-15T10:00:00"]
            mock_cache.get.return_value = {
                "timestamp": "2025-01-15T10:00:00",
                "cluster_id": "cluster-123",
                "metrics": [{"name": "hpu_utilization", "value": 85.5}]
            }
            
            cache_manager = CacheManager()
            
            # Test cache metrics batch
            metrics = [
                {"name": "hpu_utilization", "value": 85.5, "timestamp": "2025-01-15T10:00:00"},
                {"name": "memory_usage", "value": 24.5, "timestamp": "2025-01-15T10:00:00"}
            ]
            result = cache_manager.cache_metrics_batch("cluster-123", metrics)
            assert result is True
            
            # Test get recent metrics
            recent_metrics = cache_manager.get_recent_metrics("cluster-123", minutes=30)
            assert len(recent_metrics) == 1
            assert recent_metrics[0]["name"] == "hpu_utilization"
    
    def test_health_caching(self, mock_redis):
        """Test health data caching operations."""
        with patch('src.gaudi3_scale.cache.cache_manager.RedisCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            mock_cache.set.return_value = True
            mock_cache.scan_keys.return_value = ["health:cluster-123:node-1", "health:cluster-123:node-2"]
            mock_cache.get.side_effect = [
                {"status": "healthy", "hpu_utilization": 85.0},
                {"status": "healthy", "hpu_utilization": 78.0}
            ]
            
            cache_manager = CacheManager()
            
            # Test set node health
            health_data = {"status": "healthy", "hpu_utilization": 85.0, "temperature": 65.0}
            result = cache_manager.set_node_health("cluster-123", "node-1", health_data)
            assert result is True
            
            # Test get cluster health
            cluster_health = cache_manager.get_cluster_health("cluster-123")
            assert len(cluster_health) == 2
            assert "node-1" in cluster_health
            assert "node-2" in cluster_health
    
    def test_cost_analysis_caching(self, mock_redis):
        """Test cost analysis caching operations."""
        with patch('src.gaudi3_scale.cache.cache_manager.RedisCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            mock_cache.set.return_value = True
            mock_cache.get.return_value = {
                "total_cost": 5000.0,
                "cost_breakdown": {"compute": 4000.0, "storage": 500.0, "network": 500.0}
            }
            
            cache_manager = CacheManager()
            
            # Test cache cost analysis
            analysis_data = {
                "total_cost": 5000.0,
                "cost_breakdown": {"compute": 4000.0, "storage": 500.0, "network": 500.0},
                "duration_hours": 720
            }
            result = cache_manager.cache_cost_analysis("analysis-123", analysis_data)
            assert result is True
            
            # Test get cost analysis
            retrieved = cache_manager.get_cost_analysis("analysis-123")
            assert retrieved["total_cost"] == 5000.0
    
    def test_session_management(self, mock_redis):
        """Test session data management."""
        with patch('src.gaudi3_scale.cache.cache_manager.RedisCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            mock_cache.set.return_value = True
            mock_cache.get.return_value = {"user_id": "user-123", "permissions": ["read", "write"]}
            mock_cache.delete.return_value = True
            
            cache_manager = CacheManager()
            
            # Test set session data
            session_data = {"user_id": "user-123", "permissions": ["read", "write"]}
            result = cache_manager.set_session_data("session-123", session_data)
            assert result is True
            
            # Test get session data
            retrieved = cache_manager.get_session_data("session-123")
            assert retrieved["user_id"] == "user-123"
            
            # Test invalidate session
            result = cache_manager.invalidate_session("session-123")
            assert result is True
    
    def test_bulk_operations(self, mock_redis):
        """Test bulk cache operations."""
        with patch('src.gaudi3_scale.cache.cache_manager.RedisCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            mock_cache.delete_many.return_value = 3
            mock_cache.scan_keys.return_value = ["cluster:1", "cluster:2", "cluster:3"]
            
            cache_manager = CacheManager()
            
            # Test bulk invalidate
            identifiers = ["cluster-1", "cluster-2", "cluster-3"]
            result = cache_manager.bulk_invalidate("cluster", identifiers)
            assert result == 3
            
            # Test clear category
            result = cache_manager.clear_category("cluster")
            assert result == 3
    
    def test_cache_stats(self, mock_redis):
        """Test cache statistics retrieval."""
        with patch('src.gaudi3_scale.cache.cache_manager.RedisCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            mock_cache.get_stats.return_value = {
                "redis_version": "6.2.0",
                "used_memory": 1024000,
                "hit_ratio": 85.0
            }
            mock_cache.scan_keys.return_value = ["cluster:1", "cluster:2"]
            
            cache_manager = CacheManager()
            
            stats = cache_manager.get_cache_stats()
            
            assert "redis_version" in stats
            assert "category_counts" in stats
            assert stats["category_counts"]["cluster"] == 2
    
    def test_warm_up_cache(self, mock_redis):
        """Test cache warm-up functionality."""
        with patch('src.gaudi3_scale.cache.cache_manager.RedisCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            mock_cache.set.return_value = True
            
            cache_manager = CacheManager()
            
            cluster_ids = ["cluster-1", "cluster-2", "cluster-3"]
            results = cache_manager.warm_up_cache(cluster_ids)
            
            assert len(results) == 3
            for cluster_id, success in results.items():
                assert success is True
                assert cluster_id in cluster_ids
    
    def test_key_generation(self, mock_redis):
        """Test cache key generation."""
        with patch('src.gaudi3_scale.cache.cache_manager.RedisCache') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            
            cache_manager = CacheManager()
            
            # Test known category
            key = cache_manager._make_key("cluster", "test-123")
            assert key == "cluster:test-123"
            
            # Test unknown category
            key = cache_manager._make_key("unknown", "test-123")
            assert key == "unknown:test-123"