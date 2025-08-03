"""Redis-based caching implementation."""

import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Union

import redis

from ..database.connection import RedisConnection

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based cache implementation with JSON and pickle serialization."""
    
    def __init__(self, redis_connection: Optional[RedisConnection] = None):
        """Initialize Redis cache.
        
        Args:
            redis_connection: Optional Redis connection override
        """
        self.redis_conn = redis_connection or RedisConnection()
        self.client = self.redis_conn.get_client()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            use_pickle: bool = False) -> bool:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            use_pickle: Whether to use pickle for serialization
            
        Returns:
            True if set successfully, False otherwise
        """
        try:
            if use_pickle:
                serialized_value = pickle.dumps(value)
            else:
                serialized_value = json.dumps(value, default=str)
            
            if ttl:
                result = self.client.setex(key, ttl, serialized_value)
            else:
                result = self.client.set(key, serialized_value)
            
            if result:
                self.logger.debug(f"Cached value for key: {key}")
            
            return bool(result)
        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def get(self, key: str, use_pickle: bool = False) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            use_pickle: Whether to use pickle for deserialization
            
        Returns:
            Cached value or None if not found
        """
        try:
            value = self.client.get(key)
            if value is None:
                return None
            
            if use_pickle:
                return pickle.loads(value)
            else:
                return json.loads(value)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for key {key}: {e}")
            return None
        except pickle.PickleError as e:
            self.logger.error(f"Pickle decode error for key {key}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            result = self.client.delete(key)
            if result:
                self.logger.debug(f"Deleted cache key: {key}")
            return bool(result)
        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys from the cache.
        
        Args:
            keys: List of cache keys to delete
            
        Returns:
            Number of keys deleted
        """
        try:
            if not keys:
                return 0
            
            result = self.client.delete(*keys)
            self.logger.debug(f"Deleted {result} cache keys")
            return result
        except Exception as e:
            self.logger.error(f"Error deleting cache keys: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            self.logger.error(f"Error checking cache key existence {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if expiration set, False otherwise
        """
        try:
            result = self.client.expire(key, ttl)
            return bool(result)
        except Exception as e:
            self.logger.error(f"Error setting expiration for key {key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """Get time to live for a key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        try:
            return self.client.ttl(key)
        except Exception as e:
            self.logger.error(f"Error getting TTL for key {key}: {e}")
            return -2
    
    def scan_keys(self, pattern: str) -> List[str]:
        """Scan for keys matching a pattern.
        
        Args:
            pattern: Pattern to match (supports wildcards)
            
        Returns:
            List of matching keys
        """
        try:
            keys = []
            cursor = 0
            
            while True:
                cursor, partial_keys = self.client.scan(cursor, match=pattern)
                keys.extend(partial_keys)
                
                if cursor == 0:
                    break
            
            return keys
        except Exception as e:
            self.logger.error(f"Error scanning keys with pattern {pattern}: {e}")
            return []
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in the cache.
        
        Args:
            key: Cache key
            amount: Amount to increment by
            
        Returns:
            New value after increment
        """
        try:
            return self.client.incrby(key, amount)
        except Exception as e:
            self.logger.error(f"Error incrementing key {key}: {e}")
            return 0
    
    def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement a numeric value in the cache.
        
        Args:
            key: Cache key
            amount: Amount to decrement by
            
        Returns:
            New value after decrement
        """
        try:
            return self.client.decrby(key, amount)
        except Exception as e:
            self.logger.error(f"Error decrementing key {key}: {e}")
            return 0
    
    def set_hash(self, key: str, field: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a field in a hash.
        
        Args:
            key: Hash key
            field: Field name
            value: Field value
            ttl: Optional TTL for the hash
            
        Returns:
            True if set successfully
        """
        try:
            serialized_value = json.dumps(value, default=str)
            result = self.client.hset(key, field, serialized_value)
            
            if ttl and result:
                self.client.expire(key, ttl)
            
            return bool(result)
        except Exception as e:
            self.logger.error(f"Error setting hash field {key}:{field}: {e}")
            return False
    
    def get_hash(self, key: str, field: str) -> Optional[Any]:
        """Get a field from a hash.
        
        Args:
            key: Hash key
            field: Field name
            
        Returns:
            Field value or None if not found
        """
        try:
            value = self.client.hget(key, field)
            if value is None:
                return None
            
            return json.loads(value)
        except Exception as e:
            self.logger.error(f"Error getting hash field {key}:{field}: {e}")
            return None
    
    def get_all_hash(self, key: str) -> Dict[str, Any]:
        """Get all fields from a hash.
        
        Args:
            key: Hash key
            
        Returns:
            Dictionary with all field-value pairs
        """
        try:
            hash_data = self.client.hgetall(key)
            result = {}
            
            for field, value in hash_data.items():
                try:
                    result[field] = json.loads(value)
                except json.JSONDecodeError:
                    result[field] = value
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting all hash fields for {key}: {e}")
            return {}
    
    def list_append(self, key: str, value: Any, ttl: Optional[int] = None) -> int:
        """Append value to a list.
        
        Args:
            key: List key
            value: Value to append
            ttl: Optional TTL for the list
            
        Returns:
            New length of the list
        """
        try:
            serialized_value = json.dumps(value, default=str)
            result = self.client.rpush(key, serialized_value)
            
            if ttl:
                self.client.expire(key, ttl)
            
            return result
        except Exception as e:
            self.logger.error(f"Error appending to list {key}: {e}")
            return 0
    
    def list_get_range(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range of values from a list.
        
        Args:
            key: List key
            start: Start index
            end: End index (-1 for end of list)
            
        Returns:
            List of values in the specified range
        """
        try:
            values = self.client.lrange(key, start, end)
            result = []
            
            for value in values:
                try:
                    result.append(json.loads(value))
                except json.JSONDecodeError:
                    result.append(value)
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting list range for {key}: {e}")
            return []
    
    def flush_db(self) -> bool:
        """Flush all keys from the current database.
        
        Returns:
            True if successful
        """
        try:
            result = self.client.flushdb()
            self.logger.warning("Flushed all keys from Redis database")
            return bool(result)
        except Exception as e:
            self.logger.error(f"Error flushing database: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            info = self.client.info()
            
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_ratio": self._calculate_hit_ratio(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                ),
                "db_keys": self._get_db_key_count(info)
            }
        except Exception as e:
            self.logger.error(f"Error getting Redis stats: {e}")
            return {}
    
    def _calculate_hit_ratio(self, hits: int, misses: int) -> float:
        """Calculate cache hit ratio.
        
        Args:
            hits: Number of cache hits
            misses: Number of cache misses
            
        Returns:
            Hit ratio as a percentage
        """
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100
    
    def _get_db_key_count(self, info: Dict[str, Any]) -> int:
        """Get total number of keys in the database.
        
        Args:
            info: Redis info dictionary
            
        Returns:
            Total number of keys
        """
        db_info = info.get("db0")
        if db_info:
            # Parse "keys=1234,expires=56" format
            keys_part = db_info.split(',')[0]
            if '=' in keys_part:
                return int(keys_part.split('=')[1])
        return 0