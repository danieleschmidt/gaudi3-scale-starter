"""Rate limiting and DoS protection for enterprise security.

This module provides comprehensive rate limiting, request throttling, and
Denial of Service (DoS) protection mechanisms for API endpoints and services.
"""

import time
import json
import asyncio
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(default=None, **kwargs):
        return default
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
from ..logging_utils import get_logger
from ..database.connection import get_redis
from .audit_logging import SecurityAuditLogger, AuditLevel, EventCategory

logger = get_logger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class ThrottleAction(Enum):
    """Actions to take when rate limit is exceeded."""
    BLOCK = "block"
    DELAY = "delay"
    CAPTCHA = "captcha"
    TEMPORARY_BAN = "temporary_ban"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    name: str
    strategy: RateLimitStrategy
    limit: int  # Number of requests
    window: int  # Time window in seconds
    key_function: Optional[Callable] = None  # Function to generate rate limit key
    action: ThrottleAction = ThrottleAction.BLOCK
    delay_seconds: float = 0.0
    ban_duration: int = 300  # Ban duration in seconds
    exemptions: List[str] = field(default_factory=list)  # Exempt IPs/users
    priority: int = 0  # Rule priority (higher = more important)
    enabled: bool = True


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    rule_name: Optional[str] = None
    limit: Optional[int] = None
    window: Optional[int] = None
    current_count: int = 0
    remaining: int = 0
    reset_time: Optional[float] = None
    action: Optional[ThrottleAction] = None
    delay_seconds: float = 0.0
    ban_until: Optional[datetime] = None
    headers: Dict[str, str] = field(default_factory=dict)


class DoSPattern(Enum):
    """DoS attack patterns to detect."""
    HIGH_FREQUENCY = "high_frequency"
    DISTRIBUTED = "distributed"
    SLOW_LORIS = "slow_loris"
    HTTP_FLOOD = "http_flood"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class DoSAlert:
    """DoS attack alert."""
    pattern: DoSPattern
    source: str  # IP address or identifier
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    detected_at: datetime
    request_count: int
    time_window: int
    details: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""


class FixedWindowRateLimiter:
    """Fixed window rate limiter implementation."""
    
    def __init__(self, redis_client=None):
        """Initialize fixed window rate limiter."""
        self.redis_client = redis_client
        self.local_cache = {}  # Fallback for when Redis is unavailable
        self.logger = logger.getChild(self.__class__.__name__)
    
    def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under fixed window policy.
        
        Args:
            key: Rate limit key (usually IP or user ID)
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (allowed, metadata)
        """
        current_time = time.time()
        window_start = int(current_time // window) * window
        cache_key = f"rate_limit:fixed:{key}:{window_start}"
        
        try:
            if self.redis_client:
                # Use Redis for distributed rate limiting
                pipe = self.redis_client.pipeline()
                pipe.incr(cache_key)
                pipe.expire(cache_key, window)
                results = pipe.execute()
                
                current_count = results[0]
            else:
                # Fallback to local cache
                if cache_key not in self.local_cache:
                    self.local_cache[cache_key] = {"count": 0, "expires": window_start + window}
                
                # Clean expired entries
                if current_time > self.local_cache[cache_key]["expires"]:
                    self.local_cache[cache_key] = {"count": 0, "expires": window_start + window}
                
                self.local_cache[cache_key]["count"] += 1
                current_count = self.local_cache[cache_key]["count"]
            
            allowed = current_count <= limit
            remaining = max(0, limit - current_count)
            reset_time = window_start + window
            
            metadata = {
                "current_count": current_count,
                "remaining": remaining,
                "reset_time": reset_time,
                "window_start": window_start
            }
            
            return allowed, metadata
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed for {key}: {e}")
            # Fail open - allow the request
            return True, {"error": str(e)}


class SlidingWindowRateLimiter:
    """Sliding window log rate limiter implementation."""
    
    def __init__(self, redis_client=None):
        """Initialize sliding window rate limiter."""
        self.redis_client = redis_client
        self.local_cache = defaultdict(deque)
        self.logger = logger.getChild(self.__class__.__name__)
    
    def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under sliding window policy.
        
        Args:
            key: Rate limit key
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (allowed, metadata)
        """
        current_time = time.time()
        window_start = current_time - window
        
        try:
            if self.redis_client:
                # Use Redis sorted sets for distributed sliding window
                cache_key = f"rate_limit:sliding:{key}"
                
                pipe = self.redis_client.pipeline()
                # Remove old entries
                pipe.zremrangebyscore(cache_key, 0, window_start)
                # Add current request
                pipe.zadd(cache_key, {str(current_time): current_time})
                # Count current requests
                pipe.zcard(cache_key)
                # Set expiry
                pipe.expire(cache_key, window)
                
                results = pipe.execute()
                current_count = results[2]
            else:
                # Fallback to local cache
                request_times = self.local_cache[key]
                
                # Remove old requests
                while request_times and request_times[0] < window_start:
                    request_times.popleft()
                
                # Add current request
                request_times.append(current_time)
                current_count = len(request_times)
            
            allowed = current_count <= limit
            remaining = max(0, limit - current_count)
            
            metadata = {
                "current_count": current_count,
                "remaining": remaining,
                "window_start": window_start
            }
            
            return allowed, metadata
            
        except Exception as e:
            self.logger.error(f"Sliding window rate limit check failed for {key}: {e}")
            return True, {"error": str(e)}


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, redis_client=None):
        """Initialize token bucket rate limiter."""
        self.redis_client = redis_client
        self.local_buckets = {}
        self.logger = logger.getChild(self.__class__.__name__)
    
    def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under token bucket policy.
        
        Args:
            key: Rate limit key
            limit: Bucket capacity (tokens)
            window: Refill window in seconds (rate = limit/window tokens per second)
            
        Returns:
            Tuple of (allowed, metadata)
        """
        current_time = time.time()
        refill_rate = limit / window  # tokens per second
        
        try:
            if self.redis_client:
                # Use Redis for distributed token bucket
                cache_key = f"rate_limit:bucket:{key}"
                
                # Lua script for atomic token bucket operation
                lua_script = """
                local key = KEYS[1]
                local capacity = tonumber(ARGV[1])
                local refill_rate = tonumber(ARGV[2])
                local current_time = tonumber(ARGV[3])
                local tokens_requested = tonumber(ARGV[4])
                
                local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
                local tokens = tonumber(bucket[1]) or capacity
                local last_refill = tonumber(bucket[2]) or current_time
                
                -- Calculate tokens to add based on time elapsed
                local time_elapsed = current_time - last_refill
                local tokens_to_add = time_elapsed * refill_rate
                tokens = math.min(capacity, tokens + tokens_to_add)
                
                local allowed = 0
                if tokens >= tokens_requested then
                    tokens = tokens - tokens_requested
                    allowed = 1
                end
                
                -- Update bucket state
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
                redis.call('EXPIRE', key, 3600)  -- 1 hour expiry
                
                return {allowed, tokens}
                """
                
                result = self.redis_client.eval(
                    lua_script, 1, cache_key, limit, refill_rate, current_time, 1
                )
                
                allowed = bool(result[0])
                tokens_remaining = result[1]
                
            else:
                # Fallback to local cache
                if key not in self.local_buckets:
                    self.local_buckets[key] = {
                        "tokens": limit,
                        "last_refill": current_time
                    }
                
                bucket = self.local_buckets[key]
                time_elapsed = current_time - bucket["last_refill"]
                tokens_to_add = time_elapsed * refill_rate
                
                bucket["tokens"] = min(limit, bucket["tokens"] + tokens_to_add)
                bucket["last_refill"] = current_time
                
                allowed = bucket["tokens"] >= 1
                if allowed:
                    bucket["tokens"] -= 1
                
                tokens_remaining = bucket["tokens"]
            
            metadata = {
                "tokens_remaining": tokens_remaining,
                "refill_rate": refill_rate,
                "capacity": limit
            }
            
            return allowed, metadata
            
        except Exception as e:
            self.logger.error(f"Token bucket rate limit check failed for {key}: {e}")
            return True, {"error": str(e)}


class RateLimiter:
    """Main rate limiter with multiple strategies and rules."""
    
    def __init__(self, 
                 redis_client=None,
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """Initialize rate limiter.
        
        Args:
            redis_client: Redis client for distributed rate limiting
            audit_logger: Security audit logger
        """
        self.redis_client = redis_client or (get_redis().get_client() if REDIS_AVAILABLE else None)
        self.audit_logger = audit_logger
        
        # Rate limit strategies
        self.strategies = {
            RateLimitStrategy.FIXED_WINDOW: FixedWindowRateLimiter(self.redis_client),
            RateLimitStrategy.SLIDING_WINDOW: SlidingWindowRateLimiter(self.redis_client),
            RateLimitStrategy.TOKEN_BUCKET: TokenBucketRateLimiter(self.redis_client)
        }
        
        # Rate limit rules
        self.rules: List[RateLimitRule] = []
        self.global_bans: Dict[str, datetime] = {}  # Temporary bans
        
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Setup default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default rate limiting rules."""
        # API general rate limit
        self.add_rule(RateLimitRule(
            name="api_general",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            limit=1000,
            window=3600,  # 1000 requests per hour
            priority=1
        ))
        
        # Login endpoint protection
        self.add_rule(RateLimitRule(
            name="auth_login",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            limit=5,
            window=300,  # 5 attempts per 5 minutes
            action=ThrottleAction.TEMPORARY_BAN,
            ban_duration=900,  # 15 minute ban
            priority=10
        ))
        
        # High-frequency protection
        self.add_rule(RateLimitRule(
            name="burst_protection",
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            limit=10,
            window=1,  # 10 requests per second burst
            action=ThrottleAction.DELAY,
            delay_seconds=1.0,
            priority=5
        ))
    
    def add_rule(self, rule: RateLimitRule):
        """Add a rate limiting rule."""
        self.rules.append(rule)
        # Sort rules by priority (higher priority first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        self.logger.info(f"Added rate limiting rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a rate limiting rule."""
        self.rules = [r for r in self.rules if r.name != rule_name]
        self.logger.info(f"Removed rate limiting rule: {rule_name}")
    
    def check_rate_limit(self, 
                        identifier: str,
                        endpoint: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None) -> RateLimitResult:
        """Check rate limit for a request.
        
        Args:
            identifier: Request identifier (IP address, user ID, etc.)
            endpoint: API endpoint being accessed
            context: Additional context for rate limiting
            
        Returns:
            Rate limit result
        """
        context = context or {}
        
        # Check global bans first
        if identifier in self.global_bans:
            ban_until = self.global_bans[identifier]
            if datetime.now(timezone.utc) < ban_until:
                return RateLimitResult(
                    allowed=False,
                    action=ThrottleAction.TEMPORARY_BAN,
                    ban_until=ban_until,
                    headers={
                        "X-RateLimit-Banned": "true",
                        "X-RateLimit-Ban-Until": ban_until.isoformat()
                    }
                )
            else:
                # Ban expired, remove it
                del self.global_bans[identifier]
        
        # Check each rule in priority order
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Check exemptions
            if identifier in rule.exemptions:
                continue
            
            # Generate rate limit key
            if rule.key_function:
                key = rule.key_function(identifier, endpoint, context)
            else:
                key = f"{identifier}:{endpoint or 'default'}"
            
            # Check if rule applies
            if endpoint and rule.name.startswith("auth_") and "auth" not in (endpoint or ""):
                continue
            
            # Apply rate limiting strategy
            strategy = self.strategies[rule.strategy]
            allowed, metadata = strategy.is_allowed(key, rule.limit, rule.window)
            
            if not allowed:
                # Rate limit exceeded
                result = RateLimitResult(
                    allowed=False,
                    rule_name=rule.name,
                    limit=rule.limit,
                    window=rule.window,
                    current_count=metadata.get("current_count", 0),
                    remaining=0,
                    reset_time=metadata.get("reset_time"),
                    action=rule.action,
                    delay_seconds=rule.delay_seconds
                )
                
                # Apply action
                if rule.action == ThrottleAction.TEMPORARY_BAN:
                    ban_until = datetime.now(timezone.utc) + timedelta(seconds=rule.ban_duration)
                    self.global_bans[identifier] = ban_until
                    result.ban_until = ban_until
                
                # Set response headers
                result.headers = self._generate_rate_limit_headers(rule, metadata)
                
                # Log rate limit violation
                if self.audit_logger:
                    self.audit_logger.log_security_event(
                        event_type="rate_limit_exceeded",
                        risk_score=3.0,
                        threat_indicators=[
                            f"rule:{rule.name}",
                            f"identifier:{identifier}",
                            f"count:{metadata.get('current_count', 0)}"
                        ]
                    )
                
                self.logger.warning(f"Rate limit exceeded: {rule.name} for {identifier}")
                return result
        
        # All rules passed
        return RateLimitResult(
            allowed=True,
            headers={"X-RateLimit-Status": "OK"}
        )
    
    def _generate_rate_limit_headers(self, rule: RateLimitRule, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Generate HTTP headers for rate limit response."""
        headers = {
            "X-RateLimit-Limit": str(rule.limit),
            "X-RateLimit-Window": str(rule.window),
            "X-RateLimit-Remaining": str(metadata.get("remaining", 0))
        }
        
        if "reset_time" in metadata:
            headers["X-RateLimit-Reset"] = str(int(metadata["reset_time"]))
        
        if rule.action == ThrottleAction.DELAY:
            headers["X-RateLimit-Delay"] = str(rule.delay_seconds)
        
        return headers
    
    def get_rate_limit_status(self, identifier: str) -> Dict[str, Any]:
        """Get current rate limit status for identifier."""
        status = {
            "identifier": identifier,
            "banned": identifier in self.global_bans,
            "ban_until": self.global_bans.get(identifier),
            "rules": []
        }
        
        for rule in self.rules:
            if not rule.enabled or identifier in rule.exemptions:
                continue
            
            key = f"{identifier}:default"
            strategy = self.strategies[rule.strategy]
            
            # Get current status without incrementing
            try:
                if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                    current_time = time.time()
                    window_start = int(current_time // rule.window) * rule.window
                    cache_key = f"rate_limit:fixed:{key}:{window_start}"
                    
                    if self.redis_client:
                        current_count = self.redis_client.get(cache_key) or 0
                    else:
                        current_count = 0
                    
                elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                    cache_key = f"rate_limit:sliding:{key}"
                    
                    if self.redis_client:
                        current_time = time.time()
                        window_start = current_time - rule.window
                        current_count = self.redis_client.zcount(cache_key, window_start, current_time)
                    else:
                        current_count = len(self.strategies[rule.strategy].local_cache.get(key, []))
                
                elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                    cache_key = f"rate_limit:bucket:{key}"
                    
                    if self.redis_client:
                        bucket_data = self.redis_client.hmget(cache_key, "tokens")
                        current_count = int(bucket_data[0] or rule.limit)
                    else:
                        bucket = self.strategies[rule.strategy].local_buckets.get(key, {"tokens": rule.limit})
                        current_count = bucket["tokens"]
                
                else:
                    current_count = 0
                
                rule_status = {
                    "name": rule.name,
                    "strategy": rule.strategy.value,
                    "limit": rule.limit,
                    "window": rule.window,
                    "current_count": int(current_count),
                    "remaining": max(0, rule.limit - int(current_count))
                }
                
                status["rules"].append(rule_status)
                
            except Exception as e:
                self.logger.error(f"Error getting status for rule {rule.name}: {e}")
        
        return status
    
    def reset_rate_limit(self, identifier: str, rule_name: Optional[str] = None):
        """Reset rate limit for identifier.
        
        Args:
            identifier: Request identifier
            rule_name: Specific rule to reset (all if None)
        """
        if rule_name:
            rules = [r for r in self.rules if r.name == rule_name]
        else:
            rules = self.rules
        
        for rule in rules:
            key = f"{identifier}:default"
            
            try:
                if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                    if self.redis_client:
                        pattern = f"rate_limit:fixed:{key}:*"
                        for cache_key in self.redis_client.scan_iter(match=pattern):
                            self.redis_client.delete(cache_key)
                
                elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                    cache_key = f"rate_limit:sliding:{key}"
                    if self.redis_client:
                        self.redis_client.delete(cache_key)
                    else:
                        self.strategies[rule.strategy].local_cache.pop(key, None)
                
                elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                    cache_key = f"rate_limit:bucket:{key}"
                    if self.redis_client:
                        self.redis_client.hmset(cache_key, {
                            "tokens": rule.limit,
                            "last_refill": time.time()
                        })
                    else:
                        self.strategies[rule.strategy].local_buckets[key] = {
                            "tokens": rule.limit,
                            "last_refill": time.time()
                        }
                
            except Exception as e:
                self.logger.error(f"Error resetting rate limit for {rule.name}: {e}")
        
        # Remove from global bans
        if identifier in self.global_bans:
            del self.global_bans[identifier]
        
        self.logger.info(f"Rate limit reset for {identifier}")


class DoSProtection:
    """Denial of Service attack detection and protection."""
    
    def __init__(self, 
                 rate_limiter: RateLimiter,
                 audit_logger: Optional[SecurityAuditLogger] = None,
                 redis_client=None):
        """Initialize DoS protection.
        
        Args:
            rate_limiter: Rate limiter instance
            audit_logger: Security audit logger
            redis_client: Redis client
        """
        self.rate_limiter = rate_limiter
        self.audit_logger = audit_logger
        self.redis_client = redis_client or rate_limiter.redis_client
        
        # DoS detection parameters
        self.high_frequency_threshold = 100  # requests per minute
        self.distributed_threshold = 10  # unique IPs making requests
        self.slow_loris_timeout = 30  # seconds for slow connections
        
        # Attack detection state
        self.request_tracking = defaultdict(lambda: {"count": 0, "first_seen": time.time()})
        self.connection_tracking = {}
        self.alerts_sent = set()
        
        self.logger = logger.getChild(self.__class__.__name__)
    
    def analyze_request_pattern(self, 
                               ip_address: str,
                               endpoint: str,
                               request_size: int,
                               response_time: float,
                               user_agent: Optional[str] = None) -> Optional[DoSAlert]:
        """Analyze request pattern for DoS indicators.
        
        Args:
            ip_address: Client IP address
            endpoint: Requested endpoint
            request_size: Request size in bytes
            response_time: Response time in seconds
            user_agent: User agent string
            
        Returns:
            DoS alert if attack pattern detected
        """
        current_time = time.time()
        
        # Track request frequency per IP
        ip_key = f"dos_track:{ip_address}"
        self.request_tracking[ip_key]["count"] += 1
        
        # High frequency detection
        time_window = current_time - self.request_tracking[ip_key]["first_seen"]
        if time_window > 0:
            requests_per_minute = (self.request_tracking[ip_key]["count"] / time_window) * 60
            
            if requests_per_minute > self.high_frequency_threshold:
                alert = DoSAlert(
                    pattern=DoSPattern.HIGH_FREQUENCY,
                    source=ip_address,
                    severity="HIGH",
                    detected_at=datetime.now(timezone.utc),
                    request_count=self.request_tracking[ip_key]["count"],
                    time_window=int(time_window),
                    details={
                        "requests_per_minute": requests_per_minute,
                        "endpoint": endpoint,
                        "user_agent": user_agent
                    },
                    recommended_action="Rate limit or block IP address"
                )
                
                return self._process_alert(alert)
        
        # HTTP flood detection (many requests to expensive endpoints)
        if response_time > 5.0 and endpoint in ["/api/training", "/api/cluster"]:
            flood_key = f"http_flood:{ip_address}"
            flood_count = self.request_tracking.get(flood_key, {"count": 0})["count"] + 1
            
            if flood_count > 10:  # 10 slow requests
                alert = DoSAlert(
                    pattern=DoSPattern.HTTP_FLOOD,
                    source=ip_address,
                    severity="MEDIUM",
                    detected_at=datetime.now(timezone.utc),
                    request_count=flood_count,
                    time_window=300,  # 5 minutes
                    details={
                        "endpoint": endpoint,
                        "average_response_time": response_time,
                        "user_agent": user_agent
                    },
                    recommended_action="Implement request queuing or rate limiting"
                )
                
                return self._process_alert(alert)
        
        # Resource exhaustion detection (large requests)
        if request_size > 10 * 1024 * 1024:  # 10MB
            large_request_key = f"large_requests:{ip_address}"
            large_count = self.request_tracking.get(large_request_key, {"count": 0})["count"] + 1
            
            if large_count > 5:
                alert = DoSAlert(
                    pattern=DoSPattern.RESOURCE_EXHAUSTION,
                    source=ip_address,
                    severity="MEDIUM",
                    detected_at=datetime.now(timezone.utc),
                    request_count=large_count,
                    time_window=300,
                    details={
                        "request_size": request_size,
                        "endpoint": endpoint
                    },
                    recommended_action="Implement request size limits"
                )
                
                return self._process_alert(alert)
        
        return None
    
    def detect_distributed_attack(self, request_logs: List[Dict[str, Any]]) -> Optional[DoSAlert]:
        """Detect distributed DoS attacks across multiple IPs.
        
        Args:
            request_logs: List of recent request logs
            
        Returns:
            DoS alert if distributed attack detected
        """
        current_time = time.time()
        time_window = 300  # 5 minutes
        window_start = current_time - time_window
        
        # Filter requests within time window
        recent_requests = [
            log for log in request_logs 
            if log.get("timestamp", 0) >= window_start
        ]
        
        # Count unique IPs and total requests
        ip_counts = defaultdict(int)
        for log in recent_requests:
            ip_counts[log.get("ip_address", "unknown")] += 1
        
        unique_ips = len(ip_counts)
        total_requests = len(recent_requests)
        
        # Check for distributed attack pattern
        if (unique_ips >= self.distributed_threshold and 
            total_requests > unique_ips * 20):  # Average 20+ requests per IP
            
            alert = DoSAlert(
                pattern=DoSPattern.DISTRIBUTED,
                source="multiple",
                severity="CRITICAL",
                detected_at=datetime.now(timezone.utc),
                request_count=total_requests,
                time_window=int(time_window),
                details={
                    "unique_ips": unique_ips,
                    "average_requests_per_ip": total_requests / unique_ips,
                    "top_ips": dict(sorted(ip_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:10])
                },
                recommended_action="Implement distributed rate limiting and consider DDoS mitigation"
            )
            
            return self._process_alert(alert)
        
        return None
    
    def track_slow_connection(self, connection_id: str, start_time: float):
        """Track slow connection for Slow Loris detection.
        
        Args:
            connection_id: Unique connection identifier
            start_time: Connection start timestamp
        """
        self.connection_tracking[connection_id] = {
            "start_time": start_time,
            "last_activity": start_time
        }
    
    def update_connection_activity(self, connection_id: str):
        """Update connection activity timestamp."""
        if connection_id in self.connection_tracking:
            self.connection_tracking[connection_id]["last_activity"] = time.time()
    
    def detect_slow_loris(self) -> Optional[DoSAlert]:
        """Detect Slow Loris attacks (many slow connections).
        
        Returns:
            DoS alert if Slow Loris attack detected
        """
        current_time = time.time()
        slow_connections = []
        
        # Find connections that are slow (no activity for timeout period)
        for conn_id, conn_data in self.connection_tracking.items():
            last_activity = conn_data["last_activity"]
            if current_time - last_activity > self.slow_loris_timeout:
                slow_connections.append(conn_id)
        
        # Clean up old connections
        for conn_id in slow_connections:
            del self.connection_tracking[conn_id]
        
        # Alert if too many slow connections
        if len(slow_connections) > 100:
            alert = DoSAlert(
                pattern=DoSPattern.SLOW_LORIS,
                source="multiple",
                severity="HIGH",
                detected_at=datetime.now(timezone.utc),
                request_count=len(slow_connections),
                time_window=self.slow_loris_timeout,
                details={
                    "slow_connections": len(slow_connections),
                    "timeout_threshold": self.slow_loris_timeout
                },
                recommended_action="Implement connection timeouts and limits"
            )
            
            return self._process_alert(alert)
        
        return None
    
    def _process_alert(self, alert: DoSAlert) -> DoSAlert:
        """Process and log DoS alert.
        
        Args:
            alert: DoS alert to process
            
        Returns:
            Processed alert
        """
        # Avoid duplicate alerts
        alert_key = f"{alert.pattern.value}:{alert.source}"
        if alert_key in self.alerts_sent:
            return alert
        
        self.alerts_sent.add(alert_key)
        
        # Log security event
        if self.audit_logger:
            risk_score = {
                "LOW": 3.0,
                "MEDIUM": 6.0,
                "HIGH": 8.0,
                "CRITICAL": 10.0
            }.get(alert.severity, 5.0)
            
            self.audit_logger.log_security_event(
                event_type=f"dos_attack_{alert.pattern.value}",
                risk_score=risk_score,
                threat_indicators=[
                    f"source:{alert.source}",
                    f"pattern:{alert.pattern.value}",
                    f"requests:{alert.request_count}"
                ]
            )
        
        # Auto-mitigation
        if alert.severity in ["HIGH", "CRITICAL"]:
            self._apply_mitigation(alert)
        
        self.logger.critical(f"DoS attack detected: {alert.pattern.value} from {alert.source}")
        
        return alert
    
    def _apply_mitigation(self, alert: DoSAlert):
        """Apply automatic mitigation for DoS attack.
        
        Args:
            alert: DoS alert requiring mitigation
        """
        if alert.source != "multiple":
            # Single source attack - block IP
            self.rate_limiter.global_bans[alert.source] = (
                datetime.now(timezone.utc) + timedelta(minutes=30)
            )
            
            self.logger.warning(f"Auto-blocked IP {alert.source} for DoS attack")
        
        # Add aggressive rate limiting rule
        mitigation_rule = RateLimitRule(
            name=f"dos_mitigation_{int(time.time())}",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            limit=10,
            window=60,  # 10 requests per minute
            action=ThrottleAction.TEMPORARY_BAN,
            ban_duration=1800,  # 30 minute ban
            priority=100  # Highest priority
        )
        
        self.rate_limiter.add_rule(mitigation_rule)
        
        self.logger.info(f"Applied DoS mitigation rule: {mitigation_rule.name}")
    
    def get_dos_statistics(self) -> Dict[str, Any]:
        """Get DoS protection statistics."""
        current_time = time.time()
        
        # Count active tracking entries
        active_tracking = sum(
            1 for data in self.request_tracking.values()
            if current_time - data["first_seen"] < 3600  # Active in last hour
        )
        
        return {
            "active_ip_tracking": active_tracking,
            "active_connections": len(self.connection_tracking),
            "recent_alerts": len(self.alerts_sent),
            "global_bans": len(self.rate_limiter.global_bans),
            "mitigation_rules": len([
                r for r in self.rate_limiter.rules 
                if r.name.startswith("dos_mitigation_")
            ])
        }


class RequestThrottler:
    """Request throttling middleware for web applications."""
    
    def __init__(self, 
                 rate_limiter: RateLimiter,
                 dos_protection: DoSProtection):
        """Initialize request throttler.
        
        Args:
            rate_limiter: Rate limiter instance
            dos_protection: DoS protection instance
        """
        self.rate_limiter = rate_limiter
        self.dos_protection = dos_protection
        self.logger = logger.getChild(self.__class__.__name__)
    
    async def throttle_request(self, 
                              request: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Throttle incoming request.
        
        Args:
            request: Request information dictionary
            
        Returns:
            Tuple of (allowed, response_data)
        """
        start_time = time.time()
        
        # Extract request information
        ip_address = request.get("client_ip", "unknown")
        endpoint = request.get("path", "")
        method = request.get("method", "GET")
        user_agent = request.get("user_agent", "")
        request_size = request.get("content_length", 0)
        
        # Check rate limits
        rate_limit_result = self.rate_limiter.check_rate_limit(
            identifier=ip_address,
            endpoint=endpoint,
            context={"method": method, "user_agent": user_agent}
        )
        
        if not rate_limit_result.allowed:
            response_data = {
                "status_code": 429,
                "headers": rate_limit_result.headers,
                "body": {
                    "error": "Rate limit exceeded",
                    "rule": rate_limit_result.rule_name,
                    "limit": rate_limit_result.limit,
                    "window": rate_limit_result.window,
                    "reset_time": rate_limit_result.reset_time
                }
            }
            
            if rate_limit_result.action == ThrottleAction.DELAY:
                await asyncio.sleep(rate_limit_result.delay_seconds)
                response_data["body"]["delayed"] = rate_limit_result.delay_seconds
            
            elif rate_limit_result.action == ThrottleAction.TEMPORARY_BAN:
                response_data["status_code"] = 403
                response_data["body"]["banned_until"] = rate_limit_result.ban_until
            
            return False, response_data
        
        # Analyze for DoS patterns
        response_time = time.time() - start_time
        dos_alert = self.dos_protection.analyze_request_pattern(
            ip_address=ip_address,
            endpoint=endpoint,
            request_size=request_size,
            response_time=response_time,
            user_agent=user_agent
        )
        
        if dos_alert and dos_alert.severity in ["HIGH", "CRITICAL"]:
            # Block request due to DoS detection
            response_data = {
                "status_code": 503,
                "headers": {"X-DoS-Protection": "blocked"},
                "body": {
                    "error": "Request blocked due to DoS protection",
                    "pattern": dos_alert.pattern.value,
                    "severity": dos_alert.severity
                }
            }
            
            return False, response_data
        
        # Request allowed
        return True, {
            "headers": rate_limit_result.headers,
            "dos_alert": dos_alert
        }
    
    def get_throttling_status(self) -> Dict[str, Any]:
        """Get current throttling status."""
        return {
            "rate_limiter_status": {
                "total_rules": len(self.rate_limiter.rules),
                "active_bans": len(self.rate_limiter.global_bans)
            },
            "dos_protection_status": self.dos_protection.get_dos_statistics()
        }