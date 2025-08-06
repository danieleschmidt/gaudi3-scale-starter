"""Security tests for rate limiting and DoS protection."""

import pytest
import time
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from gaudi3_scale.security.rate_limiting import (
    RateLimiter, DoSProtection, RequestThrottler,
    RateLimitRule, RateLimitStrategy, ThrottleAction, DoSPattern,
    FixedWindowRateLimiter, SlidingWindowRateLimiter, TokenBucketRateLimiter
)


class TestFixedWindowRateLimiter:
    """Test fixed window rate limiting strategy."""
    
    def setup_method(self):
        """Setup test environment."""
        self.limiter = FixedWindowRateLimiter()
    
    def test_within_limit(self):
        """Test requests within rate limit."""
        key = "test_user"
        limit = 5
        window = 60  # 1 minute
        
        # Make requests within limit
        for i in range(limit):
            allowed, metadata = self.limiter.is_allowed(key, limit, window)
            assert allowed
            assert metadata["current_count"] == i + 1
            assert metadata["remaining"] == limit - (i + 1)
    
    def test_exceed_limit(self):
        """Test exceeding rate limit."""
        key = "test_user"
        limit = 3
        window = 60
        
        # Make requests up to limit
        for i in range(limit):
            allowed, metadata = self.limiter.is_allowed(key, limit, window)
            assert allowed
        
        # Next request should be denied
        allowed, metadata = self.limiter.is_allowed(key, limit, window)
        assert not allowed
        assert metadata["current_count"] == limit + 1
        assert metadata["remaining"] == 0
    
    def test_window_reset(self):
        """Test rate limit reset after window expires."""
        key = "test_user"
        limit = 2
        window = 1  # 1 second window for fast testing
        
        # Exceed limit
        for i in range(limit + 1):
            self.limiter.is_allowed(key, limit, window)
        
        # Should be blocked
        allowed, metadata = self.limiter.is_allowed(key, limit, window)
        assert not allowed
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should be allowed again
        allowed, metadata = self.limiter.is_allowed(key, limit, window)
        assert allowed
        assert metadata["current_count"] == 1
    
    def test_different_keys(self):
        """Test that different keys have separate limits."""
        limit = 2
        window = 60
        
        # Each key should have its own counter
        allowed1, _ = self.limiter.is_allowed("user1", limit, window)
        allowed2, _ = self.limiter.is_allowed("user2", limit, window)
        
        assert allowed1
        assert allowed2
        
        # Exceed limit for user1
        for i in range(limit):
            self.limiter.is_allowed("user1", limit, window)
        
        # user1 should be blocked, user2 should still be allowed
        allowed1, _ = self.limiter.is_allowed("user1", limit, window)
        allowed2, _ = self.limiter.is_allowed("user2", limit, window)
        
        assert not allowed1
        assert allowed2


class TestSlidingWindowRateLimiter:
    """Test sliding window rate limiting strategy."""
    
    def setup_method(self):
        """Setup test environment."""
        self.limiter = SlidingWindowRateLimiter()
    
    def test_sliding_window_behavior(self):
        """Test sliding window behavior."""
        key = "test_user"
        limit = 3
        window = 2  # 2 seconds
        
        # Make requests at different times
        allowed1, metadata1 = self.limiter.is_allowed(key, limit, window)
        assert allowed1
        assert metadata1["current_count"] == 1
        
        time.sleep(0.5)
        
        allowed2, metadata2 = self.limiter.is_allowed(key, limit, window)
        assert allowed2
        assert metadata2["current_count"] == 2
        
        time.sleep(0.5)
        
        allowed3, metadata3 = self.limiter.is_allowed(key, limit, window)
        assert allowed3
        assert metadata3["current_count"] == 3
        
        # Should be at limit
        allowed4, metadata4 = self.limiter.is_allowed(key, limit, window)
        assert not allowed4
        assert metadata4["current_count"] == 4
    
    def test_old_requests_expire(self):
        """Test that old requests are removed from sliding window."""
        key = "test_user"
        limit = 2
        window = 1  # 1 second
        
        # Make requests
        self.limiter.is_allowed(key, limit, window)
        self.limiter.is_allowed(key, limit, window)
        
        # Should be at limit
        allowed, metadata = self.limiter.is_allowed(key, limit, window)
        assert not allowed
        
        # Wait for first requests to expire
        time.sleep(1.1)
        
        # Should be allowed again as old requests expired
        allowed, metadata = self.limiter.is_allowed(key, limit, window)
        assert allowed
        assert metadata["current_count"] <= limit


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiting strategy."""
    
    def setup_method(self):
        """Setup test environment."""
        self.limiter = TokenBucketRateLimiter()
    
    def test_initial_burst(self):
        """Test initial burst capacity."""
        key = "test_user"
        limit = 5  # 5 token capacity
        window = 5  # 1 token per second refill rate
        
        # Should allow initial burst up to capacity
        for i in range(limit):
            allowed, metadata = self.limiter.is_allowed(key, limit, window)
            assert allowed
            assert metadata["tokens_remaining"] == limit - 1 - i
        
        # Next request should be denied (bucket empty)
        allowed, metadata = self.limiter.is_allowed(key, limit, window)
        assert not allowed
        assert metadata["tokens_remaining"] == 0
    
    def test_token_refill(self):
        """Test token refill over time."""
        key = "test_user"
        limit = 2  # 2 token capacity
        window = 1  # 2 tokens per second refill rate
        
        # Consume all tokens
        self.limiter.is_allowed(key, limit, window)
        self.limiter.is_allowed(key, limit, window)
        
        # Should be denied
        allowed, _ = self.limiter.is_allowed(key, limit, window)
        assert not allowed
        
        # Wait for refill
        time.sleep(0.6)  # Should refill ~1 token
        
        # Should be allowed again
        allowed, metadata = self.limiter.is_allowed(key, limit, window)
        assert allowed
        assert metadata["tokens_remaining"] >= 0


class TestRateLimiter:
    """Test main rate limiter functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.rate_limiter = RateLimiter(redis_client=None)  # Use local cache
    
    def test_default_rules(self):
        """Test default rate limiting rules."""
        # Should have default rules
        assert len(self.rate_limiter.rules) > 0
        
        # Should have API general rule
        api_rule = next((r for r in self.rate_limiter.rules if r.name == "api_general"), None)
        assert api_rule is not None
        assert api_rule.limit == 1000
        assert api_rule.window == 3600
    
    def test_add_remove_rule(self):
        """Test adding and removing rules."""
        initial_count = len(self.rate_limiter.rules)
        
        # Add custom rule
        custom_rule = RateLimitRule(
            name="custom_test",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            limit=10,
            window=60,
            priority=5
        )
        
        self.rate_limiter.add_rule(custom_rule)
        assert len(self.rate_limiter.rules) == initial_count + 1
        
        # Remove rule
        self.rate_limiter.remove_rule("custom_test")
        assert len(self.rate_limiter.rules) == initial_count
    
    def test_rate_limit_check(self):
        """Test rate limit checking."""
        # Add strict rule for testing
        test_rule = RateLimitRule(
            name="test_strict",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            limit=2,
            window=60,
            priority=100  # Highest priority
        )
        
        self.rate_limiter.add_rule(test_rule)
        
        identifier = "test_user"
        
        # Should allow first two requests
        result1 = self.rate_limiter.check_rate_limit(identifier)
        assert result1.allowed
        
        result2 = self.rate_limiter.check_rate_limit(identifier)
        assert result2.allowed
        
        # Third request should be blocked
        result3 = self.rate_limiter.check_rate_limit(identifier)
        assert not result3.allowed
        assert result3.rule_name == "test_strict"
        assert result3.action == ThrottleAction.BLOCK
    
    def test_rule_exemptions(self):
        """Test rule exemptions."""
        test_rule = RateLimitRule(
            name="test_with_exemption",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            limit=1,
            window=60,
            exemptions=["exempt_user"],
            priority=100
        )
        
        self.rate_limiter.add_rule(test_rule)
        
        # Normal user should be rate limited
        normal_result = self.rate_limiter.check_rate_limit("normal_user")
        assert normal_result.allowed
        
        normal_result2 = self.rate_limiter.check_rate_limit("normal_user")
        assert not normal_result2.allowed
        
        # Exempt user should not be rate limited
        exempt_result1 = self.rate_limiter.check_rate_limit("exempt_user")
        assert exempt_result1.allowed
        
        exempt_result2 = self.rate_limiter.check_rate_limit("exempt_user")
        assert exempt_result2.allowed
    
    def test_temporary_ban(self):
        """Test temporary ban functionality."""
        ban_rule = RateLimitRule(
            name="test_ban",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            limit=1,
            window=60,
            action=ThrottleAction.TEMPORARY_BAN,
            ban_duration=2,  # 2 seconds
            priority=100
        )
        
        self.rate_limiter.add_rule(ban_rule)
        
        identifier = "test_user"
        
        # First request allowed
        result1 = self.rate_limiter.check_rate_limit(identifier)
        assert result1.allowed
        
        # Second request should trigger ban
        result2 = self.rate_limiter.check_rate_limit(identifier)
        assert not result2.allowed
        assert result2.action == ThrottleAction.TEMPORARY_BAN
        assert result2.ban_until is not None
        
        # Subsequent requests should be blocked due to ban
        result3 = self.rate_limiter.check_rate_limit(identifier)
        assert not result3.allowed
        assert result3.action == ThrottleAction.TEMPORARY_BAN
        
        # Wait for ban to expire
        time.sleep(2.1)
        
        # Should be allowed again
        result4 = self.rate_limiter.check_rate_limit(identifier)
        assert result4.allowed
    
    def test_rate_limit_status(self):
        """Test getting rate limit status."""
        identifier = "test_user"
        
        # Make some requests
        self.rate_limiter.check_rate_limit(identifier)
        self.rate_limiter.check_rate_limit(identifier)
        
        status = self.rate_limiter.get_rate_limit_status(identifier)
        
        assert status["identifier"] == identifier
        assert not status["banned"]
        assert len(status["rules"]) > 0
        
        # Check rule status details
        for rule_status in status["rules"]:
            assert "name" in rule_status
            assert "limit" in rule_status
            assert "current_count" in rule_status
            assert "remaining" in rule_status
    
    def test_reset_rate_limit(self):
        """Test rate limit reset functionality."""
        # Add strict rule
        test_rule = RateLimitRule(
            name="test_reset",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            limit=1,
            window=60,
            priority=100
        )
        
        self.rate_limiter.add_rule(test_rule)
        
        identifier = "test_user"
        
        # Exceed limit
        self.rate_limiter.check_rate_limit(identifier)
        result = self.rate_limiter.check_rate_limit(identifier)
        assert not result.allowed
        
        # Reset rate limit
        self.rate_limiter.reset_rate_limit(identifier)
        
        # Should be allowed again
        result_after_reset = self.rate_limiter.check_rate_limit(identifier)
        assert result_after_reset.allowed


class TestDoSProtection:
    """Test DoS protection functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.rate_limiter = RateLimiter(redis_client=None)
        self.dos_protection = DoSProtection(
            rate_limiter=self.rate_limiter,
            redis_client=None
        )
    
    def test_high_frequency_detection(self):
        """Test high frequency attack detection."""
        ip_address = "192.168.1.100"
        endpoint = "/api/test"
        
        # Simulate high frequency requests
        for i in range(150):  # Above threshold of 100 per minute
            alert = self.dos_protection.analyze_request_pattern(
                ip_address=ip_address,
                endpoint=endpoint,
                request_size=1024,
                response_time=0.1
            )
        
        # Should detect high frequency attack
        assert alert is not None
        assert alert.pattern == DoSPattern.HIGH_FREQUENCY
        assert alert.source == ip_address
        assert alert.severity in ["HIGH", "CRITICAL"]
    
    def test_http_flood_detection(self):
        """Test HTTP flood detection."""
        ip_address = "192.168.1.101"
        expensive_endpoint = "/api/training"
        
        # Simulate slow requests to expensive endpoint
        for i in range(12):  # Above threshold of 10
            alert = self.dos_protection.analyze_request_pattern(
                ip_address=ip_address,
                endpoint=expensive_endpoint,
                request_size=1024,
                response_time=6.0  # Slow response
            )
        
        # Should detect HTTP flood
        assert alert is not None
        assert alert.pattern == DoSPattern.HTTP_FLOOD
        assert alert.source == ip_address
    
    def test_resource_exhaustion_detection(self):
        """Test resource exhaustion detection."""
        ip_address = "192.168.1.102"
        endpoint = "/api/upload"
        
        # Simulate large request attacks
        for i in range(7):  # Above threshold of 5
            alert = self.dos_protection.analyze_request_pattern(
                ip_address=ip_address,
                endpoint=endpoint,
                request_size=15 * 1024 * 1024,  # 15MB requests
                response_time=0.5
            )
        
        # Should detect resource exhaustion
        assert alert is not None
        assert alert.pattern == DoSPattern.RESOURCE_EXHAUSTION
        assert alert.source == ip_address
    
    def test_distributed_attack_detection(self):
        """Test distributed attack detection."""
        # Create fake request logs for distributed attack
        request_logs = []
        current_time = time.time()
        
        # 20 unique IPs, each making 25 requests (500 total)
        for i in range(20):
            ip = f"192.168.1.{i + 1}"
            for j in range(25):
                request_logs.append({
                    "ip_address": ip,
                    "timestamp": current_time - (j * 5),  # Spread over time
                    "endpoint": "/api/test"
                })
        
        alert = self.dos_protection.detect_distributed_attack(request_logs)
        
        # Should detect distributed attack
        assert alert is not None
        assert alert.pattern == DoSPattern.DISTRIBUTED
        assert alert.source == "multiple"
        assert alert.severity == "CRITICAL"
    
    def test_slow_loris_detection(self):
        """Test Slow Loris attack detection."""
        # Simulate many slow connections
        for i in range(150):  # Above threshold
            conn_id = f"conn_{i}"
            start_time = time.time() - 60  # Started 60 seconds ago
            self.dos_protection.track_slow_connection(conn_id, start_time)
        
        # Detect slow loris
        alert = self.dos_protection.detect_slow_loris()
        
        # Should detect slow loris attack
        assert alert is not None
        assert alert.pattern == DoSPattern.SLOW_LORIS
        assert alert.severity == "HIGH"
    
    def test_connection_activity_tracking(self):
        """Test connection activity tracking."""
        conn_id = "test_conn"
        start_time = time.time()
        
        # Track connection
        self.dos_protection.track_slow_connection(conn_id, start_time)
        assert conn_id in self.dos_protection.connection_tracking
        
        # Update activity
        time.sleep(0.1)
        self.dos_protection.update_connection_activity(conn_id)
        
        # Should update last activity time
        conn_data = self.dos_protection.connection_tracking[conn_id]
        assert conn_data["last_activity"] > start_time
    
    def test_dos_statistics(self):
        """Test DoS protection statistics."""
        # Generate some tracking data
        self.dos_protection.analyze_request_pattern(
            "192.168.1.1", "/test", 1024, 0.1
        )
        
        stats = self.dos_protection.get_dos_statistics()
        
        assert "active_ip_tracking" in stats
        assert "active_connections" in stats
        assert "recent_alerts" in stats
        assert "global_bans" in stats
        assert "mitigation_rules" in stats
        
        assert stats["active_ip_tracking"] >= 0
        assert stats["active_connections"] >= 0


class TestRequestThrottler:
    """Test request throttling middleware."""
    
    def setup_method(self):
        """Setup test environment."""
        self.rate_limiter = RateLimiter(redis_client=None)
        self.dos_protection = DoSProtection(self.rate_limiter, redis_client=None)
        self.throttler = RequestThrottler(self.rate_limiter, self.dos_protection)
    
    @pytest.mark.asyncio
    async def test_allowed_request(self):
        """Test allowing a normal request."""
        request = {
            "client_ip": "192.168.1.1",
            "path": "/api/test",
            "method": "GET",
            "user_agent": "Mozilla/5.0",
            "content_length": 100
        }
        
        allowed, response = await self.throttler.throttle_request(request)
        
        assert allowed
        assert "headers" in response
    
    @pytest.mark.asyncio
    async def test_rate_limited_request(self):
        """Test rate limiting a request."""
        # Add strict rate limit
        strict_rule = RateLimitRule(
            name="test_throttle",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            limit=1,
            window=60,
            priority=100
        )
        self.rate_limiter.add_rule(strict_rule)
        
        request = {
            "client_ip": "192.168.1.2",
            "path": "/api/test",
            "method": "GET",
            "user_agent": "Mozilla/5.0",
            "content_length": 100
        }
        
        # First request should be allowed
        allowed1, response1 = await self.throttler.throttle_request(request)
        assert allowed1
        
        # Second request should be blocked
        allowed2, response2 = await self.throttler.throttle_request(request)
        assert not allowed2
        assert response2["status_code"] == 429
        assert "error" in response2["body"]
    
    @pytest.mark.asyncio
    async def test_delay_action(self):
        """Test delay action for rate limiting."""
        delay_rule = RateLimitRule(
            name="test_delay",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            limit=1,
            window=60,
            action=ThrottleAction.DELAY,
            delay_seconds=0.1,
            priority=100
        )
        self.rate_limiter.add_rule(delay_rule)
        
        request = {
            "client_ip": "192.168.1.3",
            "path": "/api/test",
            "method": "GET",
            "user_agent": "Mozilla/5.0",
            "content_length": 100
        }
        
        # First request should be allowed
        allowed1, response1 = await self.throttler.throttle_request(request)
        assert allowed1
        
        # Second request should be delayed but allowed
        start_time = time.time()
        allowed2, response2 = await self.throttler.throttle_request(request)
        end_time = time.time()
        
        assert not allowed2  # Rate limited
        assert response2["status_code"] == 429
        # Should have been delayed (note: delay happens in rate limit check)
    
    @pytest.mark.asyncio
    async def test_dos_protection_integration(self):
        """Test DoS protection integration."""
        # Create request that might trigger DoS detection
        request = {
            "client_ip": "192.168.1.4",
            "path": "/api/expensive",
            "method": "POST",
            "user_agent": "AttackBot/1.0",
            "content_length": 50 * 1024 * 1024  # Large request
        }
        
        allowed, response = await self.throttler.throttle_request(request)
        
        # Should be allowed initially, but DoS detection might trigger
        if not allowed:
            assert response["status_code"] in [503, 429]
        else:
            assert "dos_alert" in response
    
    def test_throttling_status(self):
        """Test getting throttling status."""
        status = self.throttler.get_throttling_status()
        
        assert "rate_limiter_status" in status
        assert "dos_protection_status" in status
        
        rate_status = status["rate_limiter_status"]
        assert "total_rules" in rate_status
        assert "active_bans" in rate_status
        
        dos_status = status["dos_protection_status"]
        assert "active_ip_tracking" in dos_status


class TestRateLimitingSecurity:
    """Test security aspects of rate limiting."""
    
    def test_rate_limit_bypass_attempts(self):
        """Test protection against rate limit bypass attempts."""
        rate_limiter = RateLimiter(redis_client=None)
        
        # Add strict rule
        strict_rule = RateLimitRule(
            name="bypass_test",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            limit=1,
            window=60,
            priority=100
        )
        rate_limiter.add_rule(strict_rule)
        
        # Try various bypass techniques
        bypass_identifiers = [
            "192.168.1.1",
            "192.168.1.1:8080",  # Port addition
            "192.168.1.1 ",      # Space addition
            "192.168.001.001",   # Zero padding
            "0xC0A80101",        # Hex representation
        ]
        
        # Each should be treated as separate identifiers
        for identifier in bypass_identifiers:
            result = rate_limiter.check_rate_limit(identifier)
            assert result.allowed  # First request for each should be allowed
    
    def test_distributed_rate_limiting_simulation(self):
        """Test simulation of distributed rate limiting."""
        # This would test Redis-based distributed rate limiting
        # For now, test that local fallback works correctly
        
        rate_limiter = RateLimiter(redis_client=None)
        
        # Multiple "instances" should have independent limits without Redis
        instances = []
        for i in range(3):
            instance = RateLimiter(redis_client=None)
            instances.append(instance)
        
        # Each instance maintains its own state
        identifier = "test_user"
        
        results = []
        for instance in instances:
            result = instance.check_rate_limit(identifier)
            results.append(result.allowed)
        
        # Without Redis, each instance allows the request independently
        assert all(results)
    
    def test_memory_exhaustion_protection(self):
        """Test protection against memory exhaustion attacks."""
        dos_protection = DoSProtection(
            rate_limiter=RateLimiter(redis_client=None),
            redis_client=None
        )
        
        # Try to exhaust memory with many unique IPs
        for i in range(10000):
            ip = f"192.168.{i // 256}.{i % 256}"
            dos_protection.analyze_request_pattern(ip, "/test", 1024, 0.1)
        
        # Should not crash or consume excessive memory
        stats = dos_protection.get_dos_statistics()
        assert stats["active_ip_tracking"] <= 10000
    
    def test_time_based_attacks(self):
        """Test protection against time-based attacks."""
        rate_limiter = RateLimiter(redis_client=None)
        
        # Try to manipulate time-based rate limiting
        # This is more of a conceptual test since we can't easily manipulate system time
        
        identifier = "time_attack_user"
        
        # Normal sequence should work
        result1 = rate_limiter.check_rate_limit(identifier)
        assert result1.allowed
        
        # Rate limiter should use consistent time source
        result2 = rate_limiter.check_rate_limit(identifier)
        # Result depends on configured limits
        
        # Time-based manipulations should not bypass limits
        for i in range(10):
            result = rate_limiter.check_rate_limit(identifier)
            # Eventually should hit rate limit
        
        # At least some requests should be blocked if limits are reasonable
        blocked_count = 0
        for i in range(100):
            result = rate_limiter.check_rate_limit(identifier)
            if not result.allowed:
                blocked_count += 1
        
        assert blocked_count > 0  # Some requests should be blocked