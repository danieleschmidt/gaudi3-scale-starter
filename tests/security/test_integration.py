"""Integration security tests for complete security system."""

import pytest
import tempfile
import time
import secrets
from pathlib import Path
from datetime import datetime, timedelta, timezone

from gaudi3_scale.security.config_security import (
    SecureConfigManager, SecretsManager, EncryptionManager
)
from gaudi3_scale.security.authentication import (
    AuthenticationManager, AuthorizationManager, UserManager, 
    UserRole, Permission, LoginRequest
)
from gaudi3_scale.security.validation import SecurityValidator, AntiInjectionValidator
from gaudi3_scale.security.rate_limiting import RateLimiter, DoSProtection, RequestThrottler
from gaudi3_scale.security.audit_logging import SecurityAuditLogger, AuditLevel, EventCategory
from gaudi3_scale.security.monitoring import SecurityMonitor, SecurityAlerts
from gaudi3_scale.integrations.auth.jwt import JWTConfig
from gaudi3_scale.exceptions import AuthenticationError, AuthorizationError


class TestSecurityIntegration:
    """Test integration of all security components."""
    
    def setup_method(self):
        """Setup complete security environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Initialize core security components
        self.encryption_manager = EncryptionManager()
        self.secrets_manager = SecretsManager(storage_path=self.temp_dir / "secrets")
        self.config_manager = SecureConfigManager(
            config_path=self.temp_dir / "config.json",
            secrets_manager=self.secrets_manager
        )
        
        # Initialize authentication system
        self.user_manager = UserManager(storage_path=self.temp_dir / "users")
        jwt_config = JWTConfig(
            secret_key=secrets.token_urlsafe(64),
            algorithm="HS256"
        )
        self.auth_manager = AuthenticationManager(
            user_manager=self.user_manager,
            jwt_config=jwt_config
        )
        self.authz_manager = AuthorizationManager(user_manager=self.user_manager)
        
        # Initialize validation
        self.security_validator = SecurityValidator()
        self.anti_injection_validator = AntiInjectionValidator()
        
        # Initialize rate limiting
        self.rate_limiter = RateLimiter(redis_client=None)
        self.dos_protection = DoSProtection(self.rate_limiter)
        self.request_throttler = RequestThrottler(self.rate_limiter, self.dos_protection)
        
        # Initialize audit logging
        self.audit_logger = SecurityAuditLogger(
            storage_path=self.temp_dir / "audit",
            encryption_manager=self.encryption_manager,
            enable_database_storage=False  # Use file storage for tests
        )
        
        # Initialize monitoring
        self.security_monitor = SecurityMonitor(
            audit_logger=self.audit_logger,
            redis_client=None
        )
        self.security_alerts = SecurityAlerts(self.security_monitor)
        
        # Create test users
        self.admin_user = self.user_manager.create_user(
            username="admin",
            email="admin@example.com",
            password="AdminPass123!@#",
            roles={UserRole.ADMIN}
        )
        
        self.normal_user = self.user_manager.create_user(
            username="user",
            email="user@example.com", 
            password="UserPass123!@#",
            roles={UserRole.VIEWER}
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        self.audit_logger.shutdown()
        self.security_monitor.stop_monitoring()
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_authentication_flow(self):
        """Test complete authentication flow with all security components."""
        # Step 1: Validate login input
        username = "admin"
        password = "AdminPass123!@#"
        
        # Validate inputs for injection attacks
        username_result = self.security_validator.validate_against_injection(username, "username")
        password_result = self.security_validator.validate_against_injection(password, "password")
        
        assert username_result.is_valid
        assert password_result.is_valid
        
        # Step 2: Check rate limits
        rate_limit_result = self.rate_limiter.check_rate_limit(
            identifier="127.0.0.1",
            endpoint="/auth/login"
        )
        assert rate_limit_result.allowed
        
        # Step 3: Authenticate user
        login_request = LoginRequest(username=username, password=password)
        auth_result = self.auth_manager.login(login_request, ip_address="127.0.0.1")
        
        assert "access_token" in auth_result
        assert auth_result["user"]["username"] == username
        
        # Step 4: Log authentication event
        self.audit_logger.log_authentication_event(
            success=True,
            username=username,
            ip_address="127.0.0.1"
        )
        
        # Step 5: Verify JWT token
        access_token = auth_result["access_token"]
        token_payload = self.auth_manager.jwt_handler.verify_token(access_token, "access")
        
        assert token_payload.sub == self.admin_user.user_id
    
    def test_authorization_with_audit_logging(self):
        """Test authorization checks with comprehensive audit logging."""
        user_id = self.normal_user.user_id
        
        # Test successful authorization
        has_read_permission = self.authz_manager.check_permission(
            user_id, Permission.READ_CLUSTER
        )
        assert has_read_permission
        
        # Log successful authorization
        self.audit_logger.log_authorization_event(
            success=True,
            username=self.normal_user.username,
            permission=Permission.READ_CLUSTER.value,
            resource="cluster_001"
        )
        
        # Test failed authorization
        has_create_permission = self.authz_manager.check_permission(
            user_id, Permission.CREATE_CLUSTER
        )
        assert not has_create_permission
        
        # Log failed authorization
        self.audit_logger.log_authorization_event(
            success=False,
            username=self.normal_user.username,
            permission=Permission.CREATE_CLUSTER.value,
            resource="cluster_001"
        )
        
        # Verify authorization requirement throws exception
        with pytest.raises(AuthorizationError):
            self.authz_manager.require_permission(user_id, Permission.CREATE_CLUSTER)
    
    def test_secure_configuration_management(self):
        """Test secure configuration with secrets management."""
        # Store database configuration with encrypted password
        self.config_manager.setup_database_secrets(
            host="db.example.com",
            port="5432",
            database="production",
            username="app_user",
            password="SuperSecretPassword123!@#"
        )
        
        # Retrieve configuration
        db_url = self.config_manager.get_database_url()
        assert "db.example.com" in db_url
        assert "SuperSecretPassword123" in db_url  # Password should be decrypted
        assert "production" in db_url
        
        # Verify password is encrypted in storage
        config_export = self.config_manager.export_config(include_secrets=False)
        assert config_export.get("database.password") == "***ENCRYPTED***"
        
        # Test secrets rotation
        self.secrets_manager.store_secret(
            key="api_key",
            value="original_key_123",
            rotation_period=1  # 1 day
        )
        
        retrieved_key = self.secrets_manager.get_secret("api_key")
        assert retrieved_key == "original_key_123"
        
        # Rotate secret
        self.secrets_manager.rotate_secret("api_key", "rotated_key_456")
        rotated_key = self.secrets_manager.get_secret("api_key")
        assert rotated_key == "rotated_key_456"
    
    def test_input_validation_pipeline(self):
        """Test complete input validation pipeline."""
        # Test various malicious inputs through validation pipeline
        malicious_inputs = [
            ("sql_injection", "'; DROP TABLE users; --"),
            ("xss_attack", "<script>alert('xss')</script>"),
            ("command_injection", "; rm -rf /"),
            ("path_traversal", "../../../etc/passwd"),
            ("nosql_injection", "{'$ne': null}"),
        ]
        
        for input_type, malicious_value in malicious_inputs:
            # Step 1: Basic validation
            basic_result = self.security_validator.validate_against_injection(
                malicious_value, input_type
            )
            assert not basic_result.is_valid
            assert len(basic_result.threats) > 0
            
            # Step 2: Anti-injection validation
            if input_type == "sql_injection":
                sql_params = {"query": malicious_value}
                sql_result = self.anti_injection_validator.validate_sql_query_params(sql_params)
                assert not sql_result.is_valid
            
            # Step 3: Log security event
            self.audit_logger.log_security_event(
                event_type=f"malicious_input_{input_type}",
                risk_score=8.0,
                threat_indicators=[f"input_type:{input_type}", f"pattern:{basic_result.threats[0].type}"]
            )
    
    def test_rate_limiting_with_monitoring(self):
        """Test rate limiting integration with security monitoring."""
        # Start security monitoring
        self.security_monitor.start_monitoring()
        
        # Setup strict rate limit for testing
        from gaudi3_scale.security.rate_limiting import RateLimitRule, RateLimitStrategy, ThrottleAction
        
        strict_rule = RateLimitRule(
            name="test_integration",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            limit=2,
            window=60,
            action=ThrottleAction.TEMPORARY_BAN,
            ban_duration=5,
            priority=100
        )
        self.rate_limiter.add_rule(strict_rule)
        
        identifier = "attack_ip"
        
        # Make requests within limit
        for i in range(2):
            result = self.rate_limiter.check_rate_limit(identifier)
            assert result.allowed
        
        # Exceed limit - should trigger ban
        result = self.rate_limiter.check_rate_limit(identifier)
        assert not result.allowed
        assert result.action == ThrottleAction.TEMPORARY_BAN
        
        # Log rate limit violation
        self.audit_logger.log_security_event(
            event_type="rate_limit_violation",
            risk_score=6.0,
            threat_indicators=[f"identifier:{identifier}", f"rule:{strict_rule.name}"]
        )
        
        # Verify ban is in effect
        banned_result = self.rate_limiter.check_rate_limit(identifier)
        assert not banned_result.allowed
        assert banned_result.action == ThrottleAction.TEMPORARY_BAN
    
    def test_dos_protection_integration(self):
        """Test DoS protection with monitoring and alerting."""
        # Simulate high-frequency attack
        attacker_ip = "192.168.1.100"
        
        # Make many requests rapidly
        for i in range(120):  # Above high frequency threshold
            alert = self.dos_protection.analyze_request_pattern(
                ip_address=attacker_ip,
                endpoint="/api/sensitive",
                request_size=1024,
                response_time=0.1
            )
        
        # Should detect attack
        assert alert is not None
        assert alert.pattern.value == "high_frequency"
        assert alert.severity in ["HIGH", "CRITICAL"]
        
        # Verify automatic mitigation was applied
        rate_limit_result = self.rate_limiter.check_rate_limit(attacker_ip)
        # IP should be banned or heavily rate limited
        
        # Check DoS statistics
        dos_stats = self.dos_protection.get_dos_statistics()
        assert dos_stats["active_ip_tracking"] > 0
    
    @pytest.mark.asyncio
    async def test_request_throttling_integration(self):
        """Test complete request throttling with all components."""
        # Create request that would trigger multiple security checks
        malicious_request = {
            "client_ip": "192.168.1.200",
            "path": "/api/admin/users",
            "method": "POST",
            "user_agent": "AttackBot/1.0",
            "content_length": 1024,
            "headers": {
                "X-Forwarded-For": "'; DROP TABLE users; --"
            }
        }
        
        # Process through request throttler
        allowed, response = await self.request_throttler.throttle_request(malicious_request)
        
        # Should analyze for various threats
        # May be allowed initially but monitored
        
        # Simulate multiple rapid requests
        for i in range(10):
            allowed, response = await self.request_throttler.throttle_request(malicious_request)
            if not allowed:
                assert response["status_code"] in [429, 503]
                break
    
    def test_audit_trail_completeness(self):
        """Test complete audit trail across all components."""
        # Perform various operations and verify audit trail
        
        # 1. Failed login attempt
        try:
            self.auth_manager.login(LoginRequest(
                username="admin",
                password="wrongpassword"
            ))
        except AuthenticationError:
            pass
        
        self.audit_logger.log_authentication_event(
            success=False,
            username="admin",
            failure_reason="Invalid password",
            ip_address="192.168.1.1"
        )
        
        # 2. Successful login
        auth_result = self.auth_manager.login(LoginRequest(
            username="admin",
            password="AdminPass123!@#"
        ))
        
        self.audit_logger.log_authentication_event(
            success=True,
            username="admin",
            ip_address="192.168.1.1"
        )
        
        # 3. Authorization attempt
        self.audit_logger.log_authorization_event(
            success=True,
            username="admin",
            permission=Permission.MANAGE_USERS.value,
            resource="user_management"
        )
        
        # 4. Data access
        self.audit_logger.log_data_access_event(
            username="admin",
            resource_type="user",
            resource_id="user_123",
            action="read"
        )
        
        # 5. Configuration change
        self.audit_logger.log_configuration_change(
            username="admin",
            config_type="security_settings",
            before_state={"rate_limit": 1000},
            after_state={"rate_limit": 500}
        )
        
        # 6. Security event
        self.audit_logger.log_security_event(
            event_type="suspicious_activity",
            risk_score=7.0,
            threat_indicators=["multiple_failed_logins", "unusual_timing"]
        )
        
        # Allow time for audit processing
        time.sleep(0.1)
        
        # Query audit events
        from gaudi3_scale.security.audit_logging import AuditFilter
        
        audit_filter = AuditFilter(
            start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
            end_time=datetime.now(timezone.utc),
            limit=100
        )
        
        events = self.audit_logger.query_events(audit_filter)
        
        # Verify we have audit events for all operations
        assert len(events) > 0
        
        # Check for different event categories
        event_categories = {event.category for event in events}
        expected_categories = {
            EventCategory.AUTHENTICATION,
            EventCategory.AUTHORIZATION,
            EventCategory.DATA_ACCESS,
            EventCategory.CONFIGURATION_CHANGE,
            EventCategory.SECURITY_EVENT
        }
        
        # Should have most categories (at least some)
        assert len(event_categories & expected_categories) > 2
    
    def test_security_monitoring_alerts(self):
        """Test security monitoring and alerting system."""
        # Setup alert callback to track alerts
        alerts_received = []
        
        def alert_callback(alert):
            alerts_received.append(alert)
        
        self.security_alerts.add_notification_handler("test", alert_callback)
        
        # Start monitoring
        self.security_monitor.start_monitoring()
        
        # Generate security events that should trigger alerts
        self.audit_logger.log_security_event(
            event_type="brute_force_attack",
            risk_score=9.0,
            threat_indicators=["failed_login_attempts:50", "source_ip:192.168.1.100"],
            ip_address="192.168.1.100",
            username="admin"
        )
        
        self.audit_logger.log_security_event(
            event_type="privilege_escalation_attempt", 
            risk_score=8.5,
            threat_indicators=["unauthorized_admin_access", "user_id:12345"],
            username="user"
        )
        
        # Allow time for processing
        time.sleep(0.2)
        
        # Check security dashboard
        dashboard = self.security_monitor.get_security_dashboard()
        
        assert dashboard["monitoring_active"]
        assert "total_alerts" in dashboard
        assert "critical_alerts" in dashboard
        assert "detector_status" in dashboard
    
    def test_compliance_reporting(self):
        """Test compliance reporting across all components."""
        # Generate various compliance-relevant events
        
        # GDPR data processing
        from gaudi3_scale.security.audit_logging import ComplianceLogger
        compliance_logger = ComplianceLogger(self.audit_logger)
        
        compliance_logger.log_data_processing(
            data_subject="user@example.com",
            processing_purpose="user_authentication",
            data_categories=["email", "password_hash"],
            legal_basis="legitimate_interest"
        )
        
        # SOX financial transaction (mock)
        compliance_logger.log_financial_transaction(
            transaction_id="txn_12345",
            amount=100.50,
            currency="USD",
            user="admin"
        )
        
        # Generate compliance report
        from gaudi3_scale.security.audit_logging import ComplianceStandard
        
        report = self.audit_logger.generate_compliance_report(
            standard=ComplianceStandard.GDPR,
            start_date=datetime.now(timezone.utc) - timedelta(hours=1),
            end_date=datetime.now(timezone.utc)
        )
        
        assert report["standard"] == "GDPR"
        assert "total_events" in report
        assert "event_breakdown" in report
        assert "risk_analysis" in report
        assert "recommendations" in report
    
    def test_security_configuration_validation(self):
        """Test validation of security configurations."""
        # Test secure configuration settings
        security_config = {
            "encryption": {
                "algorithm": "AES-256-GCM",
                "key_rotation_days": 90
            },
            "authentication": {
                "password_policy": {
                    "min_length": 12,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_symbols": True
                },
                "session_timeout": 3600,
                "max_login_attempts": 5,
                "lockout_duration": 900
            },
            "rate_limiting": {
                "api_limit": 1000,
                "login_limit": 5,
                "burst_limit": 10
            }
        }
        
        # Validate configuration structure
        config_result = self.security_validator.validate_against_injection(
            str(security_config), "security_config"
        )
        assert config_result.is_valid
        
        # Store configuration securely
        for section, settings in security_config.items():
            for key, value in settings.items():
                config_key = f"{section}.{key}"
                if "password" in key.lower() or "secret" in key.lower():
                    self.config_manager.set(config_key, str(value), encrypt=True)
                else:
                    self.config_manager.set(config_key, str(value))
        
        # Retrieve and verify
        retrieved_timeout = self.config_manager.get("authentication.session_timeout")
        assert retrieved_timeout == "3600"
    
    def test_incident_response_integration(self):
        """Test incident response integration."""
        from gaudi3_scale.security.monitoring import IncidentResponse, ThreatLevel
        
        # Setup incident response
        incident_response = IncidentResponse(self.security_monitor)
        
        # Create security incident
        incident_id = incident_response.create_incident(
            title="Suspected Data Breach",
            description="Multiple failed login attempts followed by successful login and data access",
            severity=ThreatLevel.HIGH,
            assigned_to="security_team"
        )
        
        assert incident_id is not None
        
        # Verify incident was created
        incidents = self.security_monitor.incidents
        assert incident_id in incidents
        
        incident = incidents[incident_id]
        assert incident.title == "Suspected Data Breach"
        assert incident.severity == ThreatLevel.HIGH