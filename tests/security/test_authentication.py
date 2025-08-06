"""Security tests for authentication and authorization system."""

import pytest
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from gaudi3_scale.security.authentication import (
    UserManager, AuthenticationManager, AuthorizationManager,
    PasswordManager, MFAManager, User, UserRole, Permission,
    LoginRequest, ChangePasswordRequest
)
from gaudi3_scale.security.audit_logging import SecurityAuditLogger
from gaudi3_scale.integrations.auth.jwt import JWTConfig
from gaudi3_scale.exceptions import AuthenticationError, AuthorizationError


class TestPasswordManager:
    """Test password management functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.password_manager = PasswordManager()
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "Test123!@#"
        
        # Hash password
        password_hash = self.password_manager.hash_password(password)
        
        # Verify correct password
        assert self.password_manager.verify_password(password, password_hash)
        
        # Verify incorrect password
        assert not self.password_manager.verify_password("wrong", password_hash)
    
    def test_password_hash_uniqueness(self):
        """Test that same password produces different hashes."""
        password = "Test123!@#"
        
        hash1 = self.password_manager.hash_password(password)
        hash2 = self.password_manager.hash_password(password)
        
        # Hashes should be different due to salt
        assert hash1 != hash2
        
        # But both should verify correctly
        assert self.password_manager.verify_password(password, hash1)
        assert self.password_manager.verify_password(password, hash2)
    
    def test_weak_password_handling(self):
        """Test handling of weak passwords."""
        # Password manager should hash any password - validation is separate
        weak_password = "123"
        password_hash = self.password_manager.hash_password(weak_password)
        
        assert self.password_manager.verify_password(weak_password, password_hash)


class TestUserManager:
    """Test user management functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.user_manager = UserManager(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_user(self):
        """Test user creation."""
        user = self.user_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="Test123!@#",
            roles={UserRole.VIEWER}
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert UserRole.VIEWER in user.roles
        assert user.is_active
        assert not user.is_verified
        assert user.user_id in self.user_manager.users
    
    def test_duplicate_username(self):
        """Test duplicate username handling."""
        self.user_manager.create_user("testuser", "test1@example.com", "pass123")
        
        with pytest.raises(AuthenticationError, match="Username already exists"):
            self.user_manager.create_user("testuser", "test2@example.com", "pass456")
    
    def test_duplicate_email(self):
        """Test duplicate email handling."""
        self.user_manager.create_user("user1", "test@example.com", "pass123")
        
        with pytest.raises(AuthenticationError, match="Email already exists"):
            self.user_manager.create_user("user2", "test@example.com", "pass456")
    
    def test_get_user_by_username(self):
        """Test getting user by username."""
        user = self.user_manager.create_user("testuser", "test@example.com", "pass123")
        
        retrieved_user = self.user_manager.get_user_by_username("testuser")
        assert retrieved_user is not None
        assert retrieved_user.user_id == user.user_id
        
        # Test non-existent user
        assert self.user_manager.get_user_by_username("nonexistent") is None
    
    def test_update_user(self):
        """Test user updates."""
        user = self.user_manager.create_user("testuser", "test@example.com", "pass123")
        
        # Update user
        success = self.user_manager.update_user(
            user.user_id,
            is_verified=True,
            roles={UserRole.ADMIN}
        )
        
        assert success
        
        updated_user = self.user_manager.get_user_by_id(user.user_id)
        assert updated_user.is_verified
        assert UserRole.ADMIN in updated_user.roles
    
    def test_delete_user(self):
        """Test user deletion."""
        user = self.user_manager.create_user("testuser", "test@example.com", "pass123")
        
        success = self.user_manager.delete_user(user.user_id)
        assert success
        
        assert self.user_manager.get_user_by_id(user.user_id) is None


class TestAuthenticationManager:
    """Test authentication functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.user_manager = UserManager(storage_path=self.temp_dir)
        
        # Create test JWT configuration
        jwt_config = JWTConfig(
            secret_key=secrets.token_urlsafe(64),
            algorithm="HS256",
            access_token_expire_minutes=30
        )
        
        self.auth_manager = AuthenticationManager(
            user_manager=self.user_manager,
            jwt_config=jwt_config,
            max_login_attempts=3,
            lockout_duration=300
        )
        
        # Create test user
        self.test_user = self.user_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="Test123!@#",
            roles={UserRole.VIEWER}
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_successful_login(self):
        """Test successful login."""
        login_request = LoginRequest(
            username="testuser",
            password="Test123!@#"
        )
        
        result = self.auth_manager.login(login_request)
        
        assert "access_token" in result
        assert "refresh_token" in result
        assert result["token_type"] == "bearer"
        assert "user" in result
        assert result["user"]["username"] == "testuser"
    
    def test_invalid_username(self):
        """Test login with invalid username."""
        login_request = LoginRequest(
            username="nonexistent",
            password="Test123!@#"
        )
        
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            self.auth_manager.login(login_request)
    
    def test_invalid_password(self):
        """Test login with invalid password."""
        login_request = LoginRequest(
            username="testuser",
            password="wrongpassword"
        )
        
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            self.auth_manager.login(login_request)
    
    def test_account_lockout(self):
        """Test account lockout after multiple failed attempts."""
        login_request = LoginRequest(
            username="testuser",
            password="wrongpassword"
        )
        
        # Make multiple failed attempts
        for i in range(3):
            with pytest.raises(AuthenticationError, match="Invalid credentials"):
                self.auth_manager.login(login_request)
        
        # Next attempt should be blocked due to lockout
        with pytest.raises(AuthenticationError, match="Account is locked"):
            self.auth_manager.login(login_request)
    
    def test_inactive_user(self):
        """Test login with inactive user."""
        # Deactivate user
        self.user_manager.update_user(self.test_user.user_id, is_active=False)
        
        login_request = LoginRequest(
            username="testuser",
            password="Test123!@#"
        )
        
        with pytest.raises(AuthenticationError, match="Account is inactive"):
            self.auth_manager.login(login_request)
    
    def test_token_refresh(self):
        """Test token refresh functionality."""
        # First login
        login_request = LoginRequest(
            username="testuser",
            password="Test123!@#"
        )
        
        login_result = self.auth_manager.login(login_request)
        refresh_token = login_result["refresh_token"]
        
        # Refresh token
        refresh_result = self.auth_manager.refresh_token(refresh_token)
        
        assert "access_token" in refresh_result
        assert refresh_result["token_type"] == "bearer"
        assert refresh_result["expires_in"] == 3600
    
    def test_password_change(self):
        """Test password change functionality."""
        old_password = "Test123!@#"
        new_password = "NewTest456!@#"
        
        success = self.auth_manager.change_password(
            self.test_user.user_id,
            old_password,
            new_password
        )
        
        assert success
        
        # Test login with new password
        login_request = LoginRequest(
            username="testuser",
            password=new_password
        )
        
        result = self.auth_manager.login(login_request)
        assert "access_token" in result
        
        # Test that old password no longer works
        old_login_request = LoginRequest(
            username="testuser",
            password=old_password
        )
        
        with pytest.raises(AuthenticationError):
            self.auth_manager.login(old_login_request)
    
    def test_invalid_current_password_change(self):
        """Test password change with invalid current password."""
        with pytest.raises(AuthenticationError, match="Invalid current password"):
            self.auth_manager.change_password(
                self.test_user.user_id,
                "wrongpassword",
                "NewTest456!@#"
            )


class TestAuthorizationManager:
    """Test authorization functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.user_manager = UserManager(storage_path=self.temp_dir)
        self.authz_manager = AuthorizationManager(user_manager=self.user_manager)
        
        # Create test users with different roles
        self.admin_user = self.user_manager.create_user(
            username="admin",
            email="admin@example.com",
            password="Admin123!@#",
            roles={UserRole.ADMIN}
        )
        
        self.viewer_user = self.user_manager.create_user(
            username="viewer",
            email="viewer@example.com",
            password="Viewer123!@#",
            roles={UserRole.VIEWER}
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_check_permission_with_role(self):
        """Test permission checking based on role."""
        # Admin should have cluster management permissions
        assert self.authz_manager.check_permission(
            self.admin_user.user_id, 
            Permission.CREATE_CLUSTER
        )
        
        # Viewer should not have cluster creation permissions
        assert not self.authz_manager.check_permission(
            self.viewer_user.user_id,
            Permission.CREATE_CLUSTER
        )
        
        # Viewer should have read permissions
        assert self.authz_manager.check_permission(
            self.viewer_user.user_id,
            Permission.READ_CLUSTER
        )
    
    def test_require_permission_success(self):
        """Test successful permission requirement."""
        # Should not raise exception
        self.authz_manager.require_permission(
            self.admin_user.user_id,
            Permission.CREATE_CLUSTER
        )
    
    def test_require_permission_failure(self):
        """Test failed permission requirement."""
        with pytest.raises(AuthorizationError, match="Permission denied"):
            self.authz_manager.require_permission(
                self.viewer_user.user_id,
                Permission.CREATE_CLUSTER
            )
    
    def test_check_role(self):
        """Test role checking."""
        assert self.authz_manager.check_role(self.admin_user.user_id, UserRole.ADMIN)
        assert not self.authz_manager.check_role(self.viewer_user.user_id, UserRole.ADMIN)
        assert self.authz_manager.check_role(self.viewer_user.user_id, UserRole.VIEWER)
    
    def test_require_role_success(self):
        """Test successful role requirement."""
        self.authz_manager.require_role(self.admin_user.user_id, UserRole.ADMIN)
    
    def test_require_role_failure(self):
        """Test failed role requirement."""
        with pytest.raises(AuthorizationError, match="Role required"):
            self.authz_manager.require_role(self.viewer_user.user_id, UserRole.ADMIN)
    
    def test_grant_permission(self):
        """Test granting individual permissions."""
        # Grant specific permission to viewer
        success = self.authz_manager.grant_permission(
            self.viewer_user.user_id,
            Permission.CREATE_CLUSTER
        )
        
        assert success
        
        # Verify permission was granted
        assert self.authz_manager.check_permission(
            self.viewer_user.user_id,
            Permission.CREATE_CLUSTER
        )
    
    def test_revoke_permission(self):
        """Test revoking individual permissions."""
        # First grant permission
        self.authz_manager.grant_permission(
            self.viewer_user.user_id,
            Permission.CREATE_CLUSTER
        )
        
        # Then revoke it
        success = self.authz_manager.revoke_permission(
            self.viewer_user.user_id,
            Permission.CREATE_CLUSTER
        )
        
        assert success
        
        # Verify permission was revoked
        assert not self.authz_manager.check_permission(
            self.viewer_user.user_id,
            Permission.CREATE_CLUSTER
        )
    
    def test_assign_role(self):
        """Test role assignment."""
        success = self.authz_manager.assign_role(
            self.viewer_user.user_id,
            UserRole.OPERATOR
        )
        
        assert success
        
        # Verify role was assigned
        user = self.user_manager.get_user_by_id(self.viewer_user.user_id)
        assert UserRole.OPERATOR in user.roles
        assert UserRole.VIEWER in user.roles  # Should keep existing role
    
    def test_remove_role(self):
        """Test role removal."""
        # First assign additional role
        self.authz_manager.assign_role(self.viewer_user.user_id, UserRole.OPERATOR)
        
        # Then remove it
        success = self.authz_manager.remove_role(
            self.viewer_user.user_id,
            UserRole.OPERATOR
        )
        
        assert success
        
        # Verify role was removed
        user = self.user_manager.get_user_by_id(self.viewer_user.user_id)
        assert UserRole.OPERATOR not in user.roles
        assert UserRole.VIEWER in user.roles  # Should keep original role
    
    def test_inactive_user_permissions(self):
        """Test that inactive users have no permissions."""
        # Deactivate user
        self.user_manager.update_user(self.admin_user.user_id, is_active=False)
        
        # Check that permissions are denied
        assert not self.authz_manager.check_permission(
            self.admin_user.user_id,
            Permission.CREATE_CLUSTER
        )
        
        assert not self.authz_manager.check_role(
            self.admin_user.user_id,
            UserRole.ADMIN
        )


class TestMFAManager:
    """Test MFA functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.mfa_manager = MFAManager()
    
    def test_generate_secret(self):
        """Test MFA secret generation."""
        try:
            secret = self.mfa_manager.generate_secret()
            assert len(secret) == 32  # Base32 secret should be 32 characters
            assert secret.isalnum()  # Should only contain alphanumeric characters
        except ImportError:
            # Skip test if pyotp not available
            pytest.skip("pyotp not available for MFA testing")
    
    def test_verify_totp(self):
        """Test TOTP verification."""
        try:
            import pyotp
            
            secret = self.mfa_manager.generate_secret()
            totp = pyotp.TOTP(secret)
            
            # Generate current code
            current_code = totp.now()
            
            # Verify current code
            assert self.mfa_manager.verify_totp(secret, current_code)
            
            # Verify invalid code
            assert not self.mfa_manager.verify_totp(secret, "000000")
            
        except ImportError:
            # Skip test if pyotp not available
            pytest.skip("pyotp not available for MFA testing")


class TestSecurityValidation:
    """Test security validation and edge cases."""
    
    def test_password_validation(self):
        """Test password strength validation."""
        change_request = ChangePasswordRequest(
            current_password="old123",
            new_password="weak"
        )
        
        # Should raise validation error for weak password
        with pytest.raises(ValueError):
            change_request.new_password = "weak"
    
    def test_sql_injection_in_username(self):
        """Test SQL injection protection in username."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            user_manager = UserManager(storage_path=temp_dir)
            
            # Try to create user with SQL injection in username
            malicious_username = "admin'; DROP TABLE users; --"
            
            # Should not cause any issues (stored as-is, not used in SQL)
            user = user_manager.create_user(
                username=malicious_username,
                email="test@example.com",
                password="Test123!@#"
            )
            
            # Verify user was created with the exact username
            assert user.username == malicious_username
            
            # Verify we can retrieve the user
            retrieved = user_manager.get_user_by_username(malicious_username)
            assert retrieved is not None
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            user_manager = UserManager(storage_path=temp_dir)
            auth_manager = AuthenticationManager(user_manager=user_manager)
            
            # Create a user
            user_manager.create_user("testuser", "test@example.com", "Test123!@#")
            
            # Time invalid username
            start_time = time.time()
            try:
                auth_manager.login(LoginRequest(
                    username="nonexistent",
                    password="Test123!@#"
                ))
            except AuthenticationError:
                pass
            invalid_username_time = time.time() - start_time
            
            # Time invalid password
            start_time = time.time()
            try:
                auth_manager.login(LoginRequest(
                    username="testuser",
                    password="wrongpassword"
                ))
            except AuthenticationError:
                pass
            invalid_password_time = time.time() - start_time
            
            # Times should be similar (within reasonable bounds)
            time_diff = abs(invalid_username_time - invalid_password_time)
            assert time_diff < 0.1  # Should be within 100ms
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_session_fixation_protection(self):
        """Test protection against session fixation attacks."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            user_manager = UserManager(storage_path=temp_dir)
            auth_manager = AuthenticationManager(user_manager=user_manager)
            
            # Create user
            user_manager.create_user("testuser", "test@example.com", "Test123!@#")
            
            # Login twice and verify different tokens
            result1 = auth_manager.login(LoginRequest(
                username="testuser",
                password="Test123!@#"
            ))
            
            result2 = auth_manager.login(LoginRequest(
                username="testuser", 
                password="Test123!@#"
            ))
            
            # Tokens should be different
            assert result1["access_token"] != result2["access_token"]
            assert result1["refresh_token"] != result2["refresh_token"]
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)