"""Enterprise-grade authentication and authorization system.

This module provides comprehensive authentication, authorization, role-based
access control (RBAC), and session management for secure API access.
"""

import os
import json
import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Set, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    import pyotp
    import qrcode
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False

from pydantic import BaseModel, Field, validator
from ..integrations.auth.jwt import JWTHandler, JWTConfig, JWTPayload
from ..database.connection import get_redis
from ..exceptions import AuthenticationError, AuthorizationError
from ..logging_utils import get_logger
from .config_security import SecretsManager

logger = get_logger(__name__)


class UserRole(Enum):
    """User roles for RBAC."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API_USER = "api_user"


class Permission(Enum):
    """System permissions."""
    # Cluster management
    CREATE_CLUSTER = "cluster:create"
    READ_CLUSTER = "cluster:read"
    UPDATE_CLUSTER = "cluster:update"
    DELETE_CLUSTER = "cluster:delete"
    
    # Training management
    CREATE_TRAINING = "training:create"
    READ_TRAINING = "training:read"
    UPDATE_TRAINING = "training:update"
    DELETE_TRAINING = "training:delete"
    CONTROL_TRAINING = "training:control"
    
    # Model management
    CREATE_MODEL = "model:create"
    READ_MODEL = "model:read"
    UPDATE_MODEL = "model:update"
    DELETE_MODEL = "model:delete"
    DEPLOY_MODEL = "model:deploy"
    
    # System administration
    MANAGE_USERS = "system:manage_users"
    MANAGE_ROLES = "system:manage_roles"
    VIEW_AUDIT_LOGS = "system:view_audit_logs"
    MANAGE_SYSTEM = "system:manage"
    
    # API access
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"


@dataclass
class User:
    """User model with authentication and authorization data."""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: Set[UserRole] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    api_keys: Dict[str, str] = field(default_factory=dict)
    session_tokens: Set[str] = field(default_factory=set)
    password_reset_token: Optional[str] = None
    password_reset_expires: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (excluding sensitive data)."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": [role.value for role in self.roles],
            "permissions": [perm.value for perm in self.permissions],
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "mfa_enabled": self.mfa_enabled,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None
        }
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or any(
            permission in RoleManager.get_role_permissions(role) 
            for role in self.roles
        )
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role."""
        return role in self.roles


class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    mfa_code: Optional[str] = Field(None, regex=r'^\d{6}$')
    remember_me: bool = False


class ChangePasswordRequest(BaseModel):
    """Change password request model."""
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def validate_password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError('Password must contain special character')
        return v


class SessionData(BaseModel):
    """Session data model."""
    session_id: str
    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at


class RoleManager:
    """Manages role-based access control."""
    
    # Default role permissions mapping
    ROLE_PERMISSIONS = {
        UserRole.SUPER_ADMIN: {
            Permission.CREATE_CLUSTER, Permission.READ_CLUSTER, 
            Permission.UPDATE_CLUSTER, Permission.DELETE_CLUSTER,
            Permission.CREATE_TRAINING, Permission.READ_TRAINING,
            Permission.UPDATE_TRAINING, Permission.DELETE_TRAINING, 
            Permission.CONTROL_TRAINING,
            Permission.CREATE_MODEL, Permission.READ_MODEL,
            Permission.UPDATE_MODEL, Permission.DELETE_MODEL, Permission.DEPLOY_MODEL,
            Permission.MANAGE_USERS, Permission.MANAGE_ROLES,
            Permission.VIEW_AUDIT_LOGS, Permission.MANAGE_SYSTEM,
            Permission.API_READ, Permission.API_WRITE, Permission.API_ADMIN
        },
        UserRole.ADMIN: {
            Permission.CREATE_CLUSTER, Permission.READ_CLUSTER,
            Permission.UPDATE_CLUSTER, Permission.DELETE_CLUSTER,
            Permission.CREATE_TRAINING, Permission.READ_TRAINING,
            Permission.UPDATE_TRAINING, Permission.DELETE_TRAINING,
            Permission.CONTROL_TRAINING,
            Permission.CREATE_MODEL, Permission.READ_MODEL,
            Permission.UPDATE_MODEL, Permission.DELETE_MODEL, Permission.DEPLOY_MODEL,
            Permission.MANAGE_USERS, Permission.VIEW_AUDIT_LOGS,
            Permission.API_READ, Permission.API_WRITE
        },
        UserRole.OPERATOR: {
            Permission.READ_CLUSTER, Permission.UPDATE_CLUSTER,
            Permission.CREATE_TRAINING, Permission.READ_TRAINING,
            Permission.UPDATE_TRAINING, Permission.CONTROL_TRAINING,
            Permission.READ_MODEL, Permission.UPDATE_MODEL, Permission.DEPLOY_MODEL,
            Permission.API_READ, Permission.API_WRITE
        },
        UserRole.VIEWER: {
            Permission.READ_CLUSTER, Permission.READ_TRAINING,
            Permission.READ_MODEL, Permission.API_READ
        },
        UserRole.API_USER: {
            Permission.API_READ, Permission.API_WRITE
        }
    }
    
    @classmethod
    def get_role_permissions(cls, role: UserRole) -> Set[Permission]:
        """Get permissions for a role."""
        return cls.ROLE_PERMISSIONS.get(role, set())
    
    @classmethod
    def check_permission(cls, user_roles: Set[UserRole], 
                        user_permissions: Set[Permission],
                        required_permission: Permission) -> bool:
        """Check if user has required permission."""
        # Direct permission check
        if required_permission in user_permissions:
            return True
        
        # Role-based permission check
        for role in user_roles:
            if required_permission in cls.get_role_permissions(role):
                return True
        
        return False


class PasswordManager:
    """Manages password hashing and validation."""
    
    def __init__(self):
        """Initialize password manager."""
        self.logger = logger.getChild(self.__class__.__name__)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        if not BCRYPT_AVAILABLE:
            # Fallback to pbkdf2 if bcrypt not available
            return self._pbkdf2_hash(password)
        
        try:
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
            return password_hash.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Password hashing failed: {e}")
            return self._pbkdf2_hash(password)
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        if not BCRYPT_AVAILABLE or password_hash.startswith('pbkdf2:'):
            return self._verify_pbkdf2(password, password_hash)
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception as e:
            self.logger.error(f"Password verification failed: {e}")
            return False
    
    def _pbkdf2_hash(self, password: str) -> str:
        """Fallback PBKDF2 hash implementation."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), 
                                          salt.encode('utf-8'), 100000)
        return f"pbkdf2:sha256:100000:{salt}:{password_hash.hex()}"
    
    def _verify_pbkdf2(self, password: str, password_hash: str) -> bool:
        """Verify PBKDF2 hash."""
        try:
            parts = password_hash.split(':')
            if len(parts) != 5 or parts[0] != 'pbkdf2':
                return False
            
            algorithm, iterations, salt, stored_hash = parts[1], int(parts[2]), parts[3], parts[4]
            
            computed_hash = hashlib.pbkdf2_hmac(algorithm, password.encode('utf-8'),
                                              salt.encode('utf-8'), iterations)
            return computed_hash.hex() == stored_hash
        except Exception:
            return False


class MFAManager:
    """Manages multi-factor authentication."""
    
    def __init__(self):
        """Initialize MFA manager."""
        self.logger = logger.getChild(self.__class__.__name__)
    
    def generate_secret(self) -> str:
        """Generate MFA secret key."""
        if not TOTP_AVAILABLE:
            raise ImportError("pyotp required for MFA functionality")
        
        return pyotp.random_base32()
    
    def generate_qr_code(self, secret: str, username: str, issuer: str = "Gaudi3Scale") -> bytes:
        """Generate QR code for MFA setup."""
        if not TOTP_AVAILABLE:
            raise ImportError("pyotp and qrcode required for MFA QR codes")
        
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(username, issuer_name=issuer)
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL image to bytes
        import io
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    
    def verify_totp(self, secret: str, code: str) -> bool:
        """Verify TOTP code."""
        if not TOTP_AVAILABLE:
            return True  # Fallback: skip MFA if library not available
        
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)  # Allow 30s window
        except Exception as e:
            self.logger.error(f"TOTP verification failed: {e}")
            return False


class UserManager:
    """Manages user storage and operations."""
    
    def __init__(self, storage_path: Optional[Path] = None,
                 secrets_manager: Optional[SecretsManager] = None):
        """Initialize user manager.
        
        Args:
            storage_path: Path to store user data
            secrets_manager: Secrets manager for sensitive data
        """
        self.storage_path = storage_path or Path.home() / ".gaudi3_scale" / "users"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.users_file = self.storage_path / "users.json"
        self.secrets_manager = secrets_manager or SecretsManager()
        self.password_manager = PasswordManager()
        
        self.users: Dict[str, User] = self._load_users()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def _load_users(self) -> Dict[str, User]:
        """Load users from storage."""
        if not self.users_file.exists():
            return {}
        
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)
            
            users = {}
            for user_id, user_data in data.items():
                # Convert string dates back to datetime
                for date_field in ['created_at', 'updated_at', 'last_login', 
                                 'locked_until', 'password_reset_expires']:
                    if user_data.get(date_field):
                        user_data[date_field] = datetime.fromisoformat(user_data[date_field])
                
                # Convert role and permission strings back to enums
                user_data['roles'] = {UserRole(role) for role in user_data.get('roles', [])}
                user_data['permissions'] = {Permission(perm) for perm in user_data.get('permissions', [])}
                user_data['session_tokens'] = set(user_data.get('session_tokens', []))
                
                users[user_id] = User(**user_data)
            
            return users
        except Exception as e:
            self.logger.error(f"Failed to load users: {e}")
            return {}
    
    def _save_users(self):
        """Save users to storage."""
        try:
            # Convert users to serializable format
            data = {}
            for user_id, user in self.users.items():
                user_data = {
                    'user_id': user.user_id,
                    'username': user.username,
                    'email': user.email,
                    'password_hash': user.password_hash,
                    'roles': [role.value for role in user.roles],
                    'permissions': [perm.value for perm in user.permissions],
                    'is_active': user.is_active,
                    'is_verified': user.is_verified,
                    'created_at': user.created_at.isoformat(),
                    'updated_at': user.updated_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'login_attempts': user.login_attempts,
                    'locked_until': user.locked_until.isoformat() if user.locked_until else None,
                    'mfa_enabled': user.mfa_enabled,
                    'mfa_secret': user.mfa_secret,
                    'api_keys': user.api_keys,
                    'session_tokens': list(user.session_tokens),
                    'password_reset_token': user.password_reset_token,
                    'password_reset_expires': user.password_reset_expires.isoformat() 
                                            if user.password_reset_expires else None
                }
                data[user_id] = user_data
            
            # Save with secure permissions
            self.users_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.chmod(self.users_file, 0o600)
            
        except Exception as e:
            self.logger.error(f"Failed to save users: {e}")
            raise AuthenticationError(f"Failed to save user data: {e}")
    
    def create_user(self, username: str, email: str, password: str,
                   roles: Optional[Set[UserRole]] = None,
                   permissions: Optional[Set[Permission]] = None) -> User:
        """Create new user."""
        # Check if username already exists
        if any(user.username == username for user in self.users.values()):
            raise AuthenticationError("Username already exists")
        
        # Check if email already exists
        if any(user.email == email for user in self.users.values()):
            raise AuthenticationError("Email already exists")
        
        # Generate user ID
        user_id = secrets.token_urlsafe(16)
        
        # Hash password
        password_hash = self.password_manager.hash_password(password)
        
        # Create user
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or {UserRole.VIEWER},
            permissions=permissions or set()
        )
        
        # Store user
        self.users[user_id] = user
        self._save_users()
        
        self.logger.info(f"User created: {username}")
        return user
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def update_user(self, user_id: str, **updates) -> bool:
        """Update user data."""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        user.updated_at = datetime.now(timezone.utc)
        self._save_users()
        
        self.logger.info(f"User updated: {user.username}")
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        if user_id not in self.users:
            return False
        
        username = self.users[user_id].username
        del self.users[user_id]
        self._save_users()
        
        self.logger.info(f"User deleted: {username}")
        return True


class AuthenticationManager:
    """Manages user authentication."""
    
    def __init__(self, 
                 user_manager: Optional[UserManager] = None,
                 jwt_config: Optional[JWTConfig] = None,
                 max_login_attempts: int = 5,
                 lockout_duration: int = 900):  # 15 minutes
        """Initialize authentication manager.
        
        Args:
            user_manager: User manager instance
            jwt_config: JWT configuration
            max_login_attempts: Maximum login attempts before lockout
            lockout_duration: Account lockout duration in seconds
        """
        self.user_manager = user_manager or UserManager()
        
        # Setup JWT configuration
        if jwt_config is None:
            jwt_config = JWTConfig(
                secret_key=secrets.token_urlsafe(64),
                algorithm="HS256",  # Use HS256 for simplicity, RS256 for production
                access_token_expire_minutes=60,
                refresh_token_expire_days=7
            )
        
        self.jwt_handler = JWTHandler(jwt_config)
        self.mfa_manager = MFAManager()
        self.max_login_attempts = max_login_attempts
        self.lockout_duration = lockout_duration
        
        self.logger = logger.getChild(self.__class__.__name__)
    
    def login(self, login_request: LoginRequest, 
             ip_address: Optional[str] = None) -> Dict[str, Any]:
        """Authenticate user and return tokens.
        
        Args:
            login_request: Login request data
            ip_address: Client IP address
            
        Returns:
            Dictionary with tokens and user info
        """
        user = self.user_manager.get_user_by_username(login_request.username)
        
        if not user:
            self.logger.warning(f"Login attempt with invalid username: {login_request.username}")
            raise AuthenticationError("Invalid credentials")
        
        # Check if account is locked
        if user.is_locked():
            self.logger.warning(f"Login attempt on locked account: {user.username}")
            raise AuthenticationError("Account is locked due to too many failed attempts")
        
        # Check if account is active
        if not user.is_active:
            self.logger.warning(f"Login attempt on inactive account: {user.username}")
            raise AuthenticationError("Account is inactive")
        
        # Verify password
        if not self.user_manager.password_manager.verify_password(
            login_request.password, user.password_hash):
            
            # Increment login attempts
            user.login_attempts += 1
            if user.login_attempts >= self.max_login_attempts:
                user.locked_until = datetime.now(timezone.utc) + timedelta(seconds=self.lockout_duration)
                self.logger.warning(f"Account locked due to failed attempts: {user.username}")
            
            self.user_manager.update_user(user.user_id, 
                                        login_attempts=user.login_attempts,
                                        locked_until=user.locked_until)
            
            raise AuthenticationError("Invalid credentials")
        
        # Verify MFA if enabled
        if user.mfa_enabled:
            if not login_request.mfa_code:
                raise AuthenticationError("MFA code required")
            
            if not self.mfa_manager.verify_totp(user.mfa_secret, login_request.mfa_code):
                raise AuthenticationError("Invalid MFA code")
        
        # Successful login - reset attempts and update last login
        user.login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now(timezone.utc)
        
        self.user_manager.update_user(user.user_id,
                                    login_attempts=0,
                                    locked_until=None,
                                    last_login=user.last_login)
        
        # Generate tokens
        token_data = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions]
        }
        
        expires_delta = None
        if login_request.remember_me:
            expires_delta = timedelta(days=30)  # Extended session
        
        access_token = self.jwt_handler.create_access_token(
            user.user_id, token_data, expires_delta=expires_delta
        )
        refresh_token = self.jwt_handler.create_refresh_token(user.user_id)
        
        self.logger.info(f"Successful login: {user.username} from {ip_address}")
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 3600 if not login_request.remember_me else 2592000,  # 1 hour or 30 days
            "user": user.to_dict()
        }
    
    def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token."""
        try:
            payload = self.jwt_handler.verify_token(refresh_token, "refresh")
            
            # Get updated user data
            user = self.user_manager.get_user_by_id(payload.sub)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            # Generate new access token with updated user data
            token_data = {
                "user_id": user.user_id,
                "username": user.username,
                "roles": [role.value for role in user.roles],
                "permissions": [perm.value for perm in user.permissions]
            }
            
            access_token = self.jwt_handler.create_access_token(
                user.user_id, token_data
            )
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": 3600
            }
            
        except Exception as e:
            self.logger.error(f"Token refresh failed: {e}")
            raise AuthenticationError("Invalid refresh token")
    
    def logout(self, access_token: str) -> bool:
        """Logout user and invalidate token."""
        try:
            payload = self.jwt_handler.verify_token(access_token, "access")
            
            # Add token to blacklist
            self.jwt_handler.blacklist_token(access_token)
            
            self.logger.info(f"User logged out: {payload.sub}")
            return True
            
        except Exception as e:
            self.logger.error(f"Logout failed: {e}")
            return False
    
    def change_password(self, user_id: str, current_password: str, 
                       new_password: str) -> bool:
        """Change user password."""
        user = self.user_manager.get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User not found")
        
        # Verify current password
        if not self.user_manager.password_manager.verify_password(
            current_password, user.password_hash):
            raise AuthenticationError("Invalid current password")
        
        # Hash new password
        new_password_hash = self.user_manager.password_manager.hash_password(new_password)
        
        # Update password
        self.user_manager.update_user(user_id, password_hash=new_password_hash)
        
        self.logger.info(f"Password changed: {user.username}")
        return True
    
    def setup_mfa(self, user_id: str) -> Dict[str, Any]:
        """Setup MFA for user."""
        user = self.user_manager.get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User not found")
        
        # Generate MFA secret
        secret = self.mfa_manager.generate_secret()
        qr_code = self.mfa_manager.generate_qr_code(secret, user.username)
        
        # Store secret temporarily (not yet enabled)
        user.mfa_secret = secret
        self.user_manager.update_user(user_id, mfa_secret=secret)
        
        return {
            "secret": secret,
            "qr_code": qr_code
        }
    
    def enable_mfa(self, user_id: str, verification_code: str) -> bool:
        """Enable MFA after verification."""
        user = self.user_manager.get_user_by_id(user_id)
        if not user or not user.mfa_secret:
            raise AuthenticationError("MFA setup not initiated")
        
        # Verify the code
        if not self.mfa_manager.verify_totp(user.mfa_secret, verification_code):
            raise AuthenticationError("Invalid verification code")
        
        # Enable MFA
        self.user_manager.update_user(user_id, mfa_enabled=True)
        
        self.logger.info(f"MFA enabled: {user.username}")
        return True
    
    def disable_mfa(self, user_id: str, password: str) -> bool:
        """Disable MFA (requires password confirmation)."""
        user = self.user_manager.get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User not found")
        
        # Verify password
        if not self.user_manager.password_manager.verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid password")
        
        # Disable MFA
        self.user_manager.update_user(user_id, 
                                    mfa_enabled=False, 
                                    mfa_secret=None)
        
        self.logger.info(f"MFA disabled: {user.username}")
        return True


class AuthorizationManager:
    """Manages user authorization and access control."""
    
    def __init__(self, user_manager: Optional[UserManager] = None):
        """Initialize authorization manager."""
        self.user_manager = user_manager or UserManager()
        self.role_manager = RoleManager()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user = self.user_manager.get_user_by_id(user_id)
        if not user or not user.is_active:
            return False
        
        return user.has_permission(permission)
    
    def require_permission(self, user_id: str, permission: Permission) -> None:
        """Require specific permission or raise AuthorizationError."""
        if not self.check_permission(user_id, permission):
            user = self.user_manager.get_user_by_id(user_id)
            username = user.username if user else "unknown"
            
            self.logger.warning(f"Access denied: {username} lacks {permission.value}")
            raise AuthorizationError(f"Permission denied: {permission.value}")
    
    def check_role(self, user_id: str, role: UserRole) -> bool:
        """Check if user has specific role."""
        user = self.user_manager.get_user_by_id(user_id)
        if not user or not user.is_active:
            return False
        
        return user.has_role(role)
    
    def require_role(self, user_id: str, role: UserRole) -> None:
        """Require specific role or raise AuthorizationError."""
        if not self.check_role(user_id, role):
            user = self.user_manager.get_user_by_id(user_id)
            username = user.username if user else "unknown"
            
            self.logger.warning(f"Access denied: {username} lacks role {role.value}")
            raise AuthorizationError(f"Role required: {role.value}")
    
    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant permission to user."""
        user = self.user_manager.get_user_by_id(user_id)
        if not user:
            return False
        
        user.permissions.add(permission)
        self.user_manager.update_user(user_id, permissions=user.permissions)
        
        self.logger.info(f"Permission granted: {permission.value} to {user.username}")
        return True
    
    def revoke_permission(self, user_id: str, permission: Permission) -> bool:
        """Revoke permission from user."""
        user = self.user_manager.get_user_by_id(user_id)
        if not user:
            return False
        
        user.permissions.discard(permission)
        self.user_manager.update_user(user_id, permissions=user.permissions)
        
        self.logger.info(f"Permission revoked: {permission.value} from {user.username}")
        return True
    
    def assign_role(self, user_id: str, role: UserRole) -> bool:
        """Assign role to user."""
        user = self.user_manager.get_user_by_id(user_id)
        if not user:
            return False
        
        user.roles.add(role)
        self.user_manager.update_user(user_id, roles=user.roles)
        
        self.logger.info(f"Role assigned: {role.value} to {user.username}")
        return True
    
    def remove_role(self, user_id: str, role: UserRole) -> bool:
        """Remove role from user."""
        user = self.user_manager.get_user_by_id(user_id)
        if not user:
            return False
        
        user.roles.discard(role)
        self.user_manager.update_user(user_id, roles=user.roles)
        
        self.logger.info(f"Role removed: {role.value} from {user.username}")
        return True


class TokenManager:
    """Manages API tokens and session tokens."""
    
    def __init__(self, redis_connection=None):
        """Initialize token manager."""
        self.redis = redis_connection or get_redis().get_client()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def store_session(self, session_data: SessionData) -> None:
        """Store session data in Redis."""
        try:
            self.redis.setex(
                f"session:{session_data.session_id}",
                int((session_data.expires_at - session_data.created_at).total_seconds()),
                session_data.json()
            )
        except Exception as e:
            self.logger.error(f"Failed to store session: {e}")
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data from Redis."""
        try:
            data = self.redis.get(f"session:{session_id}")
            if data:
                return SessionData.parse_raw(data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get session: {e}")
            return None
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session."""
        try:
            self.redis.delete(f"session:{session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to invalidate session: {e}")
            return False
    
    def blacklist_token(self, token: str, expires_in: int) -> None:
        """Add token to blacklist."""
        try:
            self.redis.setex(f"blacklist:{token}", expires_in, "1")
        except Exception as e:
            self.logger.error(f"Failed to blacklist token: {e}")
    
    def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        try:
            return self.redis.exists(f"blacklist:{token}")
        except Exception as e:
            self.logger.error(f"Failed to check token blacklist: {e}")
            return False