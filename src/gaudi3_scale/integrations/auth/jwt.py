"""JWT token handling for authentication."""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

try:
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
except ImportError:
    jwt = None
    rsa = None
    serialization = None

try:
    from pydantic import BaseModel, ConfigDict
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class ConfigDict:
        def __init__(self, **kwargs):
            pass

logger = logging.getLogger(__name__)


class JWTConfig(BaseModel):
    """JWT configuration settings."""
    model_config = ConfigDict(extra='forbid')
    
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    issuer: str = "gaudi3-scale"
    audience: str = "gaudi3-scale-api"


class JWTToken(BaseModel):
    """JWT token information."""
    model_config = ConfigDict(extra='forbid')
    
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    scope: Optional[str] = None


class JWTPayload(BaseModel):
    """JWT payload structure."""
    model_config = ConfigDict(extra='forbid')
    
    sub: str  # Subject (user ID)
    exp: int  # Expiration time
    iat: int  # Issued at
    iss: str  # Issuer
    aud: str  # Audience
    jti: Optional[str] = None  # JWT ID
    type: str = "access"  # Token type (access, refresh)
    scope: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None


class JWTHandler:
    """JWT token handler for authentication.
    
    Provides functionality to create, verify, and manage JWT tokens
    for API authentication and authorization.
    """
    
    def __init__(self, config: JWTConfig):
        """Initialize JWT handler.
        
        Args:
            config: JWT configuration settings
            
        Raises:
            ImportError: If required JWT libraries are not available
        """
        if jwt is None:
            raise ImportError("PyJWT library not installed. Run: pip install PyJWT[crypto]")
        
        self.config = config
        self._validate_algorithm()
    
    def _validate_algorithm(self) -> None:
        """Validate JWT algorithm configuration."""
        if self.config.algorithm.startswith("RS") or self.config.algorithm.startswith("ES"):
            if rsa is None or serialization is None:
                raise ImportError("Cryptography library required for RSA/ECDSA algorithms")
    
    def create_access_token(self, user_id: str, user_data: Optional[Dict[str, Any]] = None,
                          scope: Optional[str] = None, expires_delta: Optional[timedelta] = None) -> str:
        """Create access token.
        
        Args:
            user_id: User identifier
            user_data: Additional user data to include
            scope: Token scope
            expires_delta: Custom expiration time
            
        Returns:
            JWT access token
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=self.config.access_token_expire_minutes)
        
        now = datetime.now(timezone.utc)
        expire = now + expires_delta
        
        payload = JWTPayload(
            sub=user_id,
            exp=int(expire.timestamp()),
            iat=int(now.timestamp()),
            iss=self.config.issuer,
            aud=self.config.audience,
            type="access",
            scope=scope,
            user_data=user_data
        )
        
        return jwt.encode(
            payload.dict(exclude_none=True),
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
    
    def create_refresh_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create refresh token.
        
        Args:
            user_id: User identifier
            expires_delta: Custom expiration time
            
        Returns:
            JWT refresh token
        """
        if expires_delta is None:
            expires_delta = timedelta(days=self.config.refresh_token_expire_days)
        
        now = datetime.now(timezone.utc)
        expire = now + expires_delta
        
        payload = JWTPayload(
            sub=user_id,
            exp=int(expire.timestamp()),
            iat=int(now.timestamp()),
            iss=self.config.issuer,
            aud=self.config.audience,
            type="refresh"
        )
        
        return jwt.encode(
            payload.dict(exclude_none=True),
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
    
    def create_token_pair(self, user_id: str, user_data: Optional[Dict[str, Any]] = None,
                         scope: Optional[str] = None) -> JWTToken:
        """Create access and refresh token pair.
        
        Args:
            user_id: User identifier
            user_data: Additional user data to include
            scope: Token scope
            
        Returns:
            JWT token pair
        """
        access_token = self.create_access_token(user_id, user_data, scope)
        refresh_token = self.create_refresh_token(user_id)
        
        return JWTToken(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.access_token_expire_minutes * 60,
            scope=scope
        )
    
    def verify_token(self, token: str, token_type: str = "access") -> JWTPayload:
        """Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            token_type: Expected token type (access, refresh)
            
        Returns:
            Decoded JWT payload
            
        Raises:
            jwt.InvalidTokenError: If token is invalid
            ValueError: If token type doesn't match
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer
            )
            
            jwt_payload = JWTPayload(**payload)
            
            if jwt_payload.type != token_type:
                raise ValueError(f"Token type mismatch. Expected: {token_type}, Got: {jwt_payload.type}")
            
            return jwt_payload
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise
    
    def refresh_access_token(self, refresh_token: str, user_data: Optional[Dict[str, Any]] = None,
                           scope: Optional[str] = None) -> str:
        """Create new access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            user_data: Updated user data to include
            scope: Token scope
            
        Returns:
            New JWT access token
            
        Raises:
            jwt.InvalidTokenError: If refresh token is invalid
        """
        refresh_payload = self.verify_token(refresh_token, "refresh")
        return self.create_access_token(refresh_payload.sub, user_data, scope)
    
    def decode_token_without_verification(self, token: str) -> Dict[str, Any]:
        """Decode token without verification (for debugging).
        
        Args:
            token: JWT token to decode
            
        Returns:
            Token payload (unverified)
            
        Warning:
            Only use for debugging. Never trust unverified tokens.
        """
        return jwt.decode(token, options={"verify_signature": False})
    
    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """Get token expiration time.
        
        Args:
            token: JWT token
            
        Returns:
            Token expiration datetime or None if invalid
        """
        try:
            payload = self.decode_token_without_verification(token)
            exp = payload.get("exp")
            if exp:
                return datetime.fromtimestamp(exp, tz=timezone.utc)
            return None
        except Exception as e:
            logger.error(f"Failed to get token expiry: {e}")
            return None
    
    def is_token_expired(self, token: str) -> bool:
        """Check if token is expired.
        
        Args:
            token: JWT token to check
            
        Returns:
            True if token is expired
        """
        expiry = self.get_token_expiry(token)
        if expiry is None:
            return True
        return datetime.now(timezone.utc) > expiry
    
    def blacklist_token(self, token: str) -> bool:
        """Add token to blacklist (placeholder implementation).
        
        Args:
            token: JWT token to blacklist
            
        Returns:
            True if token was blacklisted
            
        Note:
            This is a placeholder. In production, implement with Redis or database.
        """
        # TODO: Implement token blacklisting with persistent storage
        logger.info(f"Token blacklisted (placeholder): {token[:20]}...")
        return True
    
    def generate_api_key(self, user_id: str, name: str, expires_days: int = 365) -> str:
        """Generate long-lived API key.
        
        Args:
            user_id: User identifier
            name: API key name/description
            expires_days: API key expiration in days
            
        Returns:
            JWT API key
        """
        expires_delta = timedelta(days=expires_days)
        now = datetime.now(timezone.utc)
        expire = now + expires_delta
        
        payload = {
            "sub": user_id,
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "type": "api_key",
            "name": name,
            "scope": "api_access"
        }
        
        return jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )


class JWTAuthMiddleware:
    """JWT authentication middleware helper."""
    
    def __init__(self, jwt_handler: JWTHandler):
        """Initialize JWT auth middleware.
        
        Args:
            jwt_handler: JWT handler instance
        """
        self.jwt_handler = jwt_handler
    
    def extract_token_from_header(self, authorization_header: Optional[str]) -> Optional[str]:
        """Extract JWT token from Authorization header.
        
        Args:
            authorization_header: Authorization header value
            
        Returns:
            JWT token or None if not found
        """
        if not authorization_header:
            return None
        
        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        
        return parts[1]
    
    def authenticate_request(self, authorization_header: Optional[str]) -> Optional[JWTPayload]:
        """Authenticate request using JWT token.
        
        Args:
            authorization_header: Authorization header value
            
        Returns:
            JWT payload if authentication succeeds, None otherwise
        """
        token = self.extract_token_from_header(authorization_header)
        if not token:
            return None
        
        try:
            return self.jwt_handler.verify_token(token, "access")
        except jwt.InvalidTokenError:
            return None
    
    def require_scope(self, payload: JWTPayload, required_scope: str) -> bool:
        """Check if token has required scope.
        
        Args:
            payload: JWT payload
            required_scope: Required scope
            
        Returns:
            True if token has required scope
        """
        if not payload.scope:
            return False
        
        token_scopes = payload.scope.split()
        return required_scope in token_scopes or "admin" in token_scopes