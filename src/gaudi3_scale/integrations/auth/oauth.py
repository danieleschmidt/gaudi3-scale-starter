"""OAuth authentication manager."""

import logging
import secrets
import urllib.parse
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import requests
from pydantic import BaseModel, ConfigDict, HttpUrl

logger = logging.getLogger(__name__)


class OAuthConfig(BaseModel):
    """OAuth configuration settings."""
    model_config = ConfigDict(extra='forbid')
    
    client_id: str
    client_secret: str
    authorization_url: HttpUrl
    token_url: HttpUrl
    redirect_uri: HttpUrl
    scopes: List[str] = []
    provider_name: str = "oauth_provider"


class OAuthToken(BaseModel):
    """OAuth token information."""
    model_config = ConfigDict(extra='forbid')
    
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    created_at: datetime = datetime.now()
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_in:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.expires_in)


class OAuthUser(BaseModel):
    """OAuth user information."""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    username: str
    email: Optional[str] = None
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    provider: str
    raw_data: Dict[str, Any] = {}


class OAuthManager:
    """OAuth authentication manager.
    
    Handles OAuth 2.0 authentication flow for external service integration,
    including GitHub, Google, and other OAuth providers.
    """
    
    def __init__(self, config: OAuthConfig):
        """Initialize OAuth manager.
        
        Args:
            config: OAuth configuration settings
        """
        self.config = config
        self.session = requests.Session()
        self._pending_states: Dict[str, Dict[str, Any]] = {}
    
    def generate_authorization_url(self, state: Optional[str] = None, 
                                 additional_params: Optional[Dict[str, str]] = None) -> str:
        """Generate OAuth authorization URL.
        
        Args:
            state: Optional state parameter for CSRF protection
            additional_params: Additional parameters to include
            
        Returns:
            Authorization URL for redirecting user
        """
        if state is None:
            state = secrets.token_urlsafe(32)
        
        # Store state for validation
        self._pending_states[state] = {
            "created_at": datetime.now(),
            "additional_params": additional_params or {}
        }
        
        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": str(self.config.redirect_uri),
            "state": state,
        }
        
        if self.config.scopes:
            params["scope"] = " ".join(self.config.scopes)
        
        if additional_params:
            params.update(additional_params)
        
        query_string = urllib.parse.urlencode(params)
        return f"{self.config.authorization_url}?{query_string}"
    
    def exchange_code_for_token(self, code: str, state: str) -> OAuthToken:
        """Exchange authorization code for access token.
        
        Args:
            code: Authorization code from OAuth callback
            state: State parameter for CSRF validation
            
        Returns:
            OAuth token information
            
        Raises:
            ValueError: If state validation fails
            requests.HTTPError: If token exchange fails
        """
        # Validate state
        if state not in self._pending_states:
            raise ValueError("Invalid or expired state parameter")
        
        # Clean up old states (older than 1 hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self._pending_states = {
            s: data for s, data in self._pending_states.items()
            if data["created_at"] > cutoff
        }
        
        # Remove used state
        del self._pending_states[state]
        
        # Exchange code for token
        data = {
            "grant_type": "authorization_code",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "code": code,
            "redirect_uri": str(self.config.redirect_uri),
        }
        
        try:
            response = self.session.post(
                str(self.config.token_url),
                data=data,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            
            token_data = response.json()
            
            return OAuthToken(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                refresh_token=token_data.get("refresh_token"),
                scope=token_data.get("scope"),
                created_at=datetime.now()
            )
        except requests.RequestException as e:
            logger.error(f"Failed to exchange code for token: {e}")
            raise
    
    def refresh_token(self, refresh_token: str) -> OAuthToken:
        """Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token to use
            
        Returns:
            New OAuth token
            
        Raises:
            requests.HTTPError: If token refresh fails
        """
        data = {
            "grant_type": "refresh_token",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "refresh_token": refresh_token,
        }
        
        try:
            response = self.session.post(
                str(self.config.token_url),
                data=data,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            
            token_data = response.json()
            
            return OAuthToken(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                refresh_token=token_data.get("refresh_token", refresh_token),
                scope=token_data.get("scope"),
                created_at=datetime.now()
            )
        except requests.RequestException as e:
            logger.error(f"Failed to refresh token: {e}")
            raise
    
    def get_user_info(self, token: OAuthToken, user_info_url: str) -> OAuthUser:
        """Get user information using access token.
        
        Args:
            token: OAuth access token
            user_info_url: URL to fetch user information
            
        Returns:
            User information
            
        Raises:
            requests.HTTPError: If user info request fails
        """
        headers = {
            "Authorization": f"{token.token_type} {token.access_token}",
            "Accept": "application/json"
        }
        
        try:
            response = self.session.get(user_info_url, headers=headers)
            response.raise_for_status()
            
            user_data = response.json()
            
            # Provider-specific user data mapping
            if self.config.provider_name == "github":
                return self._map_github_user(user_data)
            elif self.config.provider_name == "google":
                return self._map_google_user(user_data)
            else:
                return self._map_generic_user(user_data)
        except requests.RequestException as e:
            logger.error(f"Failed to get user info: {e}")
            raise
    
    def _map_github_user(self, user_data: Dict[str, Any]) -> OAuthUser:
        """Map GitHub user data to OAuthUser.
        
        Args:
            user_data: GitHub user data
            
        Returns:
            Mapped user information
        """
        return OAuthUser(
            id=str(user_data["id"]),
            username=user_data["login"],
            email=user_data.get("email"),
            name=user_data.get("name"),
            avatar_url=user_data.get("avatar_url"),
            provider="github",
            raw_data=user_data
        )
    
    def _map_google_user(self, user_data: Dict[str, Any]) -> OAuthUser:
        """Map Google user data to OAuthUser.
        
        Args:
            user_data: Google user data
            
        Returns:
            Mapped user information
        """
        return OAuthUser(
            id=user_data["sub"],
            username=user_data.get("email", "").split("@")[0],
            email=user_data.get("email"),
            name=user_data.get("name"),
            avatar_url=user_data.get("picture"),
            provider="google",
            raw_data=user_data
        )
    
    def _map_generic_user(self, user_data: Dict[str, Any]) -> OAuthUser:
        """Map generic user data to OAuthUser.
        
        Args:
            user_data: Generic user data
            
        Returns:
            Mapped user information
        """
        return OAuthUser(
            id=str(user_data.get("id", user_data.get("sub", "unknown"))),
            username=user_data.get("username", user_data.get("login", "unknown")),
            email=user_data.get("email"),
            name=user_data.get("name"),
            avatar_url=user_data.get("avatar_url", user_data.get("picture")),
            provider=self.config.provider_name,
            raw_data=user_data
        )


class GitHubOAuthManager(OAuthManager):
    """GitHub-specific OAuth manager."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """Initialize GitHub OAuth manager.
        
        Args:
            client_id: GitHub OAuth app client ID
            client_secret: GitHub OAuth app client secret
            redirect_uri: Redirect URI for OAuth callback
        """
        config = OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            authorization_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri=redirect_uri,
            scopes=["user:email", "repo"],
            provider_name="github"
        )
        super().__init__(config)
    
    def get_user_info(self, token: OAuthToken) -> OAuthUser:
        """Get GitHub user information.
        
        Args:
            token: OAuth access token
            
        Returns:
            GitHub user information
        """
        return super().get_user_info(token, "https://api.github.com/user")


class GoogleOAuthManager(OAuthManager):
    """Google-specific OAuth manager."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """Initialize Google OAuth manager.
        
        Args:
            client_id: Google OAuth client ID
            client_secret: Google OAuth client secret
            redirect_uri: Redirect URI for OAuth callback
        """
        config = OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            redirect_uri=redirect_uri,
            scopes=["openid", "email", "profile"],
            provider_name="google"
        )
        super().__init__(config)
    
    def get_user_info(self, token: OAuthToken) -> OAuthUser:
        """Get Google user information.
        
        Args:
            token: OAuth access token
            
        Returns:
            Google user information
        """
        return super().get_user_info(token, "https://openidconnect.googleapis.com/v1/userinfo")