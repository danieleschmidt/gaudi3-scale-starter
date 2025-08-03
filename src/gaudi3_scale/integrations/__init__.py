"""External service integrations for Gaudi 3 Scale."""

from .github.client import GitHubClient
from .github.webhooks import GitHubWebhookHandler
from .github.actions import GitHubActionsIntegration
from .notifications.email.client import EmailClient
from .notifications.slack import SlackNotifier
from .auth.oauth import OAuthManager, GitHubOAuthManager, GoogleOAuthManager
from .auth.jwt import JWTHandler, JWTAuthMiddleware

__all__ = [
    "GitHubClient",
    "GitHubWebhookHandler",
    "GitHubActionsIntegration",
    "EmailClient", 
    "SlackNotifier",
    "OAuthManager",
    "GitHubOAuthManager",
    "GoogleOAuthManager",
    "JWTHandler",
    "JWTAuthMiddleware",
]