"""Security module for Gaudi 3 Scale - Enterprise-grade security hardening.

This module provides comprehensive security features including:
- Secure configuration and secrets management
- Advanced authentication and authorization
- Security monitoring and audit logging
- Secure communications and transport security
- Rate limiting and DoS protection
- Security validation and sanitization
"""

from .config_security import (
    SecureConfigManager,
    SecretsManager,
    EncryptionManager,
    ConfigEncryption
)
from .authentication import (
    AuthenticationManager,
    AuthorizationManager,
    TokenManager,
    RoleManager,
    Permission
)
from .audit_logging import (
    SecurityAuditLogger,
    AuditEvent,
    AuditLevel,
    ComplianceLogger
)
from .monitoring import (
    SecurityMonitor,
    SecurityAlerts,
    ThreatDetection,
    IncidentResponse
)
from .transport import (
    TLSManager,
    CertificateManager,
    SecureTransport
)
from .rate_limiting import (
    RateLimiter,
    DoSProtection,
    RequestThrottler
)
from .validation import (
    SecurityValidator,
    InputSanitizer,
    AntiInjectionValidator
)

__all__ = [
    # Configuration Security
    "SecureConfigManager",
    "SecretsManager", 
    "EncryptionManager",
    "ConfigEncryption",
    
    # Authentication & Authorization
    "AuthenticationManager",
    "AuthorizationManager",
    "TokenManager",
    "RoleManager",
    "Permission",
    
    # Audit & Compliance
    "SecurityAuditLogger",
    "AuditEvent", 
    "AuditLevel",
    "ComplianceLogger",
    
    # Monitoring & Alerting
    "SecurityMonitor",
    "SecurityAlerts",
    "ThreatDetection",
    "IncidentResponse",
    
    # Transport Security
    "TLSManager",
    "CertificateManager",
    "SecureTransport",
    
    # Rate Limiting & DoS Protection  
    "RateLimiter",
    "DoSProtection",
    "RequestThrottler",
    
    # Input Validation & Sanitization
    "SecurityValidator",
    "InputSanitizer", 
    "AntiInjectionValidator"
]

# Security configuration constants
DEFAULT_ENCRYPTION_ALGORITHM = "AES-256-GCM"
DEFAULT_KEY_DERIVATION = "PBKDF2"
DEFAULT_JWT_ALGORITHM = "RS256"  # Use RSA for production
DEFAULT_SESSION_TIMEOUT = 3600  # 1 hour
DEFAULT_MAX_LOGIN_ATTEMPTS = 5
DEFAULT_RATE_LIMIT = 1000  # requests per hour
DEFAULT_TLS_VERSION = "TLSv1.3"

# Security feature flags
SECURITY_FEATURES = {
    "ENCRYPTION_AT_REST": True,
    "ENCRYPTION_IN_TRANSIT": True,
    "MULTI_FACTOR_AUTH": True,
    "AUDIT_LOGGING": True,
    "RATE_LIMITING": True,
    "DOS_PROTECTION": True,
    "INTRUSION_DETECTION": True,
    "COMPLIANCE_MONITORING": True,
    "SECURE_HEADERS": True,
    "CSRF_PROTECTION": True,
    "SQL_INJECTION_PROTECTION": True,
    "XSS_PROTECTION": True
}

def get_security_config():
    """Get current security configuration."""
    return {
        "encryption_algorithm": DEFAULT_ENCRYPTION_ALGORITHM,
        "key_derivation": DEFAULT_KEY_DERIVATION,
        "jwt_algorithm": DEFAULT_JWT_ALGORITHM,
        "session_timeout": DEFAULT_SESSION_TIMEOUT,
        "max_login_attempts": DEFAULT_MAX_LOGIN_ATTEMPTS,
        "rate_limit": DEFAULT_RATE_LIMIT,
        "tls_version": DEFAULT_TLS_VERSION,
        "features": SECURITY_FEATURES
    }

def enable_security_features(features: dict):
    """Enable/disable security features dynamically."""
    global SECURITY_FEATURES
    SECURITY_FEATURES.update(features)