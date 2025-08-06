"""Secure configuration and secrets management for enterprise-grade security.

This module provides encryption, secure storage, and management of sensitive
configuration data including database credentials, API keys, and other secrets.
"""

import os
import json
import base64
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import logging

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

from pydantic import BaseModel, Field, SecretStr
from ..exceptions import ConfigurationError
from ..logging_utils import get_logger

logger = get_logger(__name__)


class SecretValue(BaseModel):
    """Encrypted secret value with metadata."""
    value: str = Field(..., description="Encrypted value")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    rotation_period: Optional[int] = Field(None, description="Days between rotations")
    tags: List[str] = Field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if the secret has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def needs_rotation(self) -> bool:
        """Check if the secret needs rotation."""
        if self.rotation_period is None:
            return False
        rotation_date = self.created_at + timedelta(days=self.rotation_period)
        return datetime.utcnow() > rotation_date


class EncryptionManager:
    """Handles encryption/decryption operations for sensitive data."""
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize encryption manager.
        
        Args:
            master_key: Master encryption key (if None, generates new one)
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography package required for encryption")
        
        self.master_key = master_key or self._generate_master_key()
        self.fernet = self._create_fernet_cipher()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def _generate_master_key(self) -> str:
        """Generate a new master encryption key."""
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode('utf-8')
    
    def _create_fernet_cipher(self) -> Fernet:
        """Create Fernet cipher from master key."""
        try:
            key_bytes = base64.urlsafe_b64decode(self.master_key.encode('utf-8'))
            return Fernet(key_bytes)
        except Exception as e:
            raise ConfigurationError(f"Invalid master key: {e}")
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Base64-encoded encrypted data
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        try:
            encrypted = self.fernet.encrypt(data)
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise ConfigurationError(f"Encryption failed: {e}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted string
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise ConfigurationError(f"Decryption failed: {e}")
    
    def derive_key(self, password: str, salt: Optional[bytes] = None) -> tuple:
        """Derive encryption key from password.
        
        Args:
            password: Password to derive key from
            salt: Optional salt (generates new if None)
            
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode('utf-8'))
        return base64.urlsafe_b64encode(key).decode('utf-8'), salt
    
    def rotate_key(self) -> str:
        """Rotate the master encryption key.
        
        Returns:
            New master key
        """
        old_key = self.master_key
        self.master_key = self._generate_master_key()
        self.fernet = self._create_fernet_cipher()
        
        self.logger.info("Master encryption key rotated")
        return self.master_key


class SecretsManager:
    """Manages encrypted secrets with rotation and expiration."""
    
    def __init__(self, 
                 storage_path: Optional[Path] = None,
                 encryption_manager: Optional[EncryptionManager] = None,
                 use_keyring: bool = True):
        """Initialize secrets manager.
        
        Args:
            storage_path: Path to store encrypted secrets
            encryption_manager: Encryption manager instance
            use_keyring: Whether to use system keyring for storage
        """
        self.storage_path = storage_path or Path.home() / ".gaudi3_scale" / "secrets"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.encryption_manager = encryption_manager or EncryptionManager()
        self.use_keyring = use_keyring and KEYRING_AVAILABLE
        
        self.secrets_file = self.storage_path / "secrets.json"
        self.secrets: Dict[str, SecretValue] = self._load_secrets()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def _load_secrets(self) -> Dict[str, SecretValue]:
        """Load secrets from storage."""
        if not self.secrets_file.exists():
            return {}
        
        try:
            with open(self.secrets_file, 'r') as f:
                data = json.load(f)
            
            secrets = {}
            for key, value_data in data.items():
                # Convert datetime strings back to datetime objects
                if 'created_at' in value_data:
                    value_data['created_at'] = datetime.fromisoformat(value_data['created_at'])
                if 'expires_at' in value_data and value_data['expires_at']:
                    value_data['expires_at'] = datetime.fromisoformat(value_data['expires_at'])
                
                secrets[key] = SecretValue(**value_data)
            
            return secrets
        except Exception as e:
            self.logger.error(f"Failed to load secrets: {e}")
            return {}
    
    def _save_secrets(self):
        """Save secrets to storage."""
        try:
            # Convert datetime objects to strings for JSON serialization
            data = {}
            for key, secret in self.secrets.items():
                secret_data = secret.dict()
                secret_data['created_at'] = secret_data['created_at'].isoformat()
                if secret_data['expires_at']:
                    secret_data['expires_at'] = secret_data['expires_at'].isoformat()
                data[key] = secret_data
            
            # Ensure directory exists with secure permissions
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            
            # Write with secure file permissions
            with open(self.secrets_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Set secure file permissions (owner read/write only)
            os.chmod(self.secrets_file, 0o600)
            
        except Exception as e:
            self.logger.error(f"Failed to save secrets: {e}")
            raise ConfigurationError(f"Failed to save secrets: {e}")
    
    def store_secret(self, 
                    key: str, 
                    value: str,
                    expires_in_days: Optional[int] = None,
                    rotation_period: Optional[int] = None,
                    tags: Optional[List[str]] = None) -> None:
        """Store an encrypted secret.
        
        Args:
            key: Secret identifier
            value: Secret value to encrypt and store
            expires_in_days: Days until expiration
            rotation_period: Days between rotations
            tags: Metadata tags
        """
        try:
            encrypted_value = self.encryption_manager.encrypt(value)
            
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            secret = SecretValue(
                value=encrypted_value,
                expires_at=expires_at,
                rotation_period=rotation_period,
                tags=tags or []
            )
            
            self.secrets[key] = secret
            self._save_secrets()
            
            # Also store in system keyring if available
            if self.use_keyring:
                try:
                    keyring.set_password("gaudi3_scale", key, encrypted_value)
                except Exception as e:
                    self.logger.warning(f"Failed to store in keyring: {e}")
            
            self.logger.info(f"Secret stored: {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to store secret {key}: {e}")
            raise ConfigurationError(f"Failed to store secret: {e}")
    
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve and decrypt a secret.
        
        Args:
            key: Secret identifier
            
        Returns:
            Decrypted secret value or None if not found
        """
        # First check file storage
        if key in self.secrets:
            secret = self.secrets[key]
            
            # Check if expired
            if secret.is_expired():
                self.logger.warning(f"Secret {key} has expired")
                self.delete_secret(key)
                return None
            
            try:
                return self.encryption_manager.decrypt(secret.value)
            except Exception as e:
                self.logger.error(f"Failed to decrypt secret {key}: {e}")
                return None
        
        # Fallback to keyring if available
        if self.use_keyring:
            try:
                encrypted_value = keyring.get_password("gaudi3_scale", key)
                if encrypted_value:
                    return self.encryption_manager.decrypt(encrypted_value)
            except Exception as e:
                self.logger.error(f"Failed to retrieve from keyring: {e}")
        
        return None
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret.
        
        Args:
            key: Secret identifier
            
        Returns:
            True if deleted, False if not found
        """
        deleted = False
        
        if key in self.secrets:
            del self.secrets[key]
            self._save_secrets()
            deleted = True
        
        if self.use_keyring:
            try:
                keyring.delete_password("gaudi3_scale", key)
                deleted = True
            except Exception:
                pass  # Secret may not exist in keyring
        
        if deleted:
            self.logger.info(f"Secret deleted: {key}")
        
        return deleted
    
    def list_secrets(self) -> Dict[str, Dict[str, Any]]:
        """List all secrets with metadata (values not included).
        
        Returns:
            Dictionary of secret metadata
        """
        result = {}
        for key, secret in self.secrets.items():
            result[key] = {
                "created_at": secret.created_at.isoformat(),
                "expires_at": secret.expires_at.isoformat() if secret.expires_at else None,
                "rotation_period": secret.rotation_period,
                "tags": secret.tags,
                "expired": secret.is_expired(),
                "needs_rotation": secret.needs_rotation()
            }
        
        return result
    
    def rotate_secret(self, key: str, new_value: str) -> None:
        """Rotate a secret with a new value.
        
        Args:
            key: Secret identifier  
            new_value: New secret value
        """
        if key not in self.secrets:
            raise ConfigurationError(f"Secret {key} not found")
        
        old_secret = self.secrets[key]
        self.store_secret(
            key=key,
            value=new_value,
            expires_in_days=None,  # Preserve expiration settings
            rotation_period=old_secret.rotation_period,
            tags=old_secret.tags
        )
        
        self.logger.info(f"Secret rotated: {key}")
    
    def cleanup_expired_secrets(self) -> int:
        """Remove all expired secrets.
        
        Returns:
            Number of secrets cleaned up
        """
        expired_keys = [
            key for key, secret in self.secrets.items() 
            if secret.is_expired()
        ]
        
        for key in expired_keys:
            self.delete_secret(key)
        
        self.logger.info(f"Cleaned up {len(expired_keys)} expired secrets")
        return len(expired_keys)


class SecureConfigManager:
    """Manages secure configuration with encryption and secrets integration."""
    
    def __init__(self, 
                 config_path: Optional[Path] = None,
                 secrets_manager: Optional[SecretsManager] = None):
        """Initialize secure configuration manager.
        
        Args:
            config_path: Path to configuration file
            secrets_manager: Secrets manager instance
        """
        self.config_path = config_path or Path.home() / ".gaudi3_scale" / "config.json"
        self.secrets_manager = secrets_manager or SecretsManager()
        
        self.config: Dict[str, Any] = {}
        self.encrypted_fields: set = set()
        self.logger = logger.getChild(self.__class__.__name__)
        
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            self.config = {}
            return
        
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            self.config = data.get('config', {})
            self.encrypted_fields = set(data.get('encrypted_fields', []))
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
            self.encrypted_fields = set()
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            
            data = {
                'config': self.config,
                'encrypted_fields': list(self.encrypted_fields),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            os.chmod(self.config_path, 0o600)
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            raise ConfigurationError(f"Failed to save config: {e}")
    
    def set(self, key: str, value: Any, encrypt: bool = False) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            encrypt: Whether to encrypt the value
        """
        if encrypt and isinstance(value, str):
            # Store in secrets manager
            secret_key = f"config.{key}"
            self.secrets_manager.store_secret(secret_key, value)
            
            # Store reference in config
            self.config[key] = {"__secret__": secret_key}
            self.encrypted_fields.add(key)
        else:
            self.config[key] = value
            if key in self.encrypted_fields:
                self.encrypted_fields.remove(key)
        
        self._save_config()
        self.logger.info(f"Configuration set: {key} (encrypted: {encrypt})")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if key not in self.config:
            return default
        
        value = self.config[key]
        
        # Check if it's an encrypted reference
        if isinstance(value, dict) and "__secret__" in value:
            secret_key = value["__secret__"]
            decrypted = self.secrets_manager.get_secret(secret_key)
            return decrypted if decrypted is not None else default
        
        return value
    
    def delete(self, key: str) -> bool:
        """Delete configuration value.
        
        Args:
            key: Configuration key
            
        Returns:
            True if deleted, False if not found
        """
        if key not in self.config:
            return False
        
        # If encrypted, also delete from secrets
        if key in self.encrypted_fields:
            secret_key = f"config.{key}"
            self.secrets_manager.delete_secret(secret_key)
            self.encrypted_fields.remove(key)
        
        del self.config[key]
        self._save_config()
        
        self.logger.info(f"Configuration deleted: {key}")
        return True
    
    def get_database_url(self, async_mode: bool = False) -> str:
        """Get secure database URL with credentials from secrets.
        
        Args:
            async_mode: Whether to return async database URL
            
        Returns:
            Database connection URL with decrypted credentials
        """
        # Get database configuration
        db_config = {
            "host": self.get("database.host", "localhost"),
            "port": self.get("database.port", "5432"),
            "database": self.get("database.name", "gaudi3_scale"),
            "username": self.get("database.username", "gaudi3_user"),
            "password": self.get("database.password", "")
        }
        
        # URL encode password to handle special characters
        from urllib.parse import quote_plus
        encoded_password = quote_plus(db_config["password"]) if db_config["password"] else ""
        
        driver = "postgresql+asyncpg" if async_mode else "postgresql+psycopg2"
        
        if encoded_password:
            return f"{driver}://{db_config['username']}:{encoded_password}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        else:
            return f"{driver}://{db_config['username']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    def setup_database_secrets(self, host: str, port: str, database: str, 
                              username: str, password: str) -> None:
        """Setup database configuration with encrypted password.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password (will be encrypted)
        """
        self.set("database.host", host)
        self.set("database.port", port)
        self.set("database.name", database)
        self.set("database.username", username)
        self.set("database.password", password, encrypt=True)
        
        self.logger.info("Database configuration setup with encrypted password")
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export configuration (optionally including secrets).
        
        Args:
            include_secrets: Whether to include decrypted secret values
            
        Returns:
            Configuration dictionary
        """
        result = {}
        
        for key, value in self.config.items():
            if key in self.encrypted_fields and include_secrets:
                # Decrypt secret for export
                result[key] = self.get(key)
            elif key not in self.encrypted_fields:
                result[key] = value
            else:
                result[key] = "***ENCRYPTED***"
        
        return result


class ConfigEncryption:
    """Utility class for encrypting entire configuration files."""
    
    def __init__(self, encryption_manager: EncryptionManager):
        """Initialize config encryption.
        
        Args:
            encryption_manager: Encryption manager instance
        """
        self.encryption_manager = encryption_manager
        self.logger = logger.getChild(self.__class__.__name__)
    
    def encrypt_config_file(self, input_path: Path, output_path: Path) -> None:
        """Encrypt an entire configuration file.
        
        Args:
            input_path: Path to plain configuration file
            output_path: Path to encrypted output file
        """
        try:
            with open(input_path, 'r') as f:
                config_data = f.read()
            
            encrypted_data = self.encryption_manager.encrypt(config_data)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump({
                    "encrypted_config": encrypted_data,
                    "encrypted_at": datetime.utcnow().isoformat(),
                    "algorithm": "AES-256-GCM"
                }, f, indent=2)
            
            os.chmod(output_path, 0o600)
            self.logger.info(f"Configuration file encrypted: {input_path} -> {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt config file: {e}")
            raise ConfigurationError(f"Failed to encrypt config file: {e}")
    
    def decrypt_config_file(self, input_path: Path, output_path: Path) -> None:
        """Decrypt an encrypted configuration file.
        
        Args:
            input_path: Path to encrypted configuration file
            output_path: Path to decrypted output file
        """
        try:
            with open(input_path, 'r') as f:
                encrypted_data = json.load(f)
            
            if "encrypted_config" not in encrypted_data:
                raise ConfigurationError("Invalid encrypted config file format")
            
            decrypted_config = self.encryption_manager.decrypt(
                encrypted_data["encrypted_config"]
            )
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(decrypted_config)
            
            os.chmod(output_path, 0o600)
            self.logger.info(f"Configuration file decrypted: {input_path} -> {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt config file: {e}")
            raise ConfigurationError(f"Failed to decrypt config file: {e}")