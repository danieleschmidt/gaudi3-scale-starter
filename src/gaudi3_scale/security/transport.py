"""Secure transport layer and TLS/SSL management.

This module provides enterprise-grade transport security including TLS/SSL
certificate management, secure HTTP clients, and encrypted communications.
"""

import ssl
import socket
import os
import hashlib
import base64
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

try:
    import certifi
    import cryptography
    from cryptography import x509
    from cryptography.x509.oid import NameOID, ExtensionOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    
    # Mock classes for fallback
    certifi = None
    cryptography = None
    
    class x509:
        class Certificate:
            pass
        
        class CertificateBuilder:
            pass
        
        class CertificateSigningRequest:
            pass
    
    class NameOID:
        COMMON_NAME = None
        ORGANIZATION_NAME = None
    
    class ExtensionOID:
        SUBJECT_ALTERNATIVE_NAME = None
    
    class hashes:
        class SHA256:
            pass
    
    class serialization:
        class Encoding:
            PEM = None
        
        class PrivateFormat:
            PKCS8 = None
        
        class NoEncryption:
            pass
    
    class rsa:
        class RSAPrivateKey:
            pass
        
        @staticmethod
        def generate_private_key(*args, **kwargs):
            return None
    
    class padding:
        class OAEP:
            pass
        
        class PSS:
            pass
    
    def default_backend():
        return None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

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
from ..exceptions import ConfigurationError
from .audit_logging import SecurityAuditLogger

logger = get_logger(__name__)


class TLSVersion(Enum):
    """Supported TLS versions."""
    TLSv1_2 = "TLSv1.2"
    TLSv1_3 = "TLSv1.3"


class CertificateType(Enum):
    """Certificate types."""
    ROOT_CA = "root_ca"
    INTERMEDIATE_CA = "intermediate_ca"
    SERVER = "server"
    CLIENT = "client"
    CODE_SIGNING = "code_signing"


class KeyUsage(Enum):
    """Certificate key usage options."""
    DIGITAL_SIGNATURE = "digital_signature"
    KEY_ENCIPHERMENT = "key_encipherment"
    KEY_AGREEMENT = "key_agreement"
    KEY_CERT_SIGN = "key_cert_sign"
    CRL_SIGN = "crl_sign"
    DATA_ENCIPHERMENT = "data_encipherment"


@dataclass
class CertificateInfo:
    """Certificate information structure."""
    subject: str
    issuer: str
    serial_number: str
    not_valid_before: datetime
    not_valid_after: datetime
    fingerprint_sha256: str
    public_key_algorithm: str
    key_size: int
    signature_algorithm: str
    extensions: Dict[str, Any]
    is_ca: bool = False
    is_self_signed: bool = False
    
    def is_expired(self) -> bool:
        """Check if certificate is expired."""
        return datetime.now(timezone.utc) > self.not_valid_after.replace(tzinfo=timezone.utc)
    
    def expires_soon(self, days: int = 30) -> bool:
        """Check if certificate expires within specified days."""
        expiry_threshold = datetime.now(timezone.utc) + timedelta(days=days)
        return self.not_valid_after.replace(tzinfo=timezone.utc) < expiry_threshold


class TLSConfig(BaseModel):
    """TLS configuration model."""
    
    min_version: TLSVersion = TLSVersion.TLSv1_2
    max_version: TLSVersion = TLSVersion.TLSv1_3
    cipher_suites: Optional[List[str]] = None
    verify_mode: bool = True
    check_hostname: bool = True
    client_cert_required: bool = False
    ca_cert_path: Optional[str] = None
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    key_password: Optional[str] = None
    
    @validator('cipher_suites')
    def validate_cipher_suites(cls, v):
        """Validate cipher suites."""
        if v is None:
            # Default secure cipher suites
            return [
                'ECDHE-RSA-AES256-GCM-SHA384',
                'ECDHE-RSA-AES128-GCM-SHA256',
                'ECDHE-RSA-AES256-SHA384',
                'ECDHE-RSA-AES128-SHA256',
                'AES256-GCM-SHA384',
                'AES128-GCM-SHA256'
            ]
        return v


class CertificateManager:
    """Manages SSL/TLS certificates and their lifecycle."""
    
    def __init__(self, 
                 cert_directory: Optional[Path] = None,
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """Initialize certificate manager.
        
        Args:
            cert_directory: Directory to store certificates
            audit_logger: Security audit logger
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for certificate management")
        
        self.cert_directory = cert_directory or Path.home() / ".gaudi3_scale" / "certs"
        self.cert_directory.mkdir(parents=True, exist_ok=True)
        
        self.audit_logger = audit_logger
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Certificate store
        self.certificates: Dict[str, CertificateInfo] = {}
        self._load_certificates()
    
    def _load_certificates(self):
        """Load existing certificates from directory."""
        for cert_file in self.cert_directory.glob("*.crt"):
            try:
                cert_info = self.parse_certificate_file(cert_file)
                self.certificates[cert_file.stem] = cert_info
                self.logger.debug(f"Loaded certificate: {cert_file.stem}")
            except Exception as e:
                self.logger.error(f"Failed to load certificate {cert_file}: {e}")
    
    def generate_private_key(self, key_size: int = 2048) -> rsa.RSAPrivateKey:
        """Generate RSA private key.
        
        Args:
            key_size: Key size in bits (minimum 2048 recommended)
            
        Returns:
            RSA private key
        """
        if key_size < 2048:
            raise ValueError("Key size must be at least 2048 bits")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="private_key_generated",
                risk_score=2.0,
                threat_indicators=[f"key_size:{key_size}"]
            )
        
        return private_key
    
    def generate_csr(self, 
                    private_key: rsa.RSAPrivateKey,
                    subject_name: str,
                    country: str = "US",
                    state: str = "CA",
                    city: str = "San Francisco",
                    organization: str = "Gaudi3Scale",
                    organizational_unit: str = "IT",
                    email: Optional[str] = None,
                    san_dns_names: Optional[List[str]] = None,
                    san_ip_addresses: Optional[List[str]] = None) -> x509.CertificateSigningRequest:
        """Generate Certificate Signing Request (CSR).
        
        Args:
            private_key: Private key for CSR
            subject_name: Common Name (CN) for certificate
            country: Country code
            state: State/Province
            city: City/Locality
            organization: Organization name
            organizational_unit: Organizational Unit
            email: Email address
            san_dns_names: Subject Alternative Name DNS entries
            san_ip_addresses: Subject Alternative Name IP addresses
            
        Returns:
            Certificate Signing Request
        """
        subject_components = [
            x509.NameAttribute(NameOID.COUNTRY_NAME, country),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, state),
            x509.NameAttribute(NameOID.LOCALITY_NAME, city),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, organizational_unit),
            x509.NameAttribute(NameOID.COMMON_NAME, subject_name)
        ]
        
        if email:
            subject_components.append(x509.NameAttribute(NameOID.EMAIL_ADDRESS, email))
        
        subject = x509.Name(subject_components)
        
        builder = x509.CertificateSigningRequestBuilder()
        builder = builder.subject_name(subject)
        
        # Add Subject Alternative Names if provided
        san_list = []
        if san_dns_names:
            for dns_name in san_dns_names:
                san_list.append(x509.DNSName(dns_name))
        
        if san_ip_addresses:
            import ipaddress
            for ip_addr in san_ip_addresses:
                san_list.append(x509.IPAddress(ipaddress.ip_address(ip_addr)))
        
        if san_list:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False
            )
        
        csr = builder.sign(private_key, hashes.SHA256(), default_backend())
        
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="csr_generated",
                risk_score=1.0,
                threat_indicators=[f"subject:{subject_name}"]
            )
        
        return csr
    
    def generate_self_signed_certificate(self,
                                       subject_name: str,
                                       private_key: Optional[rsa.RSAPrivateKey] = None,
                                       validity_days: int = 365,
                                       is_ca: bool = False,
                                       **kwargs) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Generate self-signed certificate.
        
        Args:
            subject_name: Common Name for certificate
            private_key: Private key (generates new if None)
            validity_days: Certificate validity period
            is_ca: Whether this is a CA certificate
            **kwargs: Additional arguments for CSR generation
            
        Returns:
            Tuple of (certificate, private_key)
        """
        if private_key is None:
            private_key = self.generate_private_key()
        
        # Generate CSR
        csr = self.generate_csr(private_key, subject_name, **kwargs)
        
        # Create certificate from CSR
        subject = csr.subject
        issuer = subject  # Self-signed, so subject == issuer
        
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(issuer)
        builder = builder.public_key(csr.public_key())
        
        # Set validity period
        not_valid_before = datetime.now(timezone.utc)
        not_valid_after = not_valid_before + timedelta(days=validity_days)
        builder = builder.not_valid_before(not_valid_before)
        builder = builder.not_valid_after(not_valid_after)
        
        # Generate serial number
        builder = builder.serial_number(x509.random_serial_number())
        
        # Add extensions
        if is_ca:
            # CA certificate extensions
            builder = builder.add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True
            )
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    content_commitment=False,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True
            )
        else:
            # Server/Client certificate extensions
            builder = builder.add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True
            )
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    data_encipherment=False,
                    key_agreement=False,
                    content_commitment=False,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True
            )
            
            # Extended Key Usage for server authentication
            builder = builder.add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH
                ]),
                critical=True
            )
        
        # Add Subject Key Identifier
        builder = builder.add_extension(
            x509.SubjectKeyIdentifier.from_public_key(csr.public_key()),
            critical=False
        )
        
        # Copy extensions from CSR (like SAN)
        for extension in csr.extensions:
            if extension.oid not in [ExtensionOID.BASIC_CONSTRAINTS, 
                                   ExtensionOID.KEY_USAGE,
                                   ExtensionOID.EXTENDED_KEY_USAGE]:
                builder = builder.add_extension(extension.value, extension.critical)
        
        # Sign the certificate
        certificate = builder.sign(private_key, hashes.SHA256(), default_backend())
        
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="self_signed_cert_generated",
                risk_score=3.0 if is_ca else 2.0,
                threat_indicators=[f"subject:{subject_name}", f"ca:{is_ca}"]
            )
        
        return certificate, private_key
    
    def save_certificate(self, 
                        name: str, 
                        certificate: x509.Certificate, 
                        private_key: Optional[rsa.RSAPrivateKey] = None,
                        password: Optional[bytes] = None) -> Tuple[Path, Optional[Path]]:
        """Save certificate and private key to files.
        
        Args:
            name: Name for certificate files
            certificate: Certificate to save
            private_key: Private key to save (optional)
            password: Password for private key encryption
            
        Returns:
            Tuple of (certificate_path, private_key_path)
        """
        # Save certificate
        cert_path = self.cert_directory / f"{name}.crt"
        cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
        
        cert_path.write_bytes(cert_pem)
        os.chmod(cert_path, 0o644)  # Read-only for owner and group
        
        key_path = None
        if private_key:
            # Save private key
            key_path = self.cert_directory / f"{name}.key"
            
            if password:
                encryption_algorithm = serialization.BestAvailableEncryption(password)
            else:
                encryption_algorithm = serialization.NoEncryption()
            
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption_algorithm
            )
            
            key_path.write_bytes(key_pem)
            os.chmod(key_path, 0o600)  # Read-write for owner only
        
        # Update certificate store
        cert_info = self.parse_certificate(certificate)
        self.certificates[name] = cert_info
        
        self.logger.info(f"Certificate saved: {name}")
        
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="certificate_saved",
                risk_score=1.0,
                threat_indicators=[f"name:{name}"]
            )
        
        return cert_path, key_path
    
    def parse_certificate_file(self, cert_path: Path) -> CertificateInfo:
        """Parse certificate file and extract information."""
        cert_data = cert_path.read_bytes()
        certificate = x509.load_pem_x509_certificate(cert_data, default_backend())
        return self.parse_certificate(certificate)
    
    def parse_certificate(self, certificate: x509.Certificate) -> CertificateInfo:
        """Parse certificate and extract information."""
        # Subject and Issuer
        subject = certificate.subject.rfc4514_string()
        issuer = certificate.issuer.rfc4514_string()
        
        # Serial number
        serial_number = str(certificate.serial_number)
        
        # Validity period
        not_valid_before = certificate.not_valid_before
        not_valid_after = certificate.not_valid_after
        
        # Fingerprint
        fingerprint = hashlib.sha256(certificate.public_bytes(serialization.Encoding.DER)).hexdigest()
        
        # Public key information
        public_key = certificate.public_key()
        if isinstance(public_key, rsa.RSAPublicKey):
            public_key_algorithm = "RSA"
            key_size = public_key.key_size
        else:
            public_key_algorithm = "Unknown"
            key_size = 0
        
        # Signature algorithm
        signature_algorithm = certificate.signature_algorithm_oid._name
        
        # Extensions
        extensions = {}
        is_ca = False
        
        for extension in certificate.extensions:
            ext_name = extension.oid._name
            
            if isinstance(extension.value, x509.BasicConstraints):
                is_ca = extension.value.ca
                extensions[ext_name] = {
                    "ca": extension.value.ca,
                    "path_length": extension.value.path_length
                }
            elif isinstance(extension.value, x509.SubjectAlternativeName):
                san_list = []
                for san in extension.value:
                    if isinstance(san, x509.DNSName):
                        san_list.append(f"DNS:{san.value}")
                    elif isinstance(san, x509.IPAddress):
                        san_list.append(f"IP:{san.value}")
                extensions[ext_name] = san_list
            else:
                extensions[ext_name] = str(extension.value)
        
        # Check if self-signed
        is_self_signed = subject == issuer
        
        return CertificateInfo(
            subject=subject,
            issuer=issuer,
            serial_number=serial_number,
            not_valid_before=not_valid_before,
            not_valid_after=not_valid_after,
            fingerprint_sha256=fingerprint,
            public_key_algorithm=public_key_algorithm,
            key_size=key_size,
            signature_algorithm=signature_algorithm,
            extensions=extensions,
            is_ca=is_ca,
            is_self_signed=is_self_signed
        )
    
    def validate_certificate_chain(self, 
                                 cert_chain: List[x509.Certificate],
                                 trusted_ca_certs: Optional[List[x509.Certificate]] = None) -> Tuple[bool, List[str]]:
        """Validate certificate chain.
        
        Args:
            cert_chain: Certificate chain to validate
            trusted_ca_certs: Trusted CA certificates
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not cert_chain:
            return False, ["Certificate chain is empty"]
        
        # Check certificate validity periods
        current_time = datetime.now(timezone.utc)
        for i, cert in enumerate(cert_chain):
            cert_info = self.parse_certificate(cert)
            
            if cert_info.is_expired():
                errors.append(f"Certificate {i} is expired")
            
            if current_time < cert_info.not_valid_before.replace(tzinfo=timezone.utc):
                errors.append(f"Certificate {i} is not yet valid")
        
        # Check certificate chain integrity
        for i in range(len(cert_chain) - 1):
            current_cert = cert_chain[i]
            parent_cert = cert_chain[i + 1]
            
            try:
                # Verify signature
                parent_public_key = parent_cert.public_key()
                if isinstance(parent_public_key, rsa.RSAPublicKey):
                    parent_public_key.verify(
                        current_cert.signature,
                        current_cert.tbs_certificate_bytes,
                        padding.PKCS1v15(),
                        current_cert.signature_hash_algorithm
                    )
                else:
                    errors.append(f"Unsupported public key algorithm in certificate {i+1}")
            except Exception as e:
                errors.append(f"Certificate {i} signature verification failed: {e}")
        
        return len(errors) == 0, errors
    
    def get_certificate_expiry_report(self, warning_days: int = 30) -> Dict[str, Any]:
        """Get certificate expiry report.
        
        Args:
            warning_days: Days before expiry to warn
            
        Returns:
            Certificate expiry report
        """
        report = {
            "total_certificates": len(self.certificates),
            "expired": [],
            "expiring_soon": [],
            "valid": []
        }
        
        for name, cert_info in self.certificates.items():
            if cert_info.is_expired():
                report["expired"].append({
                    "name": name,
                    "subject": cert_info.subject,
                    "expired_date": cert_info.not_valid_after.isoformat()
                })
            elif cert_info.expires_soon(warning_days):
                report["expiring_soon"].append({
                    "name": name,
                    "subject": cert_info.subject,
                    "expiry_date": cert_info.not_valid_after.isoformat(),
                    "days_until_expiry": (cert_info.not_valid_after.replace(tzinfo=timezone.utc) - 
                                        datetime.now(timezone.utc)).days
                })
            else:
                report["valid"].append({
                    "name": name,
                    "subject": cert_info.subject,
                    "expiry_date": cert_info.not_valid_after.isoformat()
                })
        
        return report
    
    def revoke_certificate(self, name: str, reason: str = "unspecified") -> bool:
        """Revoke a certificate.
        
        Args:
            name: Certificate name
            reason: Revocation reason
            
        Returns:
            True if revoked successfully
        """
        if name not in self.certificates:
            return False
        
        # In a full implementation, this would add to CRL
        # For now, we just remove from active certificates
        cert_info = self.certificates[name]
        del self.certificates[name]
        
        # Remove certificate files
        cert_path = self.cert_directory / f"{name}.crt"
        key_path = self.cert_directory / f"{name}.key"
        
        if cert_path.exists():
            cert_path.unlink()
        if key_path.exists():
            key_path.unlink()
        
        self.logger.warning(f"Certificate revoked: {name} - Reason: {reason}")
        
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="certificate_revoked",
                risk_score=4.0,
                threat_indicators=[f"name:{name}", f"reason:{reason}"]
            )
        
        return True


class TLSManager:
    """Manages TLS/SSL configurations and contexts."""
    
    def __init__(self, 
                 certificate_manager: Optional[CertificateManager] = None,
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """Initialize TLS manager.
        
        Args:
            certificate_manager: Certificate manager instance
            audit_logger: Security audit logger
        """
        self.certificate_manager = certificate_manager or CertificateManager()
        self.audit_logger = audit_logger
        self.logger = logger.getChild(self.__class__.__name__)
        
        # TLS configurations
        self.tls_configs: Dict[str, TLSConfig] = {}
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default TLS configurations."""
        # Secure server configuration
        self.tls_configs["secure_server"] = TLSConfig(
            min_version=TLSVersion.TLSv1_2,
            max_version=TLSVersion.TLSv1_3,
            verify_mode=True,
            check_hostname=False,  # Server doesn't check its own hostname
            client_cert_required=False
        )
        
        # Secure client configuration
        self.tls_configs["secure_client"] = TLSConfig(
            min_version=TLSVersion.TLSv1_2,
            max_version=TLSVersion.TLSv1_3,
            verify_mode=True,
            check_hostname=True,
            client_cert_required=False
        )
        
        # Mutual TLS configuration
        self.tls_configs["mutual_tls"] = TLSConfig(
            min_version=TLSVersion.TLSv1_2,
            max_version=TLSVersion.TLSv1_3,
            verify_mode=True,
            check_hostname=True,
            client_cert_required=True
        )
    
    def create_tls_context(self, 
                          config_name: str = "secure_client",
                          custom_config: Optional[TLSConfig] = None) -> ssl.SSLContext:
        """Create SSL context with specified configuration.
        
        Args:
            config_name: Name of predefined configuration
            custom_config: Custom TLS configuration
            
        Returns:
            SSL context
        """
        if custom_config:
            config = custom_config
        elif config_name in self.tls_configs:
            config = self.tls_configs[config_name]
        else:
            raise ConfigurationError(f"Unknown TLS configuration: {config_name}")
        
        # Create context
        if config.min_version == TLSVersion.TLSv1_3:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_3
        else:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        if config.max_version == TLSVersion.TLSv1_2:
            context.maximum_version = ssl.TLSVersion.TLSv1_2
        else:
            context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Configure verification
        if config.verify_mode:
            context.verify_mode = ssl.CERT_REQUIRED
        else:
            context.verify_mode = ssl.CERT_NONE
        
        context.check_hostname = config.check_hostname
        
        # Load CA certificates
        if config.ca_cert_path:
            context.load_verify_locations(config.ca_cert_path)
        else:
            # Load default CA bundle
            try:
                import certifi
                context.load_verify_locations(certifi.where())
            except ImportError:
                context.load_default_certs()
        
        # Load client certificate if provided
        if config.cert_path and config.key_path:
            context.load_cert_chain(config.cert_path, config.key_path, config.key_password)
        
        # Configure cipher suites
        if config.cipher_suites:
            context.set_ciphers(':'.join(config.cipher_suites))
        
        # Security options
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_COMPRESSION
        context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE
        context.options |= ssl.OP_SINGLE_DH_USE
        context.options |= ssl.OP_SINGLE_ECDH_USE
        
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="tls_context_created",
                risk_score=1.0,
                threat_indicators=[f"config:{config_name}"]
            )
        
        return context
    
    def create_server_socket(self, 
                           host: str, 
                           port: int,
                           cert_name: str,
                           config_name: str = "secure_server") -> ssl.SSLSocket:
        """Create secure server socket.
        
        Args:
            host: Server host
            port: Server port
            cert_name: Certificate name for server
            config_name: TLS configuration name
            
        Returns:
            SSL server socket
        """
        # Get certificate paths
        cert_path = self.certificate_manager.cert_directory / f"{cert_name}.crt"
        key_path = self.certificate_manager.cert_directory / f"{cert_name}.key"
        
        if not cert_path.exists() or not key_path.exists():
            raise ConfigurationError(f"Certificate or key file not found for {cert_name}")
        
        # Update TLS config with certificate paths
        config = self.tls_configs[config_name].copy()
        config.cert_path = str(cert_path)
        config.key_path = str(key_path)
        
        # Create SSL context
        context = self.create_tls_context(custom_config=config)
        
        # Create and bind socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        
        # Wrap with SSL
        ssl_sock = context.wrap_socket(sock, server_side=True)
        
        self.logger.info(f"Secure server socket created: {host}:{port}")
        
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="secure_server_socket_created",
                risk_score=2.0,
                threat_indicators=[f"host:{host}", f"port:{port}", f"cert:{cert_name}"]
            )
        
        return ssl_sock
    
    def verify_tls_connection(self, 
                            hostname: str, 
                            port: int, 
                            timeout: int = 30) -> Dict[str, Any]:
        """Verify TLS connection to a host.
        
        Args:
            hostname: Target hostname
            port: Target port
            timeout: Connection timeout
            
        Returns:
            TLS connection information
        """
        context = self.create_tls_context("secure_client")
        
        try:
            with socket.create_connection((hostname, port), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssl_sock:
                    cert_der = ssl_sock.getpeercert(binary_form=True)
                    cert_info = ssl_sock.getpeercert()
                    cipher = ssl_sock.cipher()
                    version = ssl_sock.version()
                    
                    # Parse certificate
                    if cert_der:
                        certificate = x509.load_der_x509_certificate(cert_der, default_backend())
                        parsed_cert = self.certificate_manager.parse_certificate(certificate)
                    else:
                        parsed_cert = None
                    
                    result = {
                        "success": True,
                        "hostname": hostname,
                        "port": port,
                        "tls_version": version,
                        "cipher_suite": cipher[0] if cipher else None,
                        "certificate": {
                            "subject": cert_info.get('subject', []) if cert_info else [],
                            "issuer": cert_info.get('issuer', []) if cert_info else [],
                            "expires": cert_info.get('notAfter') if cert_info else None,
                            "san": cert_info.get('subjectAltName', []) if cert_info else []
                        } if cert_info else None,
                        "parsed_certificate": parsed_cert.to_dict() if parsed_cert else None
                    }
                    
                    self.logger.info(f"TLS verification successful: {hostname}:{port}")
                    return result
                    
        except Exception as e:
            self.logger.error(f"TLS verification failed for {hostname}:{port}: {e}")
            return {
                "success": False,
                "hostname": hostname,
                "port": port,
                "error": str(e)
            }
    
    def scan_tls_configuration(self, hostname: str, port: int) -> Dict[str, Any]:
        """Scan and analyze TLS configuration of a host.
        
        Args:
            hostname: Target hostname
            port: Target port
            
        Returns:
            TLS configuration analysis
        """
        results = {
            "hostname": hostname,
            "port": port,
            "supported_versions": [],
            "supported_ciphers": [],
            "certificate_info": None,
            "security_score": 0.0,
            "vulnerabilities": [],
            "recommendations": []
        }
        
        # Test TLS versions
        tls_versions = [
            (ssl.TLSVersion.TLSv1, "TLS 1.0"),
            (ssl.TLSVersion.TLSv1_1, "TLS 1.1"),
            (ssl.TLSVersion.TLSv1_2, "TLS 1.2"),
            (ssl.TLSVersion.TLSv1_3, "TLS 1.3")
        ]
        
        for tls_version, version_name in tls_versions:
            try:
                context = ssl.create_default_context()
                context.minimum_version = tls_version
                context.maximum_version = tls_version
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                with socket.create_connection((hostname, port), timeout=10) as sock:
                    with context.wrap_socket(sock) as ssl_sock:
                        results["supported_versions"].append({
                            "version": version_name,
                            "supported": True
                        })
                        
                        if not results["certificate_info"]:
                            cert_der = ssl_sock.getpeercert(binary_form=True)
                            if cert_der:
                                certificate = x509.load_der_x509_certificate(cert_der, default_backend())
                                results["certificate_info"] = self.certificate_manager.parse_certificate(certificate).to_dict()
                        
            except Exception:
                results["supported_versions"].append({
                    "version": version_name,
                    "supported": False
                })
        
        # Analyze security
        security_score = 100.0
        
        # Check for weak TLS versions
        weak_versions = ["TLS 1.0", "TLS 1.1"]
        for version_info in results["supported_versions"]:
            if version_info["supported"] and version_info["version"] in weak_versions:
                results["vulnerabilities"].append(f"Weak TLS version supported: {version_info['version']}")
                security_score -= 20
        
        # Check certificate
        if results["certificate_info"]:
            cert_info = results["certificate_info"]
            if cert_info.get("is_expired", False):
                results["vulnerabilities"].append("Certificate is expired")
                security_score -= 30
            elif datetime.fromisoformat(cert_info["not_valid_after"]) < datetime.now(timezone.utc) + timedelta(days=30):
                results["vulnerabilities"].append("Certificate expires soon")
                security_score -= 10
        
        # Generate recommendations
        if "TLS 1.0" in [v["version"] for v in results["supported_versions"] if v["supported"]]:
            results["recommendations"].append("Disable TLS 1.0 support")
        if "TLS 1.1" in [v["version"] for v in results["supported_versions"] if v["supported"]]:
            results["recommendations"].append("Disable TLS 1.1 support")
        if not any(v["supported"] and v["version"] == "TLS 1.3" for v in results["supported_versions"]):
            results["recommendations"].append("Enable TLS 1.3 support")
        
        results["security_score"] = max(0.0, security_score)
        
        return results


class SecureTransport:
    """High-level secure transport utilities."""
    
    def __init__(self, 
                 tls_manager: Optional[TLSManager] = None,
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """Initialize secure transport.
        
        Args:
            tls_manager: TLS manager instance
            audit_logger: Security audit logger
        """
        self.tls_manager = tls_manager or TLSManager()
        self.audit_logger = audit_logger
        self.logger = logger.getChild(self.__class__.__name__)
    
    def create_secure_http_client(self, 
                                 config_name: str = "secure_client",
                                 verify_ssl: bool = True,
                                 client_cert: Optional[Tuple[str, str]] = None) -> Union['httpx.Client', 'requests.Session']:
        """Create secure HTTP client.
        
        Args:
            config_name: TLS configuration name
            verify_ssl: Whether to verify SSL certificates
            client_cert: Client certificate tuple (cert_path, key_path)
            
        Returns:
            HTTP client (httpx or requests)
        """
        if HTTPX_AVAILABLE:
            return self._create_httpx_client(config_name, verify_ssl, client_cert)
        elif REQUESTS_AVAILABLE:
            return self._create_requests_session(config_name, verify_ssl, client_cert)
        else:
            raise ImportError("Either httpx or requests package is required")
    
    def _create_httpx_client(self, 
                           config_name: str,
                           verify_ssl: bool,
                           client_cert: Optional[Tuple[str, str]]) -> 'httpx.Client':
        """Create httpx client with secure configuration."""
        ssl_context = self.tls_manager.create_tls_context(config_name)
        
        client_config = {
            "verify": ssl_context if verify_ssl else False,
            "timeout": httpx.Timeout(30.0),
            "limits": httpx.Limits(max_keepalive_connections=5, max_connections=10)
        }
        
        if client_cert:
            client_config["cert"] = client_cert
        
        client = httpx.Client(**client_config)
        
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="secure_http_client_created",
                risk_score=1.0,
                threat_indicators=[f"verify_ssl:{verify_ssl}"]
            )
        
        return client
    
    def _create_requests_session(self, 
                               config_name: str,
                               verify_ssl: bool,
                               client_cert: Optional[Tuple[str, str]]) -> 'requests.Session':
        """Create requests session with secure configuration."""
        session = requests.Session()
        
        if verify_ssl:
            try:
                import certifi
                session.verify = certifi.where()
            except ImportError:
                session.verify = True
        else:
            session.verify = False
        
        if client_cert:
            session.cert = client_cert
        
        # Set secure headers
        session.headers.update({
            'User-Agent': 'Gaudi3Scale-SecureClient/1.0',
            'Connection': 'keep-alive'
        })
        
        # Configure adapters with TLS settings
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="secure_http_session_created",
                risk_score=1.0,
                threat_indicators=[f"verify_ssl:{verify_ssl}"]
            )
        
        return session
    
    def secure_request(self, 
                      method: str,
                      url: str,
                      headers: Optional[Dict[str, str]] = None,
                      data: Optional[Any] = None,
                      timeout: int = 30,
                      **kwargs) -> Dict[str, Any]:
        """Make secure HTTP request with automatic security headers.
        
        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
            data: Request data
            timeout: Request timeout
            **kwargs: Additional arguments
            
        Returns:
            Response data
        """
        client = self.create_secure_http_client()
        
        # Add security headers
        secure_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        if headers:
            secure_headers.update(headers)
        
        try:
            if hasattr(client, 'request'):  # httpx
                response = client.request(
                    method=method,
                    url=url,
                    headers=secure_headers,
                    content=data,
                    timeout=timeout,
                    **kwargs
                )
            else:  # requests
                response = client.request(
                    method=method,
                    url=url,
                    headers=secure_headers,
                    data=data,
                    timeout=timeout,
                    **kwargs
                )
            
            result = {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.content if hasattr(response, 'content') else response.text
            }
            
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    event_type="secure_http_request",
                    risk_score=1.0,
                    threat_indicators=[f"method:{method}", f"url:{url}"]
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Secure request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if hasattr(client, 'close'):
                client.close()
    
    def validate_ssl_configuration(self, urls: List[str]) -> Dict[str, Any]:
        """Validate SSL configuration for multiple URLs.
        
        Args:
            urls: List of URLs to validate
            
        Returns:
            SSL validation report
        """
        report = {
            "total_urls": len(urls),
            "secure_urls": 0,
            "insecure_urls": 0,
            "results": {}
        }
        
        for url in urls:
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                hostname = parsed_url.hostname
                port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
                
                if parsed_url.scheme == 'https':
                    tls_info = self.tls_manager.verify_tls_connection(hostname, port)
                    scan_results = self.tls_manager.scan_tls_configuration(hostname, port)
                    
                    is_secure = (tls_info["success"] and 
                               scan_results["security_score"] >= 70.0)
                    
                    report["results"][url] = {
                        "secure": is_secure,
                        "tls_info": tls_info,
                        "scan_results": scan_results
                    }
                    
                    if is_secure:
                        report["secure_urls"] += 1
                    else:
                        report["insecure_urls"] += 1
                else:
                    report["results"][url] = {
                        "secure": False,
                        "error": "HTTP protocol is not secure"
                    }
                    report["insecure_urls"] += 1
                    
            except Exception as e:
                report["results"][url] = {
                    "secure": False,
                    "error": str(e)
                }
                report["insecure_urls"] += 1
        
        return report