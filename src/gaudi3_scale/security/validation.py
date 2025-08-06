"""Advanced security validation and sanitization for enterprise-grade protection.

This module provides comprehensive input validation, sanitization, and 
anti-injection protection to prevent security vulnerabilities.
"""

import re
import html
import json
import base64
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime

try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError:
    BLEACH_AVAILABLE = False

try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False

from pydantic import BaseModel, validator
from ..validation import ValidationResult, DataValidator
from ..exceptions import InputSanitizationError, SecurityValidationError
from ..logging_utils import get_logger

logger = get_logger(__name__)


class SecurityThreat(BaseModel):
    """Represents a detected security threat."""
    type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    matched_pattern: Optional[str] = None
    location: Optional[str] = None
    recommendation: Optional[str] = None
    cve_reference: Optional[str] = None


class SecurityValidationResult(ValidationResult):
    """Extended validation result with security threat information."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threats: List[SecurityThreat] = []
        self.security_score: float = 100.0  # Start with perfect score
    
    def add_threat(self, threat: SecurityThreat) -> None:
        """Add a security threat."""
        self.threats.append(threat)
        self.is_valid = False
        
        # Adjust security score based on severity
        severity_impact = {
            'LOW': 5,
            'MEDIUM': 15,
            'HIGH': 30,
            'CRITICAL': 50
        }
        self.security_score -= severity_impact.get(threat.severity, 10)
        self.security_score = max(0, self.security_score)
    
    def get_threat_summary(self) -> Dict[str, int]:
        """Get summary of threats by severity."""
        summary = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for threat in self.threats:
            summary[threat.severity] += 1
        return summary


class SecurityValidator(DataValidator):
    """Advanced security validator with threat detection capabilities."""
    
    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bDROP\b.*\b(TABLE|DATABASE|SCHEMA)\b)",
        r"(\bALTER\b.*\b(TABLE|DATABASE|SCHEMA)\b)",
        r"(\bCREATE\b.*\b(TABLE|DATABASE|SCHEMA|USER)\b)",
        r"(\bGRANT\b|\bREVOKE\b)",
        r"(\bEXEC\b|\bEXECUTE\b)",
        r"(\bxp_cmdshell\b|\bsp_executesql\b)",
        r"('.*'|\".*\")\s*;\s*(\bDROP\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b)",
        r"(\bOR\b|\bAND\b)\s+('.*'|\".*\"|\d+)\s*=\s*('.*'|\".*\"|\d+)",
        r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+",
        r"(\'\s*OR\s*\'1\'\s*=\s*\'1|\"\s*OR\s*\"1\"\s*=\s*\"1)",
        r"(\'\s*OR\s*\'\w+\'\s*=\s*\'\w+|\"\s*OR\s*\"\w+\"\s*=\s*\"\w+)",
        r"(\/\*.*\*\/|--[^\r\n]*|#[^\r\n]*)",  # SQL comments
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"<style[^>]*>.*?</style>",
        r"javascript\s*:",
        r"vbscript\s*:",
        r"data\s*:",
        r"on\w+\s*=",  # Event handlers
        r"<\s*\w+[^>]*on\w+\s*=",
        r"<\s*img[^>]*src\s*=[^>]*javascript:",
        r"<\s*svg[^>]*onload\s*=",
        r"<\s*body[^>]*onload\s*=",
        r"<\s*div[^>]*onclick\s*=",
        r"expression\s*\(",  # CSS expression
        r"@import\s*[\"']javascript:",
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}[\]\\]",  # Shell metacharacters
        r"\b(cat|ls|pwd|whoami|id|ps|netstat|ifconfig|ping|nslookup|dig|nc|telnet|ssh|scp|rsync|wget|curl|chmod|chown|su|sudo|passwd|mount|umount|fdisk|dd|rm|mv|cp|find|grep|awk|sed|sort|uniq|head|tail|wc|tar|gzip|gunzip|zip|unzip)\b",
        r"(\/bin\/|\/usr\/bin\/|\/sbin\/|\/usr\/sbin\/)",
        r"(\$\(.*\)|`.*`)",  # Command substitution
        r"(>|>>|<|<<)",  # Redirection
        r"(\|\s*\w+|\w+\s*\|)",  # Pipes
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
        r"..%2f",
        r"..%5c",
        r"%252e%252e%252f",
        r"%c0%ae%c0%ae%c0%af",
        r"\x2e\x2e\x2f",
        r"\x2e\x2e\x5c",
    ]
    
    # LDAP injection patterns
    LDAP_INJECTION_PATTERNS = [
        r"\*",
        r"\(",
        r"\)",
        r"\\",
        r"\x00",
        r"\/",
        r"(\|\|\|\&\&\&)",
        r"(\)\s*\(\s*\|)",
        r"(\)\s*\(\s*\&)",
    ]
    
    # NoSQL injection patterns
    NOSQL_INJECTION_PATTERNS = [
        r"\$\w+",  # MongoDB operators
        r"{\s*\$where\s*:",
        r"{\s*\$regex\s*:",
        r"{\s*\$ne\s*:",
        r"{\s*\$gt\s*:",
        r"{\s*\$lt\s*:",
        r"{\s*\$in\s*:",
        r"{\s*\$nin\s*:",
        r"function\s*\(",
        r"this\.",
        r"sleep\s*\(",
        r"benchmark\s*\(",
    ]
    
    # XXE patterns
    XXE_PATTERNS = [
        r"<!ENTITY",
        r"<!DOCTYPE",
        r"SYSTEM\s+[\"'][^\"']*[\"']",
        r"PUBLIC\s+[\"'][^\"']*[\"']\s+[\"'][^\"']*[\"']",
        r"&\w+;",
        r"file://",
        r"http://",
        r"https://",
        r"ftp://",
    ]
    
    # File upload dangerous extensions
    DANGEROUS_FILE_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.vbe',
        '.js', '.jar', '.jsp', '.php', '.asp', '.aspx', '.pl', '.py',
        '.rb', '.sh', '.bash', '.csh', '.ksh', '.tcsh', '.zsh',
        '.ps1', '.psm1', '.psd1', '.ps1xml', '.pssc', '.psrc',
        '.applescript', '.scpt', '.app', '.deb', '.rpm', '.dmg',
        '.iso', '.img', '.vhd', '.vmdk', '.ova', '.ovf'
    }
    
    def __init__(self):
        """Initialize security validator."""
        super().__init__()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Compile regex patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.compiled_sql_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.SQL_INJECTION_PATTERNS
        ]
        
        self.compiled_xss_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.XSS_PATTERNS
        ]
        
        self.compiled_command_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.COMMAND_INJECTION_PATTERNS
        ]
        
        self.compiled_path_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.PATH_TRAVERSAL_PATTERNS
        ]
        
        self.compiled_ldap_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.LDAP_INJECTION_PATTERNS
        ]
        
        self.compiled_nosql_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.NOSQL_INJECTION_PATTERNS
        ]
        
        self.compiled_xxe_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.XXE_PATTERNS
        ]
    
    def validate_against_injection(self, 
                                  value: str, 
                                  field_name: str,
                                  check_sql: bool = True,
                                  check_xss: bool = True,
                                  check_command: bool = True,
                                  check_path: bool = True,
                                  check_ldap: bool = False,
                                  check_nosql: bool = False,
                                  check_xxe: bool = False) -> SecurityValidationResult:
        """Validate input against various injection attacks.
        
        Args:
            value: Input value to validate
            field_name: Name of the field
            check_sql: Check for SQL injection
            check_xss: Check for XSS
            check_command: Check for command injection
            check_path: Check for path traversal
            check_ldap: Check for LDAP injection
            check_nosql: Check for NoSQL injection
            check_xxe: Check for XXE attacks
            
        Returns:
            SecurityValidationResult with threat information
        """
        result = SecurityValidationResult()
        result.sanitized_value = value
        
        if not isinstance(value, str):
            return result
        
        # SQL Injection Check
        if check_sql:
            self._check_sql_injection(value, field_name, result)
        
        # XSS Check
        if check_xss:
            self._check_xss(value, field_name, result)
        
        # Command Injection Check
        if check_command:
            self._check_command_injection(value, field_name, result)
        
        # Path Traversal Check
        if check_path:
            self._check_path_traversal(value, field_name, result)
        
        # LDAP Injection Check
        if check_ldap:
            self._check_ldap_injection(value, field_name, result)
        
        # NoSQL Injection Check
        if check_nosql:
            self._check_nosql_injection(value, field_name, result)
        
        # XXE Check
        if check_xxe:
            self._check_xxe(value, field_name, result)
        
        return result
    
    def _check_sql_injection(self, value: str, field_name: str, result: SecurityValidationResult):
        """Check for SQL injection patterns."""
        for pattern in self.compiled_sql_patterns:
            match = pattern.search(value)
            if match:
                threat = SecurityThreat(
                    type="SQL_INJECTION",
                    severity="CRITICAL",
                    description=f"Potential SQL injection detected in {field_name}",
                    matched_pattern=match.group(),
                    location=field_name,
                    recommendation="Use parameterized queries and input validation",
                    cve_reference="CWE-89"
                )
                result.add_threat(threat)
                self.logger.warning(f"SQL injection threat detected: {field_name}")
                break
    
    def _check_xss(self, value: str, field_name: str, result: SecurityValidationResult):
        """Check for XSS patterns."""
        for pattern in self.compiled_xss_patterns:
            match = pattern.search(value)
            if match:
                threat = SecurityThreat(
                    type="XSS",
                    severity="HIGH",
                    description=f"Potential XSS attack detected in {field_name}",
                    matched_pattern=match.group(),
                    location=field_name,
                    recommendation="Use proper output encoding and CSP headers",
                    cve_reference="CWE-79"
                )
                result.add_threat(threat)
                self.logger.warning(f"XSS threat detected: {field_name}")
                break
    
    def _check_command_injection(self, value: str, field_name: str, result: SecurityValidationResult):
        """Check for command injection patterns."""
        for pattern in self.compiled_command_patterns:
            match = pattern.search(value)
            if match:
                threat = SecurityThreat(
                    type="COMMAND_INJECTION",
                    severity="CRITICAL",
                    description=f"Potential command injection detected in {field_name}",
                    matched_pattern=match.group(),
                    location=field_name,
                    recommendation="Avoid system calls with user input or use allowlists",
                    cve_reference="CWE-78"
                )
                result.add_threat(threat)
                self.logger.warning(f"Command injection threat detected: {field_name}")
                break
    
    def _check_path_traversal(self, value: str, field_name: str, result: SecurityValidationResult):
        """Check for path traversal patterns."""
        for pattern in self.compiled_path_patterns:
            match = pattern.search(value)
            if match:
                threat = SecurityThreat(
                    type="PATH_TRAVERSAL",
                    severity="HIGH",
                    description=f"Potential path traversal detected in {field_name}",
                    matched_pattern=match.group(),
                    location=field_name,
                    recommendation="Validate and sanitize file paths",
                    cve_reference="CWE-22"
                )
                result.add_threat(threat)
                self.logger.warning(f"Path traversal threat detected: {field_name}")
                break
    
    def _check_ldap_injection(self, value: str, field_name: str, result: SecurityValidationResult):
        """Check for LDAP injection patterns."""
        for pattern in self.compiled_ldap_patterns:
            match = pattern.search(value)
            if match:
                threat = SecurityThreat(
                    type="LDAP_INJECTION",
                    severity="HIGH",
                    description=f"Potential LDAP injection detected in {field_name}",
                    matched_pattern=match.group(),
                    location=field_name,
                    recommendation="Use LDAP escaping functions",
                    cve_reference="CWE-90"
                )
                result.add_threat(threat)
                self.logger.warning(f"LDAP injection threat detected: {field_name}")
                break
    
    def _check_nosql_injection(self, value: str, field_name: str, result: SecurityValidationResult):
        """Check for NoSQL injection patterns."""
        for pattern in self.compiled_nosql_patterns:
            match = pattern.search(value)
            if match:
                threat = SecurityThreat(
                    type="NOSQL_INJECTION",
                    severity="HIGH",
                    description=f"Potential NoSQL injection detected in {field_name}",
                    matched_pattern=match.group(),
                    location=field_name,
                    recommendation="Validate input types and use proper query structure",
                    cve_reference="CWE-943"
                )
                result.add_threat(threat)
                self.logger.warning(f"NoSQL injection threat detected: {field_name}")
                break
    
    def _check_xxe(self, value: str, field_name: str, result: SecurityValidationResult):
        """Check for XXE attack patterns."""
        for pattern in self.compiled_xxe_patterns:
            match = pattern.search(value)
            if match:
                threat = SecurityThreat(
                    type="XXE",
                    severity="HIGH",
                    description=f"Potential XXE attack detected in {field_name}",
                    matched_pattern=match.group(),
                    location=field_name,
                    recommendation="Disable external entity processing in XML parsers",
                    cve_reference="CWE-611"
                )
                result.add_threat(threat)
                self.logger.warning(f"XXE threat detected: {field_name}")
                break
    
    def validate_file_upload(self, filename: str, content: bytes, 
                           max_size: int = 10 * 1024 * 1024,  # 10MB
                           allowed_extensions: Optional[Set[str]] = None) -> SecurityValidationResult:
        """Validate file upload for security threats.
        
        Args:
            filename: Name of uploaded file
            content: File content
            max_size: Maximum allowed file size
            allowed_extensions: Set of allowed file extensions
            
        Returns:
            SecurityValidationResult with file validation results
        """
        result = SecurityValidationResult()
        
        # Sanitize filename
        sanitized_filename = self._sanitize_filename(filename)
        result.sanitized_value = sanitized_filename
        
        # Check file extension
        file_ext = Path(sanitized_filename).suffix.lower()
        if file_ext in self.DANGEROUS_FILE_EXTENSIONS:
            threat = SecurityThreat(
                type="MALICIOUS_FILE_TYPE",
                severity="HIGH",
                description=f"Dangerous file extension detected: {file_ext}",
                location="filename",
                recommendation="Only allow safe file types for upload"
            )
            result.add_threat(threat)
        
        if allowed_extensions and file_ext not in allowed_extensions:
            threat = SecurityThreat(
                type="UNAUTHORIZED_FILE_TYPE",
                severity="MEDIUM",
                description=f"File extension not in allowlist: {file_ext}",
                location="filename",
                recommendation="Use strict allowlist for file types"
            )
            result.add_threat(threat)
        
        # Check file size
        if len(content) > max_size:
            threat = SecurityThreat(
                type="FILE_SIZE_LIMIT",
                severity="MEDIUM",
                description=f"File size {len(content)} exceeds limit {max_size}",
                location="file_content",
                recommendation="Enforce file size limits"
            )
            result.add_threat(threat)
        
        # Check for embedded threats in content
        content_str = content.decode('utf-8', errors='ignore')[:10000]  # First 10KB
        content_result = self.validate_against_injection(
            content_str, "file_content",
            check_xss=True, check_command=True, check_xxe=True
        )
        
        # Merge content validation results
        result.threats.extend(content_result.threats)
        if not content_result.is_valid:
            result.is_valid = False
        
        return result
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security."""
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', '_', filename)
        
        # Remove leading dots
        filename = filename.lstrip('.')
        
        # Ensure not empty
        if not filename:
            filename = f"file_{int(datetime.now().timestamp())}"
        
        # Limit length
        if len(filename) > 255:
            name, ext = Path(filename).stem[:200], Path(filename).suffix
            filename = name + ext
        
        return filename


class AntiInjectionValidator:
    """Specialized validator for preventing injection attacks."""
    
    def __init__(self):
        """Initialize anti-injection validator."""
        self.security_validator = SecurityValidator()
        self.logger = logger.getChild(self.__class__.__name__)
    
    def validate_sql_query_params(self, params: Dict[str, Any]) -> SecurityValidationResult:
        """Validate SQL query parameters for injection attacks.
        
        Args:
            params: Dictionary of query parameters
            
        Returns:
            SecurityValidationResult
        """
        result = SecurityValidationResult()
        sanitized_params = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                param_result = self.security_validator.validate_against_injection(
                    value, f"param.{key}", 
                    check_sql=True, check_xss=False, check_command=False
                )
                
                # Merge results
                result.threats.extend(param_result.threats)
                if not param_result.is_valid:
                    result.is_valid = False
                
                sanitized_params[key] = param_result.sanitized_value
            else:
                sanitized_params[key] = value
        
        result.sanitized_value = sanitized_params
        return result
    
    def validate_html_input(self, html_content: str, field_name: str) -> SecurityValidationResult:
        """Validate HTML content for XSS and other threats.
        
        Args:
            html_content: HTML content to validate
            field_name: Name of the field
            
        Returns:
            SecurityValidationResult with sanitized HTML
        """
        result = SecurityValidationResult()
        
        # Check for XSS
        xss_result = self.security_validator.validate_against_injection(
            html_content, field_name, 
            check_xss=True, check_sql=False, check_command=False
        )
        
        result.threats.extend(xss_result.threats)
        if not xss_result.is_valid:
            result.is_valid = False
        
        # Sanitize HTML if bleach is available
        if BLEACH_AVAILABLE:
            allowed_tags = [
                'p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 
                'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
            ]
            allowed_attributes = {
                'a': ['href', 'title'],
                '*': ['class']
            }
            
            sanitized_html = bleach.clean(
                html_content,
                tags=allowed_tags,
                attributes=allowed_attributes,
                strip=True
            )
            result.sanitized_value = sanitized_html
        else:
            # Fallback: escape HTML
            result.sanitized_value = html.escape(html_content)
        
        return result
    
    def validate_json_input(self, json_data: Union[str, Dict], field_name: str) -> SecurityValidationResult:
        """Validate JSON input for injection attacks.
        
        Args:
            json_data: JSON data as string or dict
            field_name: Name of the field
            
        Returns:
            SecurityValidationResult
        """
        result = SecurityValidationResult()
        
        # Parse JSON if string
        if isinstance(json_data, str):
            try:
                parsed_data = json.loads(json_data)
                result.sanitized_value = parsed_data
            except json.JSONDecodeError as e:
                result.add_error(f"Invalid JSON in {field_name}: {e}")
                return result
        else:
            parsed_data = json_data
            result.sanitized_value = parsed_data
        
        # Recursively validate JSON values
        self._validate_json_recursive(parsed_data, field_name, result)
        
        return result
    
    def _validate_json_recursive(self, data: Any, field_path: str, result: SecurityValidationResult):
        """Recursively validate JSON data."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{field_path}.{key}"
                self._validate_json_recursive(value, new_path, result)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{field_path}[{i}]"
                self._validate_json_recursive(item, new_path, result)
        elif isinstance(data, str):
            # Validate string values for injection
            str_result = self.security_validator.validate_against_injection(
                data, field_path,
                check_sql=True, check_xss=True, check_nosql=True
            )
            result.threats.extend(str_result.threats)
            if not str_result.is_valid:
                result.is_valid = False


class InputSanitizer:
    """Advanced input sanitization for security."""
    
    def __init__(self):
        """Initialize input sanitizer."""
        self.logger = logger.getChild(self.__class__.__name__)
    
    def sanitize_for_sql(self, value: str) -> str:
        """Sanitize input for SQL queries.
        
        Args:
            value: Input value
            
        Returns:
            Sanitized value safe for SQL
        """
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Escape single quotes
        value = value.replace("'", "''")
        
        # Remove SQL comments
        value = re.sub(r'(--[^\r\n]*|\/\*.*?\*\/)', '', value, flags=re.DOTALL)
        
        # Remove dangerous keywords at word boundaries
        dangerous_keywords = [
            'UNION', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP',
            'ALTER', 'CREATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE'
        ]
        
        for keyword in dangerous_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            value = re.sub(pattern, '', value, flags=re.IGNORECASE)
        
        return value.strip()
    
    def sanitize_for_html(self, value: str) -> str:
        """Sanitize input for HTML output.
        
        Args:
            value: Input value
            
        Returns:
            HTML-escaped value
        """
        if not isinstance(value, str):
            return str(value)
        
        # HTML escape
        value = html.escape(value, quote=True)
        
        # Additional JavaScript protocol prevention
        value = re.sub(r'javascript\s*:', '', value, flags=re.IGNORECASE)
        value = re.sub(r'vbscript\s*:', '', value, flags=re.IGNORECASE)
        value = re.sub(r'data\s*:', '', value, flags=re.IGNORECASE)
        
        return value
    
    def sanitize_for_shell(self, value: str) -> str:
        """Sanitize input for shell commands.
        
        Args:
            value: Input value
            
        Returns:
            Shell-escaped value
        """
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Remove shell metacharacters
        dangerous_chars = set(';&|`$(){}[]\\*?<>"\'')
        value = ''.join(char if char not in dangerous_chars else '' for char in value)
        
        # Remove newlines and carriage returns
        value = value.replace('\n', '').replace('\r', '')
        
        return value.strip()
    
    def sanitize_for_regex(self, value: str) -> str:
        """Sanitize input for regex patterns.
        
        Args:
            value: Input value
            
        Returns:
            Regex-escaped value
        """
        if not isinstance(value, str):
            return str(value)
        
        # Escape regex special characters
        return re.escape(value)
    
    def sanitize_for_ldap(self, value: str) -> str:
        """Sanitize input for LDAP queries.
        
        Args:
            value: Input value
            
        Returns:
            LDAP-escaped value
        """
        if not isinstance(value, str):
            return str(value)
        
        # LDAP special characters to escape
        escape_chars = {
            '\\': '\\5c',
            '*': '\\2a',
            '(': '\\28',
            ')': '\\29',
            '\x00': '\\00',
            '/': '\\2f'
        }
        
        for char, escaped in escape_chars.items():
            value = value.replace(char, escaped)
        
        return value
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem operations.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        if not isinstance(filename, str):
            filename = str(filename)
        
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        dangerous_chars = '<>:"|?*\x00'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure not empty
        if not filename:
            filename = 'sanitized_file'
        
        # Limit length
        if len(filename) > 255:
            name, ext = Path(filename).stem[:200], Path(filename).suffix
            filename = name + ext
        
        return filename
    
    def sanitize_url(self, url: str) -> str:
        """Sanitize URL for safe usage.
        
        Args:
            url: Original URL
            
        Returns:
            Sanitized URL
        """
        if not isinstance(url, str):
            return str(url)
        
        # Remove dangerous protocols
        dangerous_protocols = ['javascript:', 'vbscript:', 'data:', 'file:']
        url_lower = url.lower()
        
        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                return ''
        
        # URL encode special characters
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme and parsed.scheme not in ['http', 'https', 'ftp', 'ftps']:
                return ''
            
            # Reconstruct URL with proper encoding
            safe_url = urllib.parse.urlunparse(parsed)
            return safe_url
        except Exception:
            return ''
    
    def sanitize_log_message(self, message: str) -> str:
        """Sanitize log message to prevent log injection.
        
        Args:
            message: Original log message
            
        Returns:
            Sanitized log message
        """
        if not isinstance(message, str):
            message = str(message)
        
        # Remove control characters except newlines and tabs
        message = ''.join(char for char in message if ord(char) >= 32 or char in '\n\t')
        
        # Remove CRLF injection attempts
        message = message.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1b\[[0-9;]*[mGKH]')
        message = ansi_escape.sub('', message)
        
        # Limit length
        if len(message) > 1000:
            message = message[:997] + '...'
        
        return message.strip()