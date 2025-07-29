# Security Hardening Guide

Comprehensive security hardening for Gaudi 3 Scale production deployments.

## Infrastructure Security

### Container Security

#### Docker Security Best Practices

```dockerfile
# Use specific, minimal base images
FROM vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habana-torch:latest as base

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set resource limits
LABEL max-memory="32G"
LABEL max-cpu="8"

# Remove unnecessary packages
RUN apt-get update && apt-get remove -y \
    wget \
    curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Use specific package versions
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    pytorch-lightning==2.2.0

# Set security-focused environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Switch to non-root user
USER appuser
```

#### Docker Compose Security

```yaml
version: '3.8'

services:
  gaudi-trainer:
    # Use read-only root filesystem
    read_only: true
    
    # Drop all capabilities and add only required ones
    cap_drop:
      - ALL
    cap_add:
      - SYS_NICE  # For process scheduling
    
    # Security options
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    
    # Resource limits
    deploy:
      resources:
        limits:
          memory: 32G
          cpus: '8'
    
    # Tmpfs for writable directories
    tmpfs:
      - /tmp:noexec,nosuid,size=1G
      - /var/tmp:noexec,nosuid,size=1G
    
    # Restricted networks
    networks:
      - gaudi-internal

networks:
  gaudi-internal:
    driver: bridge
    internal: true  # No external access
```

### Network Security

#### Firewall Configuration

```bash
#!/bin/bash
# scripts/setup_firewall.sh

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (change port as needed)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -j ACCEPT

# Allow Grafana (internal network only)
iptables -A INPUT -p tcp --dport 3000 -s 10.0.0.0/8 -j ACCEPT

# Allow Prometheus (internal network only)
iptables -A INPUT -p tcp --dport 9090 -s 10.0.0.0/8 -j ACCEPT

# Allow HPU metrics
iptables -A INPUT -p tcp --dport 9200 -s 127.0.0.1 -j ACCEPT

# Log dropped packets
iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables denied: " --log-level 7

# Save rules
iptables-save > /etc/iptables/rules.v4
```

#### TLS Configuration

```yaml
# nginx/ssl.conf
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
ssl_stapling on;
ssl_stapling_verify on;

# Security headers
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
add_header X-Content-Type-Options nosniff;
add_header X-Frame-Options DENY;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";
```

## Application Security

### Secrets Management

#### Environment Variables

```bash
#!/bin/bash
# scripts/setup_secrets.sh

# Create secrets directory
sudo mkdir -p /etc/gaudi3-scale/secrets
sudo chmod 700 /etc/gaudi3-scale/secrets

# Generate API keys
WANDB_API_KEY=$(openssl rand -hex 32)
PROMETHEUS_TOKEN=$(openssl rand -hex 32)

# Store secrets securely
echo "$WANDB_API_KEY" | sudo tee /etc/gaudi3-scale/secrets/wandb_api_key > /dev/null
echo "$PROMETHEUS_TOKEN" | sudo tee /etc/gaudi3-scale/secrets/prometheus_token > /dev/null

# Set permissions
sudo chmod 600 /etc/gaudi3-scale/secrets/*
sudo chown root:appuser /etc/gaudi3-scale/secrets/*
```

#### Docker Secrets

```yaml
# docker-compose.secrets.yml
version: '3.8'

services:
  gaudi-trainer:
    secrets:
      - wandb_api_key
      - prometheus_token
    environment:
      - WANDB_API_KEY_FILE=/run/secrets/wandb_api_key
      - PROMETHEUS_TOKEN_FILE=/run/secrets/prometheus_token

secrets:
  wandb_api_key:
    file: /etc/gaudi3-scale/secrets/wandb_api_key
  prometheus_token:
    file: /etc/gaudi3-scale/secrets/prometheus_token
```

### Input Validation

```python
# src/gaudi3_scale/security.py
import re
from typing import Any, Dict
from pydantic import BaseModel, validator

class TrainingConfig(BaseModel):
    """Secure training configuration with validation."""
    
    model_name: str
    batch_size: int
    learning_rate: float
    epochs: int
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name against allowed patterns."""
        pattern = r'^[a-zA-Z0-9_-]+$'
        if not re.match(pattern, v):
            raise ValueError('Model name contains invalid characters')
        if len(v) > 100:
            raise ValueError('Model name too long')
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Validate batch size is within reasonable bounds."""
        if v < 1 or v > 1024:
            raise ValueError('Batch size must be between 1 and 1024')
        return v
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        """Validate learning rate is positive and reasonable."""
        if v <= 0 or v > 1.0:
            raise ValueError('Learning rate must be between 0 and 1')
        return v

# Usage in CLI
def train_command(config_dict: Dict[str, Any]):
    """Secure training command with validation."""
    try:
        config = TrainingConfig(**config_dict)
        return start_training(config)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        raise
```

### Authentication and Authorization

```python
# src/gaudi3_scale/auth.py
import jwt
import hashlib
from datetime import datetime, timedelta
from functools import wraps

class AuthManager:
    """Authentication and authorization manager."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using PBKDF2."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        ).hex()
    
    def generate_token(self, user_id: str, roles: list) -> str:
        """Generate JWT token with expiration."""
        payload = {
            'user_id': user_id,
            'roles': roles,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token."""
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise ValueError('Token has expired')
        except jwt.InvalidTokenError:
            raise ValueError('Invalid token')

def require_auth(required_roles=None):
    """Decorator for API authentication."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return {'error': 'No token provided'}, 401
            
            try:
                # Remove 'Bearer ' prefix
                token = token.replace('Bearer ', '')
                payload = auth_manager.verify_token(token)
                
                # Check roles if required
                if required_roles:
                    user_roles = payload.get('roles', [])
                    if not any(role in user_roles for role in required_roles):
                        return {'error': 'Insufficient permissions'}, 403
                
                # Add user info to request
                request.user = payload
                return f(*args, **kwargs)
                
            except ValueError as e:
                return {'error': str(e)}, 401
        
        return decorated_function
    return decorator
```

## Compliance and Auditing

### Audit Logging

```python
# src/gaudi3_scale/audit.py
import json
import logging
from datetime import datetime
from typing import Dict, Any

class AuditLogger:
    """Structured audit logging for security events."""
    
    def __init__(self):
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler('/var/log/gaudi3-scale/audit.log')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log security audit event."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'details': details,
            'source_ip': self._get_source_ip(),
            'user_agent': self._get_user_agent()
        }
        
        self.logger.info(json.dumps(audit_entry))
    
    def log_authentication(self, user_id: str, success: bool, method: str):
        """Log authentication attempt."""
        self.log_event('authentication', user_id, {
            'success': success,
            'method': method
        })
    
    def log_authorization(self, user_id: str, resource: str, action: str, success: bool):
        """Log authorization check."""
        self.log_event('authorization', user_id, {
            'resource': resource,
            'action': action,
            'success': success
        })
    
    def log_data_access(self, user_id: str, dataset: str, operation: str):
        """Log data access events."""
        self.log_event('data_access', user_id, {
            'dataset': dataset,
            'operation': operation
        })
```

### Compliance Monitoring

```python
# scripts/compliance_check.py
import os
import subprocess
from typing import List, Dict

def check_security_compliance() -> Dict[str, bool]:
    """Run comprehensive security compliance checks."""
    checks = {
        'containers_run_as_non_root': check_non_root_containers(),
        'secrets_not_in_environment': check_secrets_management(),
        'tls_enabled': check_tls_configuration(),
        'firewall_configured': check_firewall_rules(),
        'logs_protected': check_log_permissions(),
        'updates_current': check_security_updates(),
        'audit_logging_enabled': check_audit_logging()
    }
    
    return checks

def check_non_root_containers() -> bool:
    """Verify containers run as non-root."""
    try:
        result = subprocess.run(
            ['docker', 'exec', 'gaudi3-trainer', 'whoami'],
            capture_output=True, text=True
        )
        return result.stdout.strip() != 'root'
    except:
        return False

def check_secrets_management() -> bool:
    """Check that secrets are not in environment variables."""
    sensitive_patterns = ['password', 'key', 'token', 'secret']
    
    try:
        result = subprocess.run(
            ['docker', 'exec', 'gaudi3-trainer', 'env'],
            capture_output=True, text=True
        )
        
        env_vars = result.stdout.lower()
        return not any(pattern in env_vars for pattern in sensitive_patterns)
    except:
        return False

def generate_compliance_report():
    """Generate compliance report."""
    checks = check_security_compliance()
    
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'compliance_score': sum(checks.values()) / len(checks) * 100,
        'checks': checks,
        'recommendations': []
    }
    
    # Add recommendations for failed checks
    if not checks['containers_run_as_non_root']:
        report['recommendations'].append(
            'Configure containers to run as non-root user'
        )
    
    if not checks['secrets_not_in_environment']:
        report['recommendations'].append(
            'Move secrets from environment variables to secure storage'
        )
    
    # Save report
    with open(f'compliance-report-{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report
```

## Security Monitoring

### Intrusion Detection

```bash
#!/bin/bash
# scripts/setup_ids.sh

# Install OSSEC HIDS
sudo apt-get update
sudo apt-get install -y ossec-hids

# Configure OSSEC
sudo tee /var/ossec/etc/ossec.conf << EOF
<ossec_config>
  <global>
    <email_notification>yes</email_notification>
    <logall>yes</logall>
  </global>
  
  <rules>
    <include>rules_config.xml</include>
    <include>sshd_rules.xml</include>
    <include>apache_rules.xml</include>
  </rules>
  
  <syscheck>
    <directories check_all="yes">/etc,/usr/bin,/usr/sbin</directories>
    <directories check_all="yes">/root/repo</directories>
  </syscheck>
  
  <rootcheck>
    <disabled>no</disabled>
  </rootcheck>
  
  <localfile>
    <log_format>syslog</log_format>
    <location>/var/log/auth.log</location>
  </localfile>
  
  <localfile>
    <log_format>syslog</log_format>
    <location>/var/log/gaudi3-scale/audit.log</location>
  </localfile>
EOF

# Start OSSEC
sudo systemctl enable ossec
sudo systemctl start ossec
```

### Vulnerability Scanning

```yaml
# .github/workflows/security-scan.yml
name: Security Scan

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  push:
    branches: [ main ]

jobs:
  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build image
        run: docker build -t gaudi3-scale:scan .
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'gaudi3-scale:scan'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
  
  code-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install bandit safety semgrep
      
      - name: Run Bandit security scan
        run: |
          bandit -r src/ -f json -o bandit-results.json
      
      - name: Run Safety scan
        run: |
          safety check --json --output safety-results.json
      
      - name: Run Semgrep scan
        run: |
          semgrep --config=auto src/ --json --output semgrep-results.json
```

## Incident Response Security

### Forensic Data Collection

```bash
#!/bin/bash
# scripts/collect_forensics.sh

FORENSIC_DIR="forensics-$(date +%Y%m%d-%H%M%S)"
mkdir -p $FORENSIC_DIR

echo "Collecting forensic data..."

# System state
ps aux > $FORENSIC_DIR/processes.txt
ss -tuln > $FORENSIC_DIR/network_connections.txt
lsof > $FORENSIC_DIR/open_files.txt

# File system changes
find / -mtime -1 -type f > $FORENSIC_DIR/recent_files.txt 2>/dev/null

# Docker state
docker ps -a > $FORENSIC_DIR/containers.txt
docker images > $FORENSIC_DIR/images.txt
docker network ls > $FORENSIC_DIR/networks.txt

# Logs
cp /var/log/auth.log* $FORENSIC_DIR/
cp /var/log/syslog* $FORENSIC_DIR/
cp /var/log/gaudi3-scale/* $FORENSIC_DIR/ 2>/dev/null

# Memory dump (if needed)
# dd if=/dev/mem of=$FORENSIC_DIR/memory.dump bs=1M

# Create hash manifest
find $FORENSIC_DIR -type f -exec sha256sum {} \; > $FORENSIC_DIR/hashes.txt

echo "Forensic data collected in $FORENSIC_DIR/"
```

## Security Training and Awareness

### Security Checklist for Developers

- [ ] Use parameterized queries to prevent SQL injection
- [ ] Validate all input data
- [ ] Never log sensitive information
- [ ] Use HTTPS for all external communications
- [ ] Implement proper error handling
- [ ] Keep dependencies updated
- [ ] Use strong authentication mechanisms
- [ ] Follow the principle of least privilege
- [ ] Review code for security vulnerabilities
- [ ] Test security controls regularly

### Security Code Review Guidelines

1. **Authentication and Authorization**
   - Verify authentication mechanisms
   - Check authorization controls
   - Review session management

2. **Input Validation**
   - Check all input validation
   - Verify output encoding
   - Review file upload handling

3. **Cryptography**
   - Verify encryption algorithms
   - Check key management
   - Review random number generation

4. **Error Handling**
   - Check error messages don't leak information
   - Verify logging doesn't expose secrets
   - Review exception handling

5. **Dependencies**
   - Check for known vulnerabilities
   - Verify dependency integrity
   - Review third-party library usage

This security hardening guide provides comprehensive protection for Gaudi 3 Scale deployments across infrastructure, applications, and operational processes.
