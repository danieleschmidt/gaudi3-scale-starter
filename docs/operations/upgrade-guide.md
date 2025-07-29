# SDLC Enhancement Upgrade Guide

Step-by-step guide for implementing the new SDLC enhancements in your Gaudi 3 Scale deployment.

## Overview of Enhancements

This upgrade introduces comprehensive SDLC improvements:

### ✅ Added Components
- **CI/CD Workflows** - GitHub Actions for testing, security, and releases
- **Containerization** - Multi-stage Docker setup with security hardening  
- **Monitoring Stack** - Prometheus + Grafana with HPU-specific metrics
- **Security Hardening** - Comprehensive security controls and compliance
- **Performance Testing** - Benchmark suite for training performance
- **Incident Response** - Complete incident management procedures
- **Development Tooling** - Makefile, setup scripts, and dev environment

### 📊 Maturity Improvement
- **Previous State**: 65% SDLC maturity (Maturing)
- **Target State**: 85% SDLC maturity (Advanced)
- **Enhancement Level**: MATURING → ADVANCED

## Pre-Upgrade Checklist

- [ ] Backup current configuration
- [ ] Review existing workflows for conflicts
- [ ] Ensure Docker and Docker Compose are installed
- [ ] Verify GitHub repository permissions
- [ ] Check available disk space (>10GB recommended)
- [ ] Review security policies and compliance requirements

## Step-by-Step Upgrade Process

### Step 1: GitHub Actions Setup

1. **Enable GitHub Actions** (if not already enabled)
   ```bash
   # Verify Actions are enabled in repository settings
   # Settings > Actions > General > "Allow all actions and reusable workflows"
   ```

2. **Configure Repository Secrets**
   ```bash
   # Add these secrets in GitHub repository settings:
   # Settings > Secrets and variables > Actions
   
   CODECOV_TOKEN=<your_codecov_token>
   PYPI_API_TOKEN=<your_pypi_token>
   ```

3. **Test CI Pipeline**
   ```bash
   # Push changes and verify workflows run
   git add .
   git commit -m "feat: add comprehensive CI/CD workflows"
   git push origin main
   
   # Check workflow status
   gh workflow list
   gh workflow view ci.yml
   ```

### Step 2: Docker Environment Setup

1. **Build Docker Images**
   ```bash
   # Build production image
   docker build -t gaudi3-scale:latest .
   
   # Build development image
   docker build --target development -t gaudi3-scale:dev .
   
   # Verify builds
   docker images | grep gaudi3-scale
   ```

2. **Start Monitoring Stack**
   ```bash
   # Create monitoring directories
   mkdir -p monitoring/grafana/dashboards
   
   # Start monitoring services
   docker-compose up -d prometheus grafana
   
   # Verify services
   docker-compose ps
   curl http://localhost:9090  # Prometheus
   curl http://localhost:3000  # Grafana
   ```

3. **Configure Grafana Dashboards**
   ```bash
   # Access Grafana
   open http://localhost:3000
   # Login: admin / gaudi3admin
   
   # Import HPU monitoring dashboard
   # + > Import > Upload dashboard JSON
   ```

### Step 3: Development Environment Enhancement

1. **Set Up Development Environment**
   ```bash
   # Run setup script
   chmod +x scripts/setup_dev_env.sh
   ./scripts/setup_dev_env.sh
   
   # Verify installation
   source venv/bin/activate
   pytest --version
   pre-commit --version
   ```

2. **Configure Makefile Targets**
   ```bash
   # Test all make targets
   make help
   make install-dev
   make test
   make lint
   make security
   ```

3. **Performance Testing Setup**
   ```bash
   # Run performance benchmarks
   pytest tests/performance/ -v -m benchmark
   
   # Generate performance report
   pytest tests/performance/ --benchmark-only --benchmark-json=perf-report.json
   ```

### Step 4: Security Hardening Implementation

1. **Container Security**
   ```bash
   # Verify non-root user
   docker run --rm gaudi3-scale:latest whoami
   # Should output: appuser
   
   # Check security scan results
   docker run --rm -v $(pwd):/app aquasec/trivy:latest image gaudi3-scale:latest
   ```

2. **Network Security**
   ```bash
   # Configure firewall (optional, review first)
   sudo bash scripts/setup_firewall.sh
   
   # Verify network configuration
   docker network ls
   docker network inspect gaudi3-scale_gaudi-network
   ```

3. **Secrets Management**
   ```bash
   # Set up secrets (review script first)
   sudo bash scripts/setup_secrets.sh
   
   # Verify secrets are not in environment
   docker exec gaudi3-trainer env | grep -i secret
   # Should return no results
   ```

### Step 5: Monitoring and Alerting Setup

1. **HPU Metrics Configuration**
   ```bash
   # Verify HPU monitoring
   curl http://localhost:9200/hpu-metrics
   
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets
   ```

2. **Alert Configuration**
   ```bash
   # Copy alert rules
   cp monitoring/rules/*.yml prometheus/rules/
   
   # Reload Prometheus configuration
   docker exec gaudi3-prometheus kill -HUP 1
   ```

3. **Dashboard Import**
   ```bash
   # Access Grafana and import dashboards
   # Dashboards are auto-provisioned from monitoring/grafana/dashboards/
   ```

### Step 6: Incident Response Preparation

1. **Diagnostic Tools Setup**
   ```bash
   # Make scripts executable
   chmod +x scripts/collect_diagnostics.sh
   chmod +x scripts/collect_forensics.sh
   
   # Test diagnostic collection
   ./scripts/collect_diagnostics.sh
   ```

2. **Incident Response Training**
   ```bash
   # Review incident response procedures
   cat docs/operations/incident-response.md
   
   # Set up contact information
   # Edit docs/operations/incident-response.md with actual contacts
   ```

## Post-Upgrade Verification

### Automated Verification

```bash
#!/bin/bash
# scripts/verify_upgrade.sh

echo "Verifying SDLC enhancement upgrade..."

# Check CI/CD
echo "✓ Checking GitHub Actions..."
if gh workflow list | grep -q "Continuous Integration"; then
    echo "  ✅ CI workflow configured"
else
    echo "  ❌ CI workflow missing"
fi

# Check Docker
echo "✓ Checking Docker setup..."
if docker images | grep -q gaudi3-scale; then
    echo "  ✅ Docker images built"
else
    echo "  ❌ Docker images missing"
fi

# Check monitoring
echo "✓ Checking monitoring stack..."
if curl -s http://localhost:9090/api/v1/status/config > /dev/null; then
    echo "  ✅ Prometheus running"
else
    echo "  ❌ Prometheus not accessible"
fi

if curl -s http://localhost:3000/api/health | grep -q ok; then
    echo "  ✅ Grafana running"
else
    echo "  ❌ Grafana not accessible"
fi

# Check security
echo "✓ Checking security configuration..."
if docker run --rm gaudi3-scale:latest whoami | grep -q appuser; then
    echo "  ✅ Non-root container user"
else
    echo "  ❌ Container running as root"
fi

# Check development environment
echo "✓ Checking development setup..."
if [ -f "venv/bin/activate" ]; then
    echo "  ✅ Virtual environment created"
else
    echo "  ❌ Virtual environment missing"
fi

# Check performance tests
echo "✓ Checking performance tests..."
if [ -f "tests/performance/test_benchmarks.py" ]; then
    echo "  ✅ Performance tests available"
else
    echo "  ❌ Performance tests missing"
fi

echo "\nUpgrade verification complete!"
```

### Manual Verification Checklist

- [ ] **CI/CD Pipeline**: All GitHub Actions workflows run successfully
- [ ] **Docker Environment**: Containers start without errors
- [ ] **Monitoring**: Prometheus and Grafana accessible and showing data
- [ ] **Security**: Security scans pass, containers run as non-root
- [ ] **Development**: Pre-commit hooks work, tests pass
- [ ] **Performance**: Benchmark tests run successfully
- [ ] **Documentation**: All operational docs are accessible

## Rollback Procedures

If issues occur during upgrade:

### Quick Rollback

```bash
# Stop new services
docker-compose down

# Restore previous state
git checkout HEAD~1

# Rebuild if necessary
docker build -t gaudi3-scale:rollback .

# Restart with previous configuration
docker-compose up -d
```

### Selective Rollback

```bash
# Rollback specific components
git checkout HEAD~1 -- .github/workflows/
git checkout HEAD~1 -- docker-compose.yml
git checkout HEAD~1 -- Dockerfile

# Commit rollback
git commit -m "rollback: revert SDLC enhancements due to issues"
```

## Maintenance and Updates

### Regular Maintenance Tasks

1. **Weekly**
   - Review security scan results
   - Check monitoring alerts
   - Update dependencies via Dependabot

2. **Monthly**
   - Run compliance checks
   - Review incident reports
   - Update documentation

3. **Quarterly**
   - Security audit
   - Performance benchmarking
   - Tool and process review

### Upgrade Path to Advanced

To achieve 95%+ SDLC maturity (Advanced level):

1. **Add Advanced Security**
   - SIEM integration
   - Advanced threat detection
   - Zero-trust networking

2. **Enhance Automation**
   - Auto-scaling infrastructure
   - Intelligent alerting
   - Self-healing systems

3. **Add Advanced Monitoring**
   - Distributed tracing
   - Custom metrics
   - Predictive analytics

## Troubleshooting Common Issues

### Docker Issues

**Problem**: Container fails to start
```bash
# Check logs
docker logs gaudi3-trainer

# Check resource limits
docker stats

# Verify image
docker inspect gaudi3-scale:latest
```

**Problem**: HPU not accessible in container
```bash
# Check device mapping
docker run --rm --device=/dev/accel:/dev/accel gaudi3-scale:latest hl-smi

# Verify Habana drivers
hl-smi
```

### GitHub Actions Issues

**Problem**: Workflow fails
```bash
# Check workflow logs
gh run list
gh run view <run-id>

# Check secrets
echo "Verify CODECOV_TOKEN and PYPI_API_TOKEN are set"
```

### Monitoring Issues

**Problem**: Metrics not appearing
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check container connectivity
docker exec gaudi3-prometheus ping gaudi-trainer

# Verify metrics endpoint
curl http://localhost:8000/metrics
```

## Support and Resources

- **Documentation**: `/docs/` directory contains comprehensive guides
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Monitoring**: Access Grafana at http://localhost:3000
- **Metrics**: Access Prometheus at http://localhost:9090
- **Security**: Review `/docs/operations/security-hardening.md`
- **Incident Response**: Follow `/docs/operations/incident-response.md`

## Next Steps

After successful upgrade:

1. **Train Team**: Ensure all team members understand new processes
2. **Monitor Performance**: Track system performance and adjust as needed
3. **Iterate**: Continuously improve based on operational experience
4. **Plan Advanced Features**: Consider additional enhancements for full Advanced maturity

This upgrade transforms your Gaudi 3 Scale deployment from a developing project to a production-ready, enterprise-grade machine learning infrastructure with comprehensive SDLC practices.
