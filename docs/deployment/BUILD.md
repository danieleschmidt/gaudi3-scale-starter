# Build and Deployment Guide

This guide covers building, packaging, and deploying the Gaudi 3 Scale infrastructure.

## Build System Overview

The build system uses multiple tools to ensure consistent, secure, and optimized builds:

- **Docker**: Multi-stage builds for development and production
- **Make**: Development workflow automation
- **Semantic Release**: Automated versioning and publishing
- **GitHub Actions**: CI/CD pipeline automation

## Local Development Build

### Prerequisites

```bash
# Install development dependencies
make install-dev

# Verify installation
python -c "import gaudi3_scale; print(gaudi3_scale.__version__)"
```

### Development Workflow

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Run tests with coverage
make test-cov

# Security checks
make security
```

## Docker Builds

### Development Image

```bash
# Build development image
make docker-dev

# Run development container
docker run -it --rm \
  -v $(pwd):/workspace \
  gaudi3-scale:dev bash
```

### Production Image

```bash
# Build production image
make docker-build

# Run production container
docker run -it --rm \
  --device /dev/accel \
  gaudi3-scale:latest
```

### Multi-Architecture Builds

```bash
# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t gaudi3-scale:latest \
  --push .
```

## Container Optimization

### Image Size Optimization

The production Dockerfile uses several techniques to minimize image size:

1. **Multi-stage builds**: Separate development and production stages
2. **Layer optimization**: Group related commands to reduce layers
3. **Dependency pruning**: Only install required production dependencies
4. **Non-root user**: Run as non-privileged user for security

### Security Hardening

```bash
# Scan for vulnerabilities
docker scout cves gaudi3-scale:latest

# Run security analysis
trivy image gaudi3-scale:latest
```

## Package Build

### Python Package

```bash
# Clean previous builds
make clean

# Build package
make build

# Verify package
twine check dist/*
```

### Distribution

```bash
# Upload to PyPI (test)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

## Release Process

### Automated Release

The project uses semantic-release for automated versioning:

```bash
# Install semantic-release dependencies
npm install -g semantic-release

# Create release (CI only)
semantic-release
```

### Manual Release

```bash
# Update version
python scripts/update_version.py 0.2.0

# Create git tag
git tag v0.2.0

# Push tag
git push origin v0.2.0
```

## Deployment Strategies

### Docker Compose Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=gaudi3-scale

# View logs
kubectl logs -l app=gaudi3-scale -f
```

### Helm Deployment

```bash
# Install chart
helm install gaudi3-scale ./helm/gaudi3-scale

# Upgrade
helm upgrade gaudi3-scale ./helm/gaudi3-scale

# Uninstall
helm uninstall gaudi3-scale
```

## Environment Configuration

### Environment Variables

```bash
# Core configuration
export GAUDI3_SCALE_ENV=production
export GAUDI3_SCALE_LOG_LEVEL=info

# Database
export DATABASE_URL=postgresql://user:pass@host:5432/db

# Authentication
export JWT_SECRET_KEY=your-secret-key
export GITHUB_TOKEN=your-github-token

# Monitoring
export PROMETHEUS_URL=http://prometheus:9090
export GRAFANA_URL=http://grafana:3000
```

### Configuration Files

```bash
# Copy example config
cp configs/config.example.yaml configs/config.yaml

# Edit configuration
vim configs/config.yaml
```

## Monitoring and Health Checks

### Health Endpoints

```bash
# Application health
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/health/ready

# Liveness check
curl http://localhost:8000/health/live
```

### Metrics

```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Custom metrics
curl http://localhost:8000/api/v1/metrics
```

## Troubleshooting

### Common Build Issues

1. **Permission denied errors**
   ```bash
   # Fix permissions
   sudo chown -R $(id -u):$(id -g) /var/lib/docker
   ```

2. **Out of disk space**
   ```bash
   # Clean Docker
   docker system prune -a
   ```

3. **HPU device access**
   ```bash
   # Verify device access
   ls -la /dev/accel*
   
   # Add user to habana group
   sudo usermod -a -G habana $USER
   ```

### Build Performance

1. **Enable BuildKit**
   ```bash
   export DOCKER_BUILDKIT=1
   export COMPOSE_DOCKER_CLI_BUILD=1
   ```

2. **Use build cache**
   ```bash
   docker build --cache-from gaudi3-scale:latest .
   ```

3. **Parallel builds**
   ```bash
   make -j$(nproc) test
   ```

## Security Considerations

### Image Security

1. **Base image updates**: Regularly update base images
2. **Vulnerability scanning**: Use tools like Trivy or Snyk
3. **Minimal images**: Use distroless or Alpine-based images
4. **Non-root user**: Always run as non-privileged user

### Secret Management

1. **Environment variables**: Never include secrets in images
2. **Secret stores**: Use Kubernetes secrets or external stores
3. **Rotation**: Implement secret rotation policies

### Network Security

1. **Private registries**: Use private container registries
2. **Image signing**: Sign container images
3. **Network policies**: Implement Kubernetes network policies

## CI/CD Integration

### GitHub Actions

The project includes GitHub Actions workflows for:

- Automated testing
- Security scanning
- Container builds
- Deployment automation

### Pipeline Configuration

```yaml
# .github/workflows/build.yml
name: Build and Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: make install-dev
      - name: Run tests
        run: make test-cov
```

## Performance Optimization

### Build Cache

1. **Layer caching**: Optimize Dockerfile layer order
2. **Multi-stage builds**: Share layers between stages
3. **Build context**: Minimize build context size

### Runtime Optimization

1. **Resource limits**: Set appropriate CPU/memory limits
2. **Health checks**: Configure proper health check intervals
3. **Logging**: Optimize logging configuration

## Maintenance

### Regular Tasks

1. **Dependency updates**: Weekly dependency updates
2. **Security patches**: Apply security patches promptly
3. **Image rebuilds**: Rebuild images monthly
4. **Clean up**: Regular cleanup of old images and containers

### Monitoring

1. **Build metrics**: Monitor build times and success rates
2. **Image metrics**: Track image sizes and vulnerabilities
3. **Deployment metrics**: Monitor deployment success and rollback rates