# GitHub Actions Workflows

This directory contains GitHub Actions workflow templates that can be manually copied to `.github/workflows/` directory.

## Production Deployment Workflow

**File**: `production-deployment.yml`

This workflow provides:
- Automated production deployment pipeline
- Multi-stage validation (security, testing, building)
- Blue-green deployment support
- Automated rollback capabilities
- Notification and monitoring integration

### Setup Instructions

1. Copy the workflow file to your repository:
   ```bash
   mkdir -p .github/workflows
   cp deployment/workflows/production-deployment.yml .github/workflows/
   ```

2. Configure the required secrets in your GitHub repository:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `KUBE_CONFIG_DATA`
   - `DOCKER_REGISTRY_TOKEN`
   - `SLACK_WEBHOOK_URL`

3. Update the workflow variables to match your environment:
   - Registry URLs
   - Kubernetes cluster names
   - Notification channels

4. Enable the workflow in your repository settings

### Features

- **Security Scanning**: SAST, dependency vulnerability scanning
- **Testing**: Unit tests, integration tests, performance tests
- **Building**: Multi-stage Docker builds with optimization
- **Deployment**: Kubernetes deployment with health checks
- **Monitoring**: Integration with monitoring and alerting systems
- **Rollback**: Automated rollback on deployment failures

The workflow follows industry best practices for CI/CD pipelines and includes comprehensive error handling and reporting.