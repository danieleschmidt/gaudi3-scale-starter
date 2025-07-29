# GitHub Actions Workflows

This directory contains documentation and templates for GitHub Actions workflows to implement CI/CD for the Gaudi 3 Scale Starter project.

## Overview

The project uses a comprehensive CI/CD pipeline that includes:

- **Continuous Integration**: Automated testing, linting, and security scanning
- **Continuous Deployment**: Automated Docker image builds and infrastructure deployment
- **Release Management**: Automated semantic versioning and release creation
- **Security Scanning**: Dependency vulnerability scanning and secret detection

## Required Workflows

### 1. CI Pipeline (`ci.yml`)

**Purpose**: Run tests, linting, and security checks on every pull request and push.

**Triggers**:
- Pull requests to `main` branch
- Pushes to `main` branch
- Manual dispatch

**Jobs**:
- **Lint & Format**: Run black, isort, flake8, mypy
- **Test**: Run pytest with coverage reporting
- **Security**: Run bandit, safety, and GitGuardian scans
- **Terraform**: Validate and plan Terraform configurations

**Required Secrets**:
- `GITGUARDIAN_API_KEY`: For secret scanning
- `CODECOV_TOKEN`: For coverage reporting

### 2. Build Pipeline (`build.yml`)

**Purpose**: Build and publish Docker images and Python packages.

**Triggers**:
- Tags matching `v*.*.*`
- Manual dispatch

**Jobs**:
- **Build Package**: Build Python wheel and source distribution
- **Build Docker**: Build multi-arch Docker images for Gaudi 3
- **Publish**: Publish to PyPI and container registry

**Required Secrets**:
- `PYPI_API_TOKEN`: For PyPI publishing
- `DOCKER_HUB_USERNAME`: For Docker Hub
- `DOCKER_HUB_ACCESS_TOKEN`: For Docker Hub

### 3. Infrastructure Pipeline (`infrastructure.yml`)

**Purpose**: Deploy and manage cloud infrastructure using Terraform.

**Triggers**:
- Pushes to `main` branch (terraform/ directory changes)
- Manual dispatch with environment parameter

**Jobs**:
- **Plan**: Run terraform plan for infrastructure changes
- **Apply**: Apply approved terraform configurations
- **Destroy**: Clean up environments (manual only)

**Required Secrets**:
- `AWS_ACCESS_KEY_ID`: AWS credentials
- `AWS_SECRET_ACCESS_KEY`: AWS credentials
- `AZURE_CLIENT_ID`: Azure service principal
- `AZURE_CLIENT_SECRET`: Azure service principal
- `AZURE_TENANT_ID`: Azure tenant

### 4. Release Pipeline (`release.yml`)

**Purpose**: Automate semantic versioning and release creation.

**Triggers**:
- Pushes to `main` branch
- Manual dispatch

**Jobs**:
- **Release**: Generate changelog and create GitHub release
- **Notify**: Send notifications to Slack/Discord

**Required Secrets**:
- `GITHUB_TOKEN`: For release creation (automatically provided)
- `SLACK_WEBHOOK_URL`: For notifications (optional)

## Setup Instructions

### 1. Enable GitHub Actions

1. Go to repository Settings → Actions → General
2. Set "Actions permissions" to "Allow all actions and reusable workflows"
3. Enable "Allow GitHub Actions to create and approve pull requests"

### 2. Configure Secrets

Navigate to Settings → Secrets and variables → Actions and add:

**Required Secrets**:
```bash
GITGUARDIAN_API_KEY=<your-gitguardian-api-key>
CODECOV_TOKEN=<your-codecov-token>
PYPI_API_TOKEN=<your-pypi-token>
DOCKER_HUB_USERNAME=<your-dockerhub-username>
DOCKER_HUB_ACCESS_TOKEN=<your-dockerhub-token>
AWS_ACCESS_KEY_ID=<your-aws-key>
AWS_SECRET_ACCESS_KEY=<your-aws-secret>
```

**Optional Secrets**:
```bash
SLACK_WEBHOOK_URL=<your-slack-webhook>
AZURE_CLIENT_ID=<your-azure-client-id>
AZURE_CLIENT_SECRET=<your-azure-secret>
AZURE_TENANT_ID=<your-azure-tenant>
```

### 3. Configure Environments

Create the following environments in Settings → Environments:

- **development**: For development deployments
- **staging**: For staging deployments  
- **production**: For production deployments (with protection rules)

### 4. Branch Protection

Configure branch protection for `main`:

1. Go to Settings → Branches
2. Add rule for `main` branch:
   - Require pull request reviews
   - Require status checks (CI pipeline)
   - Require up-to-date branches
   - Include administrators

## Workflow Templates

See the `templates/` directory for complete workflow files that can be copied to `.github/workflows/` when ready to implement.

## Manual Setup Required

Since GitHub Actions workflows cannot be automatically committed, you must:

1. Create `.github/workflows/` directory in your repository
2. Copy templates from `docs/workflows/templates/` to `.github/workflows/`
3. Configure secrets as documented above
4. Test workflows with a small change

## Monitoring and Alerts

### Status Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/yourusername/gaudi3-scale-starter/workflows/CI/badge.svg)](https://github.com/yourusername/gaudi3-scale-starter/actions/workflows/ci.yml)
[![Build](https://github.com/yourusername/gaudi3-scale-starter/workflows/Build/badge.svg)](https://github.com/yourusername/gaudi3-scale-starter/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/yourusername/gaudi3-scale-starter/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/gaudi3-scale-starter)
```

### Failure Notifications

Configure Slack/Discord webhooks to receive notifications when:
- CI pipeline fails
- Deployment fails
- Security vulnerabilities are detected
- New releases are published

## Best Practices

1. **Security**: Never commit secrets or credentials
2. **Testing**: Ensure all tests pass before merging
3. **Dependencies**: Keep GitHub Actions versions pinned
4. **Caching**: Use action caching to speed up builds
5. **Parallelization**: Run independent jobs in parallel
6. **Environments**: Use environment-specific secrets and variables

## Troubleshooting

### Common Issues

1. **Permission denied**: Check repository secrets and GitHub token permissions
2. **Terraform fails**: Verify cloud provider credentials and permissions
3. **Tests fail**: Ensure all dependencies are properly mocked
4. **Docker build fails**: Check Dockerfile and build context
5. **Coverage too low**: Add tests or adjust coverage thresholds

### Getting Help

- Check GitHub Actions logs for detailed error messages
- Review workflow run artifacts for debugging information
- Consult GitHub Actions documentation
- Open an issue with workflow logs attached