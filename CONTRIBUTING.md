# Contributing to Gaudi 3 Scale Starter

Thank you for your interest in contributing to the Gaudi 3 Scale Starter project! This guide will help you get started with development and contribution processes.

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git
- Optional: Intel Gaudi hardware access (HPU simulator available for development)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/gaudi3-scale-starter.git
   cd gaudi3-scale-starter
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -r requirements-dev.txt

   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify setup**
   ```bash
   # Run tests
   pytest

   # Check code formatting
   black --check .
   isort --check-only .
   flake8

   # Type checking
   mypy src/
   ```

## 🎯 How to Contribute

### Areas for Contribution

1. **Model Optimization Recipes**
   - HPU-specific optimizations for popular models
   - Memory-efficient training strategies
   - Performance tuning guidelines

2. **Infrastructure Modules**
   - Multi-cloud Terraform configurations
   - Kubernetes deployment manifests
   - Monitoring and observability setups

3. **Documentation & Examples**
   - Tutorial notebooks for specific use cases
   - Best practices guides
   - Performance benchmarking results

4. **Tooling & CLI Enhancements**
   - Additional CLI commands and features
   - Integration with MLOps platforms
   - Cost optimization tools

### Contribution Process

1. **Check existing issues** or create a new one to discuss your proposed changes
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Run the full test suite**:
   ```bash
   pytest
   pre-commit run --all-files
   ```
7. **Submit a pull request** with a clear description

## 📝 Coding Standards

### Python Code Style

- **Formatter**: Black (line length: 88)
- **Import sorting**: isort
- **Linting**: flake8 with docstring requirements
- **Type hints**: Required for all public APIs
- **Documentation**: Google-style docstrings

### Example Code Style

```python
from typing import Optional, Dict, Any

import torch
import pytorch_lightning as pl


class GaudiOptimizer:
    """HPU-optimized optimizer wrapper.
    
    Args:
        base_optimizer: Base PyTorch optimizer
        use_habana: Enable Habana-specific optimizations
        
    Returns:
        Configured optimizer instance
    """
    
    def __init__(
        self, 
        base_optimizer: torch.optim.Optimizer,
        use_habana: bool = True
    ) -> None:
        self.base_optimizer = base_optimizer
        self.use_habana = use_habana
```

### Terraform Standards

- **Formatting**: `terraform fmt`
- **Validation**: `terraform validate`
- **Documentation**: Auto-generated with terraform-docs
- **Naming**: snake_case for resources and variables

### Commit Message Format

Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(training): add adaptive batch size finder`
- `fix(terraform): resolve EFA network configuration`
- `docs(guides): add multi-node scaling tutorial`

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Component integration tests
├── e2e/           # End-to-end system tests
└── fixtures/      # Test data and fixtures
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark critical paths
4. **Infrastructure Tests**: Validate Terraform configurations

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

from gaudi3_scale.training import GaudiTrainer


class TestGaudiTrainer:
    """Test suite for GaudiTrainer class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return Mock()
    
    def test_trainer_initialization(self, mock_model):
        """Test trainer initializes correctly."""
        trainer = GaudiTrainer(model=mock_model)
        assert trainer.model == mock_model
    
    @patch('gaudi3_scale.training.torch.hpu.device_count')
    def test_device_detection(self, mock_device_count, mock_model):
        """Test HPU device detection."""
        mock_device_count.return_value = 8
        trainer = GaudiTrainer(model=mock_model)
        assert trainer.num_devices == 8
```

## 📋 Pull Request Guidelines

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] Changelog entry added (if applicable)
- [ ] All CI checks pass

### PR Description Template

```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested your changes

## Performance Impact
Any performance implications

## Screenshots/Examples
If applicable, add screenshots or usage examples
```

## 🏗️ Development Workflow

### Local Development

1. **Feature development**:
   ```bash
   git checkout -b feature/new-feature
   # Make changes
   pytest tests/
   pre-commit run --all-files
   ```

2. **Documentation updates**:
   ```bash
   # Start docs server
   mkdocs serve
   # Edit docs in docs/ directory
   ```

3. **Infrastructure testing**:
   ```bash
   # Validate Terraform
   cd terraform/
   terraform init
   terraform validate
   terraform plan
   ```

### Release Process

1. Version bumping follows semantic versioning
2. Releases are automated via GitHub Actions
3. Changelog is auto-generated from conventional commits
4. Docker images are built and published automatically

## 🤝 Community Guidelines

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and inclusive in all interactions.

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and community chat
- **Documentation**: Comprehensive guides at [docs site]
- **Slack**: Real-time community support

### Recognition

Contributors are recognized in:
- README.md acknowledgments
- Release notes
- Annual contributor highlights
- Conference presentations (with permission)

## 📚 Additional Resources

- [Intel Gaudi Developer Guide](https://docs.habana.ai/)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Terraform Best Practices](https://www.terraform.io/docs/extend/best-practices/)
- [Performance Optimization Guide](docs/guides/performance-optimization.md)

## 🎉 Thank You!

Your contributions help make AI infrastructure more accessible and efficient. Every contribution, no matter how small, is valuable to the community!