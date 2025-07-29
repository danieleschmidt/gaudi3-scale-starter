# GitHub Actions Workflow Setup

**IMPORTANT**: Due to GitHub security restrictions, workflow files cannot be created automatically. Please manually create these workflow files after merging this PR.

## Required GitHub Actions Workflows

### 1. Continuous Integration Workflow

**File**: `.github/workflows/ci.yml`

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly dependency check

env:
  PYTHON_VERSION: '3.10'
  POETRY_VERSION: '1.7.1'

jobs:
  lint-and-format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
      
      - name: Run pre-commit hooks
        run: |
          pre-commit run --all-files
  
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install security tools
        run: |
          python -m pip install --upgrade pip
          pip install safety bandit
      
      - name: Run Bandit security scan
        run: bandit -r src/ -f json -o bandit-report.json
      
      - name: Run Safety dependency scan
        run: safety check --json --output safety-report.json
      
      - name: Upload security artifacts
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-reports
          path: '*-report.json'
  
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests with coverage
        run: |
          pytest --cov=gaudi3_scale --cov-report=xml --cov-report=html
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
  
  build-and-test-package:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install build tools
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Check package
        run: twine check dist/*
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist-files
          path: dist/
```

### 2. Release Workflow

**File**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

env:
  PYTHON_VERSION: '3.10'

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write
      id-token: write
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Generate changelog
        id: changelog
        run: |
          # Simple changelog generation from git log
          echo "## Changes in ${GITHUB_REF_NAME}" > CHANGELOG.md
          git log --pretty=format:"- %s" $(git describe --tags --abbrev=0 HEAD^)..HEAD >> CHANGELOG.md
          echo "CHANGELOG<<EOF" >> $GITHUB_OUTPUT
          cat CHANGELOG.md >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT
      
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref_name }}
          release_name: Release ${{ github.ref_name }}
          body: ${{ steps.changelog.outputs.CHANGELOG }}
          draft: false
          prerelease: ${{ contains(github.ref_name, 'alpha') || contains(github.ref_name, 'beta') || contains(github.ref_name, 'rc') }}
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          twine upload dist/*
```

## Setup Instructions

### Step 1: Create Workflow Files

1. Navigate to your repository on GitHub
2. Create `.github/workflows/` directory if it doesn't exist
3. Create `ci.yml` and `release.yml` files with the content above
4. Commit and push the workflow files

### Step 2: Configure Repository Secrets

Add these secrets in your GitHub repository settings:

1. Go to **Settings** > **Secrets and variables** > **Actions**
2. Add the following repository secrets:

   - `CODECOV_TOKEN`: Your Codecov token for coverage reporting
   - `PYPI_API_TOKEN`: Your PyPI API token for automated releases

### Step 3: Enable Dependabot

The Dependabot configuration is already included in this PR. It will:
- Automatically check for dependency updates weekly
- Create pull requests for security updates
- Monitor GitHub Actions for updates

### Step 4: Verify Workflow Execution

1. Push a commit to trigger the CI workflow
2. Check the **Actions** tab in your repository
3. Verify all jobs complete successfully
4. Review security scan results

## Workflow Features

### CI Pipeline Includes:
- **Code Quality**: Pre-commit hooks, linting, formatting
- **Security Scanning**: Bandit (code) and Safety (dependencies)
- **Multi-Python Testing**: Python 3.10, 3.11, 3.12
- **Coverage reporting**: Integrated with Codecov
- **Package building**: Validates distribution packages

### Release Pipeline Includes:
- **Automated releases**: Triggered by version tags
- **Changelog generation**: From git commit history
- **PyPI publishing**: Automated package distribution
- **GitHub releases**: With generated release notes

## Troubleshooting

### Common Issues:

1. **Workflow doesn't trigger**
   - Check branch protection rules
   - Verify workflow file syntax
   - Check repository permissions

2. **Security scan failures**
   - Review Bandit and Safety reports
   - Update dependencies with known vulnerabilities
   - Add security exemptions if needed

3. **Test failures**
   - Check test compatibility across Python versions
   - Verify all dependencies are installed
   - Review test environment setup

4. **Release failures**
   - Verify PyPI token is valid
   - Check package build process
   - Review changelog generation

## Manual Setup Alternative

If you prefer to set up workflows manually:

```bash
# Create workflow directory
mkdir -p .github/workflows

# Copy workflow files from this documentation
cp docs/workflows/github-actions-setup.md .github/workflows/

# Extract and create actual workflow files
# (copy the YAML content above into separate files)
```

This setup provides comprehensive CI/CD automation while maintaining security and following GitHub best practices.
