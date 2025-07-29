.PHONY: help install install-dev test test-cov lint format security clean build docker

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development setup
install: ## Install production dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

# Testing
test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=gaudi3_scale --cov-report=html --cov-report=term

test-integration: ## Run integration tests
	pytest tests/integration/ -v -m "integration"

# Code quality
lint: ## Run linting
	pre-commit run --all-files

format: ## Format code
	black src/ tests/
	isort src/ tests/

security: ## Run security checks
	bandit -r src/
	safety check

# Build and packaging
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

# Docker
docker-build: ## Build Docker image
	docker build -t gaudi3-scale:latest .

docker-dev: ## Build development Docker image
	docker build --target development -t gaudi3-scale:dev .

docker-run: ## Run Docker container
	docker-compose up -d

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-stop: ## Stop Docker containers
	docker-compose down

# Monitoring
monitor-up: ## Start monitoring stack
	docker-compose -f docker-compose.yml up prometheus grafana -d

monitor-down: ## Stop monitoring stack
	docker-compose down prometheus grafana

# Development helpers
serve-docs: ## Serve documentation locally
	python -m http.server 8080 --directory docs/

check-deps: ## Check for outdated dependencies
	pip list --outdated

update-deps: ## Update dependencies (use with caution)
	pip-review --local --auto