#!/bin/bash
# Container Optimization Script for Gaudi 3 Scale Starter
# Implements advanced container performance and security optimizations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Container Registry Configuration
REGISTRY="${CONTAINER_REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-gaudi3-scale-starter}"
TAG="${TAG:-latest}"

# Performance Optimization
optimize_container_performance() {
    log "Optimizing container performance for Gaudi 3 HPUs..."
    
    # Build optimized image with performance flags
    docker build \
        --build-arg HABANA_VERSION=1.16.0 \
        --build-arg PYTHON_VERSION=3.10 \
        --build-arg TORCH_VERSION=2.3.0 \
        --build-arg PYTORCH_LIGHTNING_VERSION=2.2.0 \
        --target production \
        --platform linux/amd64 \
        -t "${REGISTRY}/${IMAGE_NAME}:${TAG}" \
        -t "${REGISTRY}/${IMAGE_NAME}:$(git rev-parse --short HEAD)" \
        -f Dockerfile.optimized .
    
    log "Container performance optimization completed"
}

# Security Hardening
secure_container() {
    log "Applying container security hardening..."
    
    # Scan for vulnerabilities
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL "${REGISTRY}/${IMAGE_NAME}:${TAG}"
    else
        warn "Trivy not installed. Skipping vulnerability scan."
    fi
    
    # Apply security labels
    docker image inspect "${REGISTRY}/${IMAGE_NAME}:${TAG}" --format '{{json .Config.Labels}}' | \
        jq '.["org.opencontainers.image.security.scan"]'
    
    log "Container security hardening completed"
}

# Registry Automation
automate_registry_push() {
    log "Automating container registry operations..."
    
    # Login to registry
    if [[ -n "${DOCKER_PASSWORD:-}" ]]; then
        echo "${DOCKER_PASSWORD}" | docker login "${REGISTRY}" -u "${DOCKER_USERNAME}" --password-stdin
    else
        warn "Docker credentials not provided. Manual login required."
    fi
    
    # Push with multiple tags
    docker push "${REGISTRY}/${IMAGE_NAME}:${TAG}"
    docker push "${REGISTRY}/${IMAGE_NAME}:$(git rev-parse --short HEAD)"
    
    # Create latest tag for main branch
    if [[ "${GITHUB_REF:-}" == "refs/heads/main" ]]; then
        docker tag "${REGISTRY}/${IMAGE_NAME}:${TAG}" "${REGISTRY}/${IMAGE_NAME}:latest"
        docker push "${REGISTRY}/${IMAGE_NAME}:latest"
    fi
    
    log "Registry automation completed"
}

# Performance Monitoring Integration
setup_monitoring() {
    log "Setting up container performance monitoring..."
    
    # Create monitoring configuration
    cat > /tmp/container-monitoring.yml << EOF
version: '3.8'
services:
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.49.1
    container_name: cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    restart: unless-stopped
EOF
    
    log "Container monitoring setup completed"
}

# Main execution
main() {
    log "Starting container optimization process..."
    
    # Check prerequisites
    command -v docker >/dev/null 2>&1 || error "Docker not installed"
    command -v git >/dev/null 2>&1 || error "Git not installed"
    command -v jq >/dev/null 2>&1 || error "jq not installed"
    
    # Execute optimization steps
    optimize_container_performance
    secure_container
    setup_monitoring
    
    if [[ "${PUSH_TO_REGISTRY:-false}" == "true" ]]; then
        automate_registry_push
    fi
    
    log "Container optimization completed successfully!"
    
    # Display summary
    echo ""
    echo "=== Container Optimization Summary ==="
    echo "Registry: ${REGISTRY}"
    echo "Image: ${IMAGE_NAME}"
    echo "Tag: ${TAG}"
    echo "Commit: $(git rev-parse --short HEAD)"
    echo "Size: $(docker image inspect "${REGISTRY}/${IMAGE_NAME}:${TAG}" --format '{{.Size}}' | numfmt --to=iec-i --suffix=B)"
    echo "======================================="
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi