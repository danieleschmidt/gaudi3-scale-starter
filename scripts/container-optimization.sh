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

# Container runtime optimization for production
optimize_runtime_performance() {
    log "Optimizing runtime performance for Gaudi 3 HPUs..."
    
    # System information
    log "System Information:"
    echo "  CPU cores: $(nproc)"
    echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
    echo "  Disk space: $(df -h / | tail -1 | awk '{print $4}')"
    
    # Memory optimization
    if [ -w /proc/sys/vm/swappiness ]; then
        echo 1 > /proc/sys/vm/swappiness
        log "Set swappiness to 1"
    else
        warn "Cannot modify swappiness (running as non-root)"
    fi
    
    # Set memory-related environment variables
    export MALLOC_TRIM_THRESHOLD_=100000
    export MALLOC_MMAP_THRESHOLD_=131072
    
    # HPU optimization
    export HPU_VISIBLE_DEVICES=${HPU_VISIBLE_DEVICES:-0}
    export PT_HPU_POOL_STRATEGY=OPTIMIZE_UTILIZATION
    export PT_HPU_EAGER_COLLECTIVE_OPS=${PT_HPU_EAGER_COLLECTIVE_OPS:-1}
    export PT_HPU_ENABLE_RECIPE_CACHE=${PT_HPU_ENABLE_RECIPE_CACHE:-1}
    
    # Python optimization
    export PYTHONOPTIMIZE=2
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
    
    log "Runtime performance optimization completed"
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