# Multi-stage Dockerfile for Gaudi 3 Scale production deployment
FROM vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habana-torch:latest as base

# Set environment variables for Habana optimizations
ENV PT_HPU_LAZY_MODE=1 \
    PT_HPU_ENABLE_LAZY_COMPILATION=1 \
    PT_HPU_GRAPH_COMPILER_OPT_LEVEL=3 \
    PT_HPU_MAX_COMPOUND_OP_SIZE=256 \
    PT_HPU_ENABLE_SYNAPSE_LAYOUT_OPT=1 \
    PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=1 \
    PT_HPU_POOL_STRATEGY=OPTIMIZE_UTILIZATION

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base as development

WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Production stage
FROM base as production

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml README.md ./

# Install package
RUN pip install --no-cache-dir .

# Change ownership and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import gaudi3_scale; print('Health check passed')"

# Default command
CMD ["gaudi3-train", "--help"]