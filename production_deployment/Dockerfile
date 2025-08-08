# Production Dockerfile for Gaudi 3 Scale
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY standalone_simple_trainer.py .
COPY enhanced_simple_trainer.py .
COPY optimized_trainer.py .

# Create non-root user
RUN useradd -m -u 1000 gaudi3scale
RUN chown -R gaudi3scale:gaudi3scale /app
USER gaudi3scale

# Expose ports
EXPOSE 8080
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Default command
CMD ["python", "optimized_trainer.py"]
