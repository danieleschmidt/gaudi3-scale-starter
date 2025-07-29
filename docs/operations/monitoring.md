# Monitoring and Observability

Comprehensive monitoring setup for Gaudi 3 Scale training infrastructure.

## Overview

Our monitoring stack includes:
- **Prometheus** for metrics collection
- **Grafana** for visualization and alerting
- **Custom HPU metrics** for Gaudi-specific monitoring
- **Application metrics** for training performance

## Quick Start

```bash
# Start monitoring stack
make monitor-up

# Access Grafana dashboard
open http://localhost:3000
# Login: admin / gaudi3admin

# Access Prometheus
open http://localhost:9090
```

## Metrics Collection

### HPU Metrics

Key Gaudi 3 metrics we monitor:

```python
# HPU utilization
hpu_utilization_percent

# Memory usage
hpu_memory_used_bytes
hpu_memory_total_bytes

# Temperature monitoring
hpu_temperature_celsius

# Power consumption
hpu_power_watts

# Training throughput
training_samples_per_second
training_tokens_per_second
```

### Application Metrics

```python
# Training metrics
training_loss
training_accuracy
validation_loss
validation_accuracy

# Performance metrics
batch_processing_time
model_forward_time
backward_pass_time
optimizer_step_time

# Resource utilization
cpu_utilization_percent
memory_usage_bytes
disk_io_bytes
network_io_bytes
```

## Dashboard Configuration

### Grafana Dashboards

1. **HPU Monitoring Dashboard**
   - Real-time HPU utilization
   - Memory usage trends
   - Temperature alerts
   - Power consumption

2. **Training Performance Dashboard**
   - Loss curves
   - Accuracy metrics
   - Throughput monitoring
   - Batch processing times

3. **Infrastructure Dashboard**
   - System resource usage
   - Network performance
   - Storage utilization
   - Container health

### Custom Metrics Exporters

```python
# HPU metrics exporter
from prometheus_client import Gauge, Counter, start_http_server

# Define metrics
hpu_utilization = Gauge('hpu_utilization_percent', 'HPU utilization percentage')
training_samples = Counter('training_samples_total', 'Total training samples processed')

# Export metrics
def export_hpu_metrics():
    """Export HPU-specific metrics to Prometheus."""
    import habana_frameworks.torch as htorch
    
    if htorch.hpu.is_available():
        # Get HPU utilization (mock implementation)
        utilization = get_hpu_utilization()
        hpu_utilization.set(utilization)
        
        # Update sample counter
        training_samples.inc(batch_size)

# Start metrics server
start_http_server(9200)
```

## Alerting Rules

### Critical Alerts

```yaml
# High HPU temperature
- alert: HPUHighTemperature
  expr: hpu_temperature_celsius > 85
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "HPU temperature too high"
    description: "HPU temperature {{ $value }}°C exceeds 85°C threshold"

# Low HPU utilization
- alert: HPULowUtilization
  expr: hpu_utilization_percent < 20
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "HPU utilization is low"
    description: "HPU utilization {{ $value }}% is below 20% for 10 minutes"

# Training stalled
- alert: TrainingStalled
  expr: rate(training_samples_total[5m]) == 0
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Training appears to be stalled"
    description: "No training samples processed in the last 5 minutes"
```

### Warning Alerts

```yaml
# High memory usage
- alert: HighMemoryUsage
  expr: (hpu_memory_used_bytes / hpu_memory_total_bytes) > 0.9
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "HPU memory usage is high"
    description: "HPU memory usage {{ $value | humanizePercentage }} exceeds 90%"

# Slow batch processing
- alert: SlowBatchProcessing
  expr: batch_processing_time > 10
  for: 3m
  labels:
    severity: warning
  annotations:
    summary: "Batch processing is slow"
    description: "Batch processing time {{ $value }}s exceeds 10s threshold"
```

## Log Aggregation

### Structured Logging

```python
import logging
import json
from datetime import datetime

# Configure structured logging
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'hpu_id'):
            log_entry['hpu_id'] = record.hpu_id
        if hasattr(record, 'batch_idx'):
            log_entry['batch_idx'] = record.batch_idx
            
        return json.dumps(log_entry)

# Set up logger
logger = logging.getLogger('gaudi3_scale')
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage in training code
logger.info("Training started", extra={'hpu_id': 0, 'batch_idx': 1})
```

## Health Checks

### Application Health

```python
from flask import Flask, jsonify
import habana_frameworks.torch as htorch

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Application health check endpoint."""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {
            'hpu_available': htorch.hpu.is_available(),
            'hpu_count': htorch.hpu.device_count() if htorch.hpu.is_available() else 0,
            'memory_usage': get_memory_usage(),
            'disk_space': get_disk_space()
        }
    }
    
    # Determine overall health
    if not health_status['checks']['hpu_available']:
        health_status['status'] = 'unhealthy'
        return jsonify(health_status), 503
    
    return jsonify(health_status), 200

@app.route('/ready')
def readiness_check():
    """Readiness check for training."""
    ready_status = {
        'ready': True,
        'checks': {
            'model_loaded': check_model_loaded(),
            'data_available': check_data_available(),
            'hpu_initialized': check_hpu_initialized()
        }
    }
    
    if not all(ready_status['checks'].values()):
        ready_status['ready'] = False
        return jsonify(ready_status), 503
    
    return jsonify(ready_status), 200
```

### Infrastructure Health

```bash
#!/bin/bash
# infrastructure_health.sh

# Check HPU devices
echo "Checking HPU devices..."
if command -v hl-smi &> /dev/null; then
    hl-smi | grep -E "(Device|Memory|Temperature)"
else
    echo "hl-smi not available"
fi

# Check Docker containers
echo "Checking Docker containers..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check disk space
echo "Checking disk space..."
df -h | grep -E "(Filesystem|/dev/)"

# Check network connectivity
echo "Checking network connectivity..."
ping -c 1 google.com > /dev/null && echo "Network: OK" || echo "Network: FAIL"
```

## Performance Monitoring

### Training Metrics

```python
class TrainingMonitor:
    """Monitor training performance metrics."""
    
    def __init__(self):
        self.start_time = None
        self.samples_processed = 0
        self.tokens_processed = 0
        
    def start_epoch(self):
        """Start monitoring an epoch."""
        self.start_time = time.time()
        
    def update_batch(self, batch_size, sequence_length):
        """Update metrics after processing a batch."""
        self.samples_processed += batch_size
        self.tokens_processed += batch_size * sequence_length
        
    def get_throughput(self):
        """Calculate current throughput metrics."""
        if self.start_time is None:
            return None
            
        elapsed_time = time.time() - self.start_time
        
        return {
            'samples_per_second': self.samples_processed / elapsed_time,
            'tokens_per_second': self.tokens_processed / elapsed_time,
            'elapsed_time': elapsed_time
        }
```

## Troubleshooting

### Common Issues

1. **High HPU Temperature**
   - Check cooling system
   - Reduce batch size
   - Lower learning rate

2. **Low HPU Utilization**
   - Increase batch size
   - Check data loading bottlenecks
   - Optimize model graph compilation

3. **Memory Issues**
   - Enable gradient checkpointing
   - Reduce model size or batch size
   - Use mixed precision training

4. **Slow Training**
   - Profile training loop
   - Check data loading pipeline
   - Optimize network I/O

### Debug Commands

```bash
# Check HPU status
hl-smi

# Monitor system resources
htop
iotop

# Check Docker logs
docker logs gaudi3-trainer

# Monitor network
netstat -i
ss -tuln

# Check disk I/O
iostat -x 1
```

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Habana Developer Documentation](https://docs.habana.ai/)
- [Docker Compose Monitoring](https://docs.docker.com/compose/compose-file/)
