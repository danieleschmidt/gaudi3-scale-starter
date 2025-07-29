# Performance Optimization Guide

## Overview

This comprehensive guide outlines performance optimization strategies for Gaudi3 Scale infrastructure, covering hardware utilization, software optimization, and system monitoring to achieve maximum throughput and efficiency.

## Hardware Optimization

### Gaudi3 HPU Configuration

#### Optimal Cluster Topology
```python
# Configuration for maximum throughput
OPTIMAL_CLUSTER_CONFIG = {
    "nodes": 8,  # Minimum for efficient distributed training
    "hpus_per_node": 8,  # Full Gaudi3 card utilization
    "interconnect": "RoCE",  # 200Gb Ethernet recommended
    "memory_per_hpu": "96GB",  # HBM2e memory
    "host_memory": "512GB",  # DDR4 for data preprocessing
}

# Environment variables for peak performance
HPU_OPTIMIZATION_ENV = {
    "PT_HPU_LAZY_MODE": "1",
    "PT_HPU_ENABLE_LAZY_COMPILATION": "1", 
    "PT_HPU_GRAPH_COMPILER_OPT_LEVEL": "3",
    "PT_HPU_MAX_COMPOUND_OP_SIZE": "256",
    "PT_HPU_ENABLE_SYNAPSE_LAYOUT_OPT": "1",
    "PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE": "1",
    "PT_HPU_POOL_STRATEGY": "OPTIMIZE_UTILIZATION",
}
```

#### Memory Management Optimization
```python
# scripts/optimize_memory.py
class HPUMemoryOptimizer:
    def __init__(self):
        self.memory_threshold = 0.85  # 85% utilization target
        self.gc_frequency = 100  # Garbage collection frequency
        
    def optimize_batch_size(self, model, max_batch_size=512):
        """Find optimal batch size for memory utilization."""
        for batch_size in range(32, max_batch_size + 1, 32):
            try:
                memory_usage = self.estimate_memory_usage(model, batch_size)
                if memory_usage < self.memory_threshold:
                    optimal_batch_size = batch_size
                else:
                    break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                    
        return optimal_batch_size
        
    def enable_memory_optimizations(self):
        """Enable advanced memory optimization techniques."""
        import habana_frameworks.torch as htorch
        
        # Enable memory pool optimization
        htorch.hpu.memory.set_per_process_memory_fraction(0.95)
        
        # Enable gradient checkpointing for large models
        htorch.hpu.memory.empty_cache()
        
        # Configure memory pool strategy
        htorch.hpu.memory.set_memory_pool_strategy('OPTIMIZE_UTILIZATION')
```

### Network and Storage Optimization

#### High-Speed Interconnect Configuration
```yaml
# kubernetes/network-optimization.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: network-optimization
data:
  # RDMA over Converged Ethernet (RoCE) settings
  roce_config: |
    # Enable RoCE for HPU-to-HPU communication
    export HABANA_ROCE_ENABLED=1
    export HABANA_ROCE_PRIORITY=7
    export HABANA_ROCE_DSCP=46
    
    # Optimize network buffer sizes
    echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
    echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
    echo 'net.ipv4.tcp_rmem = 4096 87380 134217728' >> /etc/sysctl.conf
    echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' >> /etc/sysctl.conf
    
  # Storage optimization for dataset loading
  storage_config: |
    # NVMe SSD optimization
    echo noop > /sys/block/nvme0n1/queue/scheduler
    echo 4096 > /sys/block/nvme0n1/queue/nr_requests
    echo 2 > /sys/block/nvme0n1/queue/rq_affinity
```

## Software Optimization

### Model Optimization Techniques

#### Mixed Precision Training
```python
# src/gaudi3_scale/precision.py
class GaudiMixedPrecision:
    def __init__(self):
        self.precision_config = {
            "enabled": True,
            "opt_level": "O2",  # Aggressive mixed precision
            "keep_batchnorm_fp32": True,
            "loss_scale": "dynamic",
            "min_loss_scale": 1.0,
            "max_loss_scale": 2.**16,
        }
        
    def configure_model(self, model):
        """Configure model for optimal mixed precision."""
        import habana_frameworks.torch as htorch
        
        # Enable BF16 for optimal Gaudi3 performance
        model = model.to(dtype=torch.bfloat16)
        
        # Configure automatic mixed precision
        htorch.hpu.amp.initialize(
            model,
            opt_level=self.precision_config["opt_level"],
            keep_batchnorm_fp32=self.precision_config["keep_batchnorm_fp32"]
        )
        
        return model
        
    def optimize_optimizer(self, optimizer):
        """Configure optimizer for mixed precision."""
        # Use Habana-optimized FusedAdamW
        return htorch.optim.FusedAdamW(
            optimizer.param_groups,
            lr=optimizer.defaults['lr'],
            betas=optimizer.defaults['betas'],
            eps=optimizer.defaults['eps'],
            weight_decay=optimizer.defaults['weight_decay']
        )
```

#### Graph Compilation Optimization
```python
# src/gaudi3_scale/compiler.py
class HabanaGraphCompiler:
    def __init__(self):
        self.compiler_flags = {
            "PT_HPU_LAZY_MODE": "1",
            "PT_HPU_ENABLE_LAZY_COMPILATION": "1",
            "PT_HPU_GRAPH_COMPILER_OPT_LEVEL": "3",
            "PT_HPU_ENABLE_COMPILATION_CACHE": "1",
            "PT_HPU_COMPILATION_CACHE_PATH": "/tmp/hpu_cache",
        }
        
    def optimize_for_throughput(self):
        """Configure compiler for maximum throughput."""
        os.environ.update({
            "PT_HPU_MAX_COMPOUND_OP_SIZE": "256",
            "PT_HPU_ENABLE_SYNAPSE_LAYOUT_OPT": "1",
            "PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE": "1",
            "PT_HPU_ENABLE_CONV_FLATTEN_OPT": "1",
        })
        
    def optimize_for_latency(self):
        """Configure compiler for minimum latency."""
        os.environ.update({
            "PT_HPU_ENABLE_EAGER_EXECUTION": "1",
            "PT_HPU_LAZY_MODE": "0",
            "PT_HPU_ENABLE_DYNAMIC_COMPILATION": "1",
        })
```

### Data Pipeline Optimization

#### Efficient Data Loading
```python
# src/gaudi3_scale/dataloader.py
class OptimizedDataLoader:
    def __init__(self, dataset, batch_size, num_workers=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers or self._calculate_optimal_workers()
        
    def _calculate_optimal_workers(self):
        """Calculate optimal number of data loading workers."""
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        # Use 2 workers per physical CPU core, capped at 16
        return min(cpu_count * 2, 16)
        
    def create_dataloader(self):
        """Create optimized DataLoader with Habana extensions."""
        from habana_frameworks.torch.utils.data import DataLoader
        
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,  # Prefetch 4 batches per worker
            # Habana-specific optimizations
            habana_loader=True,
            habana_async_data_copy=True,
        )
        
    def enable_data_prefetching(self):
        """Enable aggressive data prefetching."""
        import habana_frameworks.torch as htorch
        
        # Configure data prefetching to HPU memory
        htorch.hpu.data.enable_async_data_copy()
        htorch.hpu.data.set_prefetch_count(8)
```

## Performance Monitoring and Profiling

### Comprehensive Performance Metrics

#### HPU Utilization Monitoring
```python
# scripts/monitor_performance.py
class HPUPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "hpu_utilization": [],
            "memory_utilization": [],
            "throughput": [],
            "batch_processing_time": [],
        }
        
    def collect_metrics(self):
        """Collect comprehensive HPU performance metrics."""
        import habana_frameworks.torch as htorch
        
        # HPU utilization
        utilization = htorch.hpu.utilization()
        self.metrics["hpu_utilization"].append(utilization)
        
        # Memory usage
        memory_allocated = htorch.hpu.memory_allocated()
        memory_reserved = htorch.hpu.memory_reserved()
        memory_util = memory_allocated / memory_reserved if memory_reserved > 0 else 0
        self.metrics["memory_utilization"].append(memory_util)
        
        # Performance counters
        perf_counters = htorch.hpu.get_performance_counters()
        self.metrics.update(perf_counters)
        
    def generate_performance_report(self):
        """Generate comprehensive performance analysis report."""
        report = {
            "average_hpu_utilization": np.mean(self.metrics["hpu_utilization"]),
            "peak_memory_usage": max(self.metrics["memory_utilization"]),
            "average_throughput": np.mean(self.metrics["throughput"]),
            "performance_bottlenecks": self._identify_bottlenecks(),
        }
        return report
```

#### Distributed Training Performance
```python
# scripts/distributed_profiling.py
class DistributedTrainingProfiler:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.communication_metrics = {}
        
    def profile_communication(self):
        """Profile inter-HPU communication patterns."""
        import torch.distributed as dist
        
        # Measure AllReduce performance
        tensor = torch.randn(1024, 1024).to('hpu')
        start_time = time.time()
        dist.all_reduce(tensor)
        allreduce_time = time.time() - start_time
        
        # Measure AllGather performance  
        output_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        start_time = time.time()
        dist.all_gather(output_tensors, tensor)
        allgather_time = time.time() - start_time
        
        self.communication_metrics.update({
            "allreduce_latency": allreduce_time,
            "allgather_latency": allgather_time,
            "bandwidth_utilization": self._calculate_bandwidth_utilization(),
        })
        
    def optimize_communication(self):
        """Apply communication optimizations based on profiling."""
        # Enable gradient compression
        os.environ["HABANA_ENABLE_GRADIENT_COMPRESSION"] = "1"
        
        # Optimize communication backend
        os.environ["HABANA_COMM_BACKEND"] = "hccl"  # Use Habana Collective Communications
        
        # Configure communication topology
        os.environ["HABANA_COMM_TOPOLOGY"] = "ring"  # or "tree" for larger clusters
```

### Automated Performance Optimization

#### Adaptive Batch Size Tuning
```python
# src/gaudi3_scale/adaptive_tuning.py
class AdaptiveBatchSizeTuner:
    def __init__(self, model, initial_batch_size=32):
        self.model = model
        self.current_batch_size = initial_batch_size
        self.performance_history = []
        
    def find_optimal_batch_size(self, max_batch_size=1024):
        """Automatically find optimal batch size for maximum throughput."""
        best_throughput = 0
        optimal_batch_size = self.current_batch_size
        
        for batch_size in range(32, max_batch_size + 1, 32):
            try:
                throughput = self._measure_throughput(batch_size)
                if throughput > best_throughput:
                    best_throughput = throughput
                    optimal_batch_size = batch_size
                    
                # Stop if throughput starts decreasing significantly
                if len(self.performance_history) > 3:
                    recent_trend = np.mean(self.performance_history[-3:])
                    if throughput < recent_trend * 0.95:
                        break
                        
                self.performance_history.append(throughput)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                    
        self.current_batch_size = optimal_batch_size
        return optimal_batch_size
        
    def _measure_throughput(self, batch_size):
        """Measure training throughput for given batch size."""
        # Create sample batch
        sample_batch = self._create_sample_batch(batch_size)
        
        # Warm up
        for _ in range(5):
            _ = self.model(sample_batch)
            
        # Measure performance
        start_time = time.time()
        for _ in range(10):
            _ = self.model(sample_batch)
        elapsed_time = time.time() - start_time
        
        throughput = (batch_size * 10) / elapsed_time
        return throughput
```

## Continuous Performance Monitoring

### Performance Dashboards

#### Grafana Dashboard Configuration
```yaml
# monitoring/grafana/dashboards/performance-dashboard.json
{
  "dashboard": {
    "id": null,
    "title": "Gaudi3 Performance Monitoring",
    "tags": ["gaudi3", "performance", "hpu"],
    "panels": [
      {
        "title": "HPU Utilization",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(hpu_utilization_percent)",
            "legendFormat": "Average HPU Utilization"
          }
        ],
        "thresholds": [
          {"color": "red", "value": 60},
          {"color": "yellow", "value": 80},
          {"color": "green", "value": 90}
        ]
      },
      {
        "title": "Training Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(training_samples_processed_total[5m])",
            "legendFormat": "Samples/sec"
          }
        ]
      },
      {
        "title": "Memory Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "hpu_memory_allocated_bytes / hpu_memory_total_bytes * 100",
            "legendFormat": "Memory Usage %"
          }
        ]
      }
    ]
  }
}
```

### Automated Performance Alerts

#### Prometheus Alert Rules
```yaml
# monitoring/prometheus/alert_rules.yml
groups:
- name: performance.rules
  rules:
  - alert: LowHPUUtilization
    expr: avg(hpu_utilization_percent) < 70
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Low HPU utilization detected"
      description: "HPU utilization has been below 70% for more than 5 minutes"
      
  - alert: HighMemoryUsage
    expr: hpu_memory_allocated_bytes / hpu_memory_total_bytes > 0.95
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High HPU memory usage"
      description: "HPU memory usage is above 95%"
      
  - alert: TrainingThroughputDrop
    expr: rate(training_samples_processed_total[5m]) < 1000
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Training throughput drop detected"
      description: "Training throughput has dropped below 1000 samples/sec"
```

## Performance Optimization Checklist

### Hardware Configuration ✅
- [ ] Gaudi3 drivers and firmware updated to latest versions
- [ ] HPU memory allocation optimized (85-95% utilization)
- [ ] Network interconnect configured for maximum bandwidth
- [ ] NVMe storage optimized for sequential I/O
- [ ] CPU cores allocated for data preprocessing

### Software Configuration ✅
- [ ] Mixed precision (BF16) enabled for all models
- [ ] Graph compilation optimizations applied
- [ ] Data loading pipeline optimized with prefetching
- [ ] Gradient compression enabled for distributed training
- [ ] Memory pool strategy configured for utilization

### Monitoring and Alerting ✅
- [ ] Performance metrics collection implemented
- [ ] Grafana dashboards configured for real-time monitoring
- [ ] Automated alerts configured for performance degradation
- [ ] Regular performance regression testing in place
- [ ] Capacity planning based on performance trends

## Resources and References

### Performance Tuning Guides
- [Habana Developer Docs](https://docs.habana.ai/en/latest/)
- [PyTorch Lightning Performance Guide](https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html)
- [Distributed Training Best Practices](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### Monitoring Tools
- [Habana System Management Interface](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html)
- [Prometheus HPU Exporter](https://github.com/habana-ai/prometheus-hpu-exporter)
- [Grafana Habana Plugin](https://grafana.com/grafana/plugins/habana-hpu-datasource/)