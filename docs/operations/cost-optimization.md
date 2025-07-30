# Cost Optimization Guide for Gaudi3 Scale

## Overview

This guide provides comprehensive strategies for optimizing costs when running large-scale ML training on Intel Gaudi 3 infrastructure. With Gaudi 3 offering 2.7x better price-performance than H100, proper optimization can deliver exceptional value.

## 🎯 Cost Optimization Strategies

### 1. Hardware Utilization Optimization

#### HPU Utilization Monitoring
```python
import habana_frameworks.torch as htorch
import psutil
import time

class HPUUtilizationMonitor:
    def __init__(self, target_utilization=0.95):
        self.target_utilization = target_utilization
        self.metrics = []
    
    def monitor_utilization(self, duration_seconds=60):
        """Monitor HPU utilization over time."""
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            utilization = htorch.hpu.utilization()
            memory_used = htorch.hpu.memory_allocated()
            memory_total = htorch.hpu.max_memory_allocated()
            
            self.metrics.append({
                'timestamp': time.time(),
                'hpu_utilization': utilization,
                'memory_utilization': memory_used / memory_total if memory_total > 0 else 0,
                'cpu_utilization': psutil.cpu_percent()
            })
            
            time.sleep(1)
    
    def get_cost_efficiency_score(self):
        """Calculate cost efficiency based on utilization."""
        if not self.metrics:
            return 0
        
        avg_hpu_util = sum(m['hpu_utilization'] for m in self.metrics) / len(self.metrics)
        avg_mem_util = sum(m['memory_utilization'] for m in self.metrics) / len(self.metrics)
        
        # Weight HPU utilization more heavily than memory
        efficiency_score = (avg_hpu_util * 0.7) + (avg_mem_util * 0.3)
        
        return min(efficiency_score / self.target_utilization, 1.0)
```

#### Optimal Batch Sizing
```python
from gaudi3_scale.optimization import BatchOptimizer

class CostAwareBatchOptimizer(BatchOptimizer):
    def __init__(self, model, cost_per_hour=32.77):
        super().__init__(model)
        self.cost_per_hour = cost_per_hour
    
    def find_cost_optimal_batch_size(self, dataset_size, target_epochs=3):
        """Find batch size that minimizes total training cost."""
        batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
        cost_analysis = []
        
        for batch_size in batch_sizes:
            try:
                # Estimate training time
                steps_per_epoch = dataset_size // batch_size
                throughput = self.measure_throughput(batch_size)
                
                if throughput is None:
                    continue
                
                total_steps = steps_per_epoch * target_epochs
                training_hours = total_steps / (throughput * 3600)  # Convert to hours
                total_cost = training_hours * self.cost_per_hour
                
                cost_analysis.append({
                    'batch_size': batch_size,
                    'training_hours': training_hours,
                    'total_cost': total_cost,
                    'cost_per_sample': total_cost / dataset_size,
                    'throughput': throughput
                })
                
            except Exception as e:
                print(f"Batch size {batch_size} failed: {e}")
                continue
        
        # Find minimum cost configuration
        if cost_analysis:
            optimal = min(cost_analysis, key=lambda x: x['total_cost'])
            return optimal
        
        return None
```

### 2. Infrastructure Cost Management

#### Auto-scaling Configuration
```yaml
# k8s/cost-aware-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gaudi-training-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gaudi-training
  minReplicas: 1
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: habana.ai/gaudi
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: queue_length
      target:
        type: AverageValue
        averageValue: "5"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 minutes
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

#### Spot Instance Strategy
```python
# scripts/spot_instance_manager.py
import boto3
import time
from typing import List, Dict

class SpotInstanceManager:
    def __init__(self, region='us-east-1'):
        self.ec2 = boto3.client('ec2', region_name=region)
        self.region = region
    
    def get_spot_prices(self, instance_types: List[str]) -> Dict[str, float]:
        """Get current spot prices for Gaudi instance types."""
        response = self.ec2.describe_spot_price_history(
            InstanceTypes=instance_types,
            ProductDescriptions=['Linux/UNIX'],
            MaxResults=len(instance_types),
            StartTime=time.time() - 3600  # Last hour
        )
        
        prices = {}
        for price_info in response['SpotPriceHistory']:
            instance_type = price_info['InstanceType']
            prices[instance_type] = float(price_info['SpotPrice'])
        
        return prices
    
    def calculate_savings(self, instance_type: str, training_hours: float) -> Dict:
        """Calculate potential savings with spot instances."""
        on_demand_prices = {
            'dl2q.24xlarge': 32.77,  # 8x Gaudi 3
            'dl2q.48xlarge': 65.54   # 16x Gaudi 3
        }
        
        spot_prices = self.get_spot_prices([instance_type])
        
        if instance_type not in spot_prices:
            return None
        
        on_demand_cost = on_demand_prices.get(instance_type, 0) * training_hours
        spot_cost = spot_prices[instance_type] * training_hours
        savings = on_demand_cost - spot_cost
        savings_percent = (savings / on_demand_cost) * 100 if on_demand_cost > 0 else 0
        
        return {
            'on_demand_cost': on_demand_cost,
            'spot_cost': spot_cost,
            'savings': savings,
            'savings_percent': savings_percent,
            'spot_price': spot_prices[instance_type]
        }
    
    def create_spot_fleet_config(self, target_capacity: int, subnet_ids: List[str]) -> Dict:
        """Create spot fleet configuration for cost optimization."""
        return {
            'SpotFleetRequestConfig': {
                'IamFleetRole': 'arn:aws:iam::ACCOUNT:role/aws-ec2-spot-fleet-role',
                'AllocationStrategy': 'diversified',
                'TargetCapacity': target_capacity,
                'SpotPrice': '25.00',  # Max price willing to pay
                'LaunchSpecifications': [
                    {
                        'ImageId': 'ami-0abcdef1234567890',  # Custom Gaudi AMI
                        'InstanceType': 'dl2q.24xlarge',
                        'KeyName': 'gaudi-training-key',
                        'SecurityGroups': [{'GroupId': 'sg-12345678'}],
                        'SubnetId': subnet_ids[0],
                        'UserData': self._get_user_data(),
                        'WeightedCapacity': 8  # 8 Gaudi HPUs
                    }
                ],
                'TerminateInstancesWithExpiration': True,
                'Type': 'maintain'
            }
        }
    
    def _get_user_data(self) -> str:
        """User data script for spot instances."""
        return """#!/bin/bash
        # Install checkpointing tools
        pip install torch-checkpoint-manager
        
        # Set up spot interruption handling
        cat > /opt/spot_handler.py << 'EOF'
import requests
import time
import subprocess
import sys

def check_spot_interruption():
    try:
        response = requests.get(
            'http://169.254.169.254/latest/meta-data/spot/instance-action',
            timeout=2
        )
        return response.status_code == 200
    except:
        return False

def save_checkpoint():
    # Signal training process to save checkpoint
    subprocess.run(['pkill', '-SIGUSR1', 'python'])
    time.sleep(30)  # Give time to save

if __name__ == '__main__':
    while True:
        if check_spot_interruption():
            print("Spot interruption detected, saving checkpoint...")
            save_checkpoint()
            sys.exit(0)
        time.sleep(5)
EOF

        # Start spot handler in background
        python /opt/spot_handler.py &
        """
```

### 3. Training Optimization for Cost

#### Checkpoint Strategy
```python
# src/gaudi3_scale/checkpointing.py
import torch
import os
import time
from pathlib import Path

class CostAwareCheckpointing:
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch, step, metrics):
        """Save checkpoint with cost tracking."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        
        # Calculate training cost so far
        elapsed_time = time.time() - metrics.get('start_time', time.time())
        cost_per_second = 32.77 / 3600  # $32.77/hour for dl2q.24xlarge
        accumulated_cost = elapsed_time * cost_per_second
        
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'accumulated_cost': accumulated_cost,
            'checkpoint_time': time.time(),
            'training_efficiency': metrics.get('hpu_utilization', 0)
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_best_checkpoint(self) -> Dict:
        """Load checkpoint with best cost efficiency."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        if not checkpoints:
            return None
        
        best_checkpoint = None
        best_efficiency = 0
        
        for checkpoint_path in checkpoints:
            checkpoint = torch.load(checkpoint_path)
            efficiency = checkpoint.get('training_efficiency', 0)
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_checkpoint = checkpoint
        
        return best_checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Keep only the most recent checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for checkpoint in checkpoints[self.max_checkpoints:]:
            checkpoint.unlink()
```

#### Mixed Precision Cost Impact
```python
# src/gaudi3_scale/precision_optimizer.py
import torch
import habana_frameworks.torch as htorch

class CostAwarePrecisionOptimizer:
    def __init__(self):
        self.precision_configs = {
            'fp32': {'memory_multiplier': 1.0, 'speed_multiplier': 1.0},
            'bf16': {'memory_multiplier': 0.5, 'speed_multiplier': 1.8},
            'fp16': {'memory_multiplier': 0.5, 'speed_multiplier': 1.9},
            'int8': {'memory_multiplier': 0.25, 'speed_multiplier': 2.2}
        }
    
    def calculate_precision_cost_impact(self, base_cost: float, precision: str) -> Dict:
        """Calculate cost impact of different precision modes."""
        config = self.precision_configs.get(precision, self.precision_configs['fp32'])
        
        # Speed improvement reduces training time
        time_reduction = 1.0 / config['speed_multiplier']
        cost_after_speed = base_cost * time_reduction
        
        # Memory efficiency allows larger batch sizes (indirect cost benefit)
        memory_savings = 1.0 - config['memory_multiplier']
        potential_batch_increase = 1.0 + memory_savings
        
        return {
            'precision': precision,
            'original_cost': base_cost,
            'cost_after_speed': cost_after_speed,
            'time_savings_percent': (1 - time_reduction) * 100,
            'memory_savings_percent': memory_savings * 100,
            'potential_batch_increase': potential_batch_increase,
            'cost_savings': base_cost - cost_after_speed
        }
    
    def recommend_precision(self, model_size: str, accuracy_tolerance: float = 0.02) -> str:
        """Recommend precision based on model size and accuracy requirements."""
        size_gb = self._estimate_model_size(model_size)
        
        if accuracy_tolerance < 0.01:
            return 'fp32'  # High accuracy requirement
        elif size_gb > 100:
            return 'bf16'  # Large models benefit from BF16
        elif size_gb > 50:
            return 'fp16'  # Medium models can use FP16
        else:
            return 'bf16'  # Default for Gaudi 3
    
    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in GB."""
        size_estimates = {
            'llama-7b': 14,
            'llama-13b': 26,
            'llama-30b': 60,
            'llama-70b': 140,
            'mixtral-8x7b': 90,
            'stable-diffusion-xl': 6.9
        }
        
        return size_estimates.get(model_name.lower(), 30)  # Default estimate
```

### 4. Cost Monitoring and Alerting

#### Real-time Cost Tracking
```python
# src/gaudi3_scale/cost_monitor.py
import time
import logging
from typing import Dict, List
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

class CostMonitor:
    def __init__(self, instance_cost_per_hour: float = 32.77):
        self.instance_cost_per_hour = instance_cost_per_hour
        self.start_time = time.time()
        
        # Prometheus metrics
        self.cost_counter = Counter('training_cost_total', 'Total training cost in USD')
        self.cost_rate_gauge = Gauge('training_cost_rate', 'Current cost rate per hour')
        self.efficiency_gauge = Gauge('cost_efficiency', 'Cost efficiency score (0-1)')
        self.hpu_utilization_gauge = Gauge('hpu_utilization', 'HPU utilization percentage')
        
        self.cost_history = []
        self.efficiency_history = []
    
    def update_metrics(self, hpu_utilization: float, throughput: float):
        """Update cost and efficiency metrics."""
        current_time = time.time()
        elapsed_hours = (current_time - self.start_time) / 3600
        
        # Calculate current cost
        current_cost = elapsed_hours * self.instance_cost_per_hour
        cost_rate = self.instance_cost_per_hour
        
        # Calculate efficiency (utilization * throughput normalized)
        efficiency = min(hpu_utilization * (throughput / 1000), 1.0)  # Normalize to 0-1
        
        # Update Prometheus metrics
        self.cost_counter._value._value = current_cost
        self.cost_rate_gauge.set(cost_rate)
        self.efficiency_gauge.set(efficiency)
        self.hpu_utilization_gauge.set(hpu_utilization)
        
        # Store history
        self.cost_history.append({
            'time': current_time,
            'cost': current_cost,
            'rate': cost_rate
        })
        
        self.efficiency_history.append({
            'time': current_time,
            'efficiency': efficiency,
            'utilization': hpu_utilization,
            'throughput': throughput
        })
    
    def get_cost_summary(self) -> Dict:
        """Get comprehensive cost summary."""
        if not self.cost_history:
            return {}
        
        current_cost = self.cost_history[-1]['cost']
        avg_efficiency = sum(e['efficiency'] for e in self.efficiency_history) / len(self.efficiency_history)
        
        return {
            'total_cost': current_cost,
            'average_efficiency': avg_efficiency,
            'projected_daily_cost': self.instance_cost_per_hour * 24,
            'cost_per_efficiency_point': current_cost / max(avg_efficiency, 0.01),
            'runtime_hours': (time.time() - self.start_time) / 3600
        }
    
    def check_cost_alerts(self, max_cost: float = 1000, min_efficiency: float = 0.7) -> List[str]:
        """Check for cost-related alerts."""
        alerts = []
        
        if self.cost_history:
            current_cost = self.cost_history[-1]['cost']
            if current_cost > max_cost:
                alerts.append(f"Cost alert: Current cost ${current_cost:.2f} exceeds limit ${max_cost:.2f}")
        
        if self.efficiency_history:
            recent_efficiency = sum(e['efficiency'] for e in self.efficiency_history[-10:]) / min(len(self.efficiency_history), 10)
            if recent_efficiency < min_efficiency:
                alerts.append(f"Efficiency alert: Recent efficiency {recent_efficiency:.2f} below threshold {min_efficiency:.2f}")
        
        return alerts
```

### 5. Multi-Region Cost Optimization

#### Cost Comparison Tool
```python
# scripts/multi_region_cost_comparison.py
import boto3
import json
from typing import Dict, List

class MultiRegionCostOptimizer:
    def __init__(self):
        self.regions = [
            'us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'
        ]
        self.gaudi_instance_types = ['dl2q.24xlarge', 'dl2q.48xlarge']
    
    def get_regional_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get pricing for Gaudi instances across regions."""
        pricing = {}
        
        for region in self.regions:
            pricing[region] = {}
            ec2 = boto3.client('ec2', region_name=region)
            
            try:
                # Get spot prices
                response = ec2.describe_spot_price_history(
                    InstanceTypes=self.gaudi_instance_types,
                    ProductDescriptions=['Linux/UNIX'],
                    MaxResults=10
                )
                
                for price_info in response['SpotPriceHistory']:
                    instance_type = price_info['InstanceType']
                    if instance_type not in pricing[region]:
                        pricing[region][instance_type] = {
                            'spot': float(price_info['SpotPrice']),
                            'on_demand': self._get_on_demand_price(instance_type)
                        }
                        
            except Exception as e:
                print(f"Could not get pricing for {region}: {e}")
                continue
        
        return pricing
    
    def find_cheapest_region(self, instance_type: str, training_hours: float) -> Dict:
        """Find the most cost-effective region for training."""
        pricing = self.get_regional_pricing()
        
        cheapest_region = None
        lowest_cost = float('inf')
        
        for region, region_pricing in pricing.items():
            if instance_type in region_pricing:
                spot_cost = region_pricing[instance_type]['spot'] * training_hours
                
                if spot_cost < lowest_cost:
                    lowest_cost = spot_cost
                    cheapest_region = {
                        'region': region,
                        'cost': spot_cost,
                        'hourly_rate': region_pricing[instance_type]['spot'],
                        'on_demand_rate': region_pricing[instance_type]['on_demand']
                    }
        
        return cheapest_region
    
    def _get_on_demand_price(self, instance_type: str) -> float:
        """Get on-demand pricing (approximate)."""
        on_demand_prices = {
            'dl2q.24xlarge': 32.77,
            'dl2q.48xlarge': 65.54
        }
        return on_demand_prices.get(instance_type, 0)
    
    def generate_cost_report(self, training_hours: float = 72) -> str:
        """Generate comprehensive cost comparison report."""
        pricing = self.get_regional_pricing()
        
        report = "# Multi-Region Cost Analysis\n\n"
        report += f"**Training Duration:** {training_hours} hours\n\n"
        
        for instance_type in self.gaudi_instance_types:
            report += f"## {instance_type}\n\n"
            report += "| Region | Spot Price/hr | On-Demand/hr | Spot Total | On-Demand Total | Savings |\n"
            report += "|--------|---------------|--------------|------------|-----------------|----------|\n"
            
            for region in self.regions:
                if region in pricing and instance_type in pricing[region]:
                    spot_rate = pricing[region][instance_type]['spot']
                    od_rate = pricing[region][instance_type]['on_demand']
                    spot_total = spot_rate * training_hours
                    od_total = od_rate * training_hours
                    savings = ((od_total - spot_total) / od_total) * 100 if od_total > 0 else 0
                    
                    report += f"| {region} | ${spot_rate:.3f} | ${od_rate:.3f} | ${spot_total:.2f} | ${od_total:.2f} | {savings:.1f}% |\n"
            
            report += "\n"
        
        return report
```

## 📊 Cost Optimization Dashboard

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Gaudi3 Training Cost Optimization",
    "panels": [
      {
        "title": "Real-time Cost Tracking",
        "type": "stat",
        "targets": [
          {
            "expr": "training_cost_total",
            "legendFormat": "Total Cost ($)"
          }
        ]
      },
      {
        "title": "Cost Efficiency Over Time",
        "type": "graph",
        "targets": [
          {
            "expr": "cost_efficiency",
            "legendFormat": "Efficiency Score"
          },
          {
            "expr": "hpu_utilization / 100",
            "legendFormat": "HPU Utilization"
          }
        ]
      },
      {
        "title": "Cost vs Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "training_cost_rate",
            "legendFormat": "Cost Rate ($/hr)"
          },
          {
            "expr": "training_throughput",
            "legendFormat": "Throughput (samples/sec)"
          }
        ]
      }
    ]
  }
}
```

## 🎯 Cost Optimization Checklist

### Pre-Training Optimization
- [ ] Benchmark different batch sizes for optimal throughput
- [ ] Choose appropriate precision (BF16/FP16) for model size
- [ ] Select cost-effective instance types and regions
- [ ] Configure auto-scaling policies
- [ ] Set up spot instance strategies

### During Training Optimization
- [ ] Monitor HPU utilization (target >90%)
- [ ] Track cost efficiency metrics
- [ ] Implement smart checkpointing
- [ ] Use gradient accumulation for memory efficiency
- [ ] Monitor for spot instance interruptions

### Post-Training Analysis
- [ ] Analyze cost breakdown by training phase
- [ ] Compare actual vs. projected costs
- [ ] Identify optimization opportunities
- [ ] Document lessons learned for future training runs
- [ ] Calculate ROI compared to alternative hardware

## 📈 Expected Cost Savings

With proper optimization, expect the following cost reductions:

| Optimization | Potential Savings |
|--------------|-------------------|
| Spot Instances | 60-80% |
| Optimal Batch Sizing | 15-25% |
| Mixed Precision (BF16) | 40-45% |
| HPU Utilization >90% | 20-30% |
| Multi-region Selection | 10-20% |
| **Total Combined** | **70-85%** |

## 🔧 Implementation Priority

1. **High Impact, Low Effort**
   - Enable BF16 mixed precision
   - Optimize batch sizes
   - Configure basic monitoring

2. **High Impact, Medium Effort**
   - Implement spot instance strategy
   - Set up auto-scaling
   - Add checkpointing

3. **Medium Impact, High Effort**
   - Multi-region deployment
   - Advanced cost analytics
   - Custom optimization algorithms

Remember: The key to cost optimization is continuous monitoring and iterative improvement. Start with high-impact, low-effort optimizations and gradually implement more sophisticated strategies.