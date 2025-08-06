# Gaudi 3 Scale Starter CLI Usage Guide

The Gaudi 3 Scale Starter provides three main CLI commands for working with Intel Gaudi 3 HPUs.

## Installation

Install the package with all dependencies:

```bash
pip install -e ".[habana,terraform,monitoring]"
```

## Main Commands

### 1. gaudi3-train - Model Training

Train models on Gaudi 3 HPUs with optimized settings.

#### Basic Usage

```bash
# Simple training
gaudi3-train --model llama-7b --dataset alpaca --epochs 3

# Advanced training with configuration file
gaudi3-train --config example_configs/training_config.yaml --devices 8 --batch-size 64

# Resume training from checkpoint
gaudi3-train --resume --checkpoint-dir ./checkpoints --model llama-7b
```

#### Key Options

- `--config, -c`: Training configuration file (YAML/JSON)
- `--model, -m`: Model to train (llama-7b, llama-70b, bert-large, etc.)
- `--dataset, -d`: Dataset path or HuggingFace dataset name
- `--batch-size, -b`: Training batch size per device (1-512)
- `--epochs, -e`: Number of training epochs (1-100)
- `--devices`: Number of HPU devices to use (1-64)
- `--precision`: Training precision mode (fp32, fp16, bf16, bf16-mixed)
- `--learning-rate, --lr`: Learning rate for training
- `--optimizer`: Optimizer type (adamw, fused_adamw, sgd)
- `--output-dir, -o`: Output directory for checkpoints and logs
- `--wandb-project`: Weights & Biases project name for logging
- `--dry-run`: Show training configuration without starting training

#### Example Configuration File

See `example_configs/training_config.yaml` for a complete configuration example.

### 2. gaudi3-deploy - Cluster Deployment

Deploy and manage Gaudi 3 cluster infrastructure across multiple cloud providers.

#### Basic Usage

```bash
# Quick AWS deployment
gaudi3-deploy --provider aws --cluster-size 8

# Advanced deployment with configuration
gaudi3-deploy --config example_configs/cluster_config.yaml --monitoring --spot-instances

# Dry run to see deployment plan
gaudi3-deploy --provider azure --cluster-size 16 --dry-run
```

#### Key Options

- `--provider, -p`: Cloud provider (aws, azure, gcp, onprem)
- `--cluster-size, -s`: Number of HPU nodes to deploy (1-64)
- `--instance-type, -i`: Instance type (auto-detected if not specified)
- `--region, -r`: Cloud region (auto-selected if not specified)
- `--config, -c`: Cluster configuration file (YAML/JSON)
- `--cluster-name`: Name for the cluster (auto-generated if not specified)
- `--dry-run`: Show deployment plan without applying changes
- `--auto-approve`: Automatically approve deployment without confirmation
- `--monitoring`: Deploy monitoring stack (Prometheus, Grafana)
- `--spot-instances`: Use spot instances for cost optimization
- `--enable-efa`: Enable Elastic Fabric Adapter (AWS only)
- `--storage-size`: Storage size per node in GB (100-10000)
- `--tags`: Resource tags in key=value,key2=value2 format

#### Cluster Status

```bash
# Check all clusters
gaudi3-deploy status

# Check specific cluster
gaudi3-deploy status --cluster-name my-cluster

# Detailed monitoring with auto-refresh
gaudi3-deploy status --detailed --refresh-interval 30
```

### 3. gaudi3-benchmark - Performance Benchmarking

Run comprehensive performance benchmarks on Gaudi 3 HPUs.

#### Basic Usage

```bash
# Basic benchmark
gaudi3-benchmark --model llama-7b --batch-sizes "8,16,32"

# Comprehensive analysis
gaudi3-benchmark --model llama-70b --memory-profile --network-benchmark

# Compare with H100
gaudi3-benchmark --model bert-large --compare-baseline h100
```

#### Key Options

- `--model, -m`: Model to benchmark (llama-7b, llama-70b, bert-large, etc.)
- `--batch-sizes, -b`: Comma-separated batch sizes to test
- `--devices, -d`: Number of HPU devices to use (1-64)
- `--precision, -p`: Training precision mode (fp32, fp16, bf16, bf16-mixed)
- `--sequence-length`: Input sequence length (128-8192)
- `--output, -o`: Output file for results (JSON format)
- `--warmup-steps`: Number of warmup steps before benchmarking (1-100)
- `--benchmark-steps`: Number of benchmark steps to run (10-1000)
- `--compare-baseline`: Compare results with baseline GPU (h100/a100/v100)
- `--memory-profile`: Include detailed memory profiling
- `--network-benchmark`: Include inter-device communication benchmarks
- `--config, -c`: Benchmark configuration file
- `--save-detailed`: Save detailed per-step timing information

## Global Options

All commands support these global options:

- `--verbose, -v`: Enable verbose logging
- `--config-dir`: Configuration directory path (default: ~/.gaudi3-scale)
- `--help`: Show help message

## Configuration Files

The CLI supports both YAML and JSON configuration files. Configuration files provide a convenient way to manage complex setups and share configurations across teams.

### Training Configuration

```yaml
# training_config.yaml
model_config:
  model_name: "llama-7b"
  model_type: "llama"
batch_size: 32
max_epochs: 3
learning_rate: 6e-4
precision: "bf16-mixed"
optimizer_type: "fused_adamw"
use_habana_dataloader: true
output_dir: "./output"
```

### Cluster Configuration

```yaml
# cluster_config.yaml
cluster_name: "my-cluster"
provider: "aws"
region: "us-west-2"
enable_monitoring: true
enable_spot_instances: false
storage:
  data_volume_size_gb: 1000
network:
  enable_efa: true
```

### Benchmark Configuration

```yaml
# benchmark_config.yaml
model: "llama-7b"
devices: 8
precision: "bf16-mixed"
memory_profile: true
network_benchmark: true
compare_baseline: "h100"
```

## Examples

### Complete Training Workflow

```bash
# 1. Deploy a cluster
gaudi3-deploy --provider aws --cluster-size 4 --monitoring --cluster-name training-cluster

# 2. Check cluster status
gaudi3-deploy status --cluster-name training-cluster --detailed

# 3. Run training
gaudi3-train --config training_config.yaml --devices 32

# 4. Benchmark performance
gaudi3-benchmark --model llama-7b --devices 32 --compare-baseline h100
```

### Development Workflow

```bash
# 1. Quick development cluster
gaudi3-deploy --provider aws --cluster-size 1 --spot-instances --cluster-name dev-cluster

# 2. Test training configuration
gaudi3-train --config training_config.yaml --dry-run

# 3. Short training run
gaudi3-train --model llama-7b --epochs 1 --batch-size 16 --devices 8

# 4. Performance verification
gaudi3-benchmark --model llama-7b --batch-sizes "16,32" --benchmark-steps 10
```

## Error Handling

The CLI provides comprehensive error handling and validation:

- Configuration file validation with detailed error messages
- HPU availability checking with fallback to simulation mode
- Resource requirement validation
- Network connectivity checks
- Progress indicators with error recovery

## Monitoring and Logging

- Rich console output with progress indicators and tables
- Configurable logging levels with `--verbose` flag
- Integration with Weights & Biases for training monitoring
- Prometheus and Grafana for cluster monitoring
- Comprehensive benchmark result reporting

## Cost Optimization

- Automatic cost estimation for deployments
- Spot instance support for development workloads
- Performance vs. cost comparisons with baseline GPUs
- Resource utilization monitoring and alerts
- Auto-scaling configuration options

## Support

For more information:
- Check command help: `gaudi3-train --help`
- View configuration examples in `example_configs/`
- Review benchmark results in JSON format
- Monitor cluster status with real-time updates