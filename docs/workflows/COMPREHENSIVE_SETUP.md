# Comprehensive Gaudi 3 Scale Setup Guide

This comprehensive guide covers the complete setup and configuration of the Gaudi 3 Scale infrastructure platform.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Deployment](#infrastructure-deployment)
3. [Environment Configuration](#environment-configuration)
4. [Training Pipeline Setup](#training-pipeline-setup)
5. [Monitoring Configuration](#monitoring-configuration)
6. [Security Configuration](#security-configuration)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

#### Minimum Requirements
- **Compute**: 1x Intel Gaudi 3 HPU node (8 HPUs)
- **Memory**: 256GB system RAM, 32GB HBM per HPU
- **Storage**: 1TB NVMe SSD for OS and applications
- **Network**: 100Gbps Ethernet or InfiniBand

#### Recommended Production Setup
- **Compute**: 4-8x Intel Gaudi 3 HPU nodes (32-64 HPUs total)
- **Memory**: 512GB system RAM per node, 32GB HBM per HPU
- **Storage**: 2TB NVMe SSD per node + shared storage
- **Network**: 200Gbps Ethernet or InfiniBand with EFA/RoCE

### Software Prerequisites

```bash
# Operating System
Ubuntu 22.04 LTS (recommended) or RHEL 8.6+

# Core Dependencies
Python 3.10+
Docker 24.0+
Kubernetes 1.29+
Terraform 1.8+

# Habana Software Stack
SynapseAI 1.16.0+
Habana Torch Plugin 1.16.0+
Habana Media Loader 1.16.0+

# Monitoring Stack
Prometheus 2.45+
Grafana 10.4+
Node Exporter 1.6+
```

### Cloud Account Setup

#### AWS Setup
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure credentials
aws configure
# AWS Access Key ID: [Your access key]
# AWS Secret Access Key: [Your secret key]
# Default region: us-east-1
# Default output format: json

# Verify permissions
aws sts get-caller-identity
aws ec2 describe-regions
```

#### Azure Setup
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Set subscription
az account set --subscription "your-subscription-id"

# Verify access
az account show
```

## Infrastructure Deployment

### 1. Clone and Setup Repository

```bash
# Clone repository
git clone https://github.com/yourusername/gaudi3-scale-starter.git
cd gaudi3-scale-starter

# Install development dependencies
make install-dev

# Verify installation
python -c "import gaudi3_scale; print(gaudi3_scale.__version__)"
```

### 2. Configure Infrastructure Variables

```bash
# Copy example configuration
cp terraform/aws/terraform.tfvars.example terraform/aws/terraform.tfvars

# Edit configuration
vim terraform/aws/terraform.tfvars
```

Example `terraform.tfvars`:
```hcl
# Project Configuration
project_name = "gaudi3-scale"
environment = "production"
region = "us-east-1"

# Cluster Configuration
cluster_size = 4                    # Number of nodes
instance_type = "dl2q.24xlarge"     # 8 Gaudi 3 HPUs per node
key_pair_name = "your-key-pair"     # SSH key pair

# Storage Configuration
ebs_volume_size = 1000              # GB per node
shared_storage_size = 5000          # GB shared EFS

# Network Configuration
vpc_cidr = "10.0.0.0/16"
enable_ebs_optimized = true
enable_efa = true                   # Enhanced networking

# Security Configuration
allowed_cidr_blocks = ["10.0.0.0/8"]
enable_encryption = true

# Monitoring Configuration
enable_monitoring = true
enable_logging = true

# Backup Configuration
backup_retention_days = 30
enable_automated_backups = true

# Tags
tags = {
  Project = "gaudi3-scale"
  Environment = "production"
  Owner = "ml-team"
  CostCenter = "research"
}
```

### 3. Deploy Infrastructure

```bash
# Initialize Terraform
cd terraform/aws
terraform init

# Validate configuration
terraform validate

# Plan deployment
terraform plan -var-file="terraform.tfvars" -out=tfplan

# Review plan carefully
terraform show tfplan

# Apply infrastructure
terraform apply tfplan

# Save outputs
terraform output > cluster_info.txt
```

### 4. Verify Deployment

```bash
# Get cluster endpoints
MASTER_IP=$(terraform output -raw master_public_ip)
WORKER_IPS=$(terraform output -json worker_public_ips | jq -r '.[]')

# Test SSH access
ssh -i ~/.ssh/your-key.pem ubuntu@$MASTER_IP

# Verify Gaudi devices
ssh ubuntu@$MASTER_IP "hl-smi"

# Check cluster connectivity
ssh ubuntu@$MASTER_IP "kubectl get nodes"
```

## Environment Configuration

### 1. Configure Habana Environment

```bash
# SSH to master node
ssh -i ~/.ssh/your-key.pem ubuntu@$MASTER_IP

# Source Habana environment
source /opt/habanalabs/init_env.sh

# Verify Habana installation
python -c "import habana_frameworks.torch as htorch; print(htorch.hpu.device_count())"

# Set environment variables
cat >> ~/.bashrc << 'EOF'
# Habana Environment
source /opt/habanalabs/init_env.sh

# Gaudi Optimizations
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COMPILATION=1
export PT_HPU_GRAPH_COMPILER_OPT_LEVEL=3
export PT_HPU_MAX_COMPOUND_OP_SIZE=256
export PT_HPU_ENABLE_SYNAPSE_LAYOUT_OPT=1
export PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=1
export PT_HPU_POOL_STRATEGY=OPTIMIZE_UTILIZATION

# Python path
export PYTHONPATH=/opt/gaudi3_scale/src:$PYTHONPATH
EOF

source ~/.bashrc
```

### 2. Install Gaudi 3 Scale

```bash
# Install from repository
cd /opt
sudo git clone https://github.com/yourusername/gaudi3-scale-starter.git gaudi3_scale
cd gaudi3_scale

# Install package
sudo pip install -e .

# Install additional dependencies
sudo pip install -r requirements.txt

# Verify installation
gaudi3-train --help
```

### 3. Configure Docker and Kubernetes

```bash
# Add user to docker group
sudo usermod -a -G docker $USER
newgrp docker

# Verify Docker access
docker run hello-world

# Configure kubectl
mkdir -p ~/.kube
sudo cp /etc/kubernetes/admin.conf ~/.kube/config
sudo chown $USER:$USER ~/.kube/config

# Verify Kubernetes
kubectl get nodes
kubectl get pods --all-namespaces
```

## Training Pipeline Setup

### 1. Prepare Training Data

```bash
# Create data directory
sudo mkdir -p /data/datasets
sudo chown -R $USER:$USER /data

# Example: Download and prepare dataset
cd /data/datasets

# For text data (example with OpenWebText)
wget https://example.com/openwebtext.tar.gz
tar -xzf openwebtext.tar.gz

# Tokenize data (example)
python scripts/tokenize_data.py \
  --input openwebtext/ \
  --output openwebtext_tokenized/ \
  --tokenizer gpt2 \
  --chunk_size 2048
```

### 2. Configure Training

Create training configuration file:

```yaml
# configs/training_config.yaml
model:
  name: "llama-7b"
  parameters: 7000000000
  sequence_length: 2048
  vocab_size: 32000
  hidden_size: 4096
  num_layers: 32
  num_attention_heads: 32

training:
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 6e-4
  max_epochs: 3
  warmup_steps: 2000
  weight_decay: 0.1
  
  # Gaudi optimizations
  use_mixed_precision: true
  precision: "bf16-mixed"
  use_graph_compilation: true
  enable_lazy_mode: true

data:
  dataset_path: "/data/datasets/openwebtext_tokenized"
  num_workers: 8
  pin_memory: true
  prefetch_factor: 2

hardware:
  num_nodes: 4
  hpus_per_node: 8
  strategy: "ddp"  # distributed data parallel

monitoring:
  wandb_project: "gaudi3-scale"
  log_interval: 100
  save_interval: 1000
  eval_interval: 500

checkpointing:
  save_dir: "/data/checkpoints"
  save_top_k: 3
  save_last: true
  resume_from_checkpoint: null
```

### 3. Launch Training

#### Single Node Training
```bash
gaudi3-train \
  --config configs/training_config.yaml \
  --model-name llama-7b \
  --data-path /data/datasets/openwebtext_tokenized \
  --output-dir /data/checkpoints/llama-7b-$(date +%Y%m%d_%H%M%S)
```

#### Multi-Node Training
```bash
# Launch on master node
gaudi3-train \
  --config configs/training_config.yaml \
  --model-name llama-7b \
  --data-path /data/datasets/openwebtext_tokenized \
  --output-dir /data/checkpoints/llama-7b-$(date +%Y%m%d_%H%M%S) \
  --num-nodes 4 \
  --node-rank 0 \
  --master-addr $(hostname -I | awk '{print $1}') \
  --master-port 29500

# Launch on worker nodes (run on each worker)
for i in {1..3}; do
  ssh worker-$i "gaudi3-train \
    --config configs/training_config.yaml \
    --model-name llama-7b \
    --data-path /data/datasets/openwebtext_tokenized \
    --output-dir /data/checkpoints/llama-7b-$(date +%Y%m%d_%H%M%S) \
    --num-nodes 4 \
    --node-rank $i \
    --master-addr $MASTER_IP \
    --master-port 29500"
done
```

#### Kubernetes Training Job
```yaml
# k8s/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gaudi3-training-job
spec:
  parallelism: 4
  completions: 4
  template:
    spec:
      containers:
      - name: gaudi3-trainer
        image: gaudi3-scale:latest
        command: ["gaudi3-train"]
        args:
          - "--config"
          - "/configs/training_config.yaml"
          - "--model-name"
          - "llama-7b"
          - "--num-nodes"
          - "4"
        resources:
          limits:
            habana.ai/gaudi: 8
            memory: 256Gi
          requests:
            habana.ai/gaudi: 8
            memory: 128Gi
        volumeMounts:
        - name: data-volume
          mountPath: /data
        - name: config-volume
          mountPath: /configs
        env:
        - name: WORLD_SIZE
          value: "32"  # 4 nodes * 8 HPUs
        - name: PT_HPU_LAZY_MODE
          value: "1"
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: config-volume
        configMap:
          name: training-config
      restartPolicy: Never
      nodeSelector:
        accelerator: gaudi3
```

Apply the job:
```bash
kubectl apply -f k8s/training-job.yaml
kubectl get jobs
kubectl logs -f job/gaudi3-training-job
```

## Monitoring Configuration

### 1. Deploy Monitoring Stack

```bash
# Deploy with Docker Compose
docker-compose -f monitoring/docker-compose.yml up -d

# Or deploy with Kubernetes
kubectl apply -f monitoring/k8s/
```

### 2. Configure Prometheus

Create custom Prometheus rules:

```yaml
# monitoring/rules/gaudi_training.yml
groups:
- name: gaudi_training
  rules:
  - alert: HighHPUUtilization
    expr: gaudi_hpu_utilization_percent > 95
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High HPU utilization detected"
      description: "HPU utilization is {{ $value }}% on {{ $labels.device_id }}"

  - alert: TrainingJobFailed
    expr: gaudi_training_job_status{status="failed"} > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Training job failed"
      description: "Job {{ $labels.job_id }} failed with status {{ $labels.status }}"

  - alert: LowTrainingThroughput
    expr: gaudi_training_throughput < 1000
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Low training throughput"
      description: "Training throughput is {{ $value }} tokens/sec, below expected"
```

### 3. Configure Grafana Dashboards

Import the provided Gaudi 3 dashboard:

```bash
# Import dashboard
curl -X POST \
  http://admin:gaudi3admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboards/gaudi3-performance.json
```

### 4. Setup Alerting

Configure AlertManager:

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@yourcompany.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'ml-team@yourcompany.com'
    subject: '[GAUDI3-SCALE] {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
  
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#ml-alerts'
    title: '[GAUDI3-SCALE] {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

## Security Configuration

### 1. Network Security

```bash
# Configure firewall rules
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow from 10.0.0.0/8 to any port 6443  # Kubernetes API
sudo ufw allow from 10.0.0.0/8 to any port 2379  # etcd
sudo ufw allow from 10.0.0.0/8 to any port 10250 # kubelet

# Configure fail2ban
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 2. Authentication and Authorization

#### Setup RBAC for Kubernetes
```yaml
# k8s/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: gaudi3-trainer-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "persistentvolumeclaims"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["extensions", "apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: gaudi3-trainer-binding
subjects:
- kind: ServiceAccount
  name: gaudi3-trainer
  namespace: default
roleRef:
  kind: ClusterRole
  name: gaudi3-trainer-role
  apiGroup: rbac.authorization.k8s.io
```

#### Setup Service Account
```yaml
# k8s/service-account.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: gaudi3-trainer
  namespace: default
automountServiceAccountToken: true
```

### 3. Secrets Management

```bash
# Create secrets for API keys
kubectl create secret generic gaudi3-secrets \
  --from-literal=wandb-api-key="your-wandb-key" \
  --from-literal=huggingface-token="your-hf-token" \
  --from-literal=github-token="your-github-token"

# Create TLS certificates
kubectl create secret tls gaudi3-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

### 4. Image Security

```bash
# Scan container images
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image gaudi3-scale:latest

# Sign images
docker trust key generate gaudi3-scale
docker trust signer add --key gaudi3-scale.pub gaudi3-scale-team gaudi3-scale
docker trust sign gaudi3-scale:latest
```

## Advanced Features

### 1. Auto-Scaling Configuration

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gaudi3-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gaudi3-training
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 2. Multi-Tenancy Setup

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: team-research
  labels:
    name: team-research
    tier: training

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-research-quota
  namespace: team-research
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    requests.habana.ai/gaudi: "32"
    limits.cpu: "200"
    limits.memory: 400Gi
    limits.habana.ai/gaudi: "32"
    persistentvolumeclaims: "10"
```

### 3. GitOps Integration

```yaml
# .github/workflows/deploy.yml
name: Deploy to Gaudi3 Cluster

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.29.0'
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to cluster
      run: |
        kubectl apply -f k8s/
        kubectl rollout status deployment/gaudi3-training
```

## Troubleshooting

### Common Issues

#### 1. HPU Not Detected
```bash
# Check driver installation
hl-smi
lsmod | grep habanalabs

# Reinstall drivers if needed
sudo /opt/habanalabs/install.sh

# Check device permissions
ls -la /dev/accel/
sudo usermod -a -G habana $USER
```

#### 2. Out of Memory Errors
```bash
# Check memory usage
hl-smi
free -h

# Reduce batch size in config
sed -i 's/batch_size: 32/batch_size: 16/' configs/training_config.yaml

# Enable gradient checkpointing
# Add to training config:
# gradient_checkpointing: true
```

#### 3. Network Issues in Multi-Node
```bash
# Check network connectivity
ping worker-1
ping worker-2

# Check EFA status
fi_info

# Test InfiniBand
ibv_devinfo
ibstatus
```

#### 4. Kubernetes Issues
```bash
# Check node status
kubectl get nodes
kubectl describe node master

# Check pod status
kubectl get pods --all-namespaces
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
journalctl -u kubelet
```

### Performance Tuning

#### 1. Optimize Batch Size
```python
# Run batch size finder
from gaudi3_scale.optimizer import TrainingOptimizer

optimizer = TrainingOptimizer()
result = optimizer.optimize_batch_size(
    model_config={"parameters": 7000000000},
    hardware_config={"num_hpus": 8, "memory_per_hpu_gb": 32}
)
print(f"Optimal batch size: {result.config['batch_size']}")
```

#### 2. Profile Training
```python
# Profile training step
from gaudi3_scale.monitoring import GaudiProfiler

profiler = GaudiProfiler()
session_id = profiler.start_profiling("training_profile")

# Run training...

profiler.stop_profiling()
analysis = profiler.analyze_session(f"profiling_results/{session_id}.json")
print(analysis["bottleneck_analysis"])
```

### Support Resources

- **Documentation**: [https://gaudi3-scale.readthedocs.io](https://gaudi3-scale.readthedocs.io)
- **GitHub Issues**: [https://github.com/yourusername/gaudi3-scale-starter/issues](https://github.com/yourusername/gaudi3-scale-starter/issues)
- **Community Slack**: [Join Workspace](https://gaudi3-scale.slack.com)
- **Support Email**: gaudi3-scale@yourdomain.com

For urgent issues, please create a GitHub issue with the `urgent` label and relevant logs.