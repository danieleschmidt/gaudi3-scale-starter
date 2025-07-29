# Quick Start Guide

Get up and running with Gaudi 3 Scale Starter in 5 minutes.

## Prerequisites

- Python 3.10+
- Intel Gaudi 3 hardware or HPU simulator
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gaudi3-scale-starter.git
cd gaudi3-scale-starter
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .
```

### 3. Verify Installation

```bash
# Check Gaudi availability
python -c "import habana_frameworks.torch as htorch; print(f'HPUs available: {htorch.hpu.device_count()}')"

# Test CLI
gaudi3-train --help
```

## Basic Usage

### Training a Model

```python
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from gaudi3_scale import GaudiTrainer, GaudiOptimizer

# Load model and tokenizer
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create Lightning module
class SimpleModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        return outputs.loss
    
    def configure_optimizers(self):
        return GaudiOptimizer.FusedAdamW(
            self.parameters(),
            lr=1e-4,
            use_habana=True
        )

# Setup trainer
lightning_model = SimpleModel(model)
trainer = GaudiTrainer(
    model=lightning_model,
    devices=8,
    max_epochs=1
)

# Start training (you'll need to provide your own dataloader)
# trainer.fit(train_dataloader)
```

### Using the CLI

```bash
# Train a model
gaudi3-train --model microsoft/DialoGPT-medium --batch-size 16 --epochs 1

# Deploy infrastructure
gaudi3-deploy --cluster-size 8 --cloud aws

# Run benchmarks
gaudi3-benchmark --model microsoft/DialoGPT-medium --batch-sizes 8,16,32
```

## Infrastructure Deployment

### Deploy on AWS

```bash
# Configure AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Deploy 8-HPU cluster
cd terraform/aws
terraform init
terraform plan -var="cluster_size=8"
terraform apply
```

### Access Your Cluster

```bash
# Get cluster connection info
terraform output cluster_endpoints

# SSH to master node
ssh ubuntu@$(terraform output master_ip)

# Verify HPU availability
hl-smi
```

## Next Steps

- **Learn More**: Read the [Training Guide](training.md) for detailed usage
- **Optimize Performance**: Check [Performance Tuning](performance-tuning.md)
- **Scale Up**: See [Multi-Node Setup](multi-node.md) for larger deployments
- **Monitor**: Set up [Monitoring & Observability](../infrastructure/monitoring.md)

## Common Issues

### HPU Not Detected

```bash
# Check Habana driver installation
ls /dev/accel/accel*

# Verify environment
echo $HABANA_VISIBLE_DEVICES
```

### Import Errors

```bash
# Reinstall Habana packages
pip uninstall habana-torch-plugin habana-torch-dataloader
pip install habana-torch-plugin habana-torch-dataloader
```

### Performance Issues

- Ensure BF16 mixed precision is enabled
- Check batch size optimization
- Verify graph compilation settings

## Getting Help

- **Documentation**: Continue reading the guides
- **Issues**: [GitHub Issues](https://github.com/yourusername/gaudi3-scale-starter/issues)
- **Community**: [Slack Workspace](https://gaudi3-scale.slack.com)