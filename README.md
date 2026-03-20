# gaudi3-scale-starter

A clean, dependency-light Python toolkit for training on **Intel Gaudi 3 HPUs** at scale.

Provides three composable components:
- **`GaudiTrainer`** — PyTorch Lightning-compatible trainer stub (runs on CPU, exposes HPU interface)
- **`LargeScaleBatchSampler`** — gradient accumulation-aware batch sampler
- **`LearningRateScaler`** — linear scaling rule for large-batch distributed training

No actual HPU hardware required for development or testing.

---

## Installation

```bash
pip install -e .
```

Or install from PyPI (once published):

```bash
pip install gaudi3-scale
```

---

## Quick Start

```python
from gaudi3_scale import GaudiTrainer, LargeScaleBatchSampler, LearningRateScaler

# Scale learning rate for large effective batch
scaler = LearningRateScaler(base_lr=0.1, base_batch_size=256)
sampler = LargeScaleBatchSampler(dataset_size=10000, batch_size=64, accumulation_steps=8)

scaled_lr = scaler.scale(sampler.effective_batch_size)
print(f"Effective batch: {sampler.effective_batch_size}, LR: {scaled_lr:.4f}")

# Set up trainer
trainer = GaudiTrainer(max_epochs=10, devices=8, accelerator="hpu")
trainer.fit(my_model, my_dataloader)
metrics = trainer.validate(my_model, val_loader)
```

Run the full demo:

```bash
python demo.py
```

---

## Components

### `GaudiTrainer`

A minimal PyTorch Lightning-compatible training loop for Gaudi 3.

```python
trainer = GaudiTrainer(
    max_epochs=10,
    devices=8,        # number of HPUs
    accelerator="hpu" # default
)

trainer.fit(model, train_loader)
val_metrics = trainer.validate(model, val_loader)
```

**Model interface** (duck-typed):

```python
class MyModel:
    def training_step(self, batch, batch_idx: int) -> float:
        ...
    
    def validation_step(self, batch, batch_idx: int) -> float:
        ...  # optional, falls back to training_step
```

**Attributes after training:**
- `trainer.train_losses` — list of per-epoch average losses
- `trainer.val_losses` — list of per-`validate()` call losses
- `trainer.current_epoch` — last epoch index (0-indexed)

---

### `LargeScaleBatchSampler`

Batch sampler with gradient accumulation scheduling.

```python
sampler = LargeScaleBatchSampler(
    dataset_size=50000,
    batch_size=32,
    accumulation_steps=4,  # default
)

print(sampler.effective_batch_size)  # 128
print(len(sampler))                  # number of macro batches per epoch

for macro_batch in sampler:
    # macro_batch is a list of dataset indices of length effective_batch_size
    ...
```

---

### `LearningRateScaler`

Linear scaling rule: `lr = base_lr × (actual_batch / base_batch)`.

Based on [Goyal et al., 2017](https://arxiv.org/abs/1706.02677) — "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour".

```python
scaler = LearningRateScaler(base_lr=0.1, base_batch_size=256)

scaled_lr = scaler.scale(actual_batch_size=2048)  # 0.8

# Linear warmup
for step in range(warmup_steps):
    lr = scaler.warmup_schedule(step, warmup_steps=1000, scaled_lr=scaled_lr)
    optimizer.param_groups[0]["lr"] = lr
```

---

## Infrastructure Template

The `terraform/` directory contains an **HCL template** for provisioning Intel Gaudi 3 clusters on your cloud provider of choice. It's fully commented and not tied to a specific provider — see [`terraform/main.tf`](terraform/main.tf) for details.

```bash
cd terraform
# Configure your provider and variables in terraform.tfvars
terraform init
terraform plan
terraform apply
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Design Principles

1. **No hard dependencies** — pure Python stdlib; drop in PyTorch/Lightning when ready
2. **Mock HPU, real interface** — develop locally, deploy on Gaudi without code changes
3. **Composable** — use each component independently or together
4. **Testable** — every class is fully unit-testable without hardware

---

## License

MIT
