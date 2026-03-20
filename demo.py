"""
demo.py — Toy training loop demonstrating gaudi3_scale components.

Creates a synthetic dataset, uses LargeScaleBatchSampler for batching,
GaudiTrainer for the training loop, and LearningRateScaler for LR scheduling.
"""
from __future__ import annotations

import random

from gaudi3_scale import GaudiTrainer, LargeScaleBatchSampler, LearningRateScaler


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticDataset:
    """Random float dataset for demo purposes."""

    def __init__(self, size: int = 1000, input_dim: int = 32) -> None:
        self.size = size
        self.input_dim = input_dim
        # Generate random data as plain Python lists (no numpy/torch required)
        self._data = [
            ([random.gauss(0, 1) for _ in range(input_dim)], random.randint(0, 9))
            for _ in range(size)
        ]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return self._data[idx]


# ---------------------------------------------------------------------------
# Simple linear model stub
# ---------------------------------------------------------------------------

class ToyModel:
    """Minimal model stub with training and validation steps."""

    def __init__(self, input_dim: int = 32, lr: float = 0.01) -> None:
        self.input_dim = input_dim
        self.lr = lr
        # Mock weights as a flat list
        self.weights = [random.gauss(0, 0.01) for _ in range(input_dim)]
        self.step_count = 0

    def training_step(self, batch, batch_idx: int) -> float:
        """Simulate a forward + backward pass. Returns a mock loss."""
        self.step_count += 1
        # Simulate decreasing loss over time
        base_loss = 1.0 / (1.0 + self.step_count * 0.01)
        noise = random.gauss(0, 0.05)
        return max(0.0, base_loss + noise)

    def validation_step(self, batch, batch_idx: int) -> float:
        base_loss = 1.0 / (1.0 + self.step_count * 0.01) + 0.05
        return max(0.0, base_loss)


# ---------------------------------------------------------------------------
# Demo macro-batch dataloader
# ---------------------------------------------------------------------------

class MacroBatchDataLoader:
    """Wraps a dataset + sampler to yield macro batches of samples."""

    def __init__(self, dataset: SyntheticDataset, sampler: LargeScaleBatchSampler) -> None:
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        for macro_indices in self.sampler:
            yield [self.dataset[i] for i in macro_indices]

    def __len__(self) -> int:
        return len(self.sampler)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("gaudi3_scale Demo — Intel Gaudi 3 Training Pipeline")
    print("=" * 60)

    # Configuration
    DATASET_SIZE = 1000
    BATCH_SIZE = 32
    ACCUMULATION_STEPS = 4
    MAX_EPOCHS = 3
    BASE_LR = 0.1
    BASE_BATCH_SIZE = 256
    WARMUP_STEPS = 10

    # 1. Create synthetic dataset
    print(f"\n[1] Creating synthetic dataset ({DATASET_SIZE} samples, dim=32)")
    dataset = SyntheticDataset(size=DATASET_SIZE, input_dim=32)
    print(f"    Dataset size: {len(dataset)}")

    # 2. Set up batch sampler
    print(f"\n[2] Setting up LargeScaleBatchSampler")
    sampler = LargeScaleBatchSampler(
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
        accumulation_steps=ACCUMULATION_STEPS,
    )
    print(f"    Micro batch size:     {BATCH_SIZE}")
    print(f"    Accumulation steps:   {ACCUMULATION_STEPS}")
    print(f"    Effective batch size: {sampler.effective_batch_size}")
    print(f"    Macro batches/epoch:  {len(sampler)}")

    # 3. Set up learning rate scaler
    print(f"\n[3] Configuring LearningRateScaler")
    lr_scaler = LearningRateScaler(base_lr=BASE_LR, base_batch_size=BASE_BATCH_SIZE)
    scaled_lr = lr_scaler.scale(sampler.effective_batch_size)
    print(f"    Base LR:      {BASE_LR}")
    print(f"    Scaled LR:    {scaled_lr:.4f}  (for effective_batch={sampler.effective_batch_size})")

    print("\n    Warmup schedule (first 15 steps):")
    for step in range(15):
        lr = lr_scaler.warmup_schedule(step, WARMUP_STEPS, scaled_lr)
        bar = "#" * int(lr / scaled_lr * 20)
        print(f"      step {step:2d}: lr={lr:.4f}  [{bar:<20}]")

    # 4. Create model and trainer
    print(f"\n[4] Initializing ToyModel and GaudiTrainer")
    model = ToyModel(input_dim=32, lr=scaled_lr)
    trainer = GaudiTrainer(
        max_epochs=MAX_EPOCHS,
        devices=8,
        accelerator="hpu",
    )
    print(f"    {trainer}")

    # 5. Create dataloader
    dataloader = MacroBatchDataLoader(dataset, sampler)

    # 6. Train
    print(f"\n[5] Training for {MAX_EPOCHS} epochs...")
    print("-" * 40)
    trainer.fit(model, dataloader)
    print("-" * 40)
    print("    Training losses per epoch:")
    for i, loss in enumerate(trainer.train_losses):
        print(f"      Epoch {i + 1}: {loss:.4f}")

    # 7. Validate
    print(f"\n[6] Running validation...")
    val_sampler = LargeScaleBatchSampler(DATASET_SIZE, BATCH_SIZE, 1)
    val_loader = MacroBatchDataLoader(dataset, val_sampler)
    metrics = trainer.validate(model, val_loader)
    print(f"    val_loss = {metrics['val_loss']:.4f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
