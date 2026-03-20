"""
LearningRateScaler: Linear scaling rule for large-batch distributed training.

Implements the linear scaling heuristic from "Accurate, Large Minibatch SGD:
Training ImageNet in 1 Hour" (Goyal et al., 2017).
"""
from __future__ import annotations


class LearningRateScaler:
    """Learning rate scaler using the linear scaling rule for large batches.

    When the batch size is multiplied by k, scale the learning rate by k.
    Also provides a linear warmup schedule to stabilize early training.

    Args:
        base_lr: The base learning rate corresponding to base_batch_size.
        base_batch_size: The reference batch size for base_lr. Defaults to 256.
    """

    def __init__(self, base_lr: float, base_batch_size: int = 256) -> None:
        if base_lr <= 0:
            raise ValueError(f"base_lr must be > 0, got {base_lr}")
        if base_batch_size < 1:
            raise ValueError(f"base_batch_size must be >= 1, got {base_batch_size}")

        self.base_lr = base_lr
        self.base_batch_size = base_batch_size

    def scale(self, actual_batch_size: int) -> float:
        """Compute the scaled learning rate using the linear scaling rule.

        scaled_lr = base_lr * (actual_batch_size / base_batch_size)

        Args:
            actual_batch_size: The actual batch size being used in training.

        Returns:
            The scaled learning rate.
        """
        if actual_batch_size < 1:
            raise ValueError(f"actual_batch_size must be >= 1, got {actual_batch_size}")
        return self.base_lr * actual_batch_size / self.base_batch_size

    def warmup_schedule(self, step: int, warmup_steps: int, scaled_lr: float) -> float:
        """Compute the learning rate for a given step during linear warmup.

        Linearly ramps from 0 to scaled_lr over warmup_steps steps.
        After warmup, returns scaled_lr unchanged.

        Args:
            step: Current training step (0-indexed).
            warmup_steps: Total number of warmup steps.
            scaled_lr: The target learning rate after warmup.

        Returns:
            The learning rate for the current step.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if scaled_lr < 0:
            raise ValueError(f"scaled_lr must be >= 0, got {scaled_lr}")

        if warmup_steps == 0 or step >= warmup_steps:
            return scaled_lr

        return scaled_lr * step / warmup_steps

    def __repr__(self) -> str:
        return (
            f"LearningRateScaler("
            f"base_lr={self.base_lr}, "
            f"base_batch_size={self.base_batch_size})"
        )
