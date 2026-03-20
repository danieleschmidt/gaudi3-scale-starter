"""
LargeScaleBatchSampler: Gradient accumulation-aware batch sampler for large-scale training.
"""
from __future__ import annotations

import random
from typing import Iterator


class LargeScaleBatchSampler:
    """Batch sampler with gradient accumulation scheduling for large-scale training.

    Wraps a dataset length and yields "macro batches" — groups of indices
    that represent one effective gradient update step.

    Args:
        dataset_size: Total number of samples in the dataset.
        batch_size: Number of samples per micro-batch.
        accumulation_steps: Number of micro-batches to accumulate before
            a gradient update. Defaults to 4.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        accumulation_steps: int = 4,
    ) -> None:
        if dataset_size < 1:
            raise ValueError(f"dataset_size must be >= 1, got {dataset_size}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps

    @property
    def effective_batch_size(self) -> int:
        """The effective batch size after gradient accumulation."""
        return self.batch_size * self.accumulation_steps

    def __len__(self) -> int:
        """Number of macro batches (gradient update steps) per epoch."""
        total_micro_batches = self.dataset_size // self.batch_size
        return total_micro_batches // self.accumulation_steps

    def __iter__(self) -> Iterator[list[int]]:
        """Yield macro batches as lists of indices.

        Each macro batch contains indices for (accumulation_steps * batch_size)
        samples — i.e., one effective gradient update worth of data.
        Indices within each macro batch are shuffled.
        """
        indices = list(range(self.dataset_size))
        random.shuffle(indices)

        macro_size = self.effective_batch_size
        num_macro = len(self)

        for i in range(num_macro):
            start = i * macro_size
            end = start + macro_size
            yield indices[start:end]

    def __repr__(self) -> str:
        return (
            f"LargeScaleBatchSampler("
            f"dataset_size={self.dataset_size}, "
            f"batch_size={self.batch_size}, "
            f"accumulation_steps={self.accumulation_steps}, "
            f"effective_batch_size={self.effective_batch_size})"
        )
