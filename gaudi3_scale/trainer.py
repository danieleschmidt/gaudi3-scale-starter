"""
GaudiTrainer: PyTorch Lightning-compatible trainer stub for Intel Gaudi 3 HPU.

Uses CPU internally for portability, but exposes the HPU accelerator interface.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GaudiTrainer:
    """A PyTorch Lightning-compatible trainer stub for Intel Gaudi 3 HPU.

    Internally runs on CPU but exposes the HPU accelerator interface,
    allowing code to be written targeting Gaudi hardware without requiring
    actual HPU hardware for development and testing.

    Args:
        max_epochs: Number of training epochs.
        devices: Number of HPU devices to use (mock).
        accelerator: Accelerator type, defaults to "hpu".
    """

    def __init__(self, max_epochs: int, devices: int, accelerator: str = "hpu") -> None:
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        if devices < 1:
            raise ValueError(f"devices must be >= 1, got {devices}")

        self.max_epochs = max_epochs
        self.devices = devices
        self.accelerator = accelerator
        self.current_epoch = 0
        self._is_training = False
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []

    def fit(self, model: Any, dataloader: Any) -> None:
        """Run the training loop.

        Iterates over epochs and batches, calling model.training_step for each batch.

        Args:
            model: A model with a training_step(batch, batch_idx) method.
            dataloader: An iterable of batches.
        """
        self._is_training = True
        self._train_losses = []

        logger.info(
            "Starting training: max_epochs=%d, devices=%d, accelerator=%s",
            self.max_epochs,
            self.devices,
            self.accelerator,
        )

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            epoch_losses = []

            for batch_idx, batch in enumerate(dataloader):
                loss = model.training_step(batch, batch_idx)
                # Support both tensor-like objects with .item() and plain floats
                if hasattr(loss, "item"):
                    loss_val = loss.item()
                else:
                    loss_val = float(loss)
                epoch_losses.append(loss_val)

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            self._train_losses.append(avg_loss)
            logger.info("Epoch %d/%d — train_loss=%.4f", epoch + 1, self.max_epochs, avg_loss)

        self._is_training = False
        logger.info("Training complete.")

    def validate(self, model: Any, dataloader: Any) -> dict[str, float]:
        """Run the validation loop.

        Iterates over the dataloader calling model.validation_step if available,
        otherwise falls back to training_step.

        Args:
            model: A model with a validation_step(batch, batch_idx) or
                   training_step(batch, batch_idx) method.
            dataloader: An iterable of batches.

        Returns:
            dict with "val_loss" key containing average validation loss.
        """
        step_fn = getattr(model, "validation_step", None) or getattr(model, "training_step")
        losses = []

        for batch_idx, batch in enumerate(dataloader):
            loss = step_fn(batch, batch_idx)
            if hasattr(loss, "item"):
                losses.append(loss.item())
            else:
                losses.append(float(loss))

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        self._val_losses.append(avg_loss)
        logger.info("Validation — val_loss=%.4f", avg_loss)
        return {"val_loss": avg_loss}

    @property
    def train_losses(self) -> list[float]:
        """List of average training losses per epoch."""
        return list(self._train_losses)

    @property
    def val_losses(self) -> list[float]:
        """List of average validation losses from validate() calls."""
        return list(self._val_losses)

    def __repr__(self) -> str:
        return (
            f"GaudiTrainer(max_epochs={self.max_epochs}, "
            f"devices={self.devices}, accelerator='{self.accelerator}')"
        )
