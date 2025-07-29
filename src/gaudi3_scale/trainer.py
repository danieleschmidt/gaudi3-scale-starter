"""High-level training interface for Gaudi 3 models."""

from typing import Any, Dict, Optional

try:
    import pytorch_lightning as pl
    import torch
except ImportError:
    pl = None
    torch = None


class GaudiTrainer:
    """High-level trainer for Gaudi 3 models.
    
    This class provides a simplified interface for training models
    on Intel Gaudi 3 hardware with optimized settings.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        accelerator: str = "hpu",
        devices: int = 8,
        precision: str = "bf16-mixed",
        max_epochs: int = 3,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 4,
        strategy: str = "ddp",
        **kwargs: Any
    ) -> None:
        """Initialize Gaudi trainer.
        
        Args:
            model: PyTorch Lightning model to train
            accelerator: Accelerator type ("hpu" for Gaudi)
            devices: Number of devices to use
            precision: Training precision
            max_epochs: Maximum training epochs
            gradient_clip_val: Gradient clipping value
            accumulate_grad_batches: Gradient accumulation steps
            strategy: Distributed training strategy
            **kwargs: Additional trainer arguments
        """
        self.model = model
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.strategy = strategy
        self.kwargs = kwargs
        
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Setup Gaudi environment variables."""
        import os
        
        # Optimal Habana graph compiler settings
        os.environ.setdefault('PT_HPU_LAZY_MODE', '1')
        os.environ.setdefault('PT_HPU_ENABLE_LAZY_COMPILATION', '1')
        os.environ.setdefault('PT_HPU_GRAPH_COMPILER_OPT_LEVEL', '3')
        os.environ.setdefault('PT_HPU_MAX_COMPOUND_OP_SIZE', '256')
        os.environ.setdefault('PT_HPU_ENABLE_SYNAPSE_LAYOUT_OPT', '1')
        
        # Memory optimizations
        os.environ.setdefault('PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE', '1')
        os.environ.setdefault('PT_HPU_POOL_STRATEGY', 'OPTIMIZE_UTILIZATION')
    
    def create_trainer(self) -> Optional[Any]:
        """Create configured PyTorch Lightning trainer.
        
        Returns:
            Configured PyTorch Lightning trainer
        """
        if pl is None:
            raise ImportError("PyTorch Lightning not available")
        
        from .accelerator import GaudiAccelerator
        
        trainer_config = {
            "accelerator": GaudiAccelerator() if self.accelerator == "hpu" else self.accelerator,
            "devices": self.devices,
            "precision": self.precision,
            "max_epochs": self.max_epochs,
            "gradient_clip_val": self.gradient_clip_val,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "strategy": self.strategy,
            **self.kwargs
        }
        
        return pl.Trainer(**trainer_config)
    
    def fit(self, train_dataloader: Any, val_dataloader: Optional[Any] = None) -> None:
        """Fit the model using configured trainer.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
        """
        if self.model is None:
            raise ValueError("Model must be set before training")
        
        trainer = self.create_trainer()
        trainer.fit(self.model, train_dataloader, val_dataloader)