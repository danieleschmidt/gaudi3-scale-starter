"""
gaudi3_scale — Training utilities for Intel Gaudi 3 HPU at scale.

Provides:
- GaudiTrainer: PyTorch Lightning-compatible trainer stub
- LargeScaleBatchSampler: Gradient accumulation-aware batch sampler
- LearningRateScaler: Linear scaling rule for large-batch training
"""
from gaudi3_scale.lr_scaler import LearningRateScaler
from gaudi3_scale.sampler import LargeScaleBatchSampler
from gaudi3_scale.trainer import GaudiTrainer

__all__ = ["GaudiTrainer", "LargeScaleBatchSampler", "LearningRateScaler"]
__version__ = "0.1.0"
