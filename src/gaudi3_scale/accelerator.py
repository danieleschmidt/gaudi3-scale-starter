"""Gaudi 3 accelerator integration for PyTorch Lightning."""

from typing import Any, Dict, Optional

try:
    import pytorch_lightning as pl
    from pytorch_lightning.accelerators import Accelerator
except ImportError:
    pl = None
    Accelerator = object


class GaudiAccelerator(Accelerator):
    """PyTorch Lightning accelerator for Intel Gaudi HPUs.
    
    This accelerator provides optimized training on Intel Gaudi 3 devices
    with automatic mixed precision and distributed training support.
    """
    
    def __init__(self) -> None:
        """Initialize Gaudi accelerator."""
        super().__init__()
        self._check_habana_availability()
    
    def _check_habana_availability(self) -> None:
        """Check if Habana frameworks are available."""
        try:
            import habana_frameworks.torch as htorch  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Habana frameworks not found. Please install habana-torch-plugin."
            )
    
    @staticmethod
    def is_available() -> bool:
        """Check if Gaudi accelerator is available."""
        try:
            import habana_frameworks.torch as htorch
            return htorch.hpu.is_available()
        except ImportError:
            return False
    
    def get_device_stats(self, device: Any) -> Dict[str, Any]:
        """Get device statistics."""
        try:
            import habana_frameworks.torch as htorch
            return {
                "hpu_memory_allocated": htorch.hpu.memory_allocated(device),
                "hpu_memory_reserved": htorch.hpu.memory_reserved(device),
            }
        except ImportError:
            return {}