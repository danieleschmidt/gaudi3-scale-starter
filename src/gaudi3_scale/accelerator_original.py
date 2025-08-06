"""Gaudi 3 accelerator integration for PyTorch Lightning."""

import os
from typing import Any, Dict, List, Optional, Union

try:
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.accelerators import Accelerator
    _torch_available = True
except ImportError:
    torch = None
    pl = None
    Accelerator = object
    _torch_available = False

# Import habana frameworks for easier mocking in tests
try:
    import habana_frameworks.torch as htorch
    _habana_available = True
except ImportError:
    htorch = None
    _habana_available = False


class GaudiAccelerator(Accelerator):
    """PyTorch Lightning accelerator for Intel Gaudi HPUs.
    
    This accelerator provides optimized training on Intel Gaudi 3 devices
    with automatic mixed precision and distributed training support.
    
    Features:
        - Automatic HPU device detection and management
        - Optimized environment configuration
        - Memory monitoring and statistics
        - Integration with PyTorch Lightning training workflows
    
    Example:
        >>> accelerator = GaudiAccelerator()
        >>> trainer = pl.Trainer(accelerator=accelerator, devices=8)
    """
    
    def __init__(self) -> None:
        """Initialize Gaudi accelerator.
        
        Raises:
            RuntimeError: If Habana frameworks are not available.
        """
        super().__init__()
        self._check_habana_availability()
        self._setup_environment()
    
    def _check_habana_availability(self) -> None:
        """Check if Habana frameworks are available.
        
        Raises:
            RuntimeError: If habana-torch-plugin is not installed.
        """
        if htorch is None:
            raise RuntimeError(
                "Habana frameworks not found. Please install habana-torch-plugin."
            )
    
    def _setup_environment(self) -> None:
        """Setup optimal Gaudi environment variables for training.
        
        Configures Habana graph compiler and memory optimization settings
        for best performance on Gaudi 3 devices.
        """
        # Optimal Habana graph compiler settings
        os.environ.setdefault('PT_HPU_LAZY_MODE', '1')
        os.environ.setdefault('PT_HPU_ENABLE_LAZY_COMPILATION', '1')
        os.environ.setdefault('PT_HPU_GRAPH_COMPILER_OPT_LEVEL', '3')
        os.environ.setdefault('PT_HPU_MAX_COMPOUND_OP_SIZE', '256')
        os.environ.setdefault('PT_HPU_ENABLE_SYNAPSE_LAYOUT_OPT', '1')
        
        # Memory optimizations
        os.environ.setdefault('PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE', '1')
        os.environ.setdefault('PT_HPU_POOL_STRATEGY', 'OPTIMIZE_UTILIZATION')
    
    @staticmethod
    def is_available() -> bool:
        """Check if Gaudi accelerator is available.
        
        Returns:
            bool: True if Habana HPU devices are available, False otherwise.
        """
        if htorch is None:
            return False
        return htorch.hpu.is_available()
    
    def parse_devices(self, devices: Union[int, str, List[int]]) -> Union[int, List[int]]:
        """Parse device specification for HPU devices.
        
        Args:
            devices: Device specification - int, string, or list of device indices.
        
        Returns:
            Union[int, List[int]]: Parsed device specification.
            
        Raises:
            ValueError: If device specification is invalid.
        """
        if devices == "auto":
            return self.auto_device_count()
        
        if isinstance(devices, str):
            if devices.isdigit():
                return int(devices)
            else:
                raise ValueError(f"Invalid device specification: {devices}")
        
        if isinstance(devices, int):
            if devices < 0:
                raise ValueError(f"Device count must be non-negative, got {devices}")
            return devices
        
        if isinstance(devices, list):
            if not all(isinstance(d, int) and d >= 0 for d in devices):
                raise ValueError(f"All device indices must be non-negative integers: {devices}")
            return devices
        
        raise ValueError(f"Unsupported device specification type: {type(devices)}")
    
    def get_parallel_devices(self, devices: Union[int, List[int]]) -> List[Any]:
        """Convert device indices to HPU device objects.
        
        Args:
            devices: Device count or list of device indices.
        
        Returns:
            List[torch.device]: List of HPU device objects.
            
        Raises:
            ValueError: If device specification is invalid.
        """
        if torch is None:
            raise ImportError("PyTorch not available")
        
        if isinstance(devices, int):
            if devices <= 0:
                raise ValueError(f"Device count must be positive, got {devices}")
            return [torch.device(f"hpu:{i}") for i in range(devices)]
        
        if isinstance(devices, list):
            available_count = self.auto_device_count()
            for device_idx in devices:
                if device_idx >= available_count:
                    raise ValueError(
                        f"Device index {device_idx} not available. "
                        f"Only {available_count} HPU devices found."
                    )
            return [torch.device(f"hpu:{i}") for i in devices]
        
        raise ValueError(f"Unsupported device specification: {devices}")
    
    def auto_device_count(self) -> int:
        """Get the number of available HPU devices.
        
        Returns:
            int: Number of available HPU devices.
        """
        if htorch is None:
            return 0
        if htorch.hpu.is_available():
            return htorch.hpu.device_count()
        return 0
    
    def get_device_stats(self, device: Union[Any, str, int]) -> Dict[str, Any]:
        """Get HPU device statistics for monitoring.
        
        Args:
            device: Device to get statistics for.
            
        Returns:
            Dict[str, Any]: Dictionary containing device statistics.
        """
        if htorch is None:
            return {}
        
        try:
            # Convert device to proper format if needed
            if isinstance(device, (str, int)):
                device_idx = int(str(device).replace('hpu:', ''))
            elif hasattr(device, 'index'):
                device_idx = device.index
            else:
                device_idx = 0
            
            # Create device object - handle case where torch is None
            if torch is not None:
                device_obj = torch.device(f"hpu:{device_idx}")
            else:
                # Use device string for htorch calls when torch is not available
                device_obj = f"hpu:{device_idx}"
            
            stats = {
                "hpu_memory_allocated": htorch.hpu.memory_allocated(device_obj),
                "hpu_memory_reserved": htorch.hpu.memory_reserved(device_obj),
                "hpu_device_count": htorch.hpu.device_count(),
                "hpu_current_device": htorch.hpu.current_device(),
            }
            
            # Add device name if available
            try:
                stats["hpu_device_name"] = htorch.hpu.get_device_name(device_obj)
            except (AttributeError, RuntimeError):
                stats["hpu_device_name"] = f"HPU {device_idx}"
            
            return stats
        except Exception as e:
            # Return basic info if detailed stats fail
            return {"error": str(e), "device": str(device)}
    
    @classmethod
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        """Register the Gaudi accelerator with Lightning CLI.
        
        Args:
            accelerator_registry: Lightning accelerator registry.
        """
        accelerator_registry("hpu", cls)