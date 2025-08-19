"""Mock HPU implementation for development and testing without actual Gaudi hardware.

This module provides a complete simulation of Gaudi HPU functionality that allows
developers to test and develop applications without requiring actual Intel Gaudi hardware.
"""

import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MockHPUDevice:
    """Simulates a single Gaudi HPU device."""
    
    def __init__(self, device_id: int = 0, memory_gb: int = 96):
        self.device_id = device_id
        self.memory_total = memory_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.memory_used = 0
        self.is_available = True
        self.utilization = 0.0
        self.temperature = 45.0  # Celsius
        self.power_usage = 400.0  # Watts
        
    def allocate_memory(self, size_bytes: int) -> bool:
        """Simulate memory allocation."""
        if self.memory_used + size_bytes <= self.memory_total:
            self.memory_used += size_bytes
            return True
        return False
        
    def free_memory(self, size_bytes: int):
        """Simulate memory deallocation."""
        self.memory_used = max(0, self.memory_used - size_bytes)
        
    def get_memory_info(self) -> Dict[str, int]:
        """Get memory usage information."""
        return {
            'total': self.memory_total,
            'used': self.memory_used,
            'free': self.memory_total - self.memory_used
        }
        
    def set_utilization(self, utilization: float):
        """Set device utilization (0.0 to 1.0)."""
        self.utilization = max(0.0, min(1.0, utilization))
        # Simulate temperature increase with utilization
        self.temperature = 45.0 + (self.utilization * 35.0)  # 45-80°C
        self.power_usage = 400.0 + (self.utilization * 200.0)  # 400-600W


class MockHabanaFrameworks:
    """Mock implementation of habana_frameworks.torch module."""
    
    def __init__(self, num_devices: int = 8):
        self.devices = [MockHPUDevice(i) for i in range(num_devices)]
        self.current_device = 0
        
    def device_count(self) -> int:
        """Return number of available HPU devices."""
        return len(self.devices)
        
    def is_available(self) -> bool:
        """Check if HPUs are available."""
        return len(self.devices) > 0
        
    def current_device(self) -> int:
        """Get current device ID."""
        return self.current_device
        
    def set_device(self, device_id: int):
        """Set current device."""
        if 0 <= device_id < len(self.devices):
            self.current_device = device_id
        else:
            raise ValueError(f"Invalid device ID: {device_id}")
            
    def get_device_name(self, device_id: Optional[int] = None) -> str:
        """Get device name."""
        if device_id is None:
            device_id = self.current_device
        return f"Intel Gaudi 3 (Simulated) - Device {device_id}"
        
    def get_device_properties(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """Get device properties."""
        if device_id is None:
            device_id = self.current_device
            
        device = self.devices[device_id]
        return {
            'name': self.get_device_name(device_id),
            'major': 3,
            'minor': 0,
            'total_memory': device.memory_total,
            'multi_processor_count': 32,
            'is_integrated': False,
            'is_multi_gpu_board': False,
            'driver_version': '1.16.0 (Simulated)',
            'pci_bus_id': f'0000:{device_id:02d}:00.0'
        }
        
    def memory_stats(self, device_id: Optional[int] = None) -> Dict[str, int]:
        """Get memory statistics."""
        if device_id is None:
            device_id = self.current_device
            
        device = self.devices[device_id]
        return device.get_memory_info()
        
    def utilization(self, device_id: Optional[int] = None) -> float:
        """Get device utilization."""
        if device_id is None:
            device_id = self.current_device
            
        return self.devices[device_id].utilization
        
    def temperature(self, device_id: Optional[int] = None) -> float:
        """Get device temperature."""
        if device_id is None:
            device_id = self.current_device
            
        return self.devices[device_id].temperature
        
    def power_usage(self, device_id: Optional[int] = None) -> float:
        """Get device power usage."""
        if device_id is None:
            device_id = self.current_device
            
        return self.devices[device_id].power_usage
        
    def synchronize(self, device_id: Optional[int] = None):
        """Simulate device synchronization."""
        time.sleep(0.001)  # Simulate small sync delay
        
    def empty_cache(self):
        """Simulate cache clearing."""
        for device in self.devices:
            # Simulate freeing some cached memory
            device.memory_used = int(device.memory_used * 0.8)


class MockTensorOperations:
    """Mock tensor operations for simulated training."""
    
    @staticmethod
    def simulate_forward_pass(batch_size: int, model_size: str = "medium") -> Tuple[float, float]:
        """Simulate forward pass timing and memory usage."""
        size_multipliers = {"small": 0.5, "medium": 1.0, "large": 2.0, "xl": 4.0}
        multiplier = size_multipliers.get(model_size, 1.0)
        
        # Simulate processing time (ms)
        base_time = 10.0 * multiplier
        time_ms = base_time + (batch_size * 0.5 * multiplier) + random.uniform(-2.0, 2.0)
        
        # Simulate memory usage (MB)
        base_memory = 100.0 * multiplier
        memory_mb = base_memory + (batch_size * 2.0 * multiplier)
        
        return max(0.1, time_ms), max(1.0, memory_mb)
        
    @staticmethod
    def simulate_backward_pass(forward_time: float, forward_memory: float) -> Tuple[float, float]:
        """Simulate backward pass timing and memory usage."""
        # Backward pass typically takes 1.5-2x forward pass time
        backward_time = forward_time * (1.5 + random.uniform(-0.2, 0.5))
        
        # Backward pass uses additional memory for gradients
        backward_memory = forward_memory * 1.8
        
        return backward_time, backward_memory
        
    @staticmethod
    def simulate_optimizer_step(param_count: int) -> float:
        """Simulate optimizer step timing."""
        # Time scales with parameter count
        base_time = 5.0  # ms
        param_time = (param_count / 1_000_000) * 2.0  # 2ms per million params
        return base_time + param_time + random.uniform(-1.0, 1.0)


# Global mock instance
_mock_habana = None
_mock_enabled = False


def enable_mock_mode(num_devices: int = 8):
    """Enable mock HPU mode for development/testing."""
    global _mock_habana, _mock_enabled
    _mock_habana = MockHabanaFrameworks(num_devices)
    _mock_enabled = True
    logger.info(f"Mock HPU mode enabled with {num_devices} simulated devices")


def disable_mock_mode():
    """Disable mock HPU mode."""
    global _mock_habana, _mock_enabled
    _mock_habana = None
    _mock_enabled = False
    logger.info("Mock HPU mode disabled")


def is_mock_enabled() -> bool:
    """Check if mock mode is enabled."""
    return _mock_enabled


def get_mock_instance() -> Optional[MockHabanaFrameworks]:
    """Get the mock habana instance."""
    return _mock_habana if _mock_enabled else None


# Simulate training environment detection
def detect_environment() -> str:
    """Detect the current environment type."""
    import os
    
    # Check for common CI environments
    ci_indicators = [
        'CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 
        'TRAVIS', 'CIRCLECI', 'JENKINS_URL'
    ]
    
    if any(os.getenv(indicator) for indicator in ci_indicators):
        return 'ci'
    
    # Check for container environments
    if os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER'):
        return 'container'
        
    # Check for development indicators
    dev_indicators = ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt']
    if any(os.path.exists(indicator) for indicator in dev_indicators):
        return 'development'
        
    return 'unknown'


def auto_enable_mock_if_needed():
    """Automatically enable mock mode if appropriate."""
    import os
    
    # Auto-enable in certain environments
    auto_enable_envs = ['ci', 'container', 'development']
    current_env = detect_environment()
    
    # Check if user explicitly disabled mocking
    if os.getenv('GAUDI3_DISABLE_MOCK', '').lower() in ('1', 'true', 'yes'):
        logger.info("Mock mode disabled by user preference (GAUDI3_DISABLE_MOCK)")
        return
        
    # Check if user explicitly enabled mocking
    if os.getenv('GAUDI3_ENABLE_MOCK', '').lower() in ('1', 'true', 'yes'):
        num_devices = int(os.getenv('GAUDI3_MOCK_DEVICES', '8'))
        enable_mock_mode(num_devices)
        return
        
    # Auto-enable based on environment
    if current_env in auto_enable_envs and not _mock_enabled:
        num_devices = int(os.getenv('GAUDI3_MOCK_DEVICES', '8'))
        enable_mock_mode(num_devices)
        logger.info(f"Auto-enabled mock mode for {current_env} environment")


# Context managers for temporary mock mode
@contextmanager
def temporary_mock_mode(num_devices: int = 8):
    """Temporary context manager for mock mode."""
    was_enabled = _mock_enabled
    old_instance = _mock_habana
    
    try:
        enable_mock_mode(num_devices)
        yield get_mock_instance()
    finally:
        if was_enabled:
            globals()['_mock_habana'] = old_instance
            globals()['_mock_enabled'] = True
        else:
            disable_mock_mode()


# Performance simulation utilities
class MockPerformanceProfiler:
    """Mock profiler for simulating performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, name: str):
        """Start a performance timer."""
        self.start_times[name] = time.time()
        
    def end_timer(self, name: str) -> float:
        """End a performance timer and return elapsed time."""
        if name not in self.start_times:
            return 0.0
            
        elapsed = time.time() - self.start_times[name]
        self.metrics[name] = elapsed
        del self.start_times[name]
        return elapsed
        
    def get_metrics(self) -> Dict[str, float]:
        """Get all collected metrics."""
        return self.metrics.copy()
        
    def simulate_training_metrics(self, epoch: int, batch_size: int) -> Dict[str, float]:
        """Simulate realistic training metrics."""
        # Simulate metrics that improve over time
        base_loss = 2.0
        epoch_improvement = epoch * 0.15
        noise = random.uniform(-0.1, 0.1)
        
        loss = max(0.1, base_loss - epoch_improvement + noise)
        accuracy = min(0.95, 0.3 + epoch_improvement * 0.8 + noise * 0.1)
        
        # Simulate batch processing time
        batch_time = MockTensorOperations.simulate_forward_pass(batch_size)[0] / 1000.0
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'batch_time': batch_time,
            'throughput': batch_size / batch_time,
            'memory_usage': batch_size * 2.5,  # MB per sample
            'hpu_utilization': 0.7 + random.uniform(-0.1, 0.2)
        }


# Initialize mock mode automatically if appropriate
auto_enable_mock_if_needed()