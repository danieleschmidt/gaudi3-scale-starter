#!/usr/bin/env python3
"""Manual test script for GaudiAccelerator functionality."""

import sys
import importlib.util
from unittest.mock import Mock, patch, MagicMock

# Load the accelerator module directly
spec = importlib.util.spec_from_file_location('accelerator', './gaudi3_scale/accelerator.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

GaudiAccelerator = module.GaudiAccelerator

def test_basic_functionality():
    """Test basic functionality without habana frameworks."""
    print("=== Testing Basic Functionality ===")
    
    # Test is_available
    print(f"is_available(): {GaudiAccelerator.is_available()}")
    
    # Test initialization (should fail)
    try:
        acc = GaudiAccelerator()
        print("ERROR: Should have failed without habana frameworks")
    except RuntimeError as e:
        print(f"✓ Expected RuntimeError: {e}")

def test_with_mock_habana():
    """Test functionality with mocked habana frameworks."""
    print("\n=== Testing with Mocked Habana Frameworks ===")
    
    # Create mock habana frameworks
    mock_htorch = MagicMock()
    mock_htorch.hpu.is_available.return_value = True
    mock_htorch.hpu.device_count.return_value = 8
    mock_htorch.hpu.memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
    mock_htorch.hpu.memory_reserved.return_value = 2 * 1024 * 1024 * 1024  # 2GB
    mock_htorch.hpu.current_device.return_value = 0
    mock_htorch.hpu.get_device_name.return_value = "Intel Gaudi 3"
    
    with patch.dict('sys.modules', {'habana_frameworks.torch': mock_htorch}):
        # Also need to mock torch for device creation
        mock_torch = MagicMock()
        mock_torch.device = lambda x: MagicMock(index=int(x.split(':')[1]) if ':' in x else 0)
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            # Reload the module to pick up the mocked dependencies
            importlib.reload(module)
            GaudiAccelerator = module.GaudiAccelerator
            
            print(f"✓ is_available(): {GaudiAccelerator.is_available()}")
            
            # Create accelerator instance
            acc = GaudiAccelerator()
            print("✓ Successfully created GaudiAccelerator instance")
            
            # Test auto_device_count
            device_count = acc.auto_device_count()
            print(f"✓ auto_device_count(): {device_count}")
            
            # Test parse_devices
            parsed = acc.parse_devices(4)
            print(f"✓ parse_devices(4): {parsed}")
            
            parsed_list = acc.parse_devices([0, 1, 2])
            print(f"✓ parse_devices([0,1,2]): {parsed_list}")
            
            # Test get_parallel_devices
            devices = acc.get_parallel_devices(4)
            print(f"✓ get_parallel_devices(4): {len(devices)} devices")
            
            # Test get_device_stats
            stats = acc.get_device_stats("hpu:0")
            print(f"✓ get_device_stats('hpu:0'): {list(stats.keys())}")
            
            # Test CLI registration
            mock_registry = Mock()
            GaudiAccelerator.register_accelerators(mock_registry)
            print("✓ register_accelerators() completed")

def test_error_handling():
    """Test error handling scenarios."""
    print("\n=== Testing Error Handling ===")
    
    # Mock minimal habana for error testing
    mock_htorch = MagicMock()
    mock_htorch.hpu.is_available.return_value = True
    mock_htorch.hpu.device_count.return_value = 2  # Limited devices
    
    with patch.dict('sys.modules', {'habana_frameworks.torch': mock_htorch}):
        mock_torch = MagicMock()
        mock_torch.device = lambda x: MagicMock()
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            importlib.reload(module)
            GaudiAccelerator = module.GaudiAccelerator
            
            acc = GaudiAccelerator()
            
            # Test invalid device parsing
            try:
                acc.parse_devices("invalid")
                print("ERROR: Should have failed with invalid device")
            except ValueError as e:
                print(f"✓ Expected ValueError for invalid device: {e}")
            
            # Test negative device count
            try:
                acc.parse_devices(-1)
                print("ERROR: Should have failed with negative device count")
            except ValueError as e:
                print(f"✓ Expected ValueError for negative count: {e}")
            
            # Test device index out of range  
            try:
                acc.get_parallel_devices([0, 1, 5])  # Device 5 doesn't exist (only 2 devices)
                print("ERROR: Should have failed with out of range device")
            except ValueError as e:
                print(f"✓ Expected ValueError for out of range device: {e}")

if __name__ == "__main__":
    test_basic_functionality()
    test_with_mock_habana() 
    test_error_handling()
    print("\n=== All Tests Completed Successfully! ===")