#\!/usr/bin/env python3
import sys
import importlib.util
from unittest.mock import Mock, patch, MagicMock

# Load the accelerator module directly
spec = importlib.util.spec_from_file_location('accelerator', './gaudi3_scale/accelerator.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

GaudiAccelerator = module.GaudiAccelerator

def test_basic_functionality():
    print("=== Testing Basic Functionality ===")
    print(f"is_available(): {GaudiAccelerator.is_available()}")
    
    try:
        acc = GaudiAccelerator()
        print("ERROR: Should have failed without habana frameworks")
    except RuntimeError as e:
        print(f"✓ Expected RuntimeError: {e}")

def test_with_mock_habana():
    print("\n=== Testing with Mocked Habana Frameworks ===")
    
    mock_htorch = MagicMock()
    mock_htorch.hpu.is_available.return_value = True
    mock_htorch.hpu.device_count.return_value = 8
    mock_htorch.hpu.memory_allocated.return_value = 1024 * 1024 * 1024
    mock_htorch.hpu.memory_reserved.return_value = 2 * 1024 * 1024 * 1024
    mock_htorch.hpu.current_device.return_value = 0
    mock_htorch.hpu.get_device_name.return_value = "Intel Gaudi 3"
    
    with patch.dict('sys.modules', {'habana_frameworks.torch': mock_htorch}):
        mock_torch = MagicMock()
        mock_torch.device = lambda x: MagicMock(index=int(x.split(':')[1]) if ':' in x else 0)
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            importlib.reload(module)
            GaudiAccelerator = module.GaudiAccelerator
            
            print(f"✓ is_available(): {GaudiAccelerator.is_available()}")
            
            acc = GaudiAccelerator()
            print("✓ Successfully created GaudiAccelerator instance")
            
            device_count = acc.auto_device_count()
            print(f"✓ auto_device_count(): {device_count}")
            
            parsed = acc.parse_devices(4)
            print(f"✓ parse_devices(4): {parsed}")
            
            devices = acc.get_parallel_devices(4)
            print(f"✓ get_parallel_devices(4): {len(devices)} devices")
            
            stats = acc.get_device_stats("hpu:0")
            print(f"✓ get_device_stats('hpu:0'): {list(stats.keys())}")

if __name__ == "__main__":
    test_basic_functionality()
    test_with_mock_habana()
    print("\n=== All Tests Completed Successfully\! ===")
EOF < /dev/null
