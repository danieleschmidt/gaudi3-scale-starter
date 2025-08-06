#!/usr/bin/env python3
"""Simple test script for GaudiTrainer without full module imports."""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, 'src')


def test_trainer_imports():
    """Test that trainer can be imported successfully."""
    print("Testing GaudiTrainer imports...")
    
    # Mock dependencies to avoid import issues
    mock_torch = MagicMock()
    mock_torch.nn = MagicMock()
    mock_torch.nn.Parameter = Mock
    
    mock_pl = MagicMock()
    mock_pl.callbacks = MagicMock()
    
    # Create a comprehensive mock for the modules
    sys.modules['torch'] = mock_torch
    sys.modules['pytorch_lightning'] = mock_pl
    sys.modules['pytorch_lightning.callbacks'] = mock_pl.callbacks
    sys.modules['habana_frameworks'] = MagicMock()
    sys.modules['habana_frameworks.torch'] = MagicMock()
    
    try:
        # Now try to import the trainer module directly
        from gaudi3_scale.trainer import (
            GaudiTrainer,
            GaudiTrainerCallback,
            MetricsCallback,
            GaudiTrainingError,
            GaudiValidationError
        )
        print("  ✓ All trainer classes imported successfully")
        
        # Test callback base class
        callback = GaudiTrainerCallback()
        assert hasattr(callback, 'on_train_start')
        assert hasattr(callback, 'on_train_end')
        assert hasattr(callback, 'on_epoch_start')
        assert hasattr(callback, 'on_epoch_end')
        print("  ✓ GaudiTrainerCallback has all required methods")
        
        # Test exception classes
        assert issubclass(GaudiTrainingError, Exception)
        assert issubclass(GaudiValidationError, Exception)
        print("  ✓ Custom exception classes defined correctly")
        
        print("✓ Import test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_structure():
    """Test trainer class structure and method signatures."""
    print("Testing GaudiTrainer structure...")
    
    # Continue with mocked modules
    with patch('gaudi3_scale.trainer._torch_available', True), \
         patch('gaudi3_scale.trainer.GaudiAccelerator') as mock_accelerator, \
         patch('gaudi3_scale.trainer.MetricsCollector') as mock_metrics:
        
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        mock_metrics_instance = Mock()
        mock_metrics.return_value = mock_metrics_instance
        
        from gaudi3_scale.trainer import GaudiTrainer
        
        # Test trainer can be instantiated
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(
                model_name="test_model",
                output_dir=temp_dir
            )
            
            # Check essential attributes exist
            assert hasattr(trainer, 'model')
            assert hasattr(trainer, 'model_name')
            assert hasattr(trainer, 'config')
            assert hasattr(trainer, 'callbacks')
            assert hasattr(trainer, 'accelerator')
            assert hasattr(trainer, 'devices')
            assert hasattr(trainer, 'precision')
            print("  ✓ Trainer has all essential attributes")
            
            # Check essential methods exist
            methods = [
                'fit', 'validate', 'test', 'create_trainer',
                'get_device_stats', 'get_training_summary', 'get_metrics_summary',
                'save_checkpoint', 'set_model', 'add_callback', 'remove_callback'
            ]
            
            for method in methods:
                assert hasattr(trainer, method), f"Missing method: {method}"
                assert callable(getattr(trainer, method)), f"Method not callable: {method}"
            
            print("  ✓ Trainer has all essential methods")
            
            # Test basic properties
            assert trainer.model_name == "test_model"
            assert trainer.accelerator == "hpu"
            assert trainer.enable_monitoring is True
            assert trainer.enable_checkpointing is True
            
            print("  ✓ Basic properties work correctly")
        
        print("✓ Structure test passed!")
        return True


def test_configuration_handling():
    """Test configuration handling."""
    print("Testing configuration handling...")
    
    with patch('gaudi3_scale.trainer._torch_available', True), \
         patch('gaudi3_scale.trainer.GaudiAccelerator') as mock_accelerator:
        
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        from gaudi3_scale.trainer import GaudiTrainer
        from gaudi3_scale.models.training import TrainingConfig
        
        # Test with custom config
        config = TrainingConfig(
            max_epochs=10,
            batch_size=64,
            learning_rate=1e-4
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(
                config=config,
                output_dir=temp_dir
            )
            
            assert trainer.config.max_epochs == 10
            assert trainer.config.batch_size == 64
            assert trainer.config.learning_rate == 1e-4
            print("  ✓ Custom configuration applied correctly")
            
            # Test training summary
            summary = trainer.get_training_summary()
            assert isinstance(summary, dict)
            assert 'model_name' in summary
            assert 'max_epochs' in summary
            assert 'batch_size' in summary
            print("  ✓ Training summary generated correctly")
        
        print("✓ Configuration test passed!")
        return True


def main():
    """Run simplified tests."""
    print("Running GaudiTrainer simplified tests...\n")
    
    success = True
    
    try:
        success &= test_trainer_imports()
        print()
        
        success &= test_trainer_structure()
        print()
        
        success &= test_configuration_handling()
        print()
        
        if success:
            print("🎉 All simplified tests passed!")
            print("\nThe GaudiTrainer implementation structure is correct with:")
            print("  • Proper class hierarchy and method signatures")
            print("  • All required methods and attributes")
            print("  • Correct configuration handling")
            print("  • Proper import structure")
            return 0
        else:
            print("❌ Some tests failed")
            return 1
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)