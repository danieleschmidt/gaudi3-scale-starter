"""Unit tests for GaudiTrainer."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from gaudi3_scale.trainer import (
    GaudiTrainer,
    GaudiTrainerCallback,
    MetricsCallback,
    GaudiTrainingError,
    GaudiValidationError
)
from gaudi3_scale.models.training import TrainingConfig


class TestGaudiTrainerCallback:
    """Test suite for GaudiTrainerCallback base class."""
    
    def test_callback_methods_exist(self):
        """Test that all callback methods exist and are callable."""
        callback = GaudiTrainerCallback()
        trainer = Mock()
        
        # Test all methods exist and don't raise errors
        callback.on_train_start(trainer)
        callback.on_train_end(trainer)
        callback.on_epoch_start(trainer, 1)
        callback.on_epoch_end(trainer, 1, {})
        callback.on_batch_start(trainer, 1)
        callback.on_batch_end(trainer, 1, {})


class TestMetricsCallback:
    """Test suite for MetricsCallback."""
    
    @patch('gaudi3_scale.trainer.MetricsCollector')
    def test_metrics_callback_initialization(self, mock_metrics_collector):
        """Test metrics callback initializes correctly."""
        callback = MetricsCallback()
        assert callback.metrics_collector is not None
        assert callback.epoch_start_time == 0.0
        assert callback.batch_start_time == 0.0
    
    @patch('gaudi3_scale.trainer.MetricsCollector')
    @patch('gaudi3_scale.trainer.logger')
    def test_on_train_start_logging(self, mock_logger, mock_metrics_collector):
        """Test logging on training start."""
        callback = MetricsCallback()
        trainer = Mock()
        trainer.model_name = "test_model"
        trainer.get_training_summary.return_value = {"test": "config"}
        
        callback.on_train_start(trainer)
        
        assert mock_logger.info.call_count == 2


class TestGaudiTrainer:
    """Test suite for GaudiTrainer class."""
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_trainer_initialization_defaults(self, mock_accelerator):
        """Test trainer initializes with correct defaults."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(output_dir=temp_dir)
            
            assert trainer.model is None
            assert trainer.model_name == "gaudi_model"
            assert trainer.accelerator == "hpu"
            assert trainer.devices == 8
            assert trainer.precision == "bf16-mixed"
            assert trainer.config.max_epochs == 3
            assert trainer.config.gradient_clip_val == 1.0
            assert trainer.config.gradient_accumulation_steps == 4
            assert trainer.strategy == "ddp"
            assert trainer.enable_monitoring is True
            assert trainer.enable_checkpointing is True
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_trainer_custom_parameters(self, mock_accelerator):
        """Test trainer accepts custom parameters."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        mock_model = Mock()
        config = TrainingConfig(max_epochs=10, gradient_clip_val=2.0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(
                model=mock_model,
                config=config,
                model_name="custom_model",
                output_dir=temp_dir,
                devices=16,
                precision="fp32",
                enable_monitoring=False
            )
            
            assert trainer.model == mock_model
            assert trainer.model_name == "custom_model"
            assert trainer.devices == 16
            assert trainer.precision == "fp32"
            assert trainer.config.max_epochs == 10
            assert trainer.enable_monitoring is False
    
    def test_trainer_initialization_without_torch_raises_error(self):
        """Test trainer initialization fails without PyTorch."""
        with patch('gaudi3_scale.trainer._torch_available', False):
            with pytest.raises(RuntimeError, match="PyTorch and PyTorch Lightning are required"):
                GaudiTrainer()
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_validation_error_negative_devices(self, mock_accelerator):
        """Test validation error for negative device count."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(GaudiValidationError, match="Device count must be positive"):
                GaudiTrainer(devices=-1, output_dir=temp_dir)
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_validation_error_invalid_config(self, mock_accelerator):
        """Test validation error for invalid configuration."""
        config = TrainingConfig(max_epochs=0)  # Invalid
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(GaudiValidationError, match="max_epochs must be positive"):
                GaudiTrainer(config=config, output_dir=temp_dir)
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_setup_environment_sets_variables(self, mock_accelerator):
        """Test environment variables are set correctly."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(output_dir=temp_dir)
            
            expected_vars = {
                'PT_HPU_LAZY_MODE': '1',
                'PT_HPU_ENABLE_LAZY_COMPILATION': '1',
                'PT_HPU_GRAPH_COMPILER_OPT_LEVEL': '3',
                'PT_HPU_MAX_COMPOUND_OP_SIZE': '256',
                'PT_HPU_ENABLE_SYNAPSE_LAYOUT_OPT': '1',
                'PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE': '1',
                'PT_HPU_POOL_STRATEGY': 'OPTIMIZE_UTILIZATION',
            }
            
            for var, value in expected_vars.items():
                assert os.environ[var] == value
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_output_directory_creation(self, mock_accelerator):
        """Test output directory structure is created."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output"
            trainer = GaudiTrainer(output_dir=str(output_path))
            
            assert output_path.exists()
            assert (output_path / "checkpoints").exists()
            assert (output_path / "logs").exists()
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    @patch('gaudi3_scale.trainer.pl')
    def test_create_trainer_with_pytorch_lightning(self, mock_pl, mock_accelerator):
        """Test trainer creation with PyTorch Lightning."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        mock_trainer_instance = Mock()
        mock_pl.Trainer.return_value = mock_trainer_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(output_dir=temp_dir)
            pl_trainer = trainer.create_trainer()
            
            assert pl_trainer == mock_trainer_instance
            mock_pl.Trainer.assert_called_once()
            
            # Verify trainer config includes expected parameters
            call_args = mock_pl.Trainer.call_args
            config = call_args.kwargs
            assert config['accelerator'] == mock_accelerator_instance
            assert config['devices'] == 8
            assert config['precision'] == "bf16-mixed"
            assert config['max_epochs'] == 3
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_create_trainer_without_pytorch_lightning(self, mock_accelerator):
        """Test trainer creation fails without PyTorch Lightning."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('gaudi3_scale.trainer.pl', None):
                trainer = GaudiTrainer(output_dir=temp_dir)
                
                with pytest.raises(ImportError, match="PyTorch Lightning not available"):
                    trainer.create_trainer()
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    @patch('gaudi3_scale.trainer.pl')
    def test_fit_with_model(self, mock_pl, mock_accelerator):
        """Test fitting with a model."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        mock_model = Mock()
        mock_trainer_instance = Mock()
        mock_pl.Trainer.return_value = mock_trainer_instance
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(model=mock_model, output_dir=temp_dir)
            trainer.fit(mock_train_loader, mock_val_loader)
            
            mock_trainer_instance.fit.assert_called_once()
            call_args = mock_trainer_instance.fit.call_args
            assert call_args.kwargs['model'] == mock_model
            assert call_args.kwargs['train_dataloaders'] == mock_train_loader
            assert call_args.kwargs['val_dataloaders'] == mock_val_loader
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_fit_without_model_raises_error(self, mock_accelerator):
        """Test fitting without model raises ValueError."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(output_dir=temp_dir)
            mock_train_loader = Mock()
            
            with pytest.raises(ValueError, match="Model must be set before training"):
                trainer.fit(mock_train_loader)
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    @patch('gaudi3_scale.trainer.pl')
    def test_validate_with_model(self, mock_pl, mock_accelerator):
        """Test validation with a model."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        mock_model = Mock()
        mock_trainer_instance = Mock()
        mock_trainer_instance.validate.return_value = [{"val_loss": 0.5}]
        mock_pl.Trainer.return_value = mock_trainer_instance
        mock_val_loader = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(model=mock_model, output_dir=temp_dir)
            results = trainer.validate(mock_val_loader)
            
            assert results == [{"val_loss": 0.5}]
            mock_trainer_instance.validate.assert_called_once()
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    @patch('gaudi3_scale.trainer.pl')
    def test_test_with_model(self, mock_pl, mock_accelerator):
        """Test testing with a model."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        mock_model = Mock()
        mock_trainer_instance = Mock()
        mock_trainer_instance.test.return_value = [{"test_loss": 0.3}]
        mock_pl.Trainer.return_value = mock_trainer_instance
        mock_test_loader = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(model=mock_model, output_dir=temp_dir)
            results = trainer.test(mock_test_loader)
            
            assert results == [{"test_loss": 0.3}]
            mock_trainer_instance.test.assert_called_once()
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_get_device_stats(self, mock_accelerator):
        """Test getting device statistics."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator_instance.get_device_stats.return_value = {"memory": "16GB"}
        mock_accelerator.return_value = mock_accelerator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(output_dir=temp_dir)
            stats = trainer.get_device_stats()
            
            assert stats == {"memory": "16GB"}
            mock_accelerator_instance.get_device_stats.assert_called_once_with(0)
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_get_training_summary(self, mock_accelerator):
        """Test getting training summary."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(
                model_name="test_model",
                output_dir=temp_dir,
                devices=4,
                precision="fp16",
                max_epochs=5
            )
            
            summary = trainer.get_training_summary()
            
            assert summary["model_name"] == "test_model"
            assert summary["accelerator"] == "hpu"
            assert summary["devices"] == 4
            assert summary["precision"] == "fp16"
            assert summary["max_epochs"] == 5
            assert summary["enable_monitoring"] is True
            assert summary["enable_checkpointing"] is True
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    @patch('gaudi3_scale.trainer.MetricsCollector')
    def test_get_metrics_summary(self, mock_metrics_collector_class, mock_accelerator):
        """Test getting metrics summary."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        mock_metrics_instance = Mock()
        mock_metrics_instance.get_training_summary.return_value = {"total_steps": 100}
        mock_metrics_collector_class.return_value = mock_metrics_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(
                model_name="test_model",
                output_dir=temp_dir,
                enable_monitoring=True
            )
            
            summary = trainer.get_metrics_summary()
            
            assert summary == {"total_steps": 100}
            mock_metrics_instance.get_training_summary.assert_called_once_with("test_model")
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_set_model(self, mock_accelerator):
        """Test setting a model."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(output_dir=temp_dir)
            mock_model = Mock()
            mock_model.__class__.__name__ = "TestModel"
            
            trainer.set_model(mock_model)
            
            assert trainer.model == mock_model
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_add_remove_callbacks(self, mock_accelerator):
        """Test adding and removing callbacks."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(output_dir=temp_dir, enable_monitoring=False)
            
            # Initially should have no callbacks (monitoring disabled)
            assert len(trainer.callbacks) == 0
            
            # Add a callback
            custom_callback = GaudiTrainerCallback()
            trainer.add_callback(custom_callback)
            assert len(trainer.callbacks) == 1
            assert custom_callback in trainer.callbacks
            
            # Remove callback by type
            trainer.remove_callback(GaudiTrainerCallback)
            assert len(trainer.callbacks) == 0
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    @patch('gaudi3_scale.trainer.pl')
    def test_save_checkpoint(self, mock_pl, mock_accelerator):
        """Test manual checkpoint saving."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        mock_model = Mock()
        mock_trainer_instance = Mock()
        mock_pl.Trainer.return_value = mock_trainer_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(model=mock_model, output_dir=temp_dir)
            # Create trainer instance
            trainer.create_trainer()
            
            checkpoint_path = str(Path(temp_dir) / "test_checkpoint.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            
            mock_trainer_instance.save_checkpoint.assert_called_once_with(checkpoint_path)
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    def test_repr(self, mock_accelerator):
        """Test string representation."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(
                model_name="test_model",
                output_dir=temp_dir,
                devices=4,
                precision="fp16",
                max_epochs=5
            )
            
            repr_str = repr(trainer)
            
            assert "GaudiTrainer" in repr_str
            assert "test_model" in repr_str
            assert "hpu" in repr_str
            assert "4" in repr_str
            assert "fp16" in repr_str
            assert "5" in repr_str


class TestGaudiTrainerIntegration:
    """Integration tests for GaudiTrainer with real components."""
    
    @patch('gaudi3_scale.trainer._torch_available', True)
    @patch('gaudi3_scale.trainer.GaudiAccelerator')
    @patch('gaudi3_scale.trainer.MetricsCollector')
    def test_trainer_with_callbacks(self, mock_metrics_collector_class, mock_accelerator):
        """Test trainer with callbacks working together."""
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.is_available.return_value = True
        mock_accelerator.return_value = mock_accelerator_instance
        
        mock_metrics_instance = Mock()
        mock_metrics_collector_class.return_value = mock_metrics_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GaudiTrainer(
                model_name="integration_test",
                output_dir=temp_dir,
                enable_monitoring=True
            )
            
            # Should have metrics callback
            assert len(trainer.callbacks) == 1
            assert isinstance(trainer.callbacks[0], MetricsCallback)
            
            # Test callback execution
            for callback in trainer.callbacks:
                callback.on_train_start(trainer)
                callback.on_epoch_start(trainer, 1)
                callback.on_epoch_end(trainer, 1, {"loss": 0.5})
                callback.on_train_end(trainer)