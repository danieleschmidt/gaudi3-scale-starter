"""Unit tests for GaudiTrainer."""

import os
import pytest
from unittest.mock import Mock, patch

from gaudi3_scale.trainer import GaudiTrainer


class TestGaudiTrainer:
    """Test suite for GaudiTrainer class."""
    
    def test_trainer_initialization(self):
        """Test trainer initializes with correct defaults."""
        mock_model = Mock()
        trainer = GaudiTrainer(model=mock_model)
        
        assert trainer.model == mock_model
        assert trainer.accelerator == "hpu"
        assert trainer.devices == 8
        assert trainer.precision == "bf16-mixed"
        assert trainer.max_epochs == 3
        assert trainer.gradient_clip_val == 1.0
        assert trainer.accumulate_grad_batches == 4
        assert trainer.strategy == "ddp"
    
    def test_trainer_custom_parameters(self):
        """Test trainer accepts custom parameters."""
        mock_model = Mock()
        trainer = GaudiTrainer(
            model=mock_model,
            devices=16,
            max_epochs=5,
            precision="fp32"
        )
        
        assert trainer.devices == 16
        assert trainer.max_epochs == 5
        assert trainer.precision == "fp32"
    
    def test_setup_environment(self):
        """Test environment variables are set correctly."""
        trainer = GaudiTrainer()
        
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
    
    @patch('gaudi3_scale.trainer.pl')
    def test_create_trainer_with_pytorch_lightning(self, mock_pl):
        """Test trainer creation with PyTorch Lightning."""
        mock_model = Mock()
        mock_trainer_instance = Mock()
        mock_pl.Trainer.return_value = mock_trainer_instance
        
        trainer = GaudiTrainer(model=mock_model)
        pl_trainer = trainer.create_trainer()
        
        assert pl_trainer == mock_trainer_instance
        mock_pl.Trainer.assert_called_once()
    
    def test_create_trainer_without_pytorch_lightning(self):
        """Test trainer creation fails without PyTorch Lightning."""
        with patch('gaudi3_scale.trainer.pl', None):
            trainer = GaudiTrainer()
            
            with pytest.raises(ImportError, match="PyTorch Lightning not available"):
                trainer.create_trainer()
    
    @patch('gaudi3_scale.trainer.pl')
    def test_fit_with_model(self, mock_pl):
        """Test fitting with a model."""
        mock_model = Mock()
        mock_trainer_instance = Mock()
        mock_pl.Trainer.return_value = mock_trainer_instance
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        
        trainer = GaudiTrainer(model=mock_model)
        trainer.fit(mock_train_loader, mock_val_loader)
        
        mock_trainer_instance.fit.assert_called_once_with(
            mock_model, mock_train_loader, mock_val_loader
        )
    
    def test_fit_without_model_raises_error(self):
        """Test fitting without model raises ValueError."""
        trainer = GaudiTrainer()
        mock_train_loader = Mock()
        
        with pytest.raises(ValueError, match="Model must be set before training"):
            trainer.fit(mock_train_loader)