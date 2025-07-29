"""Integration tests for training pipeline."""

import pytest
from unittest.mock import Mock, patch

from gaudi3_scale import GaudiAccelerator, GaudiOptimizer, GaudiTrainer


class TestTrainingPipeline:
    """Integration tests for complete training pipeline."""
    
    @patch('gaudi3_scale.trainer.pl')
    @patch('gaudi3_scale.accelerator.htorch')
    def test_end_to_end_training_setup(self, mock_htorch, mock_pl):
        """Test complete training pipeline setup."""
        # Setup mocks
        mock_htorch.hpu.is_available.return_value = True
        mock_trainer_instance = Mock()
        mock_pl.Trainer.return_value = mock_trainer_instance
        
        # Create components
        accelerator = GaudiAccelerator()
        optimizer_config = GaudiOptimizer.get_optimizer_config("7B")
        trainer = GaudiTrainer(accelerator="hpu")
        
        # Verify components work together
        assert accelerator.is_available()
        assert optimizer_config["lr"] == 3e-4
        
        pl_trainer = trainer.create_trainer()
        assert pl_trainer == mock_trainer_instance
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_optimizer_integration(self, mock_torch):
        """Test optimizer integration with model parameters."""
        mock_params = [Mock(), Mock()]
        mock_optimizer = Mock()
        mock_torch.optim.AdamW.return_value = mock_optimizer
        
        # Get configuration and create optimizer
        config = GaudiOptimizer.get_optimizer_config("70B")
        optimizer = GaudiOptimizer.FusedAdamW(
            mock_params,
            **config,
            use_habana=False
        )
        
        # Verify optimizer was created with correct config
        mock_torch.optim.AdamW.assert_called_once_with(
            mock_params,
            lr=6e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        assert optimizer == mock_optimizer