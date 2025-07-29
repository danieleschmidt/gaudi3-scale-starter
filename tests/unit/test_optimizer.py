"""Unit tests for GaudiOptimizer."""

import pytest
from unittest.mock import Mock, patch

from gaudi3_scale.optimizer import GaudiOptimizer


class TestGaudiOptimizer:
    """Test suite for GaudiOptimizer class."""
    
    def test_get_optimizer_config_7b(self):
        """Test optimizer configuration for 7B model."""
        config = GaudiOptimizer.get_optimizer_config("7B")
        
        expected = {
            "lr": 3e-4,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
        }
        assert config == expected
    
    def test_get_optimizer_config_70b(self):
        """Test optimizer configuration for 70B model."""
        config = GaudiOptimizer.get_optimizer_config("70B")
        
        expected = {
            "lr": 6e-4,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
        }
        assert config == expected
    
    def test_get_optimizer_config_unknown_defaults_to_7b(self):
        """Test unknown model size defaults to 7B configuration."""
        config = GaudiOptimizer.get_optimizer_config("unknown")
        
        expected = {
            "lr": 3e-4,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
        }
        assert config == expected
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_fused_adamw_fallback_to_standard(self, mock_torch):
        """Test FusedAdamW falls back to standard AdamW when Habana not available."""
        mock_params = [Mock()]
        mock_optimizer = Mock()
        mock_torch.optim.AdamW.return_value = mock_optimizer
        
        with patch.dict('sys.modules', {'habana_frameworks.torch.optimizers': None}):
            optimizer = GaudiOptimizer.FusedAdamW(
                mock_params,
                lr=1e-4,
                use_habana=True
            )
        
        mock_torch.optim.AdamW.assert_called_once_with(
            mock_params,
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2
        )
        assert optimizer == mock_optimizer
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_fused_adamw_use_habana_false(self, mock_torch):
        """Test FusedAdamW uses standard AdamW when use_habana=False."""
        mock_params = [Mock()]
        mock_optimizer = Mock()
        mock_torch.optim.AdamW.return_value = mock_optimizer
        
        optimizer = GaudiOptimizer.FusedAdamW(
            mock_params,
            lr=2e-4,
            use_habana=False
        )
        
        mock_torch.optim.AdamW.assert_called_once_with(
            mock_params,
            lr=2e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2
        )
        assert optimizer == mock_optimizer