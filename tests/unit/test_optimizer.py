"""Unit tests for GaudiOptimizer."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from gaudi3_scale.optimizer import (
    GaudiOptimizer, 
    GaudiGradScaler, 
    OptimizerType, 
    GradientScaling
)


class TestGaudiOptimizer:
    """Test suite for GaudiOptimizer class."""
    
    def test_get_optimizer_config_7b(self):
        """Test optimizer configuration for 7B model."""
        config = GaudiOptimizer.get_optimizer_config("7B")
        
        assert config["optimizer_type"] == OptimizerType.ADAMW
        assert config["lr"] == 3e-4
        assert config["weight_decay"] == 0.1
        assert config["betas"] == (0.9, 0.95)
        assert config["eps"] == 1e-8
        assert config["mixed_precision"] is True
        assert config["gradient_scaling"] == GradientScaling.DYNAMIC
    
    def test_get_optimizer_config_70b(self):
        """Test optimizer configuration for 70B model."""
        config = GaudiOptimizer.get_optimizer_config("70B")
        
        assert config["optimizer_type"] == OptimizerType.ADAMW
        assert config["lr"] == 6e-4
        assert config["weight_decay"] == 0.1
        assert config["betas"] == (0.9, 0.95)
        assert config["eps"] == 1e-8
        assert config["mixed_precision"] is True
        assert config["gradient_scaling"] == GradientScaling.DYNAMIC
    
    def test_get_optimizer_config_13b(self):
        """Test optimizer configuration for 13B model."""
        config = GaudiOptimizer.get_optimizer_config("13B")
        
        assert config["optimizer_type"] == OptimizerType.ADAMW
        assert config["lr"] == 2e-4
        assert config["weight_decay"] == 0.1
        assert config["betas"] == (0.9, 0.95)
        assert config["eps"] == 1e-8
        assert config["mixed_precision"] is True
        assert config["gradient_scaling"] == GradientScaling.DYNAMIC
    
    def test_get_optimizer_config_unknown_defaults_to_7b(self):
        """Test unknown model size defaults to 7B configuration."""
        config = GaudiOptimizer.get_optimizer_config("unknown")
        
        assert config["optimizer_type"] == OptimizerType.ADAMW
        assert config["lr"] == 3e-4
        assert config["weight_decay"] == 0.1
        assert config["betas"] == (0.9, 0.95)
        assert config["eps"] == 1e-8
        assert config["mixed_precision"] is True
        assert config["gradient_scaling"] == GradientScaling.DYNAMIC
    
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

    @patch('gaudi3_scale.optimizer.torch')
    def test_create_optimizer_with_model(self, mock_torch):
        """Test create_optimizer with model parameter."""
        mock_model = Mock()
        mock_params = [Mock()]
        mock_model.parameters.return_value = mock_params
        mock_optimizer = Mock()
        mock_torch.optim.AdamW.return_value = mock_optimizer
        
        result = GaudiOptimizer.create_optimizer(
            optimizer_type=OptimizerType.ADAMW,
            model=mock_model,
            lr=1e-3
        )
        
        assert "optimizer" in result
        assert "grad_scaler" in result
        assert "mixed_precision" in result
        assert result["optimizer"] == mock_optimizer
        assert result["mixed_precision"] is False
        assert result["grad_scaler"] is None
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_create_optimizer_with_params(self, mock_torch):
        """Test create_optimizer with params parameter."""
        mock_params = [Mock()]
        mock_optimizer = Mock()
        mock_torch.optim.AdamW.return_value = mock_optimizer
        
        result = GaudiOptimizer.create_optimizer(
            optimizer_type=OptimizerType.ADAMW,
            params=mock_params,
            lr=1e-3
        )
        
        assert result["optimizer"] == mock_optimizer
    
    def test_create_optimizer_no_model_or_params_raises(self):
        """Test create_optimizer raises when neither model nor params provided."""
        with pytest.raises(ValueError, match="Either model or params must be provided"):
            GaudiOptimizer.create_optimizer(
                optimizer_type=OptimizerType.ADAMW,
                lr=1e-3
            )
    
    @patch('gaudi3_scale.optimizer._torch_available', False)
    def test_create_optimizer_no_torch_raises(self):
        """Test create_optimizer raises when PyTorch not available."""
        with pytest.raises(RuntimeError, match="PyTorch is required"):
            GaudiOptimizer.create_optimizer(
                optimizer_type=OptimizerType.ADAMW,
                params=[Mock()],
                lr=1e-3
            )
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_create_optimizer_mixed_precision_with_grad_scaler(self, mock_torch):
        """Test create_optimizer with mixed precision creates grad_scaler."""
        mock_params = [Mock()]
        mock_optimizer = Mock()
        mock_torch.optim.AdamW.return_value = mock_optimizer
        
        result = GaudiOptimizer.create_optimizer(
            optimizer_type=OptimizerType.ADAMW,
            params=mock_params,
            mixed_precision=True,
            gradient_scaling=GradientScaling.DYNAMIC
        )
        
        assert result["mixed_precision"] is True
        assert result["grad_scaler"] is not None
        assert isinstance(result["grad_scaler"], GaudiGradScaler)
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_create_optimizer_string_types(self, mock_torch):
        """Test create_optimizer with string optimizer type."""
        mock_params = [Mock()]
        mock_optimizer = Mock()
        mock_torch.optim.SGD.return_value = mock_optimizer
        
        result = GaudiOptimizer.create_optimizer(
            optimizer_type="sgd",
            params=mock_params,
            gradient_scaling="static"
        )
        
        assert result["optimizer"] == mock_optimizer
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_create_sgd_optimizer(self, mock_torch):
        """Test creating SGD optimizer."""
        mock_params = [Mock()]
        mock_optimizer = Mock()
        mock_torch.optim.SGD.return_value = mock_optimizer
        
        result = GaudiOptimizer.create_optimizer(
            optimizer_type=OptimizerType.SGD,
            params=mock_params,
            lr=1e-2,
            momentum=0.9
        )
        
        mock_torch.optim.SGD.assert_called_once()
        assert result["optimizer"] == mock_optimizer
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_create_adam_optimizer(self, mock_torch):
        """Test creating Adam optimizer."""
        mock_params = [Mock()]
        mock_optimizer = Mock()
        mock_torch.optim.Adam.return_value = mock_optimizer
        
        result = GaudiOptimizer.create_optimizer(
            optimizer_type=OptimizerType.ADAM,
            params=mock_params,
            lr=1e-3
        )
        
        mock_torch.optim.Adam.assert_called_once()
        assert result["optimizer"] == mock_optimizer
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_create_rmsprop_optimizer(self, mock_torch):
        """Test creating RMSprop optimizer."""
        mock_params = [Mock()]
        mock_optimizer = Mock()
        mock_torch.optim.RMSprop.return_value = mock_optimizer
        
        result = GaudiOptimizer.create_optimizer(
            optimizer_type=OptimizerType.RMSPROP,
            params=mock_params,
            lr=1e-3
        )
        
        mock_torch.optim.RMSprop.assert_called_once()
        assert result["optimizer"] == mock_optimizer
    
    def test_create_distributed_optimizer(self):
        """Test create_distributed_optimizer adjusts learning rate."""
        base_config = {"lr": 1e-4, "weight_decay": 0.1}
        
        dist_config = GaudiOptimizer.create_distributed_optimizer(
            optimizer_config=base_config,
            world_size=8,
            rank=0
        )
        
        assert dist_config["lr"] == 8e-4  # 8x scaling
        assert dist_config["world_size"] == 8
        assert dist_config["rank"] == 0
        assert dist_config["sync_gradients"] is True
        assert dist_config["original_lr"] == 1e-4
    
    def test_validate_optimizer_config_valid(self):
        """Test validate_optimizer_config with valid config."""
        config = {
            "lr": 1e-4,
            "weight_decay": 0.1,
            "betas": (0.9, 0.999)
        }
        
        warnings = GaudiOptimizer.validate_optimizer_config(config)
        assert len(warnings) == 0
    
    def test_validate_optimizer_config_invalid_lr(self):
        """Test validate_optimizer_config with invalid learning rate."""
        config = {"lr": -1e-4}
        warnings = GaudiOptimizer.validate_optimizer_config(config)
        assert any("Learning rate must be positive" in w for w in warnings)
        
        config = {"lr": 1.0}  # Very high LR
        warnings = GaudiOptimizer.validate_optimizer_config(config)
        assert any("very high" in w for w in warnings)
        
        config = {"lr": 1e-8}  # Very low LR
        warnings = GaudiOptimizer.validate_optimizer_config(config)
        assert any("very low" in w for w in warnings)
    
    def test_validate_optimizer_config_invalid_weight_decay(self):
        """Test validate_optimizer_config with invalid weight decay."""
        config = {"lr": 1e-4, "weight_decay": -0.1}
        warnings = GaudiOptimizer.validate_optimizer_config(config)
        assert any("Weight decay should be non-negative" in w for w in warnings)
        
        config = {"lr": 1e-4, "weight_decay": 1.5}
        warnings = GaudiOptimizer.validate_optimizer_config(config)
        assert any("Weight decay > 1" in w for w in warnings)
    
    def test_validate_optimizer_config_invalid_betas(self):
        """Test validate_optimizer_config with invalid betas."""
        config = {"lr": 1e-4, "betas": (0.9,)}  # Wrong length
        warnings = GaudiOptimizer.validate_optimizer_config(config)
        assert any("exactly 2 values" in w for w in warnings)
        
        config = {"lr": 1e-4, "betas": (1.1, 0.999)}  # Out of range
        warnings = GaudiOptimizer.validate_optimizer_config(config)
        assert any("between 0 and 1" in w for w in warnings)
        
        config = {"lr": 1e-4, "betas": (0.999, 0.9)}  # Wrong order
        warnings = GaudiOptimizer.validate_optimizer_config(config)
        assert any("smaller than second beta" in w for w in warnings)


class TestGaudiGradScaler:
    """Test suite for GaudiGradScaler class."""
    
    @patch('gaudi3_scale.optimizer._torch_available', False)
    def test_init_no_torch_raises(self):
        """Test GaudiGradScaler raises when PyTorch not available."""
        with pytest.raises(RuntimeError, match="PyTorch is required"):
            GaudiGradScaler()
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_init_with_habana_scaler(self, mock_torch):
        """Test GaudiGradScaler initialization with Habana scaler."""
        mock_hpu_grad_scale = Mock()
        
        with patch('habana_frameworks.torch.core.hpu_grad_scale', mock_hpu_grad_scale):
            scaler = GaudiGradScaler(enabled=True)
            assert scaler._use_habana_scaler is True
            assert scaler._habana_scaler == mock_hpu_grad_scale
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_init_without_habana_scaler(self, mock_torch):
        """Test GaudiGradScaler initialization without Habana scaler."""
        with patch.dict('sys.modules', {'habana_frameworks.torch.core': None}):
            scaler = GaudiGradScaler(enabled=True)
            assert scaler._use_habana_scaler is False
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_scale_disabled(self, mock_torch):
        """Test scale method when scaler is disabled."""
        scaler = GaudiGradScaler(enabled=False)
        mock_tensor = Mock()
        
        result = scaler.scale(mock_tensor)
        assert result == mock_tensor
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_scale_enabled(self, mock_torch):
        """Test scale method when scaler is enabled."""
        scaler = GaudiGradScaler(enabled=True, init_scale=1024.0)
        mock_tensor = Mock()
        mock_scaled = Mock()
        mock_tensor.__mul__ = Mock(return_value=mock_scaled)
        
        result = scaler.scale(mock_tensor)
        mock_tensor.__mul__.assert_called_once_with(1024.0)
        assert result == mock_scaled
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_step_disabled(self, mock_torch):
        """Test step method when scaler is disabled."""
        scaler = GaudiGradScaler(enabled=False)
        mock_optimizer = Mock()
        
        scaler.step(mock_optimizer)
        mock_optimizer.step.assert_called_once()
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_step_no_overflow(self, mock_torch):
        """Test step method with no gradient overflow."""
        scaler = GaudiGradScaler(enabled=True)
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [{"params": []}]
        
        scaler.step(mock_optimizer)
        mock_optimizer.step.assert_called_once()
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_step_with_overflow(self, mock_torch):
        """Test step method with gradient overflow."""
        mock_torch.isinf.return_value.any.return_value = True
        
        scaler = GaudiGradScaler(enabled=True, init_scale=1024.0, backoff_factor=0.5)
        mock_optimizer = Mock()
        mock_param = Mock()
        mock_param.grad = Mock()
        mock_optimizer.param_groups = [{"params": [mock_param]}]
        
        initial_scale = scaler.get_scale()
        scaler.step(mock_optimizer)
        
        # Should not call optimizer.step() on overflow
        mock_optimizer.step.assert_not_called()
        
        # Scale should be reduced
        assert scaler.get_scale() == initial_scale * 0.5
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_unscale_disabled(self, mock_torch):
        """Test unscale_ method when scaler is disabled."""
        scaler = GaudiGradScaler(enabled=False)
        mock_optimizer = Mock()
        
        result = scaler.unscale_(mock_optimizer)
        assert result is True
    
    @patch('gaudi3_scale.optimizer.torch')
    def test_get_scale(self, mock_torch):
        """Test get_scale method."""
        init_scale = 2048.0
        scaler = GaudiGradScaler(enabled=True, init_scale=init_scale)
        
        assert scaler.get_scale() == init_scale


class TestOptimizerTypes:
    """Test suite for optimizer type enums."""
    
    def test_optimizer_type_enum_values(self):
        """Test OptimizerType enum values."""
        assert OptimizerType.ADAMW.value == "adamw"
        assert OptimizerType.SGD.value == "sgd"
        assert OptimizerType.RMSPROP.value == "rmsprop"
        assert OptimizerType.ADAM.value == "adam"
        assert OptimizerType.LAMB.value == "lamb"
    
    def test_gradient_scaling_enum_values(self):
        """Test GradientScaling enum values."""
        assert GradientScaling.NONE.value == "none"
        assert GradientScaling.STATIC.value == "static"
        assert GradientScaling.DYNAMIC.value == "dynamic"
        assert GradientScaling.AUTO.value == "auto"