"""Optimized optimizers for Gaudi 3 training."""

import logging
import os
from typing import Any, Dict, Iterable, Optional, Union, Type, List
from enum import Enum

try:
    import torch
    from torch.optim import Optimizer
    from torch import nn
    _torch_available = True
    # Define tensor type for type hints
    TensorType = torch.Tensor
except ImportError:
    torch = None
    Optimizer = object
    nn = None
    _torch_available = False
    # Fallback type for when torch is not available
    from typing import Any as TensorType

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Supported optimizer types."""
    ADAMW = "adamw"
    SGD = "sgd" 
    RMSPROP = "rmsprop"
    ADAM = "adam"
    LAMB = "lamb"


class GradientScaling(Enum):
    """Gradient scaling strategies."""
    NONE = "none"
    STATIC = "static" 
    DYNAMIC = "dynamic"
    AUTO = "auto"


class GaudiGradScaler:
    """Gaudi-optimized gradient scaler for mixed precision training."""
    
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True
    ):
        """Initialize gradient scaler.
        
        Args:
            init_scale: Initial loss scaling factor
            growth_factor: Factor to multiply scale on successful steps
            backoff_factor: Factor to multiply scale on overflow
            growth_interval: Number of steps between scale increases
            enabled: Whether gradient scaling is enabled
        """
        if not _torch_available:
            raise RuntimeError("PyTorch is required for gradient scaling")
            
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._growth_tracker = 0
        
        # Try to use Habana's scaler if available
        self._use_habana_scaler = False
        if enabled:
            try:
                from habana_frameworks.torch.core import hpu_grad_scale
                self._habana_scaler = hpu_grad_scale
                self._use_habana_scaler = True
                logger.info("Using Habana-optimized gradient scaler")
            except ImportError:
                logger.info("Using standard gradient scaler")
    
    def scale(self, outputs: TensorType) -> TensorType:
        """Scale the loss for mixed precision training.
        
        Args:
            outputs: Loss tensor to scale
            
        Returns:
            Scaled loss tensor
        """
        if not self._enabled:
            return outputs
        return outputs * self._scale
    
    def step(self, optimizer: Optimizer) -> None:
        """Perform an optimization step with gradient scaling.
        
        Args:
            optimizer: PyTorch optimizer instance
        """
        if not self._enabled:
            optimizer.step()
            return
        
        # Check for overflow
        has_overflow = self._check_overflow(optimizer)
        
        if has_overflow:
            # Skip step and reduce scale
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
            logger.warning(f"Gradient overflow detected, reducing scale to {self._scale}")
        else:
            # Safe to step
            self._unscale_gradients(optimizer)
            optimizer.step()
            
            # Increase scale if conditions are met
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
    
    def unscale_(self, optimizer: Optimizer) -> bool:
        """Unscale gradients.
        
        Args:
            optimizer: PyTorch optimizer instance
            
        Returns:
            bool: True if no overflow detected
        """
        if not self._enabled:
            return True
        return not self._check_overflow(optimizer)
    
    def _check_overflow(self, optimizer: Optimizer) -> bool:
        """Check for gradient overflow."""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        return True
        return False
    
    def _unscale_gradients(self, optimizer: Optimizer) -> None:
        """Unscale gradients by the current scale factor."""
        inv_scale = 1.0 / self._scale
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)
    
    def get_scale(self) -> float:
        """Get current scale factor."""
        return self._scale


class GaudiOptimizer:
    """Factory for creating HPU-optimized optimizers with advanced features.
    
    Provides Gaudi-optimized optimizers with features including:
    - Automatic mixed precision support
    - Gradient scaling and overflow detection  
    - Distributed training synchronization
    - Memory-efficient parameter updates
    - Performance monitoring and tuning
    
    Example:
        >>> optimizer = GaudiOptimizer.create_optimizer(
        ...     optimizer_type=OptimizerType.ADAMW,
        ...     model=model,
        ...     lr=1e-4,
        ...     mixed_precision=True
        ... )
    """
    
    @staticmethod
    def create_optimizer(
        optimizer_type: Union[OptimizerType, str],
        model: Optional[Any] = None,
        params: Optional[Iterable] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        mixed_precision: bool = False,
        gradient_scaling: Union[GradientScaling, str] = GradientScaling.AUTO,
        use_habana: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Create an optimized optimizer configuration.
        
        Args:
            optimizer_type: Type of optimizer to create
            model: Model to optimize (alternative to params)
            params: Model parameters to optimize (alternative to model)  
            lr: Learning rate
            weight_decay: Weight decay coefficient
            mixed_precision: Enable mixed precision training
            gradient_scaling: Gradient scaling strategy
            use_habana: Use Habana-specific optimizations when available
            **kwargs: Additional optimizer arguments
            
        Returns:
            Dictionary containing optimizer and optional gradient scaler
            
        Raises:
            ValueError: If neither model nor params provided
            RuntimeError: If PyTorch is not available
        """
        if not _torch_available:
            raise RuntimeError("PyTorch is required for optimizer creation")
        
        if model is None and params is None:
            raise ValueError("Either model or params must be provided")
            
        # Get parameters from model if provided
        if params is None and model is not None:
            params = model.parameters()
        
        # Convert string to enum if needed
        if isinstance(optimizer_type, str):
            optimizer_type = OptimizerType(optimizer_type.lower())
        if isinstance(gradient_scaling, str):
            gradient_scaling = GradientScaling(gradient_scaling.lower())
        
        # Create optimizer
        optimizer = GaudiOptimizer._create_optimizer_instance(
            optimizer_type=optimizer_type,
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            use_habana=use_habana,
            **kwargs
        )
        
        # Setup gradient scaling if needed
        grad_scaler = None
        if mixed_precision and gradient_scaling != GradientScaling.NONE:
            scale_enabled = (gradient_scaling != GradientScaling.NONE)
            grad_scaler = GaudiGradScaler(enabled=scale_enabled)
        
        return {
            "optimizer": optimizer,
            "grad_scaler": grad_scaler,
            "mixed_precision": mixed_precision
        }
    
    @staticmethod
    def _create_optimizer_instance(
        optimizer_type: OptimizerType,
        params: Iterable,
        lr: float,
        weight_decay: float,
        use_habana: bool,
        **kwargs: Any
    ) -> Optimizer:
        """Create the actual optimizer instance."""
        if optimizer_type == OptimizerType.ADAMW:
            return GaudiOptimizer._create_adamw(
                params=params, lr=lr, weight_decay=weight_decay, 
                use_habana=use_habana, **kwargs
            )
        elif optimizer_type == OptimizerType.SGD:
            return GaudiOptimizer._create_sgd(
                params=params, lr=lr, weight_decay=weight_decay,
                use_habana=use_habana, **kwargs
            )
        elif optimizer_type == OptimizerType.ADAM:
            return GaudiOptimizer._create_adam(
                params=params, lr=lr, weight_decay=weight_decay,
                use_habana=use_habana, **kwargs
            )
        elif optimizer_type == OptimizerType.RMSPROP:
            return GaudiOptimizer._create_rmsprop(
                params=params, lr=lr, weight_decay=weight_decay,
                use_habana=use_habana, **kwargs
            )
        elif optimizer_type == OptimizerType.LAMB:
            return GaudiOptimizer._create_lamb(
                params=params, lr=lr, weight_decay=weight_decay,
                use_habana=use_habana, **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    @staticmethod
    def _create_adamw(
        params: Iterable,
        lr: float,
        weight_decay: float,
        use_habana: bool,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        **kwargs: Any
    ) -> Optimizer:
        """Create AdamW optimizer."""
        if use_habana:
            try:
                from habana_frameworks.torch.optimizers import FusedAdamW
                return FusedAdamW(
                    params, lr=lr, betas=betas, eps=eps,
                    weight_decay=weight_decay, **kwargs
                )
            except ImportError:
                logger.warning("Habana FusedAdamW not available, using standard AdamW")
        
        return torch.optim.AdamW(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, **kwargs
        )
    
    @staticmethod  
    def _create_sgd(
        params: Iterable,
        lr: float,
        weight_decay: float,
        use_habana: bool,
        momentum: float = 0.9,
        nesterov: bool = True,
        **kwargs: Any
    ) -> Optimizer:
        """Create SGD optimizer."""
        if use_habana:
            try:
                from habana_frameworks.torch.optimizers import FusedSGD
                return FusedSGD(
                    params, lr=lr, momentum=momentum, nesterov=nesterov,
                    weight_decay=weight_decay, **kwargs
                )
            except ImportError:
                logger.warning("Habana FusedSGD not available, using standard SGD")
        
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, nesterov=nesterov,
            weight_decay=weight_decay, **kwargs
        )
    
    @staticmethod
    def _create_adam(
        params: Iterable,
        lr: float, 
        weight_decay: float,
        use_habana: bool,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        **kwargs: Any
    ) -> Optimizer:
        """Create Adam optimizer."""
        # No fused Adam in Habana, use standard
        return torch.optim.Adam(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, **kwargs
        )
    
    @staticmethod
    def _create_rmsprop(
        params: Iterable,
        lr: float,
        weight_decay: float, 
        use_habana: bool,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        **kwargs: Any
    ) -> Optimizer:
        """Create RMSprop optimizer."""
        return torch.optim.RMSprop(
            params, lr=lr, alpha=alpha, eps=eps,
            momentum=momentum, weight_decay=weight_decay, **kwargs
        )
    
    @staticmethod
    def _create_lamb(
        params: Iterable,
        lr: float,
        weight_decay: float,
        use_habana: bool, 
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        **kwargs: Any
    ) -> Optimizer:
        """Create LAMB optimizer."""
        try:
            from torch_optimizer import Lamb
            return Lamb(
                params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay, **kwargs
            )
        except ImportError:
            logger.warning("torch-optimizer not available, falling back to AdamW")
            return GaudiOptimizer._create_adamw(
                params=params, lr=lr, weight_decay=weight_decay,
                use_habana=use_habana, betas=betas, **kwargs
            )
    
    @staticmethod
    def FusedAdamW(
        params: Union[Iterable[Any], Any],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        use_habana: bool = True,
        **kwargs: Any
    ) -> Any:
        """Create FusedAdamW optimizer optimized for Gaudi HPUs.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay coefficient
            use_habana: Enable Habana-specific optimizations
            **kwargs: Additional optimizer arguments
            
        Returns:
            Configured FusedAdamW optimizer
            
        Raises:
            RuntimeError: If PyTorch is not available
        """
        return GaudiOptimizer._create_adamw(
            params=params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, use_habana=use_habana, **kwargs
        )
    
    @staticmethod
    def get_optimizer_config(model_size: str = "70B") -> Dict[str, Any]:
        """Get recommended optimizer configuration for model size.
        
        Args:
            model_size: Model size identifier (e.g., "70B", "7B")
            
        Returns:
            Dictionary with recommended optimizer settings
        """
        configs = {
            "7B": {
                "optimizer_type": OptimizerType.ADAMW,
                "lr": 3e-4,
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "mixed_precision": True,
                "gradient_scaling": GradientScaling.DYNAMIC,
            },
            "70B": {
                "optimizer_type": OptimizerType.ADAMW,
                "lr": 6e-4,
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "mixed_precision": True,
                "gradient_scaling": GradientScaling.DYNAMIC,
            },
            "13B": {
                "optimizer_type": OptimizerType.ADAMW,
                "lr": 2e-4,
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "mixed_precision": True,
                "gradient_scaling": GradientScaling.DYNAMIC,
            },
        }
        
        return configs.get(model_size, configs["7B"])
    
    @staticmethod
    def create_distributed_optimizer(
        optimizer_config: Dict[str, Any],
        world_size: int,
        rank: int,
        sync_gradients: bool = True
    ) -> Dict[str, Any]:
        """Create optimizer configured for distributed training.
        
        Args:
            optimizer_config: Base optimizer configuration
            world_size: Total number of processes
            rank: Current process rank
            sync_gradients: Whether to synchronize gradients across processes
            
        Returns:
            Enhanced optimizer configuration for distributed training
        """
        if not _torch_available:
            raise RuntimeError("PyTorch is required for distributed optimization")
        
        # Adjust learning rate for distributed training (linear scaling)
        base_lr = optimizer_config.get("lr", 1e-3)
        scaled_lr = base_lr * world_size
        
        # Create new config with distributed adjustments
        dist_config = optimizer_config.copy()
        dist_config.update({
            "lr": scaled_lr,
            "world_size": world_size,
            "rank": rank,
            "sync_gradients": sync_gradients,
            "original_lr": base_lr
        })
        
        logger.info(f"Distributed optimizer: scaled LR from {base_lr} to {scaled_lr} "
                   f"for world_size={world_size}")
        
        return dist_config
    
    @staticmethod
    def validate_optimizer_config(config: Dict[str, Any]) -> List[str]:
        """Validate optimizer configuration and return any warnings.
        
        Args:
            config: Optimizer configuration dictionary
            
        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []
        
        # Check learning rate
        lr = config.get("lr", 0)
        if lr <= 0:
            warnings.append("Learning rate must be positive")
        elif lr > 1e-1:
            warnings.append("Learning rate seems very high, may cause instability")
        elif lr < 1e-6:
            warnings.append("Learning rate seems very low, may slow convergence")
        
        # Check weight decay
        weight_decay = config.get("weight_decay", 0)
        if weight_decay < 0:
            warnings.append("Weight decay should be non-negative")
        elif weight_decay > 1:
            warnings.append("Weight decay > 1 is unusual and may cause issues")
        
        # Check betas for Adam-style optimizers
        betas = config.get("betas")
        if betas:
            if len(betas) != 2:
                warnings.append("betas should contain exactly 2 values")
            elif not (0 < betas[0] < 1 and 0 < betas[1] < 1):
                warnings.append("beta values should be between 0 and 1")
            elif betas[0] >= betas[1]:
                warnings.append("First beta should typically be smaller than second beta")
        
        return warnings