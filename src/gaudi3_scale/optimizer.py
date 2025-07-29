"""Optimized optimizers for Gaudi 3 training."""

from typing import Any, Dict, Iterable, Optional

try:
    import torch
    from torch.optim import Optimizer
except ImportError:
    torch = None
    Optimizer = object


class GaudiOptimizer:
    """Factory for creating HPU-optimized optimizers."""
    
    @staticmethod
    def FusedAdamW(
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        use_habana: bool = True,
        **kwargs: Any
    ) -> Optimizer:
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
        """
        if use_habana:
            try:
                from habana_frameworks.torch.optimizers import FusedAdamW
                return FusedAdamW(
                    params,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                    **kwargs
                )
            except ImportError:
                pass
        
        # Fallback to standard AdamW
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
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
                "lr": 3e-4,
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
            },
            "70B": {
                "lr": 6e-4,
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
            },
        }
        
        return configs.get(model_size, configs["7B"])