"""High-level training interface for Gaudi 3 models."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        DeviceStatsMonitor,
        RichProgressBar
    )
    import torch
    _torch_available = True
except ImportError:
    pl = None
    torch = None
    _torch_available = False

# Import project components
from .models.training import TrainingConfig, ModelConfig, DatasetConfig
from .monitoring.metrics import MetricsCollector, TrainingMetrics
from .accelerator import GaudiAccelerator

# Setup logger
logger = logging.getLogger(__name__)


class GaudiTrainingError(Exception):
    """Custom exception for Gaudi training errors."""
    pass


class GaudiValidationError(Exception):
    """Custom exception for Gaudi training validation errors."""
    pass


class GaudiTrainerCallback:
    """Base callback class for GaudiTrainer."""
    
    def on_train_start(self, trainer: 'GaudiTrainer') -> None:
        """Called when training starts."""
        pass
    
    def on_train_end(self, trainer: 'GaudiTrainer') -> None:
        """Called when training ends."""
        pass
    
    def on_epoch_start(self, trainer: 'GaudiTrainer', epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: 'GaudiTrainer', epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_start(self, trainer: 'GaudiTrainer', batch_idx: int) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, trainer: 'GaudiTrainer', batch_idx: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each batch."""
        pass


class MetricsCallback(GaudiTrainerCallback):
    """Callback for collecting and reporting training metrics."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize metrics callback.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.epoch_start_time = 0.0
        self.batch_start_time = 0.0
    
    def on_train_start(self, trainer: 'GaudiTrainer') -> None:
        """Log training start."""
        logger.info(f"Starting training for model: {trainer.model_name}")
        logger.info(f"Training configuration: {trainer.get_training_summary()}")
    
    def on_epoch_start(self, trainer: 'GaudiTrainer', epoch: int) -> None:
        """Record epoch start time."""
        self.epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch}")
    
    def on_epoch_end(self, trainer: 'GaudiTrainer', epoch: int, logs: Dict[str, Any]) -> None:
        """Collect epoch metrics."""
        epoch_duration = time.time() - self.epoch_start_time
        
        metrics = self.metrics_collector.collect_training_metrics(
            model_name=trainer.model_name,
            epoch=epoch,
            step=logs.get('step', 0),
            loss=logs.get('loss', 0.0),
            accuracy=logs.get('accuracy', 0.0),
            learning_rate=logs.get('lr', 0.0),
            throughput=logs.get('throughput', 0.0)
        )
        
        logger.info(
            f"Epoch {epoch} completed in {epoch_duration:.2f}s - "
            f"Loss: {metrics.loss:.4f}, Accuracy: {metrics.accuracy:.4f}"
        )
    
    def on_train_end(self, trainer: 'GaudiTrainer') -> None:
        """Log training completion and summary."""
        summary = self.metrics_collector.get_training_summary(trainer.model_name)
        logger.info(f"Training completed. Summary: {summary}")


class GaudiTrainer:
    """High-level trainer for Gaudi 3 models.
    
    This class provides a comprehensive interface for training models
    on Intel Gaudi 3 hardware with optimized settings, monitoring,
    and error handling.
    
    Features:
        - Automatic HPU environment setup and optimization
        - Integrated metrics collection and monitoring
        - Comprehensive error handling and validation
        - Flexible callback system for custom training logic
        - Checkpoint management and recovery
        - Integration with PyTorch Lightning
    
    Example:
        >>> trainer = GaudiTrainer(
        ...     model=my_lightning_model,
        ...     config=training_config,
        ...     output_dir="./checkpoints"
        ... )
        >>> trainer.fit(train_dataloader, val_dataloader)
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[TrainingConfig] = None,
        model_name: str = "gaudi_model",
        output_dir: str = "./output",
        accelerator: str = "hpu",
        devices: Union[int, str, List[int]] = 8,
        precision: str = "bf16-mixed",
        max_epochs: int = 3,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 4,
        strategy: str = "ddp",
        callbacks: Optional[List[GaudiTrainerCallback]] = None,
        enable_monitoring: bool = True,
        enable_checkpointing: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize Gaudi trainer.
        
        Args:
            model: PyTorch Lightning model to train
            config: Training configuration object
            model_name: Name of the model for logging and metrics
            output_dir: Directory for outputs and checkpoints
            accelerator: Accelerator type ("hpu" for Gaudi)
            devices: Number/list of devices to use
            precision: Training precision
            max_epochs: Maximum training epochs
            gradient_clip_val: Gradient clipping value
            accumulate_grad_batches: Gradient accumulation steps
            strategy: Distributed training strategy
            callbacks: List of custom callbacks
            enable_monitoring: Enable metrics collection
            enable_checkpointing: Enable checkpoint saving
            **kwargs: Additional trainer arguments
            
        Raises:
            GaudiValidationError: If configuration is invalid
            RuntimeError: If required dependencies are not available
        """
        # Validate dependencies
        if not _torch_available:
            raise RuntimeError("PyTorch and PyTorch Lightning are required")
        
        # Initialize core attributes
        self.model = model
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.enable_monitoring = enable_monitoring
        self.enable_checkpointing = enable_checkpointing
        
        # Use config if provided, otherwise create from parameters
        if config is not None:
            self.config = config
            # Override config with explicit parameters if provided
            if max_epochs != 3:  # Check if explicitly set
                self.config.max_epochs = max_epochs
            if precision != "bf16-mixed":
                # Note: This would need proper enum handling in a real implementation
                pass  # Keep the config precision for now
        else:
            # Create basic config from parameters
            self.config = TrainingConfig(
                max_epochs=max_epochs,
                gradient_accumulation_steps=accumulate_grad_batches,
                gradient_clip_val=gradient_clip_val,
                output_dir=str(output_dir)
            )
        
        # Store additional parameters
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision  # Store original precision string
        self.strategy = strategy
        self.kwargs = kwargs
        
        # Initialize components
        self.callbacks = callbacks or []
        self.metrics_collector = MetricsCollector() if enable_monitoring else None
        self._trainer: Optional[Any] = None
        self._gaudi_accelerator: Optional[GaudiAccelerator] = None
        
        # Setup
        self._validate_configuration()
        self._setup_environment()
        self._setup_output_directory()
        self._setup_callbacks()
        
        logger.info(f"GaudiTrainer initialized for model '{self.model_name}'")
    
    def _validate_configuration(self) -> None:
        """Validate trainer configuration.
        
        Raises:
            GaudiValidationError: If configuration is invalid
        """
        try:
            # Validate device specification
            if isinstance(self.devices, int) and self.devices <= 0:
                raise GaudiValidationError("Device count must be positive")
            
            # Validate epochs
            if self.config.max_epochs <= 0:
                raise GaudiValidationError("max_epochs must be positive")
            
            # Validate gradient clipping
            if self.config.gradient_clip_val < 0:
                raise GaudiValidationError("gradient_clip_val must be non-negative")
            
            # Validate batch size
            if self.config.batch_size <= 0:
                raise GaudiValidationError("batch_size must be positive")
            
            # Check HPU availability if using HPU accelerator
            if self.accelerator == "hpu":
                self._gaudi_accelerator = GaudiAccelerator()
                if not self._gaudi_accelerator.is_available():
                    logger.warning("HPU devices not detected, training may fail")
            
        except Exception as e:
            raise GaudiValidationError(f"Configuration validation failed: {str(e)}")
    
    def _setup_environment(self) -> None:
        """Setup Gaudi environment variables and optimizations."""
        logger.info("Setting up Gaudi environment variables")
        
        # Optimal Habana graph compiler settings
        env_vars = {
            'PT_HPU_LAZY_MODE': '1',
            'PT_HPU_ENABLE_LAZY_COMPILATION': '1',
            'PT_HPU_GRAPH_COMPILER_OPT_LEVEL': '3',
            'PT_HPU_MAX_COMPOUND_OP_SIZE': '256',
            'PT_HPU_ENABLE_SYNAPSE_LAYOUT_OPT': '1',
            'PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE': '1',
            'PT_HPU_POOL_STRATEGY': 'OPTIMIZE_UTILIZATION',
        }
        
        # Apply environment variables
        for var, value in env_vars.items():
            os.environ.setdefault(var, value)
            logger.debug(f"Set {var}={value}")
        
        # Additional optimizations from config
        if hasattr(self.config, 'use_lazy_mode') and self.config.use_lazy_mode:
            os.environ.setdefault('PT_HPU_LAZY_MODE', '1')
        
        if hasattr(self.config, 'use_hpu_graphs') and self.config.use_hpu_graphs:
            os.environ.setdefault('PT_HPU_ENABLE_GRAPHS', '1')
    
    def _setup_output_directory(self) -> None:
        """Setup output directory structure."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "checkpoints").mkdir(exist_ok=True)
            (self.output_dir / "logs").mkdir(exist_ok=True)
            logger.info(f"Output directory setup: {self.output_dir}")
        except Exception as e:
            raise GaudiTrainingError(f"Failed to setup output directory: {str(e)}")
    
    def _setup_callbacks(self) -> None:
        """Setup default and user callbacks."""
        # Add metrics callback if monitoring is enabled
        if self.enable_monitoring and self.metrics_collector:
            self.callbacks.append(MetricsCallback(self.metrics_collector))
            logger.info("Added metrics collection callback")
    
    def _create_lightning_callbacks(self) -> List[Any]:
        """Create PyTorch Lightning callbacks.
        
        Returns:
            List of Lightning callbacks
        """
        lightning_callbacks = []
        
        # Checkpoint callback
        if self.enable_checkpointing:
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.output_dir / "checkpoints",
                filename=f"{self.model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                verbose=True
            )
            lightning_callbacks.append(checkpoint_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        lightning_callbacks.append(lr_monitor)
        
        # Device stats monitor for HPU
        if self.accelerator == "hpu":
            device_stats = DeviceStatsMonitor()
            lightning_callbacks.append(device_stats)
        
        # Progress bar
        progress_bar = RichProgressBar()
        lightning_callbacks.append(progress_bar)
        
        # Early stopping if validation data is expected
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=True,
            mode="min"
        )
        lightning_callbacks.append(early_stopping)
        
        return lightning_callbacks
    
    def create_trainer(self) -> Any:
        """Create configured PyTorch Lightning trainer.
        
        Returns:
            Configured PyTorch Lightning trainer
            
        Raises:
            ImportError: If PyTorch Lightning is not available
            GaudiTrainingError: If trainer creation fails
        """
        if pl is None:
            raise ImportError("PyTorch Lightning not available")
        
        try:
            # Get Lightning callbacks
            lightning_callbacks = self._create_lightning_callbacks()
            
            # Configure trainer
            trainer_config = {
                "accelerator": self._gaudi_accelerator or self.accelerator,
                "devices": self.devices,
                "precision": self.precision,
                "max_epochs": self.config.max_epochs,
                "gradient_clip_val": self.config.gradient_clip_val,
                "accumulate_grad_batches": self.config.gradient_accumulation_steps,
                "strategy": self.strategy,
                "callbacks": lightning_callbacks,
                "log_every_n_steps": getattr(self.config, 'logging_steps', 10),
                "enable_checkpointing": self.enable_checkpointing,
                "enable_progress_bar": True,
                "enable_model_summary": True,
                **self.kwargs
            }
            
            # Remove None values
            trainer_config = {k: v for k, v in trainer_config.items() if v is not None}
            
            logger.info("Creating PyTorch Lightning trainer")
            logger.debug(f"Trainer config: {trainer_config}")
            
            return pl.Trainer(**trainer_config)
            
        except Exception as e:
            raise GaudiTrainingError(f"Failed to create trainer: {str(e)}")
    
    def fit(
        self,
        train_dataloader: Any,
        val_dataloader: Optional[Any] = None,
        ckpt_path: Optional[str] = None
    ) -> None:
        """Fit the model using configured trainer.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            ckpt_path: Path to checkpoint to resume from (optional)
            
        Raises:
            ValueError: If model is not set
            GaudiTrainingError: If training fails
        """
        if self.model is None:
            raise ValueError("Model must be set before training")
        
        try:
            # Call custom callbacks
            for callback in self.callbacks:
                callback.on_train_start(self)
            
            # Create trainer if not exists
            if self._trainer is None:
                self._trainer = self.create_trainer()
            
            logger.info(f"Starting training with {self.config.max_epochs} epochs")
            
            # Start training
            self._trainer.fit(
                model=self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=ckpt_path
            )
            
            # Call custom callbacks
            for callback in self.callbacks:
                callback.on_train_end(self)
                
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise GaudiTrainingError(f"Training failed: {str(e)}")
    
    def validate(self, val_dataloader: Any, ckpt_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run validation on the model.
        
        Args:
            val_dataloader: Validation data loader
            ckpt_path: Path to checkpoint to load (optional)
            
        Returns:
            List of validation results
            
        Raises:
            ValueError: If model is not set
            GaudiTrainingError: If validation fails
        """
        if self.model is None:
            raise ValueError("Model must be set before validation")
        
        try:
            # Create trainer if not exists
            if self._trainer is None:
                self._trainer = self.create_trainer()
            
            logger.info("Starting validation")
            
            results = self._trainer.validate(
                model=self.model,
                dataloaders=val_dataloader,
                ckpt_path=ckpt_path
            )
            
            logger.info("Validation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise GaudiTrainingError(f"Validation failed: {str(e)}")
    
    def test(self, test_dataloader: Any, ckpt_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run testing on the model.
        
        Args:
            test_dataloader: Test data loader
            ckpt_path: Path to checkpoint to load (optional)
            
        Returns:
            List of test results
            
        Raises:
            ValueError: If model is not set
            GaudiTrainingError: If testing fails
        """
        if self.model is None:
            raise ValueError("Model must be set before testing")
        
        try:
            # Create trainer if not exists
            if self._trainer is None:
                self._trainer = self.create_trainer()
            
            logger.info("Starting testing")
            
            results = self._trainer.test(
                model=self.model,
                dataloaders=test_dataloader,
                ckpt_path=ckpt_path
            )
            
            logger.info("Testing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Testing failed: {str(e)}")
            raise GaudiTrainingError(f"Testing failed: {str(e)}")
    
    def get_device_stats(self) -> Dict[str, Any]:
        """Get current device statistics.
        
        Returns:
            Dictionary containing device statistics
        """
        if self._gaudi_accelerator:
            return self._gaudi_accelerator.get_device_stats(0)  # Get stats for device 0
        return {}
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training configuration summary.
        
        Returns:
            Dictionary containing training summary
        """
        return {
            "model_name": self.model_name,
            "accelerator": self.accelerator,
            "devices": self.devices,
            "precision": self.precision,
            "max_epochs": self.config.max_epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "gradient_clip_val": self.config.gradient_clip_val,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "strategy": self.strategy,
            "output_dir": str(self.output_dir),
            "enable_monitoring": self.enable_monitoring,
            "enable_checkpointing": self.enable_checkpointing
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get training metrics summary.
        
        Returns:
            Dictionary containing metrics summary
        """
        if self.metrics_collector:
            return self.metrics_collector.get_training_summary(self.model_name)
        return {}
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint manually.
        
        Args:
            filepath: Path to save checkpoint
            
        Raises:
            ValueError: If model or trainer is not set
            GaudiTrainingError: If checkpoint saving fails
        """
        if self.model is None:
            raise ValueError("Model must be set before saving checkpoint")
        
        if self._trainer is None:
            raise ValueError("Trainer must be created before saving checkpoint")
        
        try:
            self._trainer.save_checkpoint(filepath)
            logger.info(f"Checkpoint saved to {filepath}")
        except Exception as e:
            raise GaudiTrainingError(f"Failed to save checkpoint: {str(e)}")
    
    def load_from_checkpoint(self, checkpoint_path: str, model: Optional[Any] = None) -> Any:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model class to instantiate (if not using existing model)
            
        Returns:
            Loaded model instance
            
        Raises:
            GaudiTrainingError: If checkpoint loading fails
        """
        try:
            if model is not None:
                # Load new model from checkpoint
                loaded_model = model.load_from_checkpoint(checkpoint_path)
                logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
                return loaded_model
            else:
                # This would require the model class to be known
                raise ValueError("Model class must be provided for checkpoint loading")
                
        except Exception as e:
            raise GaudiTrainingError(f"Failed to load checkpoint: {str(e)}")
    
    def set_model(self, model: Any) -> None:
        """Set the model to be trained.
        
        Args:
            model: PyTorch Lightning model
        """
        self.model = model
        logger.info(f"Model set: {type(model).__name__}")
    
    def add_callback(self, callback: GaudiTrainerCallback) -> None:
        """Add a custom callback.
        
        Args:
            callback: Custom callback to add
        """
        self.callbacks.append(callback)
        logger.info(f"Added callback: {type(callback).__name__}")
    
    def remove_callback(self, callback_type: type) -> None:
        """Remove callbacks of specific type.
        
        Args:
            callback_type: Type of callback to remove
        """
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback_type)]
        logger.info(f"Removed callbacks of type: {callback_type.__name__}")
    
    def __repr__(self) -> str:
        """String representation of the trainer."""
        return (
            f"GaudiTrainer(model_name='{self.model_name}', "
            f"accelerator='{self.accelerator}', devices={self.devices}, "
            f"precision='{self.precision}', max_epochs={self.config.max_epochs})"
        )