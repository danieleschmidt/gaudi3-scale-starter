"""Enhanced trainer with full mock HPU support for development and testing.

This module provides a production-ready trainer that works seamlessly with 
mock HPU devices, enabling development and testing without actual Gaudi hardware.
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .mock_hpu import (
    get_mock_instance, is_mock_enabled, MockPerformanceProfiler,
    MockTensorOperations
)
from .logging_utils import get_logger

logger = get_logger(__name__)


class MockGaudiTrainer:
    """Enhanced trainer with comprehensive mock HPU support."""
    
    def __init__(
        self,
        model_name: str = "default-model",
        batch_size: int = 32,
        learning_rate: float = 0.001,
        max_epochs: int = 10,
        precision: str = "bf16",
        output_dir: str = "output",
        enable_checkpointing: bool = True,
        enable_validation: bool = True,
        enable_profiling: bool = True,
        validation_split: float = 0.2,
        save_best_only: bool = True,
        early_stopping_patience: int = 5,
        devices: Union[int, List[int]] = "auto"
    ):
        """Initialize enhanced mock trainer.
        
        Args:
            model_name: Name of the model being trained
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            max_epochs: Maximum number of training epochs
            precision: Training precision (bf16, fp16, fp32)
            output_dir: Directory to save outputs
            enable_checkpointing: Enable model checkpointing
            enable_validation: Enable validation during training
            enable_profiling: Enable performance profiling
            validation_split: Fraction of data for validation
            save_best_only: Only save checkpoints that improve validation metrics
            early_stopping_patience: Epochs to wait before early stopping
            devices: Number of devices or list of device IDs
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.precision = precision
        self.output_dir = Path(output_dir)
        self.enable_checkpointing = enable_checkpointing
        self.enable_validation = enable_validation
        self.enable_profiling = enable_profiling
        self.validation_split = validation_split
        self.save_best_only = save_best_only
        self.early_stopping_patience = early_stopping_patience
        
        # Initialize components first
        self.profiler = MockPerformanceProfiler() if enable_profiling else None
        self.mock_instance = get_mock_instance() if is_mock_enabled() else None
        
        # Setup devices (depends on mock_instance)
        self.devices = self._setup_devices(devices)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'batch_time': [],
            'throughput': []
        }
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MockGaudiTrainer initialized: {model_name} on {self.devices} devices")
        
    def _setup_devices(self, devices: Union[int, List[int], str]) -> List[int]:
        """Setup device configuration."""
        if devices == "auto":
            if is_mock_enabled() and self.mock_instance:
                return list(range(min(8, self.mock_instance.device_count())))
            return [0]  # Fallback to single device
            
        if isinstance(devices, int):
            return list(range(devices))
            
        if isinstance(devices, list):
            return devices
            
        return [0]  # Default fallback
        
    def fit(
        self, 
        train_data: Optional[Any] = None,
        val_data: Optional[Any] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the model with comprehensive simulation.
        
        Args:
            train_data: Training dataset (simulated if None)
            val_data: Validation dataset (simulated if None)  
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting training: {self.model_name}")
        logger.info(f"Configuration: batch_size={self.batch_size}, lr={self.learning_rate}, "
                   f"epochs={self.max_epochs}, precision={self.precision}")
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self._load_checkpoint(resume_from_checkpoint)
            
        # Start profiling
        if self.profiler:
            self.profiler.start_timer('total_training')
            
        training_start_time = time.time()
        
        try:
            # Training loop
            for epoch in range(start_epoch, self.max_epochs):
                self.current_epoch = epoch + 1
                
                # Epoch timing
                epoch_start_time = time.time()
                
                # Training step
                train_metrics = self._train_epoch()
                
                # Validation step
                val_metrics = self._validate_epoch() if self.enable_validation else {}
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Update training history
                self._update_history(train_metrics, val_metrics, epoch_time)
                
                # Log progress
                self._log_epoch_progress(train_metrics, val_metrics, epoch_time)
                
                # Checkpointing
                if self.enable_checkpointing:
                    self._save_checkpoint(val_metrics.get('val_loss', train_metrics['loss']))
                    
                # Early stopping check
                if self._should_early_stop(val_metrics.get('val_loss', train_metrics['loss'])):
                    logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
        finally:
            # End profiling
            if self.profiler:
                total_time = self.profiler.end_timer('total_training')
                
        training_time = time.time() - training_start_time
        
        # Generate final results
        results = self._generate_results(training_time)
        
        # Save training results
        self._save_results(results)
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Final metrics: Loss={results['metrics']['final_loss']:.4f}, "
                   f"Accuracy={results['metrics']['final_accuracy']:.4f}")
                   
        return results
        
    def _train_epoch(self) -> Dict[str, float]:
        """Simulate training epoch."""
        if self.profiler:
            self.profiler.start_timer(f'train_epoch_{self.current_epoch}')
            
        # Simulate realistic training metrics based on epoch
        num_batches = 100  # Simulate 100 batches per epoch
        total_loss = 0.0
        total_accuracy = 0.0
        total_batch_time = 0.0
        
        for batch_idx in range(num_batches):
            # Simulate batch processing
            if self.mock_instance:
                # Simulate device utilization
                for device_id in self.devices:
                    utilization = 0.8 + random.uniform(-0.1, 0.15)
                    self.mock_instance.devices[device_id].set_utilization(utilization)
            
            # Simulate forward and backward pass
            forward_time, forward_memory = MockTensorOperations.simulate_forward_pass(
                self.batch_size, "medium"
            )
            backward_time, backward_memory = MockTensorOperations.simulate_backward_pass(
                forward_time, forward_memory
            )
            
            batch_time = (forward_time + backward_time) / 1000.0  # Convert to seconds
            total_batch_time += batch_time
            
            # Simulate improving metrics over batches
            batch_loss = self._simulate_batch_loss(batch_idx)
            batch_accuracy = self._simulate_batch_accuracy(batch_idx)
            
            total_loss += batch_loss
            total_accuracy += batch_accuracy
            
        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_batch_time = total_batch_time / num_batches
        
        if self.profiler:
            self.profiler.end_timer(f'train_epoch_{self.current_epoch}')
            
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'batch_time': avg_batch_time,
            'throughput': self.batch_size / avg_batch_time
        }
        
    def _validate_epoch(self) -> Dict[str, float]:
        """Simulate validation epoch."""
        if self.profiler:
            self.profiler.start_timer(f'val_epoch_{self.current_epoch}')
            
        # Validation typically has slightly different metrics than training
        val_noise = random.uniform(-0.05, 0.1)  # Validation can be slightly worse
        
        base_loss = self._simulate_batch_loss(50)  # Mid-training performance
        base_accuracy = self._simulate_batch_accuracy(50)
        
        val_loss = base_loss + val_noise
        val_accuracy = max(0.0, base_accuracy + val_noise * 0.5)
        
        if self.profiler:
            self.profiler.end_timer(f'val_epoch_{self.current_epoch}')
            
        return {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
        
    def _simulate_batch_loss(self, batch_idx: int) -> float:
        """Simulate realistic batch loss progression."""
        # Start high and decrease with training progression
        epoch_progress = self.current_epoch / self.max_epochs
        batch_progress = batch_idx / 100.0  # Assuming 100 batches
        
        total_progress = epoch_progress + batch_progress * 0.1
        
        # Loss starts around 2.0 and decreases
        base_loss = 2.0
        improvement = total_progress * 1.5
        noise = random.uniform(-0.1, 0.1)
        
        return max(0.1, base_loss - improvement + noise)
        
    def _simulate_batch_accuracy(self, batch_idx: int) -> float:
        """Simulate realistic batch accuracy progression."""
        # Start low and increase with training progression  
        epoch_progress = self.current_epoch / self.max_epochs
        batch_progress = batch_idx / 100.0
        
        total_progress = epoch_progress + batch_progress * 0.1
        
        # Accuracy starts around 0.3 and increases
        base_accuracy = 0.3
        improvement = total_progress * 0.6
        noise = random.uniform(-0.05, 0.05)
        
        return min(0.95, base_accuracy + improvement + noise)
        
    def _update_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch_time: float):
        """Update training history."""
        self.training_history['epoch'].append(self.current_epoch)
        self.training_history['loss'].append(train_metrics['loss'])
        self.training_history['accuracy'].append(train_metrics['accuracy'])
        self.training_history['val_loss'].append(val_metrics.get('val_loss', 0.0))
        self.training_history['val_accuracy'].append(val_metrics.get('val_accuracy', 0.0))
        self.training_history['learning_rate'].append(self.learning_rate)
        self.training_history['batch_time'].append(train_metrics['batch_time'])
        self.training_history['throughput'].append(train_metrics['throughput'])
        
    def _log_epoch_progress(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch_time: float):
        """Log epoch progress."""
        if val_metrics:
            logger.info(
                f"Epoch {self.current_epoch}/{self.max_epochs} - "
                f"Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.3f} "
                f"({epoch_time:.2f}s)"
            )
        else:
            logger.info(
                f"Epoch {self.current_epoch}/{self.max_epochs} - "
                f"Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f} "
                f"({epoch_time:.2f}s)"
            )
            
    def _save_checkpoint(self, current_loss: float):
        """Save model checkpoint."""
        # Only save if this is the best model so far
        if self.save_best_only and current_loss >= self.best_val_loss:
            return
            
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.json"
        
        checkpoint_data = {
            'epoch': self.current_epoch,
            'model_name': self.model_name,
            'loss': current_loss,
            'accuracy': self.training_history['accuracy'][-1],
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'precision': self.precision,
            'devices': self.devices,
            'training_history': self.training_history,
            'optimizer_state': {'momentum': 0.9, 'weight_decay': 0.01},  # Mock optimizer state
            'timestamp': time.time()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Update best metrics
        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss
            # Save as best model
            best_model_path = self.output_dir / "best_model.json" 
            with open(best_model_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return starting epoch."""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                
            # Restore training state
            self.current_epoch = checkpoint_data['epoch']
            self.training_history = checkpoint_data.get('training_history', self.training_history)
            self.best_val_loss = min(self.training_history.get('val_loss', [float('inf')]))
            
            logger.info(f"Resumed from checkpoint: epoch {self.current_epoch}")
            return self.current_epoch
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return 0
            
    def _should_early_stop(self, current_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        if not self.enable_validation or self.early_stopping_patience <= 0:
            return False
            
        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience
            
    def _generate_results(self, training_time: float) -> Dict[str, Any]:
        """Generate comprehensive training results."""
        final_metrics = {
            'final_loss': self.training_history['loss'][-1] if self.training_history['loss'] else 0.0,
            'final_accuracy': self.training_history['accuracy'][-1] if self.training_history['accuracy'] else 0.0,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': max(self.training_history.get('val_accuracy', [0.0])),
            'total_epochs': self.current_epoch,
            'total_time': training_time,
            'avg_epoch_time': training_time / max(1, self.current_epoch),
            'avg_throughput': sum(self.training_history.get('throughput', [0.0])) / max(1, len(self.training_history.get('throughput', [1])))
        }
        
        # Add profiling results if available
        if self.profiler:
            final_metrics['profiling'] = self.profiler.get_metrics()
            
        # Add device information
        if self.mock_instance:
            device_info = {}
            for device_id in self.devices:
                device = self.mock_instance.devices[device_id]
                device_info[f'device_{device_id}'] = {
                    'utilization': device.utilization,
                    'temperature': device.temperature,
                    'power_usage': device.power_usage,
                    'memory_used': device.memory_used,
                    'memory_total': device.memory_total
                }
            final_metrics['devices'] = device_info
            
        return {
            'success': True,
            'model_name': self.model_name,
            'config': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'max_epochs': self.max_epochs,
                'precision': self.precision,
                'devices': self.devices
            },
            'metrics': final_metrics,
            'history': self.training_history
        }
        
    def _save_results(self, results: Dict[str, Any]):
        """Save training results to file."""
        results_path = self.output_dir / "training_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Training results saved: {results_path}")
        
    def predict(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """Simulate model inference."""
        logger.info("Running inference simulation...")
        
        predictions = []
        for i, input_text in enumerate(inputs):
            # Simulate inference processing time
            time.sleep(0.01)  # Small delay to simulate processing
            
            # Generate mock prediction
            confidence = 0.85 + random.uniform(0, 0.15)  # 85-100% confidence
            predicted_class = f"Class_{(i % 3) + 1}"  # Rotate through 3 classes
            
            prediction = {
                'input': input_text,
                'prediction': predicted_class,
                'confidence': confidence,
                'processing_time_ms': random.uniform(5, 15)
            }
            
            predictions.append(prediction)
            logger.info(f"  Input: '{input_text}' → {predicted_class} (confidence: {confidence:.2f})")
            
        return predictions


# Convenience function for quick training
def quick_train(
    model_name: str = "quick-model",
    batch_size: int = 32,
    max_epochs: int = 5,
    output_dir: str = "quick_output"
) -> Dict[str, Any]:
    """Quick training function for rapid prototyping."""
    trainer = MockGaudiTrainer(
        model_name=model_name,
        batch_size=batch_size,
        max_epochs=max_epochs,
        output_dir=output_dir,
        enable_checkpointing=True,
        enable_validation=True
    )
    
    return trainer.fit()


# Advanced training with custom configuration
def advanced_train(config: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced training with custom configuration."""
    trainer = MockGaudiTrainer(**config)
    return trainer.fit()