#!/usr/bin/env python3
"""Enhanced Simple Trainer - Generation 2 Implementation.

This adds reliability features like error handling, validation,
logging, health checks, and monitoring to the simple trainer.
"""

import time
import json
import logging
import traceback
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class TrainingStatus(Enum):
    """Training status enumeration."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Any = None


# Custom Exception Classes
class Gaudi3ScaleError(Exception):
    """Base exception for Gaudi 3 Scale."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, recovery_suggestions: List[str] = None):
        super().__init__(message)
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "timestamp": self.timestamp
        }


class TrainingError(Gaudi3ScaleError):
    """Raised when training encounters an error."""
    pass


class ValidationError(Gaudi3ScaleError):
    """Raised when validation fails."""
    pass


class ConfigurationError(Gaudi3ScaleError):
    """Raised when configuration is invalid."""
    pass


# Enhanced Configuration with Validation
class EnhancedTrainingConfig:
    """Enhanced configuration with validation and sanitization."""
    
    def __init__(
        self,
        model_name: str = "enhanced-model",
        batch_size: int = 16,
        learning_rate: float = 0.001,
        max_epochs: int = 5,
        precision: str = "float32",
        output_dir: str = "./output",
        validation_split: float = 0.2,
        early_stopping_patience: int = 3,
        checkpoint_every: int = 1,
        log_level: str = "INFO"
    ):
        # Validate and sanitize inputs
        validation_result = self._validate_config(
            model_name, batch_size, learning_rate, max_epochs,
            precision, output_dir, validation_split, early_stopping_patience,
            checkpoint_every, log_level
        )
        
        if not validation_result.is_valid:
            raise ConfigurationError(
                f"Configuration validation failed: {', '.join(validation_result.errors)}",
                context={"provided_config": locals()},
                recovery_suggestions=[
                    "Check parameter ranges and types",
                    "Ensure output directory is writable",
                    "Verify model name contains only valid characters"
                ]
            )
        
        # Set validated values
        self.model_name = self._sanitize_model_name(model_name)
        self.batch_size = max(1, min(1024, batch_size))  # Clamp to reasonable range
        self.learning_rate = max(1e-6, min(1.0, learning_rate))
        self.max_epochs = max(1, min(1000, max_epochs))
        self.precision = precision.lower()
        self.output_dir = Path(output_dir)
        self.validation_split = max(0.0, min(0.5, validation_split))
        self.early_stopping_patience = max(1, early_stopping_patience)
        self.checkpoint_every = max(1, checkpoint_every)
        self.log_level = log_level.upper()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Print warnings if values were adjusted
        if validation_result.warnings:
            for warning in validation_result.warnings:
                print(f"⚠️  Configuration Warning: {warning}")
    
    def _validate_config(self, *args) -> ValidationResult:
        """Validate configuration parameters."""
        errors = []
        warnings = []
        
        model_name, batch_size, learning_rate, max_epochs, precision, output_dir, validation_split, early_stopping_patience, checkpoint_every, log_level = args
        
        # Validate model name
        if not isinstance(model_name, str) or len(model_name) < 1:
            errors.append("model_name must be a non-empty string")
        elif len(model_name) > 100:
            warnings.append("model_name is very long, truncating to 100 characters")
        
        # Validate numeric parameters
        if not isinstance(batch_size, int) or batch_size < 1:
            errors.append("batch_size must be a positive integer")
        elif batch_size > 1024:
            warnings.append("batch_size is very large, clamping to 1024")
        
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            errors.append("learning_rate must be a positive number")
        elif learning_rate > 1.0:
            warnings.append("learning_rate is very high, clamping to 1.0")
        
        if not isinstance(max_epochs, int) or max_epochs < 1:
            errors.append("max_epochs must be a positive integer")
        
        # Validate precision
        valid_precisions = ["float32", "float16", "bf16", "bf16-mixed"]
        if precision not in valid_precisions:
            warnings.append(f"precision '{precision}' not in recommended list: {valid_precisions}")
        
        # Validate validation split
        if not isinstance(validation_split, (int, float)) or not (0.0 <= validation_split <= 0.5):
            errors.append("validation_split must be between 0.0 and 0.5")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _sanitize_model_name(self, name: str) -> str:
        """Sanitize model name to remove potentially problematic characters."""
        import re
        # Remove non-alphanumeric characters except hyphens and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9\-_]', '', name)
        return sanitized[:100]  # Truncate to reasonable length
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "precision": self.precision,
            "output_dir": str(self.output_dir),
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_every": self.checkpoint_every,
            "log_level": self.log_level
        }


# Enhanced Logging System
class EnhancedLogger:
    """Enhanced logging with structured output and performance monitoring."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.performance_metrics = {}
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)
    
    def log_exception(self, exception: Exception):
        """Log exception with full traceback."""
        self.logger.error(f"Exception occurred: {exception}")
        self.logger.error(traceback.format_exc())
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        self.performance_metrics[operation] = {
            "duration": duration,
            "timestamp": time.time(),
            **metrics
        }
        self.info(f"Performance: {operation}", duration=duration, **metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        return self.performance_metrics.copy()


# Health Monitoring System
class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self):
        self.checks = []
        self.last_check_time = None
    
    def add_check(self, name: str, check_func: Callable[[], bool], critical: bool = False):
        """Add a health check."""
        self.checks.append({
            "name": name,
            "func": check_func,
            "critical": critical,
            "last_result": None,
            "last_error": None
        })
    
    def run_checks(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        
        for check in self.checks:
            try:
                result = check["func"]()
                check["last_result"] = result
                check["last_error"] = None
                
                if result:
                    results[check["name"]] = HealthStatus.HEALTHY
                else:
                    results[check["name"]] = HealthStatus.WARNING if not check["critical"] else HealthStatus.CRITICAL
                    
            except Exception as e:
                check["last_result"] = False
                check["last_error"] = str(e)
                results[check["name"]] = HealthStatus.CRITICAL
        
        self.last_check_time = time.time()
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.run_checks()
        
        if any(status == HealthStatus.CRITICAL for status in results.values()):
            return HealthStatus.CRITICAL
        elif any(status == HealthStatus.WARNING for status in results.values()):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


# Enhanced Trainer with Reliability Features
class EnhancedTrainer:
    """Enhanced trainer with reliability, monitoring, and error handling."""
    
    def __init__(
        self,
        config: Optional[EnhancedTrainingConfig] = None,
        **kwargs
    ):
        if config is None:
            config = EnhancedTrainingConfig(**kwargs)
        
        self.config = config
        self.logger = EnhancedLogger(f"trainer.{config.model_name}", config.log_level)
        self.health_monitor = HealthMonitor()
        
        # Training state
        self.current_epoch = 0
        self.training_metrics = []
        self.status = TrainingStatus.IDLE
        self.best_val_accuracy = 0.0
        self.early_stopping_counter = 0
        
        # Setup health checks
        self._setup_health_checks()
        
        self.logger.info("Enhanced trainer initialized", config=config.to_dict())
    
    def _setup_health_checks(self):
        """Setup system health checks."""
        
        def check_output_dir():
            return self.config.output_dir.exists() and self.config.output_dir.is_dir()
        
        def check_disk_space():
            import shutil
            _, _, free = shutil.disk_usage(self.config.output_dir)
            return free > 100 * 1024 * 1024  # At least 100MB free
        
        def check_memory():
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent < 90  # Less than 90% memory usage
            except ImportError:
                return True  # Skip if psutil not available
        
        self.health_monitor.add_check("output_directory", check_output_dir, critical=True)
        self.health_monitor.add_check("disk_space", check_disk_space, critical=True)
        self.health_monitor.add_check("memory_usage", check_memory, critical=False)
    
    def train(
        self,
        model: Any = None,
        train_data: Any = None,
        val_data: Any = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run enhanced training with comprehensive error handling and monitoring."""
        
        # Pre-training health checks
        health_status = self.health_monitor.get_overall_status()
        if health_status == HealthStatus.CRITICAL:
            raise TrainingError(
                "Critical health check failures detected",
                context={"health_checks": self.health_monitor.run_checks()},
                recovery_suggestions=[
                    "Check disk space and permissions",
                    "Verify output directory is accessible",
                    "Check system resources"
                ]
            )
        
        self.logger.info("Starting enhanced training", 
                        health_status=health_status.value,
                        config=self.config.to_dict())
        
        self.status = TrainingStatus.RUNNING
        training_start_time = time.time()
        
        try:
            for epoch in range(1, self.config.max_epochs + 1):
                epoch_start_time = time.time()
                
                # Run health checks periodically
                if epoch % 5 == 0:  # Every 5 epochs
                    health_status = self.health_monitor.get_overall_status()
                    if health_status == HealthStatus.CRITICAL:
                        raise TrainingError(f"Critical health check failure at epoch {epoch}")
                
                # Simulate training step with error handling
                try:
                    train_loss, train_acc = self._robust_training_step(epoch)
                    val_loss, val_acc = self._robust_validation_step(epoch)
                except Exception as e:
                    self.logger.log_exception(e)
                    raise TrainingError(
                        f"Training step failed at epoch {epoch}",
                        context={"epoch": epoch, "error": str(e)},
                        recovery_suggestions=[
                            "Check data integrity",
                            "Reduce batch size",
                            "Verify model architecture"
                        ]
                    )
                
                epoch_time = time.time() - epoch_start_time
                
                # Record metrics
                epoch_metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch_time": epoch_time,
                    "health_status": health_status.value if epoch % 5 == 0 else "not_checked"
                }
                self.training_metrics.append(epoch_metrics)
                
                # Log performance metrics
                self.logger.log_performance(
                    f"epoch_{epoch}",
                    epoch_time,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc
                )
                
                if verbose:
                    print(f"  Epoch {epoch}/{self.config.max_epochs} - "
                          f"Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f} "
                          f"({epoch_time:.2f}s)")
                
                self.current_epoch = epoch
                
                # Early stopping logic
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.early_stopping_counter = 0
                    self._save_best_model(epoch, verbose)
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    if verbose:
                        print(f"    🛑 Early stopping triggered (patience: {self.config.early_stopping_patience})")
                    break
                
                # Regular checkpointing
                if epoch % self.config.checkpoint_every == 0:
                    self._save_checkpoint(epoch, verbose)
        
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self.status = TrainingStatus.PAUSED
            raise TrainingError(
                "Training interrupted by user",
                context={"epoch": self.current_epoch},
                recovery_suggestions=["Resume training from last checkpoint"]
            )
        
        except Exception as e:
            self.logger.log_exception(e)
            self.status = TrainingStatus.FAILED
            raise
        
        finally:
            if self.status == TrainingStatus.RUNNING:
                self.status = TrainingStatus.COMPLETED
        
        total_time = time.time() - training_start_time
        
        # Final results with comprehensive metrics
        final_metrics = self.training_metrics[-1] if self.training_metrics else {}
        results = {
            "success": True,
            "status": self.status.value,
            "total_epochs": self.current_epoch,
            "total_time": total_time,
            "best_val_accuracy": self.best_val_accuracy,
            "final_train_loss": final_metrics.get("train_loss", 0.0),
            "final_train_accuracy": final_metrics.get("train_accuracy", 0.0),
            "final_val_loss": final_metrics.get("val_loss", 0.0),
            "final_val_accuracy": final_metrics.get("val_accuracy", 0.0),
            "metrics_history": self.training_metrics,
            "performance_summary": self.logger.get_performance_summary(),
            "final_health_status": self.health_monitor.get_overall_status().value
        }
        
        self.logger.info("Training completed successfully", 
                        duration=total_time,
                        best_accuracy=self.best_val_accuracy)
        
        if verbose:
            print(f"✅ Training completed in {total_time:.2f}s")
            print(f"📊 Best accuracy: {self.best_val_accuracy:.3f}")
            print(f"🏥 Final health status: {results['final_health_status']}")
        
        return results
    
    def _robust_training_step(self, epoch: int) -> tuple[float, float]:
        """Robust training step with error handling and validation."""
        try:
            # Simulate training step with potential failures
            if epoch == 7:  # Simulate a potential failure point
                import random
                if random.random() < 0.1:  # 10% chance of simulated failure
                    raise RuntimeError("Simulated training step failure")
            
            # Normal training simulation
            base_loss = 2.0
            loss = base_loss * (0.8 ** epoch) + 0.1
            
            base_acc = 0.4
            acc = min(0.95, base_acc + (epoch * 0.08))
            
            # Add realistic noise
            import random
            loss += random.uniform(-0.05, 0.05)
            acc += random.uniform(-0.02, 0.02)
            acc = max(0.0, min(1.0, acc))
            
            # Validate outputs
            if loss < 0 or loss > 10:
                raise ValueError(f"Invalid loss value: {loss}")
            if not (0.0 <= acc <= 1.0):
                raise ValueError(f"Invalid accuracy value: {acc}")
            
            time.sleep(0.1)  # Simulate computation
            return loss, acc
            
        except Exception as e:
            self.logger.error(f"Training step failed: {e}")
            raise
    
    def _robust_validation_step(self, epoch: int) -> tuple[float, float]:
        """Robust validation step with error handling."""
        try:
            train_loss, train_acc = self._robust_training_step(epoch)
            
            val_loss = train_loss * 1.1
            val_acc = train_acc * 0.98
            
            return val_loss, val_acc
            
        except Exception as e:
            self.logger.error(f"Validation step failed: {e}")
            raise
    
    def _save_checkpoint(self, epoch: int, verbose: bool = True):
        """Save checkpoint with error handling."""
        try:
            checkpoint_path = self.config.output_dir / f"checkpoint_epoch_{epoch}.pt"
            
            # Simulate checkpoint data
            checkpoint_data = {
                "epoch": epoch,
                "config": self.config.to_dict(),
                "metrics": self.training_metrics[-1] if self.training_metrics else {},
                "timestamp": time.time()
            }
            
            # Save as JSON for this simulation
            with open(checkpoint_path.with_suffix('.json'), 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            if verbose:
                print(f"    💾 Saved checkpoint: {checkpoint_path}")
            
            self.logger.info("Checkpoint saved", epoch=epoch, path=str(checkpoint_path))
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            # Don't raise - checkpoint failure shouldn't stop training
    
    def _save_best_model(self, epoch: int, verbose: bool = True):
        """Save best model checkpoint."""
        try:
            best_model_path = self.config.output_dir / "best_model.json"
            
            best_model_data = {
                "epoch": epoch,
                "val_accuracy": self.best_val_accuracy,
                "config": self.config.to_dict(),
                "timestamp": time.time()
            }
            
            with open(best_model_path, 'w') as f:
                json.dump(best_model_data, f, indent=2)
            
            if verbose:
                print(f"    🏆 New best model saved: {best_model_path} (acc: {self.best_val_accuracy:.3f})")
            
            self.logger.info("Best model saved", 
                           epoch=epoch, 
                           accuracy=self.best_val_accuracy)
            
        except Exception as e:
            self.logger.error(f"Failed to save best model: {e}")


def enhanced_quick_train(
    model_name: str = "enhanced-quick-model",
    epochs: int = 5,
    batch_size: int = 16,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Enhanced quick training with reliability features."""
    try:
        config = EnhancedTrainingConfig(
            model_name=model_name,
            max_epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
        
        trainer = EnhancedTrainer(config)
        return trainer.train(verbose=verbose)
        
    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
        if hasattr(e, 'recovery_suggestions'):
            print("💡 Recovery suggestions:")
            for suggestion in e.recovery_suggestions:
                print(f"  • {suggestion}")
        raise


def main():
    """Demonstration of enhanced trainer capabilities."""
    print("🛡️ Enhanced Simple Trainer - Gaudi 3 Scale Generation 2")
    print("=" * 65)
    
    # Example 1: Basic enhanced training
    print("\n📋 Example 1: Enhanced Training with Reliability Features")
    try:
        results1 = enhanced_quick_train(
            model_name="enhanced-demo-1",
            epochs=6,
            batch_size=32,
            early_stopping_patience=3,
            verbose=True
        )
        print(f"🎉 Enhanced training completed successfully!")
        print(f"📊 Best accuracy: {results1['best_val_accuracy']:.3f}")
        print(f"🏥 Final health: {results1['final_health_status']}")
        
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
    
    # Example 2: Configuration validation
    print("\n📋 Example 2: Configuration Validation")
    try:
        # This should trigger validation warnings
        config = EnhancedTrainingConfig(
            model_name="test@model#with$invalid%chars",
            batch_size=2000,  # Too large
            learning_rate=2.0,  # Too high
            validation_split=0.8,  # Too high
            precision="invalid_precision"
        )
        print(f"✅ Configuration created with sanitization")
        print(f"📋 Sanitized config: {config.to_dict()}")
        
    except ConfigurationError as e:
        print(f"❌ Configuration validation failed: {e}")
        print(f"💡 Recovery suggestions: {e.recovery_suggestions}")
    
    # Example 3: Error handling demonstration
    print("\n📋 Example 3: Error Handling and Recovery")
    try:
        # Create trainer that might encounter issues
        trainer = EnhancedTrainer(
            model_name="error-demo",
            max_epochs=10,  # Long enough to potentially hit simulated failures
            batch_size=16
        )
        
        print("🎯 Starting training with potential failure points...")
        results3 = trainer.train(verbose=True)
        
        print(f"✅ Training survived potential failures!")
        print(f"📊 Performance summary:")
        for operation, metrics in results3['performance_summary'].items():
            print(f"  • {operation}: {metrics['duration']:.3f}s")
            
    except TrainingError as e:
        print(f"⚠️  Training error handled gracefully: {e}")
        print(f"💡 Recovery suggestions: {e.recovery_suggestions}")
        print(f"📋 Error context: {e.context}")
    
    print("\n✅ Generation 2 Enhancement Summary:")
    print("  ✓ Comprehensive error handling with custom exceptions")
    print("  ✓ Configuration validation and sanitization")
    print("  ✓ Structured logging with performance monitoring")
    print("  ✓ Health monitoring and system checks")
    print("  ✓ Early stopping and best model tracking")
    print("  ✓ Robust checkpointing with error recovery")
    print("  ✓ Graceful failure handling and recovery suggestions")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()