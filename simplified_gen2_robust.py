#!/usr/bin/env python3
"""Simplified Generation 2 Demo - MAKE IT ROBUST implementation.

This demo showcases Generation 2 robustness features:
- Enhanced error handling and recovery
- Security validation and monitoring
- Health checks and system diagnostics
- Comprehensive audit logging
- Checkpoint management with integrity
"""

import sys
import time
import json
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gaudi3_scale import GaudiTrainer, get_logger, ValidationError, TrainingError


@dataclass
class SimpleRobustConfig:
    """Simplified robust training configuration."""
    model_name: str = "robust-llama-simple"
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 6e-4
    enable_health_checks: bool = True
    enable_audit_logging: bool = True
    checkpoint_interval: int = 50
    output_dir: str = "simplified_gen2_output"


class SimpleSecurityManager:
    """Simplified security manager for demonstration."""
    
    def __init__(self):
        self.logger = get_logger("security")
        self.session_id = secrets.token_hex(16)
        self.event_log = []
    
    def log_security_event(self, event_type: str, **details):
        """Log security event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "session_id": self.session_id,
            "details": details
        }
        self.event_log.append(event)
        self.logger.info(f"Security Event: {event_type}")
    
    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize input data."""
        self.log_security_event("input_validation_start")
        
        # Simple validation checks
        if "learning_rate" in data:
            if not (1e-6 <= data["learning_rate"] <= 1e-1):
                raise ValidationError(f"Invalid learning rate: {data['learning_rate']}")
        
        if "batch_size" in data:
            if not (1 <= data["batch_size"] <= 1024):
                raise ValidationError(f"Invalid batch size: {data['batch_size']}")
        
        self.log_security_event("input_validation_passed")
        return data


class SimpleHealthMonitor:
    """Simplified health monitoring system."""
    
    def __init__(self):
        self.logger = get_logger("health")
        self.health_status = "healthy"
        self.metrics = {}
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform basic health check."""
        try:
            # Simulate system metrics
            self.metrics = {
                "memory_usage": 65.0,  # Simulated %
                "cpu_usage": 45.0,     # Simulated %
                "disk_usage": 30.0,    # Simulated %
                "system_status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Check thresholds
            if (self.metrics["memory_usage"] > 90 or 
                self.metrics["cpu_usage"] > 95):
                self.metrics["system_status"] = "warning"
                self.health_status = "warning"
            else:
                self.health_status = "healthy"
            
            self.logger.info(f"Health check: {self.health_status}")
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.health_status = "error"
            return {"system_status": "error", "error": str(e)}


class SimpleRobustTrainer:
    """Simplified robust trainer with enhanced error handling."""
    
    def __init__(self, config: SimpleRobustConfig):
        self.config = config
        self.logger = get_logger("robust_trainer")
        self.security_manager = SimpleSecurityManager()
        self.health_monitor = SimpleHealthMonitor()
        self.trainer = None
        self.retry_count = 0
        self.max_retries = 3
        
    def initialize_trainer(self):
        """Initialize trainer with error handling."""
        try:
            self.trainer = GaudiTrainer(
                model_name=self.config.model_name,
                output_dir=self.config.output_dir,
                max_epochs=self.config.num_epochs,
                enable_monitoring=True
            )
            self.logger.info("Robust trainer initialized successfully")
            return self.trainer
        except Exception as e:
            self.logger.error(f"Failed to initialize trainer: {e}")
            raise TrainingError(f"Trainer initialization failed: {e}")
    
    def robust_training_step(self, epoch: int, batch_idx: int) -> Dict[str, float]:
        """Execute training step with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Simulate potential failure at specific batch
                if batch_idx == 25 and epoch == 0 and attempt == 0:
                    raise RuntimeError("Simulated transient error")
                
                # Simulate training metrics
                base_loss = 2.5 - (epoch * 0.3) - (batch_idx * 0.001)
                noise = (abs(hash(f"{epoch}_{batch_idx}")) % 100) / 1000.0
                loss = base_loss + noise
                
                memory_usage = 12.0 + (epoch * 0.2) + (abs(hash(str(batch_idx))) % 30) / 100.0
                throughput = 5000 + (abs(hash(f"{epoch}_{batch_idx}")) % 2000)
                
                return {
                    "loss": loss,
                    "memory_usage": memory_usage,
                    "throughput": throughput,
                    "learning_rate": self.config.learning_rate * (1 - epoch * 0.1)
                }
                
            except Exception as e:
                self.retry_count += 1
                self.logger.warning(f"Training step failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise TrainingError(f"Training step failed after {self.max_retries} attempts: {e}")
                time.sleep(0.1)  # Brief delay before retry
    
    def create_secure_checkpoint(self, epoch: int, batch_idx: int, metrics: Dict) -> str:
        """Create checkpoint with integrity verification."""
        try:
            checkpoint_data = {
                "session_id": self.security_manager.session_id,
                "epoch": epoch,
                "batch_idx": batch_idx,
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": asdict(self.config)
            }
            
            # Add integrity hash
            data_str = json.dumps(checkpoint_data, sort_keys=True)
            checkpoint_data["integrity_hash"] = hashlib.sha256(data_str.encode()).hexdigest()
            
            # Save checkpoint
            checkpoint_file = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch}_batch_{batch_idx}.json"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.info(f"Secure checkpoint created: {checkpoint_file.name}")
            return str(checkpoint_file)
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise TrainingError(f"Checkpoint creation failed: {e}")
    
    def run_robust_training(self, training_data: List[Dict], validation_data: List[Dict]) -> Dict[str, Any]:
        """Execute robust training with comprehensive monitoring."""
        self.logger.info("Starting robust training execution...")
        
        # Validate configuration
        config_dict = asdict(self.config)
        validated_config = self.security_manager.validate_input(config_dict)
        
        training_metrics = {
            "epochs": [],
            "losses": [],
            "throughput": [],
            "memory_usage": [],
            "error_recovery_count": 0,
            "checkpoint_count": 0,
            "health_checks": []
        }
        
        try:
            # Log training start
            self.security_manager.log_security_event("training_started", model_name=self.config.model_name)
            
            for epoch in range(self.config.num_epochs):
                epoch_start_time = time.time()
                self.logger.info(f"Starting robust epoch {epoch + 1}/{self.config.num_epochs}")
                
                # Health check at start of epoch
                if self.config.enable_health_checks:
                    health_status = self.health_monitor.check_system_health()
                    training_metrics["health_checks"].append(health_status)
                    
                    if health_status["system_status"] == "error":
                        raise TrainingError("System health check failed")
                
                num_batches = len(training_data) // self.config.batch_size
                epoch_loss = 0.0
                
                for batch_idx in range(num_batches):
                    try:
                        batch_metrics = self.robust_training_step(epoch, batch_idx)
                        epoch_loss += batch_metrics["loss"]
                        
                        # Log progress every 20 batches
                        if batch_idx % 20 == 0:
                            self.logger.info(
                                f"Epoch {epoch + 1}, Batch {batch_idx}: "
                                f"loss={batch_metrics['loss']:.4f}, "
                                f"throughput={batch_metrics['throughput']:.0f} samples/s"
                            )
                        
                        # Create checkpoints
                        if batch_idx % self.config.checkpoint_interval == 0:
                            self.create_secure_checkpoint(epoch, batch_idx, batch_metrics)
                            training_metrics["checkpoint_count"] += 1
                            
                    except TrainingError as e:
                        training_metrics["error_recovery_count"] += 1
                        self.logger.warning(f"Recovered from training error: {e}")
                
                # Calculate epoch metrics
                epoch_time = time.time() - epoch_start_time
                avg_loss = epoch_loss / num_batches
                epoch_throughput = (num_batches * self.config.batch_size) / epoch_time
                
                training_metrics["epochs"].append(epoch + 1)
                training_metrics["losses"].append(avg_loss)
                training_metrics["throughput"].append(epoch_throughput)
                training_metrics["memory_usage"].append(
                    max([check.get("memory_usage", 0) for check in training_metrics["health_checks"][-3:]] + [0])
                )
                
                # Simple validation
                val_loss = self.run_validation(validation_data)
                
                self.logger.info(f"Epoch {epoch + 1} completed: "
                               f"train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, "
                               f"time={epoch_time:.2f}s")
            
            self.security_manager.log_security_event("training_completed", 
                                                   epochs=self.config.num_epochs,
                                                   final_loss=training_metrics["losses"][-1])
            
            self.logger.info("Robust training completed successfully!")
            return training_metrics
            
        except Exception as e:
            self.security_manager.log_security_event("training_failed", error=str(e))
            self.logger.error(f"Training failed: {e}")
            raise
    
    def run_validation(self, validation_data: List[Dict]) -> float:
        """Run simple validation."""
        val_loss = 2.0 - 0.1  # Simplified validation loss
        return val_loss


def generate_simple_data(config: SimpleRobustConfig) -> tuple[List[Dict], List[Dict]]:
    """Generate simple training data."""
    logger = get_logger(__name__)
    logger.info("Generating secure training data...")
    
    num_samples = 800
    training_data = []
    validation_data = []
    
    for i in range(num_samples):
        sample = {
            "id": i,
            "data": f"sample_{i}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if i < 720:  # 90% training
            training_data.append(sample)
        else:
            validation_data.append(sample)
    
    logger.info(f"Generated {len(training_data)} training and {len(validation_data)} validation samples")
    return training_data, validation_data


def save_robust_results(config: SimpleRobustConfig, metrics: Dict, trainer: SimpleRobustTrainer) -> Dict:
    """Save training results with security summary."""
    logger = get_logger(__name__)
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "session_id": trainer.security_manager.session_id,
        "model_name": config.model_name,
        "training_metrics": metrics,
        "robustness_features": {
            "error_recovery": True,
            "health_monitoring": config.enable_health_checks,
            "audit_logging": config.enable_audit_logging,
            "secure_checkpoints": True,
            "input_validation": True
        },
        "performance": {
            "epochs": len(metrics["epochs"]),
            "final_loss": metrics["losses"][-1] if metrics["losses"] else None,
            "avg_throughput": sum(metrics["throughput"]) / len(metrics["throughput"]) if metrics["throughput"] else 0,
            "error_recoveries": metrics["error_recovery_count"],
            "checkpoints": metrics["checkpoint_count"]
        },
        "security_events": trainer.security_manager.event_log,
        "health_checks": metrics["health_checks"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Save results
    results_file = output_path / "robust_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    return summary


def main():
    """Main demonstration."""
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("🛡️ TERRAGON SDLC - Simplified Generation 2")
    logger.info("MAKE IT ROBUST - Enhanced Reliability & Security")
    logger.info("=" * 60)
    
    try:
        # Create configuration
        config = SimpleRobustConfig()
        logger.info(f"✅ Configuration created: {config.model_name}")
        
        # Initialize robust trainer
        trainer = SimpleRobustTrainer(config)
        trainer.initialize_trainer()
        logger.info("✅ Robust trainer initialized")
        
        # Generate data
        training_data, validation_data = generate_simple_data(config)
        logger.info("✅ Training data generated")
        
        # Run robust training
        logger.info("🚀 Starting robust training...")
        metrics = trainer.run_robust_training(training_data, validation_data)
        logger.info("✅ Robust training completed")
        
        # Save results
        summary = save_robust_results(config, metrics, trainer)
        
        # Display results
        logger.info("=" * 60)
        logger.info("🎉 Generation 2 Demo Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"📊 Results Summary:")
        logger.info(f"   • Model: {summary['model_name']}")
        logger.info(f"   • Epochs: {summary['performance']['epochs']}")
        logger.info(f"   • Final Loss: {summary['performance']['final_loss']:.4f}")
        logger.info(f"   • Avg Throughput: {summary['performance']['avg_throughput']:.0f} samples/s")
        logger.info(f"   • Error Recoveries: {summary['performance']['error_recoveries']}")
        logger.info(f"   • Checkpoints: {summary['performance']['checkpoints']}")
        logger.info(f"🛡️ Robustness Features:")
        logger.info(f"   • Security Events: {len(summary['security_events'])}")
        logger.info(f"   • Health Checks: {len(summary['health_checks'])}")
        logger.info(f"   • Error Recovery: ✅")
        logger.info(f"   • Secure Checkpoints: ✅")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)