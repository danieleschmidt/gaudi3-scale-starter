"""
Enhanced Usage Example - Gaudi 3 Scale Generation 2

This example demonstrates the new reliability and production-readiness features
including error handling, validation, logging, health checks, and monitoring.
"""

import json
from pathlib import Path

# Import enhanced Gaudi 3 Scale components
from gaudi3_scale import (
    GaudiAccelerator, GaudiTrainer, GaudiOptimizer,
    HPUNotAvailableError, TrainingError, ConfigurationError,
    get_logger, HealthMonitor, HealthStatus,
    validate_training_config, DataValidator
)
from gaudi3_scale.health_checks import (
    create_default_hpu_health_checks,
    create_system_health_checks
)
from gaudi3_scale.retry_utils import retry_on_failure
from gaudi3_scale.exceptions import Gaudi3ScaleError


def main():
    """Demonstrate enhanced Gaudi 3 Scale capabilities."""
    
    # 1. Enhanced Logging Setup
    logger = get_logger('enhanced_example')
    logger.info("Starting enhanced Gaudi 3 Scale example")
    
    # 2. Configuration Validation Example
    training_config = {
        "batch_size": 32,
        "learning_rate": 0.0006,
        "max_epochs": 10,
        "optimizer_type": "fused_adamw",
        "precision": "bf16-mixed",
        "gradient_clip_val": 1.0,
        "output_dir": "./output"
    }
    
    print("\n=== Configuration Validation ===")
    validation_result = validate_training_config(training_config)
    
    if validation_result.is_valid:
        print("✅ Configuration is valid!")
        validated_config = validation_result.sanitized_value
    else:
        print("❌ Configuration validation failed:")
        for error in validation_result.errors:
            print(f"  • {error}")
        for warning in validation_result.warnings:
            print(f"  ⚠ {warning}")
        return
    
    # 3. Enhanced Accelerator with Error Handling
    print("\n=== Enhanced HPU Accelerator ===")
    try:
        accelerator = GaudiAccelerator(
            enable_monitoring=True,
            enable_health_checks=True,
            enable_retry=True,
            validate_environment=True
        )
        
        # Check HPU availability with detailed error information
        if accelerator.is_available():
            device_count = accelerator.auto_device_count()
            print(f"✅ {device_count} HPU devices available")
            
            # Get detailed device statistics
            device_stats = accelerator.get_device_stats(0)
            print(f"📊 Device 0 stats: {json.dumps(device_stats, indent=2)}")
            
        else:
            print("❌ No HPU devices available")
            return
            
    except HPUNotAvailableError as e:
        print(f"❌ HPU Error: {e}")
        print(f"💡 Recovery suggestions: {e.recovery_suggestions}")
        print(f"📋 Error context: {e.context}")
        return
    except Exception as e:
        logger.exception("Unexpected error initializing accelerator")
        return
    
    # 4. Health Monitoring Setup
    print("\n=== Health Monitoring Setup ===")
    try:
        health_monitor = HealthMonitor(
            enable_background_monitoring=False,  # For demo purposes
            check_interval=30.0
        )
        
        # Add various health checks
        health_monitor.checks.extend(create_default_hpu_health_checks(device_count=1))
        health_monitor.checks.extend(create_system_health_checks())
        
        # Run health checks
        health_results = health_monitor.run_all_checks()
        
        overall_status = health_monitor.get_overall_status()
        print(f"🏥 Overall health status: {overall_status.value}")
        
        # Display health check results
        for check_name, result in health_results.items():
            status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}.get(result.status.value, "❓")
            print(f"  {status_emoji} {check_name}: {result.message}")
            
    except Exception as e:
        logger.warning(f"Health monitoring setup failed: {e}")
    
    # 5. Enhanced Trainer with Comprehensive Error Handling
    print("\n=== Enhanced Training Setup ===")
    try:
        # Create trainer with enhanced capabilities
        trainer = GaudiTrainer(
            config=validated_config,
            model_name="enhanced_example_model",
            output_dir="./enhanced_output",
            enable_monitoring=True,
            enable_checkpointing=True,
            accelerator="hpu",
            devices=1,
            precision="bf16-mixed"
        )
        
        print("✅ Trainer initialized successfully")
        
        # Get training summary
        training_summary = trainer.get_training_summary()
        print("📋 Training Summary:")
        for key, value in training_summary.items():
            print(f"  • {key}: {value}")
        
        # Note: In a real scenario, you would provide actual model and data
        print("💡 Training would start here with actual model and data")
        
    except TrainingError as e:
        print(f"❌ Training Error: {e}")
        print(f"💡 Recovery suggestions: {e.recovery_suggestions}")
        logger.log_exception(e)
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
        print(f"📋 Error context: {e.context}")
    except Exception as e:
        logger.exception("Unexpected error in trainer setup")
        return
    
    # 6. Retry Logic Demonstration
    print("\n=== Retry Logic Demonstration ===")
    
    @retry_on_failure(
        max_attempts=3,
        base_delay=1.0,
        retryable_exceptions=[ConnectionError, TimeoutError]
    )
    def potentially_failing_operation():
        """Simulate an operation that might fail."""
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Simulated connection failure")
        return "Operation successful!"
    
    try:
        result = potentially_failing_operation()
        print(f"✅ {result}")
    except Exception as e:
        print(f"❌ Operation failed after retries: {e}")
    
    # 7. Input Validation Example
    print("\n=== Input Validation Example ===")
    validator = DataValidator()
    
    # Test various validation scenarios
    test_cases = [
        ("model_name", "llama-7b", "Valid model name"),
        ("batch_size", 32, "Valid batch size"),
        ("learning_rate", 0.0006, "Valid learning rate"),
        ("invalid_batch", -5, "Invalid negative batch size"),
        ("malicious_input", "<script>alert('xss')</script>", "Potential XSS attack")
    ]
    
    for field_name, value, description in test_cases:
        try:
            if isinstance(value, str):
                result = validator.validate_string(value, field_name, min_length=1, max_length=100)
            else:
                result = validator.validate_number(value, field_name, min_value=1, max_value=1000)
            
            if result.is_valid:
                print(f"  ✅ {description}: Valid")
                if result.sanitized_value != value:
                    print(f"    🧹 Sanitized: {result.sanitized_value}")
            else:
                print(f"  ❌ {description}: {result.errors[0]}")
                
        except Exception as e:
            print(f"  ❌ {description}: Validation failed - {e}")
    
    # 8. Performance Monitoring Example
    print("\n=== Performance Monitoring ===")
    if hasattr(accelerator, 'get_performance_summary'):
        perf_summary = accelerator.get_performance_summary()
        print("📊 Accelerator Performance Summary:")
        for key, value in perf_summary.items():
            print(f"  • {key}: {value}")
    
    # 9. Error Information Example
    print("\n=== Structured Error Information ===")
    try:
        # Simulate a structured error
        raise HPUNotAvailableError(
            requested_devices=8,
            available_devices=4,
            context={"cluster_id": "demo-cluster", "operation": "training_init"}
        )
    except HPUNotAvailableError as e:
        error_info = e.to_dict()
        print("📋 Structured error information:")
        print(json.dumps(error_info, indent=2, default=str))
    
    print("\n=== Example Completed Successfully ===")
    logger.info("Enhanced Gaudi 3 Scale example completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()